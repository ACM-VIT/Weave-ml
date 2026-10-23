import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from api.contracts import RepositorySearchRequest
from api.search import search_repositories
from api.runtime import ServiceDeadlineExceeded
from api.v2 import router
from config import REPOSITORY_EMBEDDING_DIM, QDRANT_VECTOR_NAME
from retrieval.v2_retriever import QdrantV2Retriever
from tests.test_v2_retrieval import _repository_payload


def request(**overrides):
    return RepositorySearchRequest.model_validate({
        'schema_version': 2, 'search_id': str(uuid.uuid4()),
        'query': 'find tools for image understanding', 'limit': 10, **overrides,
    })


@pytest.mark.parametrize('override', [
    {'query': 'a'}, {'query': ' ' * 2}, {'query': 'a' * 513}, {'limit': 0},
    {'limit': 101}, {'limit': True}, {'limit': '5'}, {'schema_version': 3},
    {'search_id': 'not-a-uuid'}, {'exclude_repo_ids': ['owner/repo']},
    {'filters': {'unknown': 'x'}}, {'filters': {'topics': ['Bad Topic']}},
    {'filters': {'languages': ['Python'] * 11}},
    {'exclude_repo_ids': [str(uuid.UUID(int=1))] * 2},
])
def test_search_contract_bounds(override):
    with pytest.raises(ValidationError):
        request(**override)


def test_authenticated_versioned_search_contract(monkeypatch):
    monkeypatch.setenv('INTERNAL_API_SECRET', 'search-local-secret')
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        body = request().model_dump(mode='json')
        assert client.post('/api/v2/search/repositories', json=body).status_code == 401
        with patch('api.v2._search_repository_job', return_value={
            'schema_version': 2, 'search_id': body['search_id'],
            'embedding_version': 'repo-embedding-v2', 'items': [],
        }):
            response = client.post('/api/v2/search/repositories', json=body, headers={'X-Internal-Secret': 'search-local-secret'})
            assert response.status_code == 200
            assert set(response.json()) == {'schema_version', 'search_id', 'embedding_version', 'items'}


def test_search_deadline_returns_retryable_unavailable(monkeypatch):
    monkeypatch.setenv('INTERNAL_API_SECRET', 'search-local-secret')
    app = FastAPI(); app.include_router(router)
    async def expired(*args, **kwargs):
        raise ServiceDeadlineExceeded("search")
    with patch('api.v2.run_service_job', expired), TestClient(app) as client:
        assert client.post('/api/v2/search/repositories', json=request().model_dump(mode='json'), headers={'X-Internal-Secret': 'search-local-secret'}).status_code == 503


def test_semantic_search_local_qdrant_filters_and_no_canonical_data():
    from utils.observability import diagnostic_snapshot, span_scope
    # Embedded in-memory Qdrant only; never reads network credentials.
    client = QdrantClient(':memory:')
    collection = 'phase4_search'
    client.create_collection(collection, vectors_config={QDRANT_VECTOR_NAME: models.VectorParams(size=REPOSITORY_EMBEDDING_DIM, distance=models.Distance.COSINE)})
    ids = [str(uuid.uuid4()) for _ in range(5)]
    vector = [1.] + [0.] * (REPOSITORY_EMBEDDING_DIM - 1)
    points = []
    for i, id in enumerate(ids):
        payload = _repository_payload(id, primary_language='Python' if i != 1 else 'Rust', topics=['vision'])
        if i == 2: payload.pop('serving_eligibility_version')
        if i == 3: payload['embedding_version'] = 'obsolete'
        points.append(models.PointStruct(id=id, vector={QDRANT_VECTOR_NAME: vector}, payload=payload))
    client.upsert(collection, points)
    retriever = QdrantV2Retriever(client=client, repository_collection=collection, heavy_ranker_enabled=False)
    seen = []
    embedder = SimpleNamespace(embed_text=lambda text: seen.append(text) or vector)
    req = request(filters={'languages': ['Python'], 'topics': ['vision']}, exclude_repo_ids=[ids[4]])
    with span_scope("search", server=True, authenticated_traceparent="00-1234567890abcdef1234567890abcdef-1234567890abcdef-01"):
        result = search_repositories(req, retriever, embedder)
    assert seen == [req.query]
    assert result['items'] == [{'repo_id': ids[0], 'score': pytest.approx(1.)}]
    assert set(result['items'][0]) == {'repo_id', 'score'}
    assert result['embedding_version'] == retriever.embedding_version
    spans = [span for span in diagnostic_snapshot()["spans"]
             if span["trace_id"] == "1234567890abcdef1234567890abcdef"]
    server = next(span for span in reversed(spans) if span["kind"] == "SERVER")
    dependency = next(span for span in reversed(spans) if span["stage"] == "qdrant")
    assert dependency["parent_span_id"] == server["span_id"]
    assert dependency["outcome"] == "ok"

    client.close()


@pytest.mark.slow
def test_conceptual_query_with_real_frozen_encoder_and_local_qdrant(monkeypatch):
    import os
    if os.getenv('RUN_LOCAL_SEMANTIC_MODEL_TEST') != '1':
        pytest.skip('Opt in to the cached frozen encoder; no network services are used.')
    monkeypatch.setenv('APP_ENV', 'production')
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('V2_ALLOW_UNQUALIFIED_HEAVY_RANKER', 'false')
    from embedding.runtime import shared_embedding_model
    from embedding.embeddings import SentenceTransformerEmbedder
    from config import REPOSITORY_EMBEDDING_MODEL
    embedder = SentenceTransformerEmbedder(REPOSITORY_EMBEDDING_MODEL, model=shared_embedding_model())
    client = QdrantClient(':memory:')
    collection = 'phase4_real_encoder'
    client.create_collection(collection, vectors_config={QDRANT_VECTOR_NAME: models.VectorParams(size=REPOSITORY_EMBEDDING_DIM, distance=models.Distance.COSINE)})
    docs = ['Software that recognizes objects in photographs using neural networks and computer vision.',
            'A PostgreSQL migration manager for transactional schema changes.',
            'A calendar scheduler for meetings and appointments.']
    ids = [str(uuid.uuid4()) for _ in docs]
    points = [models.PointStruct(id=id, vector={QDRANT_VECTOR_NAME:embedder.embed_text(doc)},payload=_repository_payload(id)) for id,doc in zip(ids,docs)]
    client.upsert(collection, points)
    retriever = QdrantV2Retriever(client=client,repository_collection=collection,heavy_ranker_enabled=False,minimum_eligible_repositories=0)
    result = search_repositories(request(query='image understanding',limit=3),retriever,embedder)
    assert result['items'][0]['repo_id'] == ids[0]
    client.close()


def test_search_discards_malformed_points_bad_scores_and_duplicate_ids():
    id = str(uuid.uuid4())
    points = [
        SimpleNamespace(id=id, payload=_repository_payload(id), score=.8),
        SimpleNamespace(id=id, payload=_repository_payload(id), score=.9),
        SimpleNamespace(id=id, payload=_repository_payload(id), score=float('nan')),
        SimpleNamespace(id='owner/repo', payload={'repo_id':'owner/repo'}, score=1.),
        SimpleNamespace(id=str(uuid.uuid4()), payload=_repository_payload(str(uuid.uuid4())), score=.99),
    ]
    client = SimpleNamespace(query_points=lambda **kwargs: SimpleNamespace(points=points))
    retriever = QdrantV2Retriever(client=client, heavy_ranker_enabled=False)
    embedder = SimpleNamespace(embed_text=lambda text:[1.] + [0.] * (REPOSITORY_EMBEDDING_DIM-1))
    result = search_repositories(request(),retriever,embedder)
    assert result['items'] == [{'repo_id':id,'score':.9}]


def test_search_clamps_cosine_roundoff_and_rejects_invalid_scores():
    scores = [
        1.0000001192092896, -1.0000001192092896,
        1.00001, -1.00001, float('nan'), float('inf'), float('-inf'),
        True, '0.9', None,
    ]
    ids = [str(uuid.uuid4()) for _ in scores]
    points = [
        SimpleNamespace(id=id, payload=_repository_payload(id), score=score)
        for id, score in zip(ids, scores)
    ]
    client = SimpleNamespace(query_points=lambda **kwargs: SimpleNamespace(points=points))
    retriever = QdrantV2Retriever(client=client, heavy_ranker_enabled=False)
    embedder = SimpleNamespace(embed_text=lambda text: [1.] + [0.] * (REPOSITORY_EMBEDDING_DIM - 1))

    result = search_repositories(request(), retriever, embedder)

    assert result['items'] == [
        {'repo_id': ids[0], 'score': 1.0},
        {'repo_id': ids[1], 'score': -1.0},
    ]


def test_query_embedding_does_not_construct_or_call_a_summary_provider():
    from api.v2 import _search_repository_job
    from embedding.runtime import BoundedEmbeddingModel
    model = BoundedEmbeddingModel(object(), max_concurrency=1)
    search_retriever = SimpleNamespace(embedding_version='repo-embedding-v2', embedding_dim=REPOSITORY_EMBEDDING_DIM)
    with patch('api.v2.repository_embedding_pipeline', side_effect=AssertionError('summary runtime must not be constructed')), patch('api.v2.shared_embedding_model', return_value=model), patch('api.v2.retriever', return_value=search_retriever), patch('api.search.search_repositories', return_value={'items':[]}) as search:
        _search_repository_job(request())
        search.assert_called_once()


@pytest.mark.parametrize('adversary', ['unmatched_brackets', 'blank_lines', 'whitespace_lines'])
def test_search_and_event_loop_remain_responsive_during_malformed_readme_ingestion(adversary):
    import subprocess
    import sys

    # A regressed regex may hold the interpreter GIL, defeating asyncio's own
    # timeout. Run the complete concurrent scenario in a killable process.
    result = subprocess.run(
        [sys.executable, '-c',
         'import asyncio; from tests.test_repository_search import '
         '_malformed_ingestion_search_scenario; '
         f'asyncio.run(_malformed_ingestion_search_scenario({adversary!r}))'],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


async def _malformed_ingestion_search_scenario(adversary):
    import asyncio
    import os
    import threading
    import time

    import httpx

    from api import runtime as service_runtime
    from api.contracts import RepositoryJob
    from api.v2 import _repository_embedding_payload
    from embedding import runtime as embedding_runtime
    from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
    from embedding.embeddings import SentenceTransformerEmbedder
    from summarization.pipeline import CardSummaryPipeline

    class Encoder:
        def encode(self, texts, **kwargs):
            return [[1.0] + [0.0] * (REPOSITORY_EMBEDDING_DIM - 1) for _ in texts]

    model = embedding_runtime.BoundedEmbeddingModel(Encoder(), max_concurrency=1)
    pipeline = RepositoryEmbeddingPipeline(
        embedder=SentenceTransformerEmbedder(model=model),
        card_summarizer=CardSummaryPipeline(provider=None),
    )
    client = QdrantClient(':memory:')
    collection = 'malformed_ingestion_search'
    client.create_collection(collection, vectors_config={
        QDRANT_VECTOR_NAME: models.VectorParams(size=REPOSITORY_EMBEDDING_DIM, distance=models.Distance.COSINE),
    })
    repo_id = str(uuid.uuid4())
    client.upsert(collection, [models.PointStruct(
        id=repo_id, vector={QDRANT_VECTOR_NAME: Encoder().encode(['query'])[0]},
        payload=_repository_payload(repo_id),
    )])
    retriever = QdrantV2Retriever(client=client, repository_collection=collection, heavy_ranker_enabled=False)
    readme = {
        'unmatched_brackets': '[' * 1_000_000,
        # Keep whitespace inside non-whitespace sentinels so request-level
        # trimming cannot remove the adversary before ingestion processes it.
        'blank_lines': 'x' + '\n' * 999_998 + 'x',
        'whitespace_lines': 'x' + ' \t\n' * 333_332 + 'x',
    }[adversary]
    job = RepositoryJob.model_validate({
        'schema_version': 2, 'job_id': str(uuid.uuid4()), 'repo_id': repo_id,
        'content_version': 1,
        'repository': {
            'repo_id': repo_id, 'full_name': 'weave/malformed-readme',
            'description': 'A collaborative review workspace for developer teams.',
            'readme': readme, 'content_hash': 'isolated-malformed-fixture',
        },
    })
    assert job.repository.readme == readme
    started = threading.Event()
    stop_heartbeat = asyncio.Event()
    ticks = []

    async def heartbeat():
        while not stop_heartbeat.is_set():
            ticks.append(time.monotonic())
            await asyncio.sleep(0.01)

    def ingest():
        started.set()
        payload = _repository_embedding_payload(job)
        # Both canonical ingestion cleanup and summary source/fallback parsing
        # execute for real, using the same bounded encoder as query embedding.
        return pipeline.embed_repository(payload)

    app = FastAPI()
    app.include_router(router)
    with patch.dict(os.environ, {
        'INTERNAL_API_SECRET': 'isolated-malformed-search',
        'EMBEDDING_EXECUTOR_WORKERS': '1', 'EMBEDDING_MAX_OUTSTANDING_JOBS': '1',
        'V2_SEARCH_EXECUTOR_WORKERS': '1', 'V2_SEARCH_MAX_OUTSTANDING': '1',
    }), patch('api.v2.shared_embedding_model', return_value=model), patch('api.v2.retriever', return_value=retriever):
        began = time.monotonic()
        ticker = asyncio.create_task(heartbeat())
        ingestion = asyncio.create_task(embedding_runtime.run_embedding_job(ingest))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://isolated.test') as http:
                response = await http.post(
                    '/api/v2/search/repositories',
                    headers={'X-Internal-Secret': 'isolated-malformed-search'},
                    json=request().model_dump(mode='json'),
                )
            elapsed = time.monotonic() - began
            assert response.status_code == 200, response.text
            assert response.json()['items'] == [{'repo_id': repo_id, 'score': 1.0}]
            assert elapsed < 1.5, f'Search stalled during README ingestion for {elapsed:.3f}s'
            result = await ingestion
            assert result.card_summary is not None
            ticks.append(time.monotonic())
            assert max(b - a for a, b in zip(ticks, ticks[1:])) < 1.5
        finally:
            stop_heartbeat.set()
            await ticker
            await ingestion
            embedding_runtime.reset_embedding_runtime_for_tests()
            service_runtime.shutdown_service_runtime()
            client.close()


def test_search_reuses_only_embeddings_across_normalized_queries_and_filters(monkeypatch):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    from api.v2 import _search_repository_job
    from embedding.runtime import BoundedEmbeddingModel

    encoded = []
    started, release = threading.Event(), threading.Event()

    class Encoder:
        def encode(self, texts, **kwargs):
            encoded.extend(texts)
            started.set()
            assert release.wait(3)
            return [[1.] + [0.] * (REPOSITORY_EMBEDDING_DIM - 1) for _ in texts]

    model = BoundedEmbeddingModel(Encoder(), max_concurrency=1)
    client = QdrantClient(':memory:')
    collection = 'search_cache_filters'
    client.create_collection(collection, vectors_config={QDRANT_VECTOR_NAME: models.VectorParams(
        size=REPOSITORY_EMBEDDING_DIM, distance=models.Distance.COSINE,
    )})
    ids = [str(uuid.UUID(int=i)) for i in (1, 2)]
    vector = [1.] + [0.] * (REPOSITORY_EMBEDDING_DIM - 1)
    client.upsert(collection, [models.PointStruct(
        id=id, vector={QDRANT_VECTOR_NAME: vector},
        payload=_repository_payload(id, primary_language=language, topics=['vision']),
    ) for id, language in zip(ids, ['Python', 'Rust'])])
    retriever = QdrantV2Retriever(client=client, repository_collection=collection, heavy_ranker_enabled=False)
    monkeypatch.setattr('api.v2.shared_embedding_model', lambda: model)
    monkeypatch.setattr('api.v2.retriever', lambda: retriever)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(_search_repository_job, request(query='  Image  Understanding  ', filters={'languages': ['Python']}))
            assert started.wait(2)
            second = pool.submit(_search_repository_job, request(query='Image  Understanding', filters={'topics': ['vision']}, exclude_repo_ids=[ids[0]]))
            try:
                deadline = time.monotonic() + 2
                while model.query_embedding_cache.snapshot()['waiters'] != 1:
                    assert time.monotonic() < deadline
                    time.sleep(.001)
            finally:
                release.set()
            first, second = first.result(2), second.result(2)
        excluded = _search_repository_job(request(query='Image  Understanding', exclude_repo_ids=ids))
        assert first['items'] == [{'repo_id': ids[0], 'score': 1.}]
        assert second['items'] == [{'repo_id': ids[1], 'score': 1.}]
        assert excluded['items'] == []
        assert first['search_id'] != second['search_id']
        assert encoded == ['Image  Understanding']
        # Preserve case and internal whitespace exactly; neither is normalized
        # by the public request contract, so they must not collide in the cache.
        _search_repository_job(request(query='Image Understanding'))
        _search_repository_job(request(query='image  understanding'))
        assert encoded == ['Image  Understanding', 'Image Understanding', 'image  understanding']
        retriever.embedding_version = 'next-embedding-version'
        assert _search_repository_job(request(query='Image  Understanding'))['items'] == []
        assert len(encoded) == 4
    finally:
        release.set()
        client.close()
