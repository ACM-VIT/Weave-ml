import asyncio
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
        raise asyncio.TimeoutError()
    with patch('api.v2.run_embedding_job', expired), TestClient(app) as client:
        assert client.post('/api/v2/search/repositories', json=request().model_dump(mode='json'), headers={'X-Internal-Secret': 'search-local-secret'}).status_code == 503


def test_semantic_search_local_qdrant_filters_and_no_canonical_data():
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
    result = search_repositories(req, retriever, embedder)
    assert seen == [req.query]
    assert result['items'] == [{'repo_id': ids[0], 'score': pytest.approx(1.)}]
    assert set(result['items'][0]) == {'repo_id', 'score'}
    assert result['embedding_version'] == retriever.embedding_version
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


def test_query_embedding_does_not_construct_or_call_a_summary_provider():
    from api.v2 import _search_repository_job
    with patch('api.v2.repository_embedding_pipeline', side_effect=AssertionError('summary runtime must not be constructed')), patch('api.v2.shared_embedding_model', return_value=object()), patch('api.v2.retriever', return_value=object()), patch('api.search.search_repositories', return_value={'items':[]}) as search:
        _search_repository_job(request())
        search.assert_called_once()
