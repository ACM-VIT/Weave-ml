import uuid
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from qdrant_client import QdrantClient, models

from config import (
    EMBEDDING_MODEL_REVISION,
    QDRANT_DISTANCE,
    QDRANT_PAYLOAD_INDEX_SCHEMA,
    QDRANT_VECTOR_NAME,
    REPOSITORY_EMBEDDING_DIM,
    REPOSITORY_EMBEDDING_MODEL,
    REPOSITORY_EMBEDDING_VERSION,
)
from embedding.vector_contract import (
    REPOSITORY_SERVING_ELIGIBILITY_FIELD,
    REPOSITORY_SERVING_ELIGIBILITY_VERSION,
    repository_payload_defaults,
)
from inference.feed_assembly import FeedAssemblySystem
from retrieval.discovery_cache import DiscoveryRevision, SharedDiscoveryCache
from retrieval.v2_retriever import (
    QdrantV2Retriever,
    RankedRepository,
    RetrievalDependencyError,
)


VECTOR = [1.0] + [0.0] * (REPOSITORY_EMBEDDING_DIM - 1)


def _repository_payload(repo_id, *, stars=0, **overrides):
    payload = repository_payload_defaults()
    payload.update(
        {
            "repo_id": repo_id,
            "full_name": f"owner/{repo_id}",
            "star_count": stars,
            "content_version": 1,
            "content_hash": f"content-{repo_id}",
            "source_hash": f"source-{repo_id}",
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    payload.update(overrides)
    return payload


def _user_payload(**overrides):
    payload = {
        "last_feedback_version": 0,
        "embedding_dim": REPOSITORY_EMBEDDING_DIM,
        "embedding_model": REPOSITORY_EMBEDDING_MODEL,
        "embedding_model_revision": EMBEDDING_MODEL_REVISION,
    }
    payload.update(overrides)
    return payload


class FakeQdrant:
    def __init__(self):
        self.user_id = str(uuid.uuid4())
        self.repo_ids = [str(uuid.uuid4()), str(uuid.uuid4())]

    def retrieve(self, collection_name, ids, with_vectors, with_payload=True):
        return [SimpleNamespace(id=ids[0], vector=VECTOR, payload=_user_payload())]

    def query_points(self, **_kwargs):
        point = SimpleNamespace(
            id=self.repo_ids[0],
            score=0.9,
            vector={"repo_embedding": VECTOR},
            payload=_repository_payload(self.repo_ids[0], stars=50),
        )
        return SimpleNamespace(points=[point])

    def scroll(self, **_kwargs):
        points = [
            SimpleNamespace(
                id=self.repo_ids[0],
                vector={"repo_embedding": VECTOR},
                payload=_repository_payload(self.repo_ids[0], stars=50),
            ),
            SimpleNamespace(
                id=self.repo_ids[1],
                vector={"repo_embedding": VECTOR},
                payload=_repository_payload(
                    self.repo_ids[1], stars=1000, delta_7d=20
                ),
            ),
            SimpleNamespace(
                id=str(uuid.uuid4()),
                vector={"repo_embedding": VECTOR},
                payload={"repo_id": "owner/legacy"},
            ),
        ]
        return points, None


class HeavyRankerQdrant:
    def __init__(self):
        self.user_id = str(uuid.uuid4())
        self.repo_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        self.query_calls = []
        self.scroll_calls = []
        self.retrieve_calls = []
        self.count_calls = []
        self.vector = VECTOR

    def info(self):
        return SimpleNamespace(version="1.18.2")

    def retrieve(self, **kwargs):
        self.retrieve_calls.append(kwargs)
        if kwargs["ids"] != [self.user_id]:
            return [
                self._project(
                    self._point(repo_id, stars=100 if repo_id == self.repo_ids[0] else 10),
                    kwargs.get("with_payload"),
                )
                for repo_id in kwargs["ids"]
            ]
        return [
            SimpleNamespace(
                id=self.user_id,
                vector=self.vector,
                payload=_user_payload(
                    skills=["Python"], topics=["recommendation"]
                ),
            )
        ]

    def _point(self, repo_id, *, score=None, stars=0):
        values = {
            "id": repo_id,
            "vector": {"repo_embedding": self.vector},
            "payload": _repository_payload(
                repo_id,
                stars=stars,
                primary_language="Python",
                languages=["Python"],
                topics=["recommendation"],
            ),
        }
        if score is not None:
            values["score"] = score
        return SimpleNamespace(**values)

    @staticmethod
    def _project(point, with_payload):
        if isinstance(with_payload, list):
            point.payload = {
                key: value
                for key, value in point.payload.items()
                if key in with_payload
            }
        return point

    def query_points(self, **kwargs):
        self.query_calls.append(kwargs)
        return SimpleNamespace(points=[
            self._project(
                self._point(self.repo_ids[0], score=0.95, stars=100),
                kwargs.get("with_payload"),
            ),
            self._project(
                self._point(self.repo_ids[1], score=0.50, stars=10),
                kwargs.get("with_payload"),
            ),
        ])

    def scroll(self, **kwargs):
        self.scroll_calls.append(kwargs)
        return [
            self._project(
                self._point(self.repo_ids[0], stars=100),
                kwargs.get("with_payload"),
            ),
            self._project(
                self._point(self.repo_ids[1], stars=10),
                kwargs.get("with_payload"),
            ),
        ], None

    def get_collection(self, collection_name):
        if collection_name == "user_profiles":
            vectors = SimpleNamespace(
                size=REPOSITORY_EMBEDDING_DIM,
                distance=QDRANT_DISTANCE,
            )
        else:
            vectors = {
                "repo_embedding": SimpleNamespace(
                    size=REPOSITORY_EMBEDDING_DIM,
                    distance=QDRANT_DISTANCE,
                )
            }
        return SimpleNamespace(
            points_count=len(self.repo_ids),
            config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)),
            payload_schema={
                field: SimpleNamespace(data_type=schema)
                for field, schema in QDRANT_PAYLOAD_INDEX_SCHEMA.items()
            },
        )

    def count(self, **kwargs):
        self.count_calls.append(kwargs)
        return SimpleNamespace(count=len(self.repo_ids))


class DeterministicHeavyRanker:
    ready = True
    production_qualified = True
    model_version = "heavy-ranker-v2"

    def __init__(self):
        self.calls = []

    def score_batch(self, user_vector, user_skills, candidates):
        self.calls.append((user_vector, user_skills, candidates))
        return [
            {
                "repo_id": candidates[1]["id"],
                "final_score": 0.9,
            },
            {
                "repo_id": candidates[0]["id"],
                "final_score": 0.1,
            },
        ]


@pytest.fixture
def local_recommendation_corpus():
    # Exercise real Qdrant filtering and limit ordering, without network access.
    client = QdrantClient(":memory:")
    repository_collection = "exclusion_repositories"
    user_collection = "exclusion_users"
    client.create_collection(
        repository_collection,
        vectors_config={
            QDRANT_VECTOR_NAME: models.VectorParams(
                size=REPOSITORY_EMBEDDING_DIM, distance=models.Distance.COSINE
            )
        },
    )
    client.create_collection(
        user_collection,
        vectors_config=models.VectorParams(
            size=REPOSITORY_EMBEDDING_DIM, distance=models.Distance.COSINE
        ),
    )
    user_id = str(uuid.UUID(int=10_000))
    repo_ids = [str(uuid.UUID(int=index + 1)) for index in range(600)]
    now = datetime(2026, 10, 21, tzinfo=timezone.utc)
    points = []
    for index, repo_id in enumerate(repo_ids):
        vector = list(VECTOR)
        vector[1] = index / len(repo_ids)
        points.append(models.PointStruct(
            id=repo_id,
            vector={QDRANT_VECTOR_NAME: vector},
            payload=_repository_payload(
                repo_id,
                stars=600 - index,
                trend_velocity=float(600 - index),
                activity_score=(600 - index) / 600,
                pushed_at=(now - timedelta(days=index)).isoformat(),
                primary_language=f"language-{index % 12}",
            ),
        ))
    client.upsert(repository_collection, points=points)
    client.upsert(user_collection, points=[models.PointStruct(
        id=user_id, vector=VECTOR, payload=_user_payload()
    )])
    # Record real calls, not a fake implementation of Qdrant's filters.
    for method in ("retrieve", "query_points", "scroll"):
        setattr(client, method, Mock(wraps=getattr(client, method)))
    yield SimpleNamespace(
        client=client,
        user_id=user_id,
        repo_ids=repo_ids,
        repository_collection=repository_collection,
        user_collection=user_collection,
    )
    client.close()


def _corpus_retriever(corpus, *, heavy):
    ranker = DeterministicHeavyRanker()

    def score_all(user_vector, user_skills, candidates):
        ranker.calls.append((user_vector, user_skills, candidates))
        return [
            {"repo_id": candidate["id"], "final_score": 1 - index / 1_000}
            for index, candidate in enumerate(candidates)
        ]

    ranker.score_batch = score_all
    return QdrantV2Retriever(
        client=corpus.client,
        repository_collection=corpus.repository_collection,
        user_collection=corpus.user_collection,
        ranker=ranker if heavy else None,
        heavy_ranker_enabled=heavy,
        heavy_ranker_required=heavy,
    )


@pytest.mark.parametrize("heavy", [False, True], ids=["hybrid", "required-heavy"])
@pytest.mark.parametrize("excluded_count", [150, 500])
def test_history_exclusions_do_not_exhaust_a_bounded_candidate_pool(
    local_recommendation_corpus, heavy, excluded_count
):
    corpus = local_recommendation_corpus
    retriever = _corpus_retriever(corpus, heavy=heavy)
    excluded = corpus.repo_ids[:excluded_count]

    batch = retriever.recommend_batch(corpus.user_id, 45, excluded, "history-pool")

    returned_ids = [item.repo_id for item in batch.items]
    assert len(returned_ids) == len(set(returned_ids)) == 45
    assert set(returned_ids) <= set(corpus.repo_ids[excluded_count:])
    assert batch.ranker_applied is heavy
    assert len(corpus.client.retrieve.call_args_list) == (2 if heavy else 1)
    assert len(corpus.client.query_points.call_args_list) == 1
    assert len(corpus.client.scroll.call_args_list) == 4
    assert corpus.client.query_points.call_args.kwargs["limit"] == 150
    assert all(call.kwargs["limit"] == 75 for call in corpus.client.scroll.call_args_list)
    expected_exclusions = [models.HasIdCondition(has_id=sorted(excluded))]
    assert (
        corpus.client.query_points.call_args.kwargs["query_filter"].must_not
        == expected_exclusions
    )
    assert all(
        call.kwargs["scroll_filter"].must_not == expected_exclusions
        for call in corpus.client.scroll.call_args_list
    )
    if heavy:
        candidates = retriever.ranker.calls[0][2]
        assert 45 <= len(candidates) <= 150
        assert not {candidate["id"] for candidate in candidates} & set(excluded)


def test_discovery_only_applies_history_exclusions_before_channel_limits(
    local_recommendation_corpus,
):
    corpus = local_recommendation_corpus
    corpus.client.delete(
        corpus.user_collection,
        points_selector=models.PointIdsList(points=[corpus.user_id]),
    )
    retriever = _corpus_retriever(corpus, heavy=False)

    batch = retriever.recommend_batch(
        corpus.user_id, 45, corpus.repo_ids[:500], "history-discovery", cold_start=True
    )

    assert len(batch.items) == 45
    assert {item.repo_id for item in batch.items} <= set(corpus.repo_ids[500:])
    assert batch.retrieval_mode == "cold_start_discovery"
    corpus.client.query_points.assert_not_called()
    assert len(corpus.client.scroll.call_args_list) == 4


@pytest.mark.parametrize("heavy", [False, True], ids=["hybrid", "required-heavy"])
def test_history_exclusion_api_preserves_backend_generation_contract(
    local_recommendation_corpus, monkeypatch, heavy,
):
    from fastapi.testclient import TestClient
    from api import main as api_main

    monkeypatch.setenv("INTERNAL_API_SECRET", "history-contract-secret")
    corpus = local_recommendation_corpus
    retriever = _corpus_retriever(corpus, heavy=heavy)
    payload = {
        "schema_version": 2,
        "generation_id": str(uuid.UUID(int=20_000)),
        "user_id": corpus.user_id,
        "feed_version": 7,
        "limit": 45,
        "exclude_repo_ids": corpus.repo_ids[:500],
        "context": {"cold_start": False},
    }
    with patch("api.v2.retriever", return_value=retriever):
        response = TestClient(api_main.app).post(
            "/api/v2/recommendations/generate",
            headers={"x-internal-secret": "history-contract-secret"},
            json=payload,
        )

    assert response.status_code == 200
    body = response.json()
    assert {key: body[key] for key in (
        "schema_version", "generation_id", "user_id", "feed_version"
    )} == {key: payload[key] for key in (
        "schema_version", "generation_id", "user_id", "feed_version"
    )}
    assert len(body["items"]) == 45
    assert {item["repo_id"] for item in body["items"]} <= set(corpus.repo_ids[500:])
    assert body["served_ranker"] == ("heavy" if heavy else "hybrid")


def test_exclusion_filter_keeps_defensive_validation_and_deduplication(
    local_recommendation_corpus,
):
    corpus = local_recommendation_corpus
    retriever = _corpus_retriever(corpus, heavy=True)
    fixture = HeavyRankerQdrant()
    excluded, valid, alias, invalid_features, invalid_vector, unstamped = corpus.repo_ids[:6]
    points = [fixture._point(repo_id, score=0.9) for repo_id in (
        excluded, valid, valid, alias, invalid_features, invalid_vector, unstamped
    )]
    points[3].payload["repo_id"] = corpus.repo_ids[6]
    points[4].payload["doc_quality"] = float("nan")
    points[5].vector = {QDRANT_VECTOR_NAME: [float("nan")] * REPOSITORY_EMBEDDING_DIM}
    points[6].payload.pop(REPOSITORY_SERVING_ELIGIBILITY_FIELD)
    # An untrusted/broken client may ignore filters or repeat points. Keep the
    # existing in-process exclusion, canonical-ID, feature and vector checks.
    corpus.client.query_points.side_effect = lambda **kwargs: SimpleNamespace(points=points)
    corpus.client.scroll.side_effect = lambda **kwargs: (points, None)
    original_retrieve = corpus.client.retrieve.side_effect or corpus.client.retrieve._mock_wraps
    by_id = {point.id: point for point in points}
    corpus.client.retrieve.side_effect = lambda **kwargs: (
        original_retrieve(**kwargs)
        if kwargs["collection_name"] == corpus.user_collection
        else [by_id[repo_id] for repo_id in kwargs["ids"] if repo_id in by_id]
    )

    batch = retriever.recommend_batch(corpus.user_id, 45, [excluded, excluded])

    assert [item.repo_id for item in batch.items] == [valid]
    assert [candidate["id"] for candidate in retriever.ranker.calls[0][2]] == [valid]


def test_refill_filter_accepts_mixed_qdrant_point_id_types():
    retriever = QdrantV2Retriever(client=HeavyRankerQdrant(), heavy_ranker_enabled=False)
    repo_id = str(uuid.UUID(int=1))

    filtered = retriever._eligible_filter(exclude_repo_ids={repo_id, 42})

    assert filtered.must_not == [models.HasIdCondition(has_id=[42, repo_id])]


@pytest.mark.parametrize("semantic", [True, False], ids=["semantic", "discovery-only"])
def test_low_yield_refills_past_invalid_points_with_real_qdrant(
    local_recommendation_corpus, semantic,
):
    corpus = local_recommendation_corpus
    # These still match Qdrant's eligibility stamp but fail defensive feature
    # validation. Reading the same top pool again cannot recover the batch.
    corpus.client.set_payload(
        corpus.repository_collection, payload={"doc_quality": "invalid"},
        points=corpus.repo_ids[:180],
    )
    if not semantic:
        corpus.client.delete(
            corpus.user_collection,
            points_selector=models.PointIdsList(points=[corpus.user_id]),
        )
    retriever = _corpus_retriever(corpus, heavy=semantic)

    batch = retriever.recommend_batch(corpus.user_id, 45, [])

    ids = [item.repo_id for item in batch.items]
    assert len(ids) == len(set(ids)) == 45
    assert set(ids) <= set(corpus.repo_ids[180:])
    queries = corpus.client.query_points.call_args_list
    scrolls = corpus.client.scroll.call_args_list
    assert len(queries) == (2 if semantic else 0)
    assert len(scrolls) == (4 if semantic else 12)
    if semantic:
        assert set(queries[1].kwargs["query_filter"].must_not[0].has_id) == set(corpus.repo_ids[:150])
    else:
        assert set(scrolls[4].kwargs["scroll_filter"].must_not[0].has_id) == set(corpus.repo_ids[:75])
    assert sum(call.kwargs["limit"] for call in queries + scrolls) <= 6 * retriever.max_candidates


def test_shared_discovery_pool_keeps_refills_request_local(local_recommendation_corpus):
    corpus = local_recommendation_corpus
    corpus.client.set_payload(
        corpus.repository_collection, payload={"doc_quality": "invalid"},
        points=corpus.repo_ids[:180],
    )
    corpus.client.delete(
        corpus.user_collection,
        points_selector=models.PointIdsList(points=[corpus.user_id]),
    )
    retriever = QdrantV2Retriever(
        client=corpus.client, repository_collection=corpus.repository_collection,
        user_collection=corpus.user_collection, heavy_ranker_enabled=False,
        discovery_revision_provider=lambda: DiscoveryRevision("corpus-1", "eligible-1"),
    )

    first = retriever.recommend_batch(corpus.user_id, 45, [])
    assert len(first.items) == 45
    assert {item.repo_id for item in first.items} <= set(corpus.repo_ids[180:])
    assert retriever.discovery_cache_status()["entries"] == 1
    assert len(corpus.client.scroll.call_args_list) == 12

    second = retriever.recommend_batch(corpus.user_id, 45, corpus.repo_ids[:500])
    assert len(second.items) == 45
    assert {item.repo_id for item in second.items} <= set(corpus.repo_ids[500:])
    assert retriever.discovery_cache_status()["entries"] == 1
    assert all(
        call.kwargs["scroll_filter"].must_not == [
            models.HasIdCondition(has_id=sorted(corpus.repo_ids[:500]))
        ]
        for call in corpus.client.scroll.call_args_list[12:16]
    )


def test_exhausted_low_yield_corpus_returns_partial_without_repeats(
    local_recommendation_corpus,
):
    corpus = local_recommendation_corpus
    corpus.client.set_payload(
        corpus.repository_collection, payload={"doc_quality": "invalid"},
        points=corpus.repo_ids[500:595],
    )
    retriever = _corpus_retriever(corpus, heavy=False)

    batch = retriever.recommend_batch(corpus.user_id, 45, corpus.repo_ids[:500])

    assert {item.repo_id for item in batch.items} == set(corpus.repo_ids[595:])
    assert len(batch.items) == 5
    assert len(corpus.client.query_points.call_args_list) == 1
    assert len(corpus.client.scroll.call_args_list) == 8
    # User exclusions and refill exclusions never leak into later requests.
    retriever.recommend_batch(corpus.user_id, 2, [])
    assert corpus.client.query_points.call_args.kwargs["query_filter"].must_not is None


def test_always_invalid_candidates_cannot_exceed_shared_read_budget():
    client = HeavyRankerQdrant()
    limits = []

    def invalid_points(**kwargs):
        limits.append(kwargs["limit"])
        points = [client._point(str(uuid.uuid4()), score=0.9) for _ in range(kwargs["limit"])]
        for point in points:
            point.payload["doc_quality"] = "invalid"
        return points

    client.query_points = lambda **kwargs: SimpleNamespace(points=invalid_points(**kwargs))
    client.scroll = lambda **kwargs: (invalid_points(**kwargs), None)
    retriever = QdrantV2Retriever(client=client, heavy_ranker_enabled=False)

    assert retriever.recommend_batch(client.user_id, 45, []).items == []
    assert sum(limits) == 6 * retriever.max_candidates
    assert len(limits) <= 3 + 3 * 4


def test_filter_ignoring_client_stops_when_refill_makes_no_progress():
    client = HeavyRankerQdrant()
    invalid = client._point(client.repo_ids[0], score=0.9)
    invalid.payload["doc_quality"] = float("nan")
    client.query_points = Mock(side_effect=lambda **kwargs: SimpleNamespace(points=[invalid] * kwargs["limit"]))
    client.scroll = Mock(side_effect=lambda **kwargs: ([invalid] * kwargs["limit"], None))
    retriever = QdrantV2Retriever(client=client, heavy_ranker_enabled=False)

    assert retriever.recommend_batch(client.user_id, 45, []).items == []
    assert client.query_points.call_count == 2
    assert client.scroll.call_count == 8


def test_optional_refill_failure_keeps_successful_candidates():
    client = HeavyRankerQdrant()
    valid = client._point(client.repo_ids[0], score=0.9)
    invalid = client._point(client.repo_ids[1], score=0.8)
    invalid.payload["doc_quality"] = "invalid"
    client.query_points = Mock(side_effect=[
        SimpleNamespace(points=[valid] + [invalid] * 149),
        RuntimeError("optional refill unavailable"),
    ])
    client.scroll = Mock(return_value=([], None))
    retriever = QdrantV2Retriever(client=client, heavy_ranker_enabled=False)

    batch = retriever.recommend_batch(client.user_id, 45, [])

    assert [item.repo_id for item in batch.items] == [valid.id]
    assert client.query_points.call_count == 2


def test_expired_candidate_deadline_starts_no_more_qdrant_calls(monkeypatch):
    from inference.runtime import RecommendationDeadlineExceeded
    from retrieval.v2_retriever import _CandidateReadBudget

    client = HeavyRankerQdrant()
    retriever = QdrantV2Retriever(client=client, heavy_ranker_enabled=False)
    budget = _CandidateReadBudget(900, float("inf"))
    monkeypatch.setattr(retriever, "_candidate_read_budget", lambda: budget)
    invalid = client._point(client.repo_ids[0], score=0.9)
    invalid.payload["doc_quality"] = "invalid"

    def expire(**kwargs):
        budget.deadline = 0.0
        return SimpleNamespace(points=[invalid] * kwargs["limit"])

    client.query_points = Mock(side_effect=expire)
    # Initial discovery now fans out alongside profile retrieval. Exercise the
    # semantic/refill boundary directly so this asserts no *new* RPC after expiry.
    assert retriever._semantic(VECTOR, 150, minimum_candidates=45, budget=budget) == []
    with pytest.raises(RecommendationDeadlineExceeded):
        retriever._discovery(150, minimum_candidates=45, budget=budget)
    assert client.query_points.call_count == 1
    assert client.scroll_calls == []


def test_initial_discovery_starts_while_profile_lookup_is_waiting():
    import threading

    client = HeavyRankerQdrant()
    discovery_started = threading.Event()
    retrieve, scroll = client.retrieve, client.scroll

    def waiting_profile(**kwargs):
        assert discovery_started.wait(2), "discovery must overlap profile retrieval"
        return retrieve(**kwargs)

    def signal_discovery(**kwargs):
        discovery_started.set()
        return scroll(**kwargs)

    client.retrieve = waiting_profile
    client.scroll = signal_discovery
    retriever = QdrantV2Retriever(client=client, heavy_ranker_enabled=False)

    assert len(retriever.recommend_batch(client.user_id, 2, []).items) == 2
    assert len(client.scroll_calls) == 4


def test_cooperative_cancellation_stops_refills_and_settles_read_children():
    from retrieval.runtime import retrieval_runtime_status
    from utils.deadline import CooperativeDeadlineExceeded, RequestDeadline, deadline_scope

    client = HeavyRankerQdrant()
    invalid = client._point(client.repo_ids[0], score=0.9)
    invalid.payload["doc_quality"] = "invalid"
    deadline = RequestDeadline(10)

    def cancel_after_read(**kwargs):
        deadline.cancel()
        return SimpleNamespace(points=[invalid] * kwargs["limit"])

    client.query_points = Mock(side_effect=cancel_after_read)
    retriever = QdrantV2Retriever(client=client, heavy_ranker_enabled=False)
    with deadline_scope(deadline), pytest.raises(CooperativeDeadlineExceeded):
        retriever._semantic(VECTOR, 150, minimum_candidates=45)

    assert client.query_points.call_count == 1
    assert 1 <= client.query_points.call_args.kwargs["timeout"] <= 10
    assert retrieval_runtime_status()["outstanding"] == 0


def test_qdrant_only_retrieval_deduplicates_and_rejects_legacy_identity():
    client = FakeQdrant()
    # Isolate hybrid retrieval identity/deduplication; required-heavy repository
    # hydration is covered with HeavyRankerQdrant below.
    retriever = QdrantV2Retriever(client=client, heavy_ranker_enabled=False)
    items = retriever.recommend(client.user_id, 10, [])
    assert {item.repo_id for item in items} == set(client.repo_ids)
    assert len(items) == 2


def test_discovery_score_treats_zero_pushed_days_as_fresh():
    today_score, today_source = QdrantV2Retriever._discovery_score(
        {"pushed_days_ago": 0}
    )
    missing_score, missing_source = QdrantV2Retriever._discovery_score({})

    assert today_source == "fresh"
    assert today_score == pytest.approx(0.1)
    assert missing_source == "popular"
    assert missing_score < today_score


def test_v2_retrieval_applies_diversity_freshness_and_deterministic_exploration():
    client = FakeQdrant()
    client.repo_ids = [str(uuid.uuid4()) for _ in range(8)]
    points = []
    for index, repo_id in enumerate(client.repo_ids):
        points.append(
            SimpleNamespace(
                id=repo_id,
                vector={"repo_embedding": VECTOR},
                payload=_repository_payload(
                    repo_id,
                    stars=100 - index,
                    primary_language="Python" if index < 6 else "Rust",
                    created_at=(
                        datetime.now(timezone.utc).isoformat()
                        if index == 7
                        else None
                    ),
                ),
            )
        )
    client.retrieve = lambda **_kwargs: []
    client.scroll = lambda **_kwargs: (points, None)

    retriever = QdrantV2Retriever(
        client=client,
        assembler=FeedAssemblySystem(max_same_language=2),
        heavy_ranker_enabled=False,
    )
    first = retriever.recommend(client.user_id, 8, [], "fixed-generation")
    second = retriever.recommend(client.user_id, 8, [], "fixed-generation")

    assert first == second
    assert {item.repo_id for item in first} == set(client.repo_ids)
    assert first.index(next(item for item in first if item.repo_id == client.repo_ids[6])) < 6
    assert first[0].repo_id == client.repo_ids[7]


@pytest.mark.parametrize("compact", [False, True])
def test_v2_heavy_ranker_reranks_full_vector_candidates_and_reports_served_model(compact):
    client = HeavyRankerQdrant()
    ranker = DeterministicHeavyRanker()
    if compact:
        ranker.score_batch_compact = ranker.score_batch
        def diagnostic_must_not_run(*args, **kwargs):
            raise AssertionError("V2 should prefer the compact serving API")
        ranker.score_batch = diagnostic_must_not_run
    retriever = QdrantV2Retriever(
        client=client,
        ranker=ranker,
        heavy_ranker_enabled=True,
        heavy_ranker_required=False,
        heavy_ranker_traffic_percent=100,
    )

    batch = retriever.recommend_batch(client.user_id, 2, [], "generation")

    assert batch.ranker_applied is True
    assert batch.heavy_ranker_selected is True
    assert batch.served_ranker == "heavy"
    assert batch.model_version == "heavy-ranker-v2"
    assert batch.embedding_version == REPOSITORY_EMBEDDING_VERSION
    assert [item.repo_id for item in batch.items] == [
        client.repo_ids[1],
        client.repo_ids[0],
    ]
    assert ranker.calls[0][1] == ["Python", "recommendation"]
    assert all(len(candidate["embedding"]) == 384 for candidate in ranker.calls[0][2])
    assert client.query_calls[0]["with_vectors"] is False
    assert all(call["with_vectors"] is False for call in client.scroll_calls)
    vector_calls = [call for call in client.retrieve_calls
                    if call["collection_name"] == retriever.repository_collection]
    assert len(vector_calls) == 1
    assert vector_calls[0]["with_vectors"] == [QDRANT_VECTOR_NAME]
    assert len(vector_calls[0]["ids"]) == len(set(vector_calls[0]["ids"])) == 2
    assert set(vector_calls[0]["ids"]) == set(client.repo_ids)


def test_followed_owner_repositories_receive_only_a_small_capped_boost():
    repo_ids = [str(uuid.uuid4()) for _ in range(8)]
    ranked = [
        RankedRepository(repo_id=repo_id, score=1.0 - index * 0.01, source="semantic")
        for index, repo_id in enumerate(repo_ids)
    ]

    adjusted = QdrantV2Retriever._apply_social_boost(
        ranked,
        {repo_ids[2], repo_ids[3], repo_ids[4]},
        limit=8,
    )

    changed = {
        item.repo_id
        for item, original in zip(
            sorted(adjusted, key=lambda value: repo_ids.index(value.repo_id)),
            ranked,
        )
        if item.score != original.score
    }
    assert len(changed) == 2
    assert changed <= {repo_ids[2], repo_ids[3], repo_ids[4]}
    assert all(item.score - ranked[repo_ids.index(item.repo_id)].score <= 0.03 for item in adjusted)


def test_v2_invalid_heavy_output_falls_back_for_only_that_request():
    client = HeavyRankerQdrant()
    ranker = DeterministicHeavyRanker()
    ranker.score_batch = lambda *_args: []
    retriever = QdrantV2Retriever(
        client=client,
        ranker=ranker,
        heavy_ranker_enabled=True,
        heavy_ranker_required=False,
        heavy_ranker_traffic_percent=100,
        minimum_eligible_repositories=len(client.repo_ids),
    )

    batch = retriever.recommend_batch(client.user_id, 2, [])

    assert batch.ranker_applied is False
    assert batch.heavy_ranker_selected is True
    assert batch.fallback_code == "INVALID_HEAVY_OUTPUT"
    assert batch.model_version == "qdrant-hybrid-v2"
    assert "incomplete or invalid" in batch.fallback_reason
    assert [item.repo_id for item in batch.items] == client.repo_ids
    counters = retriever.health()["ranking_counters"]
    assert counters["requests_total"] == 1
    assert counters["heavy_selected"] == 1
    assert counters["hybrid_served"] == 1
    assert counters["fallback_invalid_heavy_output"] == 1


def test_required_v2_heavy_ranker_fails_closed_instead_of_serving_hybrid():
    client = HeavyRankerQdrant()
    ranker = DeterministicHeavyRanker()
    ranker.score_batch = lambda *_args: []
    retriever = QdrantV2Retriever(
        client=client,
        ranker=ranker,
        heavy_ranker_enabled=True,
        heavy_ranker_required=True,
        heavy_ranker_traffic_percent=100,
        minimum_eligible_repositories=len(client.repo_ids),
    )

    with pytest.raises(RetrievalDependencyError, match="required V2 heavy ranker"):
        retriever.recommend_batch(client.user_id, 2, [])

    counters = retriever.health()["ranking_counters"]
    assert counters["requests_total"] == 1
    assert counters["heavy_selected"] == 1
    assert counters["heavy_served"] == 0
    assert counters["hybrid_served"] == 0
    assert counters["fallback_invalid_heavy_output"] == 1


def test_v2_filters_eligibility_before_limits_and_uses_ordered_discovery():
    client = HeavyRankerQdrant()
    retriever = QdrantV2Retriever(client=client, heavy_ranker_enabled=False)

    retriever.recommend(client.user_id, 2, [])

    # Hybrid serving does not transfer repository vectors that it never uses;
    # Heavy requests retrieve only retained vectors in a separate batch.
    assert client.query_calls[0]["with_vectors"] is False
    assert "description" not in client.query_calls[0]["with_payload"]
    assert "topics" not in client.query_calls[0]["with_payload"]
    assert {
        "repo_id",
        "embedding_version",
        "feature_spec_version",
        "primary_language",
        "created_at",
    } <= set(client.query_calls[0]["with_payload"])
    assert all(
        call["with_payload"] == client.query_calls[0]["with_payload"]
        for call in client.scroll_calls
    )
    semantic_filter = client.query_calls[0]["query_filter"]
    semantic_conditions = {condition.key: condition for condition in semantic_filter.must}
    assert (
        semantic_conditions[REPOSITORY_SERVING_ELIGIBILITY_FIELD].match.value
        == REPOSITORY_SERVING_ELIGIBILITY_VERSION
    )
    assert semantic_conditions["content_version"].range.gte == 1
    assert (
        semantic_conditions["embedding_model"].match.value
        == REPOSITORY_EMBEDDING_MODEL
    )
    assert (
        semantic_conditions["embedding_model_revision"].match.value
        == EMBEDDING_MODEL_REVISION
    )
    assert semantic_conditions["embedding_dim"].match.value == REPOSITORY_EMBEDDING_DIM
    assert set(semantic_conditions["embedding_version"].match.any) == {
        REPOSITORY_EMBEDDING_VERSION
    }
    assert {call["order_by"].key for call in client.scroll_calls} == {
        "trend_velocity",
        "activity_score",
        "star_count",
        "pushed_at",
    }
    assert all(
        {condition.key for condition in call["scroll_filter"].must}
        >= {
            REPOSITORY_SERVING_ELIGIBILITY_FIELD,
            "content_version",
            "embedding_model",
            "embedding_model_revision",
            "embedding_version",
            "embedding_dim",
            "feature_spec_version",
        }
        for call in client.scroll_calls
    )
    assert all(call["with_vectors"] is False for call in client.scroll_calls)


def test_unstamped_payload_only_point_is_not_serving_eligible() -> None:
    client = HeavyRankerQdrant()
    point = client._point(client.repo_ids[0], stars=100)
    point.payload.pop(REPOSITORY_SERVING_ELIGIBILITY_FIELD)
    point.vector = None
    retriever = QdrantV2Retriever(client=client, heavy_ranker_enabled=False)

    candidate = retriever._eligible_repository_point(
        point,
        require_vector=False,
    )

    assert candidate is None


def test_default_recommendation_path_has_a_bounded_qdrant_call_budget() -> None:
    client = HeavyRankerQdrant()
    retriever = QdrantV2Retriever(client=client, heavy_ranker_enabled=False)

    batch = retriever.recommend_batch(
        client.user_id,
        2,
        [],
        "default-call-budget",
    )

    assert len(batch.items) == 2
    assert len(client.retrieve_calls) == 1  # canonical user profile lookup
    assert set(client.retrieve_calls[0]["with_payload"]) == {
        "skills",
        "tech_stack",
        "interests",
        "topics",
        "embedding_dim",
        "embedding_model",
        "embedding_model_revision",
        "model_policy_sha256",  # scalar guard; never fetch the full feedback ledger
    }
    assert len(client.query_calls) == 1  # semantic candidate query
    assert len(client.scroll_calls) == 4  # four bounded discovery channels
    assert (
        len(client.retrieve_calls)
        + len(client.query_calls)
        + len(client.scroll_calls)
        == 6
    )
    assert client.query_calls[0]["with_vectors"] is False
    assert all(call["with_vectors"] is False for call in client.scroll_calls)


def test_production_health_rejects_a_required_unavailable_ranker(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("V2_HEAVY_RANKER_REQUIRED", "true")
    monkeypatch.setenv("V2_ALLOW_UNQUALIFIED_HEAVY_RANKER", "false")
    monkeypatch.setenv("V2_USER_COLLECTION_REQUIRED", "false")
    retriever = QdrantV2Retriever(
        client=HeavyRankerQdrant(),
        ranker=SimpleNamespace(ready=False),
        heavy_ranker_enabled=True,
    )

    with pytest.raises(RuntimeError, match="required but unavailable"):
        retriever.health()


def test_one_failed_discovery_channel_does_not_fail_recommendations():
    client = HeavyRankerQdrant()
    original_scroll = client.scroll
    calls = 0

    def partially_failing_scroll(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary channel failure")
        return original_scroll(**kwargs)

    client.scroll = partially_failing_scroll
    retriever = QdrantV2Retriever(client=client, heavy_ranker_enabled=False)

    items = retriever.recommend(client.user_id, 2, [])

    assert len(items) == 2
    assert calls == 4


def test_heavy_ranker_is_the_default_v2_serving_path(monkeypatch):
    monkeypatch.delenv("V2_HEAVY_RANKER_ENABLED", raising=False)
    monkeypatch.delenv("V2_HEAVY_RANKER_TRAFFIC_PERCENT", raising=False)
    retriever = QdrantV2Retriever(
        client=HeavyRankerQdrant(),
        ranker=DeterministicHeavyRanker(),
    )

    assert retriever.heavy_ranker_enabled is True
    assert retriever.heavy_ranker_traffic_percent == 100.0
    assert retriever.ranker is not None


def test_heavy_ranker_canary_selection_is_deterministic_and_partitioned():
    retriever = QdrantV2Retriever(
        client=HeavyRankerQdrant(),
        ranker=DeterministicHeavyRanker(),
        heavy_ranker_enabled=True,
        heavy_ranker_required=False,
        heavy_ranker_traffic_percent=50,
    )
    user_ids = [str(uuid.UUID(int=index)) for index in range(1, 101)]

    first = [retriever._heavy_ranker_selected(user_id) for user_id in user_ids]
    second = [retriever._heavy_ranker_selected(user_id) for user_id in user_ids]

    assert first == second
    assert any(first)
    assert not all(first)


def test_cold_start_with_onboarding_profile_uses_required_heavy_ranker():
    client = HeavyRankerQdrant()
    ranker = DeterministicHeavyRanker()
    retriever = QdrantV2Retriever(
        client=client,
        ranker=ranker,
        heavy_ranker_enabled=True,
        heavy_ranker_traffic_percent=100,
    )

    batch = retriever.recommend_batch(
        client.user_id,
        2,
        [],
        "cold-start-generation",
        True,
    )

    assert batch.retrieval_mode == "cold_start_profile"
    assert batch.served_ranker == "heavy"
    assert batch.heavy_ranker_selected is True
    assert batch.ranker_applied is True
    assert ranker.calls
    assert client.query_calls  # the initial profile still personalizes retrieval


def test_required_v2_cold_start_waits_for_onboarding_vector_without_hybrid():
    client = HeavyRankerQdrant()
    client.retrieve = lambda **_kwargs: []
    retriever = QdrantV2Retriever(
        client=client,
        ranker=DeterministicHeavyRanker(),
        heavy_ranker_enabled=True,
        heavy_ranker_required=True,
        heavy_ranker_traffic_percent=100,
        minimum_eligible_repositories=len(client.repo_ids),
    )

    with pytest.raises(
        RetrievalDependencyError,
        match="user profile vector is unavailable",
    ):
        retriever.recommend_batch(
            client.user_id,
            2,
            [],
            "cold-start-vector-pending",
            True,
        )

    counters = retriever.health()["ranking_counters"]
    assert counters["heavy_served"] == 0
    assert counters["hybrid_served"] == 0
    assert counters["fallback_user_vector_unavailable"] == 1


def test_missing_profile_modes_are_explicit_for_cold_and_normal_requests():
    client = HeavyRankerQdrant()
    client.retrieve = lambda **_kwargs: []
    retriever = QdrantV2Retriever(client=client, heavy_ranker_enabled=False)

    cold = retriever.recommend_batch(client.user_id, 2, [], "cold", True)
    normal = retriever.recommend_batch(client.user_id, 2, [], "normal", False)

    assert cold.retrieval_mode == "cold_start_discovery"
    assert normal.retrieval_mode == "profile_missing_discovery"
    assert client.query_calls == []
    assert len(cold.items) == len(normal.items) == 2


def test_batch_reports_only_the_v2_embedding_version():
    client = HeavyRankerQdrant()
    original_point = client._point

    def versioned_point(repo_id, **kwargs):
        point = original_point(repo_id, **kwargs)
        point.payload["embedding_version"] = "repo-embedding-v2"
        return point

    client._point = versioned_point
    retriever = QdrantV2Retriever(
        client=client,
        heavy_ranker_enabled=False,
        compatible_embedding_versions={"repo-embedding-v2"},
    )

    batch = retriever.recommend_batch(client.user_id, 2, [], "mixed")

    assert batch.embedding_versions == ("repo-embedding-v2",)
    assert batch.embedding_version == "repo-embedding-v2"


def test_incompatible_revision_is_never_served_and_empty_batch_is_truthful():
    client = HeavyRankerQdrant()
    original_point = client._point

    def incompatible_point(repo_id, **kwargs):
        point = original_point(repo_id, **kwargs)
        point.payload["embedding_model_revision"] = "wrong-revision"
        return point

    client._point = incompatible_point
    retriever = QdrantV2Retriever(client=client, heavy_ranker_enabled=False)

    batch = retriever.recommend_batch(client.user_id, 2, [], "incompatible")

    assert batch.items == []
    assert batch.embedding_version == "none"
    assert batch.embedding_versions == ()


def test_health_validates_contract_and_fast_minimum_eligible_count():
    client = HeavyRankerQdrant()
    retriever = QdrantV2Retriever(
        client=client,
        heavy_ranker_enabled=False,
        minimum_eligible_repositories=2,
    )

    health = retriever.health()

    assert health["repository_collection_contract"] == "healthy"
    assert health["qdrant_server_version"] == "1.18.2"
    assert health["minimum_qdrant_server_version"] == "1.18.0"
    assert health["eligible_repository_points"] == 2
    assert health["eligible_repository_count_exact"] is True
    assert health["embedding_model_revision"] == EMBEDDING_MODEL_REVISION
    assert (
        health["serving_eligibility_version"]
        == REPOSITORY_SERVING_ELIGIBILITY_VERSION
    )
    assert (
        health["serving_eligibility_evidence"]
        == "validated_vector_at_atomic_upsert"
    )
    assert health["compatible_embedding_versions"] == [
        REPOSITORY_EMBEDDING_VERSION
    ]
    count_conditions = {
        condition.key
        for condition in client.count_calls[0]["count_filter"].must
    }
    assert REPOSITORY_SERVING_ELIGIBILITY_FIELD in count_conditions

    too_small = QdrantV2Retriever(
        client=client,
        heavy_ranker_enabled=False,
        minimum_eligible_repositories=3,
    )
    with pytest.raises(RuntimeError, match="below the configured minimum"):
        too_small.health()


def test_health_rejects_wrong_repository_collection_dimension():
    client = HeavyRankerQdrant()
    original_get_collection = client.get_collection

    def wrong_collection(collection_name):
        info = original_get_collection(collection_name)
        if collection_name != "user_profiles":
            info.config.params.vectors["repo_embedding"].size = 768
        return info

    client.get_collection = wrong_collection
    retriever = QdrantV2Retriever(client=client, heavy_ranker_enabled=False)

    with pytest.raises(RuntimeError, match="incompatible vector size"):
        retriever.health()


def test_health_rejects_qdrant_without_conditional_write_support():
    client = HeavyRankerQdrant()
    client.info = lambda: SimpleNamespace(version="1.17.4")
    retriever = QdrantV2Retriever(client=client, heavy_ranker_enabled=False)

    with pytest.raises(RuntimeError, match="1.18.0 or newer"):
        retriever.health()


def test_health_rejects_a_missing_required_payload_index():
    client = HeavyRankerQdrant()
    original_get_collection = client.get_collection

    def missing_index(collection_name):
        info = original_get_collection(collection_name)
        if collection_name != "user_profiles":
            info.payload_schema.pop("embedding_version")
        return info

    client.get_collection = missing_index
    retriever = QdrantV2Retriever(client=client, heavy_ranker_enabled=False)

    with pytest.raises(RuntimeError, match="embedding_version.*missing"):
        retriever.health()


@pytest.mark.parametrize("change", ["deleted", "ineligible", "version", "embedding", "feature", "shape", "nonfinite"])
def test_retained_vector_batch_rejects_changed_points_before_any_fallback(change):
    client = HeavyRankerQdrant()
    ranker = DeterministicHeavyRanker()
    # An optional ranker failure must not reintroduce a rejected snapshot.
    ranker.score_batch = lambda *_args: []
    retriever = QdrantV2Retriever(
        client=client, ranker=ranker, heavy_ranker_enabled=True,
        heavy_ranker_required=False, heavy_ranker_traffic_percent=100,
    )
    original_retrieve = client.retrieve

    def retrieve(**kwargs):
        records = original_retrieve(**kwargs)
        if kwargs["collection_name"] != retriever.repository_collection:
            return records
        bad = next(point for point in records if point.id == client.repo_ids[0])
        if change == "deleted":
            return [point for point in records if point is not bad]
        if change == "ineligible":
            bad.payload.pop(REPOSITORY_SERVING_ELIGIBILITY_FIELD)
        elif change == "version":
            bad.payload["content_version"] += 1
        elif change == "embedding":
            bad.payload["embedding_version"] = "incompatible-version"
        elif change == "feature":
            bad.payload["feature_version"] = 99
        elif change == "shape":
            bad.vector = {QDRANT_VECTOR_NAME: [1.0]}
        elif change == "nonfinite":
            bad.vector = {QDRANT_VECTOR_NAME: [float("nan")] * REPOSITORY_EMBEDDING_DIM}
        return records

    client.retrieve = retrieve
    batch = retriever.recommend_batch(client.user_id, 2, [], "changed-point")

    assert [item.repo_id for item in batch.items] == [client.repo_ids[1]]
    assert batch.ranker_applied is False
    vector_call = client.retrieve_calls[-1]
    assert set(vector_call["ids"]) == set(client.repo_ids)
    assert len(vector_call["ids"]) == 2


def test_vector_batch_only_fetches_deduplicated_retained_ids():
    client = HeavyRankerQdrant()
    client.repo_ids = [str(uuid.UUID(int=index + 1)) for index in range(80)]
    ranker = DeterministicHeavyRanker()
    def score_batch(vector, skills, candidates):
        ranker.calls.append((vector, skills, candidates))
        return [
            {"repo_id": item["id"], "final_score": 1 - index / 100}
            for index, item in enumerate(candidates)
        ]
    ranker.score_batch = score_batch
    original_query = client.query_points

    def query(**kwargs):
        response = original_query(**kwargs)
        # Many dropped, overlapping semantic candidates, with stable tie order.
        response.points = [client._project(
            client._point(repo_id, score=0.5, stars=100 if index == 0 else 10),
            kwargs.get("with_payload"),
        ) for index, repo_id in enumerate(client.repo_ids)]
        return response

    client.query_points = query
    retriever = QdrantV2Retriever(
        client=client, ranker=ranker, max_candidates=50,
        heavy_ranker_enabled=True, heavy_ranker_required=True,
    )
    retriever.recommend_batch(client.user_id, 2, [], "retained-only")

    vector_calls = [call for call in client.retrieve_calls
                    if call["collection_name"] == retriever.repository_collection]
    assert len(vector_calls) == 1
    ids = vector_calls[0]["ids"]
    assert len(ids) == len(set(ids)) == 50
    assert set(ids) == {item["id"] for item in ranker.calls[0][2]}
    assert not set(ids) & set(client.repo_ids[50:])


@pytest.mark.parametrize("required", [False, True])
def test_retained_vector_dependency_failure_preserves_required_heavy_policy(required):
    client = HeavyRankerQdrant()
    ranker = DeterministicHeavyRanker()
    retriever = QdrantV2Retriever(
        client=client, ranker=ranker, heavy_ranker_enabled=True,
        heavy_ranker_required=required, heavy_ranker_traffic_percent=100,
    )
    original_retrieve = client.retrieve

    def retrieve(**kwargs):
        if kwargs["collection_name"] == retriever.repository_collection:
            raise TimeoutError("vector dependency unavailable")
        return original_retrieve(**kwargs)

    client.retrieve = retrieve
    if required:
        with pytest.raises(RetrievalDependencyError, match="required V2 heavy ranker"):
            retriever.recommend_batch(client.user_id, 2, [])
    else:
        batch = retriever.recommend_batch(client.user_id, 2, [])
        assert batch.ranker_applied is False
        assert batch.fallback_code == "CANDIDATE_VECTORS_UNAVAILABLE"
        assert {item.repo_id for item in batch.items} == set(client.repo_ids)
    assert not ranker.calls


def test_shared_discovery_requires_authoritative_revision_and_never_caches_exclusions():
    client = HeavyRankerQdrant()
    revision = [DiscoveryRevision(corpus="corpus-1", eligibility="eligible-1")]
    retriever = QdrantV2Retriever(
        client=client, heavy_ranker_enabled=False,
        discovery_revision_provider=lambda: revision[0],
    )
    first = retriever.recommend_batch(client.user_id, 2, [], "first")
    second = retriever.recommend_batch(str(uuid.uuid4()), 2, [], "second")
    assert len(client.scroll_calls) == 4
    assert {item.repo_id for item in first.items} == {item.repo_id for item in second.items}
    assert all(call["scroll_filter"].must_not is None for call in client.scroll_calls)

    excluded = client.repo_ids[0]
    filtered = retriever.recommend_batch(client.user_id, 2, [excluded], "exclude")
    assert excluded not in {item.repo_id for item in filtered.items}
    # A bounded shared pool cannot let a user's history hide deeper results.
    assert len(client.scroll_calls) == 8
    assert all(call["scroll_filter"].must_not == [models.HasIdCondition(has_id=[excluded])]
               for call in client.scroll_calls[-4:])
    # Personalized filtering/refill must not replace the shared unfiltered pool.
    again = retriever.recommend_batch(client.user_id, 2, [], "again")
    assert {item.repo_id for item in again.items} == set(client.repo_ids)
    assert len(client.scroll_calls) == 8

    revision[0] = DiscoveryRevision(corpus="corpus-2", eligibility="eligible-1")
    retriever.recommend_batch(client.user_id, 2, [])
    assert len(client.scroll_calls) == 12
    revision[0] = None
    retriever.recommend_batch(client.user_id, 2, [])
    retriever.recommend_batch(client.user_id, 2, [])
    assert len(client.scroll_calls) == 20


def test_shared_discovery_invalidates_eligibility_model_and_feature_contracts():
    client = HeavyRankerQdrant()
    revision = [DiscoveryRevision(corpus="corpus-1", eligibility="eligible-1")]
    retriever = QdrantV2Retriever(
        client=client, heavy_ranker_enabled=False,
        discovery_revision_provider=lambda: revision[0],
    )
    retriever.recommend_batch(client.user_id, 2, [])
    revision[0] = DiscoveryRevision(corpus="corpus-1", eligibility="eligible-2")
    retriever.recommend_batch(client.user_id, 2, [])
    assert len(client.scroll_calls) == 8
    # Changing an eligibility/model feature contract cannot hit a previous key.
    retriever.required_content_version += 1
    retriever.recommend_batch(client.user_id, 2, [])
    assert len(client.scroll_calls) == 12
    retriever.embedding_model_revision = "new-model-revision"
    retriever.recommend_batch(client.user_id, 2, [])
    assert len(client.scroll_calls) == 16
    retriever.required_feature_spec_version = "new-feature-contract"
    retriever.recommend_batch(client.user_id, 2, [])
    assert len(client.scroll_calls) == 20


def test_shared_discovery_cache_ttl_byte_bounds_and_independent_copies():
    now = [0.0]
    cache = SharedDiscoveryCache(max_entries=2, max_bytes=8, ttl_seconds=5,
                                 clock=lambda: now[0])
    owner = cache.reserve(("one",))
    assert owner.owner
    cache.publish(owner, b"1234")
    hit = cache.reserve(("one",))
    assert cache.result(hit) == b"1234"
    cache.release(hit)
    for key in ("two", "three"):
        owner = cache.reserve((key,))
        cache.publish(owner, b"5678")
    assert cache.snapshot()["entries"] == 2
    assert cache.snapshot()["payload_bytes"] == 8
    now[0] = 5.0
    assert cache.snapshot()["entries"] == 0
    assert cache.snapshot()["payload_bytes"] == 0
    owner = cache.reserve(("oversized",))
    cache.publish(owner, b"012345678")
    assert cache.snapshot()["entries"] == 0
    assert cache.result(owner) == b"012345678"
    # TTL is measured from refresh start, not extended by a slow owner.
    slow_owner = cache.reserve(("slow",))
    now[0] = 10.0
    cache.publish(slow_owner, b"late")
    assert cache.snapshot()["entries"] == 0
    with pytest.raises(TimeoutError, match="expired"):
        cache.result(slow_owner)


def test_shared_discovery_cache_coalesces_and_releases_failed_flights():
    cache = SharedDiscoveryCache(max_flights=1, max_waiters=1)
    owner = cache.reserve(("same",))
    waiter = cache.reserve(("same",))
    assert owner.owner and not waiter.owner
    assert cache.reserve(("other",)) is None
    assert cache.reserve(("same",)) is None
    cache.fail(owner)
    with pytest.raises(RuntimeError, match="shared discovery refresh unavailable"):
        cache.result(waiter)
    cache.release(waiter)
    assert cache.snapshot()["flights"] == cache.snapshot()["waiters"] == 0
    replacement = cache.reserve(("same",))
    cache.publish(replacement, b"fresh")
    assert cache.result(replacement) == b"fresh"


def test_shared_discovery_does_not_publish_a_pool_across_revision_change():
    client = HeavyRankerQdrant()
    revision = [DiscoveryRevision(corpus="before", eligibility="eligible")]
    original_scroll = client.scroll

    def scroll(**kwargs):
        rows = original_scroll(**kwargs)
        revision[0] = DiscoveryRevision(corpus="after", eligibility="eligible")
        return rows

    client.scroll = scroll
    retriever = QdrantV2Retriever(
        client=client, heavy_ranker_enabled=False,
        discovery_revision_provider=lambda: revision[0],
    )
    retriever.recommend_batch(client.user_id, 2, [])
    # Revision drift discards shared results and falls back to filtered reads.
    assert retriever.discovery_cache_status()["entries"] == 0
    assert len(client.scroll_calls) == 8


def test_concurrent_discovery_misses_share_one_actual_pool_refresh():
    client = HeavyRankerQdrant()
    entered, release = threading.Event(), threading.Event()
    original_scroll = client.scroll

    def scroll(**kwargs):
        entered.set()
        assert release.wait(3)
        return original_scroll(**kwargs)

    client.scroll = scroll
    retriever = QdrantV2Retriever(
        client=client, heavy_ranker_enabled=False,
        discovery_revision_provider=lambda: DiscoveryRevision("corpus", "eligible"),
    )
    with ThreadPoolExecutor(max_workers=2) as callers:
        owner = callers.submit(retriever.recommend_batch, client.user_id, 2, [])
        assert entered.wait(3)
        waiter = callers.submit(retriever.recommend_batch, str(uuid.uuid4()), 2, [])
        deadline = time.monotonic() + 3
        try:
            while retriever.discovery_cache_status()["waiters"] != 1:
                assert time.monotonic() < deadline
                time.sleep(0.01)
        finally:
            release.set()
        first, second = owner.result(timeout=3), waiter.result(timeout=3)
    assert len(client.scroll_calls) == 4
    assert {item.repo_id for item in first.items} == set(client.repo_ids)
    assert {item.repo_id for item in second.items} == set(client.repo_ids)
    assert retriever.discovery_cache_status()["coalesced"] == 1


def test_cached_discovery_materializes_request_local_payloads_and_bypasses_failed_authority():
    client = HeavyRankerQdrant()
    revision = [DiscoveryRevision("corpus", "eligible")]

    def provider():
        if revision[0] is None:
            raise RuntimeError("authority unavailable")
        return revision[0]

    retriever = QdrantV2Retriever(
        client=client, heavy_ranker_enabled=False, discovery_revision_provider=provider,
    )
    first = retriever._discovery(2, include_vectors=False)
    first[0][0].payload["star_count"] = -999
    second = retriever._discovery(2, include_vectors=False)
    assert second[0][0].payload["star_count"] >= 0
    assert first[0][0].payload is not second[0][0].payload
    assert len(client.scroll_calls) == 4
    revision[0] = None
    retriever._discovery(2, include_vectors=False)
    retriever._discovery(2, include_vectors=False)
    assert len(client.scroll_calls) == 12


def test_partial_discovery_refresh_is_never_cached_or_lost():
    client = HeavyRankerQdrant()
    original_scroll = client.scroll

    def scroll(**kwargs):
        if kwargs["order_by"].key != "star_count":
            raise TimeoutError("channel unavailable")
        return original_scroll(**kwargs)

    client.scroll = scroll
    retriever = QdrantV2Retriever(
        client=client, heavy_ranker_enabled=False,
        discovery_revision_provider=lambda: DiscoveryRevision("corpus", "eligible"),
    )
    points = retriever._discovery(2, include_vectors=False)
    assert {point.repo_id for point, _source in points} == set(client.repo_ids)
    assert retriever.discovery_cache_status()["entries"] == 0
    assert retriever.discovery_cache_status()["flights"] == 0


def test_discovery_waiter_deadline_does_not_cancel_shared_owner():
    from utils.deadline import CooperativeDeadlineExceeded, RequestDeadline, deadline_scope

    cache = SharedDiscoveryCache()
    owner = cache.reserve(("same",))
    waiter = cache.reserve(("same",))
    deadline = RequestDeadline(1)
    deadline.cancel()
    with deadline_scope(deadline), pytest.raises(CooperativeDeadlineExceeded):
        cache.result(waiter)
    cache.release(waiter)
    assert not owner.future.cancelled()
    assert cache.snapshot()["flights"] == 1
    cache.publish(owner, b"valid")
    assert cache.result(owner) == b"valid"
    assert cache.snapshot()["waiters"] == 0


def test_cached_discovery_refill_failure_keeps_checked_missing_profile_results():
    client = HeavyRankerQdrant()
    retriever = QdrantV2Retriever(
        client=client, heavy_ranker_enabled=False,
        discovery_revision_provider=lambda: DiscoveryRevision("corpus", "eligible"),
    )
    retriever.recommend_batch(client.user_id, 2, [])
    # No semantic candidates can hide a dropped discovery fallback.
    client.retrieve = Mock(return_value=[])
    client.scroll = Mock(side_effect=TimeoutError("refill unavailable"))

    batch = retriever.recommend_batch(client.user_id, 2, [client.repo_ids[0]])

    assert batch.retrieval_mode == "profile_missing_discovery"
    assert [item.repo_id for item in batch.items] == [client.repo_ids[1]]
    assert len(client.query_calls) == 1  # Only the warm request was personalized.
    assert client.scroll.call_count == 4
    assert retriever.discovery_cache_status()["entries"] == 1


def test_oversized_shared_pool_preserves_owner_candidates_on_fallback():
    client = HeavyRankerQdrant()
    client.retrieve = Mock(return_value=[])
    retriever = QdrantV2Retriever(
        client=client, heavy_ranker_enabled=False,
        discovery_revision_provider=lambda: DiscoveryRevision("corpus", "eligible"),
        discovery_cache=SharedDiscoveryCache(max_bytes=1),
    )

    batch = retriever.recommend_batch(client.user_id, 3, [])

    assert batch.retrieval_mode == "profile_missing_discovery"
    assert {item.repo_id for item in batch.items} == set(client.repo_ids)
    assert len(client.scroll_calls) == 8
    assert retriever.discovery_cache_status()["entries"] == 0


def test_revision_change_retries_without_stale_owner_exclusions():
    client = HeavyRankerQdrant()
    calls = [0]

    def revision():
        calls[0] += 1
        return DiscoveryRevision("before" if calls[0] == 1 else "after", "eligible")

    retriever = QdrantV2Retriever(
        client=client, heavy_ranker_enabled=False,
        discovery_revision_provider=revision,
    )

    points = retriever._discovery(3, include_vectors=False)

    assert {point.repo_id for point, _source in points} == set(client.repo_ids)
    assert len(client.scroll_calls) == 8
    assert all(call["scroll_filter"].must_not is None for call in client.scroll_calls[4:])
    assert retriever.discovery_cache_status()["entries"] == 0


def test_owner_pool_expiry_retries_without_stale_read_exclusions(monkeypatch):
    client = HeavyRankerQdrant()
    now = [0.0]
    retriever = QdrantV2Retriever(
        client=client, heavy_ranker_enabled=False,
        discovery_revision_provider=lambda: DiscoveryRevision("corpus", "eligible"),
        discovery_cache=SharedDiscoveryCache(ttl_seconds=5, clock=lambda: now[0]),
    )
    encode = retriever._encode_discovery_pool

    def expire_before_publish(*args, **kwargs):
        now[0] = 5.0
        return encode(*args, **kwargs)

    monkeypatch.setattr(retriever, "_encode_discovery_pool", expire_before_publish)

    points = retriever._discovery(3, include_vectors=False)

    assert {point.repo_id for point, _source in points} == set(client.repo_ids)
    assert len(client.scroll_calls) == 8
    assert all(call["scroll_filter"].must_not is None for call in client.scroll_calls[4:])
    assert retriever.discovery_cache_status()["entries"] == 0


def test_successful_discovery_refill_merges_without_losing_checked_candidates():
    client = HeavyRankerQdrant()
    retriever = QdrantV2Retriever(
        client=client, heavy_ranker_enabled=False,
        discovery_revision_provider=lambda: DiscoveryRevision("corpus", "eligible"),
    )
    retriever._discovery(3, include_vectors=False)
    new_id = str(uuid.uuid4())
    client.scroll = Mock(side_effect=lambda **_kwargs: ([
        client._point(client.repo_ids[1], stars=10),
        client._point(new_id, stars=20),
    ], None))

    points = retriever._discovery(
        3, include_vectors=False, exclude_repo_ids={client.repo_ids[0]},
    )

    ids = [point.repo_id for point, _source in points]
    assert len(ids) == len(set(ids)) == 2
    assert ids == [client.repo_ids[1], new_id]
    assert client.scroll.call_count == 4
    # Personalized refill must not replace or augment shared pool contents.
    shared_again = retriever._discovery(3, include_vectors=False)
    assert {point.repo_id for point, _source in shared_again} == set(client.repo_ids)
    assert client.scroll.call_count == 4


@pytest.mark.parametrize("invalidation", ["revision", "ttl"])
@pytest.mark.parametrize("refill_succeeds", [False, True])
def test_refill_never_reuses_a_pool_that_expires_or_changes_during_reads(
    invalidation, refill_succeeds,
):
    client = HeavyRankerQdrant()
    now = [0.0]
    revision = [DiscoveryRevision("before", "eligible")]
    retriever = QdrantV2Retriever(
        client=client, heavy_ranker_enabled=False,
        discovery_revision_provider=lambda: revision[0],
        discovery_cache=SharedDiscoveryCache(ttl_seconds=5, clock=lambda: now[0]),
    )
    retriever._discovery(3, include_vectors=False)
    new_id = str(uuid.uuid4())

    def refill(**_kwargs):
        if invalidation == "revision":
            revision[0] = DiscoveryRevision("after", "eligible")
        else:
            now[0] = 5.0
        if not refill_succeeds:
            raise TimeoutError("refill unavailable")
        return [client._point(new_id, stars=20)], None

    client.scroll = refill
    if refill_succeeds:
        points = retriever._discovery(
            3, include_vectors=False, exclude_repo_ids={client.repo_ids[0]},
        )
        assert [point.repo_id for point, _source in points] == [new_id]
    else:
        with pytest.raises(RetrievalDependencyError):
            retriever._discovery(
                3, include_vectors=False, exclude_repo_ids={client.repo_ids[0]},
            )


def test_partial_owner_refill_cannot_extend_its_original_pool_ttl():
    client = HeavyRankerQdrant()
    now = [0.0]
    calls = [0]
    lock = threading.Lock()

    def scroll(**_kwargs):
        with lock:
            calls[0] += 1
            call = calls[0]
        if call == 1:
            return [client._point(repo_id, stars=10) for repo_id in client.repo_ids], None
        if call > 4:
            now[0] = 5.0  # The uncached partial owner's original lease expires.
        raise TimeoutError("remaining discovery/refill channels unavailable")

    client.scroll = scroll
    retriever = QdrantV2Retriever(
        client=client, heavy_ranker_enabled=False,
        discovery_revision_provider=lambda: DiscoveryRevision("corpus", "eligible"),
        discovery_cache=SharedDiscoveryCache(ttl_seconds=5, clock=lambda: now[0]),
    )
    with pytest.raises(RetrievalDependencyError):
        retriever._discovery(
            3, include_vectors=False, exclude_repo_ids={client.repo_ids[0]},
        )
    assert calls[0] == 8
    assert retriever.discovery_cache_status()["entries"] == 0
