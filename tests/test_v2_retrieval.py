import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from config import (
    EMBEDDING_MODEL_REVISION,
    QDRANT_DISTANCE,
    QDRANT_PAYLOAD_INDEX_SCHEMA,
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


def test_qdrant_only_retrieval_deduplicates_and_rejects_legacy_identity():
    client = FakeQdrant()
    retriever = QdrantV2Retriever(client=client)
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
    assert client.query_calls[0]["with_vectors"] == ["repo_embedding"]
    assert all(
        call["with_vectors"] == ["repo_embedding"] for call in client.scroll_calls
    )


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
    # heavy-canary requests opt into vectors and validate them before scoring.
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
