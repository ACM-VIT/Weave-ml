"""Deterministic, network-free benchmarks for the production v2 hot paths."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from api.contracts import RepositoryRefreshJob
from api.v2 import _refresh_repository_job_locked
from config import (
    EMBEDDING_MODEL_REVISION,
    QDRANT_VECTOR_NAME,
    REPOSITORY_EMBEDDING_DIM,
    REPOSITORY_EMBEDDING_MODEL,
    REPOSITORY_EMBEDDING_VERSION,
    REPOSITORY_FEATURE_SPEC_VERSION,
)
from embedding.vector_contract import repository_payload_defaults
from embedding.vector_contract import (
    REPOSITORY_SERVING_ELIGIBILITY_FIELD,
    REPOSITORY_SERVING_ELIGIBILITY_VERSION,
)
from feedback.v2 import DurableFeedbackProducer, OrderedFeedbackApplier
from feedback.v2_settings import V2FeedbackSettings
from inference.feed_assembly import FeedAssemblySystem
from inference.ranker_service import RankerService
from retrieval.v2_retriever import QdrantV2Retriever
from scripts.user_onboarding import UserOnboardingPipeline


pytestmark = pytest.mark.benchmark
_TIMESTAMP = "2026-07-21T00:00:00+00:00"


def _vector(index: int = 0) -> list[float]:
    vector = [0.0] * REPOSITORY_EMBEDDING_DIM
    vector[index % REPOSITORY_EMBEDDING_DIM] = 1.0
    return vector


def _payload(repo_id: str, index: int) -> dict:
    payload = repository_payload_defaults()
    payload.update(
        {
            "repo_id": repo_id,
            "github_id": str(10_000 + index),
            "full_name": f"benchmark/repository-{index}",
            "description": "bounded benchmark repository",
            "primary_language": f"language-{index % 12}",
            "languages": [f"language-{index % 12}"],
            "topics": ["benchmark"],
            "star_count": 10_000 - index,
            "pushed_days_ago": index % 90,
            "created_at": _TIMESTAMP,
            "updated_at": _TIMESTAMP,
            "pushed_at": _TIMESTAMP,
            "doc_quality": 0.8,
            "code_health": 0.7,
            "activity_score": 0.6,
            "trend_velocity": 0.5,
            "content_version": 1,
            "content_hash": f"content-{index}",
            "indexed_at": _TIMESTAMP,
            "source_hash": f"source-{index}",
        }
    )
    return payload


def _ranked_candidates(count: int = 150) -> list[dict]:
    return [
        {
            "repo_id": str(uuid.UUID(int=index + 1)),
            "score": float(count - index),
            "final_score": float(count - index),
            "source": "semantic",
            "primary_language": f"language-{index % 12}",
        }
        for index in range(count)
    ]


def test_feed_shaping_150_to_15_benchmark(benchmark) -> None:
    assembler = FeedAssemblySystem(max_same_language=5)
    candidates = _ranked_candidates()
    result = benchmark(
        assembler.shape_batch,
        candidates,
        target_size=15,
    )
    assert len(result) == 15


class _RetrievalClient:
    def __init__(self) -> None:
        self.user_id = str(uuid.uuid4())
        self.points = [
            SimpleNamespace(
                id=(repo_id := str(uuid.UUID(int=index + 1))),
                payload=_payload(repo_id, index),
                vector={QDRANT_VECTOR_NAME: _vector(index)},
                score=1.0 - index / 1_000,
            )
            for index in range(150)
        ]
        self.user = SimpleNamespace(
            id=self.user_id,
            vector=_vector(),
            payload={
                "user_id": self.user_id,
                "embedding_dim": REPOSITORY_EMBEDDING_DIM,
                "embedding_model": REPOSITORY_EMBEDDING_MODEL,
                "embedding_model_revision": EMBEDDING_MODEL_REVISION,
                "topics": ["benchmark"],
            },
        )

    def retrieve(self, **_kwargs):
        return [self.user]

    def query_points(self, **_kwargs):
        return SimpleNamespace(points=self.points)

    def scroll(self, **_kwargs):
        return self.points, None


def test_recommendation_generation_150_candidates_benchmark(benchmark) -> None:
    client = _RetrievalClient()
    retriever = QdrantV2Retriever(
        client=client,
        max_candidates=150,
        heavy_ranker_enabled=False,
        compatible_embedding_versions={REPOSITORY_EMBEDDING_VERSION},
    )
    result = benchmark(
        retriever.recommend_batch,
        client.user_id,
        15,
        [],
        "benchmark-generation",
        False,
    )
    assert len(result.items) == 15


def test_heavy_ranker_scores_150_candidates_benchmark(benchmark) -> None:
    service = RankerService(
        model_path="inference/heavy_ranker.pt",
        scaler_path="inference/feature_scaler.json",
        manifest_path="inference/model_manifest.json",
    )
    candidates = [
        {
            "id": str(uuid.UUID(int=index + 1)),
            "embedding": _vector(index),
            "languages": ["Python"],
            "topics": ["ml"],
            "tags": [],
            "doc_quality": 0.8,
            "code_health": 0.7,
            "readme_length": 2_000,
            "star_count": 1_000,
            "fork_count": 100,
            "open_issues_count": 10,
            "pushed_days_ago": 5,
            "activity_score": 0.6,
            "trend_velocity": 0.5,
        }
        for index in range(150)
    ]
    result = benchmark(service.score_batch, _vector(), ["Python", "ml"], candidates)
    assert len(result) == 150


class _DuplicateRedis:
    def eval(self, *_args):
        return "duplicate"


def test_duplicate_feedback_submission_benchmark(benchmark) -> None:
    producer = DurableFeedbackProducer(_DuplicateRedis(), V2FeedbackSettings.from_env())
    event = {
        "event_id": str(uuid.uuid4()),
        "user_id": str(uuid.uuid4()),
        "repo_id": str(uuid.uuid4()),
        "feedback_version": 1,
        "event_type": "like",
        "occurred_at": _TIMESTAMP,
    }
    result = benchmark(producer.enqueue, [event])
    assert result == (0, 1)


class _FeedbackQdrant:
    def __init__(self, user_id: str, repo_id: str) -> None:
        self.user_id = user_id
        self.repo_id = repo_id
        self.reset()

    def reset(self) -> None:
        self.user = SimpleNamespace(
            id=self.user_id,
            vector=_vector(),
            payload={"user_id": self.user_id, "last_feedback_version": 0},
        )

    def retrieve(self, *, collection_name, **_kwargs):
        if collection_name == "user_profiles":
            return [self.user]
        return [
            SimpleNamespace(
                id=self.repo_id,
                vector={QDRANT_VECTOR_NAME: _vector(1)},
                payload={
                    "repo_id": self.repo_id,
                    REPOSITORY_SERVING_ELIGIBILITY_FIELD:
                        REPOSITORY_SERVING_ELIGIBILITY_VERSION,
                    "content_version": 1,
                    "embedding_model": REPOSITORY_EMBEDDING_MODEL,
                    "embedding_model_revision": EMBEDDING_MODEL_REVISION,
                    "embedding_version": REPOSITORY_EMBEDDING_VERSION,
                    "embedding_dim": REPOSITORY_EMBEDDING_DIM,
                    "feature_spec_version": REPOSITORY_FEATURE_SPEC_VERSION,
                },
            )
        ]

    def upsert(self, *, points, **_kwargs):
        self.user = points[0]


def test_consumer_apply_latency_benchmark(benchmark) -> None:
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    qdrant = _FeedbackQdrant(user_id, repo_id)
    applier = OrderedFeedbackApplier(qdrant, V2FeedbackSettings.from_env())
    event = {
        "event_id": str(uuid.uuid4()),
        "user_id": user_id,
        "repo_id": repo_id,
        "feedback_version": 1,
        "event_type": "like",
        "occurred_at": _TIMESTAMP,
    }

    def apply_once():
        qdrant.reset()
        return applier.apply(event)

    assert benchmark(apply_once).status == "applied"


class _CachedEmbeddingModel:
    def encode(self, *_args, **_kwargs):
        return np.asarray(_vector(), dtype=np.float32)


def test_cached_onboarding_embedding_benchmark(benchmark) -> None:
    pipeline = UserOnboardingPipeline(model=_CachedEmbeddingModel())
    result = benchmark(
        pipeline.generate_interest_vector,
        {"topics": ["machine-learning"], "skills": ["Python"]},
    )
    assert len(result) == REPOSITORY_EMBEDDING_DIM


def test_repository_refresh_benchmark(benchmark, monkeypatch) -> None:
    repo_id = str(uuid.uuid4())
    point = SimpleNamespace(id=repo_id, payload=_payload(repo_id, 0))

    def compare_and_set_features(*, expected_point, feature_payload):
        return SimpleNamespace(
            id=expected_point.id,
            payload={**expected_point.payload, **feature_payload},
        )

    monkeypatch.setattr(
        "api.v2.repository_store",
        lambda: SimpleNamespace(compare_and_set_features=compare_and_set_features),
    )
    monkeypatch.setattr("api.v2._repository_points", lambda _repo_id: [point])
    request = RepositoryRefreshJob(
        schema_version=2,
        job_id=uuid.uuid4(),
        repo_id=uuid.UUID(repo_id),
        feature_version=1,
        features={"star_count": 10_001},
    )
    lock = SimpleNamespace(assert_owned=lambda: None)
    result = benchmark(_refresh_repository_job_locked, request, lock)
    assert result["status"] == "applied"
