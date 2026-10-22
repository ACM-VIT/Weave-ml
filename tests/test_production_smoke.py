"""Contract tests for the deployment smoke command."""

from __future__ import annotations

import json
from secrets import token_hex
from uuid import UUID

import pytest

from scripts import production_smoke


class _Response:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self.status = status
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit: int) -> bytes:
        return self._body


def smoke_environment() -> dict[str, str]:
    return {
        "INTERNAL_API_SECRET": token_hex(32),
        "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        "EMBEDDING_MODEL_REVISION": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        "REPOSITORY_EMBEDDING_VERSION": "repo-embedding-v2",
        "V2_COMPATIBLE_EMBEDDING_VERSIONS": "repo-embedding-v2",
        "MIN_ELIGIBLE_REPOSITORIES": "10",
        "V2_USER_COLLECTION_REQUIRED": "true",
        "V2_REQUIRED_CONTENT_VERSION": "1",
        "V2_REQUIRED_FEATURE_SPEC_VERSION": "v2",
        "V2_HEAVY_RANKER_ENABLED": "true",
        "V2_HEAVY_RANKER_REQUIRED": "true",
        "V2_HEAVY_RANKER_TRAFFIC_PERCENT": "100",
        "ML_SMOKE_USER_ID": "b7bf08f4-bc62-43a6-b27e-3705608322b7",
        "ML_SMOKE_RECOMMENDATION_LIMIT": "3",
        "ML_SMOKE_EXPECT_MIN_ITEMS": "1",
        "ML_SMOKE_TIMEOUT_SECONDS": "10",
        "ML_RELEASE_ID": "abcdef1234567890",
    }


def healthy_payload() -> dict:
    return {
        "status": "healthy",
        "embedding_runtime_ready": True,
        "inference_warmup_ready": True,
        "database_required": False,
        "feedback_consumer_active": True,
        "feedback_consumer_release_id": "abcdef1234567890",
        "feedback_consumer_release_mismatch": False,
        "redis": "healthy",
        "feedback_healthy": True,
        "feedback_status": "healthy",
        "qdrant": "healthy",
        "qdrant_server_version": "1.18.2",
        "minimum_qdrant_server_version": "1.18.0",
        "repository_collection_contract": "healthy",
        "user_collection_contract": "healthy",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_model_revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        "configured_embedding_version": "repo-embedding-v2",
        "compatible_embedding_versions": ["repo-embedding-v2"],
        "required_content_version": 1,
        "required_feature_spec_version": "v2",
        "allow_missing_embedding_revision": False,
        "serving_eligibility_version": "repository-vector-v2",
        "serving_eligibility_evidence": "validated_vector_at_atomic_upsert",
        "eligible_repository_points": 12,
        "minimum_eligible_repository_points": 10,
        "heavy_ranker_ready": True,
        "heavy_ranker_warmup_ready": True,
        "heavy_ranker_enabled": True,
        "heavy_ranker_required": True,
        "heavy_ranker_production_qualified": True,
        "heavy_ranker_traffic_percent": 100.0,
    }


def test_health_smoke_checks_consumer_and_embedding_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        production_smoke,
        "urlopen",
        lambda request, timeout: _Response(healthy_payload()),
    )

    result = production_smoke.check_health(smoke_environment())

    assert result["eligible_repository_points"] == 12


def test_health_smoke_rejects_stale_consumer_heartbeat(monkeypatch) -> None:
    payload = healthy_payload()
    payload["feedback_consumer_active"] = False
    monkeypatch.setattr(
        production_smoke,
        "urlopen",
        lambda request, timeout: _Response(payload),
    )

    with pytest.raises(production_smoke.SmokeFailure, match="consumer heartbeat"):
        production_smoke.check_health(smoke_environment())


def test_health_smoke_rejects_missing_serving_eligibility_contract(
    monkeypatch,
) -> None:
    payload = healthy_payload()
    payload.pop("serving_eligibility_version")
    monkeypatch.setattr(
        production_smoke,
        "urlopen",
        lambda request, timeout: _Response(payload),
    )

    with pytest.raises(production_smoke.SmokeFailure, match="serving-eligibility"):
        production_smoke.check_health(smoke_environment())


def test_health_smoke_rejects_qdrant_below_conditional_write_minimum(
    monkeypatch,
) -> None:
    payload = healthy_payload()
    payload["qdrant_server_version"] = "1.17.4"
    monkeypatch.setattr(
        production_smoke,
        "urlopen",
        lambda request, timeout: _Response(payload),
    )

    with pytest.raises(production_smoke.SmokeFailure, match="conditional-write"):
        production_smoke.check_health(smoke_environment())


def test_recommendation_smoke_is_deterministic_and_validates_items(monkeypatch) -> None:
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data))
        request_payload = requests[-1]
        return _Response(
            {
                "schema_version": 2,
                "generation_id": request_payload["generation_id"],
                "user_id": request_payload["user_id"],
                "feed_version": 1,
                "model_version": "heavy-ranker-v2",
                "embedding_version": "repo-embedding-v2",
                "embedding_versions": ["repo-embedding-v2"],
                "served_ranker": "heavy",
                "retrieval_mode": "personalized",
                "ranker_applied": True,
                "heavy_ranker_selected": True,
                "fallback_code": None,
                "items": [
                    {
                        "repo_id": "8e155a52-4528-46bd-b49d-8a8322ce9a1b",
                        "score": 0.75,
                        "source": "semantic",
                    }
                ],
            }
        )

    monkeypatch.setattr(production_smoke, "urlopen", fake_urlopen)

    first = production_smoke.check_recommendation(smoke_environment())
    second = production_smoke.check_recommendation(smoke_environment())

    assert first["generation_id"] == second["generation_id"]
    UUID(first["generation_id"])
    assert requests[0] == requests[1]


def test_recommendation_smoke_rejects_missing_smoke_profile(monkeypatch) -> None:
    environment = smoke_environment()

    def fake_urlopen(request, timeout):
        request_payload = json.loads(request.data)
        return _Response(
            {
                "schema_version": 2,
                "generation_id": request_payload["generation_id"],
                "user_id": request_payload["user_id"],
                "model_version": "heavy-ranker-v2",
                "embedding_version": "repo-embedding-v2",
                "embedding_versions": ["repo-embedding-v2"],
                "served_ranker": "heavy",
                "retrieval_mode": "profile_missing_discovery",
                "ranker_applied": True,
                "heavy_ranker_selected": True,
                "fallback_code": None,
                "items": [
                    {
                        "repo_id": str(UUID("8e155a52-4528-46bd-b49d-8a8322ce9a1b")),
                        "score": 0.75,
                        "source": "popular",
                    }
                ],
            }
        )

    monkeypatch.setattr(production_smoke, "urlopen", fake_urlopen)

    with pytest.raises(production_smoke.SmokeFailure, match="personalized"):
        production_smoke.check_recommendation(environment)
