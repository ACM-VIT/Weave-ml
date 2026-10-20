from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api import main as api_main
from api.contracts import OnboardingJob, RepositoryRefreshJob
from api.v2 import _onboard_user_job, _repository_job_lock_settings
from embedding import runtime as embedding_runtime
from embedding.runtime import EmbeddingDeadlineExceeded
from feedback.v2 import FeedbackEventIdConflictError
from retrieval.v2_retriever import RecommendationBatch, RankedRepository


def _onboarding_job(*, job_id: uuid.UUID | None = None, version: int = 1) -> OnboardingJob:
    return OnboardingJob(
        schema_version=2,
        job_id=job_id or uuid.uuid4(),
        user_id=uuid.uuid4(),
        profile_version=version,
        profile={"topics": ["machine-learning"]},
    )


def test_refresh_contract_rejects_protected_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RepositoryRefreshJob(
            schema_version=2,
            job_id=uuid.uuid4(),
            repo_id=uuid.uuid4(),
            feature_version=2,
            features={"repo_id": str(uuid.uuid4()), "star_count": 10},
        )

    with pytest.raises(ValidationError):
        RepositoryRefreshJob(
            schema_version=2,
            job_id=uuid.uuid4(),
            repo_id=uuid.uuid4(),
            feature_version=2,
            features={"star_count": float("inf")},
        )


def test_onboarding_contract_rejects_unknown_and_oversized_profile_data() -> None:
    payload = {
        "schema_version": 2,
        "job_id": uuid.uuid4(),
        "user_id": uuid.uuid4(),
        "profile_version": 1,
    }
    with pytest.raises(ValidationError):
        OnboardingJob(**payload, profile={"topics": ["ml"], "admin": True})
    with pytest.raises(ValidationError):
        OnboardingJob(**payload, profile={"topics": ["x" * 129]})


def test_duplicate_onboarding_skips_embedding_inference() -> None:
    job = _onboarding_job()
    existing = SimpleNamespace(
        payload={
            "job_id": str(job.job_id),
            "profile_version": job.profile_version,
        }
    )
    store = SimpleNamespace(retrieve_user=lambda _user_id: existing)
    lock = SimpleNamespace(assert_owned=MagicMock())

    @contextmanager
    def acquired_lock(*_args, **_kwargs):
        yield lock

    with patch("api.v2.producer", return_value=SimpleNamespace(redis=MagicMock())), patch(
        "api.v2.user_vector_lock", acquired_lock
    ), patch("api.v2.user_profile_store", return_value=store), patch(
        "api.v2.onboarding_pipeline"
    ) as pipeline_factory:
        result = _onboard_user_job(job)

    assert result["status"] == "duplicate"
    pipeline_factory.assert_not_called()


def test_stale_onboarding_skips_embedding_inference() -> None:
    job = _onboarding_job(version=2)
    existing = SimpleNamespace(
        payload={"job_id": str(uuid.uuid4()), "profile_version": 3}
    )
    store = SimpleNamespace(retrieve_user=lambda _user_id: existing)

    @contextmanager
    def acquired_lock(*_args, **_kwargs):
        yield SimpleNamespace(assert_owned=MagicMock())

    with patch("api.v2.producer", return_value=SimpleNamespace(redis=MagicMock())), patch(
        "api.v2.user_vector_lock", acquired_lock
    ), patch("api.v2.user_profile_store", return_value=store), patch(
        "api.v2.onboarding_pipeline"
    ) as pipeline_factory, pytest.raises(HTTPException) as exc_info:
        _onboard_user_job(job)

    assert exc_info.value.status_code == 409
    pipeline_factory.assert_not_called()


def test_onboarding_timeout_is_reported_as_retryable(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")

    async def exhausted_deadline(*_args, **_kwargs):
        raise EmbeddingDeadlineExceeded("queued too long")

    with patch("api.v2.run_embedding_job", side_effect=exhausted_deadline):
        response = TestClient(api_main.app).post(
            "/api/v2/users/onboard",
            headers={"x-internal-secret": "test-internal-secret"},
            json=_onboarding_job().model_dump(mode="json"),
        )

    assert response.status_code == 504
    assert response.json()["retryable"] is True
    assert response.headers["retry-after"] == "2"


def test_recommendation_context_reaches_retrieval_and_reports_serving_metadata(
    monkeypatch,
) -> None:
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    captured: list[object] = []
    batch = RecommendationBatch(
        items=[RankedRepository(str(uuid.uuid4()), 0.5, "fresh")],
        model_version="heavy-ranker-v2",
        embedding_version="repo-embedding-v2",
        ranker_applied=True,
        served_ranker="heavy",
        heavy_ranker_selected=True,
        retrieval_mode="cold_start_discovery",
    )

    def recommend(*args):
        captured.extend(args)
        return batch

    fake = SimpleNamespace(recommend_batch=recommend)
    with patch("api.v2.retriever", return_value=fake):
        response = TestClient(api_main.app).post(
            "/api/v2/recommendations/generate",
            headers={"x-internal-secret": "test-internal-secret"},
            json={
                "schema_version": 2,
                "generation_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()),
                "feed_version": 1,
                "limit": 15,
                "exclude_repo_ids": [],
                "context": {"cold_start": True, "locale": "en-IN"},
            },
        )

    assert response.status_code == 200
    assert captured[-2] is True
    assert captured[-1] == []
    assert response.json()["retrieval_mode"] == "cold_start_discovery"
    assert response.json()["served_ranker"] == "heavy"
    assert response.json()["context_status"] == {
        "cold_start_applied": True,
        "locale": "reserved_unused",
    }


def test_removed_v1_routes_cannot_be_reenabled(monkeypatch) -> None:
    monkeypatch.setenv("LEGACY_ML_API_ENABLED", "true")
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    response = TestClient(api_main.app).post(
        "/api/v1/recommendations",
        headers={"x-internal-secret": "test-internal-secret"},
    )

    assert response.status_code == 404


def test_validation_errors_are_stable_and_request_ids_propagate(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    request_id = "backend-request-123"
    response = TestClient(api_main.app).post(
        "/api/v2/recommendations/generate",
        headers={
            "x-internal-secret": "test-internal-secret",
            "x-request-id": request_id,
        },
        json={"schema_version": 2},
    )

    assert response.status_code == 422
    assert response.headers["x-request-id"] == request_id
    assert response.json()["code"] == "REQUEST_VALIDATION_FAILED"
    assert response.json()["retryable"] is False
    assert response.json()["request_id"] == request_id
    assert response.json()["message"] == "Request validation failed."


def test_validation_error_details_are_bounded(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    response = TestClient(api_main.app).post(
        "/api/v2/repositories/embed",
        headers={"x-internal-secret": "test-internal-secret"},
        json={f"unexpected_{index}": index for index in range(200)},
    )

    assert response.status_code == 422
    details = response.json()["details"]
    assert len(details) == api_main.MAX_VALIDATION_ERROR_DETAILS + 1
    assert details[-1]["type"] == "validation_errors_omitted"


def test_unsafe_request_id_is_replaced_before_logging(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    response = TestClient(api_main.app).post(
        "/api/v2/recommendations/generate",
        headers={
            "x-internal-secret": "test-internal-secret",
            "x-request-id": "attacker controlled value",
        },
        json={"schema_version": 2},
    )

    assert response.status_code == 422
    replacement = response.headers["x-request-id"]
    assert uuid.UUID(replacement)
    assert response.json()["request_id"] == replacement


def test_health_hides_raw_dependency_errors(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    secret_dependency_text = "redis://user:password@private-host:6379"
    failing = SimpleNamespace(
        health=MagicMock(side_effect=RuntimeError(secret_dependency_text))
    )
    with patch("api.v2.retriever", return_value=failing):
        response = TestClient(api_main.app).get(
            "/api/v2/health",
            headers={"x-internal-secret": "test-internal-secret"},
        )

    assert response.status_code == 503
    assert response.json()["code"] == "DEPENDENCY_UNAVAILABLE"
    assert secret_dependency_text not in response.text
    assert response.json()["message"] == "Dependency health check failed."


@pytest.mark.parametrize(
    ("failure", "expected_status", "expected_code", "retryable"),
    [
        (TimeoutError("private dependency timed out"), 503, "DEPENDENCY_UNAVAILABLE", True),
        (ValueError("corrupt stored payload"), 500, "INTERNAL_ERROR", False),
    ],
)
def test_recommendation_errors_only_retry_known_temporary_failures(
    monkeypatch,
    failure,
    expected_status,
    expected_code,
    retryable,
) -> None:
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    failing = SimpleNamespace(recommend_batch=MagicMock(side_effect=failure))
    with patch("api.v2.retriever", return_value=failing):
        response = TestClient(api_main.app).post(
            "/api/v2/recommendations/generate",
            headers={"x-internal-secret": "test-internal-secret"},
            json={
                "schema_version": 2,
                "generation_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()),
                "feed_version": 1,
                "limit": 15,
                "exclude_repo_ids": [],
                "context": {"cold_start": False},
            },
        )

    assert response.status_code == expected_status
    assert response.json()["code"] == expected_code
    assert response.json()["retryable"] is retryable
    assert str(failure) not in response.text


def test_declared_oversized_request_is_rejected_before_parsing(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    response = TestClient(api_main.app).post(
        "/api/v2/feedback/batch",
        headers={
            "x-internal-secret": "test-internal-secret",
            "content-length": str(api_main.MAX_REQUEST_BODY_BYTES + 1),
            "content-type": "application/json",
        },
        content=b"{}",
    )
    assert response.status_code == 413
    assert response.json()["code"] == "REQUEST_TOO_LARGE"


def test_larger_body_allowance_is_scoped_to_repository_embed(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    client = TestClient(api_main.app)
    allowed_for_repository = client.post(
        "/api/v2/repositories/embed",
        headers={
            "x-internal-secret": "test-internal-secret",
            "content-length": str(api_main.MAX_REQUEST_BODY_BYTES + 1),
            "content-type": "application/json",
        },
        content=b"{}",
    )
    too_large_for_repository = client.post(
        "/api/v2/repositories/embed",
        headers={
            "x-internal-secret": "test-internal-secret",
            "content-length": str(
                api_main.REPOSITORY_EMBED_MAX_REQUEST_BODY_BYTES + 1
            ),
            "content-type": "application/json",
        },
        content=b"{}",
    )

    assert allowed_for_repository.status_code == 422
    assert too_large_for_repository.status_code == 413
    assert too_large_for_repository.json()["code"] == "REQUEST_TOO_LARGE"


def test_feedback_conflict_reports_safe_partial_batch_progress(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    failed_event_id = str(uuid.uuid4())
    conflict = FeedbackEventIdConflictError(
        accepted=2,
        duplicates=1,
        failed_event_id=failed_event_id,
    )
    with patch("api.v2._enqueue_feedback", side_effect=conflict):
        response = TestClient(api_main.app).post(
            "/api/v2/feedback/batch",
            headers={"x-internal-secret": "test-internal-secret"},
            json={
                "schema_version": 2,
                "events": [
                    {
                        "event_id": failed_event_id,
                        "user_id": str(uuid.uuid4()),
                        "repo_id": str(uuid.uuid4()),
                        "feedback_version": 1,
                        "event_type": "like",
                        "occurred_at": "2026-07-21T00:00:00Z",
                    }
                ],
            },
        )

    assert response.status_code == 409
    assert response.json()["code"] == "EVENT_ID_PAYLOAD_CONFLICT"
    assert response.json()["retryable"] is False
    assert response.json()["details"] == {
        "failed_event_id": failed_event_id,
        "accepted": 2,
        "duplicates": 1,
        "retry_guidance": "remove the conflicting event; retrying other events is dedupe-safe",
    }


def test_embedding_runtime_loads_one_shared_model_for_both_pipelines(monkeypatch) -> None:
    embedding_runtime.reset_embedding_runtime_for_tests()
    model = MagicMock()
    loader = MagicMock(return_value=model)
    monkeypatch.setattr(embedding_runtime, "_load_sentence_transformer", loader)
    try:
        repository_pipeline = embedding_runtime.repository_embedding_pipeline()
        onboarding_pipeline = embedding_runtime.user_onboarding_pipeline()
        assert repository_pipeline.embedder._model is onboarding_pipeline._model
        assert repository_pipeline.embedder._model._model is model
        assert embedding_runtime.shared_embedding_model() is onboarding_pipeline._model
        loader.assert_called_once_with()
    finally:
        embedding_runtime.reset_embedding_runtime_for_tests()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_cancelled_embedding_keeps_capacity_until_worker_finishes(
    monkeypatch,
    anyio_backend,
) -> None:
    class RecordingAdmission:
        def __init__(self) -> None:
            self.held = False
            self.releases = 0

        def acquire(self, *, blocking: bool) -> bool:
            assert blocking is False
            if self.held:
                return False
            self.held = True
            return True

        def release(self) -> None:
            assert self.held
            self.held = False
            self.releases += 1

    started = threading.Event()
    allow_finish = threading.Event()
    finished = threading.Event()
    admission = RecordingAdmission()
    executor = ThreadPoolExecutor(max_workers=1)

    def blocking_job() -> str:
        started.set()
        assert allow_finish.wait(timeout=2)
        finished.set()
        return "finished"

    monkeypatch.setattr(embedding_runtime, "embedding_admission", lambda: admission)
    monkeypatch.setattr(embedding_runtime, "embedding_executor", lambda: executor)
    task = asyncio.create_task(embedding_runtime.run_embedding_job(blocking_job))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert admission.releases == 0
        with pytest.raises(embedding_runtime.EmbeddingCapacityError):
            await embedding_runtime.run_embedding_job(lambda: "too-early")

        allow_finish.set()
        assert await asyncio.to_thread(finished.wait, 2)
        for _ in range(100):
            if admission.releases:
                break
            await asyncio.sleep(0)
        assert admission.releases == 1
        assert await embedding_runtime.run_embedding_job(lambda: "accepted") == "accepted"
        assert admission.releases == 2
    finally:
        allow_finish.set()
        executor.shutdown(wait=True, cancel_futures=True)


def test_repository_lock_configuration_is_bounded(monkeypatch) -> None:
    monkeypatch.setenv("REPOSITORY_JOB_LOCK_TTL_MS", "999")
    with pytest.raises(RuntimeError, match="must be >= 1000"):
        _repository_job_lock_settings()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_production_lifespan_fails_before_network_on_invalid_config(
    monkeypatch,
    anyio_backend,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("REDIS_URL", raising=False)
    with pytest.raises(RuntimeError, match="Production configuration is invalid"):
        async with api_main.lifespan(api_main.app):
            pytest.fail("invalid production configuration reached application startup")
