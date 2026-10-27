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

    with patch("api.v2.run_onboarding_job", side_effect=exhausted_deadline):
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


@pytest.mark.parametrize("failed_dependency", ["qdrant", "redis"])
def test_health_hides_raw_dependency_errors(
    monkeypatch, health_probe_cache, failed_dependency,
) -> None:
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    secret_dependency_text = "redis://user:password@private-host:6379"
    qdrant = SimpleNamespace(health=MagicMock(return_value={"qdrant": "healthy"}))
    redis = SimpleNamespace(health=MagicMock(return_value={
        "redis": "healthy", "feedback_consumer_active": True,
    }))
    failing = qdrant if failed_dependency == "qdrant" else redis
    failing.health.side_effect = RuntimeError(secret_dependency_text)
    with patch("api.v2.retriever", return_value=qdrant), patch(
        "api.v2.producer", return_value=redis,
    ):
        client = TestClient(api_main.app)
        headers = {"x-internal-secret": "test-internal-secret"}
        # Exercise the failed dependency, not merely an initially empty cache.
        response = client.get("/api/v2/health/deep", headers=headers)
        cached = client.get("/api/v2/health", headers=headers)

    qdrant.health.assert_called_once()
    redis.health.assert_called_once()
    for result in (response, cached):
        assert result.status_code == 503
        assert result.json()["code"] == "DEPENDENCY_UNAVAILABLE"
        assert secret_dependency_text not in result.text
        assert result.json()["message"] == "Dependency health check failed."
        diagnostic = result.json()["details"]["dependency_diagnostics"][failed_dependency]
        assert diagnostic["status"] == "failed"
        assert diagnostic["reason"] == "dependency_check_failed"


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


@pytest.mark.parametrize("initializer", ["pipeline", "startup"])
def test_online_embedding_initialization_rejects_candidate_before_runtime_work(
    monkeypatch, initializer,
) -> None:
    embedding_runtime.reset_embedding_runtime_for_tests()
    monkeypatch.setattr(
        embedding_runtime,
        "REPOSITORY_EMBEDDING_VERSION",
        "repo-embedding-v3-stable-metadata-v1",
    )
    model_loader = MagicMock()
    summary_factory = MagicMock()
    monkeypatch.setattr(embedding_runtime, "_load_sentence_transformer", model_loader)
    monkeypatch.setattr(embedding_runtime, "card_summary_pipeline", summary_factory)
    initialize = (
        embedding_runtime.repository_embedding_pipeline
        if initializer == "pipeline"
        else embedding_runtime.warm_embedding_runtime
    )
    try:
        with pytest.raises(ValueError, match="export-only"):
            initialize()
        model_loader.assert_not_called()
        summary_factory.assert_not_called()
    finally:
        embedding_runtime.reset_embedding_runtime_for_tests()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.parametrize("job_class", ["repository", "onboarding"])
async def test_cancelled_embedding_keeps_capacity_until_worker_finishes(
    monkeypatch,
    anyio_backend,
    job_class,
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
    run_job = (
        embedding_runtime.run_embedding_job
        if job_class == "repository"
        else embedding_runtime.run_onboarding_job
    )
    if job_class == "onboarding":
        monkeypatch.setattr(embedding_runtime, "onboarding_executor", lambda: executor)
    task = asyncio.create_task(run_job(blocking_job))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert admission.releases == 0
        from utils.observability import diagnostic_snapshot
        def rejection_count():
            return sum(row["count"] for row in diagnostic_snapshot()["histograms"]
                       if row["operation"] == "embedding" and row["stage"] == "queue"
                       and row["outcome"] == "rejected")
        before = rejection_count()
        with pytest.raises(embedding_runtime.EmbeddingCapacityError):
            await embedding_runtime.run_embedding_job(lambda: "too-early")
        assert rejection_count() == before + 1

        allow_finish.set()
        assert await asyncio.to_thread(finished.wait, 2)
        for _ in range(100):
            if admission.releases:
                break
            await asyncio.sleep(0)
        assert admission.releases == 1
        assert await run_job(lambda: "accepted") == "accepted"
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


def test_trace_context_is_adopted_only_after_internal_auth(monkeypatch) -> None:
    from utils import observability

    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    captured = []
    real_scope = observability.span_scope

    @contextmanager
    def capture_scope(*args, **kwargs):
        with real_scope(*args, **kwargs) as span:
            captured.append(observability.current_correlation())
            yield span

    monkeypatch.setattr(api_main, "span_scope", capture_scope)
    parent = "00-1234567890abcdef1234567890abcdef-1234567890abcdef-01"
    headers = {"traceparent": parent, "x-request-id": "legacy-request-id"}
    client = TestClient(api_main.app)
    assert client.get("/api/v2/health/live", headers=headers).status_code == 401
    assert captured == []
    response = client.get("/api/v2/health/live", headers={
        **headers, "x-internal-secret": "test-internal-secret",
    })
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "legacy-request-id"
    assert captured[0].trace_id == "1234567890abcdef1234567890abcdef"
    assert captured[0].parent_span_id == "1234567890abcdef"
    assert captured[0].span_id != captured[0].parent_span_id
    assert captured[0].sampled is True
    assert str(uuid.UUID(captured[0].request_id)) == captured[0].request_id
    assert observability.current_correlation() is None


@pytest.mark.parametrize("header", [
    "00-00000000000000000000000000000000-1234567890abcdef-01",
    "00-1234567890abcdef1234567890abcdef-0000000000000000-01",
    "00-1234567890ABCDEF1234567890abcdef-1234567890abcdef-01",
    "00-1234567890abcdef1234567890abcdef-1234567890abcdef-03",
    "00-1234567890abcdef1234567890abcdef-1234567890abcdef-01-extra",
    " " * 10000,
])
def test_invalid_traceparent_starts_independent_root(header) -> None:
    from utils.observability import current_correlation, span_scope

    with span_scope("search", server=True, authenticated_traceparent=header):
        server = current_correlation()
        assert server.parent_span_id is None
        with span_scope("search"):
            child = current_correlation()
            assert child.trace_id == server.trace_id
            assert child.parent_span_id == server.span_id
            assert child.sampled == server.sampled
        assert current_correlation() is server
    assert current_correlation() is None


def test_sampled_stages_report_remaining_budget_without_expiring_it():
    from utils.deadline import RequestDeadline, deadline_scope
    from utils.observability import diagnostic_snapshot, span_scope

    clock = [100.0]
    deadline = RequestDeadline(5, clock=lambda: clock[0])
    trace = "abcdabcdabcdabcdabcdabcdabcdabcd"
    with deadline_scope(deadline), span_scope(
        "search", server=True,
        authenticated_traceparent=f"00-{trace}-1234567890abcdef-01",
    ):
        clock[0] = 102.0
        with span_scope("search", stage="qdrant"):
            pass
        clock[0] = 107.0
        with span_scope("search", stage="response_build"):
            pass
    spans = [span for span in diagnostic_snapshot()["spans"] if span["trace_id"] == trace]
    values = {span["stage"]: span["remaining_deadline_seconds"] for span in spans}
    assert values == {"http": 5, "qdrant": 3, "response_build": 0}
    assert deadline.expires_at == 105 and not deadline.cancelled.is_set()
    assert all("expires_at" not in span for span in spans)
    assert all(span["ended_at_unix_seconds"] > 0 for span in spans)


def test_public_liveness_is_static_and_only_exact_get_bypasses_auth(monkeypatch):
    monkeypatch.delenv("INTERNAL_API_SECRET", raising=False)
    monkeypatch.setattr("api.v2.health_diagnostics", lambda: pytest.fail("public liveness probed dependencies"))
    client = TestClient(api_main.app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-request-id"]
    for method, path in [("post", "/healthz"), ("get", "/healthz/"),
                         ("get", "/api/v2/health/live"), ("get", "/api/v2/health/ready")]:
        assert client.request(method, path, follow_redirects=False).status_code == 503


def test_feedback_auth_rotation_keeps_rejections_outside_enqueue(monkeypatch, caplog):
    import json
    from api import v2
    secret, rotated = "old-private-boundary-secret", "rotated-private-boundary-secret"
    monkeypatch.setenv("INTERNAL_API_HEADER", "x-ml-service-secret")
    monkeypatch.setenv("INTERNAL_API_SECRET", secret)
    caplog.set_level("INFO", logger="pipeline.api")
    accepted = []
    async def enqueue(operation, function, events):
        assert operation == "feedback"
        accepted.extend(events)
        return len(events), 0
    monkeypatch.setattr(v2, "run_service_job", enqueue)
    client = TestClient(api_main.app)
    payload = {"schema_version": 2, "events": [{
        "event_id": str(uuid.uuid4()), "user_id": str(uuid.uuid4()), "repo_id": str(uuid.uuid4()),
        "event_type": "like", "occurred_at": "2026-10-27T00:00:00Z", "feedback_version": 1,
    }]}
    for headers in ({}, {"x-ml-service-secret": "wrong"}, {"x-internal-secret": secret},
                    [("x-ml-service-secret", secret), ("x-ml-service-secret", "wrong")],
                    [(b"x-ml-service-secret", b"\xff")]):
        assert client.post("/api/v2/feedback/batch", headers=headers, json=payload).status_code == 401
    assert not accepted
    monkeypatch.delenv("INTERNAL_API_SECRET")
    assert client.post("/api/v2/feedback/batch", json=payload).status_code == 503
    monkeypatch.setenv("INTERNAL_API_SECRET", rotated)
    assert client.post("/api/v2/feedback/batch", headers={"x-ml-service-secret": secret}, json=payload).status_code == 401
    response = client.post("/api/v2/feedback/batch", headers={"x-ml-service-secret": rotated}, json=payload)
    assert response.status_code == 202
    assert response.json()["acknowledged_event_ids"] == [payload["events"][0]["event_id"]]
    assert len(accepted) == 1
    oversized = client.post("/api/v2/feedback/batch", headers={"x-ml-service-secret": rotated},
                            content=b"x" * (api_main.MAX_REQUEST_BODY_BYTES + 1))
    assert oversized.status_code == 413 and len(accepted) == 1
    distinct_events = [{**payload["events"][0], "event_id": str(uuid.uuid4()),
                        "feedback_version": version} for version in range(2, 103)]
    for events in ([{**payload["events"][0], "user_id": "not-a-uuid"}], distinct_events):
        rejected = client.post("/api/v2/feedback/batch", headers={"x-ml-service-secret": rotated},
                               json={"schema_version": 2, "events": events})
        assert rejected.status_code == 422
        if events is distinct_events:
            assert any(error["type"] == "too_long" for error in rejected.json()["details"])
    assert len(accepted) == 1
    at_limit = distinct_events[:100]
    response = client.post("/api/v2/feedback/batch", headers={"x-ml-service-secret": rotated},
                           json={"schema_version": 2, "events": at_limit})
    assert response.status_code == 202
    assert response.json()["acknowledged_event_ids"] == [event["event_id"] for event in at_limit]
    assert response.json()["accepted"] == 100 and len(accepted) == 101
    assert secret not in caplog.text and rotated not in caplog.text


def test_feedback_rate_window_and_safe_audit_precede_durable_admission(monkeypatch):
    import io
    import json
    import logging
    from api import v2
    clock = [10.0]
    limiter = api_main.FeedbackRequestLimiter(2, 2, clock=lambda: clock[0])
    monkeypatch.setattr(api_main, "feedback_request_limiter", lambda: limiter)
    secret = "private-rate-audit-secret"
    monkeypatch.setenv("INTERNAL_API_SECRET", secret)
    monkeypatch.delenv("INTERNAL_API_HEADER", raising=False)
    calls = []
    async def enqueue(operation, function, events):
        calls.append(events)
        return len(events), 0
    monkeypatch.setattr(v2, "run_service_job", enqueue)
    payload = {"schema_version": 2, "events": [{
        "event_id": str(uuid.uuid4()), "user_id": str(uuid.uuid4()), "repo_id": str(uuid.uuid4()),
        "event_type": "like", "occurred_at": "2026-10-27T00:00:00Z", "feedback_version": 1,
    }]}
    emitted_stream = io.StringIO()
    assert api_main.feedback_audit_logger.getEffectiveLevel() == logging.INFO
    monkeypatch.setattr(api_main.audit_handler, "stream", emitted_stream)
    client = TestClient(api_main.app)
    assert client.post("/api/v2/feedback/batch", json=payload).status_code == 401
    statuses = [client.post("/api/v2/feedback/batch", headers={"x-internal-secret": secret}, json=payload).status_code for _ in range(2)]
    assert statuses == [202, 202]
    denied = client.post("/api/v2/feedback/batch", headers={"x-internal-secret": secret}, json=payload)
    assert denied.status_code == 429 and denied.headers["retry-after"] == "1"
    assert len(calls) == 2
    clock[0] += .5
    assert client.post("/api/v2/feedback/batch", headers={"x-internal-secret": secret}, json=payload).status_code == 202
    emitted = emitted_stream.getvalue()
    audits = [json.loads(line) for line in emitted.splitlines()]
    assert [entry["status"] for entry in audits] == [401, 202, 202, 429, 202]
    assert [entry["outcome"] for entry in audits] == ["not_acknowledged", "acknowledged", "acknowledged", "not_acknowledged", "acknowledged"]
    assert all(set(entry) == {"event", "status", "outcome"} for entry in audits)
    assert all(entry["event"] == "feedback_boundary" for entry in audits)
    assert secret not in emitted and payload["events"][0]["user_id"] not in emitted


def test_feedback_rate_budget_is_shared_across_threads_and_invalid_config_fails_closed(monkeypatch):
    limiter = api_main.FeedbackRequestLimiter(1, 2, clock=lambda: 10.0)
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _: limiter.retry_after(), range(20)))
    assert outcomes.count(0) == 2 and outcomes.count(1) == 18
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    monkeypatch.setenv("V2_FEEDBACK_REQUESTS_PER_SECOND", "0")
    api_main.feedback_request_limiter.cache_clear()
    try:
        response = TestClient(api_main.app).post("/api/v2/feedback/batch",
            headers={"x-internal-secret": "test-internal-secret"}, json={})
        assert response.status_code == 503
        assert response.json()["code"] == "RATE_LIMIT_NOT_CONFIGURED"
    finally:
        api_main.feedback_request_limiter.cache_clear()


@pytest.mark.parametrize("setting", ["V2_FEEDBACK_REQUESTS_PER_SECOND", "V2_FEEDBACK_REQUEST_BURST"])
def test_invalid_feedback_limits_fail_lifespan_before_runtime_initialization(monkeypatch, setting):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("V2_FEEDBACK_REQUESTS_PER_SECOND", "20")
    monkeypatch.setenv("V2_FEEDBACK_REQUEST_BURST", "40")
    monkeypatch.setenv(setting, "0")
    monkeypatch.setattr("api.v2.validate_v2_runtime_configuration",
                        lambda: pytest.fail("initialized runtime before validating feedback limits"))
    api_main.feedback_request_limiter.cache_clear()
    try:
        with pytest.raises(ValueError, match="Feedback rate and burst"):
            with TestClient(api_main.app):
                pytest.fail("invalid feedback limits allowed startup")
    finally:
        api_main.feedback_request_limiter.cache_clear()
