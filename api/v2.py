from __future__ import annotations

import asyncio
import copy
import logging
import math
import os
import time
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from functools import lru_cache, partial
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import PlainTextResponse, Response
from api.contracts import (
    ModelOperationBatch, ModelOperationResponse, ModelAuthorityRequest, ModelAuthorityResponse,
    RecommendationResponse, RecommendationWireResponse, FeedbackResponse, RepositoryIndexResponse,
    RepositoryRefreshResponse, OnboardingResponse,
    FeedbackBatch,
    OnboardingJob,
    RecommendationRequest,
    RepositoryJob,
    RepositoryFeaturePatch,
    RepositoryRefreshJob,
    RepositorySearchRequest,
    RepositorySearchResponse,
    RequestValidationResponse,
)
from api.runtime import (
    ServiceCapacityError,
    ServiceDeadlineExceeded,
    run_service_job,
    service_runtime_status,
    service_timeout_seconds,
    service_runtime_cleanup_actions,
    service_runtime_drain_actions,
    validate_service_runtime,
)
from api.scoring_capture import CaptureWorkerError, CapturedRequest, authorized_capture_options, execute_captured_request
from api.request_budget import run_read_only_request
from api.health import HealthDiagnostics, HealthSettings
from api.metrics import record_health_dependency, record_recommendation, render_prometheus
from config import (
    QDRANT_COLLECTION_NAME, QDRANT_VECTOR_NAME,
    constant_time_secret_matches, internal_api_header_name,
)
from embedding.corpus_revision import authority_from_environment
from retrieval.discovery_cache import DiscoveryRevision
from embedding.qdrant_store import QdrantRepositoryStore
from embedding.query_cache import CachedQueryEmbedder, QueryCacheCapacityError
from embedding.runtime import (
    EmbeddingCapacityError,
    EmbeddingDeadlineExceeded,
    embedding_runtime_status,
    embedding_timeout_seconds,
    embedding_warmup_enabled,
    inference_warmup_timeout_seconds,
    repository_embedding_pipeline,
    shared_embedding_model,
    run_embedding_job,
    run_onboarding_job,
    embedding_runtime_cleanup_actions,
    embedding_runtime_drain_actions,
    user_onboarding_pipeline,
)
from embedding.user_profile_store import QdrantUserProfileStore
from embedding.vector_contract import (
    repository_point_id,
    validate_repository_payload,
)
from scripts.user_onboarding import UserProfileWriteConflict
from feedback.v2 import (
    DurableFeedbackProducer,
    FeedbackEnqueueError,
    FeedbackEventIdConflictError,
)
from feedback.v2_settings import V2FeedbackSettings
from feedback.user_lock import (
    LockAcquisitionError,
    LockLostError,
    renewable_redis_lock,
    user_vector_lock,
)
from inference.runtime import (
    RecommendationCapacityError,
    RecommendationDeadlineExceeded,
    run_recommendation_job,
    recommendation_runtime_status,
    recommendation_timeout_seconds,
    recommendation_runtime_cleanup_actions,
    recommendation_runtime_drain_actions,
    validate_recommendation_runtime,
)
from ingestion.features import (
    activity_score as calculate_activity_score,
    score_code_health,
    trend_velocity as calculate_trend_velocity,
)
from retrieval.v2_retriever import QdrantV2Retriever, RetrievalDependencyError
from retrieval.runtime import (
    retrieval_runtime_cleanup_actions, retrieval_runtime_drain_actions,
)
from summarization.contracts import (
    CARD_SUMMARY_PAYLOAD_FIELDS,
    CardSummaryArtifact,
    card_summary_from_payload,
)
from summarization.provider import SUMMARY_COMPLETION_RESERVE_SECONDS
from summarization.settings import SummarySettings
from utils.readme_processor import clean_markdown_copy, process_markdown
from utils.observability import span_scope
from utils.deadline import check_deadline
from utils.runtime_lifecycle import (
    RuntimeRetiringError, resume_runtime_admission, stop_runtime_admission,
)

router = APIRouter(prefix="/api/v2", tags=["v2"], responses={
    422: {"model": RequestValidationResponse, "description": "Request validation failed"},
})
logger = logging.getLogger("pipeline.api.v2")

_DERIVED_REPOSITORY_FEATURES = (
    "code_health",
    "activity_score",
    "trend_velocity",
)
_REPOSITORY_FEATURE_DEPENDENCIES = {
    "code_health": frozenset(
        {"fork_count", "open_issues_count", "pushed_days_ago"}
    ),
    "activity_score": frozenset(
        {"pushed_days_ago", "mentionable_users_count"}
    ),
    "trend_velocity": frozenset({"delta_3d", "delta_7d", "delta_30d"}),
}


class _RuntimeCleanupPlan:
    """Retain only failed actions for existing process-owned resources."""

    def __init__(self, capture_actions, capture_drains) -> None:
        self._capture_actions = capture_actions
        self._capture_drains = capture_drains
        self._drains = None
        self._pending = None
        self._lock = threading.Lock()
        self._status = {"state": "pending", "failed_steps": ()}

    def status(self) -> dict[str, Any]:
        status = self._status
        return {"state": status["state"], "failed_steps": list(status["failed_steps"])}

    def run(self) -> tuple[str, ...]:
        with self._lock:
            self._status = {"state": "draining", "failed_steps": ()}
            if self._drains is None:
                try:
                    self._drains = [list(phase) for phase in self._capture_drains()]
                except Exception:
                    logger.error("V2 runtime cleanup failed step=drain_capture")
                    self._status = {"state": "failed", "failed_steps": ("drain_capture",)}
                    return ("drain_capture",)
            for index, phase in enumerate(self._drains):
                failed_drains = []
                for name, action in phase:
                    try:
                        action()
                    except Exception:
                        logger.error("V2 runtime cleanup failed step=%s", name)
                        failed_drains.append((name, action))
                self._drains[index] = failed_drains
                if failed_drains:
                    failures = tuple(name for name, _action in failed_drains)
                    self._status = {"state": "failed", "failed_steps": failures}
                    return failures  # Do not advance to children or resource cleanup.
            self._status = {"state": "running", "failed_steps": ()}
            if self._pending is None:
                try:
                    # Capture only after real workers settle: any of them may
                    # have published a lazy client while retirement began.
                    self._pending = list(self._capture_actions())
                except Exception:
                    logger.error("V2 runtime cleanup failed step=cleanup_capture")
                    self._status = {"state": "failed", "failed_steps": ("cleanup_capture",)}
                    return ("cleanup_capture",)
            failed = []
            for name, action in self._pending:
                try:
                    action()
                except Exception:
                    # Source-defined names only: no exception text, dependency
                    # URLs, credentials, or object reprs in callback logs.
                    logger.error("V2 runtime cleanup failed step=%s", name)
                    failed.append((name, action))
            self._pending = failed
            failures = tuple(name for name, _action in failed)
            self._status = {
                "state": "failed" if failures else "completed",
                "failed_steps": failures,
            }
            return failures


class _RuntimeRetirement:
    """One generation owns its exact workers, cleanup plan and submitted drain."""

    def __init__(self, plan, diagnostics, startup_future) -> None:
        self.plan = plan
        self.diagnostics = diagnostics
        self.startup_future = startup_future
        self._lock = threading.Lock()
        self._startup_callback_registered = False
        self._drain_future: asyncio.Future | None = None

    def start_drain(self) -> asyncio.Future:
        """Own actual worker callbacks/submissions before the caller can yield."""
        loop = asyncio.get_running_loop()
        if self._register_startup_drain():
            # The native Future, not an asyncio Task, now owns eventual cleanup.
            registered = loop.create_future()
            registered.set_result(None)
            return registered
        with self._lock:
            if self._drain_future is None or (
                self._drain_future.done() and self.plan.status()["state"] == "failed"
            ):
                # Submit synchronously: loop teardown must not cancel a Task
                # before it has even enqueued the actual cleanup worker.
                self._drain_future = loop.run_in_executor(None, self.drain)
            return self._drain_future

    def _register_startup_drain(self) -> bool:
        with self._lock:
            barrier = self.startup_future
            if barrier is None or barrier.done():
                return False
            if self._startup_callback_registered:
                return True
            self._startup_callback_registered = True
        # add_done_callback may run inline if the native worker just finished.
        # Never register while holding the retirement lock.
        barrier.add_done_callback(lambda _future: self._drain_dependencies())
        return True

    def drain(self) -> None:
        if not self._register_startup_drain():
            self._drain_dependencies()

    def _drain_dependencies(self) -> None:
        if self.diagnostics is not None:
            self.diagnostics.retire(self.plan.run)
        else:
            self.plan.run()


_shutdown_lock = threading.Lock()
_runtime_retirement: _RuntimeRetirement | None = None


class V2ServiceHTTPException(HTTPException):
    """HTTP error with an explicit stable machine contract."""

    def __init__(
        self,
        *,
        status_code: int,
        detail: str,
        error_code: str,
        retryable: bool,
        headers: dict[str, str] | None = None,
        details: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code
        self.retryable = retryable
        self.safe_details = details


def _is_temporary_dependency_error(exc: Exception) -> bool:
    """Classify only known transport/capacity failures as retryable.

    Unknown ValueError/RuntimeError/programming failures must remain a sanitized
    non-retryable 500; treating all exceptions as temporary creates backend
    retry storms and can conceal stored-data corruption.
    """
    if isinstance(
        exc,
        (
            ConnectionError,
            TimeoutError,
            RetrievalDependencyError,
            LockAcquisitionError,
            LockLostError,
            EmbeddingCapacityError,
            RuntimeRetiringError,
        ),
    ):
        return True

    exception_type = type(exc)
    module = exception_type.__module__
    name = exception_type.__name__
    if module.startswith("redis.exceptions") and name in {
        "BusyLoadingError",
        "ClusterDownError",
        "ConnectionError",
        "MasterDownError",
        "MaxConnectionsError",
        "ReadOnlyError",
        "TimeoutError",
        "TryAgainError",
    }:
        return True
    if module.startswith(("httpx", "httpcore", "grpc")) and any(
        token in name
        for token in ("Connect", "Connection", "Network", "Pool", "Timeout")
    ):
        return True
    if module.startswith("qdrant_client"):
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return status_code in {408, 425, 429} or status_code >= 500
        source = getattr(exc, "source", None)
        if isinstance(source, Exception) and source is not exc:
            return _is_temporary_dependency_error(source)
    return False


def _operation_error(
    operation: str,
    exc: Exception,
    *,
    request_id: str,
) -> HTTPException:
    temporary = _is_temporary_dependency_error(exc)
    logger.error(
        "V2 operation failed request_id=%s operation=%s error_type=%s retryable=%s",
        request_id,
        operation,
        type(exc).__name__,
        temporary,
        extra={
            "dependency_context": {
                "operation": operation,
                "request_id": request_id,
                "error_type": type(exc).__name__,
                "retryable": temporary,
            }
        },
    )
    if not temporary:
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The ML service could not complete the operation.",
        )
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="A required ML dependency is temporarily unavailable.",
        headers={"Retry-After": "2"},
    )


async def require_internal_secret(
    request: Request,
) -> None:
    expected = os.getenv("INTERNAL_API_SECRET")
    if not expected:
        raise HTTPException(status_code=503, detail="Internal API secret is not configured.")
    supplied = request.headers.get(internal_api_header_name())
    if not constant_time_secret_matches(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing internal secret.")


@lru_cache(maxsize=1)
def corpus_authority():
    return authority_from_environment(collection=QDRANT_COLLECTION_NAME, vector=QDRANT_VECTOR_NAME)


def _corpus_discovery_revision():
    authority = corpus_authority()
    current = authority.read() if authority is not None else None
    return DiscoveryRevision(*current) if current is not None else None


@lru_cache(maxsize=1)
def retriever() -> QdrantV2Retriever:
    authority = corpus_authority()
    return QdrantV2Retriever(
        discovery_revision_provider=_corpus_discovery_revision if authority is not None else None,
    )


@lru_cache(maxsize=1)
def feedback_settings() -> V2FeedbackSettings:
    return V2FeedbackSettings.from_env()


@lru_cache(maxsize=1)
def producer() -> DurableFeedbackProducer:
    return DurableFeedbackProducer(settings=feedback_settings())


def _enqueue_feedback(events: list[dict[str, Any]]) -> tuple[int, int]:
    """Resolve the lazy Redis client inside a worker, never on the event loop."""
    return producer().enqueue(events)


def _qdrant_health() -> dict[str, Any]:
    result = dict(retriever().health())
    authority = corpus_authority()
    if authority is None:
        state = {"enabled": False, "reachable": None, "healthy": True, "state": "DISABLED", "writers": None}
    else:
        try:
            status = authority.status()
            state = {"enabled": True, "reachable": True,
                     "healthy": status["state"] in {"READY", "DIRTY"}, **status}
        except Exception:
            # Never expose connection URLs, credentials or arbitrary errors.
            state = {"enabled": True, "reachable": False, "healthy": False,
                     "state": "UNAVAILABLE", "writers": None}
    result["corpus_authority"] = state
    return result


def _feedback_health() -> dict[str, Any]:
    return producer().health()


@lru_cache(maxsize=1)
def repository_store() -> QdrantRepositoryStore:
    pipeline = repository_embedding_pipeline()
    store = QdrantRepositoryStore(
        vector_size=pipeline.config.embedding_dim,
        client=retriever().client,
        corpus_authority=corpus_authority(),
    )
    store.ensure_collection()
    return store


@lru_cache(maxsize=1)
def user_profile_store() -> QdrantUserProfileStore:
    store = QdrantUserProfileStore(client=retriever().client)
    store.ensure_collection()
    return store


def onboarding_pipeline():
    pipeline = user_onboarding_pipeline()
    pipeline.store = user_profile_store()
    return pipeline


def begin_v2_runtime_retirement(
    *, after_startup: Future | None = None,
) -> _RuntimeRetirement:
    """Atomically return the exact retired generation without closing clients."""
    global _runtime_retirement
    with _shutdown_lock:
        if _runtime_retirement is None:
            # Freeze queue/factory admission before capturing this generation.
            # Already-admitted workers and startup retain their resources until
            # the generation-owned cleanup plan drains their real completion.
            stop_runtime_admission()
            diagnostics = None
            if health_diagnostics.cache_info().currsize:
                diagnostics = health_diagnostics()
                diagnostics.close()
            _runtime_retirement = _RuntimeRetirement(
                _RuntimeCleanupPlan(_runtime_cleanup_actions, _runtime_drain_actions),
                diagnostics, after_startup,
            )
        return _runtime_retirement


def shutdown_v2_runtime(retirement: _RuntimeRetirement | None = None) -> None:
    """Operate only on a captured generation, never reread resettable globals."""
    generation = retirement if retirement is not None else begin_v2_runtime_retirement()
    generation.drain()


def _runtime_drain_actions():
    """No model/client capture or close until all admitted workers settle."""
    return [
        [
            *service_runtime_drain_actions(),
            *recommendation_runtime_drain_actions(),
            *embedding_runtime_drain_actions(),
        ],
        retrieval_runtime_drain_actions(),
    ]


def _runtime_cleanup_actions():
    """Capture close methods before cache clears can discard failed resources."""
    actions = []
    from feedback.model_runtime import model_runtime
    if model_runtime.cache_info().currsize:
        actions.append(("model_lane.close", model_runtime().close))
    actions.append(("model_lane.cache.clear", model_runtime.cache_clear))
    if producer.cache_info().currsize:
        close = getattr(producer().redis, "close", None)
        if callable(close):
            actions.append(("redis.client.close", close))
    if corpus_authority.cache_info().currsize and corpus_authority() is not None:
        actions.append(("corpus.authority.close", corpus_authority().close))
    if retriever.cache_info().currsize:
        close = getattr(retriever().client, "close", None)
        if callable(close):
            actions.append(("qdrant.client.close", close))
    return [
        *actions,
        ("repository_store.cache.clear", repository_store.cache_clear),
        ("corpus_authority.cache.clear", corpus_authority.cache_clear),
        ("user_profile_store.cache.clear", user_profile_store.cache_clear),
        ("feedback_producer.cache.clear", producer.cache_clear),
        ("retriever.cache.clear", retriever.cache_clear),
        ("feedback_settings.cache.clear", feedback_settings.cache_clear),
        ("health_diagnostics.cache.clear", health_diagnostics.cache_clear),
        *recommendation_runtime_cleanup_actions(),
        *service_runtime_cleanup_actions(),
        *embedding_runtime_cleanup_actions(),
        *retrieval_runtime_cleanup_actions(),
    ]


def _prepare_runtime_startup() -> None:
    global _runtime_retirement
    with _shutdown_lock:
        retirement = _runtime_retirement
        if retirement is not None:
            if retirement.plan.status()["state"] != "completed":
                raise RuntimeError("Previous runtime cleanup is incomplete")
            health_diagnostics.cache_clear()
            _runtime_retirement = None
            resume_runtime_admission()


def _repository_points(repo_id: str) -> list[Any]:
    canonical = repository_point_id(repo_id)
    return retriever().client.retrieve(
        collection_name=retriever().repository_collection,
        ids=[canonical],
        with_payload=True,
        with_vectors=False,
    )


def _repository_job_status(
    points: list[Any],
    *,
    version_field: str,
    job_field: str,
    requested_version: int,
    job_id: str,
) -> tuple[str, int]:
    payloads = [dict(point.payload or {}) for point in points]
    current = max(
        (int(payload.get(version_field) or 0) for payload in payloads), default=0
    )
    matching_jobs = [
        payload
        for payload in payloads
        if str(payload.get(job_field) or "") == job_id
    ]
    if matching_jobs:
        matched_version = max(
            (int(payload.get(version_field) or 0) for payload in matching_jobs),
            default=0,
        )
        if matched_version != requested_version:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"job_id was already used for {version_field} {matched_version}",
            )
        return "duplicate", current
    if requested_version < current:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"{version_field} {requested_version} is older than stored version {current}."
            ),
        )
    if requested_version == current and current > 0:
        return "current", current
    return "apply", current


def _single_repository_point(points: list[Any]) -> Any | None:
    """Return the single canonical repository point."""

    distinct = {str(point.id): point for point in points}
    if len(distinct) > 1:
        raise V2ServiceHTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Repository identity requires coordinated offline reconciliation.",
            error_code="REPOSITORY_IDENTITY_CONFLICT",
            retryable=False,
        )
    return next(iter(distinct.values()), None)


def _repository_cas_outcome(
    point: Any | None,
    *,
    version_field: str,
    job_field: str,
    requested_version: int,
    job_id: str,
) -> tuple[str, int]:
    """Verify a conditional write and classify the winner without guessing."""

    if point is None:
        raise V2ServiceHTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Repository state changed during the write; retry the request.",
            error_code="CONCURRENT_REPOSITORY_WRITE",
            retryable=True,
            headers={"Retry-After": "2"},
        )
    payload = dict(point.payload or {})
    try:
        current = int(payload.get(version_field) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stored {version_field} is invalid") from exc
    if str(payload.get(job_field) or "") == job_id and current == requested_version:
        return "applied", current

    job_status, current = _repository_job_status(
        [point],
        version_field=version_field,
        job_field=job_field,
        requested_version=requested_version,
        job_id=job_id,
    )
    if job_status == "apply":
        # The version stayed old, so another independent field changed and
        # correctly fenced this write. A retry can merge that newer snapshot.
        raise V2ServiceHTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Repository state changed during the write; retry the request.",
            error_code="CONCURRENT_REPOSITORY_WRITE",
            retryable=True,
            headers={"Retry-After": "2"},
        )
    return job_status, current


def _repository_job_lock_settings() -> tuple[int, float]:
    try:
        ttl_ms = int(os.getenv("REPOSITORY_JOB_LOCK_TTL_MS", "600000").strip())
        wait_seconds = float(
            os.getenv("REPOSITORY_JOB_LOCK_WAIT_SECONDS", "30").strip()
        )
    except ValueError as exc:
        raise RuntimeError("repository lock settings are invalid") from exc
    if ttl_ms < 1_000 or not 0 <= wait_seconds <= 300:
        raise RuntimeError(
            "REPOSITORY_JOB_LOCK_TTL_MS must be >= 1000 and lock wait must be 0-300 seconds"
        )
    return ttl_ms, wait_seconds


def _health_timeout_seconds() -> float:
    try:
        timeout = float(os.getenv("V2_HEALTH_TIMEOUT_SECONDS", "5").strip())
    except ValueError as exc:
        raise RuntimeError("V2_HEALTH_TIMEOUT_SECONDS is invalid") from exc
    if not math.isfinite(timeout) or not 0.1 <= timeout <= 30:
        raise RuntimeError("V2_HEALTH_TIMEOUT_SECONDS must be between 0.1 and 30")
    return timeout


def validate_v2_runtime_configuration() -> V2FeedbackSettings:
    """Validate network-free v2 runtime settings during process startup."""
    _prepare_runtime_startup()
    settings = feedback_settings()
    _repository_job_lock_settings()
    _health_timeout_seconds()
    HealthSettings.from_env()
    embedding_timeout_seconds()
    embedding_warmup_enabled()
    inference_warmup_timeout_seconds()
    # Constructing settings is network-free and catches malformed provider
    # configuration before an ingest request can occupy an embedding worker.
    SummarySettings.from_env()
    validate_recommendation_runtime()
    validate_service_runtime()
    # Client construction validates retrieval/ranker configuration and, when
    # enabled, verifies heavy-ranker artifacts without querying Qdrant.
    retriever()
    return settings


def _repository_job_lock(repo_id: str):
    redis = producer().redis
    key = f"ml:repository-job-lock:{repo_id}"
    ttl_ms, wait_seconds = _repository_job_lock_settings()
    try:
        return renewable_redis_lock(
            redis,
            key,
            ttl_ms=ttl_ms,
            wait_seconds=wait_seconds,
        )
    except (LockAcquisitionError, LockLostError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Repository job is temporarily locked; retry the request.",
            headers={"Retry-After": "2"},
        ) from exc


def _embed_repository_job(request: RepositoryJob, deadline: float | None = None) -> dict[str, Any]:
    repo_id = str(request.repo_id)
    job_id = str(request.job_id)
    try:
        lock_context = _repository_job_lock(repo_id)
        with lock_context as lock:
            return _embed_repository_job_locked(request, lock, deadline=deadline)
    except (LockAcquisitionError, LockLostError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Repository job lock was unavailable; retry the request.",
            headers={"Retry-After": "2"},
        ) from exc


def _embed_repository_job_locked(request: RepositoryJob, lock: Any, deadline: float | None = None) -> dict[str, Any]:
    repo_id = str(request.repo_id)
    job_id = str(request.job_id)
    lock.assert_owned()
    points = _repository_points(repo_id)
    expected_point = _single_repository_point(points)
    job_status, current = _repository_job_status(
        points,
        version_field="content_version",
        job_field="content_job_id",
        requested_version=request.content_version,
        job_id=job_id,
    )
    pipeline = repository_embedding_pipeline()
    payload: dict[str, Any] | None = None

    def summary_context(artifact: CardSummaryArtifact) -> dict[str, Any]:
        nonlocal payload
        if artifact.source == "generated":
            # Match the exact derived evidence used during generation. Cache
            # this projection within the job; fallback replay stays README-free.
            if payload is None:
                payload = _repository_embedding_payload(request)
            return payload
        return request.repository.model_dump(mode="json")

    if job_status != "apply":
        if expected_point is None:
            raise V2ServiceHTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Repository summary state changed; retry the request.",
                error_code="REPOSITORY_SUMMARY_RACE",
                retryable=True,
            )
        stored_payload = dict(expected_point.payload or {})
        requested_hash = request.repository.content_hash
        stored_hash = stored_payload.get("content_hash")
        if (
            not requested_hash
            or not isinstance(stored_hash, str)
            or not stored_hash
            or requested_hash != stored_hash
        ):
            raise V2ServiceHTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="content_version already identifies different repository content.",
                error_code="REPOSITORY_CONTENT_HASH_CONFLICT",
                retryable=False,
            )
        artifact = card_summary_from_payload(stored_payload)
        if artifact is None or not pipeline.card_summarizer.is_current(
            artifact, repo=summary_context(artifact)
        ) or (
            job_status != "duplicate"
            and pipeline.card_summarizer.should_upgrade(artifact, repo={"repo_id": repo_id, "content_hash": request.repository.content_hash})
            and _summary_upgrade_has_time(deadline)
        ):
            payload = payload if payload is not None else _repository_embedding_payload(request)
            artifact = pipeline.summarize_repository(payload, deadline=deadline)
            lock.assert_owned()
            stored = repository_store().compare_and_set_card_summary(
                expected_point=expected_point,
                artifact=artifact,
            )
            lock.assert_owned()
            if stored is None:
                raise V2ServiceHTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Repository summary state changed; retry the request.",
                    error_code="REPOSITORY_SUMMARY_RACE",
                    retryable=True,
                )
            final_status, current = _repository_job_status(
                [stored],
                version_field="content_version",
                job_field="content_job_id",
                requested_version=request.content_version,
                job_id=job_id,
            )
            stored_payload = dict(stored.payload or {})
            artifact = card_summary_from_payload(stored_payload)
            job_status = final_status
        if artifact is None or not pipeline.card_summarizer.is_current(
            artifact, repo=summary_context(artifact)
        ):
            raise V2ServiceHTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Repository summary state changed; retry the request.",
                error_code="REPOSITORY_SUMMARY_RACE",
                retryable=True,
            )
        return _repository_embedding_response(
            request=request,
            status_name=job_status,
            content_version=current,
            embedding_version=str(
                stored_payload.get("embedding_version") or pipeline.config.version
            ),
            artifact=artifact,
        )
    payload = payload if payload is not None else _repository_embedding_payload(request)
    result = pipeline.embed_repository(payload, deadline=deadline)
    result.payload["content_job_id"] = job_id

    # Content upsert replaces the complete point. Carry forward independently
    # refreshed feature state from the exact snapshot used by the CAS so it is
    # never reset by a newer content embedding.
    if expected_point is not None:
        expected_payload = dict(expected_point.payload or {})
        preserved_fields = ["feature_version", "feature_job_id"]
        if "feature_version" in expected_payload:
            preserved_fields.extend(RepositoryFeaturePatch.model_fields)
        for field in preserved_fields:
            if field in expected_payload:
                result.payload[field] = copy.deepcopy(expected_payload[field])

    lock.assert_owned()
    stored = repository_store().compare_and_set_content(
        result,
        expected_point=expected_point,
    )
    lock.assert_owned()
    final_status, final_version = _repository_cas_outcome(
        stored,
        version_field="content_version",
        job_field="content_job_id",
        requested_version=request.content_version,
        job_id=job_id,
    )
    stored_artifact = card_summary_from_payload(dict(stored.payload or {})) if stored else None
    if stored_artifact is None or not pipeline.card_summarizer.is_current(
        stored_artifact, repo=summary_context(stored_artifact)
    ):
        raise V2ServiceHTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Repository summary was not durably stored; retry the request.",
            error_code="REPOSITORY_SUMMARY_MISSING",
            retryable=True,
        )
    return _repository_embedding_response(
        request=request,
        status_name=final_status,
        content_version=final_version,
        embedding_version=str(
            dict(stored.payload or {}).get("embedding_version")
            or result.embedding_version
        ),
        artifact=stored_artifact,
    )


def _summary_upgrade_has_time(
    deadline: float | None,
) -> bool:
    """Avoid delaying replay of a durable fallback through another timeout."""

    if deadline is None:
        return True
    return (
        deadline - time.monotonic()
        # The provider caps its request timeout to the remaining usable budget.
        # Requiring the full configured timeout would disable upgrades under
        # the production 30-second job / 30-second provider configuration.
        >= SUMMARY_COMPLETION_RESERVE_SECONDS + 0.1
    )


def _repository_embedding_response(
    *,
    request: RepositoryJob,
    status_name: str,
    content_version: int,
    embedding_version: str,
    artifact: CardSummaryArtifact,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "accepted": True,
        "status": status_name,
        "job_id": str(request.job_id),
        "repo_id": str(request.repo_id),
        "content_version": content_version,
        "embedding_version": embedding_version,
        "card_summary": artifact.model_dump(mode="json"),
    }


def _repository_embedding_payload(request: RepositoryJob) -> dict[str, Any]:
    payload = request.repository.model_dump(mode="json")
    # The backend source contract permits null GitHub values. The embedding
    # payload contract publishes deterministic, non-null serving values.
    payload["description"] = payload.get("description") or ""
    payload["primary_language"] = payload.get("primary_language") or "Unknown"
    readme = payload.get("readme")
    if readme is not None:
        derived = process_markdown(readme)
        payload["extracted_paragraphs"] = derived.extracted_paragraphs or [
            clean_markdown_copy(readme)
        ]
        payload["extracted_paragraphs"] = [
            paragraph for paragraph in payload["extracted_paragraphs"] if paragraph
        ]
        payload["readme_length"] = len(readme)
    payload.update({
        "id": str(request.repo_id),
        "repo_id": str(request.repo_id),
        "content_version": request.content_version,
    })
    return payload


def _refresh_repository_job(request: RepositoryRefreshJob) -> dict[str, Any]:
    repo_id = str(request.repo_id)
    try:
        with _repository_job_lock(repo_id) as lock:
            return _refresh_repository_job_locked(request, lock)
    except (LockAcquisitionError, LockLostError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Repository job lock was unavailable; retry the request.",
            headers={"Retry-After": "2"},
        ) from exc


def _refresh_repository_job_locked(
    request: RepositoryRefreshJob, lock: Any
) -> dict[str, Any]:
    repo_id = str(request.repo_id)
    job_id = str(request.job_id)
    lock.assert_owned()
    points = _repository_points(repo_id)
    if not points:
        raise HTTPException(
            status_code=404, detail=f"Repository {repo_id} is not indexed."
        )
    expected_point = _single_repository_point(points)
    job_status, current = _repository_job_status(
        points,
        version_field="feature_version",
        job_field="feature_job_id",
        requested_version=request.feature_version,
        job_id=job_id,
    )
    if job_status != "apply":
        return {
            "accepted": True,
            "status": job_status,
            "repo_id": repo_id,
            "feature_version": current,
        }

    requested_features = request.features.model_dump(mode="json", exclude_none=True)
    observed_features = {
        field: value
        for field, value in requested_features.items()
        if field not in _DERIVED_REPOSITORY_FEATURES
    }
    resulting_observation = {
        **dict(expected_point.payload or {}),
        **observed_features,
    }
    stored_observation = dict(expected_point.payload or {})
    changed_observed_features = {
        field
        for field, value in observed_features.items()
        if stored_observation.get(field) != value
    }
    affected_derived_features = {
        field
        for field, dependencies in _REPOSITORY_FEATURE_DEPENDENCIES.items()
        if dependencies.intersection(changed_observed_features)
    }
    derived_features: dict[str, float] = {}
    if "code_health" in affected_derived_features:
        derived_features["code_health"] = score_code_health(resulting_observation)
    if "activity_score" in affected_derived_features:
        # Feature refreshes do not retain recent commit timestamps in Qdrant.
        # When an activity dependency changes, use the documented persisted
        # fallback and producer-anchored push age. Unrelated patches preserve a
        # score originally derived from richer recent-commit observations.
        derived_features["activity_score"] = calculate_activity_score(
            {**resulting_observation, "recent_commits": []}
        )
    if "trend_velocity" in affected_derived_features:
        derived_features["trend_velocity"] = calculate_trend_velocity(
            resulting_observation
        )
    expected_derived_features = {
        field: (
            derived_features[field]
            if field in affected_derived_features
            else resulting_observation[field]
        )
        for field in _DERIVED_REPOSITORY_FEATURES
    }
    inconsistent = [
        field
        for field in _DERIVED_REPOSITORY_FEATURES
        if field in requested_features
        and requested_features[field] != expected_derived_features[field]
    ]
    if inconsistent:
        expected = ", ".join(
            f"{field}={expected_derived_features[field]}" for field in inconsistent
        )
        detail = (
            "Repository derived features are inconsistent with the resulting "
            f"observation; expected {expected}."
        )
        raise V2ServiceHTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=detail,
            error_code="REQUEST_VALIDATION_FAILED",
            retryable=False,
            details=[
                {
                    "path": f"body.features.{field}",
                    "message": (
                        "Value must match the effective server-owned score "
                        f"{expected_derived_features[field]}."
                    ),
                    "type": "value_error.derived_feature_mismatch",
                }
                for field in inconsistent
            ],
        )
    features = {**observed_features, **derived_features}
    updated = {
        **dict(expected_point.payload or {}),
        **features,
        "feature_version": request.feature_version,
        "feature_job_id": job_id,
    }
    # Normalize only missing summary fields on a read copy. Core contract fields
    # remain mandatory and summary defaults must never enter the feature CAS.
    for field in CARD_SUMMARY_PAYLOAD_FIELDS:
        updated.setdefault(field, [] if field == "card_summary_highlights" else None)
    validate_repository_payload(updated)

    lock.assert_owned()
    stored = repository_store().compare_and_set_features(
        expected_point=expected_point,
        feature_payload={
            **features,
            "feature_version": request.feature_version,
            "feature_job_id": job_id,
        },
    )
    lock.assert_owned()
    final_status, final_version = _repository_cas_outcome(
        stored,
        version_field="feature_version",
        job_field="feature_job_id",
        requested_version=request.feature_version,
        job_id=job_id,
    )
    return {
        "accepted": True,
        "status": final_status,
        "repo_id": repo_id,
        "feature_version": final_version,
    }


def _user_job_status(
    point: Any | None,
    *,
    requested_version: int,
    job_id: str,
) -> tuple[str, int]:
    payload = dict(point.payload or {}) if point is not None else {}
    try:
        current = int(payload.get("profile_version") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("stored profile_version is invalid") from exc
    stored_job = str(payload.get("job_id") or "")
    if stored_job == job_id:
        if current != requested_version:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"job_id was already used for profile_version {current}",
            )
        return "duplicate", current
    if requested_version < current:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"profile_version {requested_version} is older than stored version {current}."
            ),
        )
    if requested_version == current and current > 0:
        return "current", current
    return "apply", current


def _onboard_user_job(request: OnboardingJob) -> dict[str, Any]:
    user_id = str(request.user_id)
    job_id = str(request.job_id)
    redis = producer().redis
    try:
        with user_vector_lock(
            redis,
            user_id,
            settings=feedback_settings(),
        ) as lock:
            store = user_profile_store()
            existing = store.retrieve_user(user_id)
            job_status, current = _user_job_status(
                existing,
                requested_version=request.profile_version,
                job_id=job_id,
            )
            if job_status != "apply":
                return {
                    "accepted": True,
                    "status": job_status,
                    "user_id": user_id,
                    "profile_version": current,
                }

            profile = request.profile.model_dump(mode="json", exclude_none=True)
            pipeline = onboarding_pipeline()
            vector = pipeline.generate_interest_vector(profile)
            payload = {
                **profile,
                "job_id": job_id,
                "profile_version": request.profile_version,
            }
            lock.assert_owned()
            pipeline.save_to_qdrant(
                user_id,
                vector,
                payload,
                expected_point=existing,
            )
            lock.assert_owned()
            return {
                "accepted": True,
                "status": "applied",
                "user_id": user_id,
                "profile_version": request.profile_version,
            }
    except (LockAcquisitionError, LockLostError, UserProfileWriteConflict) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User profile is temporarily locked; retry the request.",
            headers={"Retry-After": "2"},
        ) from exc


def _captured_recommendation_job(request: CapturedRequest):
    return execute_captured_request(retriever(), request)


@router.post(
    "/recommendations/generate", response_model=RecommendationWireResponse, response_model_exclude_unset=True, dependencies=[Depends(require_internal_secret)],
    openapi_extra={"parameters": [{
        "in": "header", "name": "x-scoring-capture-capability", "required": False,
        "schema": {"type": "string", "minLength": 64, "maxLength": 64, "pattern": "^[a-f0-9]{64}$"},
        "description": "Additional scoped capability required when scoring_capture is provided. Ordinary requests require only the internal service credential.",
    }]},
)
async def generate_recommendations(
    request: RecommendationRequest,
    http_request: Request,
):
    model_context = None
    if request.model_context is not None:
        from feedback.model_serving import Context
        context = request.model_context
        model_context = Context(str(request.user_id), context.session_id, int(context.epoch), context.policy_sha256,
            int(context.minimum_applied_operation_version), int(context.required_profile_version), int(context.as_of_ms))
    capture_job = None
    if request.scoring_capture is not None:
        options = authorized_capture_options(http_request, request.feed_version)
        capture_job = CapturedRequest(str(request.user_id), str(request.generation_id), request.feed_version,
            request.limit, tuple(str(item) for item in request.exclude_repo_ids),
            tuple(str(item) for item in request.social_repo_ids), tuple(request.social_candidates),
            request.context.cold_start, request.context.locale, options, model_context)
    arguments = (
        str(request.user_id), request.limit,
        [str(item) for item in request.exclude_repo_ids], str(request.generation_id),
        request.context.cold_start, [str(item) for item in request.social_repo_ids],
    )
    if request.social_candidates:
        arguments = (*arguments, request.social_candidates)
    try:
        batch = await run_read_only_request(
            http_request, recommendation_timeout_seconds(),
            (lambda: run_recommendation_job(partial(_captured_recommendation_job, capture_job)))
            if capture_job is not None else (lambda: run_recommendation_job(
                retriever().recommend_batch,
                *arguments, **({"model_context": model_context} if model_context is not None else {}),
            )),
        )
    except HTTPException:
        raise
    except RecommendationCapacityError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation capacity is temporarily exhausted; retry the request.",
            headers={"Retry-After": "1"},
        ) from exc
    except RecommendationDeadlineExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Recommendation generation exceeded its bounded deadline.",
            headers={"Retry-After": "1"},
        ) from exc
    except CaptureWorkerError:
        raise V2ServiceHTTPException(status_code=503, error_code="SCORING_CAPTURE_UNAVAILABLE", retryable=True,
            detail="Requested scoring capture is unavailable; retry after the provider is ready.",
            headers={"Retry-After": "2"}) from None
    except Exception as exc:
        raise _operation_error(
            "recommendations",
            exc,
            request_id=str(http_request.state.request_id),
        ) from exc
    if capture_job is not None:
        if batch.failure_code is not None or batch.wire is None:
            raise V2ServiceHTTPException(status_code=503, error_code="SCORING_CAPTURE_UNAVAILABLE", retryable=True,
                detail="Requested scoring capture is unavailable; retry after the provider is ready.",
                headers={"Retry-After": "2"})
        return Response(content=batch.wire, media_type="application/json", headers={"Cache-Control": "no-store"})
    items = batch.items
    invalid_scores = any(not math.isfinite(item.score) for item in items)
    if len({item.repo_id for item in items}) != len(items) or invalid_scores:
        raise HTTPException(
            status_code=500, detail="Retriever produced invalid recommendations."
        )
    record_recommendation(
        served_ranker=str(getattr(batch, "served_ranker", "hybrid")),
        fallback_code=getattr(batch, "fallback_code", None),
        item_count=len(items),
    )
    with span_scope("recommendation", stage="response_build"):
        return {
            "schema_version": 2,
            "generation_id": str(request.generation_id),
            "user_id": str(request.user_id),
            "feed_version": request.feed_version,
            "model_version": batch.model_version,
            "embedding_version": batch.embedding_version,
            "embedding_versions": list(getattr(batch, "embedding_versions", ())),
            "served_ranker": getattr(batch, "served_ranker", "heavy"),
            "heavy_ranker_selected": getattr(batch, "heavy_ranker_selected", False),
            "ranker_applied": batch.ranker_applied,
            "fallback_code": getattr(batch, "fallback_code", None),
            "fallback_reason": getattr(batch, "fallback_reason", None),
            "retrieval_mode": getattr(batch, "retrieval_mode", "personalized"),
            "context_status": {
                "cold_start_applied": request.context.cold_start,
                "locale": "reserved_unused" if request.context.locale else "not_provided",
            },
            **({"model_context": batch.model_context} if model_context is not None else {}),
            "items": [asdict(item) for item in items],
        }


@router.post(
    "/feedback/batch",
    response_model=FeedbackResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_internal_secret)],
)
async def submit_feedback(request: FeedbackBatch, http_request: Request):
    events = [event.model_dump(mode="json") for event in request.events]
    try:
        accepted, duplicates = await run_service_job(
            "feedback", _enqueue_feedback, events
        )
    except FeedbackEventIdConflictError as exc:
        raise V2ServiceHTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="event_id was already used with a different feedback payload.",
            error_code="EVENT_ID_PAYLOAD_CONFLICT",
            retryable=False,
            details={
                "failed_event_id": exc.failed_event_id,
                "accepted": exc.accepted,
                "duplicates": exc.duplicates,
                "retry_guidance": "remove the conflicting event; retrying other events is dedupe-safe",
            },
        ) from exc
    except FeedbackEnqueueError as exc:
        logger.error(
            "V2 feedback batch was only partially enqueued request_id=%s",
            str(http_request.state.request_id),
            extra={
                "feedback_batch_context": {
                    "request_id": str(http_request.state.request_id),
                    "accepted": exc.accepted,
                    "duplicates": exc.duplicates,
                    "status": "retry_required",
                }
            },
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback storage is temporarily unavailable; retry the complete batch.",
            headers={"Retry-After": "1"},
        ) from exc
    except (ServiceCapacityError, ServiceDeadlineExceeded) as exc:
        raise V2ServiceHTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback admission is temporarily unavailable; retry the complete batch.",
            error_code="FEEDBACK_CAPACITY_UNAVAILABLE",
            retryable=True,
            headers={"Retry-After": "1"},
        ) from exc
    # A completed producer call proves every requested event is durably accepted
    # or already present. Error/timeout paths above must not infer partial ACKs.
    return {
        "accepted": accepted, "duplicates": duplicates, "durable": True,
        "acknowledged_event_ids": [str(event.event_id) for event in request.events],
    }


@router.post("/repositories/embed", response_model=RepositoryIndexResponse, dependencies=[Depends(require_internal_secret)])
async def embed_repository(request: RepositoryJob, http_request: Request):
    try:
        deadline = time.monotonic() + embedding_timeout_seconds()
        return await run_embedding_job(_embed_repository_job, request, deadline=deadline)
    except HTTPException:
        raise
    except EmbeddingCapacityError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding capacity is temporarily exhausted; retry the request.",
            headers={"Retry-After": "2"},
        ) from exc
    except EmbeddingDeadlineExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Embedding generation exceeded its bounded deadline.",
            headers={"Retry-After": "2"},
        ) from exc
    except Exception as exc:
        raise _operation_error(
            "repository_embedding",
            exc,
            request_id=str(http_request.state.request_id),
        ) from exc


@router.post("/repositories/refresh", response_model=RepositoryRefreshResponse, dependencies=[Depends(require_internal_secret)])
async def refresh_repository(request: RepositoryRefreshJob, http_request: Request):
    try:
        return await run_service_job("refresh", _refresh_repository_job, request)
    except HTTPException:
        raise
    except (ServiceCapacityError, ServiceDeadlineExceeded) as exc:
        raise V2ServiceHTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Repository refresh capacity is temporarily unavailable; retry the request.",
            error_code="REFRESH_CAPACITY_UNAVAILABLE",
            retryable=True,
            headers={"Retry-After": "2"},
        ) from exc
    except Exception as exc:
        raise _operation_error(
            "repository_refresh",
            exc,
            request_id=str(http_request.state.request_id),
        ) from exc


@router.post("/users/onboard", response_model=OnboardingResponse, dependencies=[Depends(require_internal_secret)])
async def onboard_user(request: OnboardingJob, http_request: Request):
    try:
        return await run_onboarding_job(_onboard_user_job, request)
    except HTTPException:
        raise
    except EmbeddingCapacityError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding capacity is temporarily exhausted; retry the request.",
            headers={"Retry-After": "2"},
        ) from exc
    except EmbeddingDeadlineExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Onboarding generation exceeded its bounded deadline.",
            headers={"Retry-After": "2"},
        ) from exc
    except Exception as exc:
        raise _operation_error(
            "user_onboarding",
            exc,
            request_id=str(http_request.state.request_id),
        ) from exc


def _collect_health_diagnostics(stop: threading.Event) -> tuple[bool, dict[str, Any]]:
    """Own two bounded parallel checks until both actual child workers exit."""
    payload: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    healthy = True
    deadline = time.monotonic() + HealthSettings.from_env().timeout_seconds

    def check_dependency(dependency: str, check) -> tuple[bool, dict[str, Any], dict[str, Any]]:
        details: dict[str, Any] = {}
        if stop.is_set() or time.monotonic() >= deadline:
            return False, details, {
                "status": "not_checked",
                "reason": "stopping" if stop.is_set() else "deadline_exceeded",
            }
        started = time.monotonic()
        outcome = "failed"
        diagnostic: dict[str, Any] = {"status": "failed", "reason": "dependency_check_failed"}
        try:
            result = check()
            details.update(result)
            if dependency == "qdrant" and not result.get("corpus_authority", {}).get("healthy", True):
                raise RuntimeError("enabled corpus authority is unavailable for writes")
            if dependency == "redis":
                consumer_required = os.getenv(
                    "V2_FEEDBACK_CONSUMER_REQUIRED", "true"
                ).strip().lower() in {"1", "true", "yes", "on"}
                if consumer_required and not result.get("feedback_consumer_active"):
                    raise RuntimeError("feedback consumer heartbeat is missing")
                if not result.get("feedback_healthy", True):
                    raise RuntimeError("feedback thresholds exceeded")
            outcome = "healthy"
            diagnostic = {"status": "healthy"}
        except Exception:
            # Dependency exceptions can contain URLs/credentials. Keep only a
            # fixed failure category, plus successfully returned diagnostics.
            pass
        finally:
            duration = time.monotonic() - started
            diagnostic["duration_seconds"] = duration
            record_health_dependency(dependency, outcome, duration)
        return outcome == "healthy", details, diagnostic

    # Exactly two children belong to this admitted, single-flight coordinator.
    # They never reenter its reserved pool (which may have only one worker).
    # The context joins all submitted children even if submission/result handling
    # fails, so retirement cannot close shared clients under an unfinished RPC.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ml-health-dependency") as children:
        checks = [
            (dependency, children.submit(check_dependency, dependency, check))
            for dependency, check in (("qdrant", _qdrant_health), ("redis", _feedback_health))
        ]
        for dependency, future in checks:
            dependency_healthy, details, diagnostic = future.result()
            healthy = healthy and dependency_healthy
            payload.update(details)
            diagnostics[dependency] = diagnostic
    payload["dependency_diagnostics"] = diagnostics
    return healthy, payload


@lru_cache(maxsize=1)
def health_diagnostics() -> HealthDiagnostics:
    retirement = _runtime_retirement
    if retirement is not None and retirement.diagnostics is not None:
        return retirement.diagnostics
    diagnostics = HealthDiagnostics(_collect_health_diagnostics, HealthSettings.from_env())
    if retirement is not None:
        diagnostics.close()
    return diagnostics


def _readiness_response(
    snapshot: dict[str, Any], request: Request, *, tier: str = "readiness"
) -> dict[str, Any]:
    retirement = _runtime_retirement
    if retirement is not None:
        # Do not inspect lazy model/executor factories after shutdown has begun.
        raise V2ServiceHTTPException(
            status_code=503,
            detail="Dependency health check failed.",
            error_code="DEPENDENCY_UNAVAILABLE",
            retryable=True,
            details={
                **snapshot, "status": "unhealthy", "tier": tier,
                "runtime_failure": "runtime_stopping",
                "runtime_cleanup": retirement.plan.status(),
            },
        )
    embedding = embedding_runtime_status()
    if (
        os.getenv("APP_ENV", "development").strip().casefold() == "production"
        and not embedding["embedding_runtime_ready"]
    ):
        snapshot["status"] = "unhealthy"
        snapshot["runtime_failure"] = "embedding_runtime_not_ready"
    warmup_required = embedding_warmup_enabled()
    warmup_ready = getattr(request.app.state, "inference_warmup_ready", False) is True
    if warmup_required and not warmup_ready:
        snapshot["status"] = "unhealthy"
        snapshot["runtime_failure"] = "inference_warmup_not_ready"
    result = {
        **snapshot,
        "runtime_cleanup": {"state": "not_started", "failed_steps": []},
        **getattr(request.app.state, "inference_warmup", {}),
        # Startup identity is historical; live model/cache fields must win.
        **embedding,
        "inference_warmup_required": warmup_required,
        "inference_warmup_ready": warmup_ready,
        "inference_warmup_duration_seconds": getattr(
            request.app.state, "inference_warmup_duration_seconds", None
        ),
        "tier": tier,
        "service_executors": {
            **service_runtime_status(),
            "recommendation": recommendation_runtime_status(),
        },
        "database_required": False,
    }
    if tier == "deep":
        from utils.observability import diagnostic_snapshot
        result["observability"] = diagnostic_snapshot()
    if result["status"] != "healthy":
        raise V2ServiceHTTPException(
            status_code=503,
            detail="Dependency health check failed.",
            error_code="DEPENDENCY_UNAVAILABLE",
            retryable=True,
            details=result,
        )
    return result


@router.get("/health/live", dependencies=[Depends(require_internal_secret)])
async def liveness():
    """Authenticated process liveness: no clients, model loads, or executor jobs."""
    return {"status": "alive", "tier": "liveness"}


@router.get("/health/ready", dependencies=[Depends(require_internal_secret)])
@router.get("/health", dependencies=[Depends(require_internal_secret)])
async def health(request: Request):
    """Legacy-compatible detailed readiness from explicitly dated observations."""
    return _readiness_response(health_diagnostics().snapshot(), request)


@router.get("/health/deep", dependencies=[Depends(require_internal_secret)])
async def deep_health(request: Request):
    """Explicit bounded refresh, coalesced and rate-limited across all callers."""
    return _readiness_response(await health_diagnostics().refresh(), request, tier="deep")


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    dependencies=[Depends(require_internal_secret)],
)
async def metrics() -> PlainTextResponse:
    """Expose fixed-cardinality Prometheus metrics behind internal auth."""

    return PlainTextResponse(
        render_prometheus(
            {
                **service_runtime_status(),
                "recommendation": recommendation_runtime_status(),
            }
        ),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


def _search_repository_job(request: RepositorySearchRequest):
    from api.search import search_repositories
    check_deadline()
    search_retriever = retriever()
    check_deadline()
    model = shared_embedding_model()
    embedder = CachedQueryEmbedder(
        model=model, cache=model.query_embedding_cache,
        model_revision=model.query_embedding_revision,
        embedding_version=search_retriever.embedding_version,
        expected_size=search_retriever.embedding_dim,
        wait_seconds=service_timeout_seconds("search"),
    )
    try:
        return search_repositories(request, search_retriever, embedder)
    except QueryCacheCapacityError as exc:
        raise ServiceCapacityError("search") from exc
    except TimeoutError as exc:
        raise ServiceDeadlineExceeded("search") from exc


@router.post(
    "/search/repositories",
    response_model=RepositorySearchResponse,
    dependencies=[Depends(require_internal_secret)],
)
async def repository_search(request: RepositorySearchRequest, http_request: Request):
    try:
        return await run_read_only_request(
            http_request, service_timeout_seconds("search"),
            lambda: run_service_job("search", _search_repository_job, request),
        )
    except HTTPException:
        raise
    except (ServiceCapacityError, ServiceDeadlineExceeded) as exc:
        raise HTTPException(
            status_code=503, detail="Semantic search is temporarily unavailable."
        ) from exc
    except Exception as exc:
        raise _operation_error(
            "repository_search", exc, request_id=str(request.search_id)
        ) from exc


@router.post("/model-operations", response_model=ModelOperationResponse, dependencies=[Depends(require_internal_secret)])
async def submit_model_operations(request: ModelOperationBatch):
    from feedback.model_runtime import model_runtime, ModelLaneUnavailable
    from feedback.model_stream import OperationBytesConflict, OperationCapacity
    from feedback.model_wire import WireError, parse_operation
    try:
        runtime = model_runtime()
        # Validate the whole bounded batch before accepting its first element.
        # A capacity failure may still leave a prefix durably accepted; exact
        # retries report duplicate and never apply that prefix twice.
        for item in request.operations:
            parse_operation(item.operation_json, maximum_bytes=runtime.stream.limits.input_bytes,
                            allowed_policies=runtime.stream.policies)
        def enqueue():
            return {"schema_version": 1, "operations": [runtime.stream.admit(item.operation_json) for item in request.operations]}
        return await run_service_job("feedback", enqueue)
    except WireError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except OperationBytesConflict:
        raise HTTPException(status_code=409, detail="Model operation original bytes conflict.") from None
    except (ModelLaneUnavailable, OperationCapacity, ServiceCapacityError, ServiceDeadlineExceeded):
        raise HTTPException(status_code=503, detail="Model-operation acceptance unavailable; retry exact original bytes.",
                            headers={"Retry-After": "2"}) from None
    except Exception:
        # Storage errors and uncertain admission remain retryable. No original
        # feedback/profile text or credential-bearing dependency detail escapes.
        raise HTTPException(status_code=503, detail="Model-operation storage unavailable; retry exact original bytes.",
                            headers={"Retry-After": "2"}) from None


@router.post("/model-authority", response_model=ModelAuthorityResponse, dependencies=[Depends(require_internal_secret)])
async def read_model_authority(request: ModelAuthorityRequest):
    from feedback.model_runtime import model_runtime
    from feedback.model_serving import Context, owned_input, input_identity
    try:
        runtime = model_runtime()
        context = request.model_context
        owned_context = Context(str(request.user_id), context.session_id, int(context.epoch), context.policy_sha256,
            int(context.minimum_applied_operation_version), int(context.required_profile_version), int(context.as_of_ms))
        def read():
            points = runtime.qdrant.retrieve(collection_name=runtime.settings.user_collection, ids=[str(request.user_id)],
                with_vectors=True, with_payload=True, timeout=max(1, math.ceil(runtime.settings.qdrant_timeout_seconds)))
            if len(points) != 1:
                raise RuntimeError("applied user model is pending")
            effective = owned_input(points[0], runtime.users, runtime.policy, owned_context)
            return {"schema_version": 1, "model_context": input_identity(effective)}
        return await run_service_job("feedback", read)
    except Exception:
        raise HTTPException(status_code=503, detail="Required applied model/profile/session state unavailable; retry.",
                            headers={"Retry-After": "2"}) from None
