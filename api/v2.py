from __future__ import annotations

import asyncio
import copy
import logging
import math
import os
import time
from dataclasses import asdict
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import PlainTextResponse
from api.contracts import (
    FeedbackBatch,
    OnboardingJob,
    RecommendationRequest,
    RepositoryJob,
    RepositoryFeaturePatch,
    RepositoryRefreshJob,
    RepositorySearchRequest,
    RepositorySearchResponse,
)
from api.runtime import (
    ServiceCapacityError,
    ServiceDeadlineExceeded,
    run_service_job,
    service_runtime_status,
    service_timeout_seconds,
    shutdown_service_runtime,
    validate_service_runtime,
)
from api.metrics import record_recommendation, render_prometheus
from config import constant_time_secret_matches, internal_api_header_name
from embedding.qdrant_store import QdrantRepositoryStore
from embedding.query_cache import CachedQueryEmbedder, QueryCacheCapacityError
from embedding.runtime import (
    EmbeddingCapacityError,
    EmbeddingDeadlineExceeded,
    embedding_runtime_status,
    embedding_timeout_seconds,
    embedding_warmup_enabled,
    repository_embedding_pipeline,
    shared_embedding_model,
    run_embedding_job,
    shutdown_embedding_runtime,
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
    shutdown_recommendation_runtime,
    validate_recommendation_runtime,
)
from retrieval.v2_retriever import QdrantV2Retriever, RetrievalDependencyError
from summarization.contracts import (
    CARD_SUMMARY_PAYLOAD_FIELDS,
    CardSummaryArtifact,
    card_summary_from_payload,
)
from summarization.provider import SUMMARY_COMPLETION_RESERVE_SECONDS
from summarization.settings import SummarySettings
from utils.readme_processor import clean_markdown_copy, process_markdown

router = APIRouter(prefix="/api/v2", tags=["v2"])
logger = logging.getLogger("pipeline.api.v2")


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
        details: dict[str, Any] | None = None,
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
def retriever() -> QdrantV2Retriever:
    return QdrantV2Retriever()


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
    return retriever().health()


def _feedback_health() -> dict[str, Any]:
    return producer().health()


@lru_cache(maxsize=1)
def repository_store() -> QdrantRepositoryStore:
    pipeline = repository_embedding_pipeline()
    store = QdrantRepositoryStore(
        vector_size=pipeline.config.embedding_dim,
        client=retriever().client,
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


def shutdown_v2_runtime() -> None:
    """Close process-scoped clients and executors without creating new ones."""
    if producer.cache_info().currsize:
        redis = producer().redis
        close = getattr(redis, "close", None)
        if close:
            close()
    if retriever.cache_info().currsize:
        close = getattr(retriever().client, "close", None)
        if close:
            close()
    repository_store.cache_clear()
    user_profile_store.cache_clear()
    producer.cache_clear()
    retriever.cache_clear()
    feedback_settings.cache_clear()
    shutdown_recommendation_runtime()
    shutdown_service_runtime()
    shutdown_embedding_runtime()


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
    settings = feedback_settings()
    _repository_job_lock_settings()
    _health_timeout_seconds()
    embedding_timeout_seconds()
    embedding_warmup_enabled()
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
        if artifact is None or not pipeline.card_summarizer.is_current(artifact) or (
            job_status != "duplicate"
            and pipeline.card_summarizer.should_upgrade(artifact)
            and _summary_upgrade_has_time(deadline)
        ):
            payload = _repository_embedding_payload(request)
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
        if artifact is None or not pipeline.card_summarizer.is_current(artifact):
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
    payload = _repository_embedding_payload(request)
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
    if stored_artifact is None:
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

    features = request.features.model_dump(mode="json", exclude_none=True)
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


@router.post(
    "/recommendations/generate", dependencies=[Depends(require_internal_secret)]
)
async def generate_recommendations(
    request: RecommendationRequest,
    http_request: Request,
):
    try:
        batch = await run_recommendation_job(
            retriever().recommend_batch,
            str(request.user_id),
            request.limit,
            [str(item) for item in request.exclude_repo_ids],
            str(request.generation_id),
            request.context.cold_start,
            [str(item) for item in request.social_repo_ids],
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
    except Exception as exc:
        raise _operation_error(
            "recommendations",
            exc,
            request_id=str(http_request.state.request_id),
        ) from exc
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
        "items": [asdict(item) for item in items],
    }


@router.post(
    "/feedback/batch",
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
    return {"accepted": accepted, "duplicates": duplicates, "durable": True}


@router.post("/repositories/embed", dependencies=[Depends(require_internal_secret)])
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


@router.post("/repositories/refresh", dependencies=[Depends(require_internal_secret)])
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


@router.post("/users/onboard", dependencies=[Depends(require_internal_secret)])
async def onboard_user(request: OnboardingJob, http_request: Request):
    try:
        return await run_embedding_job(_onboard_user_job, request)
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


@router.get("/health", dependencies=[Depends(require_internal_secret)])
async def health(request: Request):
    try:
        qdrant, redis = await asyncio.gather(
            run_service_job("health", _qdrant_health),
            run_service_job("health", _feedback_health),
        )
        embedding = embedding_runtime_status()
        consumer_required = os.getenv(
            "V2_FEEDBACK_CONSUMER_REQUIRED",
            "true",
        ).strip().lower() in {"1", "true", "yes", "on"}
        if consumer_required and not redis.get("feedback_consumer_active"):
            raise RuntimeError("V2 feedback consumer heartbeat is missing")
        if not redis.get("feedback_healthy", True):
            raise RuntimeError("V2 feedback thresholds exceeded")
        if (
            os.getenv("APP_ENV", "development").strip().casefold() == "production"
            and not embedding["embedding_runtime_ready"]
        ):
            raise RuntimeError("embedding runtime is not ready")
        return {
            "status": "healthy",
            **qdrant,
            **redis,
            **embedding,
            "service_executors": {
                **service_runtime_status(),
                "recommendation": recommendation_runtime_status(),
            },
            "database_required": False,
        }
    except Exception as exc:
        logger.error(
            "V2 dependency health check failed request_id=%s error_type=%s",
            str(request.state.request_id),
            type(exc).__name__,
            extra={
                "health_context": {
                    "request_id": str(request.state.request_id),
                    "error_type": type(exc).__name__,
                }
            },
        )
        raise HTTPException(
            status_code=503,
            detail="Dependency health check failed.",
        ) from exc


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
    search_retriever = retriever()
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
async def repository_search(request: RepositorySearchRequest):
    try:
        return await run_service_job("search", _search_repository_job, request)
    except (ServiceCapacityError, ServiceDeadlineExceeded) as exc:
        raise HTTPException(
            status_code=503, detail="Semantic search is temporarily unavailable."
        ) from exc
    except Exception as exc:
        raise _operation_error(
            "repository_search", exc, request_id=str(request.search_id)
        ) from exc
