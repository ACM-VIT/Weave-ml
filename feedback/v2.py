"""Durable, ordered v2 feedback producer and consumer.

The stream is the source of truth until a message is either applied or copied
to the bounded dead-letter stream.  Retryable dependency failures are never
acknowledged early, and every user-vector mutation is serialized by the shared
renewable lock from :mod:`feedback.user_lock`.
"""

from __future__ import annotations

import json
import hashlib
import logging
import math
import os
import re
import signal
import socket
import threading
import uuid
from copy import deepcopy
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import PointStruct

# ``python -m feedback.v2`` is the documented local worker command. Load the
# same project environment as the API before importing configuration constants
# so both processes use one Redis/Qdrant/release contract. Library imports do
# not load developer environment files into tests or embedding callers.
if __name__ == "__main__":
    load_dotenv()

from embedding.qdrant_cas import payload_matches, payload_snapshot_filter
from config import (
    EMBEDDING_MODEL_REVISION,
    REPOSITORY_EMBEDDING_MODEL,
    REPOSITORY_EMBEDDING_VERSION,
    REPOSITORY_FEATURE_SPEC_VERSION,
)
from embedding.vector_contract import (
    FEEDBACK_STATE_REVISION_FIELD,
    REPOSITORY_SERVING_ELIGIBILITY_FIELD,
    REPOSITORY_SERVING_ELIGIBILITY_VERSION,
    repository_point_id,
    user_point_id,
)
from feedback.event_handlers import (
    ADJUSTMENTS_KEY,
    APPLIED_SIGNALS_KEY,
    LATENT_KEY,
    vector_delta,
)
from feedback.interactions import get_interaction
from feedback.safe_logging import configure_feedback_worker_logging
from feedback.user_lock import LockAcquisitionError, LockLostError, user_vector_lock
from feedback.v2_settings import V2FeedbackSettings
from feedback.user_scheduler import UserFeedbackScheduler, Work


logger = logging.getLogger(__name__)

# Compatibility exports.  New code should read the values from
# ``V2FeedbackSettings`` so producer, consumer, health, and replay agree.
STREAM = "ml:feedback:v2"
GROUP = "ml-feedback-v2"
CONSUMER_HEARTBEAT = "ml:feedback:v2:consumer-heartbeat"
CONSUMER_HEARTBEAT_TTL_SECONDS: float = 15

ACCEPT_LUA = """
local accepted_fingerprint = redis.call('get', KEYS[1])
if accepted_fingerprint then
  if accepted_fingerprint == ARGV[1] then
    return 'duplicate'
  end
  return 'event_id_conflict'
end
if redis.call('xlen', KEYS[2]) >= tonumber(ARGV[3]) then
  return 'overloaded'
end
redis.call('set', KEYS[1], ARGV[1], 'EX', ARGV[2])
local result = redis.pcall('xadd', KEYS[2], '*', unpack(ARGV, 4))
if type(result) == 'table' and result.err then
  redis.call('del', KEYS[1])
  return redis.error_reply(result.err)
end
return result
"""
ACK_DELETE_LUA = """
local acknowledged = redis.call('xack', KEYS[1], ARGV[1], ARGV[2])
if acknowledged == 1 then
  redis.call('xdel', KEYS[1], ARGV[2])
end
return acknowledged
"""
HEARTBEAT_DELETE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
end
return 0
"""
DLQ_MOVE_LUA = """
if redis.call('get', KEYS[2]) then
  return 'duplicate'
end
if redis.call('xlen', KEYS[1]) >= tonumber(ARGV[1]) then
  return 'overloaded'
end
local result = redis.pcall('xadd', KEYS[1], '*', unpack(ARGV, 3))
if type(result) == 'table' and result.err then
  return redis.error_reply(result.err)
end
redis.call('set', KEYS[2], result, 'EX', ARGV[2])
return result
"""

ALPHAS = {
    "readme_open": 0.05,
    "github_open": 0.07,
    "share": 0.10,
    "like": 0.15,
    "dislike": -0.15,
    "save": 0.20,
    "unlike": 0.0,
    "undislike": 0.0,
    "unsave": 0.0,
}

REJECTIONS_KEY = "feedback_rejections"
PENDING_REJECTION_KEY = "pending_feedback_rejection"
_USER_WRITE_CAS_FIELDS = (
    "profile_version",
    "job_id",
    "last_feedback_version",
    "last_feedback_event_id",
    FEEDBACK_STATE_REVISION_FIELD,
)
_DLQ_METADATA_FIELDS = {
    "source_stream",
    "source_message_id",
    "failure_code",
    "failure_reason",
    "retryable",
    "attempts",
    "terminal_status",
    "cursor_advanced",
    "failed_at",
}


class FeedbackProcessingError(RuntimeError):
    code = "FEEDBACK_PROCESSING_ERROR"
    retryable = False
    public_message = "feedback could not be processed"


class PermanentFeedbackError(FeedbackProcessingError, ValueError):
    code = "EVENT_INVALID"
    public_message = "feedback event is invalid"


class VersionEventConflict(PermanentFeedbackError):
    code = "VERSION_EVENT_CONFLICT"
    public_message = "feedback version is already associated with another event"


class TrackedRepositoryLimitError(PermanentFeedbackError):
    code = "TRACKED_REPOSITORY_LIMIT"
    public_message = "user feedback repository state limit is reached"


class FeedbackStateLimitError(PermanentFeedbackError):
    code = "USER_STATE_SIZE_LIMIT"
    public_message = "user feedback state size limit is reached"


class PreviouslyRejectedFeedbackError(PermanentFeedbackError):
    """Re-finalize a cursor-advanced rejection until its DLQ move succeeds."""

    def __init__(self, *, code: str, reason: str) -> None:
        self.code = _safe_text(code, 64) or "PREVIOUSLY_REJECTED"
        self.public_message = _safe_text(reason) or "feedback was previously rejected"
        super().__init__(self.public_message)


class FeedbackStateError(FeedbackProcessingError, ValueError):
    code = "USER_STATE_INVALID"
    public_message = "stored user feedback state is invalid"


class MissingVectorError(FeedbackProcessingError, LookupError):
    retryable = True

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.code = f"{kind.upper()}_VECTOR_MISSING"
        self.public_message = f"{kind} vector is not indexed yet"
        super().__init__(self.public_message)


class FeedbackDependencyError(FeedbackProcessingError):
    code = "DEPENDENCY_UNAVAILABLE"
    retryable = True
    public_message = "feedback dependency is temporarily unavailable"


class FeedbackWriteConflict(FeedbackDependencyError):
    code = "USER_PROFILE_WRITE_CONFLICT"
    public_message = "user profile changed while feedback was being applied"


class DeadLetterFullError(FeedbackProcessingError):
    code = "DEAD_LETTER_FULL"
    retryable = True
    public_message = "feedback dead-letter capacity is exhausted"


class FeedbackEnqueueError(RuntimeError):
    """Batch enqueue stopped after a durable per-event Redis failure."""

    def __init__(self, *, accepted: int, duplicates: int, failed_event_id: str) -> None:
        self.accepted = accepted
        self.duplicates = duplicates
        self.failed_event_id = failed_event_id
        super().__init__(
            "feedback enqueue failed after a partial batch; retrying the full batch is safe"
        )


class FeedbackStreamFullError(FeedbackEnqueueError):
    """The exact outstanding-work capacity rejected an enqueue atomically."""

    code = "FEEDBACK_STREAM_FULL"
    retryable = True

    def __init__(
        self,
        *,
        accepted: int,
        duplicates: int,
        failed_event_id: str,
        capacity: int,
    ) -> None:
        self.capacity = capacity
        super().__init__(
            accepted=accepted,
            duplicates=duplicates,
            failed_event_id=failed_event_id,
        )


class FeedbackEventIdConflictError(FeedbackEnqueueError):
    """The event ID was previously accepted with a different payload."""

    code = "EVENT_ID_PAYLOAD_CONFLICT"
    retryable = False


def _safe_text(value: Any, limit: int = 256) -> str:
    text = str(value or "").replace("\x00", "").replace("\r", " ").replace("\n", " ")
    return text[:limit]


def _redis_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _event_fingerprint(event: Mapping[str, Any]) -> str:
    try:
        canonical = json.dumps(
            dict(event),
            allow_nan=False,
            default=str,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("feedback event contains a non-canonical value") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _redis_client(settings: V2FeedbackSettings | None = None):
    runtime = settings or V2FeedbackSettings.from_env()
    try:
        import redis
    except ImportError as exc:
        raise RuntimeError("redis>=5 is required for the production v2 feedback boundary") from exc
    if not runtime.redis_url:
        raise RuntimeError("REDIS_URL is required for durable v2 feedback")
    client = redis.from_url(
        runtime.redis_url,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=5,
        health_check_interval=30,
    )
    client.ping()
    return client


def _require_xautoclaim_redis(redis_client: Any) -> str:
    """Require Redis 6.2+, which introduced the pending-work XAUTOCLAIM path."""

    info_method = getattr(redis_client, "info", None)
    if not callable(info_method):
        # Minimal injected test doubles do not expose server metadata.
        return "test-double"
    info = info_method(section="server")
    if not isinstance(info, Mapping):
        return "test-double"
    raw_version = str(info.get("redis_version") or "")
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?.*", raw_version)
    if not match:
        raise RuntimeError("Redis server version could not be validated")
    major, minor = int(match.group(1)), int(match.group(2))
    if (major, minor) < (6, 2):
        raise RuntimeError("Redis 6.2 or newer is required for feedback pending recovery")
    return raw_version[:32]


def _runtime_release_id(*, production: bool) -> str:
    release_id = os.getenv("ML_RELEASE_ID", "development").strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{7,64}", release_id):
        raise RuntimeError("ML_RELEASE_ID must contain 7-64 safe characters")
    if production and release_id.casefold() in {
        "development",
        "replace-me",
        "placeholder",
    }:
        raise RuntimeError("ML_RELEASE_ID must identify the immutable production image")
    return release_id


class DurableFeedbackProducer:
    def __init__(
        self,
        redis_client: Any | None = None,
        settings: V2FeedbackSettings | None = None,
    ) -> None:
        self.settings = settings or V2FeedbackSettings.from_env()
        self._owns_redis = redis_client is None
        self.redis = redis_client or _redis_client(self.settings)

    def enqueue(self, events: Iterable[dict[str, Any]]) -> tuple[int, int]:
        commands: list[tuple[str, str, list[str]]] = []
        for event in events:
            raw_event_id = event.get("event_id")
            try:
                event_id = str(uuid.UUID(str(raw_event_id)))
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError("event_id must be a canonical UUID") from exc
            fields: list[str] = []
            for key, value in event.items():
                fields.extend([str(key), _redis_field(value)])
            commands.append((event_id, _event_fingerprint(event), fields))

        if not commands:
            return 0, 0

        accepted = 0
        duplicates = 0
        if len(commands) > 1 and callable(getattr(self.redis, "pipeline", None)):
            pipeline = self.redis.pipeline(transaction=False)
            for event_id, fingerprint, fields in commands:
                pipeline.eval(
                    ACCEPT_LUA,
                    2,
                    f"{self.settings.stream_name}:accepted:{event_id}",
                    self.settings.stream_name,
                    fingerprint,
                    str(self.settings.idempotency_ttl_seconds),
                    str(self.settings.stream_maxlen),
                    *fields,
                )
            try:
                # All event scripts remain individually atomic; pipelining only
                # removes up to 99 avoidable network round trips per API batch.
                results = pipeline.execute(raise_on_error=False)
            except Exception as exc:
                failed_event_id = commands[0][0]
                logger.error(
                    "feedback batch pipeline failed",
                    extra={
                        "feedback_context": {
                            "event_id": failed_event_id,
                            "status": "enqueue_failed",
                            "accepted_before_failure": 0,
                            "duplicates_before_failure": 0,
                        }
                    },
                )
                raise FeedbackEnqueueError(
                    accepted=0,
                    duplicates=0,
                    failed_event_id=failed_event_id,
                ) from exc

            failed_event_id: str | None = None
            first_error: Exception | None = None
            overloaded_event_id: str | None = None
            conflicting_event_id: str | None = None
            for (event_id, _, _), result in zip(commands, results, strict=True):
                if isinstance(result, Exception):
                    failed_event_id = failed_event_id or event_id
                    first_error = first_error or result
                elif result == "overloaded" or result == b"overloaded":
                    overloaded_event_id = overloaded_event_id or event_id
                elif result == "event_id_conflict" or result == b"event_id_conflict":
                    conflicting_event_id = conflicting_event_id or event_id
                elif result == "duplicate" or result == b"duplicate":
                    duplicates += 1
                else:
                    accepted += 1
            if conflicting_event_id is not None:
                logger.error(
                    "feedback event ID payload conflict",
                    extra={
                        "feedback_context": {
                            "event_id": conflicting_event_id,
                            "status": "event_id_conflict",
                            "code": FeedbackEventIdConflictError.code,
                        }
                    },
                )
                raise FeedbackEventIdConflictError(
                    accepted=accepted,
                    duplicates=duplicates,
                    failed_event_id=conflicting_event_id,
                )
            if overloaded_event_id is not None:
                logger.warning(
                    "feedback stream capacity rejected a batch event",
                    extra={
                        "feedback_context": {
                            "event_id": overloaded_event_id,
                            "status": "stream_full",
                            "code": FeedbackStreamFullError.code,
                            "accepted_before_failure": accepted,
                            "duplicates_before_failure": duplicates,
                        }
                    },
                )
                raise FeedbackStreamFullError(
                    accepted=accepted,
                    duplicates=duplicates,
                    failed_event_id=overloaded_event_id,
                    capacity=self.settings.stream_maxlen,
                )
            if failed_event_id is not None:
                logger.error(
                    "feedback batch contained a failed atomic enqueue",
                    extra={
                        "feedback_context": {
                            "event_id": failed_event_id,
                            "status": "enqueue_failed",
                            "accepted_before_failure": accepted,
                            "duplicates_before_failure": duplicates,
                        }
                    },
                )
                raise FeedbackEnqueueError(
                    accepted=accepted,
                    duplicates=duplicates,
                    failed_event_id=failed_event_id,
                ) from first_error
            return accepted, duplicates

        for event_id, fingerprint, fields in commands:
            try:
                result = self.redis.eval(
                    ACCEPT_LUA,
                    2,
                    f"{self.settings.stream_name}:accepted:{event_id}",
                    self.settings.stream_name,
                    fingerprint,
                    str(self.settings.idempotency_ttl_seconds),
                    str(self.settings.stream_maxlen),
                    *fields,
                )
            except Exception as exc:
                logger.error(
                    "feedback enqueue failed",
                    extra={
                        "feedback_context": {
                            "event_id": event_id,
                            "status": "enqueue_failed",
                            "accepted_before_failure": accepted,
                            "duplicates_before_failure": duplicates,
                        }
                    },
                )
                raise FeedbackEnqueueError(
                    accepted=accepted,
                    duplicates=duplicates,
                    failed_event_id=event_id,
                ) from exc
            if result == "duplicate" or result == b"duplicate":
                duplicates += 1
            elif result == "event_id_conflict" or result == b"event_id_conflict":
                logger.error(
                    "feedback event ID payload conflict",
                    extra={
                        "feedback_context": {
                            "event_id": event_id,
                            "status": "event_id_conflict",
                            "code": FeedbackEventIdConflictError.code,
                        }
                    },
                )
                raise FeedbackEventIdConflictError(
                    accepted=accepted,
                    duplicates=duplicates,
                    failed_event_id=event_id,
                )
            elif result == "overloaded" or result == b"overloaded":
                logger.warning(
                    "feedback stream capacity rejected an event",
                    extra={
                        "feedback_context": {
                            "event_id": event_id,
                            "status": "stream_full",
                            "code": FeedbackStreamFullError.code,
                            "accepted_before_failure": accepted,
                            "duplicates_before_failure": duplicates,
                        }
                    },
                )
                raise FeedbackStreamFullError(
                    accepted=accepted,
                    duplicates=duplicates,
                    failed_event_id=event_id,
                    capacity=self.settings.stream_maxlen,
                )
            else:
                accepted += 1
        return accepted, duplicates

    @staticmethod
    def _int_metric(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def health(self) -> dict[str, Any]:
        self.redis.ping()
        redis_version = _require_xautoclaim_redis(self.redis)
        group: Mapping[str, Any] = {}
        try:
            groups = self.redis.xinfo_groups(self.settings.stream_name)
            group = next(
                (
                    item
                    for item in groups
                    if item.get("name") == self.settings.consumer_group
                    or item.get(b"name") == self.settings.consumer_group.encode()
                ),
                {},
            )
        except Exception as exc:
            message = str(exc).lower()
            if "no such key" not in message and "nogroup" not in message:
                raise

        pending = self._int_metric(group.get("pending", group.get(b"pending", 0)))
        stream_length = self._int_metric(self.redis.xlen(self.settings.stream_name))
        raw_lag = group.get("lag", group.get(b"lag"))
        # Redis 6.2 supports XAUTOCLAIM but does not always expose consumer
        # group lag. ACKed entries are atomically deleted, so XLEN - pending is
        # a bounded conservative fallback for undelivered work.
        lag = (
            max(0, stream_length - pending)
            if raw_lag is None
            else self._int_metric(raw_lag)
        )
        dead_length = self._int_metric(self.redis.xlen(self.settings.dead_letter_stream))
        raw_heartbeat = self.redis.get(self.settings.heartbeat_key)
        heartbeat = (
            raw_heartbeat.decode("utf-8")
            if isinstance(raw_heartbeat, bytes)
            else str(raw_heartbeat or "")
        )
        release_id = _runtime_release_id(production=self.settings.production)
        consumer_active = heartbeat.startswith(f"{release_id}|")

        warnings: list[str] = []
        failures: list[str] = []
        for metric, value, warning, maximum in (
            ("pending", pending, self.settings.health_warn_pending, self.settings.health_max_pending),
            ("lag", lag, self.settings.health_warn_lag, self.settings.health_max_lag),
            (
                "stream_length",
                stream_length,
                self.settings.health_warn_stream_length,
                self.settings.health_max_stream_length,
            ),
            (
                "dead_letter_length",
                dead_length,
                self.settings.health_warn_dead_letter,
                self.settings.health_max_dead_letter,
            ),
        ):
            if value >= maximum:
                failures.append(metric)
            elif value >= warning and (warning > 0 or value > 0):
                warnings.append(metric)
        feedback_status = "unhealthy" if failures else "warning" if warnings else "healthy"
        return {
            "redis": "healthy",
            "redis_version": redis_version,
            "feedback_pending": pending,
            "feedback_lag": lag,
            "feedback_stream_length": stream_length,
            "feedback_dead_letter_length": dead_length,
            "feedback_consumer_active": consumer_active,
            "feedback_consumer_release_id": release_id if consumer_active else None,
            "feedback_consumer_release_mismatch": bool(heartbeat) and not consumer_active,
            "feedback_status": feedback_status,
            "feedback_healthy": not failures,
            "feedback_warnings": warnings,
            "feedback_failures": failures,
        }

    def close(self) -> None:
        if self._owns_redis and hasattr(self.redis, "close"):
            self.redis.close()


@dataclass(frozen=True, slots=True)
class ApplyResult:
    status: str
    last_feedback_version: int
    cursor_advanced: bool = False


class OrderedFeedbackApplier:
    def __init__(
        self,
        qdrant: QdrantClient | None = None,
        settings: V2FeedbackSettings | None = None,
    ) -> None:
        self.settings = settings or V2FeedbackSettings.from_env()
        self._owns_qdrant = qdrant is None
        self._enforce_dimension = settings is not None or qdrant is None
        self._repository_batches = threading.local()
        self.qdrant = qdrant or QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            timeout=self.settings.qdrant_timeout_seconds,
        )

    @staticmethod
    def _vector(value: Any, preferred: str | None = None) -> tuple[list[float], str | None]:
        if isinstance(value, Mapping):
            if preferred and preferred in value:
                return list(value[preferred]), preferred
            if len(value) == 1:
                name, vector = next(iter(value.items()))
                return list(vector), str(name)
            raise FeedbackStateError("stored point has an ambiguous named vector")
        if value is None:
            raise FeedbackStateError("stored point is missing its vector")
        return list(value), None

    @staticmethod
    def _canonical_uuid(value: Any, field: str) -> str:
        try:
            return str(uuid.UUID(str(value)))
        except (TypeError, ValueError, AttributeError) as exc:
            raise PermanentFeedbackError(f"{field} must be a UUID") from exc

    def normalize_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(event, Mapping):
            raise PermanentFeedbackError("feedback event must be an object")
        normalized = dict(event)
        normalized["event_id"] = self._canonical_uuid(event.get("event_id"), "event_id")
        normalized["user_id"] = self._canonical_uuid(event.get("user_id"), "user_id")
        normalized["repo_id"] = self._canonical_uuid(event.get("repo_id"), "repo_id")
        raw_version = event.get("feedback_version")
        if isinstance(raw_version, bool):
            raise PermanentFeedbackError("feedback_version must be a positive integer")
        try:
            version = int(raw_version)
        except (TypeError, ValueError) as exc:
            raise PermanentFeedbackError("feedback_version must be a positive integer") from exc
        if version < 1 or str(raw_version).strip() not in {str(version), f"+{version}"}:
            raise PermanentFeedbackError("feedback_version must be a positive integer")
        normalized["feedback_version"] = version

        event_type = str(event.get("event_type") or "").strip().lower()
        try:
            definition = get_interaction(event_type)
        except (KeyError, ValueError) as exc:
            raise PermanentFeedbackError("event_type is not supported") from exc
        if event_type == "impression" or not definition.realtime:
            raise PermanentFeedbackError("offline-only events cannot modify the online vector")
        normalized["event_type"] = event_type

        raw_dwell = event.get("dwell_ms")
        if event_type == "dwell":
            if isinstance(raw_dwell, bool):
                raise PermanentFeedbackError("dwell_ms must be an integer")
            try:
                dwell_ms = int(raw_dwell)
            except (TypeError, ValueError) as exc:
                raise PermanentFeedbackError("dwell_ms must be an integer") from exc
            if not self.settings.dwell_min_ms <= dwell_ms <= self.settings.dwell_full_credit_ms:
                raise PermanentFeedbackError("dwell_ms is outside the configured range")
            normalized["dwell_ms"] = dwell_ms
        elif raw_dwell not in {None, ""}:
            raise PermanentFeedbackError("only dwell events may carry dwell_ms")

        raw_occurred_at = event.get("occurred_at")
        if raw_occurred_at is not None:
            try:
                occurred_at = (
                    raw_occurred_at
                    if isinstance(raw_occurred_at, datetime)
                    else datetime.fromisoformat(str(raw_occurred_at).replace("Z", "+00:00"))
                )
            except (TypeError, ValueError) as exc:
                raise PermanentFeedbackError("occurred_at must be an ISO-8601 timestamp") from exc
            offset = occurred_at.utcoffset()
            if occurred_at.tzinfo is None or offset is None or offset.total_seconds() != 0:
                raise PermanentFeedbackError("occurred_at must use UTC")
            normalized["occurred_at"] = occurred_at.isoformat()
        return normalized

    def _alpha(self, event: Mapping[str, Any]) -> float:
        if event["event_type"] != "dwell":
            return ALPHAS[str(event["event_type"])]
        dwell = min(
            self.settings.dwell_full_credit_ms,
            max(self.settings.dwell_min_ms, int(event["dwell_ms"])),
        )
        return (
            self.settings.dwell_max_alpha
            * math.log1p(dwell)
            / math.log1p(self.settings.dwell_full_credit_ms)
        )

    @staticmethod
    def _finite_vector(value: Any, dimension: int, *, label: str) -> np.ndarray:
        try:
            vector = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise FeedbackStateError(f"{label} is not numeric") from exc
        if vector.ndim != 1 or len(vector) != dimension:
            raise FeedbackStateError(f"{label} has the wrong dimension")
        if not np.all(np.isfinite(vector)):
            raise FeedbackStateError(f"{label} contains a non-finite value")
        return vector

    @staticmethod
    def _adjustments(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        value = payload.get(ADJUSTMENTS_KEY, {})
        if not isinstance(value, Mapping):
            raise FeedbackStateError(f"{ADJUSTMENTS_KEY} must be an object")
        adjustments = deepcopy(dict(value))
        for repo_id, repo_state in adjustments.items():
            if not isinstance(repo_id, str) or not isinstance(repo_state, Mapping):
                raise FeedbackStateError(f"{ADJUSTMENTS_KEY} contains invalid repository state")
            validated_state: dict[str, Any] = {}
            for family, stored in repo_state.items():
                if not isinstance(family, str) or not isinstance(stored, Mapping):
                    raise FeedbackStateError(f"{ADJUSTMENTS_KEY} contains invalid family state")
                action = stored.get("action")
                if not isinstance(action, str) or not action:
                    raise FeedbackStateError(f"{ADJUSTMENTS_KEY} contains an invalid action")
                validated_state[family] = dict(stored)
            adjustments[repo_id] = validated_state
        return adjustments

    @staticmethod
    def _applied_signals(payload: Mapping[str, Any]) -> dict[str, list[str]]:
        value = payload.get(APPLIED_SIGNALS_KEY, {})
        if not isinstance(value, Mapping):
            raise FeedbackStateError(f"{APPLIED_SIGNALS_KEY} must be an object")
        signals: dict[str, list[str]] = {}
        for repo_id, actions in value.items():
            if not isinstance(repo_id, str) or not isinstance(actions, list):
                raise FeedbackStateError(f"{APPLIED_SIGNALS_KEY} contains invalid repository state")
            if any(not isinstance(action, str) or not action for action in actions):
                raise FeedbackStateError(f"{APPLIED_SIGNALS_KEY} contains an invalid action")
            signals[repo_id] = list(actions)
        return signals

    @staticmethod
    def _payload(point: Any) -> dict[str, Any]:
        if point.payload is None:
            return {}
        if not isinstance(point.payload, Mapping):
            raise FeedbackStateError("user profile payload must be an object")
        return dict(point.payload)

    @staticmethod
    def _feedback_state_bytes(payload: Mapping[str, Any]) -> int:
        state = {
            LATENT_KEY: payload.get(LATENT_KEY),
            ADJUSTMENTS_KEY: payload.get(ADJUSTMENTS_KEY, {}),
            APPLIED_SIGNALS_KEY: payload.get(APPLIED_SIGNALS_KEY, {}),
        }
        try:
            encoded = json.dumps(
                state,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise FeedbackStateError("stored feedback state is not serializable") from exc
        return len(encoded)

    @staticmethod
    def _last_version(payload: Mapping[str, Any]) -> int:
        raw = payload.get("last_feedback_version", 0)
        if isinstance(raw, bool):
            raise FeedbackStateError("last_feedback_version must be a non-negative integer")
        try:
            value = int(raw or 0)
        except (TypeError, ValueError) as exc:
            raise FeedbackStateError("last_feedback_version must be a non-negative integer") from exc
        if value < 0 or str(raw or 0).strip() not in {str(value), f"+{value}"}:
            raise FeedbackStateError("last_feedback_version must be a non-negative integer")
        return value

    @staticmethod
    def _feedback_state_revision(payload: Mapping[str, Any]) -> int:
        raw = payload.get(FEEDBACK_STATE_REVISION_FIELD, 0)
        if isinstance(raw, bool):
            raise FeedbackStateError(
                f"{FEEDBACK_STATE_REVISION_FIELD} must be a non-negative integer"
            )
        try:
            value = int(raw or 0)
        except (TypeError, ValueError) as exc:
            raise FeedbackStateError(
                f"{FEEDBACK_STATE_REVISION_FIELD} must be a non-negative integer"
            ) from exc
        if (
            value < 0
            or value >= 9_223_372_036_854_775_807
            or str(raw or 0).strip() not in {str(value), f"+{value}"}
        ):
            raise FeedbackStateError(
                f"{FEEDBACK_STATE_REVISION_FIELD} must be a non-negative integer"
            )
        return value

    def _user(self, user_id: str) -> Any:
        canonical_user_id = user_point_id(user_id)
        try:
            users = self.qdrant.retrieve(
                collection_name=self.settings.user_collection,
                ids=[canonical_user_id],
                with_payload=True,
                with_vectors=True,
            )
        except Exception as exc:
            raise FeedbackDependencyError() from exc
        users_by_id = {str(point.id): point for point in users}
        user = users_by_id.get(canonical_user_id)
        if user is None:
            raise MissingVectorError("user")
        return user

    def _repository_points(self, ids: list[str]) -> list[Any]:
        try:
            return self.qdrant.retrieve(
                collection_name=self.settings.repository_collection,
                ids=ids,
                with_payload=[
                    "repo_id",
                    REPOSITORY_SERVING_ELIGIBILITY_FIELD,
                    "content_version",
                    "embedding_model",
                    "embedding_model_revision",
                    "embedding_version",
                    "embedding_dim",
                    "feature_spec_version",
                ],
                with_vectors=True,
            )
        except Exception as exc:
            raise FeedbackDependencyError() from exc

    @contextmanager
    def repository_read_batch(self, events: Iterable[Mapping[str, Any]]):
        """Reuse one bounded repository revision snapshot for this read batch.

        The snapshot includes content/model/embedding revisions and eligibility.
        Nothing survives the boundary: the next Redis read observes updates,
        removal or ineligibility afresh. User ledgers are never prefetched.
        """
        ids: list[str] = []
        for event in events:
            try:
                normalized = self.normalize_event(event)
            except PermanentFeedbackError:
                continue
            if get_interaction(normalized["event_type"]).reversal_of:
                continue
            repo_id = repository_point_id(normalized["repo_id"])
            if repo_id not in ids:
                ids.append(repo_id)
            if len(ids) >= self.settings.read_batch_size:
                break
        snapshots = None
        if ids:
            try:
                points = self._repository_points(ids)
            except FeedbackDependencyError:
                # Preserve independent per-event retries on partial transport
                # failure rather than failing unrelated events as one unit.
                pass
            else:
                snapshots = dict.fromkeys(ids)
                snapshots.update({str(point.id): deepcopy(point) for point in points if str(point.id) in snapshots})
        previous = getattr(self._repository_batches, "points", None)
        self._repository_batches.points = snapshots
        try:
            yield
        finally:
            self._repository_batches.points = previous

    def _repository_vector(self, repo_id: str, dimension: int) -> np.ndarray:
        canonical_repo_id = repository_point_id(repo_id)
        snapshots = getattr(self._repository_batches, "points", None)
        if snapshots is not None and canonical_repo_id in snapshots:
            repo = snapshots[canonical_repo_id]
        else:
            repos = self._repository_points([canonical_repo_id])
            repo = next((point for point in repos if str(point.id) == canonical_repo_id), None)
        if repo is None:
            raise MissingVectorError("repository")
        payload = self._payload(repo)
        compatible_versions = {
            value.strip()
            for value in os.getenv(
                "V2_COMPATIBLE_EMBEDDING_VERSIONS",
                REPOSITORY_EMBEDDING_VERSION,
            ).split(",")
            if value.strip()
        }
        try:
            compatible = (
                str(payload.get("repo_id")) == repo_id
                and payload.get(REPOSITORY_SERVING_ELIGIBILITY_FIELD)
                == REPOSITORY_SERVING_ELIGIBILITY_VERSION
                and int(payload.get("content_version") or 0)
                >= int(os.getenv("V2_REQUIRED_CONTENT_VERSION", "1"))
                and str(payload.get("embedding_model"))
                == os.getenv("EMBEDDING_MODEL", REPOSITORY_EMBEDDING_MODEL)
                and str(payload.get("embedding_model_revision"))
                == os.getenv("EMBEDDING_MODEL_REVISION", EMBEDDING_MODEL_REVISION)
                and str(payload.get("embedding_version")) in compatible_versions
                and int(payload.get("embedding_dim") or 0) == dimension
                and str(payload.get("feature_spec_version"))
                == os.getenv(
                    "V2_REQUIRED_FEATURE_SPEC_VERSION",
                    REPOSITORY_FEATURE_SPEC_VERSION,
                )
            )
        except (TypeError, ValueError):
            compatible = False
        if not compatible:
            raise MissingVectorError("repository")
        value, _ = self._vector(repo.vector, self.settings.repository_vector_name)
        return self._finite_vector(value, dimension, label="repository vector")

    def _upsert(self, point: PointStruct, *, expected_payload: Mapping[str, Any]) -> None:
        """Conditionally replace one user point and verify that this write won.

        The Redis user lock serializes feedback workers, while this Qdrant CAS
        also fences onboarding and any other writer that can legitimately
        modify the same point outside that lock.  A conflict is retryable: the
        next attempt reads the newer profile and reapplies the ordered event.
        """

        try:
            self.qdrant.upsert(
                collection_name=self.settings.user_collection,
                points=[point],
                wait=True,
                ordering=models.WriteOrdering.STRONG,
                update_mode=models.UpdateMode.UPDATE_ONLY,
                update_filter=payload_snapshot_filter(
                    models,
                    point_id=point.id,
                    payload=expected_payload,
                    fields=_USER_WRITE_CAS_FIELDS,
                ),
            )
            stored = self.qdrant.retrieve(
                collection_name=self.settings.user_collection,
                ids=[point.id],
                with_payload=list(_USER_WRITE_CAS_FIELDS),
                with_vectors=False,
            )
        except FeedbackProcessingError:
            raise
        except Exception as exc:
            raise FeedbackDependencyError() from exc
        if not stored or not payload_matches(
            self._payload(stored[0]),
            self._payload(point),
            _USER_WRITE_CAS_FIELDS,
        ):
            raise FeedbackWriteConflict()

    def apply(self, event: Mapping[str, Any]) -> ApplyResult:
        normalized = self.normalize_event(event)
        user_id = normalized["user_id"]
        repo_id = normalized["repo_id"]
        version = normalized["feedback_version"]
        user = self._user(user_id)
        payload = self._payload(user)
        if self._feedback_state_bytes(payload) > self.settings.max_user_state_bytes:
            raise FeedbackStateError("stored user feedback state exceeds its byte limit")
        last = self._last_version(payload)
        state_revision = self._feedback_state_revision(payload)
        pending_rejection = payload.get(PENDING_REJECTION_KEY)
        if pending_rejection is not None:
            if not isinstance(pending_rejection, Mapping):
                raise FeedbackStateError(
                    f"{PENDING_REJECTION_KEY} must be an object"
                )
            pending_event_id = str(pending_rejection.get("event_id") or "")
            try:
                pending_version = int(pending_rejection.get("feedback_version"))
            except (TypeError, ValueError) as exc:
                raise FeedbackStateError(
                    f"{PENDING_REJECTION_KEY} has an invalid version"
                ) from exc
            if (
                pending_event_id == normalized["event_id"]
                and pending_version == version
            ):
                raise PreviouslyRejectedFeedbackError(
                    code=str(
                        pending_rejection.get("error_code")
                        or "PREVIOUSLY_REJECTED"
                    ),
                    reason=str(
                        pending_rejection.get("reason")
                        or "feedback was previously rejected"
                    ),
                )
            if version > pending_version:
                return ApplyResult("gap", last)
        if version < last:
            history = payload.get(REJECTIONS_KEY, [])
            matching = next(
                (
                    item
                    for item in reversed(history)
                    if isinstance(item, Mapping)
                    and item.get("event_id") == normalized["event_id"]
                    and item.get("feedback_version") == version
                ),
                None,
            )
            if matching is not None:
                raise PreviouslyRejectedFeedbackError(
                    code=str(matching.get("error_code") or "PREVIOUSLY_REJECTED"),
                    reason=str(
                        matching.get("reason")
                        or "feedback was previously rejected"
                    ),
                )
            return ApplyResult("duplicate", last)
        if version == last:
            if str(payload.get("last_feedback_event_id") or "") == normalized["event_id"]:
                if payload.get("last_feedback_status") == "rejected":
                    history = payload.get(REJECTIONS_KEY, [])
                    matching = next(
                        (
                            item
                            for item in reversed(history)
                            if isinstance(item, Mapping)
                            and item.get("event_id") == normalized["event_id"]
                            and item.get("feedback_version") == version
                        ),
                        {},
                    )
                    raise PreviouslyRejectedFeedbackError(
                        code=str(matching.get("error_code") or "PREVIOUSLY_REJECTED"),
                        reason=str(
                            matching.get("reason")
                            or "feedback was previously rejected"
                        ),
                    )
                return ApplyResult("duplicate", last)
            raise VersionEventConflict()
        if version != last + 1:
            return ApplyResult("gap", last)

        user_vector, user_vector_name = self._vector(user.vector, self.settings.user_vector_name)
        dimension = self.settings.vector_dimension if self._enforce_dimension else len(user_vector)
        current = self._finite_vector(user_vector, dimension, label="user vector")
        accumulator = self._finite_vector(
            payload.get(LATENT_KEY, current), dimension, label="feedback latent vector"
        ).copy()
        adjustments = self._adjustments(payload)
        applied_signals = self._applied_signals(payload)
        definition = get_interaction(normalized["event_type"])
        repo_state = adjustments.get(repo_id, {})
        tracked_repositories = set(adjustments) | set(applied_signals)
        persists_repository_state = (
            bool(definition.state_family) and not definition.reversal_of
        ) or definition.apply_once
        if (
            repo_id not in tracked_repositories
            and persists_repository_state
            and len(tracked_repositories) >= self.settings.max_tracked_repositories
        ):
            raise TrackedRepositoryLimitError()

        def transition_delta() -> np.ndarray:
            return self._finite_vector(
                vector_delta(
                    accumulator,
                    self._repository_vector(repo_id, dimension),
                    self._alpha(normalized),
                ),
                dimension,
                label="feedback delta",
            )

        if definition.reversal_of:
            family = definition.state_family or ""
            stored = repo_state.get(family)
            if stored and stored.get("action") == definition.reversal_of:
                # Reversals are zero-alpha audit transitions.  Subtracting a
                # historical delta would incorrectly erase later feedback.
                repo_state.pop(family, None)
        elif definition.state_family:
            family = definition.state_family
            stored = repo_state.get(family)
            if not stored or stored.get("action") != normalized["event_type"]:
                if stored:
                    accumulator -= self._finite_vector(
                        stored.get("delta"), dimension, label="stored feedback delta"
                    )
                delta = transition_delta()
                accumulator += delta
                adjustments[repo_id] = repo_state
                repo_state[family] = {
                    "action": normalized["event_type"],
                    "delta": delta.tolist(),
                    "event_id": normalized["event_id"],
                }
        elif definition.apply_once:
            repo_signals = applied_signals.setdefault(repo_id, [])
            if normalized["event_type"] not in repo_signals:
                accumulator += transition_delta()
                repo_signals.append(normalized["event_type"])
        else:
            accumulator += transition_delta()

        if not repo_state:
            adjustments.pop(repo_id, None)
        norm = float(np.linalg.norm(accumulator))
        if not math.isfinite(norm) or norm == 0:
            raise FeedbackStateError("feedback produced an invalid vector")
        vector = (accumulator / norm).tolist()
        payload[LATENT_KEY] = accumulator.tolist()
        payload[ADJUSTMENTS_KEY] = adjustments
        payload[APPLIED_SIGNALS_KEY] = applied_signals
        if self._feedback_state_bytes(payload) > self.settings.max_user_state_bytes:
            raise FeedbackStateLimitError()
        payload["last_feedback_version"] = version
        payload["last_feedback_event_id"] = normalized["event_id"]
        payload["last_feedback_status"] = "applied"
        payload[FEEDBACK_STATE_REVISION_FIELD] = state_revision + 1
        stored_vector: Any = vector if user_vector_name is None else {user_vector_name: vector}
        self._upsert(
            PointStruct(id=user.id, vector=stored_vector, payload=payload),
            expected_payload=self._payload(user),
        )
        return ApplyResult("applied", version, cursor_advanced=True)

    def reject(self, event: Mapping[str, Any], *, code: str, reason: str) -> ApplyResult:
        """Record and skip one terminal event only when it is exactly next.

        This prevents a malformed version N from permanently blocking N+1,
        while retaining an operator-visible bounded rejection record.  Gaps are
        never skipped because advancing over an unseen version is unsafe.
        """

        user_id = self._canonical_uuid(event.get("user_id"), "user_id")
        raw_version = event.get("feedback_version")
        if isinstance(raw_version, bool):
            raise PermanentFeedbackError("feedback_version must be a positive integer")
        try:
            version = int(raw_version)
        except (TypeError, ValueError) as exc:
            raise PermanentFeedbackError("feedback_version must be a positive integer") from exc
        if version < 1:
            raise PermanentFeedbackError("feedback_version must be a positive integer")
        user = self._user(user_id)
        payload = self._payload(user)
        last = self._last_version(payload)
        if version <= last:
            return ApplyResult("rejected_duplicate", last)
        if version != last + 1:
            return ApplyResult("rejected_gap", last)
        state_revision = self._feedback_state_revision(payload)

        rejections = payload.get(REJECTIONS_KEY, [])
        if not isinstance(rejections, list) or any(not isinstance(item, Mapping) for item in rejections):
            raise FeedbackStateError(f"{REJECTIONS_KEY} must be a list of objects")
        history = [dict(item) for item in rejections]
        history.append(
            {
                "event_id": _safe_text(event.get("event_id"), 64),
                "repo_id": _safe_text(event.get("repo_id"), 64),
                "feedback_version": version,
                "error_code": _safe_text(code, 64),
                "reason": _safe_text(reason),
            }
        )
        payload[REJECTIONS_KEY] = history[-self.settings.rejection_history_size :]
        payload[PENDING_REJECTION_KEY] = {
            "event_id": _safe_text(event.get("event_id"), 64),
            "feedback_version": version,
            "error_code": _safe_text(code, 64),
            "reason": _safe_text(reason),
        }
        payload["last_feedback_version"] = version
        payload["last_feedback_event_id"] = _safe_text(event.get("event_id"), 64)
        payload["last_feedback_status"] = "rejected"
        payload[FEEDBACK_STATE_REVISION_FIELD] = state_revision + 1
        # Preserve the exact stored vector; no malformed signal is applied.
        self._upsert(
            PointStruct(id=user.id, vector=user.vector, payload=payload),
            expected_payload=self._payload(user),
        )
        return ApplyResult("rejected_advanced", version, cursor_advanced=True)

    def finalize_rejection(self, event: Mapping[str, Any]) -> None:
        """Clear the rejection fence only after its DLQ copy is durable."""

        user_id = self._canonical_uuid(event.get("user_id"), "user_id")
        # A terminally rejected event can have a malformed event_id.  The
        # fence stores the bounded raw identity precisely so that such an
        # event can still be finalized after its DLQ copy is durable.
        event_id = _safe_text(event.get("event_id"), 64)
        try:
            version = int(event.get("feedback_version"))
        except (TypeError, ValueError) as exc:
            raise PermanentFeedbackError(
                "feedback_version must be a positive integer"
            ) from exc
        user = self._user(user_id)
        payload = self._payload(user)
        pending = payload.get(PENDING_REJECTION_KEY)
        if pending is None:
            return
        if not isinstance(pending, Mapping):
            raise FeedbackStateError(f"{PENDING_REJECTION_KEY} must be an object")
        try:
            pending_version = int(pending.get("feedback_version"))
        except (TypeError, ValueError) as exc:
            raise FeedbackStateError(
                f"{PENDING_REJECTION_KEY} has an invalid version"
            ) from exc
        if (
            str(pending.get("event_id") or "") != event_id
            or pending_version != version
        ):
            raise FeedbackWriteConflict()
        state_revision = self._feedback_state_revision(payload)
        payload.pop(PENDING_REJECTION_KEY, None)
        payload[FEEDBACK_STATE_REVISION_FIELD] = state_revision + 1
        self._upsert(
            PointStruct(id=user.id, vector=user.vector, payload=payload),
            expected_payload=self._payload(user),
        )

    def close(self) -> None:
        if self._owns_qdrant and hasattr(self.qdrant, "close"):
            self.qdrant.close()


class OrderedFeedbackConsumer:
    def __init__(
        self,
        redis_client: Any | None = None,
        applier: OrderedFeedbackApplier | None = None,
        settings: V2FeedbackSettings | None = None,
    ) -> None:
        self._settings_explicit = settings is not None
        self.settings = settings or V2FeedbackSettings.from_env()
        self._owns_redis = redis_client is None
        self._owns_applier = applier is None
        self.redis = redis_client or _redis_client(self.settings)
        self.applier = applier or OrderedFeedbackApplier(settings=self.settings)
        _require_xautoclaim_redis(self.redis)
        self.consumer = (
            f"{self.settings.consumer_name_prefix}-{socket.gethostname()}-"
            f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        self.release_id = _runtime_release_id(production=self.settings.production)
        self.heartbeat_value = f"{self.release_id}|{self.consumer}"
        self._reclaim_cursor = "0-0"
        self._scheduler: UserFeedbackScheduler | None = None
        self._poll_lock = threading.Lock()
        self._closed = False
        self._admission_stop = threading.Event()
        try:
            self.redis.xgroup_create(
                self.settings.stream_name,
                self.settings.consumer_group,
                id="0",
                mkstream=True,
            )
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    @property
    def _heartbeat_ttl_seconds(self) -> float:
        # Preserve the historical monkeypatch hook used by existing tests.
        if not self._settings_explicit and os.getenv("FEEDBACK_CONSUMER_HEARTBEAT_TTL_SECONDS") is None:
            return CONSUMER_HEARTBEAT_TTL_SECONDS
        return float(self.settings.heartbeat_ttl_seconds)

    def _read_capacity(self) -> int:
        return self.settings.read_batch_size if self._scheduler is None else min(
            self.settings.read_batch_size, self._scheduler.capacity - self._scheduler.pending)

    def _messages(self):
        count = self._read_capacity()
        if count <= 0:
            return
        try:
            claimed = self.redis.xautoclaim(
                self.settings.stream_name,
                self.settings.consumer_group,
                self.consumer,
                self.settings.reclaim_idle_ms,
                self._reclaim_cursor,
                count=count,
            )
            if claimed:
                raw_cursor = claimed[0]
                next_cursor = (
                    raw_cursor.decode("utf-8")
                    if isinstance(raw_cursor, bytes)
                    else str(raw_cursor)
                )
                self._reclaim_cursor = "0-0" if next_cursor == "0-0" else next_cursor
            messages = claimed[1] if claimed and len(claimed) > 1 else []
            yield from self._repository_messages(messages)
        except Exception:
            logger.error(
                "pending feedback reclaim failed",
                extra={"feedback_context": {"status": "reclaim_failed"}},
            )
        count = self._read_capacity()
        if count <= 0:
            return
        response = self.redis.xreadgroup(
            self.settings.consumer_group,
            self.consumer,
            {self.settings.stream_name: ">"},
            count=count,
            block=self.settings.read_block_ms,
        )
        for _, messages in response or []:
            yield from self._repository_messages(messages)

    def _repository_messages(self, messages):
        # Scope snapshots to each actual reclaim/new-read response. Do not
        # buffer pending work while waiting for the following blocking read.
        context = (
            self.applier.repository_read_batch(payload for _, payload in messages)
            if isinstance(self.applier, OrderedFeedbackApplier)
            else nullcontext()
        )
        with context:
            for message_id, payload in messages:
                yield str(message_id), payload

    def _refresh_heartbeat(self) -> None:
        ttl = self._heartbeat_ttl_seconds
        kwargs = {"ex": int(ttl)} if ttl >= 1 else {"px": max(1, int(ttl * 1_000))}
        self.redis.set(self.settings.heartbeat_key, self.heartbeat_value, **kwargs)

    def _heartbeat_loop(self, stop: threading.Event) -> None:
        interval = max(0.05, self._heartbeat_ttl_seconds / 3)
        while not stop.wait(interval):
            try:
                self._refresh_heartbeat()
            except Exception:
                logger.error(
                    "feedback consumer heartbeat refresh failed",
                    extra={"feedback_context": {"status": "heartbeat_failed"}},
                )

    @staticmethod
    def _context(
        message_id: str,
        payload: Mapping[str, Any],
        *,
        status: str,
        code: str,
        attempt: int = 0,
    ) -> dict[str, Any]:
        return {
            "message_id": _safe_text(message_id, 64),
            "event_id": _safe_text(payload.get("event_id"), 64),
            "user_id": _safe_text(payload.get("user_id"), 64),
            "repo_id": _safe_text(payload.get("repo_id"), 64),
            "feedback_version": _safe_text(payload.get("feedback_version"), 32),
            "attempt": attempt,
            "status": status,
            "code": code,
        }

    def _attempt_key(self, payload: Mapping[str, Any], message_id: str) -> str:
        identity = _safe_text(payload.get("event_id"), 64) or _safe_text(message_id, 64)
        return f"{self.settings.stream_name}:attempts:{identity}"

    def _clear_attempts(self, payload: Mapping[str, Any], message_id: str) -> None:
        self.redis.delete(self._attempt_key(payload, message_id))

    def _ack(self, message_id: str) -> None:
        self.redis.eval(
            ACK_DELETE_LUA,
            1,
            self.settings.stream_name,
            self.settings.consumer_group,
            message_id,
        )

    def _dead_letter(
        self,
        message_id: str,
        payload: Mapping[str, Any],
        *,
        code: str,
        reason: str,
        retryable: bool,
        attempts: int,
        terminal_status: str,
        cursor_advanced: bool,
    ) -> None:
        dead = {
            str(key): _redis_field(value)
            for key, value in payload.items()
            if str(key) not in _DLQ_METADATA_FIELDS
        }
        dead.update(
            {
                "source_stream": self.settings.stream_name,
                "source_message_id": str(message_id),
                "failure_code": _safe_text(code, 64),
                "failure_reason": _safe_text(reason, 500),
                "retryable": "1" if retryable else "0",
                "attempts": str(attempts),
                "terminal_status": _safe_text(terminal_status, 64),
                "cursor_advanced": "1" if cursor_advanced else "0",
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        # Copying is idempotent by source message ID. Source ACK happens only
        # after any Qdrant rejection fence is finalized, so every crash window
        # leaves recoverable source work or a durable DLQ copy.
        fields: list[str] = []
        for key, value in dead.items():
            fields.extend([key, value])
        result = self.redis.eval(
            DLQ_MOVE_LUA,
            2,
            self.settings.dead_letter_stream,
            f"{self.settings.dead_letter_stream}:source:{message_id}",
            str(self.settings.dead_letter_maxlen),
            str(self.settings.idempotency_ttl_seconds),
            *fields,
        )
        if result == "overloaded" or result == b"overloaded":
            raise DeadLetterFullError()

    def _dead_letter_or_defer(
        self,
        message_id: str,
        payload: Mapping[str, Any],
        **metadata: Any,
    ) -> bool:
        try:
            self._dead_letter(message_id, payload, **metadata)
            return True
        except DeadLetterFullError as exc:
            logger.error(
                "feedback dead-letter capacity is exhausted; source remains pending",
                extra={
                    "feedback_context": self._context(
                        message_id,
                        payload,
                        status="dlq_full",
                        code=exc.code,
                        attempt=int(metadata.get("attempts") or 0),
                    )
                },
            )
            return False

    def _retry(
        self,
        message_id: str,
        payload: Mapping[str, Any],
        *,
        code: str,
        reason: str,
    ) -> bool:
        if code == "VERSION_GAP":
            # Reordering from live/reclaim/capacity deferral is not a failed
            # dependency attempt. Preserve the event until its predecessor is
            # applied or explicitly reconciled; never move healthy successors
            # to DLQ solely because scheduling changed their arrival order.
            logger.info("feedback ordering gap remains pending",
                extra={"feedback_context": self._context(message_id, payload,
                    status="retry_pending", code=code)})
            return False
        attempts_key = self._attempt_key(payload, message_id)
        attempts = int(self.redis.incr(attempts_key))
        self.redis.expire(attempts_key, self.settings.idempotency_ttl_seconds)
        if attempts < self.settings.max_delivery_attempts:
            logger.warning(
                "feedback processing deferred",
                extra={
                    "feedback_context": self._context(
                        message_id,
                        payload,
                        status="retry_pending",
                        code=code,
                        attempt=attempts,
                    )
                },
            )
            return False

        if not self._dead_letter_or_defer(
            message_id,
            payload,
            code="RETRY_EXHAUSTED",
            reason=reason,
            retryable=True,
            attempts=attempts,
            terminal_status=code,
            cursor_advanced=False,
        ):
            return False
        self._ack(message_id)
        self._clear_attempts(payload, message_id)
        logger.error(
            "feedback retry limit exhausted",
            extra={
                "feedback_context": self._context(
                    message_id,
                    payload,
                    status="dead_lettered",
                    code="RETRY_EXHAUSTED",
                    attempt=attempts,
                )
            },
        )
        return True

    def _terminal_rejection(
        self,
        message_id: str,
        payload: Mapping[str, Any],
        error: FeedbackProcessingError,
    ) -> bool:
        cursor_advanced = False
        terminal_status = "rejected_unreconciled"
        try:
            result = self.applier.reject(
                payload,
                code=error.code,
                reason=error.public_message,
            )
            if result.status == "rejected_gap":
                return self._retry(message_id, payload, code="VERSION_GAP",
                    reason="an earlier feedback version has not been applied")
            cursor_advanced = result.cursor_advanced or result.status == "rejected_duplicate"
            terminal_status = result.status
        except MissingVectorError as exc:
            return self._retry(
                message_id,
                payload,
                code=exc.code,
                reason=exc.public_message,
            )
        except FeedbackDependencyError as exc:
            return self._retry(
                message_id,
                payload,
                code=exc.code,
                reason=exc.public_message,
            )
        except (PermanentFeedbackError, FeedbackStateError):
            # Invalid identity/version or corrupt stored state means advancing
            # the ordered cursor cannot be proven safe.  Preserve in DLQ.
            terminal_status = "rejected_unreconciled"

        if not self._dead_letter_or_defer(
            message_id,
            payload,
            code=error.code,
            reason=error.public_message,
            retryable=False,
            attempts=1,
            terminal_status=terminal_status,
            cursor_advanced=cursor_advanced,
        ):
            return False
        if cursor_advanced:
            try:
                self.applier.finalize_rejection(payload)
            except FeedbackProcessingError as exc:
                logger.error(
                    "feedback rejection fence finalization deferred",
                    extra={
                        "feedback_context": self._context(
                            message_id,
                            payload,
                            status="rejection_finalize_pending",
                            code=exc.code,
                            attempt=1,
                        )
                    },
                )
                return False
        self._ack(message_id)
        self._clear_attempts(payload, message_id)
        logger.error(
            "feedback event rejected",
            extra={
                "feedback_context": self._context(
                    message_id,
                    payload,
                    status=terminal_status,
                    code=error.code,
                    attempt=1,
                )
            },
        )
        return True

    def _process_message(self, message_id: str, payload: Mapping[str, Any]) -> bool:
        finalized = False
        try:
            user_id = str(uuid.UUID(str(payload.get("user_id"))))
        except (TypeError, ValueError, AttributeError):
            error = PermanentFeedbackError("user_id must be a UUID")
            return self._terminal_rejection(message_id, payload, error)

        try:
            with user_vector_lock(
                self.redis,
                user_id,
                settings=self.settings,
            ) as lock:
                lock.assert_owned()
                try:
                    result = self.applier.apply(payload)
                except PermanentFeedbackError as exc:
                    finalized = self._terminal_rejection(message_id, payload, exc)
                    return finalized
                except FeedbackStateError as exc:
                    # Stored state corruption is terminal for this delivery but
                    # must not be "reconciled" by overwriting the cursor.
                    if not self._dead_letter_or_defer(
                        message_id,
                        payload,
                        code=exc.code,
                        reason=exc.public_message,
                        retryable=False,
                        attempts=1,
                        terminal_status="state_invalid",
                        cursor_advanced=False,
                    ):
                        return False
                    self._ack(message_id)
                    self._clear_attempts(payload, message_id)
                    finalized = True
                    logger.error(
                        "stored feedback state rejected the event",
                        extra={
                            "feedback_context": self._context(
                                message_id,
                                payload,
                                status="state_invalid",
                                code=exc.code,
                                attempt=1,
                            )
                        },
                    )
                    return True
                except MissingVectorError as exc:
                    finalized = self._retry(
                        message_id, payload, code=exc.code, reason=exc.public_message
                    )
                    return finalized
                except FeedbackDependencyError as exc:
                    finalized = self._retry(
                        message_id, payload, code=exc.code, reason=exc.public_message
                    )
                    return finalized
                except (KeyError, ValueError) as exc:
                    # Backward-compatible classification for custom appliers.
                    error = PermanentFeedbackError(str(exc))
                    finalized = self._terminal_rejection(message_id, payload, error)
                    return finalized
                except LookupError as exc:
                    finalized = self._retry(
                        message_id,
                        payload,
                        code="VECTOR_MISSING",
                        reason="a required vector is not indexed yet",
                    )
                    return finalized
                except Exception:
                    logger.error(
                        "unexpected feedback processing failure",
                        extra={
                            "feedback_context": self._context(
                                message_id,
                                payload,
                                status="retry_pending",
                                code="INTERNAL_PROCESSING_ERROR",
                            )
                        },
                    )
                    finalized = self._retry(
                        message_id,
                        payload,
                        code="INTERNAL_PROCESSING_ERROR",
                        reason="feedback processing failed unexpectedly",
                    )
                    return finalized

                lock.assert_owned()
                if result.status == "gap":
                    finalized = self._retry(
                        message_id,
                        payload,
                        code="VERSION_GAP",
                        reason="an earlier feedback version has not been applied",
                    )
                    return finalized
                if result.status not in {"applied", "duplicate"}:
                    finalized = self._retry(
                        message_id,
                        payload,
                        code="UNKNOWN_APPLY_STATUS",
                        reason="feedback applier returned an unknown status",
                    )
                    return finalized
                self._ack(message_id)
                self._clear_attempts(payload, message_id)
                finalized = True
                logger.info(
                    "feedback event processed",
                    extra={
                        "feedback_context": self._context(
                            message_id,
                            payload,
                            status=result.status,
                            code="OK",
                        )
                    },
                )
                return True
        except LockAcquisitionError:
            logger.info(
                "feedback user lock is busy",
                extra={
                    "feedback_context": self._context(
                        message_id, payload, status="lock_busy", code="LOCK_BUSY"
                    )
                },
            )
            return False
        except LockLostError:
            if finalized:
                return True
            # The write may have completed.  Ordered version checking makes a
            # redelivery a duplicate instead of applying it twice.
            return self._retry(
                message_id,
                payload,
                code="LOCK_LOST",
                reason="user-vector lock ownership was lost",
            )

    def _scheduled_message(self, work: Work) -> bool:
        if self._admission_stop.is_set():
            return False
        if isinstance(self.applier, OrderedFeedbackApplier):
            # Read snapshots are immutable and scoped to their Redis response;
            # install them on the actual worker thread, never user-ledger state.
            previous = getattr(self.applier._repository_batches, "points", None)
            self.applier._repository_batches.points = work.snapshot
            try:
                return self._process_message(work.message_id, work.payload)
            finally:
                self.applier._repository_batches.points = previous
        return self._process_message(work.message_id, work.payload)

    def run_once(self, *, wait: bool = True) -> int:
        # Default drains for CLI/tests. The production loop keeps bounded lanes
        # alive across live/reclaim polls, so one slow user does not hold a batch.
        with self._poll_lock:
            if self._closed or self._admission_stop.is_set():
                return 0
            self._refresh_heartbeat()
            heartbeat_stop = threading.Event()
            heartbeat_thread = threading.Thread(target=self._heartbeat_loop,
                args=(heartbeat_stop,), name="feedback-v2-heartbeat", daemon=True)
            heartbeat_thread.start()
            if self._scheduler is None:
                self._scheduler = UserFeedbackScheduler(self._scheduled_message,
                    workers=self.settings.consumer_workers,
                    capacity=max(self.settings.consumer_workers, self.settings.read_batch_size))
            scheduler = self._scheduler
            try:
                if scheduler.wait_capacity():
                    for message_id, payload in self._messages():
                        if self._admission_stop.is_set():
                            break
                        try:
                            user_id = str(uuid.UUID(str(payload.get("user_id"))))
                        except (TypeError, ValueError, AttributeError):
                            user_id = f"invalid:{message_id}"
                        snapshot = (getattr(self.applier._repository_batches, "points", None)
                            if isinstance(self.applier, OrderedFeedbackApplier) else None)
                        # Excess deliveries stay unacknowledged in Redis; no
                        # unbounded executor or local backlog is created.
                        work = Work(message_id, user_id, payload, snapshot)
                        if wait:
                            scheduler.wait_user(user_id)
                        scheduler.submit(work)
                if wait:
                    scheduler.drain()
                return scheduler.completed()
            finally:
                if wait:
                    scheduler.close()
                    self._scheduler = None
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=1)
                self._refresh_heartbeat()

    def stop_admission(self) -> None:
        self._admission_stop.set()

    def close(self) -> None:
        self.stop_admission()
        # Serialize close against polling. Keep heartbeats and dependency clients
        # alive while already executing synchronous vector writes drain.
        with self._poll_lock:
            self._closed = True
            if self._scheduler is not None:
                stop = threading.Event()
                heartbeat = threading.Thread(target=self._heartbeat_loop, args=(stop,), daemon=True)
                heartbeat.start()
                try:
                    self._scheduler.close()
                finally:
                    stop.set()
                    heartbeat.join(timeout=1)
                    self._scheduler = None
        try:
            self.redis.eval(
                HEARTBEAT_DELETE_LUA,
                1,
                self.settings.heartbeat_key,
                self.heartbeat_value,
            )
        except Exception:
            logger.error(
                "feedback consumer heartbeat cleanup failed",
                extra={"feedback_context": {"status": "heartbeat_cleanup_failed"}},
            )
        if self._owns_applier and hasattr(self.applier, "close"):
            self.applier.close()
        if self._owns_redis and hasattr(self.redis, "close"):
            self.redis.close()


def main() -> None:
    configure_feedback_worker_logging(os.getenv("LOG_LEVEL", "INFO"))
    consumer = OrderedFeedbackConsumer()
    stop = threading.Event()

    def request_shutdown(signum, frame) -> None:
        logger.info(
            "feedback consumer shutdown requested",
            extra={"feedback_context": {"status": "shutdown_requested", "signal": signum}},
        )
        consumer.stop_admission()
        stop.set()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    try:
        while not stop.is_set():
            consumer.run_once(wait=False)
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
