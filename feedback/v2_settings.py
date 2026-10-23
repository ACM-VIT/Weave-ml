"""Validated settings for the durable v2 feedback boundary.

The v2 API and worker intentionally share this object.  Keeping configuration
parsing here prevents a producer and consumer from silently using different
streams, groups, lock policies, or vector contracts.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from urllib.parse import urlsplit


_PRODUCTION_ENVIRONMENTS = {"prod", "production"}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._-]{0,127}$")


def _first_env(*names: str) -> tuple[str | None, str]:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value, name
    return None, names[0]


def _text(*names: str, default: str | None = None, required: bool = False) -> str | None:
    raw, selected = _first_env(*names)
    value = default if raw is None else raw.strip()
    if not value:
        if required:
            raise ValueError(f"{selected} is required")
        return None
    return value


def _name(*names: str, default: str) -> str:
    value = _text(*names, default=default, required=True)
    assert value is not None
    if not _SAFE_NAME.fullmatch(value):
        raise ValueError(
            f"{names[0]} must contain only letters, digits, colon, dot, underscore, or dash"
        )
    return value


def _integer(
    *names: str,
    default: int,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    raw, selected = _first_env(*names)
    value_text = str(default) if raw is None else raw.strip()
    try:
        value = int(value_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{selected} must be an integer") from exc
    if value < minimum or (maximum is not None and value > maximum):
        upper = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{selected} must be >= {minimum}{upper}")
    return value


def _number(
    *names: str,
    default: float,
    minimum: float = 0.0,
    maximum: float | None = None,
    strictly_greater: bool = False,
) -> float:
    raw, selected = _first_env(*names)
    value_text = str(default) if raw is None else raw.strip()
    try:
        value = float(value_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{selected} must be a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{selected} must be finite")
    below = value <= minimum if strictly_greater else value < minimum
    if below or (maximum is not None and value > maximum):
        comparator = ">" if strictly_greater else ">="
        upper = f" and <= {maximum}" if maximum is not None else ""
        raise ValueError(f"{selected} must be {comparator} {minimum}{upper}")
    return value


def _url(
    *names: str,
    default: str | None,
    schemes: set[str],
    required: bool,
) -> str | None:
    value = _text(*names, default=default, required=required)
    if value is None:
        return None
    # Do not include the URL in validation errors: it may contain credentials.
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{names[0]} must be a valid URL with a valid port") from exc
    if parsed.scheme.lower() not in schemes:
        allowed = ", ".join(sorted(schemes))
        raise ValueError(f"{names[0]} must use one of these URL schemes: {allowed}")
    if parsed.scheme.lower() != "unix" and not parsed.hostname:
        raise ValueError(f"{names[0]} must include a host")
    if parsed.fragment:
        raise ValueError(f"{names[0]} must not include a URL fragment")
    if port is not None and not 1 <= port <= 65_535:
        raise ValueError(f"{names[0]} must include a valid port")
    return value


@dataclass(frozen=True, slots=True)
class V2FeedbackSettings:
    environment: str
    redis_url: str | None
    stream_name: str
    stream_maxlen: int
    consumer_group: str
    consumer_name_prefix: str
    heartbeat_key: str
    heartbeat_ttl_seconds: int
    idempotency_ttl_seconds: int
    read_batch_size: int
    read_block_ms: int
    reclaim_idle_ms: int
    user_lock_prefix: str
    user_lock_ttl_seconds: float
    user_lock_wait_seconds: float
    user_lock_renew_interval_seconds: float
    max_delivery_attempts: int
    dead_letter_stream: str
    dead_letter_maxlen: int
    rejection_history_size: int
    max_tracked_repositories: int
    max_user_state_bytes: int
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_timeout_seconds: float
    repository_collection: str
    repository_vector_name: str
    user_collection: str
    user_vector_name: str | None
    vector_dimension: int
    dwell_min_ms: int
    dwell_full_credit_ms: int
    dwell_max_alpha: float
    health_warn_pending: int
    health_max_pending: int
    health_warn_lag: int
    health_max_lag: int
    health_warn_stream_length: int
    health_max_stream_length: int
    health_warn_dead_letter: int
    health_max_dead_letter: int

    consumer_workers: int = 4

    @property
    def production(self) -> bool:
        return self.environment in _PRODUCTION_ENVIRONMENTS

    @property
    def user_lock_ttl_ms(self) -> int:
        return int(self.user_lock_ttl_seconds * 1_000)

    def _validate_cross_fields(self) -> None:
        if self.reclaim_idle_ms <= self.read_block_ms:
            raise ValueError("FEEDBACK_RECLAIM_IDLE_MS must exceed FEEDBACK_READ_BLOCK_MS")
        if self.heartbeat_ttl_seconds * 1_000 <= self.read_block_ms:
            raise ValueError(
                "FEEDBACK_CONSUMER_HEARTBEAT_TTL_SECONDS must exceed "
                "FEEDBACK_READ_BLOCK_MS"
            )
        if self.user_lock_ttl_seconds <= self.user_lock_wait_seconds:
            raise ValueError(
                "FEEDBACK_USER_LOCK_TTL_SECONDS must exceed FEEDBACK_USER_LOCK_WAIT_SECONDS"
            )
        if self.read_batch_size > self.stream_maxlen:
            raise ValueError("FEEDBACK_READ_BATCH_SIZE must not exceed FEEDBACK_STREAM_MAXLEN")
        if self.health_max_stream_length < self.stream_maxlen:
            raise ValueError(
                "FEEDBACK_HEALTH_MAX_STREAM_LENGTH must be at least FEEDBACK_STREAM_MAXLEN"
            )
        if self.health_max_dead_letter > self.dead_letter_maxlen:
            raise ValueError(
                "FEEDBACK_HEALTH_MAX_DEAD_LETTER must not exceed "
                "FEEDBACK_DEAD_LETTER_MAXLEN"
            )

    @classmethod
    def from_env(cls) -> "V2FeedbackSettings":
        environment = (os.getenv("APP_ENV", "development").strip().lower() or "development")
        if environment not in {"development", "dev", "test", "testing", "staging", "stage", "prod", "production"}:
            raise ValueError("APP_ENV must be development, test, staging, or production")
        production = environment in _PRODUCTION_ENVIRONMENTS
        redis_url = _url(
            "REDIS_URL",
            default=None,
            schemes={"redis", "rediss", "unix"},
            required=production,
        )
        if production:
            redis_auth_mode = _text("REDIS_AUTH_MODE", required=True)
            if redis_auth_mode != "acl_url":
                raise ValueError("REDIS_AUTH_MODE must be acl_url in production")
            assert redis_url is not None
            parsed_redis = urlsplit(redis_url)
            if parsed_redis.scheme.casefold() not in {"redis", "rediss"}:
                raise ValueError("REDIS_URL must use redis or rediss in production")
            if not parsed_redis.username or not parsed_redis.password:
                raise ValueError(
                    "REDIS_URL must include Redis ACL username and password in production"
                )
            if len(parsed_redis.password) < 16:
                raise ValueError("REDIS_URL password must contain at least 16 characters")
            if parsed_redis.query:
                raise ValueError(
                    "REDIS_URL must not include query options in production"
                )
        stream_name = _name(
            "FEEDBACK_STREAM_NAME", "V2_FEEDBACK_STREAM_NAME", default="ml:feedback:v2"
        )
        stream_maxlen = _integer(
            "FEEDBACK_STREAM_MAXLEN", "V2_FEEDBACK_STREAM_MAXLEN",
            default=100_000, minimum=100, maximum=100_000_000,
        )
        dead_letter_stream = _name(
            "FEEDBACK_DEAD_LETTER_STREAM", "V2_FEEDBACK_DEAD_LETTER_STREAM",
            default=f"{stream_name}:dead",
        )
        if dead_letter_stream == stream_name:
            raise ValueError("FEEDBACK_DEAD_LETTER_STREAM must differ from FEEDBACK_STREAM_NAME")
        dead_letter_maxlen = _integer(
            "FEEDBACK_DEAD_LETTER_MAXLEN", "V2_FEEDBACK_DEAD_LETTER_MAXLEN",
            default=10_000, minimum=10, maximum=10_000_000,
        )
        heartbeat_key = _name(
            "FEEDBACK_CONSUMER_HEARTBEAT_KEY",
            default=f"{stream_name}:consumer-heartbeat",
        )
        if heartbeat_key in {stream_name, dead_letter_stream}:
            raise ValueError(
                "FEEDBACK_CONSUMER_HEARTBEAT_KEY must differ from both feedback streams"
            )
        reclaim_idle_ms = _integer(
            "FEEDBACK_RECLAIM_IDLE_MS",
            default=30_000,
            minimum=1_000,
            maximum=86_400_000,
        )
        idempotency_ttl_seconds = _integer(
            "FEEDBACK_IDEMPOTENCY_TTL_SECONDS",
            default=7 * 24 * 60 * 60,
            minimum=3_600,
            maximum=365 * 24 * 60 * 60,
        )
        if idempotency_ttl_seconds * 1_000 < reclaim_idle_ms * 2:
            raise ValueError(
                "FEEDBACK_IDEMPOTENCY_TTL_SECONDS must be at least twice "
                "FEEDBACK_RECLAIM_IDLE_MS"
            )

        lock_ttl = _number(
            "FEEDBACK_USER_LOCK_TTL_SECONDS", default=30.0,
            minimum=5.0, maximum=3_600.0,
        )
        lock_renew = _number(
            "FEEDBACK_USER_LOCK_RENEW_INTERVAL_SECONDS",
            default=lock_ttl / 3.0,
            minimum=0.05,
            maximum=1_200.0,
        )
        if lock_renew >= lock_ttl / 2.0:
            raise ValueError(
                "FEEDBACK_USER_LOCK_RENEW_INTERVAL_SECONDS must be less than half "
                "FEEDBACK_USER_LOCK_TTL_SECONDS"
            )

        dwell_min_seconds = _number(
            "FEEDBACK_DWELL_MIN_SECONDS", default=3.0,
            minimum=0.0, maximum=300.0,
        )
        dwell_full_seconds = _number(
            "FEEDBACK_DWELL_FULL_CREDIT_SECONDS", default=300.0,
            minimum=0.001, maximum=300.0,
        )
        if dwell_full_seconds <= dwell_min_seconds:
            raise ValueError(
                "FEEDBACK_DWELL_FULL_CREDIT_SECONDS must exceed FEEDBACK_DWELL_MIN_SECONDS"
            )

        warn_pending = _integer("FEEDBACK_HEALTH_WARN_PENDING", default=1_000, minimum=0)
        max_pending = _integer("FEEDBACK_HEALTH_MAX_PENDING", default=10_000, minimum=1)
        warn_lag = _integer("FEEDBACK_HEALTH_WARN_LAG", default=10_000, minimum=0)
        max_lag = _integer("FEEDBACK_HEALTH_MAX_LAG", default=50_000, minimum=1)
        warn_stream = _integer(
            "FEEDBACK_HEALTH_WARN_STREAM_LENGTH", default=max(1, int(stream_maxlen * 0.8)),
            minimum=0,
        )
        max_stream = _integer(
            "FEEDBACK_HEALTH_MAX_STREAM_LENGTH", default=stream_maxlen, minimum=1,
        )
        warn_dead = _integer("FEEDBACK_HEALTH_WARN_DEAD_LETTER", default=1, minimum=0)
        max_dead = _integer("FEEDBACK_HEALTH_MAX_DEAD_LETTER", default=1_000, minimum=1)
        for warning, maximum, label in (
            (warn_pending, max_pending, "pending"),
            (warn_lag, max_lag, "lag"),
            (warn_stream, max_stream, "stream length"),
            (warn_dead, max_dead, "dead-letter length"),
        ):
            if warning > maximum:
                raise ValueError(f"feedback health warning threshold for {label} exceeds maximum")

        qdrant_url = _url(
            "QDRANT_URL", default="http://localhost:6333",
            schemes={"http", "https"}, required=True,
        )
        assert qdrant_url is not None
        user_vector_name = _text(
            "USER_PROFILE_VECTOR_NAME",
            "USER_PROFILES_VECTOR_NAME",
            "TARGET_VECTOR_NAME",
            default=None,
        )
        if user_vector_name is not None and not _SAFE_NAME.fullmatch(user_vector_name):
            raise ValueError("USER_PROFILE_VECTOR_NAME contains invalid characters")

        runtime = cls(
            environment=environment,
            consumer_workers=_integer("FEEDBACK_CONSUMER_WORKERS", default=4, minimum=1, maximum=16),
            redis_url=redis_url,
            stream_name=stream_name,
            stream_maxlen=stream_maxlen,
            consumer_group=_name(
                "FEEDBACK_CONSUMER_GROUP", "V2_FEEDBACK_CONSUMER_GROUP",
                default="ml-feedback-v2",
            ),
            consumer_name_prefix=_name(
                "FEEDBACK_CONSUMER_PREFIX", "V2_FEEDBACK_CONSUMER_PREFIX",
                default="ml-feedback-v2",
            ),
            heartbeat_key=heartbeat_key,
            heartbeat_ttl_seconds=_integer(
                "FEEDBACK_CONSUMER_HEARTBEAT_TTL_SECONDS", default=15,
                minimum=5, maximum=3_600,
            ),
            idempotency_ttl_seconds=idempotency_ttl_seconds,
            read_batch_size=_integer(
                "FEEDBACK_READ_BATCH_SIZE", default=20, minimum=1, maximum=1_000,
            ),
            read_block_ms=_integer(
                "FEEDBACK_READ_BLOCK_MS", default=1_000, minimum=10, maximum=60_000,
            ),
            reclaim_idle_ms=reclaim_idle_ms,
            user_lock_prefix=_name(
                "FEEDBACK_USER_LOCK_PREFIX", default="ml:user-vector-lock"
            ),
            user_lock_ttl_seconds=lock_ttl,
            user_lock_wait_seconds=_number(
                "FEEDBACK_USER_LOCK_WAIT_SECONDS", default=2.0,
                minimum=0.1, maximum=300.0,
            ),
            user_lock_renew_interval_seconds=lock_renew,
            max_delivery_attempts=_integer(
                "FEEDBACK_MAX_DELIVERY_ATTEMPTS", default=5, minimum=1, maximum=100,
            ),
            dead_letter_stream=dead_letter_stream,
            dead_letter_maxlen=dead_letter_maxlen,
            rejection_history_size=_integer(
                "FEEDBACK_REJECTION_HISTORY_SIZE", default=64, minimum=1, maximum=1_000,
            ),
            max_tracked_repositories=_integer(
                "FEEDBACK_MAX_TRACKED_REPOSITORIES",
                default=256,
                minimum=1,
                maximum=512,
            ),
            max_user_state_bytes=_integer(
                "FEEDBACK_MAX_USER_STATE_BYTES",
                default=1_000_000,
                minimum=65_536,
                maximum=2_000_000,
            ),
            qdrant_url=qdrant_url,
            qdrant_api_key=_text("QDRANT_API_KEY", default=None),
            qdrant_timeout_seconds=_number(
                "FEEDBACK_QDRANT_TIMEOUT_SECONDS", default=10.0,
                minimum=0.1, maximum=120.0,
            ),
            repository_collection=_name(
                "QDRANT_COLLECTION_NAME",
                default="osiris_research_corpus_v2_20260722_r1",
            ),
            repository_vector_name=_name("QDRANT_VECTOR_NAME", default="repo_embedding"),
            user_collection=_name("USER_PROFILES_COLLECTION", default="user_profiles"),
            user_vector_name=user_vector_name,
            vector_dimension=_integer("VECTOR_DIMENSION", default=384, minimum=2, maximum=65_536),
            dwell_min_ms=int(dwell_min_seconds * 1_000),
            dwell_full_credit_ms=int(dwell_full_seconds * 1_000),
            dwell_max_alpha=_number(
                "FEEDBACK_DWELL_MAX_ALPHA", default=0.15,
                minimum=0.0, maximum=0.15, strictly_greater=True,
            ),
            health_warn_pending=warn_pending,
            health_max_pending=max_pending,
            health_warn_lag=warn_lag,
            health_max_lag=max_lag,
            health_warn_stream_length=warn_stream,
            health_max_stream_length=max_stream,
            health_warn_dead_letter=warn_dead,
            health_max_dead_letter=max_dead,
        )
        runtime._validate_cross_fields()
        return runtime
