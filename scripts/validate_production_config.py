"""Network-free validation for the production online ML configuration.

The deployment runs this module against the root-owned Docker env file before
retagging or restarting either online service.  Keep this module limited to the
Python standard library: configuration validation must not import the API,
load a model, or open Redis/Qdrant connections.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence
from urllib.parse import urlsplit


_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})
_PLACEHOLDER_PARTS = (
    "changeme",
    "change-me",
    "example",
    "placeholder",
    "replace-me",
    "replace-with",
    "replace_with",
    "your-",
    "your_",
)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._-]{0,127}$")
_HF_REVISION = re.compile(r"^[0-9a-f]{40}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SUPPORTED_PRODUCTION_EMBEDDING_MODELS = frozenset(
    {
        "all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
    }
)

# These credentials belong to acquisition, training, or legacy database jobs.
# Passing them through Docker's --env-file needlessly expands the online blast
# radius, so the production preflight rejects them even if the API ignores them.
FORBIDDEN_ONLINE_ENV_VARS = frozenset(
    {
        "DATABASE_URL",
        "TEST_DATABASE_URL",
        "LOCAL_DATABASE_URL",
        "SUPABASE_DATABASE_URL",
        "GITHUB_TOKEN",
        "OPENROUTER_API_KEY",
        "GROQ_API_KEY",
    }
)
IMAGE_OWNED_ENV_VARS = frozenset(
    {
        "ML_RELEASE_ID",
        "BAKED_ML_RELEASE_ID",
        "BAKED_EMBEDDING_MODEL",
        "BAKED_EMBEDDING_MODEL_REVISION",
    }
)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One safe-to-log configuration error."""

    name: str
    message: str

    def render(self) -> str:
        return f"{self.name}: {self.message}"


class _Validator:
    def __init__(self, environ: Mapping[str, str]) -> None:
        self.environ = environ
        self.issues: list[ValidationIssue] = []

    def issue(self, name: str, message: str) -> None:
        self.issues.append(ValidationIssue(name=name, message=message))

    def raw(self, name: str, *, required: bool = False) -> str | None:
        value = self.environ.get(name)
        if value is None or not value.strip():
            if required:
                self.issue(name, "is required and must not be empty")
            return None
        return value.strip()

    def exact(self, name: str, expected: str) -> str | None:
        value = self.raw(name, required=True)
        if value is not None and value != expected:
            self.issue(name, f"must be exactly {expected!r} in production")
        return value

    def boolean(
        self,
        name: str,
        *,
        required: bool = True,
        default: bool | None = None,
    ) -> bool | None:
        value = self.raw(name, required=required and default is None)
        if value is None:
            return default
        normalized = value.casefold()
        if normalized in _TRUE:
            return True
        if normalized in _FALSE:
            return False
        self.issue(name, "must be a boolean (true or false)")
        return None

    def integer(
        self,
        name: str,
        *,
        minimum: int,
        maximum: int,
        required: bool = True,
        default: int | None = None,
    ) -> int | None:
        value = self.raw(name, required=required and default is None)
        if value is None:
            return default
        try:
            parsed = int(value)
        except ValueError:
            self.issue(name, "must be an integer")
            return None
        if not minimum <= parsed <= maximum:
            self.issue(name, f"must be between {minimum} and {maximum}")
            return None
        return parsed

    def number(
        self,
        name: str,
        *,
        minimum: float,
        maximum: float,
        required: bool = True,
        default: float | None = None,
    ) -> float | None:
        value = self.raw(name, required=required and default is None)
        if value is None:
            return default
        try:
            parsed = float(value)
        except ValueError:
            self.issue(name, "must be a number")
            return None
        if not minimum <= parsed <= maximum:
            self.issue(name, f"must be between {minimum:g} and {maximum:g}")
            return None
        return parsed

    def name(self, name: str, *, required: bool = True) -> str | None:
        value = self.raw(name, required=required)
        if value is not None and not _SAFE_NAME.fullmatch(value):
            self.issue(
                name,
                "must contain only letters, digits, colon, dot, underscore, or dash",
            )
        return value

    def url(self, name: str, *, schemes: set[str]) -> str | None:
        value = self.raw(name, required=True)
        if value is None:
            return None
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError:
            self.issue(name, "must be a valid URL with a valid port")
            return None
        if parsed.scheme.casefold() not in schemes:
            self.issue(name, f"scheme must be one of {', '.join(sorted(schemes))}")
        if not parsed.hostname:
            self.issue(name, "must include a hostname")
        if parsed.fragment:
            self.issue(name, "must not include a URL fragment")
        if port is not None and not 1 <= port <= 65535:
            self.issue(name, "port must be between 1 and 65535")
        if _looks_like_placeholder(value):
            self.issue(name, "must not contain a placeholder value")
        return value


def _looks_like_placeholder(value: str) -> bool:
    normalized = value.strip().casefold()
    return any(part in normalized for part in _PLACEHOLDER_PARTS)


def _csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_feedback(v: _Validator) -> None:
    v.integer("FEEDBACK_CONSUMER_WORKERS", minimum=1, maximum=16, required=False)
    redis_url = v.url("REDIS_URL", schemes={"redis", "rediss"})
    auth_mode = v.raw("REDIS_AUTH_MODE", required=True)
    if auth_mode is not None and auth_mode != "acl_url":
        v.issue("REDIS_AUTH_MODE", "must be exactly 'acl_url' in production")
    if redis_url is not None:
        parsed_redis = urlsplit(redis_url)
        if not parsed_redis.username or not parsed_redis.password:
            v.issue("REDIS_URL", "must include an ACL username and password")
        elif len(parsed_redis.password) < 16 or _looks_like_placeholder(
            parsed_redis.password
        ):
            v.issue("REDIS_URL", "must contain a non-placeholder ACL password of at least 16 bytes")
        if parsed_redis.query:
            v.issue("REDIS_URL", "must not contain query options in production")
    allow_memory = v.boolean("FEEDBACK_ALLOW_MEMORY_FALLBACK")
    if allow_memory:
        v.issue(
            "FEEDBACK_ALLOW_MEMORY_FALLBACK",
            "must be false in production; durable Redis is required",
        )

    stream_name = v.name("FEEDBACK_STREAM_NAME")
    stream_maxlen = v.integer(
        "FEEDBACK_STREAM_MAXLEN", minimum=100, maximum=100_000_000
    )
    v.name("FEEDBACK_CONSUMER_GROUP")
    v.name("FEEDBACK_CONSUMER_PREFIX")
    heartbeat_key = v.name("FEEDBACK_CONSUMER_HEARTBEAT_KEY")
    heartbeat_ttl = v.integer(
        "FEEDBACK_CONSUMER_HEARTBEAT_TTL_SECONDS", minimum=5, maximum=3_600
    )
    read_batch = v.integer(
        "FEEDBACK_READ_BATCH_SIZE", minimum=1, maximum=1_000
    )
    read_block = v.integer(
        "FEEDBACK_READ_BLOCK_MS", minimum=10, maximum=60_000
    )
    reclaim_idle = v.integer(
        "FEEDBACK_RECLAIM_IDLE_MS", minimum=1_000, maximum=86_400_000
    )
    idempotency_ttl = v.integer(
        "FEEDBACK_IDEMPOTENCY_TTL_SECONDS", minimum=3_600, maximum=31_536_000
    )
    lock_ttl = v.integer(
        "FEEDBACK_USER_LOCK_TTL_SECONDS", minimum=5, maximum=3_600
    )
    lock_wait = v.number(
        "FEEDBACK_USER_LOCK_WAIT_SECONDS", minimum=0.1, maximum=300
    )
    lock_renew = v.number(
        "FEEDBACK_USER_LOCK_RENEW_INTERVAL_SECONDS",
        minimum=0.05,
        maximum=1_800,
    )
    v.integer("FEEDBACK_MAX_DELIVERY_ATTEMPTS", minimum=1, maximum=100)
    dead_letter_stream = v.name("FEEDBACK_DEAD_LETTER_STREAM")
    dead_letter_maxlen = v.integer(
        "FEEDBACK_DEAD_LETTER_MAXLEN", minimum=10, maximum=10_000_000
    )
    v.name("FEEDBACK_USER_LOCK_PREFIX")
    v.integer("FEEDBACK_REJECTION_HISTORY_SIZE", minimum=1, maximum=1_000)
    v.integer(
        "FEEDBACK_MAX_TRACKED_REPOSITORIES", minimum=1, maximum=512
    )
    v.integer(
        "FEEDBACK_MAX_USER_STATE_BYTES", minimum=65_536, maximum=2_000_000
    )
    v.number("FEEDBACK_QDRANT_TIMEOUT_SECONDS", minimum=0.1, maximum=120)

    dwell_min = v.number(
        "FEEDBACK_DWELL_MIN_SECONDS", minimum=0, maximum=300
    )
    dwell_full = v.number(
        "FEEDBACK_DWELL_FULL_CREDIT_SECONDS", minimum=0.001, maximum=300
    )
    v.number("FEEDBACK_DWELL_MAX_ALPHA", minimum=0.000001, maximum=0.15)

    warn_pending = v.integer(
        "FEEDBACK_HEALTH_WARN_PENDING", minimum=0, maximum=100_000_000
    )
    max_pending = v.integer(
        "FEEDBACK_HEALTH_MAX_PENDING", minimum=1, maximum=100_000_000
    )
    warn_lag = v.integer(
        "FEEDBACK_HEALTH_WARN_LAG", minimum=0, maximum=100_000_000
    )
    max_lag = v.integer(
        "FEEDBACK_HEALTH_MAX_LAG", minimum=1, maximum=100_000_000
    )
    warn_stream = v.integer(
        "FEEDBACK_HEALTH_WARN_STREAM_LENGTH", minimum=0, maximum=100_000_000
    )
    max_stream = v.integer(
        "FEEDBACK_HEALTH_MAX_STREAM_LENGTH", minimum=1, maximum=100_000_000
    )
    warn_dlq = v.integer(
        "FEEDBACK_HEALTH_WARN_DEAD_LETTER", minimum=0, maximum=10_000_000
    )
    max_dlq = v.integer(
        "FEEDBACK_HEALTH_MAX_DEAD_LETTER", minimum=1, maximum=10_000_000
    )

    if stream_name and dead_letter_stream and stream_name == dead_letter_stream:
        v.issue(
            "FEEDBACK_DEAD_LETTER_STREAM",
            "must differ from FEEDBACK_STREAM_NAME",
        )
    if heartbeat_key and heartbeat_key in {stream_name, dead_letter_stream}:
        v.issue(
            "FEEDBACK_CONSUMER_HEARTBEAT_KEY",
            "must differ from both feedback stream names",
        )
    if read_block is not None and reclaim_idle is not None and reclaim_idle <= read_block:
        v.issue(
            "FEEDBACK_RECLAIM_IDLE_MS",
            "must exceed FEEDBACK_READ_BLOCK_MS",
        )
    if lock_ttl is not None and lock_wait is not None and lock_ttl <= lock_wait:
        v.issue(
            "FEEDBACK_USER_LOCK_TTL_SECONDS",
            "must exceed FEEDBACK_USER_LOCK_WAIT_SECONDS",
        )
    if lock_ttl is not None and lock_renew is not None and lock_renew >= lock_ttl / 2:
        v.issue(
            "FEEDBACK_USER_LOCK_RENEW_INTERVAL_SECONDS",
            "must be shorter than half FEEDBACK_USER_LOCK_TTL_SECONDS",
        )
    if heartbeat_ttl is not None and read_block is not None:
        if heartbeat_ttl * 1_000 <= read_block:
            v.issue(
                "FEEDBACK_CONSUMER_HEARTBEAT_TTL_SECONDS",
                "must exceed FEEDBACK_READ_BLOCK_MS",
            )
    if dwell_min is not None and dwell_full is not None and dwell_full <= dwell_min:
        v.issue(
            "FEEDBACK_DWELL_FULL_CREDIT_SECONDS",
            "must exceed FEEDBACK_DWELL_MIN_SECONDS",
        )
    if (
        idempotency_ttl is not None
        and reclaim_idle is not None
        and idempotency_ttl * 1_000 < reclaim_idle * 2
    ):
        v.issue(
            "FEEDBACK_IDEMPOTENCY_TTL_SECONDS",
            "must be at least twice FEEDBACK_RECLAIM_IDLE_MS",
        )

    threshold_pairs = (
        ("FEEDBACK_HEALTH_WARN_PENDING", warn_pending, "FEEDBACK_HEALTH_MAX_PENDING", max_pending),
        ("FEEDBACK_HEALTH_WARN_LAG", warn_lag, "FEEDBACK_HEALTH_MAX_LAG", max_lag),
        (
            "FEEDBACK_HEALTH_WARN_STREAM_LENGTH",
            warn_stream,
            "FEEDBACK_HEALTH_MAX_STREAM_LENGTH",
            max_stream,
        ),
        (
            "FEEDBACK_HEALTH_WARN_DEAD_LETTER",
            warn_dlq,
            "FEEDBACK_HEALTH_MAX_DEAD_LETTER",
            max_dlq,
        ),
    )
    for warn_name, warn_value, max_name, max_value in threshold_pairs:
        if warn_value is not None and max_value is not None and warn_value > max_value:
            v.issue(warn_name, f"must not exceed {max_name}")
    if stream_maxlen is not None and max_stream is not None and max_stream < stream_maxlen:
        v.issue(
            "FEEDBACK_HEALTH_MAX_STREAM_LENGTH",
            "must be at least FEEDBACK_STREAM_MAXLEN",
        )
    if dead_letter_maxlen is not None and max_dlq is not None and max_dlq > dead_letter_maxlen:
        v.issue(
            "FEEDBACK_HEALTH_MAX_DEAD_LETTER",
            "must not exceed FEEDBACK_DEAD_LETTER_MAXLEN",
        )
    if read_batch is not None and stream_maxlen is not None and read_batch > stream_maxlen:
        v.issue(
            "FEEDBACK_READ_BATCH_SIZE",
            "must not exceed FEEDBACK_STREAM_MAXLEN",
        )


def _validate_vector_and_model(v: _Validator) -> None:
    qdrant_url = v.url("QDRANT_URL", schemes={"https"})
    if qdrant_url is not None:
        parsed_qdrant = urlsplit(qdrant_url)
        if parsed_qdrant.username is not None or parsed_qdrant.password is not None:
            v.issue(
                "QDRANT_URL",
                "must not contain credentials; use QDRANT_API_KEY",
            )
        if parsed_qdrant.query:
            v.issue("QDRANT_URL", "must not contain transport-security query options")
    auth_mode = v.raw("QDRANT_AUTH_MODE", required=True)
    if auth_mode is not None and auth_mode != "api_key":
        v.issue("QDRANT_AUTH_MODE", "must be exactly 'api_key' for this runtime")
    qdrant_key = v.raw("QDRANT_API_KEY", required=True)
    if qdrant_key is not None:
        if len(qdrant_key.encode("utf-8")) < 16 or _looks_like_placeholder(qdrant_key):
            v.issue("QDRANT_API_KEY", "must be a non-placeholder key of at least 16 bytes")

    repository_collection = v.name("QDRANT_COLLECTION_NAME")
    if (
        repository_collection is not None
        and repository_collection != "osiris_research_corpus_v2_20260722_r1"
    ):
        v.issue(
            "QDRANT_COLLECTION_NAME",
            "must target the frozen strict-V2 repository corpus",
        )
    v.number("QDRANT_TIMEOUT_SECONDS", minimum=1, maximum=120)
    distance = v.raw("QDRANT_DISTANCE", required=True)
    if distance is not None and distance.casefold() != "cosine":
        v.issue("QDRANT_DISTANCE", "must be Cosine for the frozen V2 contract")
    v.name("QDRANT_VECTOR_NAME")
    user_collection = v.name("USER_PROFILES_COLLECTION")
    user_vector = v.raw("USER_PROFILE_VECTOR_NAME")
    if user_vector is not None:
        if not _SAFE_NAME.fullmatch(user_vector):
            v.issue("USER_PROFILE_VECTOR_NAME", "contains invalid characters")
        v.issue(
            "USER_PROFILE_VECTOR_NAME",
            "must be unset while the production user collection uses its frozen unnamed vector",
        )
    if repository_collection and user_collection and repository_collection == user_collection:
        v.issue(
            "USER_PROFILES_COLLECTION",
            "must differ from QDRANT_COLLECTION_NAME",
        )
    vector_dimension = v.integer("VECTOR_DIMENSION", minimum=8, maximum=8_192)
    if vector_dimension is not None and vector_dimension != 384:
        v.issue(
            "VECTOR_DIMENSION",
            "must be 384 for the configured all-MiniLM-L6-v2 collection contract",
        )
    if v.boolean("V2_USER_COLLECTION_REQUIRED") is not True:
        v.issue(
            "V2_USER_COLLECTION_REQUIRED",
            "must be true in strict V2 production",
        )

    model = v.raw("EMBEDDING_MODEL", required=True)
    if model is not None and _looks_like_placeholder(model):
        v.issue("EMBEDDING_MODEL", "must not be a placeholder")
    if model is not None and model not in _SUPPORTED_PRODUCTION_EMBEDDING_MODELS:
        v.issue(
            "EMBEDDING_MODEL",
            "is not supported by the frozen 384-dimensional online pipelines",
        )
    baked_model = v.raw("BAKED_EMBEDDING_MODEL")
    if baked_model is not None and model is not None and baked_model != model:
        v.issue(
            "EMBEDDING_MODEL",
            "must exactly match the model baked into the image",
        )
    revision = v.raw("EMBEDDING_MODEL_REVISION", required=True)
    if revision is not None and not _HF_REVISION.fullmatch(revision):
        v.issue(
            "EMBEDDING_MODEL_REVISION",
            "must be a pinned 40-character lowercase commit SHA",
        )
    baked_revision = v.raw("BAKED_EMBEDDING_MODEL_REVISION")
    if baked_revision is not None and revision is not None and baked_revision != revision:
        v.issue(
            "EMBEDDING_MODEL_REVISION",
            "must exactly match the revision baked into the image",
        )
    embedding_version = v.name("REPOSITORY_EMBEDDING_VERSION")
    compatible_raw = v.raw("V2_COMPATIBLE_EMBEDDING_VERSIONS", required=True)
    compatible = _csv(compatible_raw)
    if compatible_raw is not None and not compatible:
        v.issue(
            "V2_COMPATIBLE_EMBEDDING_VERSIONS",
            "must contain at least one version",
        )
    if len(compatible) != len(set(compatible)):
        v.issue(
            "V2_COMPATIBLE_EMBEDDING_VERSIONS",
            "must not contain duplicate versions",
        )
    for item in compatible:
        if not _SAFE_NAME.fullmatch(item):
            v.issue(
                "V2_COMPATIBLE_EMBEDDING_VERSIONS",
                "contains an invalid version name",
            )
            break
    if embedding_version and compatible and embedding_version not in compatible:
        v.issue(
            "V2_COMPATIBLE_EMBEDDING_VERSIONS",
            "must include REPOSITORY_EMBEDDING_VERSION",
        )

    v.integer("V2_REQUIRED_CONTENT_VERSION", minimum=1, maximum=2_147_483_647)
    feature_spec = v.name("REPOSITORY_FEATURE_SPEC_VERSION")
    required_feature_spec = v.name("V2_REQUIRED_FEATURE_SPEC_VERSION")
    if feature_spec and required_feature_spec and feature_spec != required_feature_spec:
        v.issue(
            "V2_REQUIRED_FEATURE_SPEC_VERSION",
            "must match REPOSITORY_FEATURE_SPEC_VERSION",
        )
    v.integer("MIN_ELIGIBLE_REPOSITORIES", minimum=1, maximum=100_000_000)
    chunk_size = v.integer("README_CHUNK_CHARS", minimum=128, maximum=100_000)
    chunk_overlap = v.integer(
        "README_CHUNK_OVERLAP_CHARS", minimum=0, maximum=99_999
    )
    if (
        chunk_size is not None
        and chunk_overlap is not None
        and chunk_overlap >= chunk_size
    ):
        v.issue(
            "README_CHUNK_OVERLAP_CHARS",
            "must be smaller than README_CHUNK_CHARS",
        )
    if v.boolean("V2_ALLOW_MISSING_EMBEDDING_REVISION") is not False:
        v.issue(
            "V2_ALLOW_MISSING_EMBEDDING_REVISION",
            "must be false in production; reindex incompatible legacy points",
        )
    if v.boolean("EMBEDDING_WARMUP_ON_STARTUP") is not True:
        v.issue(
            "EMBEDDING_WARMUP_ON_STARTUP",
            "must be true so deployment health validates the baked model",
        )
    v.number("INFERENCE_WARMUP_TIMEOUT_SECONDS", minimum=0.1, maximum=300, default=120)
    v.integer("EMBEDDING_MAX_CONCURRENCY", minimum=1, maximum=8)
    v.number("EMBEDDING_TIMEOUT_SECONDS", minimum=0.1, maximum=300)
    embedding_workers = v.integer(
        "EMBEDDING_EXECUTOR_WORKERS", minimum=1, maximum=8
    )
    embedding_capacity = v.integer(
        "EMBEDDING_MAX_OUTSTANDING_JOBS", minimum=1, maximum=64
    )
    # Preserve the cross-field invariant even when one value is independently
    # outside its allowed range.  Otherwise a worker count of 9 and a queue of
    # 4 would report only the worker error and obscure the undersized queue.
    try:
        configured_workers = int(v.environ["EMBEDDING_EXECUTOR_WORKERS"])
        configured_capacity = int(v.environ["EMBEDDING_MAX_OUTSTANDING_JOBS"])
    except (KeyError, TypeError, ValueError):
        configured_workers = embedding_workers
        configured_capacity = embedding_capacity
    if (
        configured_workers is not None
        and configured_capacity is not None
        and configured_capacity < configured_workers
    ):
        v.issue(
            "EMBEDDING_MAX_OUTSTANDING_JOBS",
            "must be at least EMBEDDING_EXECUTOR_WORKERS",
        )
    v.integer("EMBEDDING_CPU_THREADS", minimum=1, maximum=64)

    if v.boolean("HF_HUB_OFFLINE") is not True:
        v.issue("HF_HUB_OFFLINE", "must be true to prevent runtime downloads")
    if v.boolean("TRANSFORMERS_OFFLINE") is not True:
        v.issue(
            "TRANSFORMERS_OFFLINE",
            "must be true to prevent runtime downloads",
        )


def _validate_card_summary(v: _Validator) -> None:
    provider = v.raw("SUMMARY_PROVIDER", required=True)
    if provider is not None and provider not in {"openrouter", "openai"}:
        v.issue("SUMMARY_PROVIDER", "must be 'openrouter' or 'openai'")

    api_url = v.url("SUMMARY_API_URL", schemes={"https"})
    if provider == "openai" and api_url is not None and api_url != "https://api.openai.com/v1/chat/completions":
        v.issue("SUMMARY_API_URL", "must be the official OpenAI endpoint")
    if api_url is not None:
        parsed = urlsplit(api_url)
        if parsed.username is not None or parsed.password is not None:
            v.issue("SUMMARY_API_URL", "must not contain credentials")
        if parsed.query:
            v.issue("SUMMARY_API_URL", "must not contain query options")

    api_key = v.raw("SUMMARY_API_KEY", required=True)
    if api_key is not None and (
        len(api_key.encode("utf-8")) < 16 or _looks_like_placeholder(api_key)
    ):
        v.issue(
            "SUMMARY_API_KEY",
            "must be a non-placeholder provider key of at least 16 bytes",
        )
    model = v.raw("SUMMARY_MODEL_ID", required=True)
    if model is not None and (
        len(model) > 256
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:._/-]{0,255}", model)
        or _looks_like_placeholder(model)
    ):
        v.issue("SUMMARY_MODEL_ID", "must be a safe, non-placeholder model ID")
    v.exact("SUMMARY_PROMPT_VERSION", "repo-card-summary-v1")
    v.exact("SUMMARY_FORMAT_VERSION", "repo-card-summary-json-v1")
    v.integer("SUMMARY_INPUT_MAX_CHARS", minimum=1_000, maximum=30_000)
    v.number("SUMMARY_TEMPERATURE", minimum=0, maximum=0.2)
    v.integer("SUMMARY_MAX_OUTPUT_TOKENS", minimum=64, maximum=512)
    v.number("SUMMARY_REQUEST_TIMEOUT_SECONDS", minimum=0.1, maximum=30)
    v.number("SUMMARY_RPM_LIMIT", minimum=0.01, maximum=600)
    v.integer("SUMMARY_MAX_RETRIES", minimum=0, maximum=2)
    v.number("SUMMARY_RETRY_BASE_SECONDS", minimum=0.1, maximum=1)


def _validate_ranker_and_retrieval(v: _Validator) -> None:
    device = v.environ.get("V2_HEAVY_RANKER_DEVICE", "auto").strip().casefold()
    if device not in {"auto", "cpu", "cuda"}:
        v.issue("V2_HEAVY_RANKER_DEVICE", "must be auto, cpu or cuda")
    enabled = v.boolean("V2_HEAVY_RANKER_ENABLED")
    required = v.boolean("V2_HEAVY_RANKER_REQUIRED")
    if enabled is not True:
        v.issue(
            "V2_HEAVY_RANKER_ENABLED",
            "must be true in strict V2 production",
        )
    if required is not True:
        v.issue(
            "V2_HEAVY_RANKER_REQUIRED",
            "must be true so production cannot serve the fallback ranker",
        )
    if v.boolean("V2_ALLOW_UNQUALIFIED_HEAVY_RANKER") is not False:
        v.issue(
            "V2_ALLOW_UNQUALIFIED_HEAVY_RANKER",
            "must be false in production",
        )
    traffic = v.number(
        "V2_HEAVY_RANKER_TRAFFIC_PERCENT", minimum=0, maximum=100
    )
    if traffic is not None and traffic != 100:
        v.issue(
            "V2_HEAVY_RANKER_TRAFFIC_PERCENT",
            "must be 100 in strict V2 production",
        )
    if traffic is not None and traffic > 0 and enabled is not True:
        v.issue(
            "V2_HEAVY_RANKER_TRAFFIC_PERCENT",
            "must be 0 when V2_HEAVY_RANKER_ENABLED is false",
        )
    if required and not enabled:
        v.issue(
            "V2_HEAVY_RANKER_REQUIRED",
            "cannot be true when V2_HEAVY_RANKER_ENABLED is false",
        )
    if required and traffic is not None and traffic != 100:
        v.issue(
            "V2_HEAVY_RANKER_TRAFFIC_PERCENT",
            "must be 100 when V2_HEAVY_RANKER_REQUIRED is true",
        )
    v.raw("ML_MODEL_VERSION", required=True)
    v.name("V2_HEAVY_RANKER_CANARY_SALT")
    v.number("V2_EXPLORATION_FRACTION", minimum=0, maximum=0.5)
    v.integer("V2_MAX_SAME_LANGUAGE", minimum=1, maximum=100)
    v.number("V2_RECOMMENDATION_TIMEOUT_SECONDS", minimum=0.1, maximum=120)
    health_timeout = v.number("V2_HEALTH_TIMEOUT_SECONDS", minimum=0.1, maximum=30)
    health_refresh = v.number(
        "V2_HEALTH_REFRESH_SECONDS", minimum=1, maximum=300, default=15
    )
    health_max_age = v.number(
        "V2_HEALTH_MAX_AGE_SECONDS", minimum=1, maximum=600, default=45
    )
    if (
        health_timeout is not None and health_refresh is not None
        and health_max_age is not None
        and health_max_age < health_refresh + 2 * health_timeout
    ):
        v.issue(
            "V2_HEALTH_MAX_AGE_SECONDS",
            "must cover refresh interval plus two diagnostic timeouts",
        )
    # Runtime defaults apply only when absent; an explicitly blank value is invalid.
    retrieval_workers = v.integer(
        "V2_RETRIEVAL_IO_WORKERS", minimum=1, maximum=32,
        default=4 if "V2_RETRIEVAL_IO_WORKERS" not in v.environ else None,
    )
    retrieval_capacity = v.integer(
        "V2_RETRIEVAL_IO_MAX_OUTSTANDING", minimum=1, maximum=128,
        default=8 if "V2_RETRIEVAL_IO_MAX_OUTSTANDING" not in v.environ else None,
    )
    if (
        retrieval_workers is not None and retrieval_capacity is not None
        and retrieval_capacity < retrieval_workers
    ):
        v.issue(
            "V2_RETRIEVAL_IO_MAX_OUTSTANDING",
            "must be at least V2_RETRIEVAL_IO_WORKERS",
        )

    recommendation_workers = v.integer(
        "V2_RECOMMENDATION_EXECUTOR_WORKERS", minimum=1, maximum=32
    )
    recommendation_capacity = v.integer(
        "V2_RECOMMENDATION_MAX_OUTSTANDING", minimum=1, maximum=64
    )
    if (
        recommendation_workers is not None
        and recommendation_capacity is not None
        and recommendation_capacity < recommendation_workers
    ):
        v.issue(
            "V2_RECOMMENDATION_MAX_OUTSTANDING",
            "must be at least V2_RECOMMENDATION_EXECUTOR_WORKERS",
        )

    for prefix, maximum_workers, maximum_capacity, maximum_timeout in (
        ("V2_SEARCH", 4, 16, 10),
        ("V2_FEEDBACK", 8, 64, 30),
        ("V2_REFRESH", 8, 32, 120),
        ("V2_HEALTH", 4, 16, 30),
    ):
        workers = v.integer(
            f"{prefix}_EXECUTOR_WORKERS", minimum=1, maximum=maximum_workers
        )
        capacity = v.integer(
            f"{prefix}_MAX_OUTSTANDING", minimum=1, maximum=maximum_capacity
        )
        v.number(
            f"{prefix}_TIMEOUT_SECONDS", minimum=0.1, maximum=maximum_timeout
        )
        if workers is not None and capacity is not None and capacity < workers:
            v.issue(
                f"{prefix}_MAX_OUTSTANDING",
                f"must be at least {prefix}_EXECUTOR_WORKERS",
            )


def _validate_locks_and_smoke(v: _Validator) -> None:
    repository_lock_ttl = v.integer(
        "REPOSITORY_JOB_LOCK_TTL_MS", minimum=5_000, maximum=3_600_000
    )
    repository_lock_wait = v.number(
        "REPOSITORY_JOB_LOCK_WAIT_SECONDS", minimum=0.1, maximum=300
    )
    if (
        repository_lock_ttl is not None
        and repository_lock_wait is not None
        and repository_lock_ttl <= repository_lock_wait * 1_000
    ):
        v.issue(
            "REPOSITORY_JOB_LOCK_TTL_MS",
            "must exceed REPOSITORY_JOB_LOCK_WAIT_SECONDS",
        )

    smoke_user = v.raw("ML_SMOKE_USER_ID", required=True)
    if smoke_user is not None:
        try:
            from uuid import UUID

            canonical_smoke_user = str(UUID(smoke_user))
        except ValueError:
            v.issue("ML_SMOKE_USER_ID", "must be a canonical UUID")
        else:
            if canonical_smoke_user != smoke_user:
                v.issue("ML_SMOKE_USER_ID", "must be a canonical lowercase UUID")
    smoke_limit = v.integer(
        "ML_SMOKE_RECOMMENDATION_LIMIT", minimum=1, maximum=15
    )
    smoke_minimum = v.integer(
        "ML_SMOKE_EXPECT_MIN_ITEMS", minimum=1, maximum=15
    )
    if (
        smoke_limit is not None
        and smoke_minimum is not None
        and smoke_minimum > smoke_limit
    ):
        v.issue(
            "ML_SMOKE_EXPECT_MIN_ITEMS",
            "must not exceed ML_SMOKE_RECOMMENDATION_LIMIT",
        )
    v.number("ML_SMOKE_TIMEOUT_SECONDS", minimum=1, maximum=30)


def validate_production_config(
    environ: Mapping[str, str] | None = None,
    *,
    reject_image_owned_overrides: bool = False,
) -> list[ValidationIssue]:
    """Return every network-free production configuration error.

    Values are never interpolated into errors, which keeps secrets and
    credential-bearing URLs out of deployment logs.
    """

    env = dict(os.environ if environ is None else environ)
    validator = _Validator(env)

    if reject_image_owned_overrides:
        for name in sorted(IMAGE_OWNED_ENV_VARS):
            if name in env:
                validator.issue(name, "must not be set in the host environment file")
    else:
        release_id = validator.raw("ML_RELEASE_ID", required=True)
        baked_release_id = validator.raw("BAKED_ML_RELEASE_ID", required=True)
        if release_id is not None and not re.fullmatch(r"[0-9a-f]{40}", release_id):
            validator.issue("ML_RELEASE_ID", "must be the 40-character tested commit SHA")
        if (
            release_id is not None
            and baked_release_id is not None
            and release_id != baked_release_id
        ):
            validator.issue("ML_RELEASE_ID", "must match the immutable image release ID")

    validator.exact("APP_ENV", "production")
    if validator.boolean("V2_FEEDBACK_CONSUMER_REQUIRED") is not True:
        validator.issue(
            "V2_FEEDBACK_CONSUMER_REQUIRED",
            "must be true in production",
        )
    header = validator.raw("INTERNAL_API_HEADER", required=True)
    if header is not None and header.casefold() != "x-internal-secret":
        validator.issue(
            "INTERNAL_API_HEADER",
            "must be x-internal-secret for the deployed backend contract",
        )

    secret = validator.raw("INTERNAL_API_SECRET", required=True)
    if secret is not None and not re.fullmatch(r"[0-9a-f]{64}", secret):
        validator.issue(
            "INTERNAL_API_SECRET",
            "must be exactly 64 lowercase hexadecimal characters",
        )

    for name in sorted(FORBIDDEN_ONLINE_ENV_VARS):
        if env.get(name, "").strip():
            validator.issue(
                name,
                "must not be present in the online API/feedback env file",
            )

    _validate_feedback(validator)
    _validate_vector_and_model(validator)
    _validate_card_summary(validator)
    _validate_ranker_and_retrieval(validator)
    _validate_locks_and_smoke(validator)
    return validator.issues


def load_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    """Parse the Docker-compatible subset used by the production env file."""

    values: dict[str, str] = {}
    source = Path(path)
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"line {line_number} must use NAME=VALUE syntax")
        name, value = line.split("=", 1)
        name = name.strip()
        if not _ENV_NAME.fullmatch(name):
            raise ValueError(f"line {line_number} has an invalid variable name")
        if name in values:
            raise ValueError(f"line {line_number} duplicates {name}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[name] = value
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the online ML production environment without opening "
            "network connections or importing runtime services."
        )
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="validate only values from this env file instead of process env",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable result (never includes configuration values)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        environment = load_env_file(args.env_file) if args.env_file else dict(os.environ)
    except (OSError, UnicodeError, ValueError) as exc:
        message = f"Unable to parse production env file: {exc}"
        if args.json:
            print(json.dumps({"valid": False, "errors": [message]}))
        else:
            print(message, file=sys.stderr)
        return 2

    issues = validate_production_config(
        environment,
        reject_image_owned_overrides=args.env_file is not None,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "valid": not issues,
                    "errors": [issue.render() for issue in issues],
                },
                sort_keys=True,
            )
        )
    elif issues:
        print("Production configuration is invalid:", file=sys.stderr)
        for issue in issues:
            print(f"- {issue.render()}", file=sys.stderr)
    else:
        print("Production configuration is valid (network checks not performed).")
    return 0 if not issues else 2


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
