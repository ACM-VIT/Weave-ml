"""Focused tests for the network-free production preflight."""

from __future__ import annotations

import json
from secrets import token_hex
import socket

import pytest

from scripts.validate_production_config import (
    load_env_file,
    main,
    validate_production_config,
)


PINNED_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def valid_production_env() -> dict[str, str]:
    return {
        "APP_ENV": "production",
        "V2_FEEDBACK_CONSUMER_REQUIRED": "true",
        "INTERNAL_API_HEADER": "x-internal-secret",
        "INTERNAL_API_SECRET": token_hex(32),
        "REDIS_AUTH_MODE": "acl_url",
        "REDIS_URL": "redis://ml-runtime:strong-test-password@redis.internal:6379/0",
        "FEEDBACK_ALLOW_MEMORY_FALLBACK": "false",
        "FEEDBACK_STREAM_NAME": "ml:feedback:v2",
        "FEEDBACK_STREAM_MAXLEN": "100000",
        "FEEDBACK_CONSUMER_GROUP": "ml-feedback-v2",
        "FEEDBACK_CONSUMER_PREFIX": "feedback-worker",
        "FEEDBACK_CONSUMER_HEARTBEAT_KEY": "ml:feedback:v2:heartbeat",
        "FEEDBACK_CONSUMER_HEARTBEAT_TTL_SECONDS": "30",
        "FEEDBACK_READ_BATCH_SIZE": "20",
        "FEEDBACK_READ_BLOCK_MS": "1000",
        "FEEDBACK_RECLAIM_IDLE_MS": "60000",
        "FEEDBACK_IDEMPOTENCY_TTL_SECONDS": "604800",
        "FEEDBACK_USER_LOCK_TTL_SECONDS": "60",
        "FEEDBACK_USER_LOCK_WAIT_SECONDS": "10",
        "FEEDBACK_USER_LOCK_RENEW_INTERVAL_SECONDS": "20",
        "FEEDBACK_USER_LOCK_PREFIX": "ml:user-vector-lock",
        "FEEDBACK_MAX_DELIVERY_ATTEMPTS": "5",
        "FEEDBACK_DEAD_LETTER_STREAM": "ml:feedback:v2:dead-letter",
        "FEEDBACK_DEAD_LETTER_MAXLEN": "10000",
        "FEEDBACK_REJECTION_HISTORY_SIZE": "64",
        "FEEDBACK_MAX_TRACKED_REPOSITORIES": "256",
        "FEEDBACK_MAX_USER_STATE_BYTES": "1000000",
        "FEEDBACK_QDRANT_TIMEOUT_SECONDS": "10",
        "FEEDBACK_DWELL_MIN_SECONDS": "3",
        "FEEDBACK_DWELL_FULL_CREDIT_SECONDS": "300",
        "FEEDBACK_DWELL_MAX_ALPHA": "0.15",
        "FEEDBACK_HEALTH_WARN_PENDING": "1000",
        "FEEDBACK_HEALTH_MAX_PENDING": "10000",
        "FEEDBACK_HEALTH_WARN_LAG": "10000",
        "FEEDBACK_HEALTH_MAX_LAG": "50000",
        "FEEDBACK_HEALTH_WARN_STREAM_LENGTH": "80000",
        "FEEDBACK_HEALTH_MAX_STREAM_LENGTH": "100000",
        "FEEDBACK_HEALTH_WARN_DEAD_LETTER": "1",
        "FEEDBACK_HEALTH_MAX_DEAD_LETTER": "1000",
        "QDRANT_URL": "https://qdrant.internal:6333",
        "QDRANT_AUTH_MODE": "api_key",
        "QDRANT_API_KEY": "qdrant-test-key-strong",
        "QDRANT_TIMEOUT_SECONDS": "10",
        "QDRANT_DISTANCE": "Cosine",
        "QDRANT_COLLECTION_NAME": "osiris_research_corpus_v2_20260722_r1",
        "QDRANT_VECTOR_NAME": "repo_embedding",
        "USER_PROFILES_COLLECTION": "user_profiles",
        "VECTOR_DIMENSION": "384",
        "V2_USER_COLLECTION_REQUIRED": "true",
        "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        "EMBEDDING_MODEL_REVISION": PINNED_REVISION,
        "REPOSITORY_EMBEDDING_VERSION": "repo-embedding-v2",
        "V2_COMPATIBLE_EMBEDDING_VERSIONS": "repo-embedding-v2",
        "V2_REQUIRED_CONTENT_VERSION": "1",
        "REPOSITORY_FEATURE_SPEC_VERSION": "v2",
        "V2_REQUIRED_FEATURE_SPEC_VERSION": "v2",
        "V2_ALLOW_MISSING_EMBEDDING_REVISION": "false",
        "MIN_ELIGIBLE_REPOSITORIES": "1000",
        "README_CHUNK_CHARS": "2500",
        "README_CHUNK_OVERLAP_CHARS": "250",
        "SUMMARY_PROVIDER": "openrouter",
        "SUMMARY_API_URL": "https://openrouter.ai/api/v1/chat/completions",
        "SUMMARY_API_KEY": "summary-test-key-strong",
        "SUMMARY_MODEL_ID": "meta-llama/llama-3.3-70b-instruct",
        "SUMMARY_PROMPT_VERSION": "repo-card-summary-v1",
        "SUMMARY_FORMAT_VERSION": "repo-card-summary-json-v1",
        "SUMMARY_INPUT_MAX_CHARS": "12000",
        "SUMMARY_TEMPERATURE": "0.1",
        "SUMMARY_MAX_OUTPUT_TOKENS": "240",
        "SUMMARY_REQUEST_TIMEOUT_SECONDS": "30",
        "SUMMARY_RPM_LIMIT": "60",
        "SUMMARY_MAX_RETRIES": "2",
        "SUMMARY_RETRY_BASE_SECONDS": "1",
        "EMBEDDING_WARMUP_ON_STARTUP": "true",
        "EMBEDDING_TIMEOUT_SECONDS": "30",
        "EMBEDDING_MAX_CONCURRENCY": "1",
        "EMBEDDING_EXECUTOR_WORKERS": "1",
        "EMBEDDING_MAX_OUTSTANDING_JOBS": "4",
        "EMBEDDING_CPU_THREADS": "1",
        "HF_HUB_OFFLINE": "true",
        "TRANSFORMERS_OFFLINE": "true",
        "V2_HEAVY_RANKER_ENABLED": "true",
        "V2_HEAVY_RANKER_REQUIRED": "true",
        "V2_HEAVY_RANKER_TRAFFIC_PERCENT": "100",
        "V2_ALLOW_UNQUALIFIED_HEAVY_RANKER": "false",
        "V2_HEAVY_RANKER_CANARY_SALT": "v2-heavy-ranker-2026",
        "ML_MODEL_VERSION": "qdrant-hybrid-v2",
        "V2_EXPLORATION_FRACTION": "0.333333",
        "V2_MAX_SAME_LANGUAGE": "5",
        "V2_RECOMMENDATION_TIMEOUT_SECONDS": "12",
        "V2_HEALTH_TIMEOUT_SECONDS": "5",
        "V2_RETRIEVAL_IO_WORKERS": "4",
        "V2_RETRIEVAL_IO_MAX_OUTSTANDING": "8",
        "V2_RECOMMENDATION_EXECUTOR_WORKERS": "4",
        "V2_RECOMMENDATION_MAX_OUTSTANDING": "8",
        "V2_FEEDBACK_EXECUTOR_WORKERS": "2",
        "V2_FEEDBACK_MAX_OUTSTANDING": "8",
        "V2_FEEDBACK_TIMEOUT_SECONDS": "8",
        "V2_REFRESH_EXECUTOR_WORKERS": "2",
        "V2_REFRESH_MAX_OUTSTANDING": "4",
        "V2_REFRESH_TIMEOUT_SECONDS": "45",
        "V2_SEARCH_EXECUTOR_WORKERS": "1",
        "V2_SEARCH_MAX_OUTSTANDING": "4",
        "V2_SEARCH_TIMEOUT_SECONDS": "4",
        "V2_HEALTH_EXECUTOR_WORKERS": "2",
        "V2_HEALTH_MAX_OUTSTANDING": "4",
        "REPOSITORY_JOB_LOCK_TTL_MS": "600000",
        "REPOSITORY_JOB_LOCK_WAIT_SECONDS": "30",
        "ML_SMOKE_USER_ID": "b7bf08f4-bc62-43a6-b27e-3705608322b7",
        "ML_SMOKE_RECOMMENDATION_LIMIT": "3",
        "ML_SMOKE_EXPECT_MIN_ITEMS": "1",
        "ML_SMOKE_TIMEOUT_SECONDS": "10",
        "ML_RELEASE_ID": "0123456789abcdef0123456789abcdef01234567",
        "BAKED_ML_RELEASE_ID": "0123456789abcdef0123456789abcdef01234567",
    }


def issue_names(environment: dict[str, str]) -> set[str]:
    return {issue.name for issue in validate_production_config(environment)}


def test_valid_production_environment_passes() -> None:
    assert validate_production_config(valid_production_env()) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("SUMMARY_REQUEST_TIMEOUT_SECONDS", "30.1"),
        ("SUMMARY_MAX_RETRIES", "3"),
        ("SUMMARY_RETRY_BASE_SECONDS", "1.1"),
    ],
)
def test_production_rejects_summary_settings_that_can_overrun_the_lease(
    field: str, value: str
) -> None:
    environment = valid_production_env()
    environment[field] = value

    assert field in issue_names(environment)


def test_production_requires_a_bounded_embedding_timeout() -> None:
    environment = valid_production_env()
    environment["EMBEDDING_TIMEOUT_SECONDS"] = "301"

    assert "EMBEDDING_TIMEOUT_SECONDS" in issue_names(environment)


def test_valid_tls_redis_production_environment_passes() -> None:
    environment = valid_production_env()
    environment["REDIS_URL"] = (
        "rediss://ml-runtime:strong-test-password@redis.internal:6379/0"
    )

    assert validate_production_config(environment) == []


def test_distributed_production_template_requires_secret_replacement() -> None:
    environment = load_env_file("deploy/production.env.example")

    issues = validate_production_config(environment)

    assert "INTERNAL_API_SECRET" in {issue.name for issue in issues}


def test_exact_production_flags_fail_closed() -> None:
    environment = valid_production_env()
    environment.update(
        APP_ENV="prod",
        V2_FEEDBACK_CONSUMER_REQUIRED="false",
        V2_USER_COLLECTION_REQUIRED="false",
        V2_HEAVY_RANKER_ENABLED="false",
        V2_HEAVY_RANKER_REQUIRED="false",
        V2_HEAVY_RANKER_TRAFFIC_PERCENT="0",
    )

    names = issue_names(environment)

    assert {
        "APP_ENV",
        "V2_FEEDBACK_CONSUMER_REQUIRED",
        "V2_USER_COLLECTION_REQUIRED",
        "V2_HEAVY_RANKER_ENABLED",
        "V2_HEAVY_RANKER_REQUIRED",
        "V2_HEAVY_RANKER_TRAFFIC_PERCENT",
    } <= names


def test_secret_error_never_contains_secret() -> None:
    environment = valid_production_env()
    environment["INTERNAL_API_SECRET"] = "replace-me"

    issues = validate_production_config(environment)
    rendered = "\n".join(issue.render() for issue in issues)

    assert "INTERNAL_API_SECRET" in rendered
    assert "replace-me" not in rendered


def test_secret_requires_exact_lowercase_hex_contract() -> None:
    for invalid in ("a" * 63, "A" * 64, "é" * 64, "g" * 64):
        environment = valid_production_env()
        environment["INTERNAL_API_SECRET"] = invalid
        assert "INTERNAL_API_SECRET" in issue_names(environment)


def test_service_executor_capacity_is_bounded() -> None:
    environment = valid_production_env()
    environment.update(
        V2_FEEDBACK_EXECUTOR_WORKERS="4",
        V2_FEEDBACK_MAX_OUTSTANDING="2",
        V2_REFRESH_TIMEOUT_SECONDS="121",
        V2_HEALTH_EXECUTOR_WORKERS="5",
        V2_SEARCH_EXECUTOR_WORKERS="2",
        V2_SEARCH_MAX_OUTSTANDING="1",
    )

    names = issue_names(environment)
    assert {
        "V2_FEEDBACK_MAX_OUTSTANDING",
        "V2_REFRESH_TIMEOUT_SECONDS",
        "V2_HEALTH_EXECUTOR_WORKERS",
        "V2_SEARCH_MAX_OUTSTANDING",
    } <= names


@pytest.mark.parametrize(("field", "value"), [
    ("V2_SEARCH_EXECUTOR_WORKERS", "0"),
    ("V2_SEARCH_EXECUTOR_WORKERS", "5"),
    ("V2_SEARCH_MAX_OUTSTANDING", "0"),
    ("V2_SEARCH_MAX_OUTSTANDING", "17"),
    ("V2_SEARCH_TIMEOUT_SECONDS", "0.09"),
    ("V2_SEARCH_TIMEOUT_SECONDS", "10.1"),
    ("V2_SEARCH_TIMEOUT_SECONDS", "nan"),
])
def test_search_preflight_rejects_settings_that_fail_runtime_startup(field, value, monkeypatch):
    from api.runtime import _settings

    environment = valid_production_env()
    environment[field] = value
    for name in ("V2_SEARCH_EXECUTOR_WORKERS", "V2_SEARCH_MAX_OUTSTANDING", "V2_SEARCH_TIMEOUT_SECONDS"):
        monkeypatch.setenv(name, environment[name])

    with pytest.raises(ValueError):
        _settings("search")
    assert field in issue_names(environment)


@pytest.mark.parametrize(("workers", "capacity", "timeout"), [
    ("1", "4", "4"),
    ("1", "1", "0.1"),
    ("4", "16", "10"),
], ids=["defaults", "minimum", "maximum"])
def test_search_preflight_and_runtime_accept_defaults_and_bounds(workers, capacity, timeout, monkeypatch):
    from api.runtime import _settings

    environment = valid_production_env()
    environment.update(
        V2_SEARCH_EXECUTOR_WORKERS=workers,
        V2_SEARCH_MAX_OUTSTANDING=capacity,
        V2_SEARCH_TIMEOUT_SECONDS=timeout,
    )
    for name in ("V2_SEARCH_EXECUTOR_WORKERS", "V2_SEARCH_MAX_OUTSTANDING", "V2_SEARCH_TIMEOUT_SECONDS"):
        monkeypatch.setenv(name, environment[name])

    assert _settings("search") == (int(workers), int(capacity), float(timeout))
    assert validate_production_config(environment) == []


@pytest.mark.parametrize(("field", "value"), [
    ("V2_RETRIEVAL_IO_WORKERS", "0"),
    ("V2_RETRIEVAL_IO_WORKERS", "33"),
    ("V2_RETRIEVAL_IO_WORKERS", "1.5"),
    ("V2_RETRIEVAL_IO_WORKERS", ""),
    ("V2_RETRIEVAL_IO_MAX_OUTSTANDING", "0"),
    ("V2_RETRIEVAL_IO_MAX_OUTSTANDING", "129"),
    ("V2_RETRIEVAL_IO_MAX_OUTSTANDING", "nan"),
    ("V2_RETRIEVAL_IO_MAX_OUTSTANDING", ""),
    ("V2_RETRIEVAL_IO_MAX_OUTSTANDING", "3"),
])
def test_retrieval_preflight_rejects_settings_that_fail_runtime_startup(field, value, monkeypatch):
    from retrieval.runtime import _settings

    environment = valid_production_env()
    environment[field] = value
    for name in ("V2_RETRIEVAL_IO_WORKERS", "V2_RETRIEVAL_IO_MAX_OUTSTANDING"):
        monkeypatch.setenv(name, environment[name])

    with pytest.raises(ValueError):
        _settings()
    assert field in issue_names(environment)


@pytest.mark.parametrize(("workers", "capacity"), [
    ("4", "8"), ("1", "1"), ("32", "128"),
], ids=["defaults", "minimum", "maximum"])
def test_retrieval_preflight_and_runtime_accept_defaults_and_bounds(workers, capacity, monkeypatch):
    from retrieval.runtime import _settings

    environment = valid_production_env()
    environment.update(V2_RETRIEVAL_IO_WORKERS=workers, V2_RETRIEVAL_IO_MAX_OUTSTANDING=capacity)
    for name in ("V2_RETRIEVAL_IO_WORKERS", "V2_RETRIEVAL_IO_MAX_OUTSTANDING"):
        monkeypatch.setenv(name, environment[name])

    assert _settings() == (int(workers), int(capacity))
    assert validate_production_config(environment) == []


def test_retrieval_preflight_and_runtime_use_the_same_absent_defaults(monkeypatch):
    from retrieval.runtime import _settings

    environment = valid_production_env()
    for name in ("V2_RETRIEVAL_IO_WORKERS", "V2_RETRIEVAL_IO_MAX_OUTSTANDING"):
        environment.pop(name)
        monkeypatch.delenv(name, raising=False)

    assert _settings() == (4, 8)
    assert validate_production_config(environment) == []


@pytest.mark.parametrize("timeout", ["0", "0.5", "0.999", "120.1", "nan", "inf"])
def test_qdrant_preflight_and_retriever_reject_unsupported_timeout(timeout, monkeypatch):
    from retrieval.v2_retriever import QdrantV2Retriever

    environment = valid_production_env()
    environment["QDRANT_TIMEOUT_SECONDS"] = timeout
    monkeypatch.setenv("QDRANT_TIMEOUT_SECONDS", timeout)

    with pytest.raises(ValueError, match="QDRANT_TIMEOUT_SECONDS"):
        QdrantV2Retriever(client=object(), heavy_ranker_enabled=False)
    assert "QDRANT_TIMEOUT_SECONDS" in issue_names(environment)


@pytest.mark.parametrize("timeout", ["1", "1.5", "10", "120"])
def test_qdrant_preflight_and_retriever_accept_supported_timeout(timeout, monkeypatch):
    from retrieval.v2_retriever import QdrantV2Retriever

    environment = valid_production_env()
    environment["QDRANT_TIMEOUT_SECONDS"] = timeout
    monkeypatch.setenv("QDRANT_TIMEOUT_SECONDS", timeout)

    retriever = QdrantV2Retriever(client=object(), heavy_ranker_enabled=False)
    assert retriever.qdrant_timeout_seconds == float(timeout)
    assert "QDRANT_TIMEOUT_SECONDS" not in issue_names(environment)


def test_online_env_rejects_database_and_acquisition_credentials() -> None:
    environment = valid_production_env()
    environment["DATABASE_URL"] = "postgresql://sensitive.example/db"
    environment["GITHUB_TOKEN"] = "github_pat_sensitive"

    names = issue_names(environment)

    assert {"DATABASE_URL", "GITHUB_TOKEN"} <= names


def test_summary_provider_configuration_is_scoped_and_bounded() -> None:
    environment = valid_production_env()
    environment.update(
        SUMMARY_API_URL="http://provider.internal/v1/chat/completions",
        SUMMARY_API_KEY="replace-me",
        SUMMARY_PROMPT_VERSION="unversioned",
        SUMMARY_TEMPERATURE="0.9",
    )

    names = issue_names(environment)

    assert {
        "SUMMARY_API_URL",
        "SUMMARY_API_KEY",
        "SUMMARY_PROMPT_VERSION",
        "SUMMARY_TEMPERATURE",
    } <= names


def test_url_validation_rejects_wrong_schemes_and_placeholders() -> None:
    environment = valid_production_env()
    environment["REDIS_URL"] = "http://redis.internal:6379"
    environment["QDRANT_URL"] = "https://your-qdrant.example:6333"

    names = issue_names(environment)

    assert {"REDIS_URL", "QDRANT_URL"} <= names


def test_feedback_names_and_qdrant_auth_fail_closed() -> None:
    environment = valid_production_env()
    environment.update(
        FEEDBACK_CONSUMER_HEARTBEAT_KEY=environment["FEEDBACK_STREAM_NAME"],
        QDRANT_AUTH_MODE="none",
        QDRANT_API_KEY="replace-me",
    )

    names = issue_names(environment)

    assert {"FEEDBACK_CONSUMER_HEARTBEAT_KEY", "QDRANT_AUTH_MODE", "QDRANT_API_KEY"} <= names


def test_production_rejects_non_v2_repository_collection() -> None:
    environment = valid_production_env()
    environment["QDRANT_COLLECTION_NAME"] = "osiris_research_corpus"

    assert "QDRANT_COLLECTION_NAME" in issue_names(environment)


def test_redis_requires_acl_authentication_and_image_identity_cannot_be_overridden() -> None:
    environment = valid_production_env()
    environment.update(
        REDIS_AUTH_MODE="none",
        REDIS_URL="redis://redis.internal:6379/0",
    )
    names = issue_names(environment)
    assert {"REDIS_AUTH_MODE", "REDIS_URL"} <= names

    host_file = valid_production_env()
    host_file["ML_RELEASE_ID"] = "f" * 40
    issues = validate_production_config(
        host_file,
        reject_image_owned_overrides=True,
    )
    assert "ML_RELEASE_ID" in {issue.name for issue in issues}

    environment = valid_production_env()
    environment["REDIS_URL"] += "?ssl_cert_reqs=none"
    environment["QDRANT_URL"] = "http://qdrant.internal:6333"
    names = issue_names(environment)
    assert {"REDIS_URL", "QDRANT_URL"} <= names


def test_attempt_ttl_cannot_expire_between_pending_reclaims() -> None:
    environment = valid_production_env()
    environment.update(
        FEEDBACK_IDEMPOTENCY_TTL_SECONDS="3600",
        FEEDBACK_RECLAIM_IDLE_MS="7200000",
    )

    assert "FEEDBACK_IDEMPOTENCY_TTL_SECONDS" in issue_names(environment)


def test_embedding_revision_and_allowlist_are_exact() -> None:
    environment = valid_production_env()
    environment["EMBEDDING_MODEL_REVISION"] = "main"
    environment["V2_COMPATIBLE_EMBEDDING_VERSIONS"] = "repo-embedding-v1"

    names = issue_names(environment)

    assert {"EMBEDDING_MODEL_REVISION", "V2_COMPATIBLE_EMBEDDING_VERSIONS"} <= names


def test_arbitrary_baked_model_cannot_pass_the_frozen_pipeline_contract() -> None:
    environment = valid_production_env()
    environment.update(
        EMBEDDING_MODEL="organization/arbitrary-384d-model",
        BAKED_EMBEDDING_MODEL="organization/arbitrary-384d-model",
    )

    assert "EMBEDDING_MODEL" in issue_names(environment)


def test_runtime_limits_match_the_online_contract() -> None:
    environment = valid_production_env()
    environment.update(
        VECTOR_DIMENSION="768",
        USER_PROFILE_VECTOR_NAME="profile_embedding",
        EMBEDDING_MAX_CONCURRENCY="9",
        EMBEDDING_EXECUTOR_WORKERS="9",
        EMBEDDING_MAX_OUTSTANDING_JOBS="4",
    )

    names = issue_names(environment)

    assert {
        "VECTOR_DIMENSION",
        "USER_PROFILE_VECTOR_NAME",
        "EMBEDDING_MAX_CONCURRENCY",
        "EMBEDDING_EXECUTOR_WORKERS",
        "EMBEDDING_MAX_OUTSTANDING_JOBS",
    } <= names


def test_heavy_ranker_traffic_requires_ranker_to_be_enabled() -> None:
    environment = valid_production_env()
    environment.update(
        V2_HEAVY_RANKER_ENABLED="false",
        V2_HEAVY_RANKER_REQUIRED="false",
        V2_HEAVY_RANKER_TRAFFIC_PERCENT="1",
    )

    assert "V2_HEAVY_RANKER_TRAFFIC_PERCENT" in issue_names(environment)


def test_required_heavy_ranker_requires_all_recommendation_traffic() -> None:
    environment = valid_production_env()
    environment.update(
        V2_HEAVY_RANKER_ENABLED="true",
        V2_HEAVY_RANKER_REQUIRED="true",
        V2_HEAVY_RANKER_TRAFFIC_PERCENT="99",
    )

    assert "V2_HEAVY_RANKER_TRAFFIC_PERCENT" in issue_names(environment)


def test_feedback_thresholds_and_streams_are_consistent() -> None:
    environment = valid_production_env()
    environment.update(
        FEEDBACK_DEAD_LETTER_STREAM="ml:feedback:v2",
        FEEDBACK_HEALTH_WARN_PENDING="20000",
        FEEDBACK_HEALTH_MAX_PENDING="10000",
        FEEDBACK_HEALTH_MAX_STREAM_LENGTH="50000",
    )

    names = issue_names(environment)

    assert {
        "FEEDBACK_DEAD_LETTER_STREAM",
        "FEEDBACK_HEALTH_WARN_PENDING",
        "FEEDBACK_HEALTH_MAX_STREAM_LENGTH",
    } <= names


def test_env_file_parser_rejects_duplicate_names(tmp_path) -> None:
    path = tmp_path / "ml.env"
    path.write_text("APP_ENV=production\nAPP_ENV=development\n", encoding="utf-8")

    try:
        load_env_file(path)
    except ValueError as exc:
        assert "duplicates APP_ENV" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("duplicate env name was accepted")


def test_cli_is_network_free_and_returns_safe_json(monkeypatch, capsys) -> None:
    environment = valid_production_env()
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("network connection attempted")
        ),
    )
    monkeypatch.setattr("scripts.validate_production_config.os.environ", environment)

    assert main(["--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {"errors": [], "valid": True}
