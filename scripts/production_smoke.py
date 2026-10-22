"""Bounded authenticated smoke checks for a deployed V2 ML container.

The script deliberately prints only contract metadata and counts.  It never
prints the internal secret, user vector, recommendation payload, or dependency
URLs.  Deployment executes it inside the API container so the secret is read
from the container environment rather than passed on a command line.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import NAMESPACE_URL, UUID, uuid5


MAX_RESPONSE_BYTES = 1_000_000
EXPECTED_SERVING_ELIGIBILITY_VERSION = "repository-vector-v2"
MINIMUM_QDRANT_SERVER_VERSION = (1, 18, 0)


class SmokeFailure(RuntimeError):
    """Safe-to-log smoke failure without response bodies or credentials."""


def _boolean(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "on"}


def _read_json(
    request: Request,
    *,
    timeout: float,
) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed local URL
            status = int(response.status)
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise SmokeFailure(f"endpoint returned HTTP {exc.code}") from None
    except (URLError, TimeoutError, OSError) as exc:
        raise SmokeFailure(f"endpoint is unavailable ({type(exc).__name__})") from None
    if status < 200 or status >= 300:
        raise SmokeFailure(f"endpoint returned HTTP {status}")
    if len(raw) > MAX_RESPONSE_BYTES:
        raise SmokeFailure("endpoint response exceeded the bounded smoke limit")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SmokeFailure("endpoint did not return valid JSON") from None
    if not isinstance(payload, dict):
        raise SmokeFailure("endpoint returned a non-object JSON response")
    return payload


def _request(
    path: str,
    *,
    secret: str,
    timeout: float,
    environment: Mapping[str, str] | None = None,
    body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    env = os.environ if environment is None else environment
    base_url = env.get("ML_SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    if base_url != "http://127.0.0.1:8000":
        raise SmokeFailure("ML_SMOKE_BASE_URL must target the local API container")
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request = Request(
        f"{base_url}{path}",
        data=data,
        method="GET" if body is None else "POST",
        headers={
            "x-internal-secret": secret,
            "content-type": "application/json",
            "accept": "application/json",
            "user-agent": "gh-social-ml-production-smoke/1",
        },
    )
    return _read_json(request, timeout=timeout)


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SmokeFailure(f"health is missing required field {key}")
    return value.strip()


def _semantic_version(value: str, *, field: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", value)
    if match is None:
        raise SmokeFailure(f"health {field} is invalid")
    return tuple(int(part) for part in match.groups())


def check_readiness(
    environment: Mapping[str, str] | None = None,
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Routine container probe: no diagnostic refresh or recommendation work."""
    env = os.environ if environment is None else environment
    secret = env.get("INTERNAL_API_SECRET", "")
    if not re.fullmatch(r"[0-9a-f]{64}", secret):
        raise SmokeFailure("INTERNAL_API_SECRET must be 64 lowercase hexadecimal characters")
    payload = _request(
        "/api/v2/health/ready", secret=secret,
        timeout=timeout or float(env.get("ML_SMOKE_TIMEOUT_SECONDS", "10")),
        environment=env,
    )
    if payload.get("status") != "healthy" or payload.get("diagnostics_fresh") is not True:
        raise SmokeFailure("V2 readiness diagnostics are unhealthy or stale")
    age = payload.get("diagnostics_age_seconds")
    maximum = payload.get("diagnostics_max_age_seconds")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in (age, maximum)
    ) or not 0 <= age <= maximum:
        raise SmokeFailure("V2 readiness diagnostic freshness is invalid")
    _required_string(payload, "diagnostics_checked_at")
    if (
        payload.get("embedding_runtime_ready") is not True
        or payload.get("inference_warmup_ready") is not True
    ):
        raise SmokeFailure("inference warmup is not ready")
    return payload


def check_health(
    environment: Mapping[str, str] | None = None,
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    env = os.environ if environment is None else environment
    secret = env.get("INTERNAL_API_SECRET", "")
    if not re.fullmatch(r"[0-9a-f]{64}", secret):
        raise SmokeFailure(
            "INTERNAL_API_SECRET must be 64 lowercase hexadecimal characters"
        )
    request_timeout = timeout or float(env.get("ML_SMOKE_TIMEOUT_SECONDS", "10"))
    payload = _request(
        "/api/v2/health/deep",
        secret=secret,
        timeout=request_timeout,
        environment=env,
    )

    if payload.get("status") != "healthy":
        raise SmokeFailure("V2 health did not report healthy")
    if (
        payload.get("embedding_runtime_ready") is not True
        or payload.get("inference_warmup_ready") is not True
    ):
        raise SmokeFailure("inference warmup is not ready")
    if payload.get("database_required") is not False:
        raise SmokeFailure("V2 health did not confirm the database-free online boundary")
    if payload.get("feedback_consumer_active") is not True:
        raise SmokeFailure("V2 feedback consumer heartbeat is not active")
    expected_release = env.get("ML_RELEASE_ID", "")
    if not expected_release or payload.get("feedback_consumer_release_id") != expected_release:
        raise SmokeFailure("feedback consumer heartbeat is from another release")
    if payload.get("redis") != "healthy" or payload.get("feedback_healthy") is not True:
        raise SmokeFailure("Redis feedback stream exceeded a hard readiness threshold")
    if payload.get("qdrant") != "healthy":
        raise SmokeFailure("Qdrant did not report healthy")
    qdrant_version = _required_string(payload, "qdrant_server_version")
    if _semantic_version(qdrant_version, field="Qdrant server version") < (
        MINIMUM_QDRANT_SERVER_VERSION
    ):
        raise SmokeFailure("Qdrant server is below the conditional-write minimum")
    if payload.get("minimum_qdrant_server_version") != "1.18.0":
        raise SmokeFailure("health Qdrant minimum does not match the service contract")
    if payload.get("repository_collection_contract") not in {"healthy", "compatible"}:
        raise SmokeFailure("repository collection contract is not healthy")
    if _boolean(env.get("V2_USER_COLLECTION_REQUIRED", "true")):
        if payload.get("user_collection_contract") not in {"healthy", "compatible"}:
            raise SmokeFailure("user collection contract is not healthy")

    expected_model = env.get("EMBEDDING_MODEL")
    if _required_string(payload, "embedding_model") != expected_model:
        raise SmokeFailure("health embedding model does not match production config")
    expected_revision = env.get("EMBEDDING_MODEL_REVISION", "")
    if _required_string(payload, "embedding_model_revision") != expected_revision:
        raise SmokeFailure("health embedding revision does not match production config")
    configured_version = env.get("REPOSITORY_EMBEDDING_VERSION", "")
    if _required_string(payload, "configured_embedding_version") != configured_version:
        raise SmokeFailure("health embedding version does not match production config")
    configured_versions = {
        value.strip()
        for value in env.get("V2_COMPATIBLE_EMBEDDING_VERSIONS", "").split(",")
        if value.strip()
    }
    reported_versions = payload.get("compatible_embedding_versions")
    if not isinstance(reported_versions, list) or set(reported_versions) != configured_versions:
        raise SmokeFailure("health embedding allowlist does not match production config")
    if payload.get("required_content_version") != int(
        env.get("V2_REQUIRED_CONTENT_VERSION", "1")
    ):
        raise SmokeFailure("health content-version contract does not match production config")
    if payload.get("required_feature_spec_version") != env.get(
        "V2_REQUIRED_FEATURE_SPEC_VERSION"
    ):
        raise SmokeFailure("health feature contract does not match production config")
    if payload.get("allow_missing_embedding_revision") is not False:
        raise SmokeFailure("health permits repositories with missing embedding revisions")
    if (
        payload.get("serving_eligibility_version")
        != EXPECTED_SERVING_ELIGIBILITY_VERSION
    ):
        raise SmokeFailure("health serving-eligibility contract is incompatible")
    if (
        payload.get("serving_eligibility_evidence")
        != "validated_vector_at_atomic_upsert"
    ):
        raise SmokeFailure("health serving-eligibility evidence is missing")

    eligible = payload.get("eligible_repository_points")
    minimum = int(env.get("MIN_ELIGIBLE_REPOSITORIES", "1"))
    if not isinstance(eligible, int) or isinstance(eligible, bool) or eligible < minimum:
        raise SmokeFailure("eligible repository count is below the production minimum")
    configured_minimum = payload.get("minimum_eligible_repository_points")
    if configured_minimum != minimum:
        raise SmokeFailure("health corpus minimum does not match production config")

    ranker_enabled = _boolean(env.get("V2_HEAVY_RANKER_ENABLED", "true"))
    ranker_required = _boolean(env.get("V2_HEAVY_RANKER_REQUIRED", "true"))
    if payload.get("heavy_ranker_enabled") is not ranker_enabled:
        raise SmokeFailure("health heavy-ranker enablement does not match production config")
    if payload.get("heavy_ranker_required") is not ranker_required:
        raise SmokeFailure("health heavy-ranker requirement does not match production config")
    configured_traffic = float(env.get("V2_HEAVY_RANKER_TRAFFIC_PERCENT", "100"))
    reported_traffic = payload.get("heavy_ranker_traffic_percent")
    if isinstance(reported_traffic, bool) or not isinstance(reported_traffic, (int, float)):
        raise SmokeFailure("health heavy-ranker traffic percent is invalid")
    if not math.isclose(float(reported_traffic), configured_traffic, abs_tol=1e-9):
        raise SmokeFailure("health heavy-ranker traffic does not match production config")
    if ranker_required and not math.isclose(configured_traffic, 100.0, abs_tol=1e-9):
        raise SmokeFailure("required heavy ranker must receive 100 percent of traffic")
    if ranker_enabled:
        if (
            payload.get("heavy_ranker_ready") is not True
            or payload.get("heavy_ranker_warmup_ready") is not True
        ):
            raise SmokeFailure("enabled heavy ranker is not warmed and ready")
        if payload.get("heavy_ranker_production_qualified") is not True:
            raise SmokeFailure("enabled heavy ranker is not production-qualified")
    return payload


def check_recommendation(
    environment: Mapping[str, str] | None = None,
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    env = os.environ if environment is None else environment
    secret = env.get("INTERNAL_API_SECRET", "")
    user_id_text = env.get("ML_SMOKE_USER_ID", "")
    try:
        user_id = str(UUID(user_id_text))
    except ValueError:
        raise SmokeFailure("ML_SMOKE_USER_ID must be a canonical UUID") from None
    if user_id != user_id_text:
        raise SmokeFailure("ML_SMOKE_USER_ID must be a canonical lowercase UUID")
    limit = int(env.get("ML_SMOKE_RECOMMENDATION_LIMIT", "3"))
    minimum_items = int(env.get("ML_SMOKE_EXPECT_MIN_ITEMS", "1"))
    if not 1 <= minimum_items <= limit <= 15:
        raise SmokeFailure("smoke recommendation item bounds are invalid")
    generation_id = str(uuid5(NAMESPACE_URL, f"gh-social-ml-smoke:{user_id}"))
    request_timeout = timeout or float(env.get("ML_SMOKE_TIMEOUT_SECONDS", "10"))
    payload = _request(
        "/api/v2/recommendations/generate",
        secret=secret,
        timeout=request_timeout,
        environment=env,
        body={
            "schema_version": 2,
            "generation_id": generation_id,
            "user_id": user_id,
            "feed_version": 1,
            "limit": limit,
            "exclude_repo_ids": [],
            "context": {"cold_start": False, "locale": None},
        },
    )
    if payload.get("schema_version") != 2:
        raise SmokeFailure("recommendation response schema version is not 2")
    if payload.get("generation_id") != generation_id:
        raise SmokeFailure("recommendation response generation_id changed")
    if payload.get("user_id") != user_id:
        raise SmokeFailure("recommendation response user_id changed")
    if not _required_string(payload, "model_version"):
        raise SmokeFailure("recommendation response model version is missing")
    if payload.get("served_ranker") != "heavy":
        raise SmokeFailure("production recommendation did not use the V2 heavy ranker")
    if payload.get("retrieval_mode") != "personalized":
        raise SmokeFailure("smoke user did not use its personalized profile vector")
    if not isinstance(payload.get("ranker_applied"), bool):
        raise SmokeFailure("recommendation response ranker_applied is invalid")
    if not isinstance(payload.get("heavy_ranker_selected"), bool):
        raise SmokeFailure("recommendation response heavy_ranker_selected is invalid")
    if _boolean(env.get("V2_HEAVY_RANKER_REQUIRED", "true")) and (
        payload.get("served_ranker") != "heavy"
        or payload.get("ranker_applied") is not True
        or payload.get("heavy_ranker_selected") is not True
    ):
        raise SmokeFailure("required heavy ranker did not serve the recommendation")
    fallback_code = payload.get("fallback_code")
    if fallback_code is not None and (
        not isinstance(fallback_code, str) or not fallback_code
    ):
        raise SmokeFailure("recommendation response fallback_code is invalid")
    compatible = {
        value.strip()
        for value in env.get("V2_COMPATIBLE_EMBEDDING_VERSIONS", "").split(",")
        if value.strip()
    }
    raw_served_versions = payload.get("embedding_versions")
    if (
        not isinstance(raw_served_versions, list)
        or not raw_served_versions
        or any(
            not isinstance(value, str) or not value
            for value in raw_served_versions
        )
    ):
        raise SmokeFailure(
            "recommendation response did not report served embedding versions"
        )
    served_versions = set(raw_served_versions)
    if (
        len(served_versions) != len(raw_served_versions)
        or not served_versions <= compatible
    ):
        raise SmokeFailure("recommendation response used an incompatible embedding version")
    reported_embedding_version = _required_string(payload, "embedding_version")
    expected_report = (
        next(iter(served_versions))
        if len(served_versions) == 1
        else f"compatible-mixed:{','.join(sorted(served_versions))}"
    )
    if reported_embedding_version != expected_report:
        raise SmokeFailure(
            "recommendation response embedding-version report is inaccurate"
        )

    items = payload.get("items")
    if not isinstance(items, list) or not minimum_items <= len(items) <= limit:
        raise SmokeFailure("recommendation response returned an invalid item count")
    repo_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise SmokeFailure("recommendation response contains a malformed item")
        raw_repo_id = str(item.get("repo_id", ""))
        try:
            repo_id = str(UUID(raw_repo_id))
        except ValueError:
            raise SmokeFailure("recommendation response contains a non-canonical repo_id") from None
        if repo_id != raw_repo_id:
            raise SmokeFailure("recommendation response contains a non-canonical repo_id")
        if repo_id in repo_ids:
            raise SmokeFailure("recommendation response contains duplicate repositories")
        repo_ids.add(repo_id)
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise SmokeFailure("recommendation response contains a non-numeric score")
        if not math.isfinite(float(score)):
            raise SmokeFailure("recommendation response contains a non-finite score")
        if not isinstance(item.get("source"), str) or not item["source"]:
            raise SmokeFailure("recommendation response contains an invalid source")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bounded authenticated V2 smoke checks")
    probes = parser.add_mutually_exclusive_group()
    probes.add_argument(
        "--readiness-only", action="store_true",
        help="read the cached readiness snapshot only; never refresh diagnostics",
    )
    probes.add_argument(
        "--health-only",
        action="store_true",
        help="check readiness/contracts only; skip recommendation generation",
    )
    parser.add_argument("--timeout", type=float, help="per-request timeout in seconds")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.timeout is not None and not 1 <= args.timeout <= 30:
        print("Smoke check failed: timeout must be between 1 and 30 seconds", file=sys.stderr)
        return 2
    try:
        if args.readiness_only:
            health = check_readiness(timeout=args.timeout)
            print(json.dumps({
                "status": "healthy", "tier": "readiness",
                "diagnostics_checked_at": health["diagnostics_checked_at"],
            }, sort_keys=True))
            return 0
        health = check_health(timeout=args.timeout)
        item_count = None
        model_version = None
        if not args.health_only:
            recommendation = check_recommendation(timeout=args.timeout)
            item_count = len(recommendation["items"])
            model_version = recommendation["model_version"]
    except (SmokeFailure, ValueError) as exc:
        print(f"Smoke check failed: {exc}", file=sys.stderr)
        return 1

    summary = {
        "status": "healthy",
        "eligible_repository_points": health["eligible_repository_points"],
        "embedding_model": health["embedding_model"],
        "embedding_model_revision": health["embedding_model_revision"],
        "configured_embedding_version": health["configured_embedding_version"],
        "feedback_consumer_active": True,
    }
    if item_count is not None:
        summary["recommendation_items"] = item_count
        summary["model_version"] = model_version
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
