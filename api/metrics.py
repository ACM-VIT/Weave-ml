"""Fixed-cardinality in-process metrics for the authenticated V2 boundary."""

from __future__ import annotations

from collections import Counter
import math
import threading
from typing import Mapping

from retrieval.metrics import render_candidate_metrics, reset_candidate_metrics_for_tests


_ROUTES = {
    "/api/v2/search/repositories",
    "/api/v2/recommendations/generate",
    "/api/v2/feedback/batch",
    "/api/v2/repositories/embed",
    "/api/v2/repositories/refresh",
    "/api/v2/users/onboard",
    "/api/v2/health",
    "/api/v2/health/live",
    "/api/v2/health/ready",
    "/api/v2/health/deep",
    "/api/v2/metrics",
}
_METHODS = {"GET", "POST"}
_RANKERS = {"hybrid", "heavy"}
_FALLBACKS = {
    "none",
    "HEAVY_RANKER_NOT_READY",
    "HEAVY_SCORING_FAILED",
    "INVALID_HEAVY_OUTPUT",
    "other",
}
_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
_COUNTER_LIMIT = 2**63 - 1
_lock = threading.Lock()
_requests: Counter[tuple[str, str, str]] = Counter()
_durations: Counter[tuple[str, str, float]] = Counter()
_duration_sum: Counter[tuple[str, str]] = Counter()
_recommendations: Counter[tuple[str, str]] = Counter()
_health_checks: Counter[tuple[str, str]] = Counter()
_health_duration_sum: Counter[str] = Counter()
_empty_recommendations = 0


def _bounded_increment(counter: Counter, key, amount: int | float = 1) -> None:
    counter[key] = min(_COUNTER_LIMIT, counter[key] + amount)


def _route(path: str) -> str:
    if path in _ROUTES:
        return path
    return "/api/v2/other" if path.startswith("/api/v2/") else "other"


def record_api_request(
    *,
    path: str,
    method: str,
    status_code: int,
    duration_seconds: float,
) -> None:
    route = _route(path)
    normalized_method = method if method in _METHODS else "OTHER"
    status_class = f"{status_code // 100}xx" if 100 <= status_code <= 599 else "other"
    duration = max(0.0, duration_seconds) if math.isfinite(duration_seconds) else 0.0
    with _lock:
        _bounded_increment(_requests, (normalized_method, route, status_class))
        _bounded_increment(_duration_sum, (normalized_method, route), duration)
        for bucket in _BUCKETS:
            if duration <= bucket:
                _bounded_increment(_durations, (normalized_method, route, bucket))


def record_recommendation(
    *, served_ranker: str, fallback_code: str | None, item_count: int
) -> None:
    global _empty_recommendations
    ranker = served_ranker if served_ranker in _RANKERS else "other"
    fallback = fallback_code or "none"
    if fallback not in _FALLBACKS:
        fallback = "other"
    with _lock:
        _bounded_increment(_recommendations, (ranker, fallback))
        if item_count == 0:
            _empty_recommendations = min(_COUNTER_LIMIT, _empty_recommendations + 1)


def record_health_dependency(dependency: str, outcome: str, duration: float) -> None:
    """Count logical diagnostic checks, not underlying client RPCs."""
    if dependency not in {"qdrant", "redis"} or outcome not in {"healthy", "failed"}:
        return
    with _lock:
        _bounded_increment(_health_checks, (dependency, outcome))
        _bounded_increment(_health_duration_sum, dependency, max(0.0, duration))


def _labels(**values: str) -> str:
    return "{" + ",".join(f'{key}="{value}"' for key, value in values.items()) + "}"


def render_prometheus(
    service_executors: Mapping[str, Mapping[str, int | float]],
) -> str:
    with _lock:
        requests = dict(_requests)
        durations = dict(_durations)
        duration_sum = dict(_duration_sum)
        recommendations = dict(_recommendations)
        empty_recommendations = _empty_recommendations
        health_checks = dict(_health_checks)
        health_duration_sum = dict(_health_duration_sum)

    lines = [
        "# HELP ml_api_requests_total Authenticated ML API requests.",
        "# TYPE ml_api_requests_total counter",
    ]
    for (method, route, status_class), value in sorted(requests.items()):
        lines.append(
            "ml_api_requests_total"
            + _labels(method=method, route=route, status_class=status_class)
            + f" {value}"
        )
    lines.extend(
        [
            "# HELP ml_api_request_duration_seconds ML API request latency.",
            "# TYPE ml_api_request_duration_seconds histogram",
        ]
    )
    # A request slower than the largest finite bucket still needs count/sum and
    # +Inf series; derive families from the independently recorded sum keys.
    duration_keys = sorted(duration_sum)
    for method, route in duration_keys:
        for bucket in _BUCKETS:
            value = durations.get((method, route, bucket), 0)
            lines.append(
                "ml_api_request_duration_seconds_bucket"
                + _labels(method=method, route=route, le=f"{bucket:g}")
                + f" {value}"
            )
        count = sum(
            value
            for (item_method, item_route, _status), value in requests.items()
            if item_method == method and item_route == route
        )
        lines.append(
            "ml_api_request_duration_seconds_bucket"
            + _labels(method=method, route=route, le="+Inf")
            + f" {count}"
        )
        lines.append(
            "ml_api_request_duration_seconds_count"
            + _labels(method=method, route=route)
            + f" {count}"
        )
        lines.append(
            "ml_api_request_duration_seconds_sum"
            + _labels(method=method, route=route)
            + f" {duration_sum.get((method, route), 0.0):.9f}"
        )
    lines.extend(
        [
            "# HELP ml_recommendations_total Recommendation ranker outcomes.",
            "# TYPE ml_recommendations_total counter",
        ]
    )
    for (ranker, fallback), value in sorted(recommendations.items()):
        lines.append(
            "ml_recommendations_total"
            + _labels(served_ranker=ranker, fallback_code=fallback)
            + f" {value}"
        )
    lines.extend(
        [
            "# HELP ml_empty_recommendations_total Recommendation responses with no items.",
            "# TYPE ml_empty_recommendations_total counter",
            f"ml_empty_recommendations_total {empty_recommendations}",
            "# HELP ml_service_executor_outstanding Current jobs admitted per reserved executor.",
            "# TYPE ml_service_executor_outstanding gauge",
            "# HELP ml_service_executor_capacity Maximum admitted jobs per reserved executor.",
            "# TYPE ml_service_executor_capacity gauge",
            "# HELP ml_service_executor_rejected_total Jobs rejected by executor admission.",
            "# TYPE ml_service_executor_rejected_total counter",
            "# HELP ml_service_executor_timed_out_total Jobs exceeding their response deadline.",
            "# TYPE ml_service_executor_timed_out_total counter",
        ]
    )
    for operation, values in sorted(service_executors.items()):
        lines.append(
            "ml_service_executor_outstanding"
            + _labels(operation=operation)
            + f" {int(values['outstanding'])}"
        )
        lines.append(
            "ml_service_executor_capacity"
            + _labels(operation=operation)
            + f" {int(values['capacity'])}"
        )
        lines.append(
            "ml_service_executor_rejected_total"
            + _labels(operation=operation)
            + f" {int(values['rejected_total'])}"
        )
        lines.append(
            "ml_service_executor_timed_out_total"
            + _labels(operation=operation)
            + f" {int(values['timed_out_total'])}"
        )
    lines.extend([
        "# HELP ml_health_dependency_checks_total Logical deep diagnostic checks, not client RPCs.",
        "# TYPE ml_health_dependency_checks_total counter",
        "# HELP ml_health_dependency_duration_seconds_sum Deep diagnostic dependency time.",
        "# TYPE ml_health_dependency_duration_seconds_sum counter",
    ])
    for (dependency, outcome), count in sorted(health_checks.items()):
        lines.append(
            "ml_health_dependency_checks_total"
            + _labels(dependency=dependency, outcome=outcome) + f" {count}"
        )
    for dependency, duration in sorted(health_duration_sum.items()):
        lines.append(
            "ml_health_dependency_duration_seconds_sum"
            + _labels(dependency=dependency) + f" {duration:.9f}"
        )
    lines.extend(render_candidate_metrics())
    return "\n".join(lines) + "\n"


def reset_metrics_for_tests() -> None:
    global _empty_recommendations
    reset_candidate_metrics_for_tests()
    with _lock:
        _requests.clear()
        _durations.clear()
        _duration_sum.clear()
        _recommendations.clear()
        _health_checks.clear()
        _health_duration_sum.clear()
        _empty_recommendations = 0
