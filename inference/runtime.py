"""Bounded process runtime for recommendation work.

Qdrant calls are synchronous.  A dedicated executor and admission limit keep a
slow dependency from filling Starlette's shared thread pool or building an
unbounded request queue.  The response deadline is intentionally shorter than
the worst-case worker lifetime; timed-out work retains its slot until the
underlying function actually exits.
"""

from __future__ import annotations

import asyncio
import atexit
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Callable, TypeVar


T = TypeVar("T")
_counter_lock = threading.Lock()
_outstanding = 0
_rejected = 0
_timed_out = 0
_COUNTER_LIMIT = 2**63 - 1


class RecommendationCapacityError(RuntimeError):
    """Raised when all bounded recommendation slots are occupied."""


class RecommendationDeadlineExceeded(RuntimeError):
    """Raised when the client-facing recommendation deadline expires."""


def _positive_int(name: str, default: int, *, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def recommendation_timeout_seconds() -> float:
    raw = os.getenv("V2_RECOMMENDATION_TIMEOUT_SECONDS", "12").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("V2_RECOMMENDATION_TIMEOUT_SECONDS must be a number") from exc
    if not math.isfinite(value) or not 0.1 <= value <= 120:
        raise ValueError(
            "V2_RECOMMENDATION_TIMEOUT_SECONDS must be between 0.1 and 120"
        )
    return value


def recommendation_max_outstanding() -> int:
    capacity = _positive_int(
        "V2_RECOMMENDATION_MAX_OUTSTANDING", 8, maximum=64
    )
    workers = _positive_int("V2_RECOMMENDATION_EXECUTOR_WORKERS", 4, maximum=32)
    if capacity < workers:
        raise ValueError(
            "V2_RECOMMENDATION_MAX_OUTSTANDING must be at least "
            "V2_RECOMMENDATION_EXECUTOR_WORKERS"
        )
    return capacity


def validate_recommendation_runtime() -> None:
    recommendation_timeout_seconds()
    recommendation_max_outstanding()


@lru_cache(maxsize=1)
def recommendation_executor() -> ThreadPoolExecutor:
    workers = _positive_int("V2_RECOMMENDATION_EXECUTOR_WORKERS", 4, maximum=32)
    return ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="ml-recommendation",
    )


@lru_cache(maxsize=1)
def recommendation_admission() -> threading.BoundedSemaphore:
    return threading.BoundedSemaphore(recommendation_max_outstanding())


async def run_recommendation_job(function: Callable[..., T], *args: object) -> T:
    global _outstanding, _rejected, _timed_out
    admission = recommendation_admission()
    if not admission.acquire(blocking=False):
        with _counter_lock:
            _rejected = min(_COUNTER_LIMIT, _rejected + 1)
        raise RecommendationCapacityError(
            "the bounded recommendation executor is at capacity"
        )
    with _counter_lock:
        _outstanding += 1
    try:
        future = recommendation_executor().submit(function, *args)
    except BaseException:
        _release_recommendation(admission)
        raise

    future.add_done_callback(lambda _future: _release_recommendation(admission))
    try:
        return await asyncio.wait_for(
            asyncio.wrap_future(future),
            timeout=recommendation_timeout_seconds(),
        )
    except asyncio.TimeoutError as exc:
        # Built-in TimeoutError is also used by Redis/Qdrant clients.  If the
        # worker itself completed with that exception, preserve it for the API's
        # dependency classifier; only translate a still-running/cancelled wait
        # into the request-level deadline code.
        if future.done() and not future.cancelled():
            raise
        with _counter_lock:
            _timed_out = min(_COUNTER_LIMIT, _timed_out + 1)
        raise RecommendationDeadlineExceeded(
            "the bounded recommendation deadline was exceeded"
        ) from exc


def _release_recommendation(admission: threading.BoundedSemaphore) -> None:
    global _outstanding
    with _counter_lock:
        _outstanding -= 1
    admission.release()


def recommendation_runtime_status() -> dict[str, int | float]:
    with _counter_lock:
        return {
            "workers": _positive_int(
                "V2_RECOMMENDATION_EXECUTOR_WORKERS", 4, maximum=32
            ),
            "capacity": recommendation_max_outstanding(),
            "outstanding": _outstanding,
            "rejected_total": _rejected,
            "timed_out_total": _timed_out,
            "timeout_seconds": recommendation_timeout_seconds(),
        }


def recommendation_runtime_cleanup_actions():
    """Capture the current executor so a failed close remains retryable."""
    actions = []
    if recommendation_executor.cache_info().currsize:
        executor = recommendation_executor()
        actions.append((
            "recommendation.executor.close",
            lambda: executor.shutdown(wait=False, cancel_futures=True),
        ))
    return [
        *actions,
        ("recommendation.executor.cache.clear", recommendation_executor.cache_clear),
        ("recommendation.admission.cache.clear", recommendation_admission.cache_clear),
    ]


def shutdown_recommendation_runtime() -> None:
    if recommendation_executor.cache_info().currsize:
        recommendation_executor().shutdown(wait=False, cancel_futures=True)
        recommendation_executor.cache_clear()
    recommendation_admission.cache_clear()


atexit.register(shutdown_recommendation_runtime)
