"""Bounded process-local correlation. No exporters, payloads or dynamic labels.

The authenticated HTTP boundary is the only caller allowed to adopt remote
context. Executor workers inherit context explicitly; durable queues do not.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
import math
import re
import secrets
import threading
import time
from uuid import UUID, uuid4

from utils.deadline import current_deadline

_TRACEPARENT = re.compile(r"00-([0-9a-f]{32})-([0-9a-f]{16})-(00|01)")
_OPERATIONS = frozenset({
    "search", "recommendation", "feedback", "embed", "refresh", "onboard",
    "health", "metrics", "other", "embedding",
})
_OUTCOMES = frozenset({"ok", "error", "timeout", "cancelled", "rejected"})
_ROUTES = {
    "/api/v2/search/repositories": "search",
    "/api/v2/recommendations/generate": "recommendation",
    "/api/v2/feedback/batch": "feedback",
    "/api/v2/repositories/embed": "embed",
    "/api/v2/repositories/refresh": "refresh",
    "/api/v2/users/onboard": "onboard",
    "/api/v2/health": "health",
    "/api/v2/health/live": "health",
    "/api/v2/health/ready": "health",
    "/api/v2/health/deep": "health",
    "/api/v2/metrics": "metrics",
}
_RING_ENTRIES = 256
_RING_BYTES = 128 * 1024
_SPAN_BYTES = 512
_lock = threading.Lock()
_ring: deque[bytes] = deque()
_ring_bytes = 0
# Include the default search, health, feedback, recommendation and refresh budgets.
_BUCKETS = (0.005, 0.025, 0.1, 0.5, 1.0, 4.0, 5.0, 8.0, 12.0, 30.0, 45.0, 120.0)
_STAGES = frozenset({"http", "span", "queue", "service", "late_work",
                     "model_lock", "encoding", "qdrant", "ranking", "response_build"})
_LIMIT = 2**63 - 1
_histograms: dict[tuple[str, str, str], list[float | int]] = {}


@dataclass(frozen=True, slots=True)
class Correlation:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    request_id: str
    sampled: bool


_current: ContextVar[Correlation | None] = ContextVar("ml_correlation", default=None)


def current_correlation() -> Correlation | None:
    return _current.get()


def operation_for_path(path: str) -> str:
    return _ROUTES.get(path, "other")


def authenticated_request_id(value: str | None) -> str:
    """Accept only canonical UUIDs after auth, never arbitrary caller labels."""
    if value is not None and len(value) == 36:
        try:
            parsed = UUID(value)
            if str(parsed) == value and parsed.int:
                return value
        except ValueError:
            pass
    return str(uuid4())


def _remote_parent(value: str | None) -> tuple[str, str, bool] | None:
    # Check length before regex/int work; no extension fields or whitespace.
    if value is None or len(value) != 55:
        return None
    match = _TRACEPARENT.fullmatch(value)
    if match is None:
        return None
    trace_id, span_id, flags = match.groups()
    if int(trace_id, 16) == 0 or int(span_id, 16) == 0:
        return None
    return trace_id, span_id, flags == "01"


def _id(byte_count: int) -> str:
    # Exclude the all-zero identifier even in the extremely unlikely RNG case.
    while True:
        value = secrets.token_hex(byte_count)
        if int(value, 16):
            return value


def _record(context: Correlation, operation: str, kind: str,
            outcome: str, elapsed: float, stage: str, remaining_deadline: float | None) -> None:
    global _ring_bytes
    if not context.sampled:
        return
    # All strings are internal enums or strictly bounded generated/validated IDs.
    record = {
        "trace_id": context.trace_id, "span_id": context.span_id,
        "parent_span_id": context.parent_span_id,
        "operation": operation, "kind": kind, "stage": stage, "outcome": outcome,
        "duration_seconds": round(max(0.0, elapsed), 6),
        "ended_at_unix_seconds": round(time.time(), 6),
    }
    if remaining_deadline is not None:
        record["remaining_deadline_seconds"] = remaining_deadline
    encoded = json.dumps(record, separators=(",", ":")).encode("ascii")
    if len(encoded) > _SPAN_BYTES:
        return
    with _lock:
        while _ring and (len(_ring) >= _RING_ENTRIES or
                         _ring_bytes + len(encoded) > _RING_BYTES):
            _ring_bytes -= len(_ring.popleft())
        if len(encoded) <= _RING_BYTES:
            _ring.append(encoded)
            _ring_bytes += len(encoded)


@dataclass(slots=True)
class Span:
    outcome: str = "ok"

    def set_http_status(self, status: int) -> None:
        self.outcome = ("timeout" if status == 504 else "rejected" if status == 429
                        else "error" if status >= 400 else "ok")


@contextmanager
def span_scope(operation: str, *, request_id: str | None = None,
               authenticated_traceparent: str | None = None, server: bool = False,
               stage: str = "span"):
    """Create a SERVER scope post-auth, or an INTERNAL child in a copied context."""
    operation = operation if operation in _OPERATIONS else "other"
    stage = "http" if server else stage if stage in _STAGES else "span"
    parent = _current.get() if not server else None
    remote = _remote_parent(authenticated_traceparent) if server else None
    if remote is not None:
        trace_id, parent_span_id, sampled = remote
    elif parent is not None:
        trace_id, parent_span_id, sampled = parent.trace_id, parent.span_id, parent.sampled
    else:
        trace_id, parent_span_id = _id(16), None
        sampled = int(trace_id[:2], 16) < 16  # 1/16 independently generated roots.
    context = Correlation(
        trace_id, _id(8), parent_span_id,
        authenticated_request_id(request_id) if server or parent is None else parent.request_id,
        sampled,
    )
    token = _current.set(context)
    remaining_deadline = None
    if context.sampled and (deadline := current_deadline()) is not None:
        # Read the existing clock without remaining(), which cancels on expiry.
        value = deadline.expires_at - deadline.clock()
        if math.isfinite(value):
            remaining_deadline = round(max(0.0, value), 6)
    started = time.monotonic()
    span = Span()
    try:
        yield span
    except BaseException as exc:
        # Names are not recorded. Cancellation covers asyncio/concurrent cancellation.
        span.outcome = _exception_outcome(exc)
        raise
    finally:
        _current.reset(token)
        outcome = span.outcome if span.outcome in _OUTCOMES else "error"
        elapsed = time.monotonic() - started
        observe_duration(operation, stage, outcome, elapsed)
        _record(context, operation, "SERVER" if server else "INTERNAL", outcome, elapsed, stage, remaining_deadline)


def diagnostic_snapshot() -> dict:
    """Only call from existing authenticated deep diagnostics, not readiness."""
    with _lock:
        records = list(_ring)
        size = _ring_bytes
        histograms = [
            {"operation": operation, "stage": stage, "outcome": outcome,
             "count": values[0], "sum_seconds": values[1],
             "cumulative_buckets": list(values[2:])}
            for (operation, stage, outcome), values in sorted(_histograms.items())
        ]
    return {
        "scope": "process_local_sampled_not_complete_traces",
        "root_sample_rate": "1/16", "max_spans": _RING_ENTRIES,
        "max_encoded_bytes": _RING_BYTES, "max_span_bytes": _SPAN_BYTES, "encoded_bytes": size,
        "spans": [json.loads(record) for record in records],
        "histogram_bounds_seconds": list(_BUCKETS), "histograms": histograms,
        "queue_scope": "in_process_submission_to_worker_start_not_durable_queue_age",
        "late_work_scope": "actual_running_work_after_caller_abandonment_or_shared_deadline",
        "http_scope": "middleware_entry_after_auth_to_response_object_not_body_completion",
        "durable_enqueue_context": "not_propagated",
        "client_app_timing": "not_measured",
    }


def _exception_outcome(exc: BaseException | None) -> str:
    if exc is None:
        return "ok"
    # Follow only the known service wrapper, not arbitrary chained application
    # errors. RuntimeError inheritance is retained and no API import cycle is
    # introduced. Only fixed class identity strings are compared, never emitted.
    seen: set[int] = set()
    for _ in range(8):
        if id(exc) in seen:
            break
        seen.add(id(exc))
        if isinstance(exc, TimeoutError):
            return "timeout"
        if type(exc).__name__ == "CancelledError":
            return "cancelled"
        if (type(exc).__module__, type(exc).__name__) != ("api.runtime", "ServiceDeadlineExceeded"):
            break
        exc = exc.__cause__
        if exc is None:
            break
    return "error"


def observe_duration(operation: str, stage: str, outcome: str, elapsed: float) -> None:
    """Sample-independent histograms with a finite Cartesian label space."""
    if stage not in _STAGES:
        return
    operation = operation if operation in _OPERATIONS else "other"
    outcome = outcome if outcome in _OUTCOMES else "error"
    if not math.isfinite(elapsed):
        return
    elapsed = max(0.0, elapsed)
    with _lock:
        values = _histograms.setdefault((operation, stage, outcome), [0] * (2 + len(_BUCKETS)))
        values[0] = min(_LIMIT, values[0] + 1)
        values[1] = min(_LIMIT, values[1] + elapsed)
        for index, bound in enumerate(_BUCKETS, start=2):
            if elapsed <= bound:
                values[index] = min(_LIMIT, values[index] + 1)


class WorkObservation:
    """One in-process submitted job, not persisted row age or durable queue time.

    start belongs to the actual worker; finish belongs to its native Future.
    abandon merely marks when a caller stopped waiting. It never finishes work.
    """
    def __init__(self, operation: str, *, shared_deadline: float | None = None):
        if shared_deadline is not None and not math.isfinite(shared_deadline):
            raise ValueError("shared observation deadline must be finite")
        self.operation = operation if operation in _OPERATIONS else "other"
        self._lock = threading.Lock()
        self._submitted = time.monotonic()
        # A process coordinator may own a shared expiry independently of any
        # HTTP waiter. Store its existing clock value, never start another timer.
        self._shared_deadline = shared_deadline
        self._started: float | None = None
        self._worker_stopped: float | None = None
        self._abandoned: float | None = None
        self._abandon_outcome = "cancelled"
        self._finished = False

    def start(self) -> None:
        with self._lock:
            if self._started is None and not self._finished:
                self._started = time.monotonic()
                observe_duration(self.operation, "queue", "ok", self._started - self._submitted)

    def worker_stopped(self) -> None:
        # Capture the physical exit before Future callbacks can be delayed.
        with self._lock:
            if self._worker_stopped is None:
                self._worker_stopped = time.monotonic()

    def abandon(self, outcome: str) -> None:
        with self._lock:
            if (not self._finished and self._worker_stopped is None
                    and self._abandoned is None):
                self._abandoned = time.monotonic()
                self._abandon_outcome = "timeout" if outcome == "timeout" else "cancelled"

    def complete_future(self, future: Future) -> None:
        """Attach only to the native concurrent Future, never an asyncio wrapper."""
        self.finish("cancelled" if future.cancelled() else _exception_outcome(future.exception()))

    def finish(self, outcome: str) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
            now = self._worker_stopped if self._worker_stopped is not None else time.monotonic()
            if self._started is None:
                observe_duration(self.operation, "queue", outcome, now - self._submitted)
            else:
                observe_duration(self.operation, "service", outcome, now - self._started)
                abandoned, abandon_outcome = self._abandoned, self._abandon_outcome
                if (self._shared_deadline is not None and now > self._shared_deadline
                        and (abandoned is None or self._shared_deadline < abandoned)):
                    abandoned, abandon_outcome = self._shared_deadline, "timeout"
                if abandoned is not None:
                    observe_duration(self.operation, "late_work", abandon_outcome,
                                     now - max(self._started, abandoned))


def render_stage_histograms() -> list[str]:
    """Export unsampled cumulative seconds; identical buckets can be fleet-summed."""
    with _lock:
        snapshot = [(key, tuple(values)) for key, values in sorted(_histograms.items())]
    lines = [
        "# HELP ml_stage_duration_seconds ML process-local stage duration, including failures.",
        "# TYPE ml_stage_duration_seconds histogram",
    ]
    for (operation, stage, outcome), values in snapshot:
        labels = f'operation="{operation}",stage="{stage}",outcome="{outcome}"'
        for bound, count in zip(_BUCKETS, values[2:]):
            lines.append(f'ml_stage_duration_seconds_bucket{{{labels},le="{bound:g}"}} {count}')
        lines.extend((
            f'ml_stage_duration_seconds_bucket{{{labels},le="+Inf"}} {values[0]}',
            f'ml_stage_duration_seconds_count{{{labels}}} {values[0]}',
            f'ml_stage_duration_seconds_sum{{{labels}}} {values[1]:.9f}',
        ))
    return lines


def reset_stage_histograms_for_tests() -> None:
    with _lock:
        _histograms.clear()
