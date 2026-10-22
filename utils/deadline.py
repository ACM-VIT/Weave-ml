"""Cooperative, process-local budgets for read-only online work.

Deadlines never cross the wire. Callers may send a relative duration; admission
creates one local monotonic expiry before any executor queue wait. Cancellation
only prevents new stages. It cannot interrupt a running Python/native call.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
import math
import threading
import time
from typing import Callable, Iterator


class CooperativeDeadlineExceeded(TimeoutError):
    """The request no longer has a budget for another stage."""


class DependencyBudgetUnavailable(TimeoutError):
    """One dependency call cannot fit a supported timeout in its budget."""


class RequestDeadline:
    def __init__(
        self, seconds: float, *, clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("request budget must be finite and positive")
        self.clock = clock
        self.expires_at = clock() + seconds
        self.cancelled = threading.Event()
        self._lock = threading.Lock()
        self._stopped_at: float | None = None
        self._stop_reason: str | None = None
        self.dependency_calls = 0
        self.prevented_calls = 0

    def remaining(self) -> float:
        remaining = self.expires_at - self.clock()
        if self.cancelled.is_set() or remaining <= 0:
            if remaining <= 0:
                self.cancel("timeout")
            raise CooperativeDeadlineExceeded("read-only request budget was exhausted")
        return remaining

    def cancel(self, reason: str = "cancelled") -> None:
        with self._lock:
            if self._stopped_at is None:
                self._stopped_at = min(self.clock(), self.expires_at)
                self._stop_reason = reason
            self.cancelled.set()

    def dependency_timeout(self, maximum: float) -> int:
        # Qdrant's public timeout contract is integer seconds >= 1. Rounding up
        # would exceed the remaining budget; do not launch a sub-second RPC.
        try:
            timeout = math.floor(min(maximum, self.remaining()))
            if timeout < 1:
                # A still-positive request budget can retain completed channels.
                # Refuse this RPC without cancelling the shared request deadline.
                raise DependencyBudgetUnavailable("insufficient budget for a Qdrant call")
        except (CooperativeDeadlineExceeded, DependencyBudgetUnavailable):
            with self._lock:
                self.prevented_calls += 1
            raise
        with self._lock:
            self.dependency_calls += 1
        return timeout

    def finish(self, operation: str) -> None:
        """Observe actual worker exit, never the end of its cancelled await."""
        now = self.clock()
        with self._lock:
            stopped_at = self._stopped_at
            reason = self._stop_reason
            if stopped_at is None and now >= self.expires_at:
                stopped_at, reason = self.expires_at, "timeout"
            calls, prevented = self.dependency_calls, self.prevented_calls
        with _metrics_lock:
            counters = _metrics.setdefault(operation, Counter())
            counters["dependency_calls_total"] += calls
            counters["dependency_calls_prevented_total"] += prevented
            if stopped_at is not None:
                reason = "cancelled" if reason == "cancelled" else "timeout"
                counters[f"{reason}_worker_exits_total"] += 1
                counters[f"{reason}_to_worker_exit_seconds_total"] += max(0.0, now - stopped_at)


_current: ContextVar[RequestDeadline | None] = ContextVar("read_only_deadline", default=None)
_metrics_lock = threading.Lock()
_metrics: dict[str, Counter] = {}


def current_deadline() -> RequestDeadline | None:
    return _current.get()


@contextmanager
def deadline_scope(deadline: RequestDeadline | None) -> Iterator[None]:
    token = _current.set(deadline)
    try:
        yield
    finally:
        _current.reset(token)


def check_deadline() -> None:
    if (deadline := current_deadline()) is not None:
        deadline.remaining()


def remaining_seconds(maximum: float) -> float:
    deadline = current_deadline()
    return min(maximum, deadline.remaining()) if deadline is not None else maximum


def qdrant_timeout(maximum: float = 10.0) -> dict[str, int]:
    deadline = current_deadline()
    return {"timeout": deadline.dependency_timeout(maximum)} if deadline is not None else {}


def deadline_status(operation: str) -> dict[str, int | float]:
    with _metrics_lock:
        counters = _metrics.get(operation, {})
        return {name: counters.get(name, 0) for name in (
            "dependency_calls_total", "dependency_calls_prevented_total",
            "timeout_worker_exits_total", "timeout_to_worker_exit_seconds_total",
            "cancelled_worker_exits_total", "cancelled_to_worker_exit_seconds_total",
        )}
