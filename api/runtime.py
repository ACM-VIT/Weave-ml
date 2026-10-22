"""Reserved bounded executors for synchronous V2 service operations."""

from __future__ import annotations

import asyncio
import atexit
import math
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, TypeVar


from utils.deadline import (
    CooperativeDeadlineExceeded, RequestDeadline, check_deadline, current_deadline,
    deadline_scope, deadline_status,
)
from utils.runtime_lifecycle import runtime_admission


T = TypeVar("T")


class ServiceCapacityError(RuntimeError):
    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(f"{operation} executor is at capacity")


class ServiceDeadlineExceeded(RuntimeError):
    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(f"{operation} deadline was exceeded")


@dataclass(frozen=True, slots=True)
class _Spec:
    prefix: str
    default_workers: int
    default_capacity: int
    default_timeout: float
    maximum_workers: int
    maximum_capacity: int
    maximum_timeout: float


_SPECS = {
    # Short query jobs must not queue behind repository summary-provider waits.
    # The model itself remains shared and concurrency-bounded.
    "search": _Spec("V2_SEARCH", 1, 4, 4.0, 4, 16, 10.0),
    "feedback": _Spec("V2_FEEDBACK", 2, 8, 8.0, 8, 64, 30.0),
    "refresh": _Spec("V2_REFRESH", 2, 4, 45.0, 8, 32, 120.0),
    "health": _Spec("V2_HEALTH", 2, 4, 5.0, 4, 16, 30.0),
}
_instances: set["_BoundedExecutor"] = set()
_instances_lock = threading.Lock()


def _integer(name: str, default: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def _number(name: str, default: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value) or not 0.1 <= value <= maximum:
        raise ValueError(f"{name} must be between 0.1 and {maximum:g}")
    return value


def _settings(operation: str) -> tuple[int, int, float]:
    try:
        spec = _SPECS[operation]
    except KeyError as exc:
        raise ValueError("unknown service executor") from exc
    workers = _integer(
        f"{spec.prefix}_EXECUTOR_WORKERS",
        spec.default_workers,
        spec.maximum_workers,
    )
    capacity = _integer(
        f"{spec.prefix}_MAX_OUTSTANDING",
        spec.default_capacity,
        spec.maximum_capacity,
    )
    timeout = _number(
        f"{spec.prefix}_TIMEOUT_SECONDS",
        spec.default_timeout,
        spec.maximum_timeout,
    )
    if capacity < workers:
        raise ValueError(
            f"{spec.prefix}_MAX_OUTSTANDING must be at least "
            f"{spec.prefix}_EXECUTOR_WORKERS"
        )
    return workers, capacity, timeout


class _BoundedExecutor:
    def __init__(self, operation: str) -> None:
        workers, capacity, timeout = _settings(operation)
        self.operation = operation
        self.workers = workers
        self.capacity = capacity
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=f"ml-{operation}",
        )
        self.admission = threading.BoundedSemaphore(capacity)
        self._lock = threading.Lock()
        self._settled = threading.Condition(self._lock)
        self._outstanding = 0
        self._rejected = 0
        self._timed_out = 0
        with _instances_lock:
            _instances.add(self)

    def submit(
        self, function: Callable[..., T], *args: Any,
        deadline: RequestDeadline | None = None,
    ) -> Future[T]:
        """Retain admission until actual worker completion, including cancellation."""
        with runtime_admission():
            if not self.admission.acquire(blocking=False):
                with self._lock:
                    self._rejected += 1
                raise ServiceCapacityError(self.operation)
            with self._lock:
                self._outstanding += 1
            try:
                with deadline_scope(deadline):
                    context = copy_context()
                def invoke() -> T:
                    check_deadline()
                    return function(*args)
                future = self.executor.submit(context.run, invoke)
            except BaseException:
                self._release()
                raise
        def completed(_future) -> None:
            try:
                if deadline is not None and not _future.cancelled():
                    deadline.finish(self.operation)
            finally:
                self._release()
        future.add_done_callback(completed)
        return future

    async def run(self, function: Callable[..., T], *args: Any) -> T:
        # Durable feedback/refresh jobs deliberately keep their existing semantics.
        deadline = (current_deadline() or RequestDeadline(self.timeout)) if self.operation == "search" else None
        if deadline is not None:
            try:
                deadline.remaining()
            except CooperativeDeadlineExceeded as exc:
                raise ServiceDeadlineExceeded(self.operation) from exc
        future = self.submit(function, *args, deadline=deadline)
        try:
            return await asyncio.wait_for(
                asyncio.wrap_future(future),
                timeout=deadline.remaining() if deadline is not None else self.timeout,
            )
        except CooperativeDeadlineExceeded as exc:
            if deadline is not None:
                deadline.cancel("timeout")
            future.cancel()  # A pre-wait expiry must also remove queued work.
            self.record_timeout()
            raise ServiceDeadlineExceeded(self.operation) from exc
        except asyncio.CancelledError:
            if deadline is not None:
                deadline.cancel()
            future.cancel()
            raise
        except asyncio.TimeoutError as exc:
            if future.done() and not future.cancelled():
                return future.result()
            if deadline is not None:
                deadline.cancel("timeout")
            self.record_timeout()
            raise ServiceDeadlineExceeded(self.operation) from exc

    def record_timeout(self) -> None:
        with self._lock:
            self._timed_out += 1

    def _release(self) -> None:
        with self._settled:
            self._outstanding -= 1
            self.admission.release()
            self._settled.notify_all()

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "workers": self.workers,
                "capacity": self.capacity,
                "outstanding": self._outstanding,
                "rejected_total": self._rejected,
                "timed_out_total": self._timed_out,
                "timeout_seconds": self.timeout,
                **deadline_status(self.operation),
            }

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def drain(self) -> None:
        # Future.done() is visible before completion callbacks release slots.
        # Wait for that settlement, not a thread join: diagnostic retirement
        # may invoke cleanup on the health worker after its own release.
        if self.operation == "health":
            with self._settled:
                settled = self._settled.wait_for(
                    lambda: self._outstanding == 0,
                    timeout=min(self.timeout, 1.0),
                )
                if not settled:
                    # Retain this failed drain for the existing owner's retry;
                    # genuinely stuck work must not permit resource closure.
                    raise RuntimeError("health workers have not settled")
        self.executor.shutdown(wait=self.operation != "health", cancel_futures=True)


@lru_cache(maxsize=4)
def _runtime(operation: str) -> _BoundedExecutor:
    with runtime_admission():
        return _BoundedExecutor(operation)


async def run_service_job(
    operation: str,
    function: Callable[..., T],
    *args: Any,
) -> T:
    return await _runtime(operation).run(function, *args)


def submit_service_job(
    operation: str,
    function: Callable[..., T],
    *args: Any,
) -> Future[T]:
    """Submit bounded work for process-owned, single-flight diagnostics."""
    return _runtime(operation).submit(function, *args)


def record_service_job_timeout(operation: str) -> None:
    """Record a deadline owned by a process-level single-flight coordinator."""
    _runtime(operation).record_timeout()


def service_timeout_seconds(operation: str) -> float:
    """Return the immutable deadline configured for this admitted worker pool."""
    return _runtime(operation).timeout


def validate_service_runtime() -> None:
    for operation in _SPECS:
        _settings(operation)


def service_runtime_status() -> dict[str, dict[str, int | float]]:
    return {operation: _runtime(operation).snapshot() for operation in _SPECS}


def service_runtime_drain_actions():
    """Capture existing executors for the cleanup plan's pre-resource barrier."""
    with _instances_lock:
        instances = tuple(_instances)
    return [(f"service.{instance.operation}.drain", instance.drain) for instance in instances]


def service_runtime_cleanup_actions():
    """Capture every existing pool without allocating one during shutdown."""
    with _instances_lock:
        instances = tuple(_instances)

    def forget_instances():
        with _instances_lock:
            _instances.difference_update(instances)

    return [
        *[(f"service.{instance.operation}.close", instance.close) for instance in instances],
        ("service.instances.clear", forget_instances),
        ("service.cache.clear", _runtime.cache_clear),
    ]


def shutdown_service_runtime() -> None:
    # Close only pools that were actually created.  Shutdown must not allocate
    # new threads or clients as a side effect.
    with _instances_lock:
        instances = list(_instances)
        _instances.clear()
    for instance in instances:
        try:
            instance.close()
        except RuntimeError:
            pass
    _runtime.cache_clear()


atexit.register(shutdown_service_runtime)
