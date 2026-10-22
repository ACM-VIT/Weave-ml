"""Process-wide, FIFO-admitted fan-out for read-only retrieval I/O.

Both queued and running child futures count against admission. Request workers
drain their children before exiting, including after a cancelled HTTP await.
"""

from __future__ import annotations

import atexit
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from contextvars import copy_context
import os
import threading
import time
from typing import Any, Callable, TypeVar

from utils.deadline import check_deadline, remaining_seconds


T = TypeVar("T")
_STAGES = ("profile", "semantic", "trend", "activity", "popularity", "freshness", "search")
_pool_lock = threading.Lock()
_pool: _RetrievalIO | None = None


def _settings() -> tuple[int, int]:
    values = []
    for name, default, maximum in (
        ("V2_RETRIEVAL_IO_WORKERS", 4, 32),
        ("V2_RETRIEVAL_IO_MAX_OUTSTANDING", 8, 128),
    ):
        try:
            value = int(os.getenv(name, str(default)).strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if not 1 <= value <= maximum:
            raise ValueError(f"{name} must be between 1 and {maximum}")
        values.append(value)
    workers, capacity = values
    if capacity < workers:
        raise ValueError("V2_RETRIEVAL_IO_MAX_OUTSTANDING must be at least V2_RETRIEVAL_IO_WORKERS")
    return workers, capacity


def validate_retrieval_runtime() -> None:
    _settings()


class _RetrievalIO:
    def __init__(self) -> None:
        self.workers, self.capacity = _settings()
        self.executor = ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix="ml-retrieval-io",
        )
        self.condition = threading.Condition()
        self.waiting: deque[object] = deque()
        self.outstanding = self.running = self.peak_running = 0
        self.closed = False
        self.stages = {name: Counter() for name in _STAGES}

    def submit(self, stage: str, function: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
        if stage not in _STAGES:
            raise ValueError("unknown retrieval stage")
        began = time.monotonic()
        ticket = object()
        with self.condition:
            self.waiting.append(ticket)
            try:
                while True:
                    check_deadline()
                    if self.closed:
                        raise RuntimeError("retrieval I/O executor is closed")
                    if self.waiting[0] is ticket and self.outstanding < self.capacity:
                        break
                    self.condition.wait(timeout=remaining_seconds(0.05))
            except BaseException:
                self.waiting.remove(ticket)
                self.condition.notify_all()
                raise
            self.waiting.popleft()
            self.outstanding += 1
            self.condition.notify_all()

        def invoke() -> T:
            check_deadline()  # An admitted child may expire in the pool queue.
            started = time.monotonic()
            with self.condition:
                self.running += 1
                self.peak_running = max(self.peak_running, self.running)
                self.stages[stage]["calls_total"] += 1
                self.stages[stage]["queue_seconds_total"] += started - began
            try:
                return function(*args, **kwargs)
            except BaseException:
                with self.condition:
                    self.stages[stage]["errors_total"] += 1
                raise
            finally:
                with self.condition:
                    self.running -= 1
                    self.stages[stage]["completed_total"] += 1
                    self.stages[stage]["duration_seconds_total"] += time.monotonic() - started

        try:
            # Never enter one Context concurrently from independent channels.
            context = copy_context()
            future = self.executor.submit(context.run, invoke)
        except BaseException:
            self._release()
            raise
        future.add_done_callback(lambda _future: self._release())
        return future

    def _release(self) -> None:
        with self.condition:
            self.outstanding -= 1
            self.condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self.condition:
            return {
                "workers": self.workers, "capacity": self.capacity,
                "outstanding": self.outstanding, "running": self.running,
                "peak_running": self.peak_running, "waiting_submitters": len(self.waiting),
                "stages": {name: dict(values) for name, values in self.stages.items()},
            }

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.executor.shutdown(wait=False, cancel_futures=True)


def _runtime() -> _RetrievalIO:
    global _pool
    with _pool_lock:
        if _pool is not None:
            with _pool.condition:
                if _pool.closed:
                    if _pool.outstanding:
                        raise RuntimeError("previous retrieval I/O workers are still settling")
                    _pool = None
        if _pool is None:
            _pool = _RetrievalIO()
        return _pool


class RetrievalWork:
    """One request owns each child until actual settlement, not await expiry."""

    def __init__(self) -> None:
        self.futures: list[Future] = []

    def __enter__(self) -> RetrievalWork:
        return self

    def submit(self, stage: str, function: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
        future = _runtime().submit(stage, function, *args, **kwargs)
        self.futures.append(future)
        return future

    @staticmethod
    def result(future: Future[T]) -> T:
        while True:
            check_deadline()
            try:
                return future.result(timeout=remaining_seconds(0.05))
            except TimeoutError as exc:
                if future.done() and not future.cancelled() and future.exception() is exc:
                    raise  # The child raised it; this was not a polling timeout.

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        if exc_type is not None:
            for future in self.futures:
                future.cancel()  # Running I/O is not cancellable by this call.
        # This is intentionally not deadline-bounded: outer request admission
        # must remain occupied until all already-running child workers exit.
        for future in self.futures:
            try:
                future.result()
            except BaseException:
                # result() also handles executor-shutdown cancellations; wait()
                # can await a notification for a removed queued work item.
                pass


def run_retrieval_io(stage: str, function: Callable[[], T]) -> T:
    with RetrievalWork() as work:
        return work.result(work.submit(stage, function))


def retrieval_runtime_status() -> dict[str, Any]:
    with _pool_lock:
        if _pool is not None:
            return _pool.snapshot()
    workers, capacity = _settings()
    return {"workers": workers, "capacity": capacity, "outstanding": 0, "running": 0,
            "peak_running": 0, "waiting_submitters": 0, "stages": {name: {} for name in _STAGES}}


def shutdown_retrieval_runtime() -> None:
    with _pool_lock:
        pool = _pool
    if pool is not None:
        pool.close()


atexit.register(shutdown_retrieval_runtime)
