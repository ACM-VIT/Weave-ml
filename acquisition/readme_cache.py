"""Bounded, authorization-scoped immutable README artifacts and single flight."""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Callable, Hashable
from concurrent.futures import CancelledError, Future, TimeoutError as FutureTimeout
from dataclasses import dataclass
import threading
import time
from typing import Any, TypeVar

from .github_rate_limit import GitHubRequestBudget, GitHubRetryLater

T = TypeVar("T")


class SharedReadmeLoadUnavailable(RuntimeError):
    """Retryable shared-load failure, never another caller's cancellation."""


@dataclass
class _Flight:
    owner: GitHubRequestBudget
    future: Future[Any]


class BoundedReadmeCache:
    """An LRU byte/entry bound and bounded cooperative single-flight registry.

    A scheduler-owned producer retains its flight through retry deferrals.
    Followers defer too, rather than consuming all transport workers waiting
    for their producer. Finishing/cancelling the logical job cleans its flights.
    Producer cancellation is published as an ordinary retryable load failure;
    it never cancels followers belonging to another logical job/run. Existing
    followers do not re-elect a producer and reset the failed flight's budget.
    Keys must include the opaque authorization scope, never credential bytes.
    """

    def __init__(self, *, max_entries: int, max_bytes: int, max_inflight: int = 32) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.max_inflight = max_inflight
        self._lock = threading.Lock()
        self._values: OrderedDict[Hashable, tuple[Any, int]] = OrderedDict()
        self._flights: dict[Hashable, _Flight] = {}
        self._bytes = 0
        self._counters: Counter[str] = Counter()

    def get_or_load(
        self, key: Hashable, load: Callable[[], T], budget: GitHubRequestBudget,
        *, size: Callable[[T], int], cacheable: Callable[[T], bool] = lambda value: True,
    ) -> T:
        marker = (id(self), key)
        while True:
            budget.check()
            # Remember a follower's flight across deferrals so a producer's
            # failure is delivered to the whole flight, not silently retried
            # with a new producer/budget for every queued duplicate.
            remembered = budget.shared_flights.get(marker)
            if remembered is not None:
                if remembered.done():
                    return remembered.result()
                self._wait(budget, remembered)
                continue
            with self._lock:
                cached = self._values.get(key)
                if cached is not None:
                    self._values.move_to_end(key)
                    self._counters["hits"] += 1
                    self._counters["bytes_reused"] += cached[1]
                    return cached[0]
                flight = self._flights.get(key)
                if flight is None and len(self._flights) < self.max_inflight:
                    flight = _Flight(budget, Future())
                    self._flights[key] = flight
                    self._counters["misses"] += 1
                    budget.cleanups[marker] = lambda key=key, flight=flight: self._abandon(key, flight)
                elif flight is not None and flight.owner is not budget:
                    budget.shared_flights[marker] = flight.future
                    self._counters["coalesced"] += 1
            if flight is None:
                self._wait(budget)
                continue
            if flight.owner is not budget:
                self._wait(budget, flight.future)
                continue
            try:
                value = load()
                footprint = max(0, size(value))
                should_cache = cacheable(value) and footprint <= self.max_bytes
                budget.check()
            except GitHubRetryLater:
                # Same owner and attempt identity resume this producer later.
                raise
            except BaseException as exc:
                # Cancellation/termination belongs to the producer's logical
                # job, not to healthy followers in another acquisition run.
                # Deliver a retryable ordinary failure to that flight instead;
                # the producer still observes its original local exception.
                shared_error = exc
                if isinstance(exc, CancelledError) or not isinstance(exc, Exception):
                    shared_error = SharedReadmeLoadUnavailable("README shared producer ended before completion")
                self._complete(key, flight, error=shared_error)
                budget.cleanups.pop(marker, None)
                raise
            self._complete(key, flight, value=value, footprint=footprint, cacheable=should_cache)
            budget.cleanups.pop(marker, None)
            return value

    @staticmethod
    def _wait(budget: GitHubRequestBudget, future: Future[Any] | None = None) -> None:
        ready_at = time.monotonic() + 0.05
        budget.check_wait(ready_at)
        if budget.defer_retries:
            raise GitHubRetryLater(ready_at)
        if future is None:
            time.sleep(0.05)
        else:
            try:
                future.result(timeout=0.05)
            except FutureTimeout:
                pass
        budget.check()

    def _complete(
        self, key: Hashable, flight: _Flight, *, value: Any = None,
        footprint: int = 0, cacheable: bool = False, error: BaseException | None = None,
    ) -> None:
        with self._lock:
            if self._flights.get(key) is not flight:
                return
            self._flights.pop(key)
            if error is None and cacheable:
                while self._values and (
                    len(self._values) >= self.max_entries or self._bytes + footprint > self.max_bytes
                ):
                    _old_key, (_old_value, old_size) = self._values.popitem(last=False)
                    self._bytes -= old_size
                    self._counters["evictions"] += 1
                self._values[key] = (value, footprint)
                self._bytes += footprint
            if error is not None:
                self._counters["failures"] += 1
        # Complete outside the cache lock; arbitrary future callbacks must not
        # be able to deadlock cache admission or logical-job cleanup.
        if error is None:
            flight.future.set_result(value)
        else:
            flight.future.set_exception(error)

    def _abandon(self, key: Hashable, flight: _Flight) -> None:
        self._complete(
            key, flight,
            error=SharedReadmeLoadUnavailable("README shared producer ended before completion"),
        )

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                **self._counters, "entries": len(self._values),
                "bytes": self._bytes, "inflight": len(self._flights),
            }


# Whole canonical blobs are retained or not cached, never truncated to fit.
README_MANIFEST_CACHE = BoundedReadmeCache(max_entries=512, max_bytes=2 * 1024 * 1024)
README_BLOB_CACHE = BoundedReadmeCache(max_entries=256, max_bytes=64 * 1024 * 1024)
README_DOCUMENT_CACHE = BoundedReadmeCache(max_entries=256, max_bytes=32 * 1024 * 1024)
