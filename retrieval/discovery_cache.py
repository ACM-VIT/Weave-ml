"""Bounded reuse of public, payload-only discovery pools, never user state.

A serving integration must supply an authoritative revision covering *every*
corpus/eligibility write, deletion and collection replacement before activating
this adapter. Counts, collection configuration and process-local write counters
are not such an authority. Without it the retriever uses uncached reads.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Callable

from retrieval.runtime import RetrievalWork


@dataclass(frozen=True, slots=True)
class DiscoveryRevision:
    corpus: str
    eligibility: str

    def __post_init__(self) -> None:
        for value in (self.corpus, self.eligibility):
            if not isinstance(value, str) or not value.strip() or len(value) > 128:
                raise ValueError("discovery revision components must be 1 to 128 characters")


@dataclass(frozen=True, slots=True)
class _Pool:
    data: bytes
    created_at: float
    expires_at: float


@dataclass(slots=True)
class PoolLease:
    key: bytes
    future: Future[_Pool]
    owner: bool
    started_at: float
    waiting: bool = False


class SharedDiscoveryCache:
    """Per-retriever LRU/TTL, bounded in payload bytes, entries and flights.

    The existing admitted request worker owns refresh. Waiters do not cancel
    that owner. Owners release only after their RetrievalWork children settle.
    Immutable encoded bytes prevent callers from mutating another user's pool.
    Entry/flight/waiter bounds additionally bound Python metadata overhead.
    """

    def __init__(
        self, *, max_entries: int = 4, max_bytes: int = 2_097_152,
        ttl_seconds: float = 15.0, max_flights: int = 4, max_waiters: int = 32,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for name, value, upper in (
            ("max_entries", max_entries, 32), ("max_bytes", max_bytes, 8_388_608),
            ("max_flights", max_flights, 16), ("max_waiters", max_waiters, 128),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
                raise ValueError(f"{name} must be between 1 and {upper}")
        if not math.isfinite(ttl_seconds) or not 0 < ttl_seconds <= 30:
            raise ValueError("discovery TTL must be finite and at most 30 seconds")
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.ttl_seconds = ttl_seconds
        self.max_flights, self.max_waiters = max_flights, max_waiters
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[bytes, _Pool] = OrderedDict()
        self._flights: dict[bytes, Future[_Pool]] = {}
        self._bytes = self._waiters = 0
        self._hits = self._misses = self._coalesced = self._bypasses = 0

    def _expire(self) -> None:
        now = self._clock()
        for key, pool in list(self._entries.items()):
            if pool.expires_at <= now:
                self._bytes -= len(self._entries.pop(key).data)

    def reserve(self, namespace: tuple) -> PoolLease | None:
        key = hashlib.sha256(json.dumps(namespace, separators=(",", ":")).encode()).digest()
        with self._lock:
            self._expire()
            pool = self._entries.get(key)
            if pool is not None:
                self._entries.move_to_end(key)
                self._hits += 1
                ready: Future[_Pool] = Future()
                ready.set_result(pool)
                return PoolLease(key, ready, False, pool.created_at)
            future = self._flights.get(key)
            if future is not None:
                if self._waiters >= self.max_waiters:
                    self._bypasses += 1
                    return None
                self._waiters += 1
                self._coalesced += 1
                return PoolLease(key, future, False, self._clock(), True)
            if len(self._flights) >= self.max_flights:
                self._bypasses += 1
                return None
            future = Future()
            self._flights[key] = future
            self._misses += 1
            return PoolLease(key, future, True, self._clock())

    def result(self, lease: PoolLease) -> bytes:
        # Budget-aware polling does not cancel a shared owner on waiter expiry.
        pool = RetrievalWork.result(lease.future)
        if pool.expires_at <= self._clock():
            raise TimeoutError("shared discovery pool expired during refresh/wait")
        return pool.data

    def publish(self, lease: PoolLease, data: bytes) -> None:
        if not lease.owner or not isinstance(data, bytes):
            raise ValueError("only the owner may publish immutable pool bytes")
        with self._lock:
            if self._flights.get(lease.key) is not lease.future:
                raise RuntimeError("discovery refresh ownership was lost")
            pool = _Pool(data, lease.started_at, lease.started_at + self.ttl_seconds)
            if len(data) <= self.max_bytes and pool.expires_at > self._clock():
                while self._entries and (
                    len(self._entries) >= self.max_entries
                    or self._bytes + len(data) > self.max_bytes
                ):
                    _, evicted = self._entries.popitem(last=False)
                    self._bytes -= len(evicted.data)
                self._entries[lease.key] = pool
                self._bytes += len(data)
            self._flights.pop(lease.key)
        lease.future.set_result(pool)

    def fail(self, lease: PoolLease) -> None:
        with self._lock:
            if not lease.owner or self._flights.get(lease.key) is not lease.future:
                return
            self._flights.pop(lease.key)
        # Never expose another request's exception, identifiers or metadata.
        lease.future.set_exception(RuntimeError("shared discovery refresh unavailable"))

    def release(self, lease: PoolLease) -> None:
        with self._lock:
            if lease.waiting:
                lease.waiting = False
                self._waiters -= 1

    def lease_fresh(self, lease: PoolLease) -> bool:
        # A partial refresh can fail before publishing a pool Future. Its
        # per-request fallback still cannot extend the original refresh TTL.
        return self._clock() < lease.started_at + self.ttl_seconds

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            self._expire()
            now = self._clock()
            return {
                "entries": len(self._entries), "payload_bytes": self._bytes,
                "flights": len(self._flights), "waiters": self._waiters,
                "hits": self._hits, "misses": self._misses,
                "coalesced": self._coalesced, "bypasses": self._bypasses,
                "oldest_pool_age_seconds": max(
                    (now - pool.created_at for pool in self._entries.values()), default=0.0,
                ),
            }
