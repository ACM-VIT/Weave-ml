"""Bounded, model-owned reuse of query vectors, never search results.

The request contract strips outer whitespace. This module hashes the exact
result without case folding, whitespace collapsing or other preprocessing.
Only digests and validated, immutable vector bytes survive completed calls.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
import time
from array import array
from collections import OrderedDict
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Callable

from config import REPOSITORY_EMBEDDING_MODEL
from embedding.embeddings import SentenceTransformerEmbedder
from embedding.vector_contract import validate_embedding_vector


class QueryCacheCapacityError(RuntimeError):
    """A bounded cache flight/waiter slot is unavailable."""


@dataclass(frozen=True, slots=True)
class _Vector:
    data: bytes
    expires_at: float
    compute_seconds: float

    def copy(self) -> list[float]:
        vector = array('d')
        vector.frombytes(self.data)
        return vector.tolist()


class QueryEmbeddingCache:
    """Thread-safe LRU/TTL cache with synchronous, bounded single-flight.

    Owners execute inside the already-admitted search worker, not a new pool.
    A waiter timing out never cancels its owner's future. Owner flights are
    removed only when the actual encoder exits, including exceptional exits.
    Byte bounds count packed vector payload; the entry/flight/waiter limits
    additionally bound metadata overhead. A snapshot estimates retained bytes.
    """

    def __init__(
        self, *, max_entries: int = 256, max_bytes: int = 1_048_576,
        ttl_seconds: float = 60., max_flights: int = 16, max_waiters: int = 16,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for name, value in (('max_entries', max_entries), ('max_bytes', max_bytes),
                            ('max_flights', max_flights), ('max_waiters', max_waiters)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError('ttl_seconds must be finite and positive')
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self.max_flights = max_flights
        self.max_waiters = max_waiters
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[bytes, _Vector] = OrderedDict()
        self._flights: dict[bytes, Future[_Vector]] = {}
        self._bytes = 0
        self._waiters = 0
        self._hits = self._misses = self._coalesced = 0
        self._compute_seconds = self._saved_seconds = 0.

    def _expire(self) -> None:
        now = self._clock()
        # LRU order is not expiry order: a hit must not extend TTL.
        for key, value in list(self._entries.items()):
            if value.expires_at <= now:
                self._bytes -= len(self._entries.pop(key).data)

    def get_or_compute(
        self, query: str, *, namespace: tuple, expected_size: int,
        compute: Callable[[], list[float]], wait_seconds: float = 4.,
    ) -> list[float]:
        if not math.isfinite(wait_seconds) or wait_seconds <= 0:
            raise ValueError('wait_seconds must be finite and positive')
        # Namespace serialization is unambiguous. Only the digest is retained,
        # including for flights; raw query text is never a key or metric label.
        digest = hashlib.sha256(json.dumps(
            [namespace, expected_size, query], ensure_ascii=True, separators=(',', ':'),
        ).encode('utf-8')).digest()
        with self._lock:
            self._expire()
            value = self._entries.get(digest)
            if value is not None:
                self._entries.move_to_end(digest)
                self._hits += 1
                self._saved_seconds += value.compute_seconds
                return value.copy()
            future = self._flights.get(digest)
            owner = future is None
            if owner:
                if len(self._flights) >= self.max_flights:
                    raise QueryCacheCapacityError('query embedding flights are at capacity')
                future = Future()
                self._flights[digest] = future
                self._misses += 1
            else:
                if self._waiters >= self.max_waiters:
                    raise QueryCacheCapacityError('query embedding waiters are at capacity')
                self._waiters += 1
                self._coalesced += 1

        if not owner:
            try:
                try:
                    value = future.result(timeout=wait_seconds)
                except FutureTimeoutError as exc:
                    # Before Python 3.11 this is not the built-in TimeoutError
                    # caught by the search route. Keep one boundary contract.
                    raise TimeoutError('query embedding wait deadline was exceeded') from exc
                with self._lock:
                    self._saved_seconds += value.compute_seconds
                return value.copy()
            finally:
                with self._lock:
                    self._waiters -= 1

        began = time.perf_counter()
        try:
            computed = compute()
            compute_seconds = time.perf_counter() - began
            vector = validate_embedding_vector(
                computed, expected_size=expected_size, field_name='search query embedding',
            )
            value = _Vector(
                array('d', vector).tobytes(), self._clock() + self.ttl_seconds, compute_seconds,
            )
        except BaseException as exc:
            with self._lock:
                self._compute_seconds += time.perf_counter() - began
                self._flights.pop(digest)
                future.set_exception(exc)
            raise
        with self._lock:
            self._compute_seconds += value.compute_seconds
            if len(value.data) <= self.max_bytes:
                while self._entries and (
                    len(self._entries) >= self.max_entries
                    or self._bytes + len(value.data) > self.max_bytes
                ):
                    _, evicted = self._entries.popitem(last=False)
                    self._bytes -= len(evicted.data)
                self._entries[digest] = value
                self._bytes += len(value.data)
            self._flights.pop(digest)
            future.set_result(value)
        return value.copy()

    def snapshot(self) -> dict[str, int | float]:
        """Fixed-cardinality diagnostics, never keys or query text.

        The retained-byte estimate includes Python containers, digest keys,
        records, numeric metadata and packed vectors, but not interpreter or
        allocator fragmentation. Use tracemalloc for workload peak memory.
        """
        with self._lock:
            self._expire()
            retained = (
                sys.getsizeof(self) + sys.getsizeof(self.__dict__) + sys.getsizeof(self._lock)
                + sys.getsizeof(self._entries) + sys.getsizeof(self._flights)
            )
            retained += sum(
                sys.getsizeof(key) + sys.getsizeof(value) + sys.getsizeof(value.data)
                + sys.getsizeof(value.expires_at) + sys.getsizeof(value.compute_seconds)
                for key, value in self._entries.items()
            )
            retained += sum(sys.getsizeof(key) + sys.getsizeof(future)
                            for key, future in self._flights.items())
            return {
                'hits': self._hits, 'misses': self._misses, 'coalesced': self._coalesced,
                'entries': len(self._entries), 'payload_bytes': self._bytes,
                'retained_bytes_estimate': retained, 'flights': len(self._flights),
                'waiters': self._waiters, 'embedding_compute_seconds': self._compute_seconds,
                'embedding_compute_seconds_avoided_estimate': self._saved_seconds,
            }


class CachedQueryEmbedder(SentenceTransformerEmbedder):
    """Query-only adapter; ingestion and onboarding never enter this cache.

    The caller supplies a model-owned cache, so replacing the loaded model
    cannot reuse old vectors even when its declared name/revision is unchanged.
    Current search preprocessing is public and tenant-independent. A future
    tenant-specific preprocessor MUST supply a separate security_scope and
    normalization_version before using the cache.
    """

    def __init__(
        self, *, model: object, cache: QueryEmbeddingCache, model_revision: str,
        embedding_version: str, expected_size: int, wait_seconds: float = 4.,
        model_name: str = REPOSITORY_EMBEDDING_MODEL,
        normalization_version: str = 'repository-search-contract-v2',
        security_scope: str = 'public-repository-search',
    ) -> None:
        super().__init__(model_name, model=model)
        self._cache = cache
        self._namespace = (self.model_name, model_revision, normalization_version,
                           embedding_version, security_scope)
        self._expected_size = expected_size
        self._wait_seconds = wait_seconds

    def embed_text(self, text: str, *, normalize: bool = True) -> list[float]:
        return self._cache.get_or_compute(
            text, namespace=(*self._namespace, normalize), expected_size=self._expected_size,
            compute=lambda: super(CachedQueryEmbedder, self).embed_text(text, normalize=normalize),
            wait_seconds=self._wait_seconds,
        )
