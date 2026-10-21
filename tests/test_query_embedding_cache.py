"""Query cache contracts: exact reuse, bounded resources and safe sharing."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from embedding.query_cache import CachedQueryEmbedder, QueryCacheCapacityError, QueryEmbeddingCache
from embedding.runtime import BoundedEmbeddingModel


def lookup(cache, query='private query', *, namespace=('model', 'revision', True, 'version', 'public'),
           compute=lambda: [1., 0.], expected_size=2, wait_seconds=1.):
    return cache.get_or_compute(query, namespace=namespace, expected_size=expected_size,
                                compute=compute, wait_seconds=wait_seconds)


def eventually(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(.001)


def test_concurrent_queries_encode_once_and_vectors_are_request_owned():
    cache = QueryEmbeddingCache()
    started, release = threading.Event(), threading.Event()
    source = [1., 0.]
    calls = []

    def encode():
        calls.append(1)
        started.set()
        assert release.wait(3)
        return source

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(lookup, cache, compute=encode) for _ in range(8)]
        try:
            assert started.wait(2)
            eventually(lambda: cache.snapshot()['waiters'] == 7)
            assert cache.snapshot()['flights'] == 1
        finally:
            release.set()
        vectors = [future.result(2) for future in futures]
    source[0] = 8.
    vectors[0][0] = 9.
    assert all(vector == [1., 0.] for vector in vectors[1:])
    assert lookup(cache) == [1., 0.]
    assert calls == [1]
    assert cache.snapshot()['waiters'] == cache.snapshot()['flights'] == 0
    # Both completed entries and active-flight keys must retain only digests.
    assert all(isinstance(key, bytes) and len(key) == 32 for key in cache._entries)
    assert b'private query' not in repr(cache._entries).encode()


@pytest.mark.parametrize('field', [
    'model_name', 'model_revision', 'normalization_version', 'embedding_version', 'security_scope',
])
def test_model_revision_normalization_embedding_version_and_scope_do_not_collide(field):
    class Encoder:
        def __init__(self):
            self.calls = 0

        def encode(self, texts, **kwargs):
            self.calls += 1
            return [[float(self.calls), 0.]]

    model = BoundedEmbeddingModel(Encoder(), max_concurrency=1)
    fields = dict(model_name='model', model_revision='revision',
                  normalization_version='normalization-v1', embedding_version='embedding-v1',
                  security_scope='tenant-a')
    original = CachedQueryEmbedder(model=model, cache=model.query_embedding_cache, expected_size=2, **fields)
    altered = CachedQueryEmbedder(model=model, cache=model.query_embedding_cache, expected_size=2,
                                  **{**fields, field: 'changed'})
    assert original.embed_text('same query') == [1., 0.]
    assert altered.embed_text('same query') == [2., 0.]
    assert original.embed_text('same query') == [1., 0.]
    assert model._model.calls == 2


def test_model_generations_do_not_share_cache_and_normalize_flag_is_namespaced():
    class Encoder:
        def __init__(self):
            self.calls = []

        def encode(self, texts, **kwargs):
            self.calls.append(kwargs['normalize_embeddings'])
            return [[float(len(self.calls)), 0.]]

    def adapter(model):
        return CachedQueryEmbedder(model=model, cache=model.query_embedding_cache,
                                   model_revision=model.query_embedding_revision,
                                   embedding_version='embedding-v1', expected_size=2)

    first, second = [BoundedEmbeddingModel(Encoder(), max_concurrency=1) for _ in range(2)]
    assert adapter(first).embed_text('same query') == [1., 0.]
    assert adapter(first).embed_text('same query') == [1., 0.]
    assert adapter(first).embed_text('same query', normalize=False) == [2., 0.]
    assert adapter(second).embed_text('same query') == [1., 0.]
    assert first._model.calls == [True, False]
    assert second._model.calls == [True]


@pytest.mark.parametrize('invalid', [[float('nan'), 0.], [float('inf'), 0.], [True, 0.], [1.], ['bad', 0.]])
def test_invalid_vectors_never_enter_cache_or_leave_flights(invalid):
    cache = QueryEmbeddingCache()
    with pytest.raises((ValueError, TypeError)):
        lookup(cache, compute=lambda: invalid)
    assert cache.snapshot()['entries'] == cache.snapshot()['flights'] == 0
    assert lookup(cache) == [1., 0.]
    assert cache.snapshot()['misses'] == 2


@pytest.mark.parametrize('error', [RuntimeError, TimeoutError])
def test_encoder_failure_wakes_waiters_and_next_call_retries(error):
    cache = QueryEmbeddingCache()
    started, release = threading.Event(), threading.Event()
    failure = error('synthetic encoder failure')

    def fail():
        started.set()
        assert release.wait(3)
        raise failure

    with ThreadPoolExecutor(max_workers=3) as pool:
        owner = pool.submit(lookup, cache, compute=fail)
        assert started.wait(2)
        waiters = [pool.submit(lookup, cache) for _ in range(2)]
        try:
            eventually(lambda: cache.snapshot()['waiters'] == 2)
        finally:
            release.set()
        for future in [owner, *waiters]:
            with pytest.raises(error, match='synthetic encoder failure') as caught:
                future.result(2)
            assert caught.value is failure
    assert cache.snapshot()['entries'] == cache.snapshot()['flights'] == cache.snapshot()['waiters'] == 0
    assert lookup(cache) == [1., 0.]


@pytest.mark.parametrize('legacy_future_timeout', [False, True])
def test_timed_out_waiter_does_not_cancel_or_retire_live_owner(monkeypatch, legacy_future_timeout):
    if legacy_future_timeout:
        import concurrent.futures
        from embedding import query_cache

        class LegacyFuturesTimeoutError(Exception):
            pass

        class LegacyFuture(concurrent.futures.Future):
            def result(self, timeout=None):
                try:
                    return super().result(timeout=timeout)
                except concurrent.futures.TimeoutError as exc:
                    # Python 3.10's futures timeout is not a built-in timeout.
                    raise LegacyFuturesTimeoutError() from exc

        monkeypatch.setattr(query_cache, 'Future', LegacyFuture)
        monkeypatch.setattr(query_cache, 'FutureTimeoutError', LegacyFuturesTimeoutError, raising=False)
    cache = QueryEmbeddingCache()
    started, release = threading.Event(), threading.Event()

    def encode():
        started.set()
        assert release.wait(3)
        return [1., 0.]

    with ThreadPoolExecutor(max_workers=1) as pool:
        owner = pool.submit(lookup, cache, compute=encode)
        try:
            assert started.wait(2)
            with pytest.raises(TimeoutError):
                lookup(cache, wait_seconds=.01)
            assert cache.snapshot()['flights'] == 1
            assert cache.snapshot()['waiters'] == 0
        finally:
            release.set()
        assert owner.result(2) == [1., 0.]
    assert lookup(cache) == [1., 0.]
    assert cache.snapshot()['misses'] == 1


def test_lru_ttl_byte_bounds_and_oversize_bypass():
    now = [0.]
    cache = QueryEmbeddingCache(max_entries=2, max_bytes=32, ttl_seconds=10, clock=lambda: now[0])
    for query in ['one', 'two', 'one', 'three']:
        lookup(cache, query)
    assert cache.snapshot()['hits'] == 1
    assert cache.snapshot()['entries'] == 2
    assert cache.snapshot()['payload_bytes'] == 32
    lookup(cache, 'two')
    assert cache.snapshot()['misses'] == 4  # two, not one, was least recently used
    now[0] = 9.
    lookup(cache, 'two')
    now[0] = 10.
    assert cache.snapshot()['entries'] == cache.snapshot()['payload_bytes'] == 0
    lookup(cache, 'two')
    assert cache.snapshot()['misses'] == 5  # hits do not extend TTL
    for i in range(100):
        lookup(cache, f'unique-{i}')
    assert cache.snapshot()['entries'] == 2
    assert cache.snapshot()['payload_bytes'] <= 32
    tiny = QueryEmbeddingCache(max_bytes=8)
    assert lookup(tiny) == lookup(tiny) == [1., 0.]
    assert tiny.snapshot()['entries'] == tiny.snapshot()['payload_bytes'] == 0
    assert tiny.snapshot()['misses'] == 2


def test_flight_and_waiter_capacity_is_bounded():
    cache = QueryEmbeddingCache(max_flights=1, max_waiters=1)
    started, release = threading.Event(), threading.Event()

    def encode():
        started.set()
        assert release.wait(3)
        return [1., 0.]

    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(lookup, cache, compute=encode)
        assert started.wait(2)
        waiter = pool.submit(lookup, cache)
        try:
            eventually(lambda: cache.snapshot()['waiters'] == 1)
            with pytest.raises(QueryCacheCapacityError, match='waiters'):
                lookup(cache)
            with pytest.raises(QueryCacheCapacityError, match='flights'):
                lookup(cache, 'different')
        finally:
            release.set()
        assert owner.result(2) == waiter.result(2) == [1., 0.]
    assert cache.snapshot()['waiters'] == cache.snapshot()['flights'] == 0


def test_dimension_namespace_and_byte_bound_are_independent_of_entry_bound():
    cache = QueryEmbeddingCache(max_entries=100, max_bytes=24)
    assert lookup(cache, expected_size=1, compute=lambda: [2.]) == [2.]
    assert lookup(cache, expected_size=2, compute=lambda: [3., 4.]) == [3., 4.]
    assert cache.snapshot()['misses'] == 2
    assert cache.snapshot()['entries'] == 2
    assert cache.snapshot()['payload_bytes'] == 24
    lookup(cache, 'third')
    assert cache.snapshot()['entries'] == 1
    assert cache.snapshot()['payload_bytes'] == 16


@pytest.mark.parametrize('field,value', [
    ('max_entries', 0), ('max_entries', True), ('max_bytes', -1),
    ('max_flights', 0), ('max_waiters', 0),
    ('ttl_seconds', 0), ('ttl_seconds', float('nan')), ('ttl_seconds', float('inf')),
])
def test_cache_rejects_unbounded_or_invalid_limits(field, value):
    with pytest.raises(ValueError):
        QueryEmbeddingCache(**{field: value})


def test_concurrent_cold_search_model_loads_share_one_query_cache(monkeypatch):
    from embedding import runtime

    loaded = []
    ready = threading.Barrier(4)

    class Encoder:
        def __init__(self):
            self.calls = 0

        def encode(self, texts, **kwargs):
            self.calls += 1
            return [[1., 0.]]

    def loader():
        encoder = Encoder()
        loaded.append(encoder)
        # A cold local model load blocks before the lru_cache can publish it.
        # Concurrent callers must not each build a model-owned query cache.
        time.sleep(.05)
        return encoder

    def search():
        ready.wait(timeout=2)
        model = runtime.shared_embedding_model()
        return CachedQueryEmbedder(
            model=model, cache=model.query_embedding_cache,
            model_revision=model.query_embedding_revision,
            embedding_version='embedding-v1', expected_size=2,
        ).embed_text('same cold query')

    runtime.reset_embedding_runtime_for_tests()
    monkeypatch.setattr(runtime, '_load_sentence_transformer', loader)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            vectors = [future.result(2) for future in [pool.submit(search) for _ in range(4)]]
        assert vectors == [[1., 0.]] * 4
        assert len(loaded) == 1
        assert loaded[0].calls == 1
        assert runtime.shared_embedding_model.cache_info().currsize == 1
        assert runtime.shared_embedding_model.cache_parameters() == {'maxsize': 1, 'typed': False}
        runtime.shared_embedding_model.cache_clear()
        assert runtime.shared_embedding_model.cache_info().currsize == 0
        runtime.shared_embedding_model()
        assert len(loaded) == 2
    finally:
        runtime.reset_embedding_runtime_for_tests()


@pytest.mark.parametrize('reorder', ['coordinates', 'requests'])
def test_profile_fingerprint_detects_vector_and_request_reordering(monkeypatch, reorder):
    from scripts.profile_query_embeddings import SyntheticEncoder, measure

    queries = ['first synthetic query', 'second synthetic query']
    baseline = measure(queries, cached=False)['vector_checksum']
    if reorder == 'coordinates':
        encode = SyntheticEncoder.encode

        def reversed_coordinates(self, texts, **kwargs):
            return [list(reversed(vector)) for vector in encode(self, texts, **kwargs)]

        monkeypatch.setattr(SyntheticEncoder, 'encode', reversed_coordinates)
    else:
        queries.reverse()
    reordered = measure(queries, cached=True)['vector_checksum']
    assert baseline != reordered
