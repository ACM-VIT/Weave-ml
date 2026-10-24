"""Owned local Redis exercises real Lua; no service URL or cleanup of user data."""
import shutil
import subprocess
import tempfile
import threading
import time

import pytest
import redis

from feedback.model_runtime import BudgetedRecords, ModelLaneUnavailable
from feedback.model_retention import VerifiedJournalTier
from feedback.model_stream import OperationCapacity
from feedback.model_journal import Journal, Limits, JournalConflict


class Records:
    def __init__(self):
        self.rows = {}

    def get(self, key):
        return self.rows.get(key)

    def put_once(self, key, raw):
        return self.rows.setdefault(key, raw)

    def delete_verified(self, key, raw):
        if key in self.rows:
            assert self.rows[key] == raw
            del self.rows[key]


@pytest.fixture
def r(tmp_path):
    executable = shutil.which('redis-server')
    if executable is None:
        pytest.skip('local redis-server required for real Lua retention tests')
    owned = tempfile.TemporaryDirectory(prefix='ret-', dir='/tmp')
    socket = owned.name + '/redis.sock'
    process = subprocess.Popen([executable, '--port', '0', '--unixsocket', socket,
        '--save', '', '--appendonly', 'no'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    client = redis.Redis(unix_socket_path=socket, socket_connect_timeout=.1, socket_timeout=.1, protocol=2)
    try:
        for _ in range(200):
            try:
                client.ping()
                break
            except redis.ConnectionError:
                time.sleep(.01)
        else:
            pytest.fail('owned Redis failed to start')
        yield client
    finally:
        client.close()
        process.terminate()
        process.wait(timeout=5)
        owned.cleanup()


def tier(r, *, hot=2, cold=10):
    archive = BudgetedRecords(Records(), r, key='cold', max_records=cold, max_bytes=10000000)
    return VerifiedJournalTier(Records(), r, key='hot', max_records=hot, max_bytes=1000000,
        archive=archive, signing_key=b'x'*32, decision_sha256='a'*64)


def test_small_hot_budget_continues_until_explicit_archive_ceiling(r):
    store = tier(r)
    before = store.metrics()
    costs = []
    for n in range(10):
        key, raw = f'operation:{n}', ('{"original":"検索", "n":%d}' % n).encode()
        assert store.put_once(key, raw) == raw
        costs.append(store.compact(key))
        assert store.get(key) == raw
        assert store.put_once(key, b'conflict') == raw
        assert store.metrics()['hot_records'] == 0
        assert len(store.records.rows) == 0
    assert sum(x['released_records'] for x in costs) == 10
    assert all(x['elapsed_seconds'] >= 0 for x in costs)
    after = store.metrics()
    assert before['hot_record_headroom'] == after['hot_record_headroom'] == 2
    assert after['archive_records'] == after['routing_records'] == 10
    assert after['archive_record_headroom'] == 0
    store.put_once('operation:next', b'next')
    with pytest.raises(OperationCapacity):
        store.compact('operation:next')
    assert store.get('operation:next') == b'next'
    assert store.metrics()['hot_records'] == 1


@pytest.mark.parametrize('boundary', ['archive_write', 'publication', 'delete', 'release'])
def test_crash_boundaries_resume_without_double_refund(r, monkeypatch, boundary):
    store = tier(r)
    store.put_once('key', b'exact original bytes')
    if boundary == 'archive_write':
        owner, name = store.archive.records, 'put_once'
    elif boundary == 'delete':
        owner, name = store.records, 'delete_verified'
    else:
        owner, name = r, 'eval'
    original = getattr(owner, name)
    def crash(*args, **kwargs):
        result = original(*args, **kwargs)
        if boundary not in {'publication', 'release'} or ('hset\', KEYS[3]' in args[0] if boundary == 'publication' else "'_count', count - 1" in args[0]):
            raise RuntimeError('lost response / worker stopped')
        return result
    monkeypatch.setattr(owner, name, crash)
    with pytest.raises(RuntimeError):
        store.compact('key')
    monkeypatch.setattr(owner, name, original)
    assert store.get('key') == b'exact original bytes'
    store.compact('key')
    assert store.compact('key')['released_records'] == 0
    assert store.metrics()['hot_records'] == store.metrics()['hot_bytes'] == 0
    assert store.metrics()['archive_records'] == 1


@pytest.mark.parametrize('corrupt', ['route', 'missing_route', 'archive', 'missing_archive', 'decision'])
def test_corrupt_or_incomplete_proof_never_admits_replacement(r, corrupt):
    store = tier(r)
    store.put_once('key', b'original')
    store.compact('key')
    if corrupt == 'route':
        r.hset(store.routes, 'key', b'bad')
    elif corrupt == 'missing_route':
        r.hdel(store.routes, 'key')
    elif corrupt == 'archive':
        store.archive.records.rows['key'] = b'changed'
    elif corrupt == 'missing_archive':
        del store.archive.records.rows['key']
    else:
        store.decision = 'b'*64
    with pytest.raises(ModelLaneUnavailable):
        store.get('key')
    with pytest.raises(ModelLaneUnavailable):
        store.put_once('key', b'replacement')
    assert store.metrics()['hot_records'] == 0
    assert store.records.get('key') is None


def test_inflight_recovery_cannot_reinsert_after_delete_and_refund(r, monkeypatch):
    store = tier(r)
    original = store.records.put_once
    blocked, release = threading.Event(), threading.Event()
    errors = []
    def paused(key, raw):
        if threading.current_thread().name == 'recovering':
            blocked.set()
            assert release.wait(5)
        return original(key, raw)
    monkeypatch.setattr(store.records, 'put_once', paused)
    def write():
        try:
            store.put_once('key', b'original')
        except BaseException as exc:
            errors.append(exc)
    worker = threading.Thread(target=write, name='recovering')
    worker.start()
    try:
        assert blocked.wait(5)
        with pytest.raises(ModelLaneUnavailable, match='reader'):
            store.compact('key')
        assert store.records.get('key') == b'original'
        assert store.metrics()['hot_records'] == 1
    finally:
        release.set()
        worker.join(5)
    assert not errors and not worker.is_alive()
    store.compact('key')
    assert store.records.get('key') is None
    assert store.metrics()['reader_records'] == store.metrics()['hot_records'] == 0


def test_interrupted_reader_fails_closed_without_expiring_io_fence(r):
    store = tier(r)
    store.put_once('key', b'original')
    r.hset(store.readers, 'key', 1)  # worker killed during Qdrant IO
    with pytest.raises(ModelLaneUnavailable, match='reader'):
        store.compact('key')
    assert store.metrics()['hot_records'] == 1
    assert store.get('key') == b'original'


def test_actual_journal_content_addresses_and_conflicts_use_archive(r):
    store = tier(r, hot=2)
    journal = Journal(store, Limits(100, 2048*1024, 1024*1024, 100))
    digest = journal.put('checkpoint', {'fixture': 'original exact state'})
    store.compact('blob:' + digest)
    assert journal.get(digest, 'checkpoint') == {'fixture': 'original exact state'}
    assert journal.put('checkpoint', {'fixture': 'original exact state'}) == digest
    assert store.metrics()['hot_records'] == 0
    store.archive.records.rows['blob:' + digest] = b'changed'
    with pytest.raises(ModelLaneUnavailable):
        journal.get(digest, 'checkpoint')


def test_qdrant_delete_adapter_verifies_bytes_and_missing_retry():
    from types import SimpleNamespace
    from feedback.model_journal import QdrantRecords
    class Client:
        def __init__(self):
            self.payload = None
            self.deletes = 0
        def retrieve(self, **kwargs):
            return [] if self.payload is None else [SimpleNamespace(payload=self.payload)]
        def upsert(self, **kwargs):
            self.payload = kwargs['points'][0].payload
        def delete(self, **kwargs):
            assert kwargs['wait'] is True
            self.deletes += 1
            self.payload = None
    from qdrant_client import models
    client = Client()
    records = QdrantRecords(client, models, collection='fixture', record_bytes=1024)
    records.put_once('key', b'original')
    with pytest.raises(JournalConflict):
        records.delete_verified('key', b'changed')
    assert client.deletes == 0
    records.delete_verified('key', b'original')
    records.delete_verified('key', b'original')
    assert client.deletes == 1


def test_same_physical_collection_and_overlapping_namespaces_are_refused(r):
    hot, cold = Records(), Records()
    hot.collection = cold.collection = 'same-physical-collection'
    archive = BudgetedRecords(cold, r, key='cold', max_records=10, max_bytes=1000)
    with pytest.raises(ValueError, match='dedicated'):
        VerifiedJournalTier(hot, r, key='hot', max_records=2, max_bytes=1000,
            archive=archive, signing_key=b'x'*32, decision_sha256='a'*64)
    del hot.collection
    archive.key = 'hot:candidates'
    with pytest.raises(ValueError, match='dedicated'):
        VerifiedJournalTier(hot, r, key='hot', max_records=2, max_bytes=1000,
            archive=archive, signing_key=b'x'*32, decision_sha256='a'*64)


def test_local_qdrant_actual_runtime_reversal_withdrawal_and_replay(r):
    from dataclasses import replace
    from qdrant_client import QdrantClient, models
    from feedback.model_runtime import ModelRuntime
    from feedback.v2_settings import V2FeedbackSettings
    from tests.test_model_operations import policy, runtime_config, uid, basis
    from tests.test_model_operations_integration import (
        test_explicit_verified_tier_preserves_reversal_withdrawal_and_rebuild as scenario)
    from config import (REPOSITORY_EMBEDDING_VERSION, REPOSITORY_FEATURE_SPEC_VERSION,
        REPOSITORY_EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION)
    from embedding.vector_contract import REPOSITORY_SERVING_ELIGIBILITY_FIELD, REPOSITORY_SERVING_ELIGIBILITY_VERSION
    q = QdrantClient(':memory:')
    try:
        q.create_collection('fixture_model_users', vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
        q.create_collection('fixture_model_journal', vectors_config={})
        q.create_collection('fixture_model_repositories', vectors_config={'repo_embedding': models.VectorParams(size=384, distance=models.Distance.COSINE)})
        settings = replace(V2FeedbackSettings.from_env(), user_collection='fixture_model_users', repository_collection='fixture_model_repositories')
        q.upsert(settings.repository_collection, points=[models.PointStruct(id=uid(100), vector={'repo_embedding': basis(1)}, payload={
            'repo_id': uid(100), REPOSITORY_SERVING_ELIGIBILITY_FIELD: REPOSITORY_SERVING_ELIGIBILITY_VERSION,
            'content_version': 1, 'feature_version': 1, 'embedding_model': REPOSITORY_EMBEDDING_MODEL,
            'embedding_model_revision': EMBEDDING_MODEL_REVISION, 'embedding_version': REPOSITORY_EMBEDDING_VERSION,
            'embedding_dim': 384, 'feature_spec_version': REPOSITORY_FEATURE_SPEC_VERSION})], wait=True)
        runtime = ModelRuntime(runtime_config(policy()), redis=r, qdrant=q,
            encoder=lambda _: basis(0), settings=settings)
        scenario(runtime)
    finally:
        q.close()


def test_head_bound_status_index_remains_readable_after_verified_archive_routing(r):
    from tests.test_model_operations import status_history
    history=status_history(40);store=tier(r,hot=2,cold=500)
    rows=list(history.records.rows.items())
    assert any(b'"kind":"status_index"' in raw for _,raw in rows)
    for key,raw in rows:
        assert store.put_once(key,raw)==raw
        store.compact(key)
    history.journal.records=store
    for version in (1,16,17,32,39,40):
        proof=history.coordinator.original_status(history.users.snapshot,history.headers[version])
        assert proof['status']=='applied' and proof['operation_head']==history.snapshots[version].head
    assert store.metrics()['hot_records']==0


def test_timed_out_write_retains_fence_until_remote_request_is_quiescent(r, monkeypatch):
    store = tier(r)
    original = store.records.put_once
    release, arrived = threading.Event(), threading.Event()
    errors = []
    def delayed():
        try:
            assert release.wait(5)
            original('key', b'original')
            arrived.set()
        except BaseException as exc:
            errors.append(exc)
    worker = threading.Thread(target=delayed)
    def timeout(key, raw):
        worker.start()
        raise TimeoutError('request still in flight after client timeout')
    monkeypatch.setattr(store.records, 'put_once', timeout)
    try:
        with pytest.raises(TimeoutError):
            store.put_once('key', b'original')
        monkeypatch.setattr(store.records, 'put_once', original)
        # A second worker recovers and refunds the temporary candidate copy.
        assert store.get('key') == b'original'
        assert r.hget(store.candidates, 'key') is None
        with pytest.raises(ModelLaneUnavailable, match='reader'):
            store.compact('key')
        assert store.metrics()['hot_records'] == 1
        assert r.hget(store.routes, 'key') is None
        release.set()
        worker.join(5)
        assert arrived.is_set() and not errors and not worker.is_alive()
        # Even observing this write does not automatically prove all remote IO
        # quiescent. The abandoned fence requires controlled operator recovery.
        with pytest.raises(ModelLaneUnavailable, match='reader'):
            store.compact('key')
        assert store.records.get('key') == b'original'
        assert store.metrics()['hot_records'] == 1
    finally:
        release.set()
        if worker.ident is not None:
            worker.join(5)


@pytest.mark.parametrize('routed', [False, True])
@pytest.mark.parametrize('loss', ['identity', 'ledger'])
def test_missing_archive_reservation_blocks_route_and_hot_deletion(r, routed, loss):
    store = tier(r)
    store.put_once('key', b'original')
    if routed:
        store.compact('key')
    else:
        store.archive.put_once('key', b'original')
    if loss == 'identity':
        r.hdel(store.archive.key, 'key')
    else:
        r.delete(store.archive.key)
    if routed:
        with pytest.raises(ModelLaneUnavailable, match='reservation'):
            store.get('key')
        with pytest.raises(ModelLaneUnavailable, match='reservation'):
            store.put_once('key', b'replacement')
    with pytest.raises(ModelLaneUnavailable, match='reservation'):
        store.compact('key')
    assert store.records.get('key') == (None if routed else b'original')
    assert store.metrics()['hot_records'] == (0 if routed else 1)
