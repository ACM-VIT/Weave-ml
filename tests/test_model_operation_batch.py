"""Owned local Redis verification of bounded producer pipelines."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import tempfile
import shutil
import subprocess
import time

import pytest
import redis

from feedback.model_stream import OperationStream, StreamLimits, OperationCapacity, OperationBytesConflict
from feedback.model_wire import WireError, canonical_json
from tests.test_model_operations import policy, wire, begin_payload, profile_payload, feedback_payload, uid


@pytest.fixture
def local_redis():
    executable = shutil.which('redis-server')
    if executable is None:
        pytest.skip('local redis-server required')
    directory = tempfile.TemporaryDirectory(prefix='weave-redis-', dir='/tmp')
    socket = directory.name + '/redis.sock'
    process = subprocess.Popen([executable, '--port', '0', '--unixsocket', socket,
                               '--save', '', '--appendonly', 'no'], stdout=subprocess.DEVNULL)
    client = redis.Redis(unix_socket_path=socket, decode_responses=True, protocol=2)
    try:
        for _ in range(200):
            try:
                client.ping()
                break
            except redis.ConnectionError:
                time.sleep(.01)
        else:
            raise RuntimeError('owned Redis did not start')
        yield client
    finally:
        client.close()
        process.terminate()
        process.wait(timeout=5)
        directory.cleanup()


def stream(client, **limits):
    s = OperationStream(client, replace(StreamLimits('batch', 100, 1000, 131072, 10, 10), **limits),
                        {policy().sha256}, initialization_scope='batch-tests', max_guard_users=1000)
    s.ready_predecessor = lambda operation: False
    return s


def originals(count, same_user=False, start=1):
    return [wire(policy(), 'begin', i, begin_payload(i), user=uid(1 if same_user else i))
            for i in range(start, start + count)]


class CountedRedis:
    def __init__(self, client):
        self.client, self.commands, self.roundtrips, self.sizes = client, 0, 0, []
        self.failure = None

    def __getattr__(self, name):
        return getattr(self.client, name)

    def eval(self, *args):
        self.commands += 1
        self.roundtrips += 1
        return self.client.eval(*args)

    def pipeline(self, *, transaction):
        assert transaction is False
        owner = self
        class Pipeline:
            def __init__(self):
                self.pipe = owner.client.pipeline(transaction=False)
                self.count = 0
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.pipe.reset()
            def eval(self, *args):
                self.count += 1
                self.pipe.eval(*args)
            def execute(self, *, raise_on_error):
                assert raise_on_error is False
                owner.commands += self.count
                owner.roundtrips += 1
                owner.sizes.append(self.count)
                results = self.pipe.execute(raise_on_error=False)
                failure, owner.failure = owner.failure, None
                if failure == 'lost':
                    raise redis.ConnectionError('response lost after commands committed')
                if failure == 'command':
                    results[0] = redis.ResponseError('injected command error')
                if failure == 'short':
                    results.pop()
                return results
        return Pipeline()


def test_batch_parses_once_preserves_bytes_and_duplicate_hashes(local_redis, monkeypatch):
    from feedback import model_stream
    counted = CountedRedis(local_redis)
    s = stream(counted)
    raws = originals(10)
    parsed = []
    original = model_stream.parse_operation
    def parse(raw, **kwargs):
        parsed.append(raw)
        return original(raw, **kwargs)
    monkeypatch.setattr(model_stream, 'parse_operation', parse)
    first = s.admit_batch(raws)
    assert parsed == raws
    assert counted.commands == 10 and counted.roundtrips == 1
    assert [fields['operation_json'] for _, fields in local_redis.xrange(s.stream)] == raws
    second = s.admit_batch(raws)
    assert all(x['status'] == 'duplicate' for x in second)
    assert [x['source_envelope_sha256'] for x in first] == [x['source_envelope_sha256'] for x in second]
    changed = raws[0].replace('"witness_sha256":"b', '"witness_sha256":"c')
    with pytest.raises(OperationBytesConflict):
        s.admit_batch([changed, raws[1]])


def test_whole_batch_validation_and_hard_count_bound(local_redis):
    counted = CountedRedis(local_redis); s = stream(counted)
    for raws in [originals(2) + ['invalid'], [], originals(51)]:
        with pytest.raises(WireError):
            s.admit_batch(raws)
    assert counted.commands == 0


@pytest.mark.parametrize('failure', ['lost', 'command', 'short'])
def test_uncertain_or_individual_errors_require_exact_retry(local_redis, failure):
    counted = CountedRedis(local_redis); s = stream(counted)
    counted.failure = failure
    raws = originals(3)
    with pytest.raises(Exception):
        s.admit_batch(raws)
    assert local_redis.xlen(s.stream) == 3
    assert all(x['status'] == 'duplicate' for x in s.admit_batch(raws))
    assert local_redis.xlen(s.stream) == 3


def test_capacity_partial_admission_and_ready_slot(local_redis):
    s = stream(local_redis, max_pending=1)
    raws = originals(3)
    seen = []
    s.ready_predecessor = lambda op: seen.append(op.original_json) or True
    with pytest.raises(OperationCapacity):
        s.admit_batch(raws)
    assert local_redis.xlen(s.stream) == 2
    assert seen == raws[1:]
    with pytest.raises(OperationCapacity):
        s.admit_batch(raws)
    assert local_redis.xlen(s.stream) == 2


def test_same_user_recovery_is_resolved_before_next_input(local_redis):
    counted = CountedRedis(local_redis); s = stream(counted, max_pending=1)
    s.ready_predecessor = lambda op: op.header.operation_version == 2
    raws = originals(3, same_user=True)
    with pytest.raises(OperationCapacity):
        s.admit_batch(raws)
    assert [fields['operation_json'] for _, fields in local_redis.xrange(s.stream)] == raws[:2]
    assert counted.sizes == []


def test_profile_preflight_is_causal_barrier_with_mixed_kinds(local_redis):
    s = stream(local_redis); p = policy()
    raws = originals(2)
    profile = wire(p, 'profile', 3, profile_payload())
    feedback = wire(p, 'feedback', 4, feedback_payload())
    def preflight(operation):
        assert [fields['operation_json'] for _, fields in local_redis.xrange(s.stream)] == raws
        return None
    s.profile_preflight = preflight
    result = s.admit_batch([*raws, profile, feedback])
    assert len(result) == 4
    assert [fields['operation_json'] for _, fields in local_redis.xrange(s.stream)] == [*raws, profile, feedback]


def test_concurrent_producers_preserve_each_batch_order_and_atomic_capacity(local_redis):
    s = stream(local_redis, max_pending=30)
    def produce(index):
        try:
            return s.admit_batch(originals(10, start=index * 10 + 1))
        except OperationCapacity:
            return None
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(produce, range(6)))
    assert local_redis.xlen(s.stream) == 30
    import json
    versions = [int(json.loads(fields['operation_json'])['model_operation_version'])
                for _, fields in local_redis.xrange(s.stream)]
    for index in range(6):
        group = [v for v in versions if index * 10 < v <= (index + 1) * 10]
        assert group == sorted(group)


def test_wire_budget_splits_large_original_feedback(local_redis):
    import json
    counted = CountedRedis(local_redis); s = stream(counted)
    raws = []
    for i in range(1, 21):
        payload = feedback_payload(i)
        event = json.loads(payload['original_feedback_json'])
        event['user_id'] = uid(i)
        payload['original_feedback_json'] = ' ' * 100000 + canonical_json(event)
        raws.append(wire(policy(), 'feedback', i, payload, user=uid(i)))
    s.admit_batch(raws)
    assert len(counted.sizes) == 2
    assert sum(counted.sizes) == 20
    assert [fields['operation_json'] for _, fields in local_redis.xrange(s.stream)] == raws


def test_profile_withdrawal_between_preflight_and_admission_proves_ack(local_redis):
    from tests.test_model_operations import withdrawal
    s = stream(local_redis)
    raw = wire(policy(), 'profile', 1, profile_payload())
    attestation = withdrawal(policy(), raw)
    def preflight(operation):
        s.admit_withdrawal(attestation)
        return None
    s.profile_preflight = preflight
    ack = s.admit_batch([raw])[0]
    assert ack['status'] == 'withdrawal_accepted'
    assert ack['withdrawal_sha256'] == s.lookup_withdrawal(uid(1), 1).withdrawal_sha256
    assert local_redis.xlen(s.stream) == 1


def test_real_command_error_and_guard_capacity_fail_closed(local_redis):
    s = stream(local_redis)
    local_redis.set(s.identities, 'wrong-type')
    with pytest.raises(redis.ResponseError):
        s.admit_batch(originals(2))
    assert local_redis.xlen(s.stream) == 0
    local_redis.delete(s.identities)
    s.max_guard_users = 1
    with pytest.raises(OperationCapacity):
        s.admit_batch(originals(2))
    assert local_redis.hlen(s.guard_key) == 1
    assert local_redis.xlen(s.stream) == 1


def test_malformed_script_results_are_never_acknowledged(local_redis):
    s = stream(local_redis)
    for result in [True, 1.5, '1', None, 3]:
        with pytest.raises(RuntimeError):
            s._recover('', None, result, ())


def test_bound_stops_consuming_oversized_iterable(local_redis):
    s = stream(local_redis)
    consumed = []
    def generate():
        for i in range(100):
            consumed.append(i)
            yield 'invalid'
    with pytest.raises(WireError):
        s.admit_batch(generate())
    assert len(consumed) == 51


def test_route_validates_entire_batch_and_retries_lost_reply(local_redis, monkeypatch):
    from types import SimpleNamespace
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import feedback.model_runtime as provider
    from api.v2 import router
    counted = CountedRedis(local_redis); s = stream(counted)
    monkeypatch.setattr(provider, 'model_runtime', lambda: SimpleNamespace(stream=s))
    monkeypatch.setenv('INTERNAL_API_SECRET', 'fixture-secret')
    app = FastAPI(); app.include_router(router)
    client = TestClient(app)
    headers = {'x-internal-secret': 'fixture-secret'}
    raws = originals(2)
    def post(values):
        return client.post('/api/v2/model-operations', headers=headers,
                           json={'schema_version': 1, 'operations': [{'operation_json': raw} for raw in values]})
    assert post([raws[0], 'invalid']).status_code == 422
    assert counted.commands == 0
    counted.failure = 'lost'
    response = post(raws)
    assert response.status_code == 503
    assert 'operations' not in response.json()
    retry = post(raws)
    assert retry.status_code == 200
    assert [item['status'] for item in retry.json()['operations']] == ['duplicate', 'duplicate']


def test_concurrent_exact_retries_share_one_ordered_identity(local_redis):
    s = stream(local_redis)
    raws = originals(10, same_user=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: s.admit_batch(raws), range(6)))
    assert sum(item['status'] == 'accepted' for result in results for item in result) == 10
    assert sum(item['status'] == 'duplicate' for result in results for item in result) == 50
    assert [fields['operation_json'] for _, fields in local_redis.xrange(s.stream)] == raws
