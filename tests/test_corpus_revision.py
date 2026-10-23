"""Bounded authority protocol; real Redis fixtures own a network-isolated container."""
import json
import os
import subprocess
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest

from embedding.corpus_revision import (
    CorpusAuthorityUnavailable, CorpusRevisionAuthority, MaintenanceProof,
    authority_from_environment, operation_completed,
)


def proof():
    return MaintenanceProof('isolated-test-proof', old_writers_retired=True,
                            qdrant_operations_completed=True, former_primaries_fenced=True,
                            admin_boundary_managed=True)


def test_disabled_authority_factory_does_not_create_a_network_client(monkeypatch):
    import redis
    monkeypatch.delenv('CORPUS_AUTHORITY_ENABLED', raising=False)
    monkeypatch.setattr(redis, 'from_url', lambda *a, **k: pytest.fail('disabled must remain uncached'))
    assert authority_from_environment(collection='corpus', vector='repo_embedding') is None


def test_recovery_never_accepts_timeout_as_old_writer_retirement():
    with pytest.raises(ValueError, match='all'):
        MaintenanceProof('timeout-is-not-proof', old_writers_retired=False,
                         qdrant_operations_completed=True, former_primaries_fenced=True,
                         admin_boundary_managed=True)


@pytest.mark.parametrize('status,completed', [('completed', True), ('acknowledged', False), (None, False)])
def test_only_completed_update_status_can_release_writer(status, completed):
    assert operation_completed(SimpleNamespace(status=status)) is completed


class DockerRedis:
    def __init__(self, name, port=6379, credentials=()):
        self.name, self.port, self.credentials = name, port, credentials
    def command(self, *args):
        result = subprocess.run(['docker', 'exec', self.name, 'redis-cli', '--json', '-p', str(self.port), *self.credentials, *map(str,args)],
                                check=True, capture_output=True, text=True)
        value = json.loads(result.stdout)
        if isinstance(value, dict) and 'error' in value:
            raise RuntimeError(value['error'])
        return value
    def eval(self, source, numkeys, *args):
        return self.command('EVAL', source, numkeys, *args)


@pytest.fixture(scope='module')
def isolated_redis():
    if os.getenv('RUN_CORPUS_AUTHORITY_INTEGRATION') != '1':
        pytest.skip('explicit owned-container authority integration required')
    name = 'weave-authority-test-' + uuid4().hex[:12]
    subprocess.run(['docker','run','--detach','--name',name,'--network','none','redis:7.4.5-alpine',
                    'redis-server','--save','','--appendonly','no'], check=True, capture_output=True)
    client = DockerRedis(name)
    try:
        deadline = time.monotonic()+8
        while True:
            try:
                if client.command('PING') == 'PONG': break
            except (subprocess.CalledProcessError, json.JSONDecodeError): pass
            if time.monotonic() >= deadline: pytest.fail('isolated Redis failed bounded startup')
            time.sleep(.05)
        yield client
    finally:
        subprocess.run(['docker','rm','--force',name], check=False, capture_output=True)


@pytest.fixture
def authority(isolated_redis):
    return CorpusRevisionAuthority(isolated_redis, key='authority:' + uuid4().hex,
                                  scope='test-corpus', managed_boundary_attested=True, max_writers=2)


@pytest.mark.integration
class TestRealRedis:
    def test_no_automatic_bootstrap_and_two_writers_hold_dirty_until_actual_completion(self, authority):
        assert authority.read() is None
        with pytest.raises(CorpusAuthorityUnavailable): authority.begin()
        authority.bootstrap(proof())
        initial = authority.read()
        first, second = authority.begin(), authority.begin()
        assert first != second and authority.read() is None
        authority.finish(second, completed=True)
        assert authority.read() is None
        authority.finish(first, completed=True)
        assert authority.read() not in (None, initial)

    def test_ambiguous_write_never_expires_or_releases_dirty_token(self, authority):
        authority.bootstrap(proof())
        token = authority.begin()
        assert authority.finish(token, completed=False) is False
        assert authority.read() is None
        assert authority.status()['writers'] == 1
        assert authority.client.command('TTL', authority.key) == -1
        with pytest.raises(CorpusAuthorityUnavailable): authority.finish(uuid4().hex, completed=True)
        assert authority.status()['writers'] == 1

    def test_mutation_marks_dirty_before_dispatch_and_keeps_it_on_lost_ack(self, authority):
        authority.bootstrap(proof())
        calls = []
        def operation():
            calls.append(authority.read())
            raise TimeoutError('actual completion remains unknown')
        with pytest.raises(TimeoutError): authority.mutate(operation)
        assert calls == [None] and authority.status()['writers'] == 1
        assert authority.read() is None

    def test_acknowledged_status_is_not_completed(self, authority):
        authority.bootstrap(proof())
        with pytest.raises(CorpusAuthorityUnavailable):
            authority.mutate(lambda: SimpleNamespace(status='acknowledged'))
        assert authority.read() is None and authority.status()['writers'] == 1

    def test_successful_completed_mutation_changes_revision(self, authority):
        authority.bootstrap(proof())
        before = authority.read()
        result = authority.mutate(lambda: SimpleNamespace(status='completed'))
        assert result.status == 'completed' and authority.read() not in (None, before)

    def test_bounded_overflow_never_evicts_an_unknown_writer(self, authority):
        authority.bootstrap(proof())
        first, second = authority.begin(), authority.begin()
        with pytest.raises(CorpusAuthorityUnavailable): authority.begin()
        assert authority.status()['state'] == 'UNTRUSTED'
        assert authority.status()['writers'] == 2
        with pytest.raises(CorpusAuthorityUnavailable): authority.finish(first, completed=True)
        assert authority.read() is None

    def test_deleted_record_cannot_clean_unknown_writes_or_auto_bootstrap(self, authority):
        authority.bootstrap(proof())
        token = authority.begin()
        authority.client.command('DEL', authority.key)
        assert authority.read() is None
        with pytest.raises(CorpusAuthorityUnavailable): authority.finish(token, completed=True)
        with pytest.raises(CorpusAuthorityUnavailable): authority.begin()
        authority.bootstrap(proof())  # Explicit controlled maintenance, never a timeout.
        assert authority.read() is not None

    def test_begin_failure_prevents_operation_dispatch(self, authority):
        calls = []
        with pytest.raises(CorpusAuthorityUnavailable):
            authority.mutate(lambda: calls.append('must not run'))
        assert calls == []


    def test_real_scoped_acl_supports_exact_atomic_protocol(self, authority):
        user = 'authority_' + uuid4().hex[:8]
        authority.client.command('ACL','SETUSER',user,'on','>isolated-test-only',
                                 '~authority:*','+eval','+info','+hget','+hset')
        scoped = DockerRedis(authority.client.name, credentials=('--user',user,'--pass','isolated-test-only'))
        guarded = CorpusRevisionAuthority(scoped, key=authority.key, scope=authority.scope,
                                          managed_boundary_attested=True)
        guarded.bootstrap(proof())
        token = guarded.begin()
        assert guarded.read() is None
        guarded.finish(token, completed=True)
        assert guarded.read() is not None

    def test_unsupported_protocol_or_epoch_never_authorizes_reads_or_writes(self, authority):
        for field,value in [('v','unsupported'),('epoch','old-epoch'),('writers','not-json')]:
            authority.bootstrap(proof())
            authority.client.command('HSET',authority.key,field,value)
            assert authority.read() is None
            with pytest.raises(CorpusAuthorityUnavailable): authority.begin()

    def test_restoring_ready_after_actual_restart_requires_new_maintenance_epoch(self, authority):
        authority.bootstrap(proof())
        before = authority.read()
        raw = subprocess.run(['docker','exec',authority.client.name,'redis-cli','--raw','DUMP',authority.key],
                             check=True,capture_output=True).stdout[:-1]
        subprocess.run(['docker','restart',authority.client.name],check=True,capture_output=True)
        deadline = time.monotonic()+8
        while True:
            try:
                if authority.client.command('PING') == 'PONG': break
            except (subprocess.CalledProcessError,json.JSONDecodeError): pass
            if time.monotonic()>=deadline: pytest.fail('restart did not settle')
            time.sleep(.05)
        subprocess.run(['docker','exec','-i',authority.client.name,'redis-cli','-x','RESTORE',authority.key,'0'],
                       input=raw,check=True,capture_output=True)
        assert authority.read() is None
        with pytest.raises(CorpusAuthorityUnavailable): authority.begin()
        authority.bootstrap(proof())
        assert authority.read() not in (None,before)

    def test_lost_ack_after_actual_local_qdrant_write_keeps_dirty(self, authority):
        from qdrant_client import QdrantClient, models
        authority.bootstrap(proof())
        client=QdrantClient(':memory:')
        try:
            client.create_collection('isolated',vectors_config=models.VectorParams(size=2,distance=models.Distance.COSINE))
            def write_then_lose_ack():
                client.upsert('isolated',[models.PointStruct(id=1,vector=[1.,0.],payload={'version':1})],wait=True)
                raise TimeoutError('simulated lost client acknowledgment after actual storage write')
            with pytest.raises(TimeoutError): authority.mutate(write_then_lose_ack)
            assert client.retrieve('isolated',[1])[0].payload['version']==1
            assert authority.read() is None and authority.status()['writers']==1
        finally:
            client.close()


def test_unavailable_transport_bypasses_reader_and_prevents_writer_dispatch():
    def disconnected(*args): raise ConnectionError('isolated transport failure')
    authority=CorpusRevisionAuthority(SimpleNamespace(eval=disconnected),key='authority:unit',scope='unit',
                                      managed_boundary_attested=True)
    assert authority.read() is None
    calls=[]
    with pytest.raises(CorpusAuthorityUnavailable): authority.mutate(lambda: calls.append('unsafe'))
    assert calls==[]






@pytest.mark.integration
def test_actual_same_process_promotion_rejects_pre_promotion_replication_identity(authority):
    client=authority.client
    subprocess.run(['docker','exec',client.name,'redis-server','--port','6380','--daemonize','yes',
                    '--save','','--appendonly','no','--replicaof','127.0.0.1','6379'],check=True,capture_output=True)
    replica=DockerRedis(client.name,port=6380)
    deadline=time.monotonic()+10
    while True:
        try:
            info=replica.eval("return redis.call('INFO',ARGV[1])",0,'replication')
            if 'master_link_status:up' in info: break
        except (subprocess.CalledProcessError,json.JSONDecodeError): pass
        if time.monotonic()>=deadline: pytest.fail('isolated replication failed bounded startup')
        time.sleep(.05)
    authority.bootstrap(proof())
    assert client.command('WAIT',1,2000)==1  # Test setup only, never authority consistency.
    def identity(redis):
        fields={}
        for section in ['server','replication']:
            for line in redis.eval("return redis.call('INFO',ARGV[1])",0,section).splitlines():
                if ':' in line:
                    key,value=line.split(':',1)
                    fields[key]=value
        return fields
    before=identity(replica)
    replica.command('REPLICAOF','127.0.0.1',6399)
    authority.begin()  # Acknowledged DIRTY is deliberately absent from replica.
    assert replica.command('HGET',authority.key,'state')=='READY'
    replica.command('REPLICAOF','NO','ONE')
    after=identity(replica)
    assert before['run_id']==after['run_id']
    assert before['master_replid']!=after['master_replid'] and after['role']=='master'
    promoted=CorpusRevisionAuthority(replica,key=authority.key,scope=authority.scope,
                                     managed_boundary_attested=True)
    assert promoted.read() is None
    # Isolate the replication component: even this process's own old run_id
    # cannot authorize a READY record from its pre-promotion replication epoch.
    replica.command('HSET',authority.key,'epoch',before['run_id']+'|'+before['master_replid'])
    assert promoted.read() is None
    with pytest.raises(CorpusAuthorityUnavailable): promoted.begin()






@pytest.mark.parametrize('url',['http://localhost:6379','redis://localhost:6379',
    'redis://user:short@localhost:6379','redis://user:test-only-long-password@localhost:6379?decode_responses=1'])
def test_enabled_production_authority_rejects_unscoped_transport_before_client_creation(monkeypatch,url):
    import redis
    monkeypatch.setenv('CORPUS_AUTHORITY_ENABLED','true')
    monkeypatch.setenv('CORPUS_AUTHORITY_MANAGED_BOUNDARY','single-endpoint-fenced-admin-v1')
    monkeypatch.setenv('CORPUS_AUTHORITY_REDIS_URL',url)
    monkeypatch.setenv('APP_ENV','production')
    monkeypatch.setattr(redis,'from_url',lambda *a,**k: pytest.fail('invalid config must not connect'))
    with pytest.raises(ValueError,match='production ACL'):
        authority_from_environment(collection='corpus',vector='repo_embedding')
