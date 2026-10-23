"""Measurement correctness only; no cache freshness policy is installed."""
import uuid
from contextlib import closing

import pytest
from qdrant_client import QdrantClient, models

from scripts.measure_profile_fetch import (
    Opportunities, VERSION_FIELDS, WireSpans, profile_payload, run_scenario,
)
from retrieval.v2_retriever import QdrantV2Retriever


@pytest.fixture
def reader(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("V2_ALLOW_MISSING_EMBEDDING_REVISION", "false")
    with closing(QdrantClient(":memory:")) as client:
        client.create_collection("user_profiles", vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
        from scripts.measure_profile_fetch import fixture_reader
        yield fixture_reader(client)


def test_identical_profiles_are_separated_by_user():
    analyzer = Opportunities()
    payload = profile_payload()
    for user in ("a", "b", "a", "b"):
        analyzer.observe(user, [1.] + [0.] * 383, payload, payload)
    assert analyzer.content_repeats == analyzer.version_compatible_repeats == 2
    assert analyzer.reads == 4


@pytest.mark.parametrize("version", [None, -1, True, "2"])
def test_missing_or_invalid_versions_cannot_authorize_opportunities(version):
    analyzer = Opportunities()
    oracle = profile_payload() | {"last_feedback_version": version}
    for _ in range(2):
        analyzer.observe("a", [1.], {}, oracle)
    assert analyzer.content_repeats == 1
    assert analyzer.version_compatible_repeats == 0


def test_version_only_change_is_not_a_compatible_repeat():
    analyzer = Opportunities()
    oracle = profile_payload()
    analyzer.observe("a", [1.], {}, oracle)
    analyzer.observe("a", [1.], {}, oracle | {"profile_version": 2})
    assert analyzer.content_repeats == 1
    assert analyzer.version_compatible_repeats == 0


def test_invalid_read_breaks_previous_observation():
    analyzer = Opportunities()
    for vector in ([1.], None, [1.]):
        analyzer.observe("a", vector, {}, profile_payload())
    assert analyzer.valid == 2
    assert analyzer.content_repeats == 0


def test_actual_projected_reader_observes_atomic_update_without_ledger(reader):
    uid = str(uuid.uuid4())
    first = [1.] + [0.] * 383
    second = [0., 1.] + [0.] * 382
    reader.client.upsert("user_profiles", points=[models.PointStruct(id=uid, vector=first, payload=profile_payload())], wait=True)
    vector, payload = reader._user_profile(uid)
    assert vector == first
    assert "fixture_ledger" not in payload
    assert not set(VERSION_FIELDS).intersection(payload)
    changed = profile_payload() | {"skills": ["Rust"], "profile_version": 2}
    reader.client.upsert("user_profiles", points=[models.PointStruct(id=uid, vector=second, payload=changed)], wait=True)
    vector, payload = reader._user_profile(uid)
    assert vector == second
    assert payload["skills"] == ["Rust"]
    assert "profile_version" not in payload


@pytest.mark.parametrize("bad", [{"embedding_model_revision": "other"}, {"embedding_dim": 1024}])
def test_incompatible_actual_profile_is_not_a_valid_measurement(reader, bad):
    uid = str(uuid.uuid4())
    reader.client.upsert("user_profiles", points=[models.PointStruct(id=uid, vector=[1.] + [0.] * 383, payload=profile_payload() | bad)], wait=True)
    vector, payload = reader._user_profile(uid)
    analyzer = Opportunities()
    analyzer.observe(uid, vector, payload, profile_payload())
    assert vector is None
    assert analyzer.valid == analyzer.content_repeats == 0


def test_missing_profile_has_no_opportunity(reader):
    assert reader._user_profile(str(uuid.uuid4())) == (None, {})


def test_http_hook_measures_only_actual_profile_bodies():
    import httpx
    spans = WireSpans()
    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"result": []})),
                      event_hooks={"request": [spans.request], "response": [spans.response]}) as client:
        client.get("http://fixture/collections")
        response = client.post("http://fixture/collections/user_profiles/points", json={"ids": []})
    assert spans.decoded_body_bytes == [len(response.content)]
    assert len(spans.milliseconds) == 1 and spans.milliseconds[0] >= 0


def test_scenario_never_silently_treats_local_calls_as_measured_http(reader):
    # QdrantLocal exercises the real retrieval path but cannot establish wire cost.
    with pytest.raises(AssertionError, match="real HTTP"):
        run_scenario(reader.client, reader, WireSpans(), name="local", users=1, reads=2)


def test_measurement_contract_is_independent_of_ambient_collection_and_dimension():
    import os
    from pathlib import Path
    import subprocess
    import sys
    import textwrap
    script=textwrap.dedent("""
        from contextlib import contextmanager
        import json
        from uuid import UUID
        from qdrant_client import QdrantClient, models
        from scripts import measure_profile_fetch as bench
        class Owned(QdrantClient):
            def close(self): pass
        client=Owned(":memory:")
        @contextmanager
        def owned(): yield bench.OwnedProfileTarget(url="owned-memory-only",image_id="owned-fixture-image")
        bench.owned_qdrant=owned
        bench.QdrantClient=lambda **kwargs: client
        bench.source_hashes=lambda: {}
        def verify(writer,reader,spans,**kwargs):
            uid=str(UUID(int=63))
            writer.upsert("user_profiles",points=[models.PointStruct(id=uid,
                vector=[1.]+[0.]*383,payload=bench.profile_payload())],wait=True)
            vector,payload=reader._user_profile(uid)
            assert len(vector)==384 and payload['embedding_dim']==384
            assert reader.user_collection=="user_profiles"
            import os
            assert os.environ['V2_DIVERSITY_MODE']=='enabled'
            assert reader.diversity_mode=='disabled'
            return {"name":kwargs['name']}
        bench.run_scenario=verify
        try:
            result=bench.measure(32)
            assert len(result['scenarios'])==5
        finally:
            QdrantClient.close(client)
    """)
    env={**os.environ,'APP_ENV':'development','USER_PROFILES_COLLECTION':'ambient-profile-collection',
         'VECTOR_DIMENSION':'32','V2_DIVERSITY_MODE':'enabled'}
    subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[1],
                   env=env,check=True,capture_output=True,text=True,timeout=30)


def _fake_owned_docker(monkeypatch, *, cleanup_failure=None, startup_timeout=False, wrong_owner=False, host_ip="127.0.0.1"):
    from scripts import measure_profile_fetch as bench
    from types import SimpleNamespace
    import json
    import subprocess
    class Fake(bench._LocalDocker):
        def __init__(self):
            self.commands=[];self.owner=None;self.name=None;self.exists=False
        def run(self,*arguments,check=True):
            self.commands.append(arguments)
            if arguments[:2]==('image','inspect'):
                assert arguments[2]==bench.IMAGE and '@sha256:' in bench.IMAGE
                return SimpleNamespace(returncode=0,stdout=json.dumps([{'Id':'owned-fixture-image'}]),stderr='')
            if arguments[0]=='run':
                self.name=arguments[arguments.index('--name')+1]
                label=arguments[arguments.index('--label')+1]
                assert label.startswith(bench.OWNER_LABEL+'=')
                self.owner=label.split('=',1)[1];self.exists=True
                assert arguments[arguments.index('-p')+1]=='127.0.0.1::6333'
                assert arguments[arguments.index('--memory')+1]=='1g'
                assert arguments[arguments.index('--cpus')+1]=='2'
                assert arguments[arguments.index('--pids-limit')+1]=='256'
                if startup_timeout:raise subprocess.TimeoutExpired(arguments,40)
            elif arguments[0]=='inspect':
                if not self.exists:return SimpleNamespace(returncode=1,stdout='',stderr='Error: No such object')
                labels={bench.OWNER_LABEL:'someone-else' if wrong_owner else self.owner}
                return SimpleNamespace(returncode=0,stderr='',stdout=json.dumps([{'Config':{'Labels':labels},
                    'NetworkSettings':{'Ports':{'6333/tcp':[{'HostIp':host_ip,'HostPort':'12345'}]}}}]))
            elif arguments[0]=='rm':
                if cleanup_failure=='timeout':raise subprocess.TimeoutExpired(arguments,40)
                if cleanup_failure=='nonzero':raise RuntimeError('owned profile Docker operation failed: rm')
                self.exists=False
            return SimpleNamespace(returncode=0,stdout='owned-id',stderr='')
    fake=Fake()
    monkeypatch.setattr(bench,'_LocalDocker',lambda:fake)
    class Probe:
        def get_collections(self):return []
        def close(self):pass
    monkeypatch.setattr(bench,'QdrantClient',lambda **kwargs:Probe())
    return bench,fake


@pytest.mark.parametrize('failure',['nonzero','timeout'])
@pytest.mark.parametrize('primary',[False,True])
def test_owned_cleanup_reports_name_and_preserves_primary_failure(monkeypatch,capsys,failure,primary):
    bench,fake=_fake_owned_docker(monkeypatch,cleanup_failure=failure)
    original=ValueError('measurement failed')
    with pytest.raises(ValueError if primary else RuntimeError) as caught:
        with bench.owned_qdrant() as target:
            assert target.url=='http://127.0.0.1:12345' and target.image_id=='owned-fixture-image'
            if primary:raise original
    names=[arguments[-1] for arguments in fake.commands if arguments[0]=='rm']
    assert names==[fake.name] and fake.name.startswith('weave-profile-measure-')
    if primary:
        assert caught.value is original
        assert fake.name in capsys.readouterr().err
    else:
        assert fake.name in str(caught.value)


def test_startup_timeout_still_finds_and_removes_only_labelled_container(monkeypatch):
    import subprocess
    bench,fake=_fake_owned_docker(monkeypatch,startup_timeout=True)
    with pytest.raises(subprocess.TimeoutExpired):
        with bench.owned_qdrant():pytest.fail('timed out startup yielded')
    assert not fake.exists
    assert [arguments[0] for arguments in fake.commands][-3:]==['inspect','rm','inspect']


def test_foreign_container_label_is_never_removed(monkeypatch):
    bench,fake=_fake_owned_docker(monkeypatch,wrong_owner=True)
    with pytest.raises(RuntimeError,match='label mismatch'):
        with bench.owned_qdrant():pytest.fail('foreign resource yielded')
    assert fake.exists and not any(arguments[0]=='rm' for arguments in fake.commands)


def test_nonloopback_binding_fails_and_owned_resource_is_cleaned(monkeypatch):
    bench,fake=_fake_owned_docker(monkeypatch,host_ip='0.0.0.0')
    with pytest.raises(ValueError,match='loopback'):
        with bench.owned_qdrant():pytest.fail('nonloopback target yielded')
    assert not fake.exists


@pytest.mark.parametrize('host',['tcp://remote:2375','ssh://remote','unix://remote/run/docker.sock',
    'unix:///run/docker.sock?context=remote','unix:///run/docker.sock#remote','unix:///run/docker.sock\n'])
def test_nonlocal_docker_target_is_rejected_before_subprocess(monkeypatch,host):
    from scripts import measure_profile_fetch as bench
    monkeypatch.setenv('DOCKER_HOST',host)
    monkeypatch.setattr(bench.subprocess,'run',lambda *args,**kwargs:pytest.fail('unsafe Docker command launched'))
    with pytest.raises(ValueError,match='local Docker Unix socket'):
        bench._LocalDocker()


def test_every_docker_operation_pins_local_daemon_and_filters_context_env(monkeypatch):
    from scripts import measure_profile_fetch as bench
    from types import SimpleNamespace
    import stat
    monkeypatch.setenv('DOCKER_HOST','unix:///tmp/profile-rootless.sock')
    monkeypatch.setenv('DOCKER_CONTEXT','remote-context')
    monkeypatch.setenv('DOCKER_TLS_VERIFY','1')
    monkeypatch.setenv('DOCKER_CERT_PATH','private-path')
    monkeypatch.setattr(bench,'Path',lambda path:SimpleNamespace(stat=lambda:SimpleNamespace(st_mode=stat.S_IFSOCK)))
    calls=[]
    def capture(arguments,**kwargs):
        calls.append((arguments,kwargs));return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(bench.subprocess,'run',capture)
    docker=bench._LocalDocker()
    # Later caller environment mutation cannot redirect cleanup/inspect.
    monkeypatch.setenv('DOCKER_HOST','ssh://later-remote')
    for operation in [('image','inspect',bench.IMAGE),('inspect','owned'),('rm','-f','owned')]:docker.run(*operation)
    assert len(calls)==3
    for arguments,options in calls:
        assert arguments[:3]==['docker','--host','unix:///tmp/profile-rootless.sock']
        assert not any(key.startswith('DOCKER_') for key in options['env'])
        assert options['timeout']==40 and options['check'] is False


def test_missing_docker_cli_is_bounded_and_has_no_cleanup_target(monkeypatch):
    from scripts import measure_profile_fetch as bench
    from types import SimpleNamespace
    import stat
    monkeypatch.delenv('DOCKER_HOST',raising=False)
    monkeypatch.setattr(bench,'Path',lambda path:SimpleNamespace(stat=lambda:SimpleNamespace(st_mode=stat.S_IFSOCK)))
    def missing(*args,**kwargs):raise FileNotFoundError('docker')
    monkeypatch.setattr(bench.subprocess,'run',missing)
    with pytest.raises(FileNotFoundError):
        with bench.owned_qdrant():pytest.fail('missing Docker yielded')


@pytest.mark.parametrize('appears', [True, False])
def test_timeout_cleanup_retries_late_creation_or_warns(monkeypatch, capsys, appears):
    import subprocess
    bench, fake = _fake_owned_docker(monkeypatch, startup_timeout=True)
    original = fake.owned_info
    reads = []
    def delayed(name, owner):
        reads.append(name)
        if len(reads) < 3 or not appears:
            return None
        return original(name, owner)
    monkeypatch.setattr(fake, 'owned_info', delayed)
    monkeypatch.setattr(bench.time, 'sleep', lambda _: None)
    with pytest.raises(subprocess.TimeoutExpired):
        with bench.owned_qdrant(): pytest.fail('startup timeout yielded')
    assert len(reads) >= 3
    if appears:
        assert not fake.exists
    else:
        assert fake.name in capsys.readouterr().err


@pytest.mark.parametrize('stderr,reason', [
    ('permission denied: private/token', 'Docker socket permission denied'),
    ('No such image: private/token', 'pinned image is unavailable locally'),
    ('invalid memory resource private/token', 'container resource limit rejected'),
    ('unknown private/token', 'unclassified Docker failure')])
def test_docker_failure_reasons_are_actionable_without_echoing_secrets(monkeypatch, stderr, reason):
    from scripts import measure_profile_fetch as bench
    from types import SimpleNamespace
    docker = object.__new__(bench._LocalDocker)
    docker.prefix = ('docker',); docker.environment = {}
    monkeypatch.setattr(bench.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=1, stderr=stderr))
    with pytest.raises(RuntimeError) as caught:
        docker.run('run')
    assert reason in str(caught.value)
    assert 'private/token' not in str(caught.value)
