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
        def owned(): yield "owned-memory-only"
        bench.owned_qdrant=owned
        bench.QdrantClient=lambda **kwargs: client
        bench.source_hashes=lambda: {}
        bench.docker=lambda *args: json.dumps([{"Id":"owned-fixture-image"}])
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


@pytest.mark.parametrize('failure', ['nonzero', 'timeout'])
@pytest.mark.parametrize('primary', [False, True])
def test_owned_cleanup_reports_name_and_preserves_primary_failure(monkeypatch,capsys,failure,primary):
    from scripts import measure_profile_fetch as bench
    from types import SimpleNamespace
    import subprocess
    import json
    class Probe:
        def get_collections(self): return []
        def close(self): pass
    monkeypatch.setattr(bench,'QdrantClient',lambda **kwargs: Probe())
    def docker(*args):
        if args[0]=='inspect':
            return json.dumps([{'NetworkSettings':{'Ports':{'6333/tcp':[{'HostPort':'1'}]}}}])
        return 'owned-id'
    monkeypatch.setattr(bench,'docker',docker)
    names=[]
    def cleanup(command,**kwargs):
        names.append(command[-1])
        if failure=='timeout': raise subprocess.TimeoutExpired(command,40)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(bench.subprocess,'run',cleanup)
    original=ValueError('measurement failed')
    with pytest.raises(ValueError if primary else RuntimeError) as caught:
        with bench.owned_qdrant():
            if primary: raise original
    assert len(names)==1 and names[0].startswith('weave-profile-measure-')
    if primary:
        assert caught.value is original
        assert names[0] in capsys.readouterr().err
    else:
        assert names[0] in str(caught.value)
