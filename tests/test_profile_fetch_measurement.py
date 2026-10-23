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
        yield QdrantV2Retriever(client=client, heavy_ranker_enabled=False,
                                minimum_eligible_repositories=0)


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
