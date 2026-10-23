"""Registered FastAPI -> actual retrieval/Qdrant -> trained ranker capture."""
from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context

from fastapi.testclient import TestClient
import numpy as np
import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from api import main, v2, scoring_capture
from api.contracts import CapturedRecommendationResponse, UnscoredEmptyRecommendationResponse, RecommendationRequest
from config import EMBEDDING_MODEL_REVISION, REPOSITORY_EMBEDDING_MODEL
from embedding.vector_contract import repository_payload_defaults
from inference.ranker_service import RankerService
from inference.scoring_provenance import ScoredPacket, decode_owned_packet
from retrieval.v2_retriever import QdrantV2Retriever, _USER_RETRIEVAL_PAYLOAD_FIELDS

SECRET = "private-test-internal"
CAPABILITY = "7" * 64
COMMIT = "a" * 40
USER = str(uuid.UUID(int=7001))


@pytest.fixture(scope="module")
def trained_ranker():
    folder = Path(inspect.getfile(RankerService)).parent
    return RankerService(model_path=str(folder / "heavy_ranker.pt"), scaler_path=str(folder / "feature_scaler.json"),
                         manifest_path=str(folder / "model_manifest.json"), require_production_manifest=True, device="cpu")


def make_retriever(client, ranker, *, prefix=None, count=12):
    suffix = prefix or uuid.uuid4().hex
    repositories, users = "capture_repos_" + suffix, "capture_users_" + suffix
    client.create_collection(repositories, vectors_config={"repo_embedding": models.VectorParams(size=384, distance=models.Distance.COSINE)})
    client.create_collection(users, vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
    user = np.zeros(384, np.float32)
    user[0] = 1.
    client.upsert(users, points=[models.PointStruct(id=USER, vector=user.tolist(), payload={
        "embedding_dim": 384, "embedding_model": REPOSITORY_EMBEDDING_MODEL,
        "embedding_model_revision": EMBEDDING_MODEL_REVISION, "skills": ["python"],
        "profile_version": 2, "last_feedback_version": 3, "feedback_state_revision": 4})], wait=True)
    for index in range(count):
        repo_id = str(uuid.UUID(int=8000 + index))
        vector = np.zeros(384, np.float32)
        vector[0], vector[index + 1] = .8, .6
        payload = repository_payload_defaults()
        payload.update(repo_id=repo_id, full_name=f"diagnostic/repo-{index}", content_version=1,
            content_hash="a" * 64, source_hash="b" * 64, indexed_at="2026-10-24T00:00:00Z",
            star_count=index * 100, primary_language="Python", languages=["Python"], topics=["python"],
            pushed_days_ago=1, delta_7d=1, activity_score=.5)
        # Fresh indexes really omit feature_version: the existing codec's
        # authoritative zero-version compatibility must work in this route.
        payload.pop("feature_version", None)
        client.upsert(repositories, points=[models.PointStruct(id=repo_id,
            vector={"repo_embedding": vector.tolist()}, payload=payload)], wait=True)
    return QdrantV2Retriever(client=client, repository_collection=repositories, user_collection=users,
        ranker=ranker, heavy_ranker_enabled=True, heavy_ranker_required=True,
        heavy_ranker_traffic_percent=100., minimum_eligible_repositories=1), repositories, users


@pytest.fixture
def actual_pipeline(monkeypatch, trained_ranker):
    monkeypatch.setenv("INTERNAL_API_SECRET", SECRET)
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("SCORING_CAPTURE_ENABLED", "true")
    monkeypatch.setenv("SCORING_CAPTURE_CAPABILITY", CAPABILITY)
    monkeypatch.setenv("SCORING_CAPTURE_CODE_COMMIT", COMMIT)
    monkeypatch.setenv("BAKED_ML_RELEASE_ID", COMMIT)
    client = QdrantClient(":memory:")
    retriever, repositories, users = make_retriever(client, trained_ranker)
    monkeypatch.setattr(v2, "retriever", lambda: retriever)
    matrices = []
    hook = trained_ranker.model.register_forward_pre_hook(lambda _model, args: matrices.append(args[0].detach().cpu().numpy().copy()))
    try:
        yield retriever, client, repositories, users, matrices
    finally:
        hook.remove()
        client.close()


def payload(*, captured=True, generation=None, feed_version=None):
    result = {"schema_version": 2, "user_id": USER, "generation_id": generation or str(uuid.uuid4()),
              "feed_version": feed_version if feed_version is not None else ("9007199254740993" if captured else 1), "limit": 5,
              "exclude_repo_ids": [], "social_repo_ids": [], "context": {"cold_start": False}}
    if captured:
        result["scoring_capture"] = {"schema_version": 1, "encoding": "float32-le-v1"}
    return result


def headers(*, capability=CAPABILITY):
    return {"x-internal-secret": SECRET, "x-scoring-capture-capability": capability}


def post(body, request_headers=None):
    return TestClient(main.app).post("/api/v2/recommendations/generate", json=body,
                                     headers=request_headers or headers())


def test_registered_route_captures_actual_trained_tensor_and_exact_large_feed_version(actual_pipeline):
    retriever, client, _, users, matrices = actual_pipeline
    calls = []
    original = client.retrieve

    def read(*args, **kwargs):
        rows = original(*args, **kwargs)
        if kwargs.get("collection_name") == users:
            calls.append(copy.deepcopy(kwargs["with_payload"]))
            # Update after the same-read snapshot. Context must not perform a
            # second read or borrow these later applied cursor revisions.
            client.set_payload(users, {"profile_version": 12, "last_feedback_version": 13, "feedback_state_revision": 14}, points=[USER], wait=True)
        return rows
    client.retrieve = read
    result = post(payload())
    assert result.status_code == 200, result.text
    body = result.json()
    assert result.headers["cache-control"] == "no-store"
    assert len(result.content) <= 1_000_000
    assert body["feed_version"] == "9007199254740993"
    assert body["scoring_context"]["profile_version"] == "2"
    assert body["scoring_context"]["feedback_version"] == "3"
    assert body["scoring_context"]["feedback_state_revision"] == "4"
    assert len(calls) == 1 and "last_feedback_version" in calls[0]
    model = CapturedRecommendationResponse.model_validate_json(result.content)
    assert len(matrices) == 1 and matrices[0].shape[1] == 778
    actual_rows = {row.astype("<f4", copy=False).tobytes() for row in matrices[0]}
    for item in model.items:
        raw = decode_owned_packet(item.scored_input, model.scoring_context, owner_user_id=USER,
                                  generation_id=str(model.generation_id), repo_id=str(item.repo_id))
        assert raw in actual_rows
        assert item.scored_input.snapshot.feature_version == "0"
        assert np.isfinite(item.raw_heavy_score)
    assert body["scoring_capture"] == {"schema_version": 1, "status": "captured"}
    assert "input_base64" not in repr(model)


def test_disabled_normal_path_does_not_validate_new_configuration_or_capture_projection(actual_pipeline, monkeypatch):
    _, client, _, users, _ = actual_pipeline
    monkeypatch.setenv("SCORING_CAPTURE_ENABLED", "invalid-on-purpose")
    monkeypatch.setenv("SCORING_CAPTURE_CAPABILITY", "malformed")
    projections = []
    original = client.retrieve
    def read(*args, **kwargs):
        if kwargs.get("collection_name") == users:
            projections.append(kwargs["with_payload"])
        return original(*args, **kwargs)
    client.retrieve = read
    result = post(payload(captured=False))
    assert result.status_code == 200, result.text
    body = result.json()
    assert "scoring_capture" not in body and "scoring_context" not in body
    assert all("scored_input" not in item and "raw_heavy_score" not in item for item in body["items"])
    assert projections == [_USER_RETRIEVAL_PAYLOAD_FIELDS]


@pytest.mark.parametrize(("setting", "value", "status"), [("SCORING_CAPTURE_ENABLED", "false", 503),
    ("SCORING_CAPTURE_ENABLED", "invalid", 503), ("SCORING_CAPTURE_CAPABILITY", "short", 503),
    ("SCORING_CAPTURE_CODE_COMMIT", "main", 503)])
def test_capture_configuration_fails_closed_before_scheduling(actual_pipeline, monkeypatch, setting, value, status):
    monkeypatch.setenv(setting, value)
    monkeypatch.setattr(v2, "run_recommendation_job", lambda *a: pytest.fail("unauthorized capture scheduled"))
    result = post(payload())
    assert result.status_code == status
    assert CAPABILITY not in result.text
    assert result.json()["code"] in {"SCORING_CAPTURE_DISABLED", "SCORING_CAPTURE_CONFIG_INVALID"}


def test_scoped_capability_is_independent_and_removed_before_downstream_observers(actual_pipeline, monkeypatch):
    result = post(payload(), headers(capability="wrong"))
    assert result.status_code == 403
    result = post(payload(), {"x-scoring-capture-capability": CAPABILITY})
    assert result.status_code == 401
    original = v2.authorized_capture_options
    observed = []
    def authorize(request, version):
        options = original(request, version)
        observed.append(dict(request.headers))
        return options
    monkeypatch.setattr(v2, "authorized_capture_options", authorize)
    assert post(payload()).status_code == 200
    assert observed and all("x-scoring-capture-capability" not in row for row in observed)


@pytest.mark.parametrize("version", [True, 1.0, 0, -1, 2**53, "01", "+1", " 1", "1 ", "1.0", str(2**63)])
def test_request_feed_versions_are_exact_and_never_coerced(version):
    with pytest.raises(ValidationError):
        RecommendationRequest.model_validate(payload(feed_version=version))


def test_required_heavy_no_eligible_failure_is_preserved(actual_pipeline):
    body = payload()
    body["exclude_repo_ids"] = [str(uuid.UUID(int=8000 + i)) for i in range(12)]
    result = post(body)
    assert result.status_code != 200
    assert "not_scored_empty" not in result.text


@pytest.mark.parametrize("fault", ["packet", "oversize"])
def test_actual_route_validation_and_wire_failure_redact_private_payload(actual_pipeline, monkeypatch, caplog, fault):
    good = post(payload()).json()
    private_bytes = good["items"][0]["scored_input"]["input_base64"]
    if fault == "packet":
        original = scoring_capture.packet_for_frame
        monkeypatch.setattr(scoring_capture, "packet_for_frame", lambda *a: original(*a).model_copy(update={"binding_sha256": "0" * 64}))
    else:
        monkeypatch.setattr(scoring_capture, "DEFAULT_WIRE_LIMIT", 128)
    result = post(payload())
    assert result.status_code == 503
    assert "SCORING_CAPTURE_UNAVAILABLE" in result.text
    assert private_bytes not in result.text + caplog.text
    assert CAPABILITY not in result.text + caplog.text
    assert "input_base64" not in caplog.text and "ValidationError" not in caplog.text


def test_raw_heavy_score_precedes_actual_social_adjustment(actual_pipeline):
    body = payload()
    body["social_repo_ids"] = [str(uuid.UUID(int=8000 + index)) for index in range(12)]
    result = post(body)
    assert result.status_code == 200, result.text
    assert any(item["score"] != item["raw_heavy_score"] for item in result.json()["items"])


def test_concurrent_actual_requests_keep_owner_generation_and_input_bytes_isolated(actual_pipeline):
    _, client, _, users, matrices = actual_pipeline
    other = str(uuid.UUID(int=7002))
    vector = [0., 1.] + [0.] * 382
    client.upsert(users, points=[models.PointStruct(id=other, vector=vector, payload={
        "embedding_dim": 384, "embedding_model": REPOSITORY_EMBEDDING_MODEL,
        "embedding_model_revision": EMBEDDING_MODEL_REVISION, "profile_version": 22,
        "last_feedback_version": 23, "feedback_state_revision": 24})], wait=True)
    bodies = [payload(), payload()]
    bodies[1]["user_id"] = other
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(post, bodies))
    assert [result.status_code for result in results] == [200, 200]
    assert len(matrices) == 2
    for expected, result in zip(bodies, results):
        body = result.json()
        assert body["scoring_context"]["user_id"] == expected["user_id"]
        assert body["scoring_context"]["generation_id"] == expected["generation_id"]
        model = CapturedRecommendationResponse.model_validate_json(result.content)
        raw = decode_owned_packet(model.items[0].scored_input, model.scoring_context,
            owner_user_id=expected["user_id"], generation_id=expected["generation_id"], repo_id=str(model.items[0].repo_id))
        values = np.frombuffer(raw, dtype="<f4")
        assert values[0] == (1. if expected["user_id"] == USER else 0.)
        assert values[1] == (0. if expected["user_id"] == USER else 1.)


def test_collector_failure_is_cleared_before_next_ordinary_and_captured_request(actual_pipeline, monkeypatch):
    from inference import scoring_runtime
    original = scoring_runtime.scoring_context_for
    # Retriever imports this helper directly; poison only its generated hash.
    import retrieval.v2_retriever as module
    def wrong(*args, **kwargs):
        return original(*args, **kwargs).model_copy(update={"user_vector_sha256": "0" * 64})
    with monkeypatch.context() as scope:
        scope.setattr(module, "scoring_context_for", wrong)
        assert post(payload()).status_code == 503
    assert post(payload(captured=False)).status_code == 200
    assert post(payload()).status_code == 200


@pytest.mark.parametrize("mutation", ["owner", "feed", "raw_nan", "extra", "schema_bool"])
def test_captured_response_rejects_altered_owner_scores_and_extra_fields_without_private_repr(actual_pipeline, mutation):
    result = post(payload())
    assert result.status_code == 200
    body = result.json()
    secret = body["items"][0]["scored_input"]["input_base64"]
    if mutation == "owner": body["user_id"] = str(uuid.uuid4())
    elif mutation == "feed": body["feed_version"] = "2"
    elif mutation == "raw_nan": body["items"][0]["raw_heavy_score"] = "not-numeric"
    elif mutation == "schema_bool": body["scoring_capture"]["schema_version"] = True
    else: body["items"][0]["scored_input"]["unknown"] = secret
    with pytest.raises(ValidationError) as error:
        CapturedRecommendationResponse.model_validate_json(json.dumps(body))
    assert secret not in str(error.value) and secret not in repr(error.value)


def test_actual_capture_scope_belongs_to_worker_and_closes_even_with_copied_context_attack(actual_pipeline, monkeypatch):
    from inference import scoring_runtime
    original = scoring_runtime.capture_scored_inputs
    collectors, threads = [], []
    @contextmanager
    def scope(context):
        with original(context) as capture:
            collectors.append(capture)
            threads.append(threading.current_thread().name)
            copied = copy_context()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(copied.run, capture.record, actual_pipeline[0].ranker,
                                         [], np.empty((0, 778), dtype=np.float32))
                with pytest.raises(ValueError, match="owning execution thread"):
                    future.result(timeout=5)
            yield capture
    monkeypatch.setattr(scoring_runtime, "capture_scored_inputs", scope)
    result = post(payload())
    assert result.status_code == 200, result.text
    assert threads and all(name.startswith("ml-recommendation") for name in threads)
    assert all(capture.frames == () for capture in collectors)


def test_actual_timeout_keeps_worker_admission_until_exit_and_next_scope_is_clean(actual_pipeline, monkeypatch):
    from inference.runtime import recommendation_runtime_status
    entered, release = threading.Event(), threading.Event()
    def hold(_model, _args):
        entered.set()
        assert release.wait(timeout=5)
    handle = actual_pipeline[0].ranker.model.register_forward_pre_hook(hold)
    # Qdrant deliberately refuses RPCs with <1second remaining; provide enough
    # budget to reach actual scoring, then block that synchronous model call.
    monkeypatch.setenv("V2_RECOMMENDATION_TIMEOUT_SECONDS", "2")
    try:
        result = post(payload())
        assert result.status_code == 504, result.text
        assert entered.is_set()
        assert recommendation_runtime_status()["outstanding"] > 0
    finally:
        release.set()
        handle.remove()
    deadline = time.monotonic() + 5
    while recommendation_runtime_status()["outstanding"]:
        assert time.monotonic() < deadline
        time.sleep(.01)
    monkeypatch.setenv("V2_RECOMMENDATION_TIMEOUT_SECONDS", "12")
    assert post(payload(captured=False)).status_code == 200
    assert post(payload()).status_code == 200


def test_actual_maximum_candidate_pool_fits_exact_wire_budget(actual_pipeline, monkeypatch, trained_ranker):
    _, client, _, _, matrices = actual_pipeline
    larger, _, _ = make_retriever(client, trained_ranker, count=150)
    monkeypatch.setattr(v2, "retriever", lambda: larger)
    body = payload()
    body["limit"] = 100
    result = post(body)
    assert result.status_code == 200, result.text
    assert matrices[-1].shape == (150, 778)
    assert 0 < len(result.json()["items"]) <= 100
    assert len(result.content) <= 1_000_000


@pytest.mark.parametrize(("field", "value"), [("profile_version", None), ("profile_version", 0),
    ("last_feedback_version", "03"), ("feedback_state_revision", True)])
def test_capture_never_guesses_missing_positive_or_malformed_same_read_versions(actual_pipeline, field, value):
    _, client, _, users, _ = actual_pipeline
    client.set_payload(users, {field: value}, points=[USER], wait=True)
    assert post(payload()).status_code == 503
    assert post(payload(captured=False)).status_code == 200


def test_legacy_absent_feedback_cursors_are_honest_zero_not_backend_accepted_versions(actual_pipeline):
    _, client, _, users, _ = actual_pipeline
    client.delete_payload(users, keys=["last_feedback_version", "feedback_state_revision"], points=[USER], wait=True)
    result = post(payload())
    assert result.status_code == 200, result.text
    assert result.json()["scoring_context"]["feedback_version"] == "0"
    assert result.json()["scoring_context"]["feedback_state_revision"] == "0"


def test_empty_capture_schema_has_no_invented_context_and_does_not_change_heavy_failure(actual_pipeline):
    # Schema-only shape check. The real no-eligible required-heavy route remains
    # a failure in the separate actual-pipeline test, not a fabricated success.
    response = post(payload())
    assert response.status_code == 200
    body = response.json()
    body.pop("scoring_context")
    body.pop("scoring_metadata_complete")
    body["items"] = []
    body["scoring_capture"]["status"] = "not_scored_empty"
    empty = UnscoredEmptyRecommendationResponse.model_validate_json(json.dumps(body))
    assert empty.items == []
    body["scoring_context"] = {"invented": True}
    with pytest.raises(ValidationError):
        UnscoredEmptyRecommendationResponse.model_validate_json(json.dumps(body))


def test_duplicate_scoped_capability_is_not_accepted_as_one_secret(actual_pipeline):
    result = post(payload(), [("x-internal-secret", SECRET),
        ("x-scoring-capture-capability", CAPABILITY), ("x-scoring-capture-capability", CAPABILITY)])
    assert result.status_code == 403
    assert result.json()["code"] == "SCORING_CAPTURE_FORBIDDEN"
