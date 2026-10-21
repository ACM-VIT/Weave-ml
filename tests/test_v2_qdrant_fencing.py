"""Deterministic stale-holder tests for Qdrant storage-level fencing."""

from __future__ import annotations

from dataclasses import replace
import json
import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

from embedding.qdrant_store import QdrantRepositoryStore
from embedding.repository_embedding import (
    RepositoryEmbeddingConfig,
    RepositoryEmbeddingResult,
    build_vector_payload,
)
from embedding.user_profile_store import QdrantUserProfileStore
from embedding.vector_contract import (
    FEEDBACK_STATE_REVISION_FIELD,
    USER_PROFILE_COLLECTION_CONTRACT,
    repository_point_id,
    user_point_id,
)
from feedback.v2 import (
    FeedbackWriteConflict,
    OrderedFeedbackApplier,
    PENDING_REJECTION_KEY,
)
from feedback.v2_settings import V2FeedbackSettings


REPO_ID = "00000000-0000-4000-8000-000000000711"
USER_ID = "00000000-0000-4000-8000-000000000712"
VECTOR_SIZE = 384


def _vector(index: int = 0) -> list[float]:
    vector = [0.0] * VECTOR_SIZE
    vector[index] = 1.0
    return vector


def _job() -> str:
    return str(uuid.uuid4())


def _repository_result(
    content_version: int,
    content_job_id: str,
    *,
    feature_version: int | None = None,
    feature_job_id: str | None = None,
) -> RepositoryEmbeddingResult:
    vector = _vector(content_version % 2)
    config = RepositoryEmbeddingConfig()
    payload = build_vector_payload(
        {
            "repo_id": REPO_ID,
            "full_name": "owner/fenced-repository",
            "description": f"content revision {content_version}",
            "primary_language": "Python",
            "languages": ["Python"],
            "topics": ["qdrant"],
            "content_version": content_version,
        },
        repo_id=REPO_ID,
        final_embedding=vector,
        readme_chunks=1,
        source_hash=f"source-{content_version}",
        config=config,
    )
    payload["content_job_id"] = content_job_id
    if feature_version is not None:
        payload["feature_version"] = feature_version
    if feature_job_id is not None:
        payload["feature_job_id"] = feature_job_id
    return RepositoryEmbeddingResult(
        repo_id=REPO_ID,
        final_embedding=vector,
        readme_embedding=vector,
        metadata_embedding=vector,
        topic_embedding=vector,
        payload=payload,
        readme_chunks=1,
        source_hash=f"source-{content_version}",
        embedding_model=config.model_name,
        embedding_version=config.version,
    )


def _repository_store() -> tuple[QdrantClient, QdrantRepositoryStore]:
    client = QdrantClient(":memory:")
    store = QdrantRepositoryStore(client=client)
    client.create_collection(
        store.collection_name,
        vectors_config={
            store.vector_name: models.VectorParams(
                size=store.vector_size,
                distance=models.Distance.COSINE,
            )
        },
    )
    return client, store


def _repository_point(client: QdrantClient, store: QdrantRepositoryStore, point_id: str):
    return client.retrieve(
        store.collection_name,
        ids=[point_id],
        with_payload=True,
        with_vectors=True,
    )[0]


def test_stale_content_holder_cannot_overwrite_a_newer_content_revision() -> None:
    client, store = _repository_store()
    store.upsert([_repository_result(1, _job())])
    stale_snapshot = _repository_point(client, store, repository_point_id(REPO_ID))

    newest_job = _job()
    store.compare_and_set_content(
        _repository_result(3, newest_job),
        expected_point=stale_snapshot,
    )
    stale_result = store.compare_and_set_content(
        _repository_result(2, _job()),
        expected_point=stale_snapshot,
    )

    assert stale_result.payload["content_version"] == 3
    assert stale_result.payload["content_job_id"] == newest_job
    assert stale_result.payload["description"] == "content revision 3"


def test_stale_feature_holder_cannot_overwrite_a_newer_feature_revision() -> None:
    client, store = _repository_store()
    store.upsert([_repository_result(1, _job())])
    stale_snapshot = _repository_point(client, store, repository_point_id(REPO_ID))

    newest_job = _job()
    store.compare_and_set_features(
        expected_point=stale_snapshot,
        feature_payload={
            "star_count": 300,
            "feature_version": 3,
            "feature_job_id": newest_job,
        },
    )
    stale_result = store.compare_and_set_features(
        expected_point=stale_snapshot,
        feature_payload={
            "star_count": 200,
            "feature_version": 2,
            "feature_job_id": _job(),
        },
    )

    assert stale_result.payload["feature_version"] == 3
    assert stale_result.payload["feature_job_id"] == newest_job
    assert stale_result.payload["star_count"] == 300


def test_feature_refresh_fences_a_content_holder_that_would_erase_it() -> None:
    client, store = _repository_store()
    store.upsert([_repository_result(1, _job())])
    stale_content_snapshot = _repository_point(
        client,
        store,
        repository_point_id(REPO_ID),
    )

    refresh_job = _job()
    store.compare_and_set_features(
        expected_point=stale_content_snapshot,
        feature_payload={
            "activity_score": 0.9,
            "feature_version": 2,
            "feature_job_id": refresh_job,
        },
    )
    stale_result = store.compare_and_set_content(
        _repository_result(2, _job()),
        expected_point=stale_content_snapshot,
    )

    assert stale_result.payload["content_version"] == 1
    assert stale_result.payload["feature_version"] == 2
    assert stale_result.payload["feature_job_id"] == refresh_job
    assert stale_result.payload["activity_score"] == 0.9


def _user_store() -> tuple[QdrantClient, QdrantUserProfileStore]:
    client = QdrantClient(":memory:")
    store = QdrantUserProfileStore(client=client)
    client.create_collection(
        USER_PROFILE_COLLECTION_CONTRACT.collection_name,
        vectors_config=models.VectorParams(
            size=USER_PROFILE_COLLECTION_CONTRACT.vector_size,
            distance=models.Distance.COSINE,
        ),
    )
    return client, store


def _user_payload(
    profile_version: int,
    job_id: str,
    *,
    feedback_version: int = 0,
    feedback_event_id: str | None = None,
) -> dict:
    payload = {
        "profile_version": profile_version,
        "job_id": job_id,
        "last_feedback_version": feedback_version,
        "topics": [f"profile-{profile_version}"],
    }
    if feedback_event_id is not None:
        payload["last_feedback_event_id"] = feedback_event_id
    return payload


def _user_point(client: QdrantClient, point_id: str):
    return client.retrieve(
        USER_PROFILE_COLLECTION_CONTRACT.collection_name,
        [point_id],
        with_payload=True,
        with_vectors=True,
    )[0]


def test_feedback_cursor_change_fences_stale_onboarding() -> None:
    client, store = _user_store()
    store.upsert_user(
        USER_ID,
        _vector(),
        payload=_user_payload(1, _job()),
    )
    point_id = user_point_id(USER_ID)
    stale_snapshot = _user_point(client, point_id)

    feedback_event_id = _job()
    feedback_payload = dict(stale_snapshot.payload)
    feedback_payload.update(
        last_feedback_version=1,
        last_feedback_event_id=feedback_event_id,
    )
    client.upsert(
        USER_PROFILE_COLLECTION_CONTRACT.collection_name,
        [
            models.PointStruct(
                id=point_id,
                vector=_vector(1),
                payload=feedback_payload,
            )
        ],
        wait=True,
    )

    applied = store.compare_and_set_user(
        USER_ID,
        _vector(2),
        payload=_user_payload(2, _job()),
        expected_point=stale_snapshot,
    )
    final = _user_point(client, point_id)

    assert applied is False
    assert final.payload["profile_version"] == 1
    assert final.payload["last_feedback_version"] == 1
    assert final.payload["last_feedback_event_id"] == feedback_event_id
    assert final.vector == _vector(1)
def test_rejection_finalization_fences_stale_onboarding_snapshot() -> None:
    client, store = _user_store()
    initial_job = _job()
    rejected_event_id = _job()
    pending_rejection = {
        "event_id": rejected_event_id,
        "feedback_version": 1,
        "error_code": "EVENT_INVALID",
        "reason": "feedback event is invalid",
    }
    initial_payload = _user_payload(
        1,
        initial_job,
        feedback_version=1,
        feedback_event_id=rejected_event_id,
    )
    initial_payload.update(
        {
            "last_feedback_status": "rejected",
            PENDING_REJECTION_KEY: pending_rejection,
            FEEDBACK_STATE_REVISION_FIELD: 1,
        }
    )
    store.upsert_user(USER_ID, _vector(), payload=initial_payload)
    point_id = user_point_id(USER_ID)
    stale_onboarding_snapshot = _user_point(client, point_id)

    settings = replace(
        V2FeedbackSettings.from_env(),
        vector_dimension=VECTOR_SIZE,
        user_collection=USER_PROFILE_COLLECTION_CONTRACT.collection_name,
    )
    OrderedFeedbackApplier(client, settings).finalize_rejection(
        {
            "user_id": USER_ID,
            "event_id": rejected_event_id,
            "feedback_version": 1,
        }
    )

    stale_profile_payload = dict(stale_onboarding_snapshot.payload)
    stale_profile_payload.update(profile_version=2, job_id=_job())
    assert not store.compare_and_set_user(
        USER_ID,
        _vector(2),
        payload=stale_profile_payload,
        expected_point=stale_onboarding_snapshot,
    )
    final = _user_point(client, point_id)

    assert final.payload["profile_version"] == 1
    assert final.payload["job_id"] == initial_job
    assert final.payload[FEEDBACK_STATE_REVISION_FIELD] == 2
    assert PENDING_REJECTION_KEY not in final.payload
    assert final.vector == _vector()


def test_stale_onboarding_holder_cannot_overwrite_newer_profile() -> None:
    client, store = _user_store()
    initial_job = _job()
    store.upsert_user(
        USER_ID,
        _vector(),
        payload=_user_payload(1, initial_job),
    )
    point_id = user_point_id(USER_ID)
    stale_snapshot = _user_point(client, point_id)

    newest_job = _job()
    assert store.compare_and_set_user(
        USER_ID,
        _vector(1),
        payload=_user_payload(3, newest_job),
        expected_point=stale_snapshot,
    )
    assert not store.compare_and_set_user(
        USER_ID,
        _vector(2),
        payload=_user_payload(2, _job()),
        expected_point=stale_snapshot,
    )
    final = _user_point(client, point_id)

    assert final.payload["profile_version"] == 3
    assert final.payload["job_id"] == newest_job
    assert final.vector == _vector(1)


_USER_CAS_FIELDS = (
    "profile_version",
    "job_id",
    "last_feedback_version",
    "last_feedback_event_id",
    FEEDBACK_STATE_REVISION_FIELD,
)


def _user_writer(client, store, writer):
    if writer == "onboarding":
        return store.retrieve_user, lambda snapshot, payload: store.compare_and_set_user(
            USER_ID, _vector(1), payload=payload, expected_point=snapshot
        )
    applier = OrderedFeedbackApplier(
        client,
        replace(
            V2FeedbackSettings.from_env(),
            vector_dimension=VECTOR_SIZE,
            user_collection=USER_PROFILE_COLLECTION_CONTRACT.collection_name,
        ),
    )

    def write(snapshot, payload):
        try:
            applier._upsert(
                models.PointStruct(id=snapshot.id, vector=_vector(1), payload=payload),
                expected_payload=snapshot.payload,
            )
        except FeedbackWriteConflict:
            return False
        return True

    return applier._user, write


@pytest.mark.parametrize("writer", ["onboarding", "feedback"])
@pytest.mark.parametrize("cursor", ["absent", None, 0])
def test_user_verification_projects_only_cas_fields(monkeypatch, writer, cursor):
    client, store = _user_store()
    payload = {
        "profile_version": 1,
        "job_id": _job(),
        "feedback_rejections": [{"reason": "large-ledger-" * 50}] * 2000,
    }
    if cursor != "absent":
        payload.update(last_feedback_version=cursor, feedback_state_revision=cursor)
        payload["last_feedback_event_id"] = None
    store.upsert_user(USER_ID, _vector(), payload=payload)
    read, write = _user_writer(client, store, writer)
    retrieve = client.retrieve
    readbacks = []

    def capture(**kwargs):
        records = retrieve(**kwargs)
        readbacks.append((kwargs, records))
        return records

    monkeypatch.setattr(client, "retrieve", capture)
    snapshot = read(USER_ID)
    desired = dict(snapshot.payload)
    assert write(snapshot, desired)

    initial, verification = readbacks
    assert initial[0]["with_payload"] is True
    assert initial[0]["with_vectors"] is True
    assert "feedback_rejections" in initial[1][0].payload
    assert isinstance(verification[0]["with_payload"], list)
    assert set(verification[0]["with_payload"]) == set(_USER_CAS_FIELDS)
    assert verification[0]["with_vectors"] is False
    assert set(verification[1][0].payload) <= set(_USER_CAS_FIELDS)
    assert len(json.dumps(verification[1][0].payload)) < 500
    assert len(json.dumps(initial[1][0].payload)) > 1_000_000


@pytest.mark.parametrize("writer", ["onboarding", "feedback"])
@pytest.mark.parametrize("field", _USER_CAS_FIELDS)
def test_projected_user_verification_preserves_conflicts_and_retry(writer, field):
    client, store = _user_store()
    payload = _user_payload(1, _job())
    store.upsert_user(USER_ID, _vector(), payload=payload)
    read, write = _user_writer(client, store, writer)
    stale = read(USER_ID)
    competing_value = _job() if field in {"job_id", "last_feedback_event_id"} else 2
    client.set_payload(
        USER_PROFILE_COLLECTION_CONTRACT.collection_name,
        {field: competing_value},
        points=[stale.id],
        wait=True,
    )

    assert not write(stale, dict(stale.payload))
    fresh = read(USER_ID)
    assert fresh.payload[field] == competing_value
    assert write(fresh, dict(fresh.payload))
