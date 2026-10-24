"""Phase 4 tests for contract-compatible user onboarding."""

import math
from types import SimpleNamespace

import pytest
from qdrant_client.http import models

from embedding.user_profile_store import QdrantUserProfileStore
from embedding.vector_contract import (
    FEEDBACK_STATE_REVISION_FIELD,
    USER_PROFILE_COLLECTION_CONTRACT,
    user_point_id,
)
from scripts.user_onboarding import (
    EMBEDDING_MODEL,
    TARGET_VECTOR_NAME,
    USER_PROFILES_COLLECTION,
    VECTOR_DIMENSION,
    UserOnboardingPipeline,
    synthesize_user_context,
)

USER_ID = "00000000-0000-4000-8000-000000000101"
JOB_ID = "00000000-0000-4000-8000-000000000201"


def _vector(value=1.0, *, dimension=VECTOR_DIMENSION):
    return [value] + [0.0] * (dimension - 1)


class FakeModel:
    def __init__(self, vector=None):
        self.vector = vector if vector is not None else _vector(2.0)
        self.calls = []

    def encode(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return self.vector


class FakeUserStore:
    def __init__(self):
        self.ensure_calls = 0
        self.upsert_call = None

    def ensure_collection(self):
        self.ensure_calls += 1

    def upsert_user(self, user_id, vector, *, payload=None):
        self.upsert_call = (user_id, vector, payload)


class FakeQdrantClient:
    def __init__(self, *, exists=True, vectors=None):
        self.exists = exists
        self.vectors = vectors or models.VectorParams(
            size=VECTOR_DIMENSION,
            distance=models.Distance.COSINE,
        )
        self.create_call = None
        self.upsert_call = None

    def collection_exists(self, collection_name):
        return self.exists

    def create_collection(self, **kwargs):
        self.create_call = kwargs
        self.vectors = kwargs["vectors_config"]
        self.exists = True

    def get_collection(self, collection_name):
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=self.vectors))
        )

    def upsert(self, **kwargs):
        self.upsert_call = kwargs

    def retrieve(self, **kwargs):
        return self.upsert_call["points"] if self.upsert_call else []


class CollectionCreationRaceClient(FakeQdrantClient):
    def __init__(self):
        super().__init__(exists=False)

    def create_collection(self, **kwargs):
        self.vectors = kwargs["vectors_config"]
        self.exists = True
        raise RuntimeError("collection already exists")


def test_onboarding_exports_match_the_frozen_user_collection_contract():
    assert USER_PROFILES_COLLECTION == USER_PROFILE_COLLECTION_CONTRACT.collection_name
    assert VECTOR_DIMENSION == USER_PROFILE_COLLECTION_CONTRACT.vector_size == 384
    assert EMBEDDING_MODEL == USER_PROFILE_COLLECTION_CONTRACT.model_name
    assert TARGET_VECTOR_NAME is USER_PROFILE_COLLECTION_CONTRACT.vector_name is None


def test_synthesize_user_context_is_deterministic_and_ignores_empty_values():
    context = synthesize_user_context(
        {
            "skills": ["Python", "  ML  ", None],
            "tech_stack": ["Qdrant"],
            "interests": [],
            "topics": ["vector-search", "machine-learning"],
            "bio": "  Builds search systems.  ",
        }
    )

    assert context == (
        "Skills: Python, ML. Tech Stack: Qdrant. "
        "Interests: vector-search, machine-learning. Bio: Builds search systems."
    )


@pytest.mark.parametrize("user_data", [{}, {"skills": [], "bio": "   "}, None])
def test_synthesize_user_context_rejects_empty_or_invalid_profiles(user_data):
    with pytest.raises(ValueError):
        synthesize_user_context(user_data)


def test_generate_interest_vector_is_finite_normalized_and_384_dimensional():
    model = FakeModel(_vector(2.0))
    pipeline = UserOnboardingPipeline(model=model)

    vector = pipeline.generate_interest_vector({"skills": ["Python"]})

    assert len(vector) == VECTOR_DIMENSION
    assert all(math.isfinite(value) for value in vector)
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0)
    assert model.calls[0][1]["normalize_embeddings"] is True
    vector[0] = 99.0
    reused = pipeline.generate_interest_vector({"skills": ["Python"], "profile_version": 2, "username": "new"})
    assert reused == _vector()
    reused[0] = 77.0
    assert pipeline.generate_interest_vector({"skills": ["Python"]}) == _vector()
    assert len(model.calls) == 1


@pytest.mark.parametrize("changed", ["bio", "skills", "revision", "sequence_length", "preprocessing", "model", "wrapped_encoder"])
def test_baseline_reuse_requires_identical_semantics_and_encoder_identity(changed, monkeypatch):
    import scripts.user_onboarding as onboarding
    model = FakeModel()
    model.query_embedding_revision = "revision-1"
    model._model = object()
    pipeline = UserOnboardingPipeline(model=model)
    profile = {"skills": ["Python"], "bio": "Builds tools"}
    pipeline.generate_interest_vector(profile)
    assert pipeline.generate_interest_vector(profile) == _vector()
    assert len(model.calls) == 1
    if changed in {"bio", "skills"}:
        profile[changed] = "changed"
    elif changed == "revision":
        model.query_embedding_revision = "revision-2"
    elif changed == "sequence_length":
        model.max_seq_length = 512
    elif changed == "preprocessing":
        monkeypatch.setattr(onboarding, "USER_BASELINE_PREPROCESSING_VERSION", "new-recipe")
    elif changed == "model":
        pipeline._model = FakeModel()
    else:
        model._model = object()
    assert pipeline.generate_interest_vector(profile) == _vector()
    assert len(model.calls) + (len(pipeline._model.calls) if pipeline._model is not model else 0) == 2


def test_baseline_cache_bounds_and_invalid_unit_vector_recompute():
    from array import array
    from dataclasses import replace
    model = FakeModel()
    pipeline = UserOnboardingPipeline(model=model)
    for index in range(129):
        pipeline.generate_interest_vector({"bio": f"Profile {index}"})
    cache = pipeline._baseline_cache
    assert cache.snapshot()["entries"] == 128
    assert cache.snapshot()["vector_bytes"] == 128 * VECTOR_DIMENSION * 8
    pipeline.generate_interest_vector({"bio": "Profile 0"})
    assert len(model.calls) == 130, "evicted baselines must be encoded again"
    key = next(reversed(cache._entries))
    entry = cache._entries[key]
    cache._entries[key] = replace(entry, data=array("d", _vector(2.0)).tobytes())
    assert pipeline.generate_interest_vector({"bio": "Profile 0"}) == _vector()
    assert len(model.calls) == 131, "a finite but non-unit cached vector is unusable"


def test_warm_baseline_does_not_skip_metadata_or_deadline_validation():
    from utils.deadline import CooperativeDeadlineExceeded, RequestDeadline, deadline_scope
    model = FakeModel()
    pipeline = UserOnboardingPipeline(model=model)
    pipeline.generate_interest_vector({"skills": ["Python"]})
    with pytest.raises(ValueError, match="profile_version"):
        pipeline.generate_interest_vector({"skills": ["Python"], "profile_version": True})
    deadline = RequestDeadline(1)
    deadline.cancel()
    with deadline_scope(deadline), pytest.raises(CooperativeDeadlineExceeded):
        pipeline.generate_interest_vector({"skills": ["Python"]})
    assert len(model.calls) == 1


@pytest.mark.parametrize(
    ("model_vector", "message"),
    [
        (_vector(dimension=3), "dimension 3, expected 384"),
        ([float("nan")] + [0.0] * 383, "must be finite"),
        ([0.0] * 384, "zero-length"),
    ],
)
def test_generate_interest_vector_rejects_incompatible_model_output(
    model_vector, message
):
    pipeline = UserOnboardingPipeline(model=FakeModel(model_vector))

    with pytest.raises(ValueError, match=message):
        pipeline.generate_interest_vector({"skills": ["Python"]})


def test_onboarding_rejects_a_model_that_does_not_match_the_384_dim_contract():
    with pytest.raises(ValueError, match="Unsupported onboarding embedding model"):
        UserOnboardingPipeline(embedding_model="bge-large-en-v1.5", model=FakeModel())


def test_pipeline_onboarding_stores_contract_metadata_and_feedback_accumulator():
    store = FakeUserStore()
    pipeline = UserOnboardingPipeline(model=FakeModel(), store=store)
    profile = {
        "topics": ["machine-learning"],
        "bio": "Builds recommendation systems.",
        "job_id": JOB_ID,
        "profile_version": 3,
    }

    assert pipeline.onboard_user(USER_ID, profile) is True

    assert store.ensure_calls == 1
    user_id, vector, payload = store.upsert_call
    assert user_id == USER_ID
    assert len(vector) == VECTOR_DIMENSION
    assert payload["user_id"] == USER_ID
    assert payload["job_id"] == JOB_ID
    assert payload["profile_version"] == 3
    assert payload["embedding_dim"] == VECTOR_DIMENSION
    assert payload["embedding_model"] == EMBEDDING_MODEL
    assert payload["preference_accumulator"] == vector
    assert profile["topics"] == ["machine-learning"]
    assert pipeline.onboard_user(USER_ID, {**profile, "profile_version": 4, "username": "renamed"}) is True
    assert len(pipeline.model.calls) == 1
    assert store.upsert_call[2]["profile_version"] == 4
    assert store.upsert_call[2]["username"] == "renamed"
    assert store.ensure_calls == 2, "vector reuse must not skip the current profile write"


def test_profile_update_preserves_learned_vector_and_feedback_cursor():
    learned_vector = [0.0, 1.0] + [0.0] * (VECTOR_DIMENSION - 2)
    learned_latent = [0.25, 0.75] + [0.0] * (VECTOR_DIMENSION - 2)

    class ExistingUserStore(FakeUserStore):
        def retrieve_user(self, user_id):
            return SimpleNamespace(
                id=user_id,
                vector=learned_vector,
                payload={
                    "user_id": user_id,
                    "profile_version": 3,
                    "last_feedback_version": 12,
                    "last_feedback_event_id": "event-12",
                    "feedback_latent_vector": learned_latent,
                    "feedback_adjustments": {"repo": {"reaction": {"action": "like"}}},
                    "feedback_applied_signals": {"repo": ["readme_open"]},
                    "feedback_rejections": [
                        {"event_id": "rejected-event", "reason": "invalid"}
                    ],
                    FEEDBACK_STATE_REVISION_FIELD: 13,
                    "last_feedback_status": "rejected",
                    "preference_accumulator": learned_latent,
                },
            )

    store = ExistingUserStore()
    pipeline = UserOnboardingPipeline(model=FakeModel(), store=store)
    new_baseline = pipeline.generate_interest_vector({"topics": ["databases"]})
    new_baseline = pipeline.generate_interest_vector({"topics": ["databases"], "profile_version": 4})
    assert len(pipeline.model.calls) == 1

    pipeline.save_to_qdrant(
        USER_ID,
        new_baseline,
        {"topics": ["databases"], "profile_version": 4, "job_id": JOB_ID},
    )

    _, stored_vector, payload = store.upsert_call
    assert stored_vector == learned_vector
    assert payload["profile_version"] == 4
    assert payload["last_feedback_version"] == 12
    assert payload["last_feedback_event_id"] == "event-12"
    assert payload["feedback_latent_vector"] == learned_latent
    assert payload["feedback_rejections"] == [
        {"event_id": "rejected-event", "reason": "invalid"}
    ]
    assert payload[FEEDBACK_STATE_REVISION_FIELD] == 13
    assert payload["last_feedback_status"] == "rejected"
    assert payload["preference_accumulator"] == learned_latent
    assert payload["profile_baseline_vector"] == new_baseline


def test_user_store_creates_an_unnamed_collection_and_deterministic_point():
    client = FakeQdrantClient(exists=False)
    store = QdrantUserProfileStore(client=client)

    store.ensure_collection()
    caller_payload = {"skills": ["Python"]}
    store.upsert_user(f" {USER_ID} ", _vector(2.0), payload=caller_payload)

    assert client.create_call["collection_name"] == USER_PROFILES_COLLECTION
    params = client.create_call["vectors_config"]
    assert params.size == VECTOR_DIMENSION
    assert params.distance == models.Distance.COSINE
    point = client.upsert_call["points"][0]
    assert point.id == user_point_id(USER_ID)
    assert isinstance(point.vector, list)
    assert len(point.vector) == VECTOR_DIMENSION
    assert point.payload["user_id"] == USER_ID
    assert caller_payload == {"skills": ["Python"]}


def test_user_identity_is_canonicalized_before_storage():
    user_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    client = FakeQdrantClient()
    store = QdrantUserProfileStore(client=client)

    store.upsert_user(user_id.upper(), _vector())

    point = client.upsert_call["points"][0]
    assert point.id == user_point_id(user_id)
    assert point.payload["user_id"] == user_id


def test_user_store_handles_first_onboarding_collection_creation_race():
    QdrantUserProfileStore(client=CollectionCreationRaceClient()).ensure_collection()


def test_user_store_preserves_the_previous_30_second_timeout_contract():
    store = QdrantUserProfileStore(client=FakeQdrantClient())
    assert store.timeout == 30


@pytest.mark.parametrize(
    "profile",
    [
        {"topics": ["ml"], "job_id": "not-a-uuid"},
        {"topics": ["ml"], "profile_version": 0},
        {"topics": ["ml"], "profile_version": True},
    ],
)
def test_onboarding_rejects_invalid_backend_job_metadata(profile):
    pipeline = UserOnboardingPipeline(model=FakeModel(), store=FakeUserStore())
    with pytest.raises(ValueError):
        pipeline.onboard_user(USER_ID, profile)


def test_onboarding_rejects_non_uuid_user_identity():
    model = FakeModel()
    pipeline = UserOnboardingPipeline(model=model, store=FakeUserStore())
    with pytest.raises(ValueError, match="backend-issued UUID"):
        pipeline.onboard_user("user-123", {"topics": ["ml"]})
    assert model.calls == []


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        (
            {"named_user_vector": models.VectorParams(
                size=VECTOR_DIMENSION,
                distance=models.Distance.COSINE,
            )},
            "must use one unnamed user vector",
        ),
        (
            models.VectorParams(size=3, distance=models.Distance.COSINE),
            "vector size 3, expected 384",
        ),
        (
            models.VectorParams(size=VECTOR_DIMENSION, distance=models.Distance.DOT),
            "uses distance",
        ),
    ],
)
def test_user_store_rejects_incompatible_existing_collection(vectors, message):
    store = QdrantUserProfileStore(client=FakeQdrantClient(vectors=vectors))

    with pytest.raises(ValueError, match=message):
        store.validate_collection()
