import threading
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from config import (
    EMBEDDING_MODEL_REVISION,
    REPOSITORY_EMBEDDING_MODEL,
    REPOSITORY_EMBEDDING_VERSION,
    REPOSITORY_FEATURE_SPEC_VERSION,
)
from embedding.vector_contract import (
    REPOSITORY_SERVING_ELIGIBILITY_FIELD,
    REPOSITORY_SERVING_ELIGIBILITY_VERSION,
)
from feedback.event_handlers import ADJUSTMENTS_KEY, APPLIED_SIGNALS_KEY, LATENT_KEY, vector_delta
from feedback.interactions import INTERACTIONS, get_interaction
from feedback.v2 import (
    CONSUMER_HEARTBEAT,
    DurableFeedbackProducer,
    OrderedFeedbackApplier,
    OrderedFeedbackConsumer,
)


class FakeQdrant:
    def __init__(self, user_id, repo_id, last=0):
        self.user_id = user_id
        self.repo_id = repo_id
        self.user = SimpleNamespace(
            id=user_id,
            vector=[1.0, 0.0],
            payload={"last_feedback_version": last},
        )
        self.upserts = []

    def retrieve(self, collection_name, ids, with_payload, with_vectors):
        if ids[0] == self.user_id:
            return [self.user]
        return [SimpleNamespace(id=self.repo_id, vector={"repo_embedding": [0.0, 1.0]}, payload={
            "repo_id": self.repo_id,
            REPOSITORY_SERVING_ELIGIBILITY_FIELD: REPOSITORY_SERVING_ELIGIBILITY_VERSION,
            "content_version": 1,
            "embedding_model": REPOSITORY_EMBEDDING_MODEL,
            "embedding_model_revision": EMBEDDING_MODEL_REVISION,
            "embedding_version": REPOSITORY_EMBEDDING_VERSION,
            "embedding_dim": 2,
            "feature_spec_version": REPOSITORY_FEATURE_SPEC_VERSION,
        })]

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)
        self.user = kwargs["points"][0]


def event(user_id, repo_id, version, event_type="like"):
    return {"event_id": str(uuid.uuid4()), "user_id": user_id, "repo_id": repo_id,
            "feedback_version": str(version), "event_type": event_type, "dwell_ms": ""}


def test_v2_feedback_health_reports_dedicated_consumer_heartbeat():
    redis = MagicMock()
    redis.xinfo_groups.return_value = [
        {"name": "ml-feedback-v2", "pending": 2, "lag": 3}
    ]
    redis.get.return_value = "development|feedback-worker"

    health = DurableFeedbackProducer(redis).health()

    assert health["feedback_pending"] == 2
    assert health["feedback_lag"] == 3
    assert health["feedback_consumer_active"] is True


def test_consumer_refreshes_heartbeat_during_slow_processing(monkeypatch):
    redis = MagicMock()
    heartbeat_renewed = threading.Event()
    heartbeat_writes = 0

    def record_set(key, *args, **kwargs):
        nonlocal heartbeat_writes
        if key == CONSUMER_HEARTBEAT:
            heartbeat_writes += 1
            if heartbeat_writes >= 2:
                heartbeat_renewed.set()
        return True

    redis.set.side_effect = record_set
    applier = MagicMock()
    applier.apply.side_effect = lambda payload: (
        SimpleNamespace(status="applied")
        if heartbeat_renewed.wait(timeout=1)
        else pytest.fail("heartbeat was not refreshed while processing")
    )
    consumer = OrderedFeedbackConsumer(redis_client=redis, applier=applier)
    consumer._messages = lambda: iter(
        [("1-0", {"user_id": str(uuid.uuid4())})]
    )
    monkeypatch.setattr("feedback.v2.CONSUMER_HEARTBEAT_TTL_SECONDS", 0.3)

    assert consumer.run_once() == 1
    assert heartbeat_writes >= 3


def test_feedback_applies_version_with_vector_in_one_upsert():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    client = FakeQdrant(user_id, repo_id)
    result = OrderedFeedbackApplier(client).apply(event(user_id, repo_id, 1))
    assert result.status == "applied"
    point = client.upserts[0]["points"][0]
    assert point.payload["last_feedback_version"] == 1
    assert np.allclose(point.payload[LATENT_KEY], [0.85, 0.15])


def test_feedback_skips_duplicate_and_holds_version_gap():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    duplicate_event = event(user_id, repo_id, 2)
    duplicate_client = FakeQdrant(user_id, repo_id, last=2)
    duplicate_client.user.payload["last_feedback_event_id"] = duplicate_event["event_id"]
    duplicate = OrderedFeedbackApplier(duplicate_client).apply(duplicate_event)
    gap_client = FakeQdrant(user_id, repo_id, last=2)
    gap = OrderedFeedbackApplier(gap_client).apply(event(user_id, repo_id, 4))
    assert duplicate.status == "duplicate"
    assert gap.status == "gap"
    assert gap_client.upserts == []


@pytest.mark.parametrize(
    ("forward", "reverse"),
    [("like", "unlike"), ("dislike", "undislike"), ("save", "unsave")],
)
def test_reversal_clears_state_without_rewriting_learned_vector(forward, reverse):
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    client = FakeQdrant(user_id, repo_id)
    applier = OrderedFeedbackApplier(client)

    applier.apply(event(user_id, repo_id, 1, forward))
    assert not np.allclose(client.user.vector, [1.0, 0.0])
    learned_vector = np.asarray(client.user.vector)
    learned_latent = np.asarray(client.user.payload[LATENT_KEY])

    result = applier.apply(event(user_id, repo_id, 2, reverse))

    assert result.status == "applied"
    assert np.allclose(client.user.vector, learned_vector)
    assert np.allclose(client.user.payload[LATENT_KEY], learned_latent)
    assert repo_id not in client.user.payload[ADJUSTMENTS_KEY]
    assert client.user.payload["last_feedback_version"] == 2


def test_reversal_preserves_later_unrelated_feedback():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    client = FakeQdrant(user_id, repo_id)
    applier = OrderedFeedbackApplier(client)

    applier.apply(event(user_id, repo_id, 1, "like"))
    applier.apply(event(user_id, repo_id, 2, "readme_open"))
    learned_vector = np.asarray(client.user.vector)
    learned_latent = np.asarray(client.user.payload[LATENT_KEY])
    applier.apply(event(user_id, repo_id, 3, "unlike"))

    assert np.allclose(client.user.vector, learned_vector)
    assert np.allclose(client.user.payload[LATENT_KEY], learned_latent)
    assert repo_id not in client.user.payload[ADJUSTMENTS_KEY]


def test_forward_action_can_apply_again_after_zero_alpha_reversal():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    client = FakeQdrant(user_id, repo_id)
    applier = OrderedFeedbackApplier(client)

    applier.apply(event(user_id, repo_id, 1, "like"))
    applier.apply(event(user_id, repo_id, 2, "unlike"))
    latent_after_reversal = np.asarray(client.user.payload[LATENT_KEY])
    applier.apply(event(user_id, repo_id, 3, "like"))

    assert not np.allclose(client.user.payload[LATENT_KEY], latent_after_reversal)
    assert client.user.payload[ADJUSTMENTS_KEY][repo_id]["reaction"]["action"] == "like"
    assert client.user.payload["last_feedback_version"] == 3


@pytest.mark.parametrize("passive_action", ["readme_open", "github_open", "share"])
def test_passive_signal_is_applied_only_once(passive_action):
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    client = FakeQdrant(user_id, repo_id)
    applier = OrderedFeedbackApplier(client)

    applier.apply(event(user_id, repo_id, 1, passive_action))
    first_vector = np.asarray(client.user.vector)
    result = applier.apply(event(user_id, repo_id, 2, passive_action))

    assert result.status == "applied"
    assert np.allclose(client.user.vector, first_vector)
    assert client.user.payload[APPLIED_SIGNALS_KEY][repo_id] == [passive_action]
    assert client.user.payload["last_feedback_version"] == 2


def test_action_registry_is_complete_and_immutable():
    assert set(INTERACTIONS) == {
        "impression", "dwell", "readme_open", "github_open", "share", "like",
        "unlike", "dislike", "undislike", "save", "unsave",
    }
    assert get_interaction(" LIKE ").embedding_alpha == 0.15
    assert get_interaction("dislike").embedding_alpha == -0.15
    assert get_interaction("save").embedding_alpha == 0.20
    assert get_interaction("impression").realtime is False
    assert get_interaction("readme_open").apply_once is True
    assert get_interaction("dwell").apply_once is False
    assert {
        action: definition.reference_score
        for action, definition in INTERACTIONS.items()
    } == {
        "impression": 0.0,
        "dwell": 0.0,
        "readme_open": 0.2,
        "github_open": 0.3,
        "share": 0.6,
        "like": 1.0,
        "unlike": 0.0,
        "dislike": -1.0,
        "undislike": 0.0,
        "save": 0.8,
        "unsave": 0.0,
    }
    assert get_interaction("like").feedback_score == 1.0
    with pytest.raises(TypeError):
        INTERACTIONS["new"] = get_interaction("like")
    with pytest.raises(ValueError):
        get_interaction("star")


def test_active_vector_delta_preserves_interpolation_direction():
    assert vector_delta([1.0, 0.0], [0.0, 1.0], 0.5) == pytest.approx([-0.5, 0.5])
