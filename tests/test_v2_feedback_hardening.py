from __future__ import annotations

import json
import logging
import sys
import threading
import time
import uuid
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import (
    EMBEDDING_MODEL_REVISION,
    REPOSITORY_EMBEDDING_MODEL,
    REPOSITORY_EMBEDDING_VERSION,
    REPOSITORY_FEATURE_SPEC_VERSION,
)
from embedding.vector_contract import (
    FEEDBACK_STATE_REVISION_FIELD,
    REPOSITORY_SERVING_ELIGIBILITY_FIELD,
    REPOSITORY_SERVING_ELIGIBILITY_VERSION,
)
from feedback.event_handlers import ADJUSTMENTS_KEY, APPLIED_SIGNALS_KEY, LATENT_KEY
from feedback.safe_logging import BoundedJsonFormatter, configure_feedback_worker_logging
from feedback.user_lock import LockLostError, renewable_redis_lock
from feedback.v2 import (
    ACCEPT_LUA,
    ACK_DELETE_LUA,
    ApplyResult,
    DurableFeedbackProducer,
    FeedbackEventIdConflictError,
    FeedbackEnqueueError,
    FeedbackStateLimitError,
    FeedbackStreamFullError,
    FeedbackWriteConflict,
    MissingVectorError,
    OrderedFeedbackApplier,
    OrderedFeedbackConsumer,
    TrackedRepositoryLimitError,
    VersionEventConflict,
    DLQ_MOVE_LUA,
)
from feedback.v2_replay import DeadLetterReplayer
from feedback.v2_settings import V2FeedbackSettings


def _settings(**updates) -> V2FeedbackSettings:
    return replace(V2FeedbackSettings.from_env(), vector_dimension=2, **updates)


def _event(user_id: str, repo_id: str, version: int, event_type: str = "like"):
    return {
        "event_id": str(uuid.uuid4()),
        "user_id": user_id,
        "repo_id": repo_id,
        "feedback_version": str(version),
        "event_type": event_type,
        "dwell_ms": "",
        "occurred_at": "2026-07-21T00:00:00Z",
    }


class FakeQdrant:
    def __init__(self, user_id: str, repo_id: str, *, user_exists: bool = True) -> None:
        self.user_id = user_id
        self.repo_id = repo_id
        self.user = (
            SimpleNamespace(
                id=user_id,
                vector=[1.0, 0.0],
                payload={"user_id": user_id, "last_feedback_version": 0},
            )
            if user_exists
            else None
        )
        self.repo = SimpleNamespace(
            id=repo_id,
            vector={"repo_embedding": [0.0, 1.0]},
            payload={
                "repo_id": repo_id,
                REPOSITORY_SERVING_ELIGIBILITY_FIELD: REPOSITORY_SERVING_ELIGIBILITY_VERSION,
                "content_version": 1,
                "embedding_model": REPOSITORY_EMBEDDING_MODEL,
                "embedding_model_revision": EMBEDDING_MODEL_REVISION,
                "embedding_version": REPOSITORY_EMBEDDING_VERSION,
                "embedding_dim": 2,
                "feature_spec_version": REPOSITORY_FEATURE_SPEC_VERSION,
            },
        )
        self.upserts = 0
        self.repository_reads = 0

    def retrieve(self, *, collection_name, ids, with_payload, with_vectors):
        if collection_name == "user_profiles":
            return [self.user] if self.user is not None and str(self.user.id) in ids else []
        self.repository_reads += 1
        return [self.repo] if self.repo is not None and str(self.repo.id) in ids else []

    def upsert(self, *, collection_name, points, wait, **_kwargs):
        self.user = points[0]
        self.upserts += 1


class FakeRedis:
    """Small thread-safe Redis subset used to exercise lock/consumer behavior."""

    def __init__(self) -> None:
        self.values: dict[str, tuple[str, float | None]] = {}
        self.guard = threading.Lock()
        self.acks: list[str] = []
        self.dead: list[tuple[str, dict, dict]] = []
        self.attempts: dict[str, int] = {}
        self.renewals = 0
        self.renewed = threading.Event()
        self.group = None
        self.stream_lengths: dict[str, int] = {}

    def _current(self, key: str) -> str | None:
        stored = self.values.get(key)
        if stored is None:
            return None
        value, expiry = stored
        if expiry is not None and time.monotonic() >= expiry:
            self.values.pop(key, None)
            return None
        return value

    def set(self, key, value, nx=False, px=None, ex=None):
        with self.guard:
            if nx and self._current(key) is not None:
                return False
            ttl = (px / 1_000) if px is not None else ex
            expiry = time.monotonic() + ttl if ttl else None
            self.values[key] = (str(value), expiry)
            return True

    def get(self, key):
        with self.guard:
            return self._current(key)

    def eval(self, script, number_of_keys, *args):
        if "pexpire" in script:
            key, token, ttl = args
            with self.guard:
                if self._current(key) != token:
                    return 0
                self.values[key] = (token, time.monotonic() + int(ttl) / 1_000)
                self.renewals += 1
                self.renewed.set()
                return 1
        if script == DLQ_MOVE_LUA and number_of_keys == 2:
            dead_stream, dedupe_key, capacity, ttl, *fields = args
            if self._current(dedupe_key) is not None:
                return "duplicate"
            if self.stream_lengths.get(dead_stream, 0) >= int(capacity):
                return "overloaded"
            payload = dict(zip(fields[::2], fields[1::2], strict=True))
            self.dead.append((dead_stream, payload, {}))
            self.stream_lengths[dead_stream] = (
                self.stream_lengths.get(dead_stream, 0) + 1
            )
            self.set(dedupe_key, f"{len(self.dead)}-0", ex=int(ttl))
            return f"{len(self.dead)}-0"
        if "xack" in script and "xdel" in script:
            stream, group, message_id = args
            self.acks.append(str(message_id))
            self.stream_lengths[stream] = max(0, self.stream_lengths.get(stream, 0) - 1)
            return 1
        if "redis.call('get'" in script and "redis.call('del'" in script:
            key, token = args
            with self.guard:
                if self._current(key) != token:
                    return 0
                self.values.pop(key, None)
                return 1
        return "1-0"

    def xgroup_create(self, stream, group, id, mkstream):
        self.group = (stream, group)

    def xadd(self, stream, payload, **kwargs):
        self.dead.append((stream, dict(payload), dict(kwargs)))
        self.stream_lengths[stream] = self.stream_lengths.get(stream, 0) + 1
        return f"{len(self.dead)}-0"

    def incr(self, key):
        self.attempts[key] = self.attempts.get(key, 0) + 1
        return self.attempts[key]

    def expire(self, key, ttl):
        return True

    def delete(self, key):
        self.attempts.pop(key, None)
        self.values.pop(key, None)
        return 1

    def ping(self):
        return True

    def xinfo_groups(self, stream):
        return []

    def xlen(self, stream):
        return self.stream_lengths.get(stream, 0)

    def exists(self, key):
        return int(self._current(key) is not None)


class ProducerRedis:
    """Stateful ACCEPT/ACK Lua model for exact-capacity producer tests."""

    def __init__(self) -> None:
        self.accepted: dict[str, str] = {}
        self.messages: list[str] = []
        self.acks: list[str] = []

    def eval(self, script, number_of_keys, *args):
        if script == ACCEPT_LUA:
            dedupe_key, stream, fingerprint, ttl, capacity, *fields = args
            if dedupe_key in self.accepted:
                return (
                    "duplicate"
                    if self.accepted[dedupe_key] == fingerprint
                    else "event_id_conflict"
                )
            if len(self.messages) >= int(capacity):
                return "overloaded"
            self.accepted[dedupe_key] = fingerprint
            message_id = f"{len(self.messages) + 1}-0"
            self.messages.append(message_id)
            return message_id
        if script == ACK_DELETE_LUA:
            stream, group, message_id = args
            self.acks.append(message_id)
            if message_id in self.messages:
                self.messages.remove(message_id)
                return 1
            return 0
        raise AssertionError("unexpected Lua script")


@pytest.mark.parametrize("scheme", ["redis", "rediss"])
def test_settings_accept_authenticated_redis_transports_in_production(
    monkeypatch,
    scheme,
):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("REDIS_AUTH_MODE", "acl_url")
    monkeypatch.setenv(
        "REDIS_URL",
        f"{scheme}://ml-runtime:strong-test-password@redis.internal:6379/0",
    )

    settings = V2FeedbackSettings.from_env()

    assert settings.redis_url is not None
    assert settings.redis_url.startswith(f"{scheme}://")


def test_settings_reject_unsafe_production_and_lock_configuration(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("REDIS_URL", raising=False)
    with pytest.raises(ValueError, match="REDIS_URL is required"):
        V2FeedbackSettings.from_env()

    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("FEEDBACK_USER_LOCK_TTL_SECONDS", "10")
    monkeypatch.setenv("FEEDBACK_USER_LOCK_RENEW_INTERVAL_SECONDS", "5")
    with pytest.raises(ValueError, match="less than half"):
        V2FeedbackSettings.from_env()


def test_producer_atomic_script_uses_exact_outstanding_work_capacity():
    redis = MagicMock()
    redis.eval.return_value = "1-0"
    settings = _settings(stream_maxlen=321)
    producer = DurableFeedbackProducer(redis, settings)
    payload = _event(str(uuid.uuid4()), str(uuid.uuid4()), 1)

    assert producer.enqueue([payload]) == (1, 0)
    args = redis.eval.call_args.args
    assert "xlen" in args[0]
    assert "MAXLEN" not in args[0]
    assert args[6] == "321"


def test_exact_capacity_preserves_dedupe_and_ack_deletes_outstanding_work():
    redis = ProducerRedis()
    settings = _settings(stream_maxlen=1)
    producer = DurableFeedbackProducer(redis, settings)
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    first = _event(user_id, repo_id, 1)
    second = _event(user_id, repo_id, 2)

    assert producer.enqueue([first]) == (1, 0)
    assert producer.enqueue([first]) == (0, 1)
    with pytest.raises(FeedbackStreamFullError) as full:
        producer.enqueue([second])
    assert full.value.code == "FEEDBACK_STREAM_FULL"
    assert f"{settings.stream_name}:accepted:{second['event_id']}" not in redis.accepted
    assert len(redis.messages) == 1

    conflicting = {**first, "event_type": "save"}
    with pytest.raises(FeedbackEventIdConflictError) as conflict:
        producer.enqueue([conflicting])
    assert conflict.value.code == "EVENT_ID_PAYLOAD_CONFLICT"
    assert conflict.value.retryable is False

    consumer_stub = SimpleNamespace(redis=redis, settings=settings)
    OrderedFeedbackConsumer._ack(consumer_stub, "1-0")
    assert redis.messages == []
    assert redis.acks == ["1-0"]
    assert producer.enqueue([second]) == (1, 0)


def test_partial_batch_failure_is_retry_safe_and_reports_progress():
    redis = MagicMock()
    pipeline = redis.pipeline.return_value
    pipeline.execute.side_effect = [
        ["1-0", RuntimeError("redis unavailable")],
        ["duplicate", "2-0"],
    ]
    producer = DurableFeedbackProducer(redis, _settings())
    events = [
        _event(str(uuid.uuid4()), str(uuid.uuid4()), 1),
        _event(str(uuid.uuid4()), str(uuid.uuid4()), 1),
    ]
    with pytest.raises(FeedbackEnqueueError) as exc_info:
        producer.enqueue(events)
    assert exc_info.value.accepted == 1
    assert exc_info.value.duplicates == 0
    assert exc_info.value.failed_event_id == events[1]["event_id"]
    assert producer.enqueue(events) == (1, 1)


def test_health_reports_constant_time_stream_and_dlq_thresholds():
    redis = FakeRedis()
    settings = _settings(
        health_warn_dead_letter=0,
        health_max_dead_letter=1,
    )
    redis.stream_lengths[settings.stream_name] = 20
    redis.stream_lengths[settings.dead_letter_stream] = 2
    health = DurableFeedbackProducer(redis, settings).health()
    assert health["feedback_stream_length"] == 20
    assert health["feedback_dead_letter_length"] == 2
    assert health["feedback_status"] == "unhealthy"
    assert health["feedback_failures"] == ["dead_letter_length"]


@pytest.mark.parametrize(
    ("metric", "maximum_field"),
    [
        ("pending", "health_max_pending"),
        ("lag", "health_max_lag"),
        ("stream_length", "health_max_stream_length"),
        ("dead_letter_length", "health_max_dead_letter"),
    ],
)
def test_health_fails_at_the_exact_hard_threshold(metric, maximum_field):
    settings = _settings()
    maximum = getattr(settings, maximum_field)
    redis = FakeRedis()
    if metric in {"pending", "lag"}:
        redis.xinfo_groups = lambda _stream: [
            {
                "name": settings.consumer_group,
                "pending": maximum if metric == "pending" else 0,
                "lag": maximum if metric == "lag" else 0,
            }
        ]
    elif metric == "stream_length":
        redis.stream_lengths[settings.stream_name] = maximum
    else:
        redis.stream_lengths[settings.dead_letter_stream] = maximum

    health = DurableFeedbackProducer(redis, settings).health()

    assert health["feedback_healthy"] is False
    assert metric in health["feedback_failures"]


def test_health_derives_lag_when_redis_does_not_report_it():
    settings = _settings()
    redis = FakeRedis()
    redis.stream_lengths[settings.stream_name] = 12
    redis.xinfo_groups = lambda _stream: [
        {"name": settings.consumer_group, "pending": 5}
    ]

    health = DurableFeedbackProducer(redis, settings).health()

    assert health["feedback_lag"] == 7


def test_redis_older_than_xautoclaim_is_rejected_at_startup_and_health():
    class OldRedis(FakeRedis):
        def info(self, *, section):
            assert section == "server"
            return {"redis_version": "6.0.20"}

    settings = _settings()
    with pytest.raises(RuntimeError, match="Redis 6.2 or newer"):
        OrderedFeedbackConsumer(OldRedis(), MagicMock(), settings)
    with pytest.raises(RuntimeError, match="Redis 6.2 or newer"):
        DurableFeedbackProducer(OldRedis(), settings).health()


def test_heartbeat_key_cannot_alias_a_feedback_stream(monkeypatch):
    monkeypatch.setenv("FEEDBACK_STREAM_NAME", "ml:feedback:v2")
    monkeypatch.setenv("FEEDBACK_CONSUMER_HEARTBEAT_KEY", "ml:feedback:v2")

    with pytest.raises(ValueError, match="must differ from both feedback streams"):
        V2FeedbackSettings.from_env()


def test_missing_user_vector_stays_pending_until_bounded_retry_exhaustion():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    redis = FakeRedis()
    settings = _settings(max_delivery_attempts=2, dead_letter_maxlen=17)
    applier = OrderedFeedbackApplier(FakeQdrant(user_id, repo_id, user_exists=False), settings)
    consumer = OrderedFeedbackConsumer(redis, applier, settings)
    payload = _event(user_id, repo_id, 1)

    assert consumer._process_message("1-0", payload) is False
    assert redis.acks == []
    assert redis.dead == []

    assert consumer._process_message("1-0", payload) is True
    assert redis.acks == ["1-0"]
    stream, dead, kwargs = redis.dead[0]
    assert stream == settings.dead_letter_stream
    assert dead["failure_code"] == "RETRY_EXHAUSTED"
    assert dead["terminal_status"] == "USER_VECTOR_MISSING"
    assert kwargs == {}


def test_full_dlq_keeps_terminal_source_pending_without_trimming():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    settings = _settings(dead_letter_maxlen=1)
    redis = FakeRedis()
    redis.stream_lengths[settings.dead_letter_stream] = 1
    qdrant = FakeQdrant(user_id, repo_id)
    consumer = OrderedFeedbackConsumer(
        redis,
        OrderedFeedbackApplier(qdrant, settings),
        settings,
    )

    assert consumer._process_message(
        "1-0", _event(user_id, repo_id, 1, "not_supported")
    ) is False
    assert redis.acks == []
    assert redis.dead == []
    assert redis.stream_lengths[settings.dead_letter_stream] == 1
    assert qdrant.user.payload["last_feedback_version"] == 1


def test_cursor_advanced_rejection_remains_pending_until_dlq_move_succeeds():
    user_id, existing_repo_id, new_repo_id = (
        str(uuid.uuid4()),
        str(uuid.uuid4()),
        str(uuid.uuid4()),
    )
    settings = _settings(max_tracked_repositories=1, dead_letter_maxlen=1)
    redis = FakeRedis()
    redis.stream_lengths[settings.dead_letter_stream] = 1
    qdrant = FakeQdrant(user_id, new_repo_id)
    qdrant.user.payload[ADJUSTMENTS_KEY] = {
        existing_repo_id: {
            "reaction": {
                "action": "like",
                "delta": [-0.15, 0.15],
                "event_id": str(uuid.uuid4()),
            }
        }
    }
    event = _event(user_id, new_repo_id, 1, "like")
    consumer = OrderedFeedbackConsumer(
        redis,
        OrderedFeedbackApplier(qdrant, settings),
        settings,
    )

    assert consumer._process_message("1-0", event) is False
    assert qdrant.user.payload["last_feedback_status"] == "rejected"
    assert redis.acks == []
    assert consumer._process_message("1-0", event) is False
    assert redis.acks == []

    redis.stream_lengths[settings.dead_letter_stream] = 0
    assert consumer._process_message("1-0", event) is True
    assert redis.acks == ["1-0"]
    assert redis.dead[0][1]["failure_code"] == "TRACKED_REPOSITORY_LIMIT"


def test_version_gap_remains_pending_without_acknowledgement():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    redis = FakeRedis()
    settings = _settings(max_delivery_attempts=3)
    qdrant = FakeQdrant(user_id, repo_id)
    qdrant.user.payload["last_feedback_version"] = 1
    consumer = OrderedFeedbackConsumer(
        redis, OrderedFeedbackApplier(qdrant, settings), settings
    )

    assert consumer._process_message("3-0", _event(user_id, repo_id, 3)) is False
    assert redis.acks == []
    assert redis.attempts == {}, "ordering waits must not exhaust dependency-failure retries"


def test_terminal_invalid_event_advances_only_next_cursor_and_unblocks_followup():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    redis = FakeRedis()
    settings = _settings()
    qdrant = FakeQdrant(user_id, repo_id)
    consumer = OrderedFeedbackConsumer(
        redis, OrderedFeedbackApplier(qdrant, settings), settings
    )
    invalid = _event(user_id, repo_id, 1, "not_supported")

    assert consumer._process_message("1-0", invalid) is True
    assert qdrant.user.payload["last_feedback_version"] == 1
    assert qdrant.user.payload["last_feedback_status"] == "rejected"
    assert qdrant.user.payload["feedback_rejections"][0]["feedback_version"] == 1
    assert qdrant.user.payload[FEEDBACK_STATE_REVISION_FIELD] == 2
    assert redis.dead[0][1]["cursor_advanced"] == "1"

    assert consumer._process_message("2-0", _event(user_id, repo_id, 2)) is True
    assert qdrant.user.payload["last_feedback_version"] == 2
    assert qdrant.user.payload["last_feedback_status"] == "applied"
    assert qdrant.user.payload[FEEDBACK_STATE_REVISION_FIELD] == 3


def test_malformed_event_id_is_finalized_without_blocking_the_next_version():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    redis = FakeRedis()
    settings = _settings()
    qdrant = FakeQdrant(user_id, repo_id)
    consumer = OrderedFeedbackConsumer(
        redis, OrderedFeedbackApplier(qdrant, settings), settings
    )
    malformed = _event(user_id, repo_id, 1)
    malformed["event_id"] = "not-a-uuid"

    assert consumer._process_message("1-0", malformed) is True
    assert redis.acks == ["1-0"]
    assert redis.dead[0][1]["failure_code"] == "EVENT_INVALID"
    assert qdrant.user.payload["last_feedback_version"] == 1
    assert qdrant.user.payload["last_feedback_status"] == "rejected"
    assert "pending_feedback_rejection" not in qdrant.user.payload

    assert consumer._process_message("2-0", _event(user_id, repo_id, 2)) is True
    assert qdrant.user.payload["last_feedback_version"] == 2
    assert qdrant.user.payload["last_feedback_status"] == "applied"


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (("user_id", "not-a-uuid"), ("feedback_version", "1.0")),
)
def test_unreconcilable_identity_or_version_is_dlqed_without_worker_failure(
    field, invalid_value
):
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    redis = FakeRedis()
    settings = _settings()
    qdrant = FakeQdrant(user_id, repo_id)
    consumer = OrderedFeedbackConsumer(
        redis, OrderedFeedbackApplier(qdrant, settings), settings
    )
    malformed = _event(user_id, repo_id, 1)
    malformed[field] = invalid_value

    assert consumer._process_message("1-0", malformed) is True
    assert redis.acks == ["1-0"]
    assert redis.dead[0][1]["terminal_status"] == "rejected_unreconciled"
    assert redis.dead[0][1]["cursor_advanced"] == "0"
    assert qdrant.user.payload["last_feedback_version"] == 0


def test_duplicate_delivery_does_not_apply_vector_twice():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    redis = FakeRedis()
    settings = _settings()
    qdrant = FakeQdrant(user_id, repo_id)
    consumer = OrderedFeedbackConsumer(
        redis, OrderedFeedbackApplier(qdrant, settings), settings
    )
    payload = _event(user_id, repo_id, 1)
    assert consumer._process_message("1-0", payload)
    assert consumer._process_message("2-0", payload)
    assert qdrant.upserts == 1
    assert redis.acks == ["1-0", "2-0"]


def test_same_version_with_different_event_id_is_terminal_conflict():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    prior_event_id = str(uuid.uuid4())
    qdrant = FakeQdrant(user_id, repo_id)
    qdrant.user.payload.update(
        {"last_feedback_version": 1, "last_feedback_event_id": prior_event_id}
    )
    redis = FakeRedis()
    consumer = OrderedFeedbackConsumer(
        redis,
        OrderedFeedbackApplier(qdrant, _settings()),
        _settings(),
    )

    assert consumer._process_message("2-0", _event(user_id, repo_id, 1)) is True
    assert qdrant.upserts == 0
    assert redis.dead[0][1]["failure_code"] == "VERSION_EVENT_CONFLICT"
    assert redis.dead[0][1]["retryable"] == "0"
    assert redis.acks == ["2-0"]


def test_stale_feedback_cannot_overwrite_newer_onboarding_in_memory_qdrant():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    first_job_id, newer_job_id = str(uuid.uuid4()), str(uuid.uuid4())
    settings = _settings()
    client = QdrantClient(":memory:")
    client.create_collection(
        settings.user_collection,
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    client.create_collection(
        settings.repository_collection,
        vectors_config={
            settings.repository_vector_name: models.VectorParams(
                size=2,
                distance=models.Distance.COSINE,
            )
        },
    )
    client.upsert(
        settings.user_collection,
        [
            models.PointStruct(
                id=user_id,
                vector=[1.0, 0.0],
                payload={
                    "user_id": user_id,
                    "profile_version": 1,
                    "job_id": first_job_id,
                    "last_feedback_version": 0,
                },
            )
        ],
        wait=True,
    )
    client.upsert(
        settings.repository_collection,
        [
            models.PointStruct(
                id=repo_id,
                vector={settings.repository_vector_name: [0.0, 1.0]},
                payload={
                    "repo_id": repo_id,
                    REPOSITORY_SERVING_ELIGIBILITY_FIELD: REPOSITORY_SERVING_ELIGIBILITY_VERSION,
                    "content_version": 1,
                    "embedding_model": REPOSITORY_EMBEDDING_MODEL,
                    "embedding_model_revision": EMBEDDING_MODEL_REVISION,
                    "embedding_version": REPOSITORY_EMBEDDING_VERSION,
                    "embedding_dim": 2,
                    "feature_spec_version": REPOSITORY_FEATURE_SPEC_VERSION,
                },
            )
        ],
        wait=True,
    )

    class OnboardingWinsBeforeFeedbackCas:
        def __init__(self):
            self.raced = False

        def retrieve(self, **kwargs):
            return client.retrieve(**kwargs)

        def upsert(self, **kwargs):
            if kwargs.get("update_filter") is not None and not self.raced:
                self.raced = True
                client.upsert(
                    settings.user_collection,
                    [
                        models.PointStruct(
                            id=user_id,
                            vector=[0.0, 1.0],
                            payload={
                                "user_id": user_id,
                                "profile_version": 2,
                                "job_id": newer_job_id,
                                "last_feedback_version": 0,
                            },
                        )
                    ],
                    wait=True,
                    ordering=models.WriteOrdering.STRONG,
                )
            return client.upsert(**kwargs)

    racing_client = OnboardingWinsBeforeFeedbackCas()
    with pytest.raises(FeedbackWriteConflict):
        OrderedFeedbackApplier(racing_client, settings).apply(
            _event(user_id, repo_id, 1)
        )

    stored = client.retrieve(
        settings.user_collection,
        [user_id],
        with_payload=True,
        with_vectors=True,
    )[0]
    assert racing_client.raced is True
    assert stored.payload["profile_version"] == 2
    assert stored.payload["job_id"] == newer_job_id
    assert stored.payload["last_feedback_version"] == 0
    assert "last_feedback_event_id" not in stored.payload
    assert stored.vector == pytest.approx([0.0, 1.0])
    client.close()


def test_tracked_repository_cap_blocks_new_state_but_preserves_reversal():
    user_id, tracked_repo_id, new_repo_id = (
        str(uuid.uuid4()),
        str(uuid.uuid4()),
        str(uuid.uuid4()),
    )
    first_event_id = str(uuid.uuid4())
    qdrant = FakeQdrant(user_id, new_repo_id)
    qdrant.user.payload.update(
        {
            "last_feedback_version": 1,
            "last_feedback_event_id": first_event_id,
            LATENT_KEY: [0.85, 0.15],
            ADJUSTMENTS_KEY: {
                tracked_repo_id: {
                    "reaction": {
                        "action": "like",
                        "delta": [-0.15, 0.15],
                        "event_id": first_event_id,
                    }
                }
            },
            APPLIED_SIGNALS_KEY: {},
        }
    )
    settings = _settings(max_tracked_repositories=1)
    applier = OrderedFeedbackApplier(qdrant, settings)

    with pytest.raises(TrackedRepositoryLimitError) as limit:
        applier.apply(_event(user_id, new_repo_id, 2, "like"))
    assert limit.value.code == "TRACKED_REPOSITORY_LIMIT"
    assert qdrant.upserts == 0

    assert applier.apply(_event(user_id, tracked_repo_id, 2, "unlike")).status == "applied"
    assert tracked_repo_id not in qdrant.user.payload[ADJUSTMENTS_KEY]
    assert applier.apply(_event(user_id, new_repo_id, 3, "like")).status == "applied"
    assert new_repo_id in qdrant.user.payload[ADJUSTMENTS_KEY]


def test_feedback_ledger_has_an_exact_serialized_byte_ceiling():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    qdrant = FakeQdrant(user_id, repo_id)
    base_size = OrderedFeedbackApplier._feedback_state_bytes(qdrant.user.payload)
    settings = _settings(max_user_state_bytes=base_size + 16)
    applier = OrderedFeedbackApplier(qdrant, settings)

    with pytest.raises(FeedbackStateLimitError) as limit:
        applier.apply(_event(user_id, repo_id, 1, "like"))

    assert limit.value.code == "USER_STATE_SIZE_LIMIT"
    assert qdrant.upserts == 0


def test_incompatible_repository_vector_never_updates_user_profile():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    qdrant = FakeQdrant(user_id, repo_id)
    qdrant.repo.payload["embedding_version"] = "incompatible-version"
    applier = OrderedFeedbackApplier(qdrant, _settings())

    with pytest.raises(MissingVectorError) as missing:
        applier.apply(_event(user_id, repo_id, 1, "like"))

    assert missing.value.code == "REPOSITORY_VECTOR_MISSING"
    assert qdrant.upserts == 0


def test_two_consumers_serialize_same_user():
    user_id = str(uuid.uuid4())
    redis = FakeRedis()
    settings = _settings(user_lock_wait_seconds=1.0)
    active = 0
    maximum_active = 0
    guard = threading.Lock()
    entered = threading.Event()
    release = threading.Event()

    class SlowApplier:
        def apply(self, payload):
            nonlocal active, maximum_active
            with guard:
                active += 1
                maximum_active = max(maximum_active, active)
                entered.set()
            release.wait(timeout=1)
            with guard:
                active -= 1
            return ApplyResult("applied", 1)

    first = OrderedFeedbackConsumer(redis, SlowApplier(), settings)
    second = OrderedFeedbackConsumer(redis, SlowApplier(), settings)
    payload_one = {"user_id": user_id, "event_id": str(uuid.uuid4())}
    payload_two = {"user_id": user_id, "event_id": str(uuid.uuid4())}
    thread_one = threading.Thread(target=first._process_message, args=("1-0", payload_one))
    thread_two = threading.Thread(target=second._process_message, args=("2-0", payload_two))
    thread_one.start()
    assert entered.wait(timeout=1)
    thread_two.start()
    release.set()
    thread_one.join(timeout=1)
    thread_two.join(timeout=1)
    assert maximum_active == 1


def test_renewable_lock_renews_and_detects_lost_ownership():
    redis = FakeRedis()
    with renewable_redis_lock(
        redis,
        "test:lock",
        ttl_ms=300,
        wait_seconds=0,
        renew_interval_seconds=0.05,
    ) as lock:
        assert redis.renewed.wait(timeout=1)
        lock.assert_owned()
    assert redis.renewals >= 1

    class LostRedis(FakeRedis):
        def eval(self, script, number_of_keys, *args):
            if "pexpire" in script:
                self.renewed.set()
                return 0
            return super().eval(script, number_of_keys, *args)

    lost = LostRedis()
    with pytest.raises(LockLostError):
        with renewable_redis_lock(
            lost,
            "test:lost",
            ttl_ms=300,
            wait_seconds=0,
            renew_interval_seconds=0.05,
        ) as lock:
            assert lost.renewed.wait(timeout=1)
            lock.assert_owned()


def test_dlq_replay_is_dry_run_by_default_and_execute_is_idempotent():
    source_id = "123-0"
    payload = {
        **_event(str(uuid.uuid4()), str(uuid.uuid4()), 1),
        "retryable": "1",
        "failure_code": "RETRY_EXHAUSTED",
    }
    redis = MagicMock()
    redis.xrange.return_value = [(source_id, payload)]
    redis.get.side_effect = [None, None, f"{payload['event_id']}|456-0"]
    redis.eval.return_value = "456-0"
    replayer = DeadLetterReplayer(redis, _settings(stream_maxlen=777))

    assert replayer.replay(source_id).status == "dry_run"
    redis.eval.assert_not_called()
    assert replayer.replay(source_id, execute=True).status == "replayed"
    assert replayer.replay(source_id, execute=True).status == "duplicate"
    assert redis.eval.call_count == 1
    assert redis.eval.call_args.args[7] == "777"
    assert "xdel" in redis.eval.call_args.args[0]
    assert "MAXLEN" not in redis.eval.call_args.args[0]


def test_dlq_replay_overload_leaves_source_entry_untouched():
    source_id = "123-0"
    payload = {
        **_event(str(uuid.uuid4()), str(uuid.uuid4()), 1),
        "retryable": "1",
    }
    redis = MagicMock()
    redis.get.return_value = None
    redis.xrange.return_value = [(source_id, payload)]
    redis.eval.return_value = "overloaded"

    with pytest.raises(FeedbackStreamFullError):
        DeadLetterReplayer(redis, _settings(stream_maxlen=1)).replay(
            source_id, execute=True
        )

    redis.xrange.assert_called_once()
    assert redis.eval.call_args.args[5].endswith(":dead")


def test_dlq_replay_refuses_terminal_event_without_override():
    source_id = "123-0"
    redis = MagicMock()
    redis.get.return_value = None
    redis.xrange.return_value = [
        (source_id, {**_event(str(uuid.uuid4()), str(uuid.uuid4()), 1), "retryable": "0"})
    ]
    with pytest.raises(ValueError, match="terminal-invalid"):
        DeadLetterReplayer(redis, _settings()).replay(source_id)


def test_dlq_replay_refuses_cursor_advanced_terminal_event_even_with_override():
    source_id = "123-0"
    redis = MagicMock()
    redis.get.return_value = None
    redis.xrange.return_value = [
        (
            source_id,
            {
                **_event(str(uuid.uuid4()), str(uuid.uuid4()), 1),
                "retryable": "0",
                "cursor_advanced": "1",
            },
        )
    ]

    with pytest.raises(ValueError, match="compensating event"):
        DeadLetterReplayer(redis, _settings()).replay(
            source_id,
            execute=True,
            allow_terminal=True,
        )

    redis.eval.assert_not_called()


def test_feedback_json_formatter_bounds_context_and_never_emits_tracebacks():
    try:
        raise RuntimeError("redis://worker:password@redis:6379 token=raw-secret")
    except RuntimeError:
        exception_info = sys.exc_info()
    record = logging.LogRecord(
        name="feedback.v2",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="dependency failed at redis://worker:password@redis:6379 token=raw-secret",
        args=(),
        exc_info=exception_info,
    )
    record.feedback_context = {
        "event_id": "event-1",
        "api_token": "raw-secret",
        "user_vector": [1.0, 2.0],
        "detail": "x" * 1_000,
    }

    rendered = BoundedJsonFormatter().format(record)
    payload = json.loads(rendered)
    assert payload["context"]["event_id"] == "event-1"
    assert payload["context"]["api_token"] == "[redacted]"
    assert payload["context"]["user_vector"] == "[redacted]"
    assert len(payload["context"]["detail"]) == 256
    assert "raw-secret" not in rendered
    assert "worker:password" not in rendered
    assert "Traceback" not in rendered
    assert "RuntimeError" not in rendered


def test_feedback_worker_logging_writes_structured_json_to_stdout(capsys):
    package_logger = logging.getLogger("feedback")
    previous_handlers = list(package_logger.handlers)
    previous_level = package_logger.level
    previous_propagate = package_logger.propagate
    try:
        configure_feedback_worker_logging("INFO")
        logging.getLogger("feedback.worker").info(
            "event processed",
            extra={
                "feedback_context": {
                    "event_id": "event-1",
                    "status": "applied",
                    "code": "OK",
                }
            },
        )
        output = capsys.readouterr().out.strip()
        assert json.loads(output)["context"] == {
            "code": "OK",
            "event_id": "event-1",
            "status": "applied",
        }
    finally:
        for handler in package_logger.handlers:
            handler.close()
        package_logger.handlers = previous_handlers
        package_logger.setLevel(previous_level)
        package_logger.propagate = previous_propagate


def test_consumer_read_batch_reuses_repository_snapshot_without_reusing_user_state():
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    settings = _settings()
    events = [_event(user_id, repo_id, i, action) for i, action in enumerate(
        ["like", "readme_open", "save", "unlike", "like"], start=1,
    )]
    baseline = FakeQdrant(user_id, repo_id)
    baseline_applier = OrderedFeedbackApplier(baseline, settings)
    for payload in events:
        baseline_applier.apply(payload)

    qdrant, redis = FakeQdrant(user_id, repo_id), FakeRedis()
    redis.xautoclaim = lambda *args, **kwargs: ("0-0", [])
    redis.xreadgroup = lambda *args, **kwargs: [
        (settings.stream_name, [(f"{i}-0", payload) for i, payload in enumerate(events, start=1)])
    ]
    consumer = OrderedFeedbackConsumer(redis, OrderedFeedbackApplier(qdrant, settings), settings)

    assert consumer.run_once() == len(events)
    assert qdrant.repository_reads == 1
    assert baseline.repository_reads == 4
    assert qdrant.user.vector == baseline.user.vector
    assert qdrant.user.payload == baseline.user.payload
    assert qdrant.upserts == len(events)
    assert len(redis.acks) == len(events)


@pytest.mark.parametrize("change", ["update", "removed", "ineligible", "model_revision", "content_revision"])
def test_repository_read_batch_refetches_revision_and_eligibility_at_each_boundary(change):
    user_id, repo_id = str(uuid.uuid4()), str(uuid.uuid4())
    qdrant = FakeQdrant(user_id, repo_id)
    applier = OrderedFeedbackApplier(qdrant, _settings())
    first = _event(user_id, repo_id, 1, "like")
    second = _event(user_id, repo_id, 2, "readme_open")
    with applier.repository_read_batch([first]):
        assert applier.apply(first).status == "applied"

    previous_payload = json.loads(json.dumps(qdrant.user.payload))
    if change == "update":
        qdrant.repo.payload["content_version"] = 2
        qdrant.repo.vector = {"repo_embedding": [1.0, 0.0]}
    elif change == "removed":
        qdrant.repo = None
    elif change == "ineligible":
        qdrant.repo.payload.pop(REPOSITORY_SERVING_ELIGIBILITY_FIELD)
    elif change == "model_revision":
        qdrant.repo.payload["embedding_model_revision"] = "stale-model-revision"
    else:
        qdrant.repo.payload["content_version"] = 0

    with applier.repository_read_batch([second]):
        if change == "update":
            assert applier.apply(second).status == "applied"
            assert qdrant.user.payload[LATENT_KEY] == pytest.approx([0.8575, 0.1425])
        else:
            with pytest.raises(MissingVectorError):
                applier.apply(second)
            assert qdrant.user.payload == previous_payload
    assert qdrant.repository_reads == 2


@pytest.mark.parametrize("prefetch_fails", [False, True])
def test_repository_read_batch_partial_failure_does_not_poison_future_batches(prefetch_fails):
    user_id, repo_id, missing_id = (str(uuid.uuid4()) for _ in range(3))
    qdrant = FakeQdrant(user_id, repo_id)
    retrieve = qdrant.retrieve

    def maybe_fail(**kwargs):
        if prefetch_fails and kwargs["collection_name"] != "user_profiles" and len(kwargs["ids"]) > 1:
            qdrant.repository_reads += 1
            raise RuntimeError("batch transport fixture")
        return retrieve(**kwargs)

    qdrant.retrieve = maybe_fail
    applier = OrderedFeedbackApplier(qdrant, _settings())
    first, second = _event(user_id, repo_id, 1), _event(user_id, missing_id, 2)
    with applier.repository_read_batch([first, second]):
        assert applier.apply(first).status == "applied"
        with pytest.raises(MissingVectorError):
            applier.apply(second)
    assert qdrant.user.payload["last_feedback_version"] == 1
    # Missing results and failed prefetches are discarded at the batch boundary.
    qdrant.repo.id = missing_id
    qdrant.repo.payload["repo_id"] = missing_id
    with applier.repository_read_batch([second]):
        assert applier.apply(second).status == "applied"
    assert qdrant.user.payload["last_feedback_version"] == 2
    assert qdrant.repository_reads == (4 if prefetch_fails else 2)
