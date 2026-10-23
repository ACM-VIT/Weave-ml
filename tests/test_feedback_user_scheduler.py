"""Bounded user lanes across polls, failure/reclaim and shutdown."""

import threading
import time
import uuid
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace

import pytest

from feedback.user_scheduler import UserFeedbackScheduler, Work
from feedback.v2 import OrderedFeedbackConsumer
from tests.test_v2_feedback_hardening import FakeRedis, _settings


def wait_until(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("worker did not reach expected state")
        time.sleep(0.005)


def test_parallel_users_and_single_ordered_lane_across_repeated_polls():
    hot, cold = str(uuid.uuid4()), str(uuid.uuid4())
    entered, release, cold_done = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    active, order = defaultdict(int), defaultdict(list)
    guard = threading.Lock()

    def apply(work):
        with guard:
            active[work.user_id] += 1
            assert active[work.user_id] == 1
        if work.user_id == hot and work.payload == 1:
            entered.set()
            assert release.wait(2)
        if work.user_id == cold:
            cold_done.set()
        with guard:
            order[work.user_id].append(work.payload)
            active[work.user_id] -= 1
        return True

    scheduler = UserFeedbackScheduler(apply, workers=2, capacity=4)
    try:
        assert scheduler.submit(Work("1-0", hot, 1))
        assert entered.wait(1)
        assert scheduler.submit(Work("2-0", hot, 2))
        for version in range(3, 10_003):
            assert not scheduler.submit(Work(f"{version}-hot", hot, version))
        # Hot lane cap does not retain the ten thousand deferred deliveries.
        assert not scheduler.submit(
            Work("1-0", hot, 1)
        )  # Active reclaim is not double scheduled.
        assert scheduler.submit(Work("4-0", cold, 1))
        assert cold_done.wait(1) and order[hot] == []
        release.set()
        scheduler.drain()
        assert scheduler.submit(Work("3-0", hot, 3))  # Later reclaim poll.
        scheduler.drain()
        assert order[hot] == [1, 2, 3]
        assert scheduler.completed() == 4 and scheduler.high_water <= 4
    finally:
        release.set()
        scheduler.close()


def test_failed_lane_retains_queued_work_for_reclaim_and_does_not_poison_other_users():
    entered, release = threading.Event(), threading.Event()
    calls = []

    def apply(work):
        calls.append(work.message_id)
        if work.message_id == "bad":
            entered.set()
            assert release.wait(2)
            raise RuntimeError("worker fault before durable acknowledgement")
        return True

    scheduler = UserFeedbackScheduler(apply, workers=2, capacity=4)
    try:
        scheduler.submit(Work("bad", "user", 1))
        assert entered.wait(1)
        scheduler.submit(Work("later", "user", 2))
        scheduler.submit(Work("other", "other", 1))
        wait_until(lambda: "other" in calls)
        release.set()
        scheduler.drain()
        assert (
            "later" not in calls and scheduler.errors == 1 and scheduler.deferred == 1
        )
        assert scheduler.submit(Work("later", "user", 2))
        scheduler.drain()
        assert "later" in calls
    finally:
        release.set()
        scheduler.close()


def test_close_rejects_new_work_releases_only_queued_and_waits_for_actual_worker_exit():
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def apply(work):
        calls.append(work.message_id)
        entered.set()
        assert release.wait(2)
        return True

    scheduler = UserFeedbackScheduler(apply, workers=1, capacity=2)
    scheduler.submit(Work("running", "user", 1))
    assert entered.wait(1)
    scheduler.submit(Work("queued", "user", 2))
    closer = threading.Thread(target=lambda: (scheduler.close(), closed.set()))
    closer.start()
    try:
        wait_until(lambda: scheduler.pending == 1)
        assert not closed.is_set()
        assert not scheduler.submit(Work("new", "other", 1))
        release.set()
        closer.join(2)
        assert closed.is_set() and scheduler.pending == 0 and calls == ["running"]
    finally:
        release.set()
        closer.join(2)


def test_consumer_keeps_lanes_across_live_polls_and_shutdown_keeps_lock_heartbeat():
    redis = FakeRedis()
    settings = replace(
        _settings(),
        consumer_workers=2,
        user_lock_ttl_seconds=0.3,
        user_lock_renew_interval_seconds=0.05,
        user_lock_wait_seconds=0.1,
    )
    hot, cold = str(uuid.uuid4()), str(uuid.uuid4())
    entered, release, cold_done = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    versions = defaultdict(list)

    def apply(payload):
        if payload["user_id"] == hot and payload["feedback_version"] == "1":
            entered.set()
            assert release.wait(2)
        versions[payload["user_id"]].append(payload["feedback_version"])
        if payload["user_id"] == cold:
            cold_done.set()
        return SimpleNamespace(status="applied")

    consumer = OrderedFeedbackConsumer(redis, SimpleNamespace(apply=apply), settings)
    consumer._messages = lambda: iter(
        [("1-0", {"user_id": hot, "feedback_version": "1"})]
    )
    consumer.run_once(wait=False)
    assert entered.wait(1)
    consumer._messages = lambda: iter(
        [
            ("2-0", {"user_id": hot, "feedback_version": "2"}),
            ("3-0", {"user_id": cold, "feedback_version": "1"}),
        ]
    )
    consumer.run_once(wait=False)
    try:
        assert cold_done.wait(1) and "1-0" not in redis.acks
        assert redis.renewed.wait(1)
        release.set()
        consumer._scheduler.drain()
        assert versions[hot] == ["1", "2"] and set(redis.acks) == {"1-0", "2-0", "3-0"}
    finally:
        release.set()
        consumer.close()


@pytest.mark.parametrize("workers", [1, 4])
def test_mixed_user_workload_measurement(workers, record_property):
    count, active, maximum = 40, 0, 0
    guard = threading.Lock()
    completions = []

    def apply(work):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.005)  # Matched dependency latency, never production throughput.
        with guard:
            active -= 1
            completions.append(time.monotonic())
        return True

    scheduler = UserFeedbackScheduler(apply, workers=workers, capacity=40)
    started = time.monotonic()
    try:
        for i in range(count):
            assert scheduler.submit(Work(str(i), str(i), i))
        scheduler.drain()
        elapsed = time.monotonic() - started
        assert len(completions) == count and maximum <= workers
        record_property("workers", workers)
        record_property("events_per_second", count / elapsed)
        record_property(
            "oldest_event_completion_age_seconds", max(completions) - started
        )
        record_property("retry_deferrals", scheduler.deferred)
        record_property("max_parallel_applies", maximum)
        record_property("high_water", scheduler.high_water)
    finally:
        scheduler.close()


def test_poison_gap_stays_pending_until_predecessor_is_reclaimed():
    from feedback.v2 import OrderedFeedbackApplier
    from tests.test_v2_feedback_hardening import FakeQdrant, _event

    user, repo = str(uuid.uuid4()), str(uuid.uuid4())
    redis, qdrant = FakeRedis(), FakeQdrant(user, repo)
    settings = _settings()
    consumer = OrderedFeedbackConsumer(
        redis, OrderedFeedbackApplier(qdrant, settings), settings
    )
    poison = _event(user, repo, 2, "invalid-action")
    try:
        assert not consumer._process_message("2-0", poison)
        assert redis.acks == [] and redis.dead == []
        assert qdrant.user.payload["last_feedback_version"] == 0
        assert consumer._process_message("1-0", _event(user, repo, 1))
        assert consumer._process_message("2-0", poison)
        assert consumer._process_message("3-0", _event(user, repo, 3, "save"))
        assert qdrant.user.payload["last_feedback_version"] == 3
        assert len(redis.dead) == 1
    finally:
        consumer.close()


@pytest.mark.parametrize("value", ["0", "17", "invalid"])
def test_worker_configuration_is_bounded(monkeypatch, value):
    from feedback.v2_settings import V2FeedbackSettings

    monkeypatch.setenv("FEEDBACK_CONSUMER_WORKERS", value)
    with pytest.raises(ValueError, match="FEEDBACK_CONSUMER_WORKERS"):
        V2FeedbackSettings.from_env()


def test_consumer_close_retains_owned_clients_and_renewal_until_apply_exits():
    redis = FakeRedis()
    settings = replace(
        _settings(),
        consumer_workers=1,
        user_lock_ttl_seconds=0.3,
        user_lock_renew_interval_seconds=0.05,
        user_lock_wait_seconds=0.1,
    )
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    client_closes = []

    def apply(_):
        entered.set()
        assert release.wait(2)
        assert client_closes == []
        return SimpleNamespace(status="applied")

    applier = SimpleNamespace(
        apply=apply, close=lambda: client_closes.append("applier")
    )
    redis.close = lambda: client_closes.append("redis")
    consumer = OrderedFeedbackConsumer(redis, applier, settings)
    consumer._owns_applier = consumer._owns_redis = True
    consumer._messages = lambda: iter([("1-0", {"user_id": str(uuid.uuid4())})])
    consumer.run_once(wait=False)
    assert entered.wait(1)
    closer = threading.Thread(target=lambda: (consumer.close(), closed.set()))
    closer.start()
    try:
        assert redis.renewed.wait(1)
        assert not closed.is_set() and client_closes == []
        release.set()
        closer.join(2)
        assert closed.is_set() and client_closes == ["applier", "redis"]
        assert redis.acks == ["1-0"]
    finally:
        release.set()
        closer.join(2)
