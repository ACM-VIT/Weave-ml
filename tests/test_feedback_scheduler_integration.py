"""Opt-in two-process scheduler/reclaim proof on disposable Redis and Qdrant."""

from __future__ import annotations

import multiprocessing
import os
import threading
import time
import uuid
from dataclasses import replace
from urllib.parse import urlsplit

import pytest
import redis
from qdrant_client import QdrantClient, models

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
from feedback.v2 import (
    DurableFeedbackProducer,
    OrderedFeedbackApplier,
    OrderedFeedbackConsumer,
)
from feedback.v2_settings import V2FeedbackSettings

pytestmark = pytest.mark.integration


def consume(settings, stop, results):
    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    counts = {"lock_contention": 0, "retry_attempts": 0}
    guard = threading.Lock()
    original_set, original_incr = client.set, client.incr

    def measured_set(*args, **kwargs):
        result = original_set(*args, **kwargs)
        if kwargs.get("nx") and not result:
            with guard:
                counts["lock_contention"] += 1
        return result

    def measured_incr(*args, **kwargs):
        result = original_incr(*args, **kwargs)
        with guard:
            counts["retry_attempts"] += 1
        return result

    client.set, client.incr = measured_set, measured_incr
    qdrant = QdrantClient(url=settings.qdrant_url)
    consumer = OrderedFeedbackConsumer(
        client, OrderedFeedbackApplier(qdrant, settings), settings
    )
    try:
        while not stop.is_set():
            consumer.run_once(wait=False)
    finally:
        scheduler = consumer._scheduler
        consumer.close()
        results.put({**counts, "high_water": scheduler.high_water if scheduler else 0})
        qdrant.close()
        client.close()


def test_two_processes_reclaim_duplicates_poison_and_ordered_cursors(record_property):
    redis_url = os.getenv("FEEDBACK_SCHEDULER_TEST_REDIS_URL")
    qdrant_url = os.getenv("FEEDBACK_SCHEDULER_TEST_QDRANT_URL")
    if not redis_url or not qdrant_url:
        pytest.skip("Disposable scheduler services required")
    for url, port in [(redis_url, 6397), (qdrant_url, 6347)]:
        parsed = urlsplit(url)
        assert parsed.hostname in {"localhost", "127.0.0.1"} and parsed.port == port
    suffix = uuid.uuid4().hex
    settings = replace(
        V2FeedbackSettings.from_env(),
        environment="test",
        redis_url=redis_url,
        qdrant_url=qdrant_url,
        qdrant_api_key=None,
        vector_dimension=2,
        consumer_workers=2,
        stream_name=f"test:scheduler:{suffix}",
        consumer_group=f"test-{suffix}",
        heartbeat_key=f"test:scheduler:{suffix}:heartbeat",
        user_lock_prefix=f"test:scheduler:{suffix}:lock",
        dead_letter_stream=f"test:scheduler:{suffix}:dead",
        user_collection=f"test_users_{suffix}",
        repository_collection=f"test_repos_{suffix}",
        read_batch_size=4,
        read_block_ms=10,
        reclaim_idle_ms=100,
        max_delivery_attempts=1,
    )
    client = redis.Redis.from_url(redis_url, decode_responses=True)
    qdrant = QdrantClient(url=qdrant_url)
    users, repo = [str(uuid.uuid4()) for _ in range(4)], str(uuid.uuid4())
    processes = []
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    results = context.Queue()
    started = time.monotonic()
    try:
        qdrant.create_collection(
            settings.user_collection,
            vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
        )
        qdrant.create_collection(
            settings.repository_collection,
            vectors_config={
                settings.repository_vector_name: models.VectorParams(
                    size=2, distance=models.Distance.COSINE
                )
            },
        )
        qdrant.upsert(
            settings.user_collection,
            points=[
                models.PointStruct(
                    id=user,
                    vector=[1.0, 0.0],
                    payload={"user_id": user, "last_feedback_version": 0},
                )
                for user in users
            ],
            wait=True,
        )
        qdrant.upsert(
            settings.repository_collection,
            points=[
                models.PointStruct(
                    id=repo,
                    vector={settings.repository_vector_name: [0.0, 1.0]},
                    payload={
                        "repo_id": repo,
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
        events = []
        for user in users:
            # A poison version must be durably rejected before version3 can apply.
            for version, action in [(1, "like"), (2, "invalid-action"), (3, "save")]:
                events.append(
                    {
                        "event_id": str(uuid.uuid4()),
                        "user_id": user,
                        "repo_id": repo,
                        "feedback_version": str(version),
                        "event_type": action,
                        "dwell_ms": "",
                        "occurred_at": "2026-07-21T00:00:00Z",
                    }
                )
        producer = DurableFeedbackProducer(client, settings)
        assert producer.enqueue(events) == (12, 0)
        assert producer.enqueue([events[0]]) == (0, 1)
        client.xgroup_create(settings.stream_name, settings.consumer_group, id="0")
        # A terminated delivery owner has read the first event, without apply/ACK.
        assert client.xreadgroup(
            settings.consumer_group,
            "crashed-owner",
            {settings.stream_name: ">"},
            count=1,
        )
        # An ambiguous post-write redelivery must not apply the same version twice.
        client.xadd(settings.stream_name, events[0])
        for _ in range(2):
            process = context.Process(target=consume, args=(settings, stop, results))
            process.start()
            processes.append(process)
        deadline = time.monotonic() + 12
        points = []
        while time.monotonic() < deadline:
            points = qdrant.retrieve(
                settings.user_collection,
                ids=users,
                with_payload=True,
                with_vectors=True,
            )
            if (
                all(point.payload.get("last_feedback_version") == 3 for point in points)
                and client.xpending(settings.stream_name, settings.consumer_group)[
                    "pending"
                ]
                == 0
            ):
                break
            time.sleep(0.02)
        assert len(points) == 4 and all(
            point.payload["last_feedback_version"] == 3 for point in points
        )
        assert (
            client.xpending(settings.stream_name, settings.consumer_group)["pending"]
            == 0
        )
        assert client.xlen(settings.dead_letter_stream) == 4
        assert client.xlen(settings.stream_name) == 0
        record_property(
            "oldest_event_completion_age_seconds", time.monotonic() - started
        )
        for point in points:
            assert point.payload["last_feedback_event_id"] == next(
                e["event_id"]
                for e in events
                if e["user_id"] == point.id and e["feedback_version"] == "3"
            )
    finally:
        stop.set()
        for process in processes:
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(2)
            assert process.exitcode == 0
        if processes:
            measurements = [results.get(timeout=2) for _ in processes]
            record_property(
                "lock_acquisition_retries",
                sum(item["lock_contention"] for item in measurements),
            )
            record_property(
                "retry_attempts", sum(item["retry_attempts"] for item in measurements)
            )
            assert all(
                item["high_water"] <= settings.read_batch_size for item in measurements
            )
        for name in [settings.user_collection, settings.repository_collection]:
            if qdrant.collection_exists(name):
                qdrant.delete_collection(name)
        # Unique test namespace only. Never clear a shared Redis database.
        keys = list(client.scan_iter(f"test:scheduler:{suffix}*"))
        if keys:
            client.delete(*keys)
        client.close()
        qdrant.close()
