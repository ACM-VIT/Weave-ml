"""HTTP ACKs identify only feedback whose durable enqueue completed successfully."""
from __future__ import annotations

from dataclasses import replace
import asyncio
import threading
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api import runtime, v2
from api.contracts import FeedbackResponse
from api.main import app
from feedback.v2 import DurableFeedbackProducer
from feedback.v2_settings import V2FeedbackSettings
from tests.test_v2_feedback_hardening import ProducerRedis


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch):
    monkeypatch.setenv('INTERNAL_API_SECRET', 'isolated-feedback-ack-secret')
    runtime.shutdown_service_runtime()
    yield
    for _, drain in runtime.service_runtime_drain_actions():
        drain()
    runtime.shutdown_service_runtime()


def event(version=1):
    return {'event_id': str(uuid.uuid4()), 'user_id': str(uuid.uuid4()),
            'repo_id': str(uuid.uuid4()), 'feedback_version': version, 'event_type': 'like',
            'occurred_at': '2026-10-23T00:00:00Z'}


def post(events):
    return TestClient(app).post('/api/v2/feedback/batch',
        headers={'x-internal-secret': 'isolated-feedback-ack-secret'},
        json={'schema_version': 2, 'events': events})


def producer_for(redis):
    return DurableFeedbackProducer(redis, replace(V2FeedbackSettings.from_env(), stream_name='isolated-ack-test'))


def test_actual_producer_mixed_replay_and_canonical_http_ack_ids(monkeypatch):
    redis = ProducerRedis()
    monkeypatch.setattr(v2, 'producer', lambda: producer_for(redis))
    first, second = event(), event(2)
    first['event_id'] = first['event_id'].upper()
    initial = post([first])
    assert initial.status_code == 202, initial.text
    assert initial.json() == {'accepted': 1, 'duplicates': 0, 'durable': True,
        'acknowledged_event_ids': [first['event_id'].lower()]}
    mixed = post([first, second])
    assert mixed.status_code == 202, mixed.text
    assert mixed.json() == {'accepted': 1, 'duplicates': 1, 'durable': True,
        'acknowledged_event_ids': [first['event_id'].lower(), second['event_id']]}
    replay = post([first, second])
    assert replay.status_code == 202
    assert replay.json() == {'accepted': 0, 'duplicates': 2, 'durable': True,
        'acknowledged_event_ids': [first['event_id'].lower(), second['event_id']]}
    assert len(redis.messages) == 2


def test_conflicting_payload_after_partial_progress_returns_no_ack(monkeypatch):
    redis = ProducerRedis()
    monkeypatch.setattr(v2, 'producer', lambda: producer_for(redis))
    original = event()
    assert post([original]).status_code == 202
    conflicting = {**original, 'event_type': 'save'}
    response = post([event(2), conflicting])
    assert response.status_code == 409
    assert 'acknowledged_event_ids' not in response.json()
    assert response.json()['details']['accepted'] == 1
    assert len(redis.messages) == 2  # Genuine partial durable progress is not a full ACK.


def test_partial_script_failure_and_pipeline_failure_return_no_ack(monkeypatch):
    for failure in (["1-0", RuntimeError('script unavailable')], RuntimeError('pipeline unavailable')):
        redis = MagicMock()
        redis.pipeline.return_value.execute.side_effect = failure if isinstance(failure, Exception) else [failure]
        monkeypatch.setattr(v2, 'producer', lambda: producer_for(redis))
        response = post([event(), event(2)])
        assert response.status_code == 503
        assert 'acknowledged_event_ids' not in response.json()
        assert response.headers['retry-after'] == '1'


def test_timeout_after_durable_acceptance_never_guesses_ack_ids(monkeypatch):
    monkeypatch.setenv('V2_FEEDBACK_TIMEOUT_SECONDS', '0.1')
    redis = ProducerRedis()
    producer = producer_for(redis)
    started, release = threading.Event(), threading.Event()
    original_wait_for = asyncio.wait_for
    controlled_deadlines = []

    async def wait_after_durable_acceptance(awaitable, timeout):
        if timeout == .1:
            # Exercise timeout AFTER durable enqueue. Starting the observation
            # clock before the worker arrives tests a different valid scenario.
            assert await asyncio.to_thread(started.wait, 5), 'durable worker did not start'
            controlled_deadlines.append(timeout)
        return await original_wait_for(awaitable, timeout)

    monkeypatch.setattr(runtime.asyncio, 'wait_for', wait_after_durable_acceptance)

    def delayed_return(events):
        counts = producer.enqueue(events)
        started.set()
        assert release.wait(10)
        return counts

    monkeypatch.setattr(v2, '_enqueue_feedback', delayed_return)
    try:
        response = post([event()])
        assert controlled_deadlines == [.1]
        assert started.is_set() and len(redis.messages) == 1
        assert response.status_code == 503
        assert 'acknowledged_event_ids' not in response.json()
        assert runtime.service_runtime_status()['feedback']['outstanding'] == 1
    finally:
        release.set()


def test_duplicate_request_identity_is_rejected_before_enqueue(monkeypatch):
    enqueue = MagicMock()
    monkeypatch.setattr(v2, '_enqueue_feedback', enqueue)
    original = event()
    response = post([original, {**original, 'event_id': original['event_id'].upper()}])
    assert response.status_code == 422
    assert 'acknowledged_event_ids' not in response.json()
    enqueue.assert_not_called()


def test_success_schema_requires_bounded_unique_ids_and_matching_counts():
    identifier = str(uuid.uuid4())
    parsed = FeedbackResponse.model_validate({'accepted': 1, 'duplicates': 0, 'durable': True,
        'acknowledged_event_ids': [identifier.upper()]})
    assert parsed.model_dump(mode='json')['acknowledged_event_ids'] == [identifier]
    for changes in [
        {'acknowledged_event_ids': []}, {'acknowledged_event_ids': [identifier, identifier.upper()], 'duplicates': 1},
        {'accepted': 0}, {'accepted': True}, {'duplicates': '0'},
        {'acknowledged_event_ids': [str(uuid.uuid4()) for _ in range(101)]},
    ]:
        with pytest.raises(ValidationError):
            FeedbackResponse.model_validate({'accepted': 1, 'duplicates': 0, 'durable': True,
                'acknowledged_event_ids': [identifier], **changes})
    schema = FeedbackResponse.model_json_schema()
    assert 'acknowledged_event_ids' in schema['required']
    field = schema['properties']['acknowledged_event_ids']
    assert field['minItems'] == 1 and field['maxItems'] == 100 and field['uniqueItems'] is True


def test_incomplete_producer_counts_cannot_be_published_as_a_success_ack(monkeypatch):
    monkeypatch.setattr(v2, '_enqueue_feedback', lambda events: (0, 0))
    response = TestClient(app, raise_server_exceptions=False).post('/api/v2/feedback/batch',
        headers={'x-internal-secret': 'isolated-feedback-ack-secret'},
        json={'schema_version': 2, 'events': [event()]})
    assert response.status_code == 500
    assert 'acknowledged_event_ids' not in response.json()
