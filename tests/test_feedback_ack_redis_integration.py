"""Opt-in HTTP + real Redis durable acceptance; never uses the runtime Redis URL."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import socket
import threading
import time
import os
from urllib.parse import urlsplit
import uuid

import pytest
import redis
import httpx
import uvicorn

from api import v2
from api.main import app
from feedback.v2 import DurableFeedbackProducer
from feedback.v2_settings import V2FeedbackSettings
from tests.test_feedback_ack_contract import event, isolated_runtime

pytestmark = pytest.mark.integration


@contextmanager
def live_http():
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(128)
    server = uvicorn.Server(uvicorn.Config(app, lifespan='off', log_level='error'))
    thread = threading.Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.01)
        assert server.started, 'isolated HTTP fixture did not start'
        with httpx.Client(base_url=f'http://127.0.0.1:{listener.getsockname()[1]}', timeout=5) as client:
            def post(events):
                return client.post('/api/v2/feedback/batch',
                    headers={'x-internal-secret': 'isolated-feedback-ack-secret'},
                    json={'schema_version': 2, 'events': events})
            yield post
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        assert not thread.is_alive(), 'isolated HTTP server failed to stop'


def test_http_ack_ids_match_real_redis_acceptance_and_partial_retry(monkeypatch):
    if os.getenv('RUN_FEEDBACK_ACK_REDIS_INTEGRATION') != '1':
        pytest.skip('requires an explicitly selected disposable Redis database')
    # This separate variable never falls back to REDIS_URL or .env runtime data.
    target = os.getenv('FEEDBACK_ACK_TEST_REDIS_URL', '')
    parsed = urlsplit(target)
    if not (parsed.scheme == 'redis' and parsed.hostname in {'127.0.0.1', 'localhost'}
            and parsed.port is not None and parsed.username is None and parsed.password is None
            and parsed.path == '/15' and not parsed.query and not parsed.fragment):
        pytest.fail('FEEDBACK_ACK_TEST_REDIS_URL must explicitly select disposable plain loopback Redis database 15')
    monkeypatch.setenv('APP_ENV', 'development')
    suffix = uuid.uuid4().hex
    stream = f'ml:feedback:v2:ack-test:{suffix}'
    client = redis.from_url(target, decode_responses=True, socket_timeout=2, socket_connect_timeout=2)
    settings = replace(V2FeedbackSettings.from_env(), redis_url=target, stream_name=stream, stream_maxlen=3)
    producer = DurableFeedbackProducer(client, settings)
    monkeypatch.setattr(v2, 'producer', lambda: producer)
    first, second, third, fourth = (event(index) for index in range(1, 5))
    first['event_id'] = first['event_id'].upper()
    ids = [str(uuid.UUID(row['event_id'])) for row in (first, second, third, fourth)]
    keys = [stream, *(f'{stream}:accepted:{identifier}' for identifier in ids)]
    with live_http() as post:
        try:
            assert client.ping()
            assert post([first]).json()['acknowledged_event_ids'] == [ids[0]]
            mixed = post([first, second])
            assert mixed.status_code == 202, mixed.text
            assert mixed.json() == {'accepted': 1, 'duplicates': 1, 'durable': True,
                                    'acknowledged_event_ids': ids[:2]}
            assert client.xlen(stream) == 2
            assert [fields['event_id'] for _, fields in client.xrange(stream)] == ids[:2]
            assert all(client.exists(key) for key in keys[1:3])

            conflict = post([third, {**first, 'event_type': 'save'}])
            assert conflict.status_code == 409, conflict.text
            assert 'acknowledged_event_ids' not in conflict.json()
            assert conflict.json()['details']['accepted'] == 1
            assert client.xlen(stream) == 3
            replay = post([third, first])
            assert replay.status_code == 202
            assert replay.json() == {'accepted': 0, 'duplicates': 2, 'durable': True,
                                     'acknowledged_event_ids': [ids[2], ids[0]]}
            assert client.xlen(stream) == 3

            overloaded = post([fourth])
            assert overloaded.status_code == 503
            assert 'acknowledged_event_ids' not in overloaded.json()
            assert not client.exists(keys[-1])
            assert client.xlen(stream) == 3
        finally:
            client.delete(*keys)  # Exact random test identities only, never FLUSHDB.
            client.close()
