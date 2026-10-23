"""Authority deployment/diagnostic boundaries; owned services only when opted in."""
from contextlib import closing
import threading
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from embedding.corpus_revision import authority_from_environment
from tests.test_corpus_revision import proof
from tests.test_corpus_revision_operations import owned_authority_services


def configure(monkeypatch, redis_url, qdrant_url):
    for name, value in {
        'CORPUS_AUTHORITY_ENABLED': 'true',
        'CORPUS_AUTHORITY_MANAGED_BOUNDARY': 'single-endpoint-fenced-admin-v1',
        'CORPUS_AUTHORITY_REDIS_URL': redis_url, 'QDRANT_URL': qdrant_url,
        'APP_ENV': 'development',
    }.items():
        monkeypatch.setenv(name, value)


@pytest.mark.integration
def test_enabled_bootstrap_supplies_matching_authority_and_closes_owned_clients(
    owned_authority_services, monkeypatch,
):
    from qdrant_client import QdrantClient
    from config import QDRANT_COLLECTION_NAME, QDRANT_VECTOR_NAME
    from embedding.user_profile_store import QdrantUserProfileStore
    from scripts import bootstrap_qdrant
    redis_url, qdrant_url = owned_authority_services
    configure(monkeypatch, redis_url, qdrant_url)
    with closing(QdrantClient(url=qdrant_url, timeout=5, check_compatibility=False)) as setup:
        QdrantUserProfileStore(client=setup).ensure_collection()
    authority = authority_from_environment(collection=QDRANT_COLLECTION_NAME,
        vector=QDRANT_VECTOR_NAME, endpoint=qdrant_url)
    authority.bootstrap(proof())
    authority.close = Mock(wraps=authority.close)
    created = []
    def actual_client(**kwargs):
        client = QdrantClient(**kwargs)
        client.close = Mock(wraps=client.close)
        created.append(client)
        return client
    monkeypatch.setattr(bootstrap_qdrant, 'QDRANT_URL', qdrant_url)
    monkeypatch.setattr(bootstrap_qdrant, 'QDRANT_API_KEY', None)
    monkeypatch.setattr(bootstrap_qdrant, 'QdrantClient', actual_client)
    monkeypatch.setattr(bootstrap_qdrant, 'authority_from_environment', lambda **kwargs: authority, raising=False)
    try:
        result = bootstrap_qdrant.bootstrap()
        assert result['status'] == 'ready'
        assert result['qdrant_server_version'].startswith('1.18.')
        authority.close.assert_called_once()
        assert len(created) == 1
        created[0].close.assert_called_once()
    finally:
        authority.close()
        for client in created:
            client.close()


@pytest.mark.integration
def test_actual_scoped_factory_survives_reconnect_and_overdue_idle_probe(
    owned_authority_services, monkeypatch,
):
    import redis
    from urllib.parse import urlsplit
    redis_url, qdrant_url = owned_authority_services
    user = 'corpus_' + uuid4().hex[:8]
    password = 'owned-fixture-password-only'
    endpoint = urlsplit(redis_url)
    scoped_url = f'redis://{user}:{password}@{endpoint.netloc}'
    configure(monkeypatch, scoped_url, qdrant_url)
    with closing(redis.from_url(redis_url)) as admin:
        admin.execute_command('ACL', 'SETUSER', user, 'on', '>' + password,
            '~weave:corpus-authority:v1:*', '+eval', '+info', '+hget', '+hset')
        guarded = authority_from_environment(collection='owned-corpus', vector='repo_embedding', endpoint=qdrant_url)
        try:
            guarded.bootstrap(proof())
            assert guarded.status()['state'] == 'READY'
            connection = guarded.client.connection_pool.get_connection()
            connection.next_health_check = 0  # An overdue idle probe, without a 30s sleep.
            guarded.client.connection_pool.release(connection)
            assert guarded.read() is not None
            guarded.client.connection_pool.disconnect()
            token = guarded.begin()
            assert guarded.status()['state'] == 'DIRTY'
            guarded.finish(token, completed=True)
            assert guarded.status()['state'] == 'READY'
            # The actual scoped user does not have PING; protocol reads still work.
            with pytest.raises(redis.ResponseError):
                guarded.client.ping()
            assert guarded.read() is not None
        finally:
            guarded.close()


@pytest.mark.parametrize('state, healthy', [
    ('READY', True), ('DIRTY', True), ('UNTRUSTED', False), ('UNAVAILABLE', False), ('outage', False),
])
def test_readiness_includes_enabled_authority_state_without_rejecting_dirty(monkeypatch, state, healthy):
    from api import v2
    monkeypatch.setenv('V2_FEEDBACK_CONSUMER_REQUIRED', 'true')
    monkeypatch.setattr(v2, 'retriever', lambda: SimpleNamespace(health=lambda: {'qdrant_status': 'healthy'}))
    monkeypatch.setattr(v2, 'producer', lambda: SimpleNamespace(health=lambda: {
        'feedback_healthy': True, 'feedback_consumer_active': True,
    }))
    def status():
        if state == 'outage':
            raise RuntimeError('redis://sensitive-credentials-must-not-leak')
        return {'state': state, 'writers': 1 if state == 'DIRTY' else 0}
    monkeypatch.setattr(v2, 'corpus_authority', lambda: SimpleNamespace(status=status))
    actual_healthy, details = v2._collect_health_diagnostics(threading.Event())
    assert actual_healthy is healthy
    info = details['corpus_authority']
    assert info['enabled'] is True
    assert info['healthy'] is healthy
    assert info['reachable'] is (state != 'outage')
    assert 'sensitive-credentials' not in str(details)


def test_disabled_authority_reports_no_reachability_observation(monkeypatch):
    from api import v2
    monkeypatch.setattr(v2, 'retriever', lambda: SimpleNamespace(health=lambda: {'qdrant_status': 'healthy'}))
    monkeypatch.setattr(v2, 'corpus_authority', lambda: None)
    assert v2._qdrant_health()['corpus_authority'] == {
        'enabled': False, 'reachable': None, 'healthy': True, 'state': 'DISABLED', 'writers': None,
    }


@pytest.mark.integration
def test_real_qdrant_accepted_write_with_lost_http_reply_stays_dirty(
    owned_authority_services, monkeypatch,
):
    import socket
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import httpx
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import ResponseHandlingException
    from config import QDRANT_COLLECTION_NAME, QDRANT_VECTOR_NAME
    from embedding.qdrant_store import QdrantRepositoryStore
    from tests.test_v2_qdrant_fencing import _repository_result, REPO_ID

    redis_url, qdrant_url = owned_authority_services
    configure(monkeypatch, redis_url, qdrant_url)
    authority = authority_from_environment(collection=QDRANT_COLLECTION_NAME,
        vector=QDRANT_VECTOR_NAME, endpoint=qdrant_url)
    accepted = threading.Event()
    statuses, errors = [], []

    class LoseAcceptedReply(BaseHTTPRequestHandler):
        def do_PUT(self):
            try:
                body = self.rfile.read(int(self.headers['Content-Length']))
                with httpx.Client(timeout=5, trust_env=False) as forward:
                    response = forward.put(qdrant_url + self.path, content=body,
                        headers={'content-type': 'application/json'})
                    response.raise_for_status()
                    statuses.append(response.json()['result']['status'])
                accepted.set()  # The real server replied before the client loses its response.
            except Exception as exc:
                errors.append(type(exc).__name__)
            finally:
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), LoseAcceptedReply)
    server.daemon_threads = False  # server_close joins every owned forwarding handler.
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=.05))
    worker.start()
    direct = QdrantClient(url=qdrant_url, timeout=5, check_compatibility=False)
    faulted = QdrantClient(url=f'http://127.0.0.1:{server.server_port}', timeout=10, check_compatibility=False)
    try:
        authority.bootstrap(proof())
        store = QdrantRepositoryStore(client=direct, corpus_authority=authority)
        store.ensure_collection()
        store.upsert([_repository_result(1, str(uuid4()))])
        store.client = faulted  # Transparent owned fault proxy, same corpus and authority.
        with pytest.raises(ResponseHandlingException):
            store.upsert([_repository_result(2, str(uuid4()))])
        assert accepted.is_set() and statuses == ['completed'] and not errors
        current = direct.retrieve(store.collection_name, ids=[REPO_ID], with_payload=True)[0]
        assert current.payload['content_version'] == 2
        assert authority.status() == {'state': 'DIRTY', 'writers': 1}
        assert authority.read() is None
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)
        assert not worker.is_alive()
        faulted.close()
        direct.close()
        authority.close()
