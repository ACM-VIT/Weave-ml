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
