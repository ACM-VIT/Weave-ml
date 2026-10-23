"""Lifecycle regression evidence using real Qdrant filtering and feature CAS."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch
import uuid

import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from api.contracts import RepositoryRefreshJob
from api.v2 import _refresh_repository_job_locked
from config import QDRANT_VECTOR_NAME, REPOSITORY_EMBEDDING_DIM
from ingestion.lifecycle import RepositoryLifecycleObservation, merge_lifecycle
from retrieval.v2_retriever import QdrantV2Retriever, REPOSITORY_DISCOVERY_CHANNELS
from tests.test_v2_retrieval import _repository_payload, VECTOR
from tests.test_v2_summary_contract import _indexed_feature_refresh_fixture, OwnedLock


def observation(archived=False, disabled=False, *, day=1, status='verified'):
    item = dict(status=status, source='github-graphql-repository', observed_at=f'2026-01-{day:02}T00:00:00Z')
    if status == 'verified':
        item.update(archived=archived, disabled=disabled)
    return item


def test_verified_evidence_orders_restoration_and_preserves_unavailable_status():
    archived = merge_lifecycle({}, observation(archived=True, day=2))
    failed = merge_lifecycle(archived, observation(day=4, status='unavailable'))
    assert failed['lifecycle_archived'] is True
    assert failed['lifecycle_observed_at'] == archived['lifecycle_observed_at']
    assert merge_lifecycle(failed, observation(day=1))['lifecycle_archived'] is True
    # A verified restoration is newer than verified evidence; a failed attempt
    # does not suppress it just because that attempt happened later.
    revived = merge_lifecycle(failed, observation(day=3))
    assert revived['lifecycle_archived'] is False
    assert revived['lifecycle_last_attempt_at'] == failed['lifecycle_last_attempt_at']
    assert merge_lifecycle(revived, None) == revived
    with pytest.raises(ValueError, match='conflicting'):
        merge_lifecycle(archived, observation(day=2))


@pytest.mark.parametrize('change', [
    {'archived': None}, {'disabled': 'false'}, {'source': 'push-recency'},
    {'observed_at': '2026-01-01T00:00:00'}, {'unexpected': True},
    {'status': 'unavailable'},
])
def test_lifecycle_contract_rejects_partial_inferred_and_ambiguous_evidence(change):
    with pytest.raises(ValidationError):
        RepositoryLifecycleObservation.model_validate({**observation(), **change})


def test_future_observation_cannot_poison_monotone_status():
    with pytest.raises(ValueError, match='future'):
        merge_lifecycle({}, {**observation(), 'observed_at': (datetime.now(timezone.utc)+timedelta(days=1)).isoformat()})


@pytest.mark.parametrize('flag', ['archived', 'disabled'])
def test_prelimit_exclusion_reaches_semantic_and_every_discovery_channel(monkeypatch, flag):
    monkeypatch.setenv('V2_REPOSITORY_LIFECYCLE_POLICY', 'verified-status-v1')
    client = QdrantClient(':memory:')
    collection = 'lifecycle-regression'
    client.create_collection(collection, vectors_config={QDRANT_VECTOR_NAME: models.VectorParams(
        size=REPOSITORY_EMBEDDING_DIM, distance=models.Distance.COSINE)})
    excluded, stable, unknown = [str(uuid.uuid4()) for _ in range(3)]
    rows = []
    for repo_id, status, high in [(excluded, observation(**{flag: True}), 1000),
                                  (stable, observation(), 2), (unknown, None, 1)]:
        payload = _repository_payload(repo_id, stars=high, **merge_lifecycle({}, status))
        payload.update({key: high for key in REPOSITORY_DISCOVERY_CHANNELS.values()})
        # Age does not establish death; stable repositories remain selectable.
        payload['pushed_days_ago'] = 50_000
        rows.append(models.PointStruct(id=repo_id, vector={QDRANT_VECTOR_NAME: VECTOR}, payload=payload))
    client.upsert(collection, rows)
    retriever = QdrantV2Retriever(client=client, repository_collection=collection, heavy_ranker_enabled=False)
    points = retriever._fetch_semantic(VECTOR, 2, include_vectors=False, exclude_repo_ids=None).points
    assert {str(point.id) for point in points} == {stable, unknown}
    for channel in REPOSITORY_DISCOVERY_CHANNELS:
        points = retriever._fetch_discovery_channel(channel, 1, include_vectors=False, exclude_repo_ids=None)
        assert [str(point.id) for point in points] == [stable]
    bad = SimpleNamespace(id=excluded, payload=rows[0].payload)
    assert retriever._eligible_repository_point(bad, require_vector=False) is None
    # Completed status refresh is observed by the next bounded request.
    client.set_payload(collection, merge_lifecycle(rows[0].payload, observation(day=2)), [excluded])
    points = retriever._fetch_discovery_channel(next(iter(REPOSITORY_DISCOVERY_CHANNELS)), 1,
                                               include_vectors=False, exclude_repo_ids=None)
    assert [str(point.id) for point in points] == [excluded]
    client.close()


def test_default_policy_keeps_prepared_observations_inert(monkeypatch):
    monkeypatch.delenv('V2_REPOSITORY_LIFECYCLE_POLICY', raising=False)
    retriever = QdrantV2Retriever(client=object(), heavy_ranker_enabled=False)
    assert not any(getattr(item, 'key', '').startswith('lifecycle_') for item in (retriever._eligible_filter().must_not or []))


def test_refresh_cas_persists_status_without_reembedding_and_rejects_conflict():
    client, store, original = _indexed_feature_refresh_fixture()
    original_vector = original.vector
    def apply(status, version):
        job = RepositoryRefreshJob(schema_version=2, job_id=uuid.uuid4(), repo_id=original.id,
                                   feature_version=version, features={'lifecycle': status})
        with patch('api.v2.repository_store', return_value=store), patch('api.v2._repository_points', side_effect=lambda _:
            client.retrieve(store.collection_name, [original.id], with_payload=True, with_vectors=True)):
            return _refresh_repository_job_locked(job, OwnedLock())
    assert apply(observation(archived=True, day=2), 1)['status'] == 'applied'
    assert apply(observation(status='unavailable', day=4), 2)['status'] == 'applied'
    assert apply(observation(day=1), 3)['status'] == 'applied'
    stored = client.retrieve(store.collection_name, [original.id], with_vectors=True)[0]
    assert stored.payload['lifecycle_archived'] is True
    assert stored.vector == original_vector
    assert stored.payload['content_version'] == original.payload['content_version']
    assert apply(observation(day=3), 4)['status'] == 'applied'
    stored = client.retrieve(store.collection_name, [original.id], with_vectors=True)[0]
    assert stored.payload['lifecycle_archived'] is False
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as conflict:
        apply(observation(archived=True, day=3), 5)
    assert conflict.value.status_code == 422
    client.close()


@pytest.mark.parametrize('incoming', [None, observation(day=1), observation(status='unavailable', day=4)])
def test_new_content_cannot_erase_newer_lifecycle(incoming):
    from api.contracts import RepositoryJob
    from api.v2 import _embed_repository_job_locked
    from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
    from tests.test_v2_summary_contract import _job, FakeEmbedder
    client, store, point = _indexed_feature_refresh_fixture()
    client.set_payload(store.collection_name, merge_lifecycle({}, observation(archived=True, day=2)), [point.id])
    point = client.retrieve(store.collection_name, [point.id], with_payload=True, with_vectors=True)[0]
    body = _job(content_hash='changed-content').model_dump(mode='json')
    body['content_version'] = point.payload['content_version'] + 1
    body['repository']['lifecycle'] = incoming
    job = RepositoryJob.model_validate(body)
    with patch('api.v2.repository_store', return_value=store), patch('api.v2._repository_points', return_value=[point]), patch(
        'api.v2.repository_embedding_pipeline', return_value=RepositoryEmbeddingPipeline(embedder=FakeEmbedder())):
        result = _embed_repository_job_locked(job, OwnedLock())
    assert result['status'] == 'applied'
    stored = client.retrieve(store.collection_name, [point.id], with_payload=True)[0]
    assert stored.payload['lifecycle_archived'] is True
    assert stored.payload['lifecycle_observed_at'] == point.payload['lifecycle_observed_at']
    client.close()


def test_delayed_archive_refresh_survives_newer_stats_version_without_rewinding_it():
    client, store, original = _indexed_feature_refresh_fixture()
    client.set_payload(store.collection_name, {'feature_version': 20, 'feature_job_id': str(uuid.uuid4()), 'star_count': 99}, [original.id])
    request = RepositoryRefreshJob(schema_version=2, job_id=uuid.uuid4(), repo_id=original.id,
        feature_version=10, features={'star_count': 1, 'lifecycle': observation(archived=True, day=2)})
    with patch('api.v2.repository_store', return_value=store), patch('api.v2._repository_points', side_effect=lambda _:
        client.retrieve(store.collection_name, [original.id], with_payload=True, with_vectors=True)):
        assert _refresh_repository_job_locked(request, OwnedLock())['status'] == 'current'
    stored = client.retrieve(store.collection_name, [original.id], with_payload=True)[0]
    assert stored.payload['lifecycle_archived'] is True
    assert stored.payload['feature_version'] == 20 and stored.payload['star_count'] == 99
    client.close()


def test_independent_lifecycle_cas_fences_stale_content_and_feature_writers():
    from api.v2 import _repository_embedding_payload
    from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
    from tests.test_v2_summary_contract import _job, FakeEmbedder
    client, store, point = _indexed_feature_refresh_fixture()
    # A writer already holding an old snapshot cannot erase the independent
    # status update even though content/feature versions did not advance.
    store.compare_and_set_lifecycle(expected_point=point,
        lifecycle_payload=merge_lifecycle({}, observation(archived=True, day=2)))
    body = _job(content_hash='newer-content').model_dump(mode='json')
    body['content_version'] = point.payload['content_version'] + 1
    from api.contracts import RepositoryJob
    result = RepositoryEmbeddingPipeline(embedder=FakeEmbedder()).embed_repository(
        _repository_embedding_payload(RepositoryJob.model_validate(body)))
    result.payload['content_job_id'] = str(uuid.uuid4())
    store.compare_and_set_content(result, expected_point=point)
    store.compare_and_set_features(expected_point=point, feature_payload={
        'feature_version': 2, 'feature_job_id': str(uuid.uuid4()), 'star_count': 999})
    stored = client.retrieve(store.collection_name, [point.id], with_payload=True)[0]
    assert stored.payload['lifecycle_archived'] is True
    assert stored.payload['content_version'] == point.payload['content_version']
    assert stored.payload['star_count'] == point.payload['star_count']
    client.close()


def test_lifecycle_health_requires_new_indexes_only_when_policy_enabled(monkeypatch):
    from config import QDRANT_PAYLOAD_INDEX_SCHEMA
    legacy_indexes = {key: value for key, value in QDRANT_PAYLOAD_INDEX_SCHEMA.items() if not key.startswith('lifecycle_')}
    monkeypatch.delenv('V2_REPOSITORY_LIFECYCLE_POLICY', raising=False)
    retriever = QdrantV2Retriever(client=object(), heavy_ranker_enabled=False)
    retriever._validate_repository_payload_indexes(SimpleNamespace(payload_schema=legacy_indexes), lifecycle_policy=retriever.lifecycle_policy)
    monkeypatch.setenv('V2_REPOSITORY_LIFECYCLE_POLICY', 'verified-status-v1')
    retriever = QdrantV2Retriever(client=object(), heavy_ranker_enabled=False)
    with pytest.raises(RuntimeError, match='lifecycle_archived'):
        retriever._validate_repository_payload_indexes(SimpleNamespace(payload_schema=legacy_indexes), lifecycle_policy=retriever.lifecycle_policy)


# Reuse the real local Qdrant corpus/model boundary from existing retrieval tests.
from tests.test_v2_retrieval import local_recommendation_corpus, _corpus_retriever, HeavyRankerQdrant
from retrieval.social_candidates import SocialCandidate, SocialPolicy
from retrieval.discovery_cache import DiscoveryRevision


def test_archived_social_leader_cannot_hide_active_candidate_before_tiny_lookup_bound(local_recommendation_corpus, monkeypatch):
    monkeypatch.setenv('V2_REPOSITORY_LIFECYCLE_POLICY', 'verified-status-v1')
    corpus = local_recommendation_corpus
    retriever = _corpus_retriever(corpus, heavy=True)
    retriever.social_policy = SocialPolicy(mode='enabled', max_lookup=1, max_candidates=1)
    archived, active = corpus.repo_ids[-2:]
    corpus.client.set_payload(corpus.repository_collection, merge_lifecycle({}, observation(archived=True)), [archived])
    social = [SocialCandidate(repo_id=archived, owner_github_id='123', source='followed_owner', source_score=1., support=1),
              SocialCandidate(repo_id=active, owner_github_id='456', source='followed_owner', source_score=.5, support=1)]
    retriever.recommend_batch(corpus.user_id, 10, [], 'lifecycle-social', False, [], social)
    ranked = {row['id'] for row in retriever.ranker.calls[-1][2]}
    assert active in ranked and archived not in ranked
    social_reads = [call.kwargs for call in corpus.client.scroll.call_args_list
                    if any(isinstance(item, models.HasIdCondition) and set(item.has_id) == {archived, active}
                           for item in (call.kwargs['scroll_filter'].must or []))]
    assert len(social_reads) == 1
    assert any(getattr(item, 'key', '') == 'lifecycle_archived' for item in social_reads[0]['scroll_filter'].must_not)


def test_active_lifecycle_bypasses_discovery_snapshot_cache(monkeypatch):
    monkeypatch.setenv('V2_REPOSITORY_LIFECYCLE_POLICY', 'verified-status-v1')
    client = HeavyRankerQdrant()
    retriever = QdrantV2Retriever(client=client, heavy_ranker_enabled=False,
        discovery_revision_provider=lambda: DiscoveryRevision('corpus', 'eligibility'))
    retriever.recommend_batch(client.user_id, 2, [], 'first')
    retriever.recommend_batch(client.user_id, 2, [], 'second')
    assert len(client.scroll_calls) == 8


def test_archive_transition_during_deferred_hydration_cannot_reach_heavy_ranker(local_recommendation_corpus, monkeypatch):
    monkeypatch.setenv('V2_REPOSITORY_LIFECYCLE_POLICY', 'verified-status-v1')
    corpus = local_recommendation_corpus
    retriever = _corpus_retriever(corpus, heavy=True)
    original = retriever._fetch_retained_vectors
    changed = []
    def archive_after_selection(ids):
        changed.append(ids[0])
        corpus.client.set_payload(corpus.repository_collection,
            merge_lifecycle({}, observation(archived=True)), [ids[0]])
        return original(ids)
    monkeypatch.setattr(retriever, '_fetch_retained_vectors', archive_after_selection)
    retriever.recommend_batch(corpus.user_id, 10, [], 'hydrate-lifecycle')
    assert changed
    assert changed[0] not in {row['id'] for row in retriever.ranker.calls[-1][2]}


@pytest.mark.parametrize('timestamp', [
    '0000-01-01T00:00:00Z',
    '0001-01-01T00:00:00+23:59',
    '9999-12-31T23:59:59-23:59',
])
def test_lifecycle_http_rejects_unrepresentable_utc_dates_with_422(monkeypatch, timestamp):
    from fastapi.testclient import TestClient
    from api.main import app
    monkeypatch.setenv('INTERNAL_API_SECRET', 'lifecycle-range-secret')
    response = TestClient(app).post('/api/v2/repositories/refresh',
        headers={'x-internal-secret': 'lifecycle-range-secret'}, json={
            'schema_version': 2, 'job_id': str(uuid.uuid4()), 'repo_id': str(uuid.uuid4()),
            'feature_version': 1,
            'features': {'lifecycle': {**observation(), 'observed_at': timestamp}},
        })
    assert response.status_code == 422, response.text
    assert any('observed_at' in row['path'] for row in response.json()['details'])


def test_lifecycle_utc_normalization_retains_supported_extreme_offset_and_microseconds():
    item = RepositoryLifecycleObservation.model_validate({
        **observation(), 'observed_at': '0001-01-02T00:00:00.123456+23:59'})
    assert item.observed_at.isoformat() == '0001-01-01T00:01:00.123456+00:00'


@pytest.mark.parametrize('lifecycle', [observation(archived=True), observation(disabled=True)])
def test_same_content_replay_persists_lifecycle_without_embedding(lifecycle):
    from api.v2 import _embed_repository_job_locked
    from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
    from tests.test_v2_summary_contract import _job, FakeEmbedder
    client, store, point = _indexed_feature_refresh_fixture()
    body = _job().model_dump(mode='json')
    body['repository']['lifecycle'] = lifecycle
    client.set_payload(store.collection_name, {'content_version': 7}, [point.id])
    point.payload['content_version'] = 7
    body['repository']['content_hash'] = point.payload['content_hash']
    from api.contracts import RepositoryJob
    with patch('api.v2.repository_store', return_value=store), patch('api.v2._repository_points', return_value=[point]), patch(
        'api.v2.repository_embedding_pipeline', return_value=RepositoryEmbeddingPipeline(embedder=FakeEmbedder())) as provider:
        with patch.object(provider.return_value, 'embed_repository', side_effect=AssertionError('same content must not embed')):
            response = _embed_repository_job_locked(RepositoryJob.model_validate(body), OwnedLock())
    assert response['status'] == 'current'
    stored = client.retrieve(store.collection_name, [point.id], with_payload=True)[0]
    assert stored.payload['lifecycle_archived'] == lifecycle['archived']
    assert stored.payload['lifecycle_disabled'] == lifecycle['disabled']
    client.close()


def test_future_lifecycle_source_rejected_before_embedding_work(monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    from tests.test_v2_summary_contract import _job
    monkeypatch.setenv('INTERNAL_API_SECRET', 'lifecycle-future-secret')
    body = _job().model_dump(mode='json')
    body['repository']['lifecycle'] = {**observation(), 'observed_at': (datetime.now(timezone.utc)+timedelta(days=1)).isoformat()}
    with patch('api.v2.repository_embedding_pipeline', side_effect=AssertionError('validation must precede model work')):
        response = TestClient(app).post('/api/v2/repositories/embed', json=body,
            headers={'x-internal-secret': 'lifecycle-future-secret'})
    assert response.status_code == 422


def test_bootstrap_can_validate_indexes_without_a_retriever_instance():
    from config import QDRANT_PAYLOAD_INDEX_SCHEMA
    QdrantV2Retriever._validate_repository_payload_indexes(SimpleNamespace(payload_schema=QDRANT_PAYLOAD_INDEX_SCHEMA))
