"""The documented wire contract must describe actual registered product routes."""
from api.main import app

PRODUCT_OPERATIONS = {
    '/api/v2/recommendations/generate': 200,
    '/api/v2/feedback/batch': 202,
    '/api/v2/repositories/embed': 200,
    '/api/v2/repositories/refresh': 200,
    '/api/v2/users/onboard': 200,
    '/api/v2/search/repositories': 200,
}


def test_all_product_success_responses_have_explicit_wire_models():
    schema = app.openapi()
    for path, status in PRODUCT_OPERATIONS.items():
        response = schema['paths'][path]['post']['responses'][str(status)]
        assert response['content']['application/json']['schema'].get('$ref'), path


def test_feedback_schema_preserves_exact_decimal_versions():
    from api.contracts import FeedbackEvent
    from datetime import datetime, timezone
    event = FeedbackEvent(
        event_id='00000000-0000-4000-8000-000000000001',
        user_id='00000000-0000-4000-8000-000000000002',
        repo_id='00000000-0000-4000-8000-000000000003',
        feedback_version='9223372036854775807', event_type='like',
        occurred_at=datetime.now(timezone.utc),
    )
    assert event.feedback_version == 9223372036854775807
    alternatives = FeedbackEvent.model_json_schema()['properties']['feedback_version']['anyOf']
    assert next(item for item in alternatives if item['type'] == 'integer')['minimum'] == 1


def test_feedback_schema_matches_impression_exclusion():
    from api.contracts import FeedbackEvent
    assert 'impression' not in FeedbackEvent.model_json_schema()['properties']['event_type']['enum']


def test_recommendations_accept_the_full_supported_mixed_embedding_identity(monkeypatch):
    from fastapi.testclient import TestClient
    from retrieval.v2_retriever import RecommendationBatch, QdrantV2Retriever, RankedRepository
    from types import SimpleNamespace
    from uuid import uuid4

    monkeypatch.setenv('INTERNAL_API_SECRET', 'isolated-contract-secret')
    versions = tuple(str(index) * 128 for index in range(8))
    # Use the active retriever's identity formatter, not a test-only join.
    candidates = [RankedRepository(str(uuid4()), .5, 'semantic') for version in versions]
    metadata = {item.repo_id: {'embedding_version': version} for item, version in zip(candidates, versions)}
    identity, reported_versions = QdrantV2Retriever._served_embedding_versions(candidates, metadata)
    batch = RecommendationBatch(
        items=[], model_version='heavy-ranker-v2', embedding_version=identity,
        embedding_versions=reported_versions, ranker_applied=True,
    )
    monkeypatch.setattr('api.v2.retriever', lambda: SimpleNamespace(recommend_batch=lambda *_args: batch))
    # No context manager: do not enter the production service lifespan.
    response = TestClient(app).post('/api/v2/recommendations/generate',
        headers={'x-internal-secret': 'isolated-contract-secret'}, json={
            'schema_version': 2, 'generation_id': str(uuid4()), 'user_id': str(uuid4()),
            'feed_version': 1, 'limit': 10, 'context': {'cold_start': False},
        })
    assert response.status_code == 200, response.text
    assert response.json()['embedding_version'] == identity
    assert response.json()['embedding_versions'] == list(versions)
