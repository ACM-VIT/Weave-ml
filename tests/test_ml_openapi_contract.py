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
    assert 'anyOf' in FeedbackEvent.model_json_schema()['properties']['feedback_version']


def test_feedback_schema_matches_impression_exclusion():
    from api.contracts import FeedbackEvent
    assert 'impression' not in FeedbackEvent.model_json_schema()['properties']['event_type']['enum']
