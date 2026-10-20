from __future__ import annotations

from fastapi.testclient import TestClient

from api import main
from api import metrics
from api.runtime import shutdown_service_runtime


def setup_function() -> None:
    metrics.reset_metrics_for_tests()
    shutdown_service_runtime()


def teardown_function() -> None:
    shutdown_service_runtime()
    metrics.reset_metrics_for_tests()


def test_metrics_are_fixed_cardinality_and_cumulative() -> None:
    metrics.record_api_request(
        path="/api/v2/users/secret-user-id",
        method="DELETE",
        status_code=599,
        duration_seconds=0.02,
    )
    metrics.record_recommendation(
        served_ranker="unexpected-ranker",
        fallback_code="exception-with-user-data",
        item_count=0,
    )
    metrics.record_api_request(
        path="/api/v2/repositories/refresh",
        method="POST",
        status_code=503,
        duration_seconds=45.0,
    )

    rendered = metrics.render_prometheus({})

    assert 'method="OTHER",route="/api/v2/other",status_class="5xx"' in rendered
    assert 'served_ranker="other",fallback_code="other"' in rendered
    assert "secret-user-id" not in rendered
    assert "exception-with-user-data" not in rendered
    assert "ml_empty_recommendations_total 1" in rendered
    assert (
        'ml_api_request_duration_seconds_bucket{method="POST",route="/api/v2/repositories/refresh",le="+Inf"} 1'
        in rendered
    )


def test_metrics_endpoint_is_authenticated_and_exposes_executor_saturation(
    monkeypatch,
) -> None:
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    client = TestClient(main.app)

    assert client.get("/api/v2/metrics").status_code == 401
    response = client.get(
        "/api/v2/metrics",
        headers={"x-internal-secret": "test-internal-secret"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "ml_api_requests_total" in response.text
    assert 'ml_service_executor_capacity{operation="feedback"} 8' in response.text
    assert 'ml_service_executor_capacity{operation="recommendation"} 8' in response.text
    assert 'le="+Inf"' in response.text


def test_search_success_and_failure_have_their_own_fixed_metric_route() -> None:
    for status in [200, 503]:
        metrics.record_api_request(
            path="/api/v2/search/repositories", method="POST",
            status_code=status, duration_seconds=0.2,
        )
    rendered = metrics.render_prometheus({})
    assert 'route="/api/v2/search/repositories",status_class="2xx"' in rendered
    assert 'route="/api/v2/search/repositories",status_class="5xx"' in rendered
    assert 'route="/api/v2/search/repositories",le="+Inf"} 2' in rendered
    assert '/api/v2/other' not in rendered
