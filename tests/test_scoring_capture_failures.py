"""Private worker failures remain observable without leaking input exceptions."""
import traceback
import uuid
import pytest
from api import scoring_capture
from api.metrics import record_scoring_capture_failure, render_prometheus, reset_metrics_for_tests
from inference.scoring_runtime import CaptureOptions
from tests.test_scoring_provider import actual_pipeline, trained_ranker, CAPABILITY, COMMIT, USER, payload, post


@pytest.mark.parametrize("fault,expected_stage", [
    ("packet", "packet_binding"), ("validation", "response_validation"), ("seal", "sealing"),
])
def test_private_worker_exception_is_unreachable_and_native_future_marks_error(actual_pipeline, monkeypatch, caplog, fault, expected_stage):
    private = "private-tensor-or-validation-input-DO-NOT-LOG"
    def fail(*_args, **_kwargs):
        raise ValueError(private)
    if fault == "packet":
        monkeypatch.setattr(scoring_capture, "packet_for_frame", fail)
    elif fault == "validation":
        monkeypatch.setattr(scoring_capture.CapturedRecommendationResponse, "model_validate_json", fail)
    else:
        monkeypatch.setattr(scoring_capture, "seal_scoring_response", fail)
    request = scoring_capture.CapturedRequest(USER, str(uuid.uuid4()), "1", 5, (), (), (), False, None,
        CaptureOptions(feed_version="1", code_commit=COMMIT))
    with pytest.raises(scoring_capture.CaptureWorkerError) as caught:
        scoring_capture.execute_captured_request(actual_pipeline[0], request)
    error = caught.value
    assert error.__context__ is None and error.__cause__ is None
    assert error.args == ("SCORING_CAPTURE_UNAVAILABLE",)
    assert "_execute_captured_request_private" not in [frame.name for frame in traceback.extract_tb(error.__traceback__)]
    assert private not in "".join(traceback.format_exception(error))
    reset_metrics_for_tests()
    response = post(payload())
    assert response.status_code == 503
    assert response.json()["code"] == "SCORING_CAPTURE_UNAVAILABLE"
    metrics = render_prometheus({})
    assert f'ml_scoring_capture_failures_total{{stage="{expected_stage}"}} 1' in metrics
    assert 'ml_stage_duration_seconds_count{operation="recommendation",stage="service",outcome="error"} 1' in metrics
    assert private not in response.text + caplog.text + metrics
    assert CAPABILITY not in response.text + caplog.text + metrics
    assert "ValidationError" not in caplog.text


def test_same_read_profile_revision_failure_is_bounded_scoring_stage(actual_pipeline):
    _, client, _, users, _ = actual_pipeline
    client.set_payload(users, {"profile_version": True}, points=[USER], wait=True)
    reset_metrics_for_tests()
    assert post(payload()).status_code == 503
    metrics = render_prometheus({})
    assert 'ml_scoring_capture_failures_total{stage="retrieval_scoring"} 1' in metrics
    assert 'ml_stage_duration_seconds_count{operation="recommendation",stage="service",outcome="error"} 1' in metrics


def test_failure_stage_counter_has_fixed_cardinality_and_never_adopts_exception_text():
    reset_metrics_for_tests()
    record_scoring_capture_failure("untrusted raw detail, owner or packet")
    metrics = render_prometheus({})
    assert 'ml_scoring_capture_failures_total{stage="other"} 1' in metrics
    assert "untrusted raw detail" not in metrics
