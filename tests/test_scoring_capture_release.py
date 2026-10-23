"""Capture binds to the baked release while ordinary requests stay unchanged."""
import pytest
from tests.test_scoring_provider import actual_pipeline, trained_ranker, CAPABILITY, COMMIT, payload, post


@pytest.mark.parametrize("setting,value", [
    ("BAKED_ML_RELEASE_ID", None), ("BAKED_ML_RELEASE_ID", "main"),
    ("BAKED_ML_RELEASE_ID", "A" * 40), ("BAKED_ML_RELEASE_ID", "b" * 40),
    ("SCORING_CAPTURE_CODE_COMMIT", "b" * 40),
])
def test_capture_release_must_equal_valid_baked_image_identity(actual_pipeline, monkeypatch, setting, value):
    if value is None:
        monkeypatch.delenv(setting, raising=False)
    else:
        monkeypatch.setenv(setting, value)
    result = post(payload())
    assert result.status_code == 503
    assert result.json()["code"] == "SCORING_CAPTURE_CONFIG_INVALID"
    assert CAPABILITY not in result.text
    # Ordinary wire and admission never inspect the new release configuration.
    assert post(payload(captured=False)).status_code == 200


def test_capture_records_matching_baked_identity(actual_pipeline):
    response = post(payload())
    assert response.status_code == 200
    assert response.json()["scoring_context"]["code_commit"] == COMMIT
