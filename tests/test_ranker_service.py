import json
import hashlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from config import EMBEDDING_MODEL_REVISION, REPOSITORY_EMBEDDING_MODEL
from inference.feature_spec import (
    EMBEDDING_DIM,
    FEATURE_COUNT,
    FEATURE_SPEC_VERSION,
    INPUT_DIM,
)
from inference.ranker_service import MMoEHeavyRanker, RankerService
from inference.value_function import VALUE_WEIGHTS, compute_value_score


def _write_scaler(path, feature_count=FEATURE_COUNT):
    path.write_text(
        json.dumps(
            {
                "mean": [0.0] * feature_count,
                "scale": [1.0] * feature_count,
            }
        )
    )


def _write_manifest(path, **overrides):
    manifest = {
        "model_version": "heavy-ranker-test",
        "embedding_version": "repo-embedding-test",
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "input_dim": INPUT_DIM,
        "embedding_dim": EMBEDDING_DIM,
        "feature_count": FEATURE_COUNT,
    }
    manifest.update(overrides)
    path.write_text(json.dumps(manifest))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_production_manifest(path, *, model_path, scaler_path, **overrides):
    manifest = {
        "model_file": model_path.name,
        "scaler_file": scaler_path.name,
        "model_sha256": _sha256(model_path),
        "scaler_sha256": _sha256(scaler_path),
        "model_version": "heavy-ranker-production-test",
        "embedding_version": "repo-embedding-test",
        "embedding_model": REPOSITORY_EMBEDDING_MODEL,
        "embedding_model_revision": EMBEDDING_MODEL_REVISION,
        "compatible_embedding_versions": ["repo-embedding-test"],
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "input_dim": INPUT_DIM,
        "embedding_dim": EMBEDDING_DIM,
        "feature_count": FEATURE_COUNT,
        "value_function_version": "v1",
        "value_weights": VALUE_WEIGHTS,
        "training_data": {
            "identity": "telemetry-snapshot-2026-07-01",
            "type": "versioned_production_telemetry",
        },
        "training_timestamp": "2026-07-02T12:00:00+00:00",
        "offline_metrics": {"ndcg_at_15": 0.51},
        "calibration_metrics": {"expected_calibration_error": 0.04},
        "code_version": "0123456789abcdef",
    }
    manifest.update(overrides)
    path.write_text(json.dumps(manifest))


@pytest.fixture
def loaded_ranker_service(tmp_path):
    scaler_path = tmp_path / "feature_scaler.json"
    model_path = tmp_path / "heavy_ranker.pt"
    _write_scaler(scaler_path)
    model_path.touch()

    state_dict = MMoEHeavyRanker(INPUT_DIM).state_dict()
    with patch("inference.ranker_service.torch.load", return_value=state_dict):
        return RankerService(
            model_path=str(model_path),
            scaler_path=str(scaler_path),
        )


def test_ranker_service_rejects_scaler_with_wrong_feature_count(tmp_path):
    scaler_path = tmp_path / "feature_scaler.json"
    _write_scaler(scaler_path, feature_count=FEATURE_COUNT - 1)

    with pytest.raises(ValueError, match="expected 10"):
        RankerService(
            model_path=str(tmp_path / "missing_model.pt"),
            scaler_path=str(scaler_path),
        )


def test_missing_model_disables_scoring(tmp_path, caplog):
    scaler_path = tmp_path / "feature_scaler.json"
    _write_scaler(scaler_path)

    service = RankerService(
        model_path=str(tmp_path / "missing_model.pt"),
        scaler_path=str(scaler_path),
    )

    assert service._model_loaded is False
    with caplog.at_level("WARNING", logger="pipeline.ranker"):
        assert service.score_batch(np.zeros(384), [], [{}]) == []
    assert "model is not loaded" in caplog.text


def test_ranker_service_exposes_manifest_versions(tmp_path):
    scaler_path = tmp_path / "feature_scaler.json"
    manifest_path = tmp_path / "model_manifest.json"
    _write_scaler(scaler_path)
    _write_manifest(manifest_path)

    service = RankerService(
        model_path=str(tmp_path / "missing_model.pt"),
        scaler_path=str(scaler_path),
        manifest_path=str(manifest_path),
    )

    assert service.model_version == "heavy-ranker-test"
    assert service.embedding_version == "repo-embedding-test"


def test_ranker_service_rejects_incompatible_manifest(tmp_path):
    scaler_path = tmp_path / "feature_scaler.json"
    manifest_path = tmp_path / "model_manifest.json"
    _write_scaler(scaler_path)
    _write_manifest(manifest_path, embedding_dim=768)

    with pytest.raises(ValueError, match="Model manifest is incompatible"):
        RankerService(
            model_path=str(tmp_path / "missing_model.pt"),
            scaler_path=str(scaler_path),
            manifest_path=str(manifest_path),
        )


def test_ranker_service_accepts_explicit_v2_embedding_compatibility(tmp_path):
    scaler_path = tmp_path / "feature_scaler.json"
    manifest_path = tmp_path / "model_manifest.json"
    _write_scaler(scaler_path)
    _write_manifest(
        manifest_path,
        compatible_embedding_versions=[
            "repo-embedding-test",
            "repo-embedding-v2",
        ],
    )

    service = RankerService(
        model_path=str(tmp_path / "missing_model.pt"),
        scaler_path=str(scaler_path),
        manifest_path=str(manifest_path),
        expected_embedding_version="repo-embedding-v2",
    )

    assert "repo-embedding-v2" in service.compatible_embedding_versions


def test_ranker_service_rejects_unlisted_serving_embedding_version(tmp_path):
    scaler_path = tmp_path / "feature_scaler.json"
    manifest_path = tmp_path / "model_manifest.json"
    _write_scaler(scaler_path)
    _write_manifest(manifest_path)

    with pytest.raises(ValueError, match="embedding contract is incompatible"):
        RankerService(
            model_path=str(tmp_path / "missing_model.pt"),
            scaler_path=str(scaler_path),
            manifest_path=str(manifest_path),
            expected_embedding_version="repo-embedding-v2",
        )


def test_legacy_manifest_loads_for_inspection_but_is_not_production_qualified(
    tmp_path,
):
    scaler_path = tmp_path / "feature_scaler.json"
    manifest_path = tmp_path / "model_manifest.json"
    _write_scaler(scaler_path)
    _write_manifest(manifest_path)

    service = RankerService(
        model_path=str(tmp_path / "missing_model.pt"),
        scaler_path=str(scaler_path),
        manifest_path=str(manifest_path),
    )

    assert service.production_qualified is False
    assert "missing or invalid model_sha256" in service.qualification_errors
    assert "missing or invalid training_data" in service.qualification_errors


def test_production_manifest_requires_verified_provenance_and_metrics(tmp_path):
    scaler_path = tmp_path / "feature_scaler.json"
    model_path = tmp_path / "heavy_ranker.pt"
    manifest_path = tmp_path / "model_manifest.json"
    _write_scaler(scaler_path)
    model_path.write_bytes(b"model-artifact")
    _write_production_manifest(
        manifest_path,
        model_path=model_path,
        scaler_path=scaler_path,
    )
    state_dict = MMoEHeavyRanker(INPUT_DIM).state_dict()

    with patch("inference.ranker_service.torch.load", return_value=state_dict):
        service = RankerService(
            model_path=str(model_path),
            scaler_path=str(scaler_path),
            manifest_path=str(manifest_path),
            expected_embedding_versions={"repo-embedding-test"},
            expected_embedding_model=REPOSITORY_EMBEDDING_MODEL,
            expected_embedding_model_revision=EMBEDDING_MODEL_REVISION,
            require_production_manifest=True,
        )

    assert service.ready is True
    assert service.production_qualified is True
    assert service.qualification_errors == ()


def test_synthetic_training_manifest_cannot_be_production_qualified(tmp_path):
    scaler_path = tmp_path / "feature_scaler.json"
    model_path = tmp_path / "heavy_ranker.pt"
    manifest_path = tmp_path / "model_manifest.json"
    _write_scaler(scaler_path)
    model_path.write_bytes(b"model-artifact")
    _write_production_manifest(
        manifest_path,
        model_path=model_path,
        scaler_path=scaler_path,
        training_data={"identity": "generator-v1", "type": "synthetic"},
    )

    with pytest.raises(RuntimeError, match="synthetic training data"):
        RankerService(
            model_path=str(model_path),
            scaler_path=str(scaler_path),
            manifest_path=str(manifest_path),
            require_production_manifest=True,
        )


def test_synthetic_manifest_can_be_explicitly_manually_qualified(tmp_path):
    scaler_path = tmp_path / "feature_scaler.json"
    model_path = tmp_path / "heavy_ranker.pt"
    manifest_path = tmp_path / "model_manifest.json"
    _write_scaler(scaler_path)
    model_path.write_bytes(b"model-artifact")
    _write_production_manifest(
        manifest_path,
        model_path=model_path,
        scaler_path=scaler_path,
        training_data={"identity": "generator-v1", "type": "synthetic"},
        production_qualified=True,
        qualification_method="manual_override",
    )
    state_dict = MMoEHeavyRanker(INPUT_DIM).state_dict()

    with patch("inference.ranker_service.torch.load", return_value=state_dict):
        service = RankerService(
            model_path=str(model_path),
            scaler_path=str(scaler_path),
            manifest_path=str(manifest_path),
            require_production_manifest=True,
        )

    assert service.production_qualified is True
    assert service.qualification_overridden is True
    assert "synthetic training data is not production-qualified" in (
        service.qualification_errors
    )



@pytest.mark.parametrize("count", [0, 1, 150, 2000])
def test_compact_and_diagnostic_outputs_match_scalar_baseline(loaded_ranker_service, count):
    rng = np.random.default_rng(68)
    candidates = [
        {"id": str(index), "embedding": rng.normal(size=EMBEDDING_DIM),
         "languages": ["Python"], "topics": ["Σ"],
         "doc_quality": index / max(count, 1), "activity_score": 0.75}
        for index in range(count)
    ]
    user = rng.normal(size=EMBEDDING_DIM)
    loaded_ranker_service.model = MagicMock(wraps=loaded_ranker_service.model)
    diagnostic = loaded_ranker_service.score_batch(user, ["Python", "ς"], candidates)
    compact = loaded_ranker_service.score_batch_compact(user, ["Python", "ς"], candidates)
    assert compact == [{"repo_id": result["repo_id"], "final_score": result["final_score"]}
                       for result in diagnostic]
    if not count:
        assert not loaded_ranker_service.model.called
        return
    tensor = loaded_ranker_service.model.call_args.args[0]
    with torch.no_grad():
        outputs = loaded_ranker_service.model(tensor)
    # Deliberately preserve the previous five transfers, clipping and scalar
    # value function as an independent compatibility oracle.
    predictions = np.column_stack([
        np.clip(output.reshape(-1).cpu().numpy(), 0, 1) for output in outputs
    ])
    expected = []
    for repo, row in zip(candidates, predictions):
        values = dict(zip(("p_ctr", "p_save", "p_gh", "pred_dwell_fraction", "p_follow"),
                          map(float, row)))
        expected.append({"repo_id": repo["id"], "final_score": compute_value_score(values),
                         "skill_match": 1.0, "predictions": values})
    expected.sort(key=lambda result: result["final_score"], reverse=True)
    assert diagnostic == expected  # Exact equality, including stable ordering.


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compact_value_precision_preserves_near_ties_and_stable_order(loaded_ranker_service, dtype):
    ctr = torch.tensor([0.5, 0.5, 0.50000006], dtype=dtype)
    outputs = (ctr, *(torch.full((3,), 0.1, dtype=dtype) for _ in range(4)))
    loaded_ranker_service.model = MagicMock(return_value=outputs)
    candidates = [{"id": str(index), "embedding": np.zeros(EMBEDDING_DIM)}
                  for index in range(3)]
    results = loaded_ranker_service.score_batch_compact(np.zeros(EMBEDDING_DIM), [], candidates)
    expected = [compute_value_score(dict(zip(
        ("p_ctr", "p_save", "p_gh", "pred_dwell_fraction", "p_follow"),
        (float(output[index]) for output in outputs),
    ))) for index in range(3)]
    assert [result["repo_id"] for result in results] == ["2", "0", "1"]
    assert [result["final_score"] for result in results] == [expected[2], expected[0], expected[1]]


@pytest.mark.parametrize("method", ["score_batch", "score_batch_compact"])
def test_ranker_output_uses_one_transfer_in_inference_mode(loaded_ranker_service, monkeypatch, method):
    original_cpu = torch.Tensor.cpu
    original_model = loaded_ranker_service.model
    transfers = []

    def cpu(tensor, *args, **kwargs):
        transfers.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    def model(tensor):
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        return original_model(tensor)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    loaded_ranker_service.model = model
    getattr(loaded_ranker_service, method)(np.zeros(EMBEDDING_DIM), [],
        [{"id": str(index), "embedding": np.zeros(EMBEDDING_DIM)} for index in range(3)])
    assert transfers == [(3, 5)]
    assert not torch.is_inference_mode_enabled()  # Context must not leak to training.


@pytest.mark.parametrize("head", [0, 4])
def test_compact_path_retains_nonfinite_prediction_rejection(loaded_ranker_service, head):
    outputs = [torch.tensor(0.0) for _ in range(5)]
    outputs[head] = torch.tensor(float("nan"))
    loaded_ranker_service.model = MagicMock(return_value=outputs)
    with pytest.raises(ValueError, match="non-finite predictions"):
        loaded_ranker_service.score_batch_compact(np.zeros(EMBEDDING_DIM), [],
            [{"id": "invalid", "embedding": np.zeros(EMBEDDING_DIM)}])


def test_direct_ranker_training_still_supports_autograd():
    model = MMoEHeavyRanker(INPUT_DIM).train()
    inputs = torch.randn(2, INPUT_DIM, requires_grad=True)
    loss = sum(output.sum() for output in model(inputs))
    loss.backward()
    assert inputs.grad is not None and torch.all(torch.isfinite(inputs.grad))
    assert torch.count_nonzero(inputs.grad) > 0


def test_score_batch_empty_candidates_returns_empty(loaded_ranker_service):
    assert loaded_ranker_service.score_batch(
        np.zeros(384, dtype=np.float32),
        ["Python"],
        [],
    ) == []


def test_score_batch_single_candidate_shape(loaded_ranker_service):
    candidate = {
        "id": "repo-1",
        "embedding": np.zeros(384, dtype=np.float32),
        "languages": ["Python"],
        "topics": [],
        "tags": [],
    }
    results = loaded_ranker_service.score_batch(
        np.zeros(384, dtype=np.float32),
        ["Python"],
        [candidate],
    )

    assert isinstance(results, list)
    assert len(results) == 1
    assert set(results[0]) == {
        "repo_id",
        "final_score",
        "skill_match",
        "predictions",
    }
    assert results[0]["repo_id"] == "repo-1"
    assert isinstance(results[0]["final_score"], float)
    assert set(results[0]["predictions"]) == {
        "p_ctr",
        "p_save",
        "p_gh",
        "pred_dwell_fraction",
        "p_follow",
    }


def test_score_batch_candidate_missing_optional_fields(loaded_ranker_service):
    candidate = {
        "id": "minimal-repo",
        "embedding": np.zeros(384, dtype=np.float32),
    }

    results = loaded_ranker_service.score_batch(
        np.zeros(384, dtype=np.float32),
        [],
        [candidate],
    )

    assert len(results) == 1
    assert results[0]["repo_id"] == "minimal-repo"


def test_score_batch_preserves_repo_id_order_mapping(loaded_ranker_service):
    candidates = [
        {
            "id": f"repo-{index}",
            "embedding": np.zeros(384, dtype=np.float32),
        }
        for index in range(3)
    ]
    loaded_ranker_service.model = MagicMock(
        return_value=(
            torch.tensor([0.9, 0.8, 0.7]),
            torch.zeros(3),
            torch.zeros(3),
            torch.zeros(3),
            torch.zeros(3),
        )
    )

    results = loaded_ranker_service.score_batch(
        np.zeros(384, dtype=np.float32),
        [],
        candidates,
    )

    assert [result["repo_id"] for result in results] == [
        candidate["id"] for candidate in candidates
    ]


def test_score_batch_rejects_wrong_embedding_dimensions(loaded_ranker_service):
    with pytest.raises(ValueError, match="exactly 384"):
        loaded_ranker_service.score_batch(
            np.zeros(383, dtype=np.float32),
            [],
            [{"id": "repo-1", "embedding": np.zeros(384, dtype=np.float32)}],
        )


def test_score_batch_clips_outliers_and_prediction_ranges(loaded_ranker_service):
    loaded_ranker_service.model = MagicMock(
        return_value=(
            torch.tensor(2.0),
            torch.tensor(-1.0),
            torch.tensor(0.5),
            torch.tensor(4.0),
            torch.tensor(0.25),
        )
    )

    results = loaded_ranker_service.score_batch(
        np.zeros(384, dtype=np.float32),
        [],
        [
            {
                "id": "repo-outlier",
                "embedding": np.zeros(384, dtype=np.float32),
                "star_count": 10**12,
            }
        ],
    )

    model_input = loaded_ranker_service.model.call_args.args[0].cpu().numpy()
    assert model_input[0, 771] == pytest.approx(8.0)
    assert results[0]["predictions"] == {
        "p_ctr": 1.0,
        "p_save": 0.0,
        "p_gh": 0.5,
        "pred_dwell_fraction": 1.0,
        "p_follow": 0.25,
    }


def test_score_batch_rejects_non_finite_model_output(loaded_ranker_service):
    loaded_ranker_service.model = MagicMock(
        return_value=(
            torch.tensor(float("nan")),
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )
    )

    with pytest.raises(ValueError, match="non-finite predictions"):
        loaded_ranker_service.score_batch(
            np.zeros(384, dtype=np.float32),
            [],
            [{"id": "repo-1", "embedding": np.zeros(384, dtype=np.float32)}],
        )
