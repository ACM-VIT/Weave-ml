import json
import hashlib
import math
import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from config import EMBEDDING_MODEL_REVISION, REPOSITORY_EMBEDDING_MODEL
from inference.feature_spec import EMBEDDING_DIM, FEATURE_COUNT, FEATURE_SPEC_VERSION, INPUT_DIM
from inference.ranker_service import MMoEHeavyRanker, RankerService, _cached_repository_keywords, _repository_keywords, calculate_match_score
from inference.value_function import VALUE_WEIGHTS, compute_value_score
from pathlib import Path
from inference.ranker_service import RankerService
from retrieval.v2_retriever import QdrantV2Retriever
from Ranking_Training_module import train_ranker


# Ranker service

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

@pytest.fixture
def cpu_ranker_service(loaded_ranker_service):
    # These tests inspect CPU NumPy/storage sharing, not CUDA transfer behavior.
    loaded_ranker_service.device = torch.device("cpu")
    loaded_ranker_service.model.to(loaded_ranker_service.device)
    return loaded_ranker_service

@pytest.mark.parametrize("count", [1, 150, 2000])
def test_contiguous_ranker_input_matches_reference_for_pool_sizes(
    cpu_ranker_service, count
):
    user = np.arange(EMBEDDING_DIM * 2, dtype=np.float32)[::2] / 1000
    vectors = [np.roll(user, i)[::-1] for i in range(count)]
    for vector in vectors:
        vector.setflags(write=False)
    candidates = [{"id": str(i), "embedding": vector}
                  for i, vector in enumerate(vectors)]
    cpu_ranker_service.model = MagicMock(wraps=cpu_ranker_service.model)
    scores = cpu_ranker_service.score_batch(user, [], candidates)
    tensor = cpu_ranker_service.model.call_args.args[0]
    # Legacy tile/vstack/hstack assembly is the parity reference, including the
    # existing scaler/clipping and all ten default feature values.
    defaults = np.array([0.5, 0.5, 1000, 0, 0, 0, 365, 0, 0, 0], np.float32)
    dense = np.clip((defaults - cpu_ranker_service.scaler_mean)
                    / cpu_ranker_service.scaler_scale, -8, 8)
    expected = np.hstack((np.tile(user, (count, 1)), np.vstack(vectors),
                          np.tile(dense, (count, 1))))
    np.testing.assert_array_equal(tensor.cpu().numpy(), expected)
    assert tensor.is_contiguous() and tensor.dtype == torch.float32
    with torch.no_grad():
        outputs = cpu_ranker_service.model(torch.from_numpy(expected))
    predictions = np.clip(np.column_stack([
        output.reshape(-1).cpu().numpy() for output in outputs
    ]), 0, 1)
    expected_scores = [
        {"repo_id": str(index), "final_score": compute_value_score(dict(zip(
            ["p_ctr", "p_save", "p_gh", "pred_dwell_fraction", "p_follow"],
            map(float, row),
        )))}
        for index, row in enumerate(predictions)
    ]
    expected_scores.sort(key=lambda result: result["final_score"], reverse=True)
    assert [score["repo_id"] for score in scores] == [score["repo_id"] for score in expected_scores]
    np.testing.assert_allclose([score["final_score"] for score in scores],
                               [score["final_score"] for score in expected_scores],
                               rtol=0, atol=0)

@pytest.mark.parametrize("field", ["user", "repo"])
@pytest.mark.parametrize("bad", [np.zeros(383), np.zeros((1, 384)),
                                 np.full(384, np.nan), np.full(384, np.inf)])
def test_ranker_input_rejects_malformed_external_vectors(
    cpu_ranker_service, field, bad
):
    user = bad if field == "user" else np.zeros(EMBEDDING_DIM)
    repo = bad if field == "repo" else np.zeros(EMBEDDING_DIM)
    with pytest.raises(ValueError, match="exactly 384|only finite"):
        cpu_ranker_service.score_batch(user, [], [{"id": "bad", "embedding": repo}])

def test_ranker_input_does_not_alias_caller_arrays_or_later_requests(cpu_ranker_service):
    user = np.zeros(EMBEDDING_DIM, np.float32)
    repo = np.ones(EMBEDDING_DIM, np.float32)
    cpu_ranker_service.model = MagicMock(wraps=cpu_ranker_service.model)
    cpu_ranker_service.score_batch(user, [], [{"id": "one", "embedding": repo}])
    first = cpu_ranker_service.model.call_args.args[0]
    user[:] = 0.25
    repo[:] = 0.5
    assert torch.all(first[0, :EMBEDDING_DIM] == 0)
    assert torch.all(first[0, EMBEDDING_DIM:2 * EMBEDDING_DIM] == 1)
    first[0, 0] = 9
    assert user[0] == 0.25
    cpu_ranker_service.score_batch(user, [], [{"id": "two", "embedding": repo}])
    second = cpu_ranker_service.model.call_args.args[0]
    assert not np.shares_memory(first.numpy(), second.numpy())
    assert second[0, 0] == 0.25 and second[0, EMBEDDING_DIM] == 0.5

def test_concurrent_ranker_requests_keep_independent_buffers(cpu_ranker_service):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(2)
    inputs = []

    def model(tensor):
        inputs.append(tensor)
        barrier.wait(timeout=10)
        return tensor[:, 0].squeeze(), *(torch.zeros(len(tensor)).squeeze() for _ in range(4))

    cpu_ranker_service.model = model

    def score(value):
        return cpu_ranker_service.score_batch(
            np.full(EMBEDDING_DIM, value), [],
            [{"id": str(value), "embedding": np.full(EMBEDDING_DIM, value)}],
        )[0]["final_score"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        scores = list(executor.map(score, [0.25, 0.75]))
    assert scores == [0.25, 0.75]
    assert not np.shares_memory(inputs[0].numpy(), inputs[1].numpy())

@pytest.mark.parametrize("shape", [(2, 1), (1, FEATURE_COUNT), (3, FEATURE_COUNT),
                                   (2, FEATURE_COUNT - 1), (2, FEATURE_COUNT + 1),
                                   (FEATURE_COUNT,)])
def test_ranker_input_rejects_dense_feature_broadcasting(cpu_ranker_service, shape):
    candidates = [{"id": str(index), "embedding": np.zeros(EMBEDDING_DIM)}
                  for index in range(2)]
    with pytest.raises(ValueError, match="dense_features must have shape"):
        cpu_ranker_service._input_tensor(
            np.zeros(EMBEDDING_DIM), candidates, np.full(shape, 7.0, dtype=np.float32)
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


class _MetricFixtureRanker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "probabilities", torch.tensor([0.2, 0.4, 0.6, 0.3, 0.8])
        )

    def forward(self, features):
        return tuple(
            self.probabilities[index].expand(features.shape[0])
            for index in range(5)
        )


def _metric_fixture_loader():
    return DataLoader(
        TensorDataset(
            torch.zeros((3, 2)),
            torch.tensor([0.0, 1.0, 1.0]),
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([0.1, 0.9, 0.5]),
            torch.tensor([1.0, 0.0, 1.0]),
        ),
        batch_size=2,
        shuffle=False,
    )


def _track_training_module_items(monkeypatch):
    calls = []
    module_name = train_ranker.__name__
    original_item = torch.Tensor.item

    def track_item(tensor, *args, **kwargs):
        caller = sys._getframe(1)
        if caller.f_globals.get("__name__") == module_name:
            calls.append(caller.f_code.co_name)
        return original_item(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", track_item)
    return calls


def _expected_fixture_loss():
    # Hand-calculated per-example BCE/MSE means for the three fixture rows.
    return (
        (-math.log(0.8) - 2 * math.log(0.2)) / 3
        + (-math.log(0.4) - 2 * math.log(0.6)) / 3
        + (-2 * math.log(0.4) - math.log(0.6)) / 3
        + (0.04 + 0.36 + 0.04) / 3
        + (-2 * math.log(0.8) - math.log(0.2)) / 3
    )


def _expected_fixture_calibration():
    return {
        "ctr_brier": (0.04 + 0.64 + 0.64) / 3,
        "save_brier": (0.36 + 0.16 + 0.16) / 3,
        "github_open_brier": (0.36 + 0.16 + 0.36) / 3,
        "dwell_mse": (0.04 + 0.36 + 0.04) / 3,
        "follow_brier": (0.04 + 0.64 + 0.04) / 3,
    }


def test_weighted_metric_accumulator_detaches_and_preserves_example_weights():
    total = torch.zeros((), dtype=torch.float64)
    first_loss = torch.tensor(0.25, requires_grad=True)
    second_loss = torch.tensor(0.75, requires_grad=True)

    train_ranker._add_detached_metric(total, first_loss, 3)
    train_ranker._add_detached_metric(total, second_loss, 1)

    assert total.dtype == torch.float64
    assert total.item() == pytest.approx(1.5, rel=1e-7, abs=1e-9)
    assert not total.requires_grad
    assert total.grad_fn is None


def test_cpu_metric_paths_preserve_weighting_and_calibration_values(monkeypatch):
    model = _MetricFixtureRanker()
    loader = list(_metric_fixture_loader())
    cpu_shapes = []
    original_cpu = torch.Tensor.cpu

    def track_cpu(tensor, *args, **kwargs):
        cpu_shapes.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    actual_loss = train_ranker.evaluate(
        model, loader, torch.device("cpu"), torch.nn.BCELoss(), torch.nn.MSELoss()
    )
    monkeypatch.setattr(torch.Tensor, "cpu", track_cpu)
    actual_calibration = train_ranker.calibration_metrics(
        model, loader, torch.device("cpu")
    )

    # Preserve example weighting and per-head metrics within explicit tolerances.
    assert actual_loss == pytest.approx(_expected_fixture_loss(), rel=1e-7, abs=1e-7)
    assert actual_calibration == pytest.approx(
        _expected_fixture_calibration(), rel=1e-7, abs=1e-9
    )
    assert cpu_shapes == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_metric_paths_extract_only_at_reporting_boundaries(monkeypatch):
    model = _MetricFixtureRanker()
    loader = list(_metric_fixture_loader())
    expected_loss = _expected_fixture_loss()
    item_calls = _track_training_module_items(monkeypatch)
    actual_loss = train_ranker.evaluate(
        model.cuda(), loader, torch.device("cuda"), torch.nn.BCELoss(), torch.nn.MSELoss()
    )
    assert actual_loss == pytest.approx(expected_loss, rel=1e-7, abs=1e-9)
    assert item_calls == ["evaluate"]

    cpu_shapes = []
    original_cpu = torch.Tensor.cpu

    def track_cpu(tensor, *args, **kwargs):
        cpu_shapes.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", track_cpu)
    actual_calibration = train_ranker.calibration_metrics(
        model, loader, torch.device("cuda")
    )
    assert actual_calibration == pytest.approx(
        _expected_fixture_calibration(), rel=1e-7, abs=1e-9
    )
    assert cpu_shapes == [(5,)]
    assert item_calls == ["evaluate"]


def test_train_preserves_small_cpu_training_path(tmp_path, monkeypatch):
    class TinyTrainableRanker(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.logits = torch.nn.Parameter(
                torch.tensor([-1.0, -0.4, 0.2, -0.8, 0.7])
            )

        def forward(self, features):
            probabilities = torch.sigmoid(self.logits)
            return tuple(
                probabilities[index].expand(features.shape[0])
                for index in range(5)
            )

    data_path = tmp_path / "training.npz"
    labels = np.array([0, 1, 0, 1, 0], dtype=np.float32)
    np.savez(
        data_path,
        user_embs=np.zeros((5, EMBEDDING_DIM), dtype=np.float32),
        repo_embs=np.zeros((5, EMBEDDING_DIM), dtype=np.float32),
        dense_features=np.zeros((5, FEATURE_COUNT), dtype=np.float32),
        y_ctr=labels,
        y_save=1 - labels,
        y_gh=labels,
        y_dwell=labels,
        y_follow=1 - labels,
    )
    monkeypatch.setattr(train_ranker, "MMoEHeavyRanker", TinyTrainableRanker)
    manifest = train_ranker.train(
        data_path=data_path,
        epochs=1,
        lr=1e-3,
        batch_size=2,
        output_dir=tmp_path / "artifacts",
    )

    assert np.isfinite(manifest["offline_metrics"]["final_train_loss"])
    assert np.isfinite(manifest["offline_metrics"]["final_validation_loss"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_training_aggregates_weighted_loss_once_per_epoch(tmp_path, monkeypatch):
    class TinyTrainableRanker(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.logits = torch.nn.Parameter(
                torch.tensor([-1.0, -0.4, 0.2, -0.8, 0.7])
            )

        def forward(self, features):
            probabilities = torch.sigmoid(self.logits)
            return tuple(
                probabilities[index].expand(features.shape[0])
                for index in range(5)
            )

    row_count = 11
    labels = np.zeros(row_count, dtype=np.float32)
    data_path = tmp_path / "training.npz"
    np.savez(
        data_path,
        user_embs=np.zeros((row_count, EMBEDDING_DIM), dtype=np.float32),
        repo_embs=np.zeros((row_count, EMBEDDING_DIM), dtype=np.float32),
        dense_features=np.zeros((row_count, FEATURE_COUNT), dtype=np.float32),
        y_ctr=labels,
        y_save=labels,
        y_gh=labels,
        y_dwell=labels,
        y_follow=labels,
    )

    observed_train_batches = []
    original_compute_batch_loss = train_ranker.compute_batch_loss

    def record_real_training_loss(model, batch, device, bce_loss, mse_loss):
        loss, count = original_compute_batch_loss(
            model, batch, device, bce_loss, mse_loss
        )
        if model.training:
            observed_train_batches.append((loss.detach(), count))
        return loss, count

    monkeypatch.setattr(train_ranker, "MMoEHeavyRanker", TinyTrainableRanker)
    monkeypatch.setattr(
        train_ranker, "compute_batch_loss", record_real_training_loss
    )
    item_calls = _track_training_module_items(monkeypatch)
    epochs = 2
    manifest = train_ranker.train(
        data_path=data_path,
        epochs=epochs,
        lr=1e-2,
        batch_size=3,
        output_dir=tmp_path / "artifacts",
    )

    # 80% of 11 rows gives eight training examples: two full batches and a tail.
    assert len(observed_train_batches) == epochs * 3
    final_epoch = observed_train_batches[-3:]
    batch_sizes = [count for _, count in final_epoch]
    assert sorted(batch_sizes) == [2, 3, 3]

    # Derive the expected metric from the real pre-update batch losses in the
    # final epoch, weighting each batch mean by its example count.
    batch_losses = torch.stack([loss for loss, _ in final_epoch]).cpu().tolist()
    weighted_loss = sum(
        loss * count for loss, count in zip(batch_losses, batch_sizes)
    ) / sum(batch_sizes)
    unweighted_batch_mean = sum(batch_losses) / len(batch_losses)
    assert not math.isclose(
        weighted_loss, unweighted_batch_mean, rel_tol=1e-8, abs_tol=1e-8
    )
    assert manifest["offline_metrics"]["final_train_loss"] == pytest.approx(
        weighted_loss, rel=1e-6, abs=1e-6
    )

    # The trainer extracts one scalar for train and one for validation per epoch.
    assert item_calls.count("train") == epochs
    assert item_calls.count("evaluate") == epochs
    assert len(item_calls) == epochs * 2


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

@pytest.mark.parametrize(
    "skills,languages,topics,tags,expected",
    [
        ([" Python ", "RUST", "python"], [" python"], [" Rust "], [], 1.0),
        (["STRASSE", "Σ", "missing"], ["Straße"], ["ς"], None, 2 / 3),
        (["İ", "i"], ["i\u0307"], [], [], 0.5),
        ([None, "", "   "], ["Python"], None, None, 0.0),
        (None, None, None, None, 0.0),
        (" Python ", "PYTHON", None, None, 1.0),
        ([False, 7], False, [7, None], {}, 1.0),
        (["py"], ["python"], None, None, 0.0),
    ],
)
def test_keyword_features_preserve_golden_scores(
    loaded_ranker_service, skills, languages, topics, tags, expected
):
    # Public helper and the actual batch feature must retain exact match rules,
    # including Unicode casefolding, deduplication and missing/default values.
    assert calculate_match_score(skills, languages, topics, tags) == expected
    result = loaded_ranker_service.score_batch(
        np.zeros(EMBEDDING_DIM), skills,
        [{"id": "golden", "embedding": np.zeros(EMBEDDING_DIM),
          "languages": languages, "topics": topics, "tags": tags}],
    )
    assert result[0]["skill_match"] == pytest.approx(expected)

def test_keyword_cache_invalidates_mutable_repository_content(loaded_ranker_service):
    candidate = {"id": "same-id", "embedding": np.zeros(EMBEDDING_DIM),
                 "languages": ["Python"], "topics": [], "tags": []}
    _cached_repository_keywords.cache_clear()
    for field, value, expected in [
        (None, None, 1 / 3),
        ("languages", "RUST", 1 / 3),
        ("topics", "Python", 2 / 3),
        ("tags", "SQL", 1.0),
    ]:
        if field == "languages":
            candidate[field][:] = [value]
        elif field:
            candidate[field].append(value)
        scores = loaded_ranker_service.score_batch(
            np.zeros(EMBEDDING_DIM), ["Python", "Rust", "SQL"], [candidate]
        )
        assert scores[0]["skill_match"] == pytest.approx(expected)
    assert _cached_repository_keywords.cache_info().misses == 4
    _repository_keywords(candidate["languages"], candidate["topics"], candidate["tags"])
    assert _cached_repository_keywords.cache_info().hits == 1

def test_keyword_cache_is_bounded_and_results_are_immutable():
    _cached_repository_keywords.cache_clear()
    for index in range(4100):
        result = _repository_keywords([f"language-{index}"], None, None)
    assert isinstance(result, frozenset)
    assert _cached_repository_keywords.cache_info().currsize == 4096
    _cached_repository_keywords.cache_clear()

def test_score_batch_normalizes_single_use_user_keywords_once(loaded_ranker_service):
    candidates = [
        {"id": str(index), "embedding": np.zeros(EMBEDDING_DIM),
         "languages": ["Python"]}
        for index in range(3)
    ]
    scores = loaded_ranker_service.score_batch(
        np.zeros(EMBEDDING_DIM), iter([" Python "]), candidates
    )
    assert [score["skill_match"] for score in scores] == [1.0, 1.0, 1.0]

@pytest.mark.parametrize("topics", [["Σ" * 1025], [f"topic-{i}" for i in range(65)]])
def test_oversized_keyword_metadata_preserves_scores_without_cache_retention(topics):
    _cached_repository_keywords.cache_clear()
    expected = {topic.casefold() for topic in topics}
    assert _repository_keywords(None, topics, None) == expected
    assert calculate_match_score(topics, None, topics, None) == 1.0
    assert _cached_repository_keywords.cache_info().currsize == 0


# Ranker device selection

ARTIFACTS = Path(__file__).resolve().parents[1] / "inference"

def load_ranker(**kwargs):
    return RankerService(model_path=str(ARTIFACTS / "heavy_ranker.pt"),
                         scaler_path=str(ARTIFACTS / "feature_scaler.json"),
                         manifest_path=str(ARTIFACTS / "model_manifest.json"),
                         require_production_manifest=True, **kwargs)

def test_explicit_cpu_ignores_automatic_cuda_preference(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    service = load_ranker(device="cpu")
    assert service.ready and service.production_qualified
    assert service.device.type == "cpu"
    assert next(service.model.parameters()).device.type == "cpu"
    assert service.qualification_overridden  # Preserve the existing manifest provenance.

def test_factory_honors_explicit_cpu_on_cuda_capable_host(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("V2_HEAVY_RANKER_DEVICE", "cpu")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    retriever = QdrantV2Retriever(client=object(), heavy_ranker_enabled=True,
                                  heavy_ranker_required=True, allow_unqualified_ranker=False,
                                  minimum_eligible_repositories=0)
    assert retriever.ranker is not None
    assert retriever.ranker.device.type == "cpu"
    assert retriever.heavy_ranker_required

def test_explicit_unavailable_cuda_does_not_fall_back_to_cpu(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        load_ranker(device="cuda")

def test_invalid_runtime_device_is_a_configuration_error(monkeypatch):
    monkeypatch.setenv("V2_HEAVY_RANKER_DEVICE", "cdua")
    with pytest.raises(ValueError, match="V2_HEAVY_RANKER_DEVICE"):
        QdrantV2Retriever(client=object(), heavy_ranker_enabled=False,
                          minimum_eligible_repositories=0)
