"""Training-only statistics and the exported preprocessing/serving contract."""
import json

import numpy as np
import pytest

from Ranking_Training_module.train_ranker import (
    build_dataloaders, chronological_split_index, standardize_dense_features, train,
)
from inference.feature_spec import EMBEDDING_DIM, FEATURE_COUNT, FEATURE_ORDER
from inference.preprocessing import (
    FEATURE_DEFAULTS, PREPROCESSING_VERSION, load_scaler_document,
    scaler_document, transform_dense_features,
)
from inference.ranker_service import RankerService


def write_training_rows(path, dense):
    count = len(dense)
    np.savez(path, dense_features=dense,
             user_embs=np.zeros((count, EMBEDDING_DIM), dtype=np.float32),
             repo_embs=np.zeros((count, EMBEDDING_DIM), dtype=np.float32),
             **{key: np.zeros(count, dtype=np.float32)
                for key in ("y_ctr", "y_save", "y_gh", "y_dwell", "y_follow")})


def export_fixture(tmp_path, dense, name):
    data = tmp_path / f"{name}.npz"
    write_training_rows(data, dense)
    output = tmp_path / name
    # Tiny disposable code/export fixture, never the checked-in model artifacts.
    train(data_path=data, epochs=1, lr=.001, batch_size=4, output_dir=output)
    return output, json.loads((output / "feature_scaler.json").read_text())


def test_validation_only_changes_cannot_leak_into_fitted_scaler_or_training_rows(tmp_path):
    baseline = np.zeros((5, FEATURE_COUNT), dtype=np.float32)
    shifted = baseline.copy()
    shifted[-1] = 100
    split = chronological_split_index(len(baseline))
    for name, dense in (("baseline", baseline), ("changed-holdout", shifted)):
        _, document = export_fixture(tmp_path, dense, name)
        np.testing.assert_array_equal(document["mean"], np.zeros(FEATURE_COUNT))
        np.testing.assert_array_equal(document["scale"], np.ones(FEATURE_COUNT))
    first, _, _ = standardize_dense_features(baseline, fit_rows=split)
    second, _, _ = standardize_dense_features(shifted, fit_rows=split)
    np.testing.assert_array_equal(first[:split], second[:split])
    np.testing.assert_array_equal(second[-1], np.full(FEATURE_COUNT, 8))
    labels = tuple(np.zeros(5, dtype=np.float32) for _ in range(5))
    train_loader, validation_loader = build_dataloaders(second, labels, 4, split_index=split)
    np.testing.assert_array_equal(train_loader.dataset.tensors[0], first[:split])
    np.testing.assert_array_equal(validation_loader.dataset.tensors[0], second[split:])


def test_export_reload_matches_real_serving_inputs_for_normal_extreme_and_missing_values(tmp_path):
    dense = np.arange(10, dtype=np.float32)[:, None] * np.arange(1, FEATURE_COUNT + 1)
    dense[:, -1] = 0  # No user keywords => the serving skill-match feature is zero.
    output, document = export_fixture(tmp_path, dense, "export")
    assert document["preprocessing_version"] == PREPROCESSING_VERSION
    assert document["feature_order"] == FEATURE_ORDER
    manifest = json.loads((output / "model_manifest.json").read_text())
    assert manifest["preprocessing_version"] == PREPROCESSING_VERSION
    assert manifest["preprocessing_fit"] == {
        "split": "input_order_80_20", "training_rows": 8, "validation_rows": 2,
    }
    split = chronological_split_index(len(dense))
    _, mean, scale = standardize_dense_features(dense, fit_rows=split)
    loaded_mean, loaded_scale = load_scaler_document(json.loads(json.dumps(document)))
    rows = np.vstack([dense[2], np.full(FEATURE_COUNT, 1e100),
                      np.full(FEATURE_COUNT, -1e100), np.full(FEATURE_COUNT, np.nan)])
    rows[:, -1] = 0
    rows[3, 0], rows[3, 1] = np.inf, -np.inf
    training_transform = transform_dense_features(rows, mean, scale)
    np.testing.assert_array_equal(training_transform, transform_dense_features(rows, loaded_mean, loaded_scale))
    np.testing.assert_array_equal(training_transform[1, :-1], np.full(FEATURE_COUNT - 1, 8))
    np.testing.assert_array_equal(training_transform[2, :-1], np.full(FEATURE_COUNT - 1, -8))
    defaults = np.asarray([FEATURE_DEFAULTS[name] for name in FEATURE_ORDER], dtype=np.float32)
    expected_missing = np.clip((defaults - loaded_mean) / loaded_scale, -8, 8)
    np.testing.assert_array_equal(training_transform[-1], expected_missing)
    service = RankerService(model_path=output / "heavy_ranker.pt", scaler_path=output / "feature_scaler.json", device="cpu")
    inputs = []
    hook = service.model.register_forward_pre_hook(lambda _model, args: inputs.append(args[0].detach().cpu().numpy().copy()))
    try:
        candidates = [{**dict(zip(FEATURE_ORDER, row)), "id": str(index),
                       "embedding": np.zeros(EMBEDDING_DIM, np.float32)} for index, row in enumerate(rows)]
        assert len(service.score_batch(np.zeros(EMBEDDING_DIM), [], candidates)) == len(rows)
    finally:
        hook.remove()
    np.testing.assert_array_equal(inputs[0][:, -FEATURE_COUNT:], training_transform)
    manifest["preprocessing_version"] = "incompatible-manifest-version"
    (output / "model_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="preprocessing versions do not match"):
        RankerService(model_path=output / "heavy_ranker.pt", scaler_path=output / "feature_scaler.json", device="cpu")


@pytest.mark.parametrize("field,value", [
    ("preprocessing_version", "unknown-v999"), ("feature_order", list(reversed(FEATURE_ORDER))),
    ("standardized_clip", 9), ("missing_value_policy", "zero"),
    ("feature_defaults", [0] * FEATURE_COUNT),
])
def test_versioned_scalers_reject_incompatible_preprocessing(field, value):
    document = scaler_document(np.zeros(FEATURE_COUNT), np.ones(FEATURE_COUNT))
    document[field] = value
    with pytest.raises(ValueError, match=field):
        load_scaler_document(document)


def test_legacy_scaler_keeps_the_existing_float32_transform():
    mean = np.arange(FEATURE_COUNT, dtype=np.float64) / 3
    scale = np.arange(1, FEATURE_COUNT + 1, dtype=np.float64) / 7
    legacy = {"mean": mean.tolist(), "scale": scale.tolist()}
    loaded_mean, loaded_scale = load_scaler_document(legacy)
    rows = np.arange(3 * FEATURE_COUNT, dtype=np.float32).reshape(3, FEATURE_COUNT)
    expected = np.clip((rows - mean.astype(np.float32)) / scale.astype(np.float32), -8, 8)
    np.testing.assert_array_equal(transform_dense_features(rows, loaded_mean, loaded_scale), expected)


@pytest.mark.parametrize("mean,scale", [
    (np.zeros((2, FEATURE_COUNT)), np.ones(FEATURE_COUNT)),
    (np.full(FEATURE_COUNT, np.nan), np.ones(FEATURE_COUNT)),
    (np.zeros(FEATURE_COUNT), np.zeros(FEATURE_COUNT)),
    (np.zeros(FEATURE_COUNT), np.full(FEATURE_COUNT, np.inf)),
])
def test_invalid_scaler_parameters_fail_before_serving(mean, scale):
    with pytest.raises(ValueError, match="Feature scaler"):
        load_scaler_document({"mean": mean.tolist(), "scale": scale.tolist()})
