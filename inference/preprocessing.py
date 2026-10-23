"""Versioned dense-feature preprocessing shared by training and serving."""
from __future__ import annotations

import numpy as np

from inference.feature_spec import FEATURE_COUNT, FEATURE_ORDER

PREPROCESSING_VERSION = "dense-standardize-clip-v1"
STANDARDIZED_FEATURE_CLIP = 8.0
FEATURE_DEFAULTS = dict(zip(FEATURE_ORDER, (0.5, 0.5, 1000.0, 0.0, 0.0, 0.0, 365.0, 0.0, 0.0, 0.0)))
_METADATA = {
    "preprocessing_version": PREPROCESSING_VERSION,
    "feature_order": FEATURE_ORDER,
    "standardized_clip": STANDARDIZED_FEATURE_CLIP,
    "missing_value_policy": "nonfinite_to_feature_default",
    "feature_defaults": [FEATURE_DEFAULTS[name] for name in FEATURE_ORDER],
}


def _dense_values(values: np.ndarray) -> np.ndarray:
    dense = np.asarray(values, dtype=np.float64)
    if dense.ndim != 2 or dense.shape[1] != FEATURE_COUNT:
        raise ValueError(f"Dense features must have shape (rows, {FEATURE_COUNT})")
    # The same fixed defaults are applied before fitting and before transforming.
    # Zero is a real observation, not a missing-value marker.
    return np.where(np.isfinite(dense), dense, _METADATA["feature_defaults"])


def validate_scaler(mean: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(over="ignore", under="ignore"):
        mean = np.asarray(mean, dtype=np.float32)
        scale = np.asarray(scale, dtype=np.float32)
    if mean.shape != (FEATURE_COUNT,) or scale.shape != (FEATURE_COUNT,):
        raise ValueError(f"Feature scaler vectors must have expected {FEATURE_COUNT} features")
    if not np.all(np.isfinite(mean)):
        raise ValueError("Feature scaler means must all be finite float32 values")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Feature scaler scales must be finite and positive float32 values")
    return mean, scale


def fit_dense_features(training_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit population statistics using only explicitly selected training rows."""
    dense = _dense_values(training_rows)
    if not len(dense):
        raise ValueError("Cannot fit a feature scaler without training rows")
    with np.errstate(over="ignore", invalid="ignore"):
        mean = dense.mean(axis=0)
        scale = dense.std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    validate_scaler(mean, scale)
    return mean, scale


def transform_dense_features(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Impute fixed defaults, standardize in serving float32 arithmetic, clip."""
    mean, scale = validate_scaler(mean, scale)
    with np.errstate(over="ignore", under="ignore"):
        dense = _dense_values(values).astype(np.float32)
        scaled = (dense - mean) / scale
    return np.clip(scaled, -STANDARDIZED_FEATURE_CLIP, STANDARDIZED_FEATURE_CLIP)


def scaler_document(mean: np.ndarray, scale: np.ndarray) -> dict:
    validate_scaler(mean, scale)
    metadata = {key: value.copy() if isinstance(value, list) else value for key, value in _METADATA.items()}
    return {**metadata, "mean": np.asarray(mean).tolist(), "scale": np.asarray(scale).tolist()}


def load_scaler_document(document: dict) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(document, dict):
        raise ValueError("Feature scaler must be a JSON object")
    # The checked-in historical scaler has only mean/scale. Keep it compatible
    # with its existing float32 serving transform without rewriting the artifact.
    if any(key in document for key in _METADATA):
        for key, expected in _METADATA.items():
            if document.get(key) != expected:
                raise ValueError(f"Feature scaler has incompatible {key}")
    return validate_scaler(document["mean"], document["scale"])
