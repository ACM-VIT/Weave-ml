"""Train and export the production MMoE heavy ranker."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.feature_spec import (  # noqa: E402
    EMBEDDING_DIM,
    FEATURE_COUNT,
    FEATURE_ORDER,
    FEATURE_SPEC_VERSION,
    INPUT_DIM,
    RANKER_MODEL_VERSION,
)
from inference.ranker_service import MMoEHeavyRanker  # noqa: E402
from inference.preprocessing import (  # noqa: E402
    PREPROCESSING_VERSION, fit_dense_features, scaler_document, transform_dense_features,
)
from inference.value_function import VALUE_WEIGHTS  # noqa: E402
from config import (  # noqa: E402
    EMBEDDING_MODEL_REVISION,
    REPOSITORY_EMBEDDING_MODEL,
    REPOSITORY_EMBEDDING_VERSION,
)


DATA_KEYS = (
    "user_embs",
    "repo_embs",
    "dense_features",
    "y_ctr",
    "y_save",
    "y_gh",
    "y_dwell",
    "y_follow",
)




def training_array_sha256(arrays):
    """Hash the exact validated arrays used, independent of NPZ compression."""
    checksum = hashlib.sha256()
    for key in DATA_KEYS:
        value = np.ascontiguousarray(arrays[key])
        if value.dtype.hasobject:
            raise ValueError("training source cannot contain object arrays")
        header = json.dumps({"name": key, "dtype": value.dtype.str,
                             "shape": list(value.shape)}, sort_keys=True,
                            separators=(",", ":")).encode()
        checksum.update(len(header).to_bytes(8, "big"))
        checksum.update(header)
        checksum.update(value.nbytes.to_bytes(8, "big"))
        checksum.update(value.tobytes(order="C"))
    return checksum.hexdigest()


def training_source_identity():
    """Local-only exact code identity; no remote fallback or symbolic revision."""
    import os
    import subprocess
    environment = {**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0",
                   "GIT_OPTIONAL_LOCKS": "0"}
    def git(*arguments):
        return subprocess.check_output(["git", "--no-lazy-fetch", "--no-pager", *arguments],
            cwd=REPO_ROOT, env=environment, timeout=5)
    try:
        commit = git("rev-parse", "HEAD").decode().strip()
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            raise ValueError("training requires an exact Git commit")
        names = git("ls-files", "-z", "--", "*.py", "pyproject.toml", "uv.lock").decode().split("\0")
        hashes = {name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
                  for name in sorted(names) if name}
        if not hashes:
            raise ValueError("training source inventory is empty")
        return {"code_commit": commit, "source_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
                "source_sha256": hashes}
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("training provenance requires compatible local-only Git and readable source") from exc


def load_training_data(data_path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate the arrays required by the ranker trainer."""
    with np.load(data_path) as data:
        missing_keys = [key for key in DATA_KEYS if key not in data]
        if missing_keys:
            raise ValueError(
                f"Training data is missing required arrays: {missing_keys}"
            )
        arrays = {key: np.asarray(data[key]) for key in DATA_KEYS}

    dense_features = arrays["dense_features"]
    if dense_features.ndim != 2 or dense_features.shape[1] != FEATURE_COUNT:
        actual_count = dense_features.shape[1] if dense_features.ndim == 2 else None
        raise ValueError(
            f"dense_features has {actual_count} columns; expected {FEATURE_COUNT} "
            f"in FEATURE_ORDER {FEATURE_ORDER}"
        )

    row_count = len(dense_features)
    mismatched = {
        key: len(value)
        for key, value in arrays.items()
        if len(value) != row_count
    }
    if mismatched:
        raise ValueError(
            f"All training arrays must contain {row_count} rows; got {mismatched}"
        )
    if row_count < 3:
        raise ValueError("Training data must contain at least 3 rows")

    for embedding_key in ("user_embs", "repo_embs"):
        embeddings = arrays[embedding_key]
        if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"{embedding_key} must have shape (rows, {EMBEDDING_DIM})"
            )

    return arrays


def standardize_dense_features(
    dense_features: np.ndarray,
    *, fit_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the chronological training prefix, then transform both partitions."""
    if not 1 <= fit_rows <= len(dense_features):
        raise ValueError("fit_rows must select a non-empty training prefix")
    mean, scale = fit_dense_features(dense_features[:fit_rows])
    return transform_dense_features(dense_features, mean, scale), mean, scale


def chronological_split_index(row_count: int) -> int:
    split_index = int(row_count * 0.8)
    if split_index < 2 or split_index >= row_count:
        raise ValueError("The chronological split requires at least 2 training rows")
    return split_index


def build_dataloaders(
    X: np.ndarray,
    labels: tuple[np.ndarray, ...],
    batch_size: int,
    *, split_index: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create an 80/20 chronological train/validation split."""
    split_index = chronological_split_index(len(X)) if split_index is None else split_index
    if split_index < 2 or split_index >= len(X):
        raise ValueError("The chronological split requires at least 2 training rows")

    def make_dataset(start: int, end: int) -> TensorDataset:
        tensors = [torch.from_numpy(X[start:end])]
        tensors.extend(
            torch.from_numpy(label[start:end])
            for label in labels
        )
        return TensorDataset(*tensors)

    train_dataset = make_dataset(0, split_index)
    val_dataset = make_dataset(split_index, len(X))
    effective_batch_size = min(batch_size, len(train_dataset))
    drop_last = len(train_dataset) % effective_batch_size == 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


def compute_batch_loss(
    model: MMoEHeavyRanker,
    batch: tuple[torch.Tensor, ...] | list[torch.Tensor],
    device: torch.device,
    bce_loss: nn.BCELoss,
    mse_loss: nn.MSELoss,
) -> tuple[torch.Tensor, int]:
    """Compute the summed five-head training objective for one batch."""
    x, y_ctr, y_save, y_gh, y_dwell, y_follow = (
        tensor.to(device) for tensor in batch
    )
    predictions = [prediction.reshape(-1) for prediction in model(x)]
    p_ctr, p_save, p_gh, p_dwell, p_follow = predictions
    loss = (
        bce_loss(p_ctr, y_ctr)
        + bce_loss(p_save, y_save)
        + bce_loss(p_gh, y_gh)
        + mse_loss(p_dwell, y_dwell)
        + bce_loss(p_follow, y_follow)
    )
    return loss, x.shape[0]


def evaluate(
    model: MMoEHeavyRanker,
    data_loader: DataLoader,
    device: torch.device,
    bce_loss: nn.BCELoss,
    mse_loss: nn.MSELoss,
) -> float:
    """Return the example-weighted validation loss."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch in data_loader:
            loss, batch_count = compute_batch_loss(
                model,
                batch,
                device,
                bce_loss,
                mse_loss,
            )
            total_loss += loss.item() * batch_count
            total_examples += batch_count

    if total_examples == 0:
        raise ValueError("Validation split is empty")
    return total_loss / total_examples


def calibration_metrics(
    model: MMoEHeavyRanker,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute validation Brier/MSE metrics for artifact qualification."""
    names = ("ctr", "save", "github_open", "dwell", "follow")
    squared_error = {name: 0.0 for name in names}
    total_examples = 0
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            x, *targets = (tensor.to(device) for tensor in batch)
            predictions = [value.reshape(-1) for value in model(x)]
            for name, prediction, target in zip(names, predictions, targets):
                squared_error[name] += float(
                    torch.sum((prediction - target.reshape(-1)) ** 2).item()
                )
            total_examples += int(x.shape[0])
    if total_examples == 0:
        raise ValueError("Validation split is empty")
    return {
        f"{name}_brier" if name != "dwell" else "dwell_mse": value / total_examples
        for name, value in squared_error.items()
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train(
    *,
    data_path: str | Path,
    epochs: int,
    lr: float,
    batch_size: int,
    output_dir: str | Path,
    model_version: str = RANKER_MODEL_VERSION,
    training_data_identity: str | None = None,
    training_data_type: str = "synthetic",
    code_version: str = "heavy-ranker-runtime-v2",
) -> dict[str, object]:
    """Train the heavy ranker and write its production artifacts."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 for BatchNorm training")
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("model_version must be a non-empty string")
    model_version = model_version.strip()
    if training_data_type not in {"synthetic", "real_telemetry"}:
        raise ValueError("training_data_type must be synthetic or real_telemetry")
    training_data_identity = (
        str(training_data_identity).strip()
        if training_data_identity is not None
        else Path(data_path).name
    )
    if not training_data_identity:
        raise ValueError("training_data_identity must be non-empty")
    if not isinstance(code_version, str) or not code_version.strip():
        raise ValueError("code_version must be a non-empty string")

    arrays = load_training_data(data_path)
    dataset_sha256 = training_array_sha256(arrays)
    source_identity = training_source_identity()
    split_index = chronological_split_index(len(arrays["dense_features"]))
    scaled_dense, scaler_mean, scaler_scale = standardize_dense_features(
        arrays["dense_features"], fit_rows=split_index,
    )
    X = np.concatenate(
        [
            arrays["user_embs"].astype(np.float32, copy=False),
            arrays["repo_embs"].astype(np.float32, copy=False),
            scaled_dense,
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    if X.shape[1] != INPUT_DIM:
        raise ValueError(f"Training input has {X.shape[1]} columns; expected {INPUT_DIM}")

    labels = tuple(
        arrays[key].astype(np.float32, copy=False).reshape(-1)
        for key in ("y_ctr", "y_save", "y_gh", "y_dwell", "y_follow")
    )
    train_loader, val_loader = build_dataloaders(X, labels, batch_size, split_index=split_index)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MMoEHeavyRanker(INPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    final_train_loss = 0.0
    final_val_loss = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_examples = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, batch_count = compute_batch_loss(
                model,
                batch,
                device,
                bce_loss,
                mse_loss,
            )
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch_count
            total_train_examples += batch_count

        if total_train_examples == 0:
            raise ValueError("Training split did not produce any batches")
        final_train_loss = total_train_loss / total_train_examples
        final_val_loss = evaluate(
            model,
            val_loader,
            device,
            bce_loss,
            mse_loss,
        )
        print(
            f"Epoch {epoch}/{epochs} - "
            f"train_loss={final_train_loss:.6f} - "
            f"val_loss={final_val_loss:.6f}"
        )

    calibration = calibration_metrics(model, val_loader, device)
    if training_source_identity() != source_identity:
        raise ValueError("training source changed before artifact export")
    if training_array_sha256(arrays) != dataset_sha256:
        raise ValueError("training input arrays changed during training")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "heavy_ranker.pt"
    scaler_path = destination / "feature_scaler.json"
    manifest_path = destination / "model_manifest.json"

    torch.save(model.state_dict(), model_path)
    scaler_path.write_text(
        json.dumps(
            scaler_document(scaler_mean, scaler_scale),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "model_file": model_path.name,
        "scaler_file": scaler_path.name,
        "model_version": model_version,
        "embedding_version": REPOSITORY_EMBEDDING_VERSION,
        "embedding_model": REPOSITORY_EMBEDDING_MODEL,
        "embedding_identity": {"model": REPOSITORY_EMBEDDING_MODEL, "revision": EMBEDDING_MODEL_REVISION,
            "dimension": EMBEDDING_DIM, "pooling": "mean", "normalized": True,
            "query_prompt": "", "document_prompt": "", "recipe": REPOSITORY_EMBEDDING_VERSION},
        "embedding_model_revision": EMBEDDING_MODEL_REVISION,
        "compatible_embedding_versions": [REPOSITORY_EMBEDDING_VERSION],
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "input_dim": INPUT_DIM,
        "embedding_dim": EMBEDDING_DIM,
        "feature_count": FEATURE_COUNT,
        "model_sha256": _sha256(model_path),
        "scaler_sha256": _sha256(scaler_path),
        "value_function_version": "v2",
        "code_version": code_version.strip(),
        **source_identity,
        "value_weights": VALUE_WEIGHTS,
        "training_data": {
            "identity": training_data_identity,
            "type": training_data_type,
            "content_sha256": dataset_sha256,
            "content_hash_scope": "validated ordered DATA_KEYS; dtype/shape and contiguous array bytes",
        },
        "training_split_sha256": hashlib.sha256(json.dumps({"dataset_sha256": dataset_sha256,
            "split": "input_order_80_20", "training_range": [0, split_index],
            "validation_range": [split_index, len(X)]}, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "preprocessing_fit": {
            "split": "input_order_80_20",
            "training_rows": split_index,
            "validation_rows": len(X) - split_index,
        },
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "offline_metrics": {
            "final_train_loss": final_train_loss,
            "final_validation_loss": final_val_loss,
        },
        "calibration_metrics": calibration,
        "epochs": epochs,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved manifest to {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="training_data.npz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--output-dir", default="inference/")
    parser.add_argument("--model-version", default=RANKER_MODEL_VERSION)
    parser.add_argument("--training-data-identity", default=None)
    parser.add_argument(
        "--training-data-type",
        choices=("synthetic", "real_telemetry"),
        default="synthetic",
    )
    parser.add_argument("--code-version", default="heavy-ranker-runtime-v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        data_path=args.data_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        model_version=args.model_version,
        training_data_identity=args.training_data_identity,
        training_data_type=args.training_data_type,
        code_version=args.code_version,
    )


if __name__ == "__main__":
    main()
