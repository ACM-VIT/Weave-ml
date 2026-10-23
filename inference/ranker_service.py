import json
import hashlib
import logging
import math
import os
import threading
import time
from collections.abc import Iterable
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from inference.feature_spec import (
    EMBEDDING_DIM,
    FEATURE_COUNT,
    FEATURE_ORDER,
    FEATURE_SPEC_VERSION,
    INPUT_DIM,
    RANKER_MODEL_VERSION,
)
from inference.value_function import VALUE_WEIGHTS
from config import REPOSITORY_EMBEDDING_VERSION
from utils.deadline import check_deadline


logger = logging.getLogger("pipeline.ranker")
STANDARDIZED_FEATURE_CLIP = 8.0
_SHA256_HEX_LENGTH = 64
_KEYWORD_CACHE_MAX_ELEMENTS = 64
_KEYWORD_CACHE_MAX_CHARACTERS = 1024

class MMoEHeavyRanker(nn.Module):
    def __init__(self, input_dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = 5
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU()
            ) for _ in range(self.num_experts)
        ])
        
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, self.num_experts), nn.Softmax(dim=1))
            for _ in range(self.num_tasks)
        ])
        
        self.head_ctr = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.head_save = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.head_gh = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.head_dwell = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.ReLU())
        self.head_follow = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        gate_ctr = self.gates[0](x).unsqueeze(2)
        out_ctr = self.head_ctr(torch.sum(expert_outputs * gate_ctr, dim=1)).squeeze()
        
        gate_save = self.gates[1](x).unsqueeze(2)
        out_save = self.head_save(torch.sum(expert_outputs * gate_save, dim=1)).squeeze()
        
        gate_gh = self.gates[2](x).unsqueeze(2)
        out_gh = self.head_gh(torch.sum(expert_outputs * gate_gh, dim=1)).squeeze()
        
        gate_dwell = self.gates[3](x).unsqueeze(2)
        out_dwell = self.head_dwell(torch.sum(expert_outputs * gate_dwell, dim=1)).squeeze()
        
        gate_follow = self.gates[4](x).unsqueeze(2)
        out_follow = self.head_follow(torch.sum(expert_outputs * gate_follow, dim=1)).squeeze()
        
        return out_ctr, out_save, out_gh, out_dwell, out_follow

def _keywords(values: object) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Iterable):
        values = [values]
    return {
        text.casefold()
        for value in values
        if value is not None and (text := str(value).strip())
    }


def _keyword_content(values: object) -> tuple[str, ...]:
    """Snapshot content, not identity, so mutable metadata cannot stale the cache."""
    if values is None:
        return ()
    if isinstance(values, str):
        return (str(values),)
    if not isinstance(values, Iterable):
        values = [values]
    return tuple(str(value) for value in values if value is not None)


@lru_cache(maxsize=4096)
def _cached_repository_keywords(
    languages: tuple[str, ...], topics: tuple[str, ...], tags: tuple[str, ...]
) -> frozenset[str]:
    # Immutable results are safe to share across ranking requests. The bounded,
    # content-keyed cache invalidates for any language/topic/tag content change.
    return frozenset(_keywords(languages) | _keywords(topics) | _keywords(tags))


def _repository_keywords(languages, topics, tags) -> frozenset[str]:
    content = (
        _keyword_content(languages), _keyword_content(topics), _keyword_content(tags)
    )
    # Entry count alone cannot bound retained metadata. Bypass reuse for large
    # values without truncating them or changing their matching semantics.
    if (
        sum(map(len, content)) > _KEYWORD_CACHE_MAX_ELEMENTS
        or sum(len(value) for values in content for value in values)
        > _KEYWORD_CACHE_MAX_CHARACTERS
    ):
        return _cached_repository_keywords.__wrapped__(*content)
    return _cached_repository_keywords(*content)


def _keyword_match_score(user_keywords, repo_keywords):
    return len(user_keywords & repo_keywords) / len(user_keywords)


def calculate_match_score(user_interests_skills, repo_languages, repo_topics, repo_tags):
    """Calculates the percentage of overlapping keywords dynamically."""
    user_keywords = _keywords(user_interests_skills)
    if not user_keywords:
        return 0.0
    return _keyword_match_score(
        user_keywords, _repository_keywords(repo_languages, repo_topics, repo_tags)
    )

class RankerService:
    def __init__(
        self,
        model_path="heavy_ranker.pt",
        scaler_path="feature_scaler.json",
        emb_dim=384,
        manifest_path=None,
        expected_embedding_version=None,
        expected_embedding_versions=None,
        expected_embedding_model=None,
        expected_embedding_model_revision=None,
        require_production_manifest=False,
        device="auto",
    ):
        self._warmup_lock = threading.Lock()
        self._warmup_status = {"state": "not_started", "duration_seconds": None}
        if not isinstance(device, str) or device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Ranker device must be auto, cpu or cuda")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        self.device = torch.device(device)
        self.emb_dim = EMBEDDING_DIM
        self.model_version = RANKER_MODEL_VERSION
        self.embedding_version = REPOSITORY_EMBEDDING_VERSION
        self.compatible_embedding_versions = {self.embedding_version}
        self.feature_spec_version = FEATURE_SPEC_VERSION
        self.embedding_model: str | None = None
        self.embedding_model_revision: str | None = None
        self.production_qualified = False
        self.qualification_overridden = False
        self.qualification_errors: tuple[str, ...] = (
            "model manifest is unavailable",
        )
        self.manifest: dict = {}
        
        # Total input dim = User_emb(384) + Repo_emb(384) + DenseFeatures(10)
        self.input_dim = INPUT_DIM

        if manifest_path is None:
            manifest_path = os.path.join(
                os.path.dirname(os.path.abspath(model_path)),
                "model_manifest.json",
            )
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
            if not isinstance(manifest, dict):
                raise ValueError("Model manifest must be a JSON object")
            self.manifest = manifest
            expected_contract = {
                "input_dim": INPUT_DIM,
                "embedding_dim": EMBEDDING_DIM,
                "feature_count": FEATURE_COUNT,
                "feature_spec_version": FEATURE_SPEC_VERSION,
            }
            mismatches = {
                key: (manifest.get(key), expected)
                for key, expected in expected_contract.items()
                if manifest.get(key) != expected
            }
            if mismatches:
                raise ValueError(
                    f"Model manifest is incompatible with inference: {mismatches}"
                )
            self.model_version = str(
                manifest.get("model_version") or RANKER_MODEL_VERSION
            )
            self.embedding_version = str(
                manifest.get("embedding_version")
                or REPOSITORY_EMBEDDING_VERSION
            )
            raw_embedding_model = manifest.get("embedding_model")
            if raw_embedding_model is not None:
                self.embedding_model = str(raw_embedding_model).strip() or None
            raw_embedding_revision = manifest.get("embedding_model_revision")
            if raw_embedding_revision is not None:
                self.embedding_model_revision = (
                    str(raw_embedding_revision).strip() or None
                )
            raw_compatible_versions = manifest.get("compatible_embedding_versions")
            if raw_compatible_versions is None:
                raw_compatible_versions = [self.embedding_version]
            if (
                not isinstance(raw_compatible_versions, list)
                or not raw_compatible_versions
                or not all(
                    isinstance(version, str) and version.strip()
                    for version in raw_compatible_versions
                )
            ):
                raise ValueError(
                    "Model manifest compatible_embedding_versions must be a "
                    "non-empty list of strings"
                )
            self.compatible_embedding_versions = {
                version.strip() for version in raw_compatible_versions
            }
            self.qualification_errors = tuple(
                self._production_manifest_errors(
                    manifest,
                    model_path=Path(model_path),
                    scaler_path=Path(scaler_path),
                    expected_embedding_model=expected_embedding_model,
                    expected_embedding_model_revision=expected_embedding_model_revision,
                )
            )
            self.qualification_overridden = (
                manifest.get("production_qualified") is True
                and manifest.get("qualification_method") == "manual_override"
            )
            self.production_qualified = (
                self.qualification_overridden or not self.qualification_errors
            )
            if self.qualification_overridden and self.qualification_errors:
                logger.warning(
                    "Heavy ranker is manually production-qualified despite "
                    "automatic qualification failures: %s",
                    "; ".join(self.qualification_errors),
                )
            logger.info("Model manifest loaded successfully from %s", manifest_path)

        requested_embedding_versions = set(expected_embedding_versions or ())
        if expected_embedding_version:
            requested_embedding_versions.add(expected_embedding_version)
        incompatible_versions = (
            requested_embedding_versions - self.compatible_embedding_versions
        )
        if incompatible_versions:
            raise ValueError(
                "Heavy ranker embedding contract is incompatible with serving: "
                f"expected {sorted(requested_embedding_versions)!r}, "
                "incompatible versions are "
                f"{sorted(incompatible_versions)!r}, compatible versions are "
                f"{sorted(self.compatible_embedding_versions)!r}"
            )
        if require_production_manifest and not self.production_qualified:
            raise RuntimeError(
                "Heavy ranker manifest is not production-qualified: "
                + "; ".join(self.qualification_errors)
            )
        
        # Load Model
        self.model = MMoEHeavyRanker(self.input_dim).to(self.device)
        self._model_loaded = False
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self._model_loaded = True
            logger.info("Heavy Ranker model loaded successfully from %s", model_path)
        else:
            logger.warning("Heavy Ranker model not found at %s", model_path)
        self.model.eval()
            
        # Load Scaler safely from JSON
        if os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                scaler_params = json.load(f)
            self.scaler_mean = np.array(scaler_params['mean'], dtype=np.float32)
            self.scaler_scale = np.array(scaler_params['scale'], dtype=np.float32)
            if len(self.scaler_mean) != FEATURE_COUNT:
                raise ValueError(
                    f"Feature scaler has {len(self.scaler_mean)} features; "
                    f"expected {FEATURE_COUNT}"
                )
            if len(self.scaler_scale) != FEATURE_COUNT:
                raise ValueError(
                    f"Feature scaler has {len(self.scaler_scale)} scale values; "
                    f"expected {FEATURE_COUNT}"
                )
            if not np.all(np.isfinite(self.scaler_mean)):
                raise ValueError("Feature scaler means must all be finite")
            if not np.all(np.isfinite(self.scaler_scale)) or np.any(
                self.scaler_scale <= 0
            ):
                raise ValueError("Feature scaler scales must be finite and positive")
            logger.info("Feature scaler loaded successfully from %s", scaler_path)
        else:
            raise RuntimeError(f"{scaler_path} not found; refusing to use unscaled features")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as artifact:
            for block in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @classmethod
    def _production_manifest_errors(
        cls,
        manifest: dict,
        *,
        model_path: Path,
        scaler_path: Path,
        expected_embedding_model: str | None,
        expected_embedding_model_revision: str | None,
    ) -> list[str]:
        """Return bounded, non-secret production qualification failures.

        Loading an artifact and qualifying it for broad production traffic are
        intentionally separate.  Development can inspect older artifacts,
        while serving keeps them disabled until provenance, evaluation, and
        calibration are all explicit and verifiable.
        """
        errors: list[str] = []

        def non_empty_string(key: str) -> str | None:
            value = manifest.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"missing or invalid {key}")
                return None
            return value.strip()

        model_file = non_empty_string("model_file")
        scaler_file = non_empty_string("scaler_file")
        non_empty_string("model_version")
        non_empty_string("embedding_version")
        model_hash = non_empty_string("model_sha256")
        scaler_hash = non_empty_string("scaler_sha256")
        embedding_model = non_empty_string("embedding_model")
        embedding_revision = non_empty_string("embedding_model_revision")
        non_empty_string("value_function_version")
        non_empty_string("code_version")

        if model_file is not None and model_file != model_path.name:
            errors.append("model_file does not match the loaded model artifact")
        if scaler_file is not None and scaler_file != scaler_path.name:
            errors.append("scaler_file does not match the loaded scaler artifact")

        compatible_versions = manifest.get("compatible_embedding_versions")
        if (
            not isinstance(compatible_versions, list)
            or not compatible_versions
            or any(
                not isinstance(version, str) or not version.strip()
                for version in compatible_versions
            )
        ):
            errors.append("missing or invalid compatible_embedding_versions")

        if model_hash is not None:
            normalized = model_hash.casefold()
            if (
                len(normalized) != _SHA256_HEX_LENGTH
                or any(character not in "0123456789abcdef" for character in normalized)
            ):
                errors.append("model_sha256 must be a 64-character hex digest")
            elif not model_path.is_file() or cls._sha256(model_path) != normalized:
                errors.append("model_sha256 does not match the model artifact")
        if scaler_hash is not None:
            normalized = scaler_hash.casefold()
            if (
                len(normalized) != _SHA256_HEX_LENGTH
                or any(character not in "0123456789abcdef" for character in normalized)
            ):
                errors.append("scaler_sha256 must be a 64-character hex digest")
            elif not scaler_path.is_file() or cls._sha256(scaler_path) != normalized:
                errors.append("scaler_sha256 does not match the scaler artifact")

        if expected_embedding_model and embedding_model != expected_embedding_model:
            errors.append("embedding_model does not match the serving contract")
        if (
            expected_embedding_model_revision
            and embedding_revision != expected_embedding_model_revision
        ):
            errors.append(
                "embedding_model_revision does not match the serving contract"
            )

        weights = manifest.get("value_weights")
        if weights != VALUE_WEIGHTS:
            errors.append("value_weights do not match the serving value function")

        training_data = manifest.get("training_data")
        if not isinstance(training_data, dict):
            errors.append("missing or invalid training_data")
        else:
            identity = training_data.get("identity")
            data_type = training_data.get("type")
            if not isinstance(identity, str) or not identity.strip():
                errors.append("missing or invalid training_data.identity")
            if not isinstance(data_type, str) or not data_type.strip():
                errors.append("missing or invalid training_data.type")
            elif "synthetic" in data_type.casefold():
                errors.append("synthetic training data is not production-qualified")

        raw_training_timestamp = manifest.get("training_timestamp")
        if not isinstance(raw_training_timestamp, str):
            errors.append("missing or invalid training_timestamp")
        else:
            try:
                parsed_timestamp = datetime.fromisoformat(
                    raw_training_timestamp.replace("Z", "+00:00")
                )
                if parsed_timestamp.tzinfo is None:
                    raise ValueError
            except ValueError:
                errors.append("training_timestamp must be an ISO-8601 timestamp")

        for key in ("offline_metrics", "calibration_metrics"):
            metrics = manifest.get(key)
            if not isinstance(metrics, dict) or not metrics:
                errors.append(f"missing or invalid {key}")
                continue
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in metrics.values()
            ):
                errors.append(f"{key} values must be finite numbers")

        return list(dict.fromkeys(errors))

    @property
    def ready(self) -> bool:
        return self._model_loaded

    def warmup_status(self) -> dict:
        """Read warmup state without acquiring the inference/warmup lock."""
        status = self._warmup_status
        return {
            "heavy_ranker_warmup_ready": status["state"] == "ready",
            "heavy_ranker_warmup_state": status["state"],
            "heavy_ranker_warmup_duration_seconds": status["duration_seconds"],
        }

    def warmup(self, *, stop: threading.Event | None = None) -> dict:
        """Warm real serving input preparation and forward once per model."""
        with self._warmup_lock:
            if self._warmup_status["state"] == "ready":
                return self.warmup_status()
            if self._warmup_status["state"] == "failed":
                raise RuntimeError("Heavy ranker inference warmup previously failed")
            started = time.monotonic()
            self._warmup_status = {"state": "warming", "duration_seconds": None}
            try:
                if not self.ready:
                    raise RuntimeError("Heavy ranker artifact is not loaded")
                if stop is not None and stop.is_set():
                    raise RuntimeError("Inference warmup was stopped")
                # A normalized, fixed synthetic embedding and one candidate.
                # No retriever, storage, real IDs, or user/corpus writes occur.
                vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                vector[0] = 1.0
                inputs = self._input_tensor(
                    vector, [{"id": "runtime-warmup", "embedding": vector}],
                    np.zeros((1, FEATURE_COUNT), dtype=np.float32),
                )
                with torch.inference_mode():
                    outputs = self.model(inputs)
                    if len(outputs) != 5 or any(output.numel() != 1 for output in outputs):
                        raise ValueError("Heavy ranker warmup returned invalid prediction shapes")
                    # Transfer/synchronize the real device work before declaring
                    # ready; inspect raw outputs before serving's clipping step.
                    predictions = torch.stack([
                        output.reshape(-1) for output in outputs
                    ], dim=1).cpu().numpy()
                if predictions.shape != (1, 5) or not np.all(np.isfinite(predictions)):
                    raise ValueError("Heavy ranker warmup returned non-finite predictions")
                if stop is not None and stop.is_set():
                    raise RuntimeError("Inference warmup was stopped")
            except Exception:
                self._warmup_status = {
                    "state": "failed", "duration_seconds": time.monotonic() - started,
                }
                raise
            self._warmup_status = {
                "state": "ready", "duration_seconds": time.monotonic() - started,
            }
            return self.warmup_status()

    def score_batch(self, user_embedding, user_skills, candidate_repos):
        """Compatibility diagnostic API, including skill match and predictions.

        Serving callers should use score_batch_compact to avoid debug objects.
        Training continues to call MMoEHeavyRanker.forward directly, not here.
        """
        return self._score_batch(
            user_embedding, user_skills, candidate_repos, diagnostics=True
        )

    def score_batch_compact(self, user_embedding, user_skills, candidate_repos):
        """Preferred serving API: only repository ID and final score."""
        return self._score_batch(user_embedding, user_skills, candidate_repos)

    def _embedding(self, value, *, field_name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (self.emb_dim,):
            raise ValueError(
                f"{field_name} must contain exactly {self.emb_dim} values; "
                f"got shape {vector.shape}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{field_name} must contain only finite values")
        return vector

    @staticmethod
    def _finite_feature(value, *, default: float) -> float:
        if value is None or isinstance(value, bool):
            return default
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if np.isfinite(number) else default

    def _input_tensor(self, user_vector, candidate_repos, dense_features):
        """Build a request-owned C-contiguous float32 inference buffer.

        Assignment copies external vectors (including strided/read-only arrays)
        into this one buffer. CPU from_numpy shares only this private buffer,
        retained by the tensor for its full lifetime; no caller array is aliased.
        Device transfer is blocking by default and owns its destination storage.
        No request buffer is stored or reused on this service.
        """
        dense_features = np.asarray(dense_features, dtype=np.float32)
        expected_dense_shape = (len(candidate_repos), FEATURE_COUNT)
        if dense_features.shape != expected_dense_shape:
            raise ValueError(
                f"dense_features must have shape {expected_dense_shape}; "
                f"got shape {dense_features.shape}"
            )
        X = np.empty((len(candidate_repos), self.input_dim), dtype=np.float32)
        X[:, :self.emb_dim] = user_vector
        for index, repo in enumerate(candidate_repos):
            X[index, self.emb_dim:2 * self.emb_dim] = self._embedding(
                repo.get("embedding"),
                field_name=f"candidate {repo.get('id', '<unknown>')} embedding",
            )
        X[:, 2 * self.emb_dim:] = dense_features
        if not np.all(np.isfinite(X)):
            raise ValueError("Heavy ranker input contains non-finite values")
        return torch.from_numpy(X).to(self.device)

    def _score_batch(self, user_embedding, user_skills, candidate_repos, *, diagnostics=False):
        """
        Executes the MMoE network on a micro-batch (e.g. 15 or 150 repos).
        """
        check_deadline()
        if not self._model_loaded:
            logger.warning("Heavy Ranker model is not loaded; returning no scores")
            return []

        if not candidate_repos:
            return []

        # 1. Prepare inputs
        user_vector = self._embedding(user_embedding, field_name="user_embedding")
        user_keywords = _keywords(user_skills)
        dense_features = []
        for repo in candidate_repos:
            # Dynamically calculate the cross-feature!
            skill_match = (
                _keyword_match_score(
                    user_keywords,
                    _repository_keywords(
                        repo.get('languages'), repo.get('topics'), repo.get('tags')
                    ),
                )
                if user_keywords else 0.0
            )
            
            # The EXACT 10 features generated in DataGen:
            # batch_doc, batch_health, batch_readme, batch_stars, batch_forks, batch_issues, batch_pushed, batch_activity, batch_trend, skill_match_score
            feature_values = {
                'doc_quality': self._finite_feature(repo.get('doc_quality'), default=0.5),
                'code_health': self._finite_feature(repo.get('code_health'), default=0.5),
                'readme_length': self._finite_feature(repo.get('readme_length'), default=1000),
                'star_count': self._finite_feature(repo.get('star_count'), default=0),
                'fork_count': self._finite_feature(repo.get('fork_count'), default=0),
                'open_issues_count': self._finite_feature(repo.get('open_issues_count'), default=0),
                'pushed_days_ago': self._finite_feature(repo.get('pushed_days_ago'), default=365),
                'activity_score': self._finite_feature(repo.get('activity_score'), default=0.0),
                'trend_velocity': self._finite_feature(repo.get('trend_velocity'), default=0.0),
                'skill_match_score': skill_match,
            }
            row = [feature_values[name] for name in FEATURE_ORDER]
            dense_features.append(row)
            
        dense_features = np.asarray(dense_features, dtype=np.float32)
        skill_matches = dense_features[:, 9].copy() if diagnostics else None
        
        # Manually apply StandardScaler math
        dense_features = (dense_features - self.scaler_mean) / self.scaler_scale
        dense_features = np.clip(
            dense_features,
            -STANDARDIZED_FEATURE_CLIP,
            STANDARDIZED_FEATURE_CLIP,
        )
            
        X_tensor = self._input_tensor(user_vector, candidate_repos, dense_features)

        return self._rank_results(X_tensor, candidate_repos, skill_matches)

    def _rank_results(self, X_tensor, candidate_repos, skill_matches=None):
        """Inference-only output path; one stacked device-to-host transfer.

        Float64 arithmetic follows the scalar value function's left-to-right
        operation order, avoiding float32 rounding changes around score ties.
        Python's stable sort retains candidate order for equal final scores.
        """
        check_deadline()
        with torch.inference_mode():
            outputs = self.model(X_tensor)
            if len(outputs) != 5 or any(
                output.numel() != len(candidate_repos) for output in outputs
            ):
                raise ValueError("Heavy ranker produced invalid prediction shapes")
            predictions = torch.stack(
                [output.reshape(-1) for output in outputs], dim=1
            ).cpu().numpy()
            # Preserve legacy range and finite-check ordering (clip, then check).
            np.clip(predictions, 0.0, 1.0, out=predictions)
        if not np.all(np.isfinite(predictions)):
            raise ValueError("Heavy ranker produced non-finite predictions")

        values = predictions.astype(np.float64, copy=False)
        scores = (
            VALUE_WEIGHTS["ctr"] * values[:, 0]
            + VALUE_WEIGHTS["save"] * values[:, 1]
            + VALUE_WEIGHTS["gh_open"] * values[:, 2]
            + VALUE_WEIGHTS["dwell"] * values[:, 3]
            + VALUE_WEIGHTS["follow"] * values[:, 4]
        )
        if not np.all(np.isfinite(scores)):
            raise ValueError("Heavy ranker produced a non-finite value score")
        results = []
        for index, repo in enumerate(candidate_repos):
            result = {"repo_id": repo["id"], "final_score": float(scores[index])}
            if skill_matches is not None:
                result["skill_match"] = float(skill_matches[index])
                result["predictions"] = dict(zip(
                    ("p_ctr", "p_save", "p_gh", "pred_dwell_fraction", "p_follow"),
                    map(float, predictions[index]),
                ))
            results.append(result)
        return sorted(results, key=lambda result: result["final_score"], reverse=True)
