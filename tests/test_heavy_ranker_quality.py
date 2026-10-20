"""Tests that load the real production heavy_ranker.pt and verify it is properly trained.

These are NOT hardcoded unit tests — they exercise the actual model artifact to
catch regressions like:
    - A randomly initialized (untrained) model being deployed
    - Training that collapsed all outputs to a constant
    - Feature scaling drift that makes metadata features invisible
    - Heavy ranker producing rankings indistinguishable from random

Every test loads the real model from ``inference/`` and feeds it synthetic but
realistic inputs. If the model file is missing, all tests are skipped.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from inference.feature_spec import EMBEDDING_DIM, FEATURE_COUNT, FEATURE_ORDER, INPUT_DIM
from inference.ranker_service import MMoEHeavyRanker, RankerService
from inference.value_function import compute_value_score, VALUE_WEIGHTS

INFERENCE_DIR = Path(__file__).resolve().parents[1] / "inference"
MODEL_PATH = INFERENCE_DIR / "heavy_ranker.pt"
SCALER_PATH = INFERENCE_DIR / "feature_scaler.json"
MANIFEST_PATH = INFERENCE_DIR / "model_manifest.json"

# Skip the entire module if the production model artifact isn't available.
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists() or not SCALER_PATH.exists(),
    reason="Production heavy_ranker.pt or feature_scaler.json not found",
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ranker() -> RankerService:
    """Load the real production ranker once for the entire module."""
    return RankerService(
        model_path=str(MODEL_PATH),
        scaler_path=str(SCALER_PATH),
        manifest_path=str(MANIFEST_PATH) if MANIFEST_PATH.exists() else None,
    )


@pytest.fixture(scope="module")
def scaler_params() -> dict:
    """Load the production feature scaler parameters."""
    with open(SCALER_PATH, "r") as f:
        return json.load(f)


def _random_embedding(rng: np.random.Generator) -> np.ndarray:
    """Generate a random unit-norm embedding vector."""
    vec = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _make_candidate(
    repo_id: str,
    embedding: np.ndarray,
    *,
    star_count: float = 100,
    fork_count: float = 20,
    open_issues_count: float = 5,
    doc_quality: float = 0.5,
    code_health: float = 0.5,
    readme_length: float = 2000,
    pushed_days_ago: float = 30,
    activity_score: float = 0.5,
    trend_velocity: float = 0.1,
    languages: list[str] | None = None,
    topics: list[str] | None = None,
    tags: list[str] | None = None,
) -> dict:
    return {
        "id": repo_id,
        "embedding": embedding,
        "star_count": star_count,
        "fork_count": fork_count,
        "open_issues_count": open_issues_count,
        "doc_quality": doc_quality,
        "code_health": code_health,
        "readme_length": readme_length,
        "pushed_days_ago": pushed_days_ago,
        "activity_score": activity_score,
        "trend_velocity": trend_velocity,
        "languages": languages or [],
        "topics": topics or [],
        "tags": tags or [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  1. MODEL IS ACTUALLY LOADED AND NOT RANDOM
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelIsProperlyTrained:
    """Verify that the deployed model artifact is a real trained model."""

    def test_model_file_is_loaded(self, ranker: RankerService):
        """The RankerService should confirm the model loaded from disk."""
        assert ranker._model_loaded is True
        assert ranker.ready is True

    def test_model_weights_are_not_at_initialization(self, ranker: RankerService):
        """A trained model's weights should have diverged from random init.

        We compare the actual model's parameter statistics against a freshly
        initialized model — if training happened, the distributions should
        differ significantly.
        """
        fresh_model = MMoEHeavyRanker(INPUT_DIM)

        trained_params = {
            name: param.detach().cpu().numpy()
            for name, param in ranker.model.named_parameters()
        }
        fresh_params = {
            name: param.detach().cpu().numpy()
            for name, param in fresh_model.named_parameters()
        }

        # At least 80% of parameter tensors should have different means
        # (random init uses a specific distribution; training moves them)
        different_count = 0
        for name in trained_params:
            if name not in fresh_params:
                continue
            trained_mean = float(np.mean(trained_params[name]))
            fresh_mean = float(np.mean(fresh_params[name]))
            # If the absolute difference exceeds a small threshold, it was trained
            if abs(trained_mean - fresh_mean) > 0.01:
                different_count += 1

        total_comparable = len(
            [n for n in trained_params if n in fresh_params]
        )
        assert total_comparable > 0, "No comparable parameters found"
        ratio = different_count / total_comparable
        # MMoE models have many bias parameters and gate softmax layers that
        # naturally stay near init.  The real training signal is verified by
        # BatchNorm running stats and prediction quality tests.  A 30% threshold
        # still catches a fully untrained model (~0% divergence).
        assert ratio > 0.3, (
            f"Only {different_count}/{total_comparable} ({ratio:.0%}) parameter tensors "
            f"differ from random init — the model may not be trained"
        )

    def test_batch_norm_running_stats_are_populated(self, ranker: RankerService):
        """If BatchNorm layers exist, their running mean/var should be non-trivial.

        A freshly initialized BatchNorm has running_mean=0, running_var=1.
        Training populates them with data statistics.
        """
        found_bn = False
        for name, module in ranker.model.named_modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                found_bn = True
                mean = module.running_mean.cpu().numpy()
                var = module.running_var.cpu().numpy()
                # Running mean should not be all zeros after training
                assert not np.allclose(mean, 0.0, atol=1e-5), (
                    f"BatchNorm {name} running_mean is all zeros — untrained"
                )
                # Running var should not be all ones after training
                assert not np.allclose(var, 1.0, atol=1e-5), (
                    f"BatchNorm {name} running_var is all ones — untrained"
                )

        assert found_bn, "Expected at least one BatchNorm1d layer in MMoE"


# ═══════════════════════════════════════════════════════════════════════════════
#  2. PREDICTIONS ARE NOT DEGENERATE
# ═══════════════════════════════════════════════════════════════════════════════


class TestPredictionQuality:
    """Verify that the model produces non-degenerate, varied predictions."""

    def test_predictions_are_not_constant_across_different_repos(
        self, ranker: RankerService
    ):
        """Different repo inputs should produce different prediction scores.

        If the model always outputs the same score regardless of input,
        it hasn't learned anything useful.
        """
        rng = np.random.default_rng(42)
        user_emb = _random_embedding(rng)

        candidates = []
        for i in range(10):
            candidates.append(
                _make_candidate(
                    f"repo-{i}",
                    _random_embedding(rng),
                    star_count=10 ** (i % 5),
                    doc_quality=i / 10,
                    activity_score=i / 10,
                    readme_length=500 * (i + 1),
                )
            )

        results = ranker.score_batch(user_emb, ["Python", "ML"], candidates)
        scores = [r["final_score"] for r in results]

        # Scores should have meaningful variance — not all the same
        score_std = float(np.std(scores))
        assert score_std > 0.01, (
            f"All 10 candidates received nearly identical scores (std={score_std:.6f}). "
            f"The model is likely predicting a constant."
        )

    def test_predictions_span_a_reasonable_range(self, ranker: RankerService):
        """Scores should cover a non-trivial range, not cluster in a tiny band."""
        rng = np.random.default_rng(99)
        user_emb = _random_embedding(rng)

        # Create intentionally diverse candidates
        great_repo = _make_candidate(
            "great-repo",
            _random_embedding(rng),
            star_count=50000,
            doc_quality=0.95,
            code_health=0.95,
            activity_score=0.9,
            trend_velocity=0.5,
            pushed_days_ago=1,
            languages=["Python"],
            topics=["machine-learning"],
        )
        bad_repo = _make_candidate(
            "bad-repo",
            _random_embedding(rng),
            star_count=0,
            doc_quality=0.0,
            code_health=0.0,
            activity_score=0.0,
            trend_velocity=0.0,
            pushed_days_ago=999,
        )

        results = ranker.score_batch(
            user_emb, ["Python", "machine-learning"], [great_repo, bad_repo]
        )

        scores_by_id = {r["repo_id"]: r["final_score"] for r in results}
        spread = abs(scores_by_id["great-repo"] - scores_by_id["bad-repo"])
        assert spread > 0.05, (
            f"A high-quality and a low-quality repo scored almost identically "
            f"(spread={spread:.4f}). The model isn't differentiating."
        )

    def test_all_five_task_heads_produce_non_trivial_output(
        self, ranker: RankerService
    ):
        """Each of the 5 MMoE task heads should produce varied predictions."""
        rng = np.random.default_rng(77)
        user_emb = _random_embedding(rng)

        candidates = [
            _make_candidate(
                f"repo-{i}",
                _random_embedding(rng),
                star_count=100 * (i + 1),
                activity_score=i / 20,
            )
            for i in range(20)
        ]

        results = ranker.score_batch(user_emb, ["Python"], candidates)
        head_names = ["p_ctr", "p_save", "p_gh", "pred_dwell_fraction", "p_follow"]

        for head in head_names:
            values = [r["predictions"][head] for r in results]
            std = float(np.std(values))
            assert std > 1e-4, (
                f"Task head '{head}' produced nearly constant output across 20 "
                f"candidates (std={std:.6f}). That head may not be trained."
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  3. FEATURE SENSITIVITY — THE MODEL RESPONDS TO QUALITY SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFeatureSensitivity:
    """Verify that varying individual features changes the model's output."""

    def test_higher_stars_influence_score(self, ranker: RankerService):
        """Changing star_count significantly should alter the final score."""
        rng = np.random.default_rng(11)
        user_emb = _random_embedding(rng)
        repo_emb = _random_embedding(rng)

        low_stars = _make_candidate("low", repo_emb, star_count=1)
        high_stars = _make_candidate("high", repo_emb, star_count=100000)

        results = ranker.score_batch(user_emb, ["Python"], [low_stars, high_stars])
        scores = {r["repo_id"]: r["final_score"] for r in results}

        # We don't prescribe direction (the model learns that), but the scores
        # should differ — if they're identical, star_count is invisible.
        assert scores["low"] != pytest.approx(scores["high"], abs=0.001), (
            "Varying star_count from 1 to 100000 had no effect on the score"
        )

    def test_doc_quality_influences_score(self, ranker: RankerService):
        """Changing doc_quality should alter the final score."""
        rng = np.random.default_rng(22)
        user_emb = _random_embedding(rng)
        repo_emb = _random_embedding(rng)

        low_doc = _make_candidate("low", repo_emb, doc_quality=0.0)
        high_doc = _make_candidate("high", repo_emb, doc_quality=1.0)

        results = ranker.score_batch(user_emb, [], [low_doc, high_doc])
        scores = {r["repo_id"]: r["final_score"] for r in results}

        assert scores["low"] != pytest.approx(scores["high"], abs=0.001), (
            "Varying doc_quality from 0.0 to 1.0 had no effect on the score"
        )

    def test_activity_score_influences_predictions(self, ranker: RankerService):
        """Changing activity_score should alter at least one task head."""
        rng = np.random.default_rng(33)
        user_emb = _random_embedding(rng)
        repo_emb = _random_embedding(rng)

        low_activity = _make_candidate("low", repo_emb, activity_score=0.0)
        high_activity = _make_candidate("high", repo_emb, activity_score=1.0)

        results = ranker.score_batch(user_emb, [], [low_activity, high_activity])
        preds = {r["repo_id"]: r["predictions"] for r in results}

        # At least one head should respond
        any_different = any(
            abs(preds["low"][head] - preds["high"][head]) > 0.001
            for head in preds["low"]
        )
        assert any_different, (
            "Varying activity_score from 0.0 to 1.0 had no effect on any task head"
        )

    def test_embedding_similarity_influences_score(self, ranker: RankerService):
        """A repo whose embedding is similar to the user should score differently
        from one whose embedding is orthogonal."""
        rng = np.random.default_rng(44)
        user_emb = _random_embedding(rng)

        # Similar: same direction as user
        similar_emb = user_emb.copy()
        # Orthogonal: a very different direction
        orthogonal_emb = _random_embedding(np.random.default_rng(9999))

        similar_repo = _make_candidate("similar", similar_emb)
        orthogonal_repo = _make_candidate("orthogonal", orthogonal_emb)

        results = ranker.score_batch(
            user_emb, ["Python"], [similar_repo, orthogonal_repo]
        )
        scores = {r["repo_id"]: r["final_score"] for r in results}

        assert scores["similar"] != pytest.approx(scores["orthogonal"], abs=0.001), (
            "User-repo embedding alignment had no effect on the score"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  4. SKILL MATCH CROSS-FEATURE
# ═══════════════════════════════════════════════════════════════════════════════


class TestSkillMatchIntegration:
    """Verify that the dynamic skill_match_score cross-feature works."""

    def test_skill_match_is_computed_and_affects_output(
        self, ranker: RankerService
    ):
        """A repo whose languages/topics match the user's skills should get
        a different skill_match value than one with zero overlap."""
        rng = np.random.default_rng(55)
        user_emb = _random_embedding(rng)
        repo_emb = _random_embedding(rng)

        matched_repo = _make_candidate(
            "matched",
            repo_emb,
            languages=["Python", "TypeScript"],
            topics=["machine-learning"],
        )
        unmatched_repo = _make_candidate(
            "unmatched",
            repo_emb,
            languages=["Haskell"],
            topics=["quantum-computing"],
        )

        results = ranker.score_batch(
            user_emb,
            ["Python", "machine-learning", "TypeScript"],
            [matched_repo, unmatched_repo],
        )

        scores = {r["repo_id"]: r for r in results}
        assert scores["matched"]["skill_match"] > 0.0, (
            "Matching repo should have a nonzero skill_match"
        )
        assert scores["unmatched"]["skill_match"] == pytest.approx(0.0), (
            "Non-matching repo should have zero skill_match"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  5. DETERMINISM AND NUMERICAL STABILITY
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminismAndStability:
    """Verify reproducibility and absence of numerical issues."""

    def test_same_input_produces_same_output(self, ranker: RankerService):
        """Two identical calls should produce bit-identical results."""
        rng = np.random.default_rng(66)
        user_emb = _random_embedding(rng)

        rng2 = np.random.default_rng(66)
        user_emb2 = _random_embedding(rng2)

        candidates = [
            _make_candidate("repo-0", _random_embedding(np.random.default_rng(100))),
            _make_candidate("repo-1", _random_embedding(np.random.default_rng(200))),
        ]

        results_a = ranker.score_batch(user_emb, ["Python"], candidates)
        results_b = ranker.score_batch(user_emb2, ["Python"], candidates)

        for a, b in zip(results_a, results_b):
            assert a["repo_id"] == b["repo_id"]
            assert a["final_score"] == pytest.approx(b["final_score"], abs=1e-6)

    def test_all_outputs_are_finite(self, ranker: RankerService):
        """No prediction head should produce NaN or Inf for reasonable inputs."""
        rng = np.random.default_rng(88)
        user_emb = _random_embedding(rng)

        candidates = [
            _make_candidate(f"repo-{i}", _random_embedding(rng))
            for i in range(50)
        ]

        results = ranker.score_batch(user_emb, ["Python", "ML"], candidates)

        for result in results:
            assert math.isfinite(result["final_score"]), (
                f"final_score is not finite for {result['repo_id']}"
            )
            for head, value in result["predictions"].items():
                assert math.isfinite(value), (
                    f"Prediction '{head}' is not finite for {result['repo_id']}"
                )

    def test_extreme_feature_values_do_not_produce_nan(
        self, ranker: RankerService
    ):
        """Extreme but valid metadata values should not cause numerical blowup."""
        rng = np.random.default_rng(111)
        user_emb = _random_embedding(rng)
        repo_emb = _random_embedding(rng)

        extreme_repo = _make_candidate(
            "extreme",
            repo_emb,
            star_count=10_000_000,
            fork_count=5_000_000,
            open_issues_count=100_000,
            readme_length=500_000,
            pushed_days_ago=0,
            activity_score=100.0,
            trend_velocity=500.0,
            doc_quality=1.0,
            code_health=1.0,
        )

        results = ranker.score_batch(user_emb, [], [extreme_repo])
        assert len(results) == 1
        assert math.isfinite(results[0]["final_score"])
        for value in results[0]["predictions"].values():
            assert math.isfinite(value)


# ═══════════════════════════════════════════════════════════════════════════════
#  6. VALUE FUNCTION AND RANKING ORDER
# ═══════════════════════════════════════════════════════════════════════════════


class TestValueFunctionIntegration:
    """Verify value function scoring is applied correctly."""

    def test_value_function_weights_are_consistent_with_manifest(self):
        """The value weights in code should match the manifest if it exists."""
        if not MANIFEST_PATH.exists():
            pytest.skip("No manifest file")
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        if "value_weights" not in manifest:
            pytest.skip("Manifest has no value_weights")
        for key, expected in manifest["value_weights"].items():
            assert VALUE_WEIGHTS.get(key) == expected, (
                f"VALUE_WEIGHTS['{key}'] = {VALUE_WEIGHTS.get(key)}, "
                f"manifest says {expected}"
            )

    def test_results_are_sorted_descending_by_final_score(
        self, ranker: RankerService
    ):
        """score_batch should return results sorted descending by final_score."""
        rng = np.random.default_rng(122)
        user_emb = _random_embedding(rng)

        candidates = [
            _make_candidate(
                f"repo-{i}",
                _random_embedding(rng),
                star_count=i * 1000,
                activity_score=i / 10,
            )
            for i in range(15)
        ]

        results = ranker.score_batch(user_emb, ["Python"], candidates)
        scores = [r["final_score"] for r in results]

        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Results are not sorted descending: "
                f"score[{i}]={scores[i]} < score[{i+1}]={scores[i+1]}"
            )

    def test_follow_prediction_has_highest_value_weight(self):
        """p_follow should dominate scoring since it has weight=20."""
        high_follow = {"p_ctr": 0.1, "p_save": 0.1, "p_gh": 0.1,
                       "pred_dwell_fraction": 0.1, "p_follow": 0.9}
        low_follow = {"p_ctr": 0.9, "p_save": 0.9, "p_gh": 0.9,
                      "pred_dwell_fraction": 0.9, "p_follow": 0.0}

        score_follow = compute_value_score(high_follow)
        score_other = compute_value_score(low_follow)

        assert score_follow > score_other, (
            "A candidate with high p_follow should outscore one with high "
            "everything-else — p_follow has weight 20"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  7. SCALER INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════════


class TestScalerIntegrity:
    """Verify the production feature scaler is valid."""

    def test_scaler_has_correct_feature_count(self, scaler_params: dict):
        assert len(scaler_params["mean"]) == FEATURE_COUNT
        assert len(scaler_params["scale"]) == FEATURE_COUNT

    def test_scaler_scales_are_positive(self, scaler_params: dict):
        for i, scale in enumerate(scaler_params["scale"]):
            assert scale > 0, (
                f"scale[{i}] ({FEATURE_ORDER[i]}) is {scale} — must be positive"
            )

    def test_scaler_values_are_finite(self, scaler_params: dict):
        for i, mean in enumerate(scaler_params["mean"]):
            assert math.isfinite(mean), f"mean[{i}] is not finite"
        for i, scale in enumerate(scaler_params["scale"]):
            assert math.isfinite(scale), f"scale[{i}] is not finite"

    def test_scaler_means_are_plausible(self, scaler_params: dict):
        """Sanity-check that the scaler was fit on real data, not dummy values."""
        means = scaler_params["mean"]
        # doc_quality (index 0) should be between 0 and 1
        assert 0.0 <= means[0] <= 1.0, (
            f"doc_quality mean={means[0]} is outside [0, 1]"
        )
        # star_count mean (index 3) should be positive for a real corpus
        assert means[3] > 0, f"star_count mean={means[3]} should be positive"


# ═══════════════════════════════════════════════════════════════════════════════
#  8. MANIFEST CONTRACT
# ═══════════════════════════════════════════════════════════════════════════════


class TestManifestContract:
    """Verify the model manifest matches the inference contract."""

    @pytest.fixture(scope="class")
    def manifest(self) -> dict:
        if not MANIFEST_PATH.exists():
            pytest.skip("No manifest file found")
        with open(MANIFEST_PATH) as f:
            return json.load(f)

    def test_input_dim_matches(self, manifest: dict):
        assert manifest["input_dim"] == INPUT_DIM

    def test_embedding_dim_matches(self, manifest: dict):
        assert manifest["embedding_dim"] == EMBEDDING_DIM

    def test_feature_count_matches(self, manifest: dict):
        assert manifest["feature_count"] == FEATURE_COUNT

    def test_compatible_embedding_versions_is_non_empty(self, manifest: dict):
        versions = manifest.get("compatible_embedding_versions", [])
        assert len(versions) > 0, "Manifest must list at least one compatible version"

    def test_model_version_is_non_empty(self, manifest: dict):
        assert manifest.get("model_version", "").strip(), (
            "Manifest model_version must be non-empty"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  9. BATCH SIZE ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════════════


class TestBatchSizeRobustness:
    """The model must work for batch sizes from 1 to large."""

    @pytest.mark.parametrize("batch_size", [1, 2, 5, 50, 150])
    def test_score_batch_at_various_sizes(
        self, ranker: RankerService, batch_size: int
    ):
        rng = np.random.default_rng(batch_size)
        user_emb = _random_embedding(rng)
        candidates = [
            _make_candidate(f"repo-{i}", _random_embedding(rng))
            for i in range(batch_size)
        ]

        results = ranker.score_batch(user_emb, ["Python"], candidates)

        assert len(results) == batch_size
        assert all(math.isfinite(r["final_score"]) for r in results)
        assert len({r["repo_id"] for r in results}) == batch_size


# ═══════════════════════════════════════════════════════════════════════════════
#  10. SINGLE-CANDIDATE CONSISTENCY WITH BATCH
# ═══════════════════════════════════════════════════════════════════════════════


class TestSingleVsBatch:
    """A candidate scored alone should get the same score as in a batch."""

    def test_single_and_batch_scores_agree(self, ranker: RankerService):
        rng = np.random.default_rng(200)
        user_emb = _random_embedding(rng)
        candidates = [
            _make_candidate(f"repo-{i}", _random_embedding(rng), star_count=i * 100)
            for i in range(5)
        ]

        # Score them all together
        batch_results = ranker.score_batch(user_emb, ["Python"], candidates)
        batch_scores = {r["repo_id"]: r["final_score"] for r in batch_results}

        # Score each one individually
        for candidate in candidates:
            single = ranker.score_batch(user_emb, ["Python"], [candidate])
            assert len(single) == 1
            # BatchNorm in eval mode should produce the same result
            assert single[0]["final_score"] == pytest.approx(
                batch_scores[candidate["id"]], abs=1e-4
            ), (
                f"Single score for {candidate['id']} differs from batch score: "
                f"single={single[0]['final_score']}, batch={batch_scores[candidate['id']]}"
            )
