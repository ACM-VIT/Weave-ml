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
from inference.ranker_service import (
    MMoEHeavyRanker,
    RankerService,
    _cached_repository_keywords,
    _repository_keywords,
    calculate_match_score,
)
from inference.value_function import VALUE_WEIGHTS


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
