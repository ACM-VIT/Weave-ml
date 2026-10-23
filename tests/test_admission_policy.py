"""Admission observations agree without changing unapproved live gates."""

from types import SimpleNamespace

import pytest

from contracts.admission import (
    ADMISSION_OBSERVATION_FIELD,
    ADMISSION_POLICY_VERSION,
    observe_repository_admission,
)
from embedding.repository_embedding import RepositoryEmbeddingConfig, build_vector_payload
from ingestion.corpus import CorpusStore
from ingestion.pipeline import ingest_repository
from main import filter_enriched


@pytest.mark.parametrize(
    "overrides,candidate,reasons,acquired",
    [
        ({"readme_length": 10}, "review", ["readme_thin"], False),
        ({"pushed_days_ago": 5000}, "pass", [], True),
        ({"star_count": 0}, "pass", [], True),
        ({"readme_length": 5000}, "pass", [], True),
        ({"description": "", "languages": [], "topics": []},
         "review", ["repository_metadata_empty"], False),
    ],
    ids=["thin", "stale", "low-star", "well-documented", "shell"],
)
def test_shared_observation_across_acquisition_compatibility_and_v2_vector_path(
    overrides, candidate, reasons, acquired
):
    repo = {
        "id": "owner/example",
        "full_name": "owner/example",
        "readme_length": 200,
        "description": "An example project",
        "languages": ["Python"],
        "topics": [],
        "star_count": 100,
        "pushed_days_ago": 0,
        "extracted_paragraphs": ["Installation usage API contributing license FAQ"],
        **overrides,
    }
    source = SimpleNamespace(payload=dict(repo))
    kept, dropped = filter_enriched([source])
    assert bool(kept) is acquired
    assert bool(dropped) is not acquired
    compatibility = ingest_repository(repo, corpus_store=CorpusStore(), auto_index=False)
    vector_payload = build_vector_payload(
        repo,
        repo_id="00000000-0000-4000-8000-000000000001",
        final_embedding=[1.0] + [0.0] * 383,
        readme_chunks=1,
        source_hash="source-hash",
        config=RepositoryEmbeddingConfig(),
    )
    observation = source.payload[ADMISSION_OBSERVATION_FIELD]
    assert observation == compatibility.admission_observation
    assert observation == vector_payload[ADMISSION_OBSERVATION_FIELD]
    assert observation["policy_version"] == ADMISSION_POLICY_VERSION
    assert observation["mode"] == "observe_only"
    assert observation["content_candidate"] == candidate
    assert observation["reason_codes"] == reasons
    assert observation["quality_approval"] == "not_evaluated"
    # Existing gate divergence is explicit and remains a pending owner choice.
    if repo["star_count"] == 0:
        assert compatibility.decision == "REJECTED"


def test_raw_readme_overrides_claimed_length_and_forged_quality_approval():
    repo = {
        "full_name": "owner/example", "description": "Example",
        "readme": "short", "readme_length": 99999,
        ADMISSION_OBSERVATION_FIELD: {"quality_approval": "approved"},
    }
    observed = observe_repository_admission(repo)
    assert observed["readme_length"] == 5
    assert observed["reason_codes"] == ["readme_thin"]
    payload = build_vector_payload(
        repo, repo_id="00000000-0000-4000-8000-000000000001",
        final_embedding=[1.0] + [0.0] * 383, readme_chunks=1,
        source_hash="hash", config=RepositoryEmbeddingConfig(),
    )
    assert payload[ADMISSION_OBSERVATION_FIELD] == observed
    assert observed["quality_approval"] == "not_evaluated"


@pytest.mark.parametrize("length", [None, -1, True, "200"])
def test_unknown_readme_evidence_never_looks_like_a_pass(length):
    observation = observe_repository_admission({"description": "Example", "readme_length": length})
    assert observation["content_candidate"] == "review"
    assert observation["reason_codes"] == ["readme_length_unknown"]
    assert observation["readme_length"] is None


def test_existing_acquisition_threshold_does_not_silently_version_the_proposal():
    source = SimpleNamespace(payload={"description": "Example", "readme_length": 200})
    kept, dropped = filter_enriched([source], min_readme_chars=500)
    assert not kept and dropped
    assert source.payload[ADMISSION_OBSERVATION_FIELD]["content_candidate"] == "pass"
    assert source.payload[ADMISSION_OBSERVATION_FIELD]["proposed_min_readme_chars"] == 200


@pytest.mark.parametrize("value", ["Python", 123, [False], [" "]])
def test_malformed_truthy_metadata_cannot_supply_candidate_evidence(value):
    observation = observe_repository_admission({"readme_length": 200, "languages": value, "topics": value})
    assert observation["reason_codes"] == ["repository_metadata_empty"]
    assert observation["quality_approval"] == "not_evaluated"


@pytest.mark.parametrize("readme", [None, "x" * 10, "x" * 200])
def test_real_v2_source_projection_recomputes_from_available_readme(readme):
    from api.contracts import RepositoryJob
    from api.v2 import _repository_embedding_payload

    request = RepositoryJob.model_validate({
        "schema_version": 2,
        "job_id": "00000000-0000-4000-8000-000000000002",
        "repo_id": "00000000-0000-4000-8000-000000000001",
        "content_version": 1,
        "repository": {"full_name": "owner/example", "description": "Example", "readme": readme, "content_hash": "hash"},
    })
    source = _repository_embedding_payload(request)
    payload = build_vector_payload(
        source, repo_id=str(request.repo_id),
        final_embedding=[1.0] + [0.0] * 383, readme_chunks=1,
        source_hash="hash", config=RepositoryEmbeddingConfig(),
    )
    observation = payload[ADMISSION_OBSERVATION_FIELD]
    assert observation == observe_repository_admission(source)
    assert observation["content_candidate"] == ("pass" if readme and len(readme) >= 200 else "review")
    assert observation["quality_approval"] == "not_evaluated"
    if readme is None:
        # V2 defaults absent length to zero; absent README never invents a pass.
        assert observation["reason_codes"] == ["readme_missing"]
