"""Admission observations agree without changing unapproved live gates."""

from types import SimpleNamespace

import pytest

from contracts.admission import (
    ADMISSION_OBSERVATION_FIELD,
    ADMISSION_POLICY_VERSION,
    observe_repository_admission,
)
from embedding.repository_embedding import RepositoryEmbeddingConfig, build_vector_payload
from ingestion.corpus import CorpusStore, dynamic_cluster_discovery
from ingestion.pipeline import ingest_repository, print_batch_summary
from main import filter_enriched


def _approved_compatibility_repo(repo_id="owner/example"):
    return {
        "id": repo_id,
        "full_name": repo_id,
        "readme_length": 3000,
        "readme_to_codebase_ratio": 0.01,
        "description": "An example project",
        "languages": ["Python"],
        "topics": ["api"],
        "star_count": 100,
        "pushed_days_ago": 0,
        "open_issues_count": 0,
        "fork_count": 1,
        "extracted_paragraphs": [
            "Installation, usage examples, API docs, contributing, license, and FAQ."
        ],
    }


def _topology_node(category, vector):
    return {
        "category": category,
        "vector": vector,
        "embedding_model": "test-model",
        "embedding_model_revision": "test-revision",
        "embedding_version": "test-embedding-v1",
        "feature_spec_version": "test-features-v1",
    }


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
    assert compatibility.novelty.measurement_status == "unmeasured"
    assert (
        compatibility.novelty.final,
        compatibility.novelty.semantic_penalty,
        compatibility.novelty.category_penalty,
        compatibility.novelty.tech_stack_penalty,
        compatibility.novelty.activity_penalty,
    ) == (None, None, None, None, None)
    rendered = str(compatibility)
    assert "Corpus Novelty Measurement: UNMEASURED" in rendered
    assert "Aggregate System Novelty Target Index" not in rendered
    assert "SECURITY/INTEGRITY ANOMALY DETECTED" not in rendered
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


def test_compatibility_batch_summary_preserves_unmeasured_values(capsys):
    store = CorpusStore()
    approved = ingest_repository(_approved_compatibility_repo(), corpus_store=store)
    rejected_repo = _approved_compatibility_repo("owner/rejected")
    rejected_repo["star_count"] = 0
    rejected = ingest_repository(rejected_repo, corpus_store=store)

    assert approved.decision == "APPROVED"
    assert rejected.decision == "REJECTED"
    assert store.get_timeline() == [{
        "growth_index": 1,
        "repo_id": "owner/example",
        "novelty_index_point": None,
    }]

    print_batch_summary([approved, rejected], store.get_timeline())
    output = capsys.readouterr().out
    assert "Corpus Novelty Measurements         : UNAVAILABLE" in output
    assert output.count("No measured novelty values are available.") == 2
    assert "Registered Novelty Tracking: UNMEASURED" in output
    assert "owner/rejected" in output
    assert "Matrix Evaluation Value: UNMEASURED" in output
    assert "Total Flagged System Anomalies    : 0" in output
    assert "Novelty Vector Index:" not in output


def test_compatibility_placeholders_cannot_produce_topology():
    store = CorpusStore()
    for index in range(4):
        result = ingest_repository(
            _approved_compatibility_repo(f"owner/example-{index}"),
            corpus_store=store,
        )
        assert result.decision == "APPROVED"

    result = ingest_repository(
        _approved_compatibility_repo("owner/topology-check"),
        corpus_store=store,
        auto_index=False,
        compute_topology=True,
    )
    assert result.topological_metrics == {
        "current_cluster": "Unavailable",
        "nearest_cluster": "Unavailable",
        "distance": None,
        "measurement_status": "unmeasured",
        "reason": "Fewer than four valid non-zero repository vectors are available.",
    }
    assert "Corpus Cluster Topology Structural Space Vector" not in str(result)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (
            lambda nodes: nodes[-1].update(embedding_model_revision="other-revision"),
            "Repository vectors use incompatible model or feature versions.",
        ),
        (
            lambda nodes: nodes[-1].update(vector=[0.0, 1.0, 0.0]),
            "Repository vectors use incompatible dimensions.",
        ),
        (
            lambda nodes: nodes[-1].pop("embedding_model_revision"),
            "Repository vector model and version provenance is incomplete.",
        ),
        (
            lambda nodes: nodes[-1].update(vector=[float("nan"), 1.0]),
            "Fewer than four valid non-zero repository vectors are available.",
        ),
    ],
    ids=["mixed-provenance", "mixed-dimensions", "missing-provenance", "non-finite"],
)
def test_topology_rejects_incompatible_vector_contracts(mutation, reason):
    nodes = [
        _topology_node("alpha", [1.0, 0.0]),
        _topology_node("alpha", [2.0, 0.0]),
        _topology_node("beta", [0.0, 1.0]),
        _topology_node("beta", [0.0, 2.0]),
    ]
    mutation(nodes)

    topology = dynamic_cluster_discovery(nodes)
    assert topology["measurement_status"] == "unmeasured"
    assert topology["distance"] is None
    assert topology["reason"] == reason


def test_topology_measures_compatible_synthetic_centroids():
    topology = dynamic_cluster_discovery([
        _topology_node("alpha", [1.0, 0.0]),
        _topology_node("alpha", [2.0, 0.0]),
        _topology_node("beta", [0.0, 1.0]),
        _topology_node("beta", [0.0, 2.0]),
    ])

    assert topology == {
        "current_cluster": "alpha",
        "nearest_cluster": "beta",
        "distance": 1.0,
        "measurement_status": "measured",
    }
