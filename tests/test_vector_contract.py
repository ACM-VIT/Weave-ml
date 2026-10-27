"""Phase 1 tests for the vector-platform contract."""

import pytest

from config import (
    QDRANT_COLLECTION_NAME,
    QDRANT_DISTANCE,
    QDRANT_VECTOR_NAME,
    REPOSITORY_EMBEDDING_DIM,
    REPOSITORY_EMBEDDING_MODEL,
    USER_PROFILES_COLLECTION_NAME,
)
from embedding.qdrant_store import QdrantRepositoryStore
from embedding.repository_embedding import RepositoryEmbeddingConfig, build_vector_payload
from embedding.vector_contract import (
    REPOSITORY_COLLECTION_CONTRACT,
    REPOSITORY_PAYLOAD_FIELD_TYPES,
    REPOSITORY_PAYLOAD_REQUIRED_FIELDS,
    REPOSITORY_SERVING_ELIGIBILITY_FIELD,
    REPOSITORY_SERVING_ELIGIBILITY_VERSION,
    USER_PROFILE_COLLECTION_CONTRACT,
    canonical_backend_uuid,
    repository_payload_defaults,
    repository_point_id,
    user_point_id,
)

REPO_ID = "00000000-0000-4000-8000-000000000001"
OTHER_REPO_ID = "00000000-0000-4000-8000-000000000002"


def test_offline_lookahead_snapshots_mutable_topics():
    import hashlib
    from embedding.embedding_pipeline import RepositoryEmbeddingPipeline

    class Encoder:
        def embed_texts(self, texts):
            vectors = []
            for text in texts:
                vector = [0.0] * REPOSITORY_EMBEDDING_DIM
                vector[int.from_bytes(hashlib.sha256(text.encode()).digest()[:2]) % len(vector)] = 1.0
                vectors.append(vector)
            return vectors

    pipeline = RepositoryEmbeddingPipeline(embedder=Encoder())
    sources = [
        {"repo_id": REPO_ID, "full_name": "owner/first", "topics": ["search"]},
        {"repo_id": OTHER_REPO_ID, "full_name": "owner/second", "topics": ["storage"]},
    ]
    expected = [pipeline.embed_repository(source) for source in sources]
    shared_topics = []

    def reused_rows():
        for source in sources:
            shared_topics[:] = source["topics"]
            yield {**source, "topics": shared_topics}

    actual = list(pipeline.iter_embed(reused_rows()))
    assert [result.payload["topics"] for result in actual] == [["search"], ["storage"]]
    assert [result.source_hash for result in actual] == [result.source_hash for result in expected]
    assert [result.final_embedding for result in actual] == [result.final_embedding for result in expected]


@pytest.mark.parametrize(
    "value", [b"x" * 1_100_000, bytearray(b"large"), object(), 1 << 10000],
    ids=["large-bytes", "bytearray", "object", "large-integer"],
)
def test_offline_budget_rejects_unbounded_values_before_preparing_text(value, monkeypatch):
    from embedding.embedding_pipeline import RepositoryEmbeddingPipeline

    pipeline = RepositoryEmbeddingPipeline()
    monkeypatch.setattr(pipeline, "_prepare_repository", lambda *args, **kwargs: pytest.fail("prepared rejected source"))
    with pytest.raises((TypeError, ValueError), match="scalar"):
        next(pipeline.iter_embed([{
            "repo_id": REPO_ID, "full_name": "owner/repo", "extracted_paragraphs": [value],
        }]))


def test_cancelled_offline_job_does_not_advance_lazy_source():
    from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
    from utils.deadline import CooperativeDeadlineExceeded, RequestDeadline, deadline_scope

    fetched = []

    def sources():
        fetched.append(True)
        yield {"repo_id": REPO_ID, "full_name": "owner/repo"}

    deadline = RequestDeadline(30)
    deadline.cancel()
    with deadline_scope(deadline), pytest.raises(CooperativeDeadlineExceeded):
        next(RepositoryEmbeddingPipeline().iter_embed(sources()))
    assert fetched == []


def test_repository_collection_contract_matches_central_config():
    assert REPOSITORY_COLLECTION_CONTRACT.collection_name == QDRANT_COLLECTION_NAME
    assert REPOSITORY_COLLECTION_CONTRACT.vector_name == QDRANT_VECTOR_NAME
    assert REPOSITORY_COLLECTION_CONTRACT.vector_size == REPOSITORY_EMBEDDING_DIM == 384
    assert REPOSITORY_COLLECTION_CONTRACT.distance == QDRANT_DISTANCE == "Cosine"
    assert REPOSITORY_COLLECTION_CONTRACT.model_name == REPOSITORY_EMBEDDING_MODEL


def test_user_collection_contract_preserves_existing_unnamed_vector():
    assert USER_PROFILE_COLLECTION_CONTRACT.collection_name == USER_PROFILES_COLLECTION_NAME
    assert USER_PROFILE_COLLECTION_CONTRACT.vector_name is None
    assert USER_PROFILE_COLLECTION_CONTRACT.vector_size == REPOSITORY_EMBEDDING_DIM
    assert USER_PROFILE_COLLECTION_CONTRACT.distance == QDRANT_DISTANCE
    assert USER_PROFILE_COLLECTION_CONTRACT.model_name == REPOSITORY_EMBEDDING_MODEL


def test_point_ids_are_canonical_backend_uuids():
    assert repository_point_id(REPO_ID) == repository_point_id(f" {REPO_ID} ")
    assert user_point_id(REPO_ID) == user_point_id(f" {REPO_ID} ")
    assert repository_point_id(REPO_ID) == user_point_id(REPO_ID) == REPO_ID
    assert repository_point_id(REPO_ID) != repository_point_id(OTHER_REPO_ID)


def test_backend_ids_are_canonical_valid_uuids():
    uppercase_uuid = "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"
    assert canonical_backend_uuid(uppercase_uuid, field_name="repo_id") == uppercase_uuid.lower()
    with pytest.raises(ValueError, match="backend-issued UUID"):
        canonical_backend_uuid("owner/repository", field_name="repo_id")


@pytest.mark.parametrize("value", ["", "   ", "repo-123", "owner/repository"])
def test_point_ids_reject_empty_identifiers(value):
    with pytest.raises(ValueError):
        repository_point_id(value)
    with pytest.raises(ValueError):
        user_point_id(value)


def test_point_ids_reject_non_string_identifiers():
    with pytest.raises(TypeError):
        repository_point_id(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        user_point_id(None)  # type: ignore[arg-type]


def test_store_uses_the_published_repository_point_id_helper():
    assert QdrantRepositoryStore._point_id(REPO_ID) == repository_point_id(REPO_ID)


def test_repository_payload_contract_has_identity_and_embedding_fields():
    assert tuple(REPOSITORY_PAYLOAD_FIELD_TYPES) == REPOSITORY_PAYLOAD_REQUIRED_FIELDS
    assert {"repo_id", "full_name"}.issubset(REPOSITORY_PAYLOAD_REQUIRED_FIELDS)
    assert {
        "github_id",
        "content_version",
        "content_hash",
        "embedding_dim",
        "embedding_model",
        "embedding_version",
        "model_version",
        "indexed_at",
        "source_hash",
        REPOSITORY_SERVING_ELIGIBILITY_FIELD,
    }.issubset(REPOSITORY_PAYLOAD_REQUIRED_FIELDS)


def test_repository_payload_defaults_are_fresh_and_match_current_model():
    first = repository_payload_defaults()
    second = repository_payload_defaults()

    first["languages"].append("Python")  # type: ignore[union-attr]

    assert second["languages"] == []
    assert second["embedding_dim"] == 384
    assert second["embedding_model"] == REPOSITORY_EMBEDDING_MODEL
    assert (
        second[REPOSITORY_SERVING_ELIGIBILITY_FIELD]
        == REPOSITORY_SERVING_ELIGIBILITY_VERSION
    )


def test_current_payload_builder_publishes_the_frozen_contract():
    payload = build_vector_payload(
        {
            "full_name": "owner/repository",
            "description": "Example repository",
            "primary_language": "Python",
            "languages": ["Python"],
            "topics": ["machine-learning"],
            "extracted_paragraphs": ["A documented project."],
        },
        repo_id=REPO_ID,
        final_embedding=[0.0] * 384,
        readme_chunks=1,
        source_hash="source-hash",
        config=RepositoryEmbeddingConfig(),
    )

    # Summary provenance and admission observations are optional internal metadata; existing vectors keep
    # the frozen required serving contract and need no reindex.
    assert set(payload) == (set(REPOSITORY_PAYLOAD_REQUIRED_FIELDS) - {
        REPOSITORY_SERVING_ELIGIBILITY_FIELD
    }) | {"card_summary_reuse_key", "admission_observation"}
    assert payload["card_summary_reuse_key"] is None
    assert payload["admission_observation"]["quality_approval"] == "not_evaluated"
    for field_name, expected_type in REPOSITORY_PAYLOAD_FIELD_TYPES.items():
        if field_name == REPOSITORY_SERVING_ELIGIBILITY_FIELD:
            continue
        assert isinstance(payload[field_name], expected_type), field_name


def test_language_numeric_metadata_does_not_change_semantic_towers_or_source_hash():
    import hashlib
    from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
    class Encoder:
        def embed_texts(self, texts):
            result = []
            for text in texts:
                vector = [0.] * 384
                vector[hashlib.sha256(text.encode()).digest()[0]] = 1.
                result.append(vector)
            return result
    pipeline = RepositoryEmbeddingPipeline(embedder=Encoder())
    source = dict(repo_id=REPO_ID, full_name="owner/repo", languages=["Python"], primary_language="Python",
        readme="A search project.", topics=["search"], language_total_size=1000,
        language_breakdown_complete=False,
        language_breakdown=[dict(name="Python", color="#3572A5", size=300, percentage=30.)])
    first = pipeline.embed_repository(source)
    changed = {**source, "language_breakdown": [dict(name="Python", color="#FFFFFF", size=500, percentage=50.)]}
    second = pipeline.embed_repository(changed)
    assert first.payload["language_breakdown"] == source["language_breakdown"]
    assert second.payload["language_breakdown"] == changed["language_breakdown"]
    assert first.source_hash == second.source_hash
    for tower in ("readme_embedding", "metadata_embedding", "topic_embedding", "final_embedding"):
        assert getattr(first, tower) == getattr(second, tower)
    # Old provider model dumps include optional nulls; those are unknown, not new evidence.
    from contracts.repository_languages import language_observation
    assert language_observation(dict(language_breakdown=None, language_total_size=None,
                                    language_breakdown_complete=None)) == {}


def test_language_percentage_rejects_contradictory_observed_bytes():
    from contracts.repository_languages import language_observation
    with pytest.raises(ValueError, match="percentage.*byte"):
        language_observation(dict(language_total_size=1000,
            language_breakdown=[dict(name="Python", size=300, percentage=99.)]))


def test_language_percentage_accepts_rounded_rows_without_inventing_unknowns():
    from contracts.repository_languages import language_observation
    rows = [dict(name=f"language{number}", size=1, percentage=16.67) for number in range(6)]
    observed = language_observation(dict(language_total_size=6, language_breakdown_complete=True,
                                        language_breakdown=rows))
    assert observed["language_breakdown"] == rows
    assert observed["language_breakdown_complete"] is True
    for total, row in [(3, dict(name="Python", size=1, percentage=33.33)),
                       (3, dict(name="Python", size=1, percentage=None)),
                       (None, dict(name="Python", size=1, percentage=None)),
                       (0, dict(name="Python", size=0, percentage=None))]:
        result = language_observation(dict(language_total_size=total, language_breakdown=[row]))
        assert result["language_breakdown"] == [row]


def test_unknown_language_percentages_do_not_expand_rounding_allowance():
    from contracts.repository_languages import language_observation
    unknown = [dict(name=f"unknown{number}", percentage=None) for number in range(98)]
    with pytest.raises(ValueError, match="percentages exceed"):
        language_observation(dict(language_breakdown=unknown + [
            dict(name="Python", percentage=50.4), dict(name="Rust", percentage=50.4)]))
