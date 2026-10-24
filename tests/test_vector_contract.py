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
