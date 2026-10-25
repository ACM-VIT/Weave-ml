from .embedding_pipeline import (
    RepositoryEmbeddingPipeline, embed_repositories, index_repositories,
    iter_embed_repositories, iter_index_repositories,
)
from .repository_embedding import RepositoryEmbeddingConfig, RepositoryEmbeddingResult
from .qdrant_store import QdrantRepositoryStore
from .user_profile_store import QdrantUserProfileStore
from .vector_contract import (
    REPOSITORY_COLLECTION_CONTRACT,
    REPOSITORY_DISCOVERY_CHANNELS,
    REPOSITORY_PAYLOAD_FIELD_TYPES,
    REPOSITORY_PAYLOAD_REQUIRED_FIELDS,
    USER_PROFILE_COLLECTION_CONTRACT,
    VectorCollectionContract,
    canonical_backend_uuid,
    repository_payload_defaults,
    repository_point_id,
    resolve_repository_identity,
    user_point_id,
    validate_embedding_vector,
    validate_repository_payload,
)

__all__ = [
    "RepositoryEmbeddingPipeline",
    "RepositoryEmbeddingConfig",
    "RepositoryEmbeddingResult",
    "QdrantRepositoryStore",
    "QdrantUserProfileStore",
    "embed_repositories",
    "index_repositories",
    "iter_embed_repositories",
    "iter_index_repositories",
    "VectorCollectionContract",
    "canonical_backend_uuid",
    "REPOSITORY_COLLECTION_CONTRACT",
    "REPOSITORY_DISCOVERY_CHANNELS",
    "USER_PROFILE_COLLECTION_CONTRACT",
    "REPOSITORY_PAYLOAD_FIELD_TYPES",
    "REPOSITORY_PAYLOAD_REQUIRED_FIELDS",
    "repository_payload_defaults",
    "repository_point_id",
    "resolve_repository_identity",
    "user_point_id",
    "validate_embedding_vector",
    "validate_repository_payload",
]
