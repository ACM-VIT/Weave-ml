"""Day-1 user-interest embedding and Qdrant onboarding workflow."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import threading
from collections.abc import Mapping
from typing import Any

from config import (
    EMBEDDING_MODEL_REVISION,
    QDRANT_API_KEY,
    QDRANT_URL,
    REPOSITORY_EMBEDDING_MODEL,
)
from embedding.embeddings import aggregate_vectors
from embedding.repository_embedding import SUPPORTED_REPOSITORY_EMBEDDING_DIMS
from embedding.tower_cache import TowerEmbeddingCache
from embedding.user_profile_store import QdrantUserProfileStore
from embedding.vector_contract import (
    FEEDBACK_STATE_REVISION_FIELD,
    USER_PROFILE_COLLECTION_CONTRACT,
    canonical_backend_uuid,
    validate_embedding_vector,
)
from utils.deadline import check_deadline

logger = logging.getLogger("pipeline.user_onboarding")

# Compatibility exports used by retrieval and feedback.  The frozen user
# collection deliberately has one unnamed vector, so TARGET_VECTOR_NAME is None.
EMBEDDING_MODEL = REPOSITORY_EMBEDDING_MODEL
VECTOR_DIMENSION = USER_PROFILE_COLLECTION_CONTRACT.vector_size
USER_PROFILES_COLLECTION = USER_PROFILE_COLLECTION_CONTRACT.collection_name
TARGET_VECTOR_NAME = USER_PROFILE_COLLECTION_CONTRACT.vector_name
# Bump when semantic projection, encoding options or normalization changes.
USER_BASELINE_PREPROCESSING_VERSION = "user-context-l2-v1"

_FEEDBACK_PAYLOAD_KEYS = {
    "feedback_latent_vector",
    "feedback_adjustments",
    "feedback_applied_signals",
    "feedback_processed_events",
    "feedback_rejections",
    FEEDBACK_STATE_REVISION_FIELD,
    "last_feedback_version",
    "last_feedback_event_id",
    "last_feedback_status",
    "preference_accumulator",
}

_EXPECTED_USER_NOT_SUPPLIED = object()


class UserProfileWriteConflict(RuntimeError):
    """The user snapshot changed before its conditional profile write."""


def _is_feedback_payload_key(key: str) -> bool:
    return (
        key in _FEEDBACK_PAYLOAD_KEYS
        or key.startswith("feedback_")
        or key.startswith("last_feedback_")
    )


def _stored_vector(point: Any) -> list[float]:
    value = point.vector
    if isinstance(value, Mapping):
        if TARGET_VECTOR_NAME and TARGET_VECTOR_NAME in value:
            value = value[TARGET_VECTOR_NAME]
        elif len(value) == 1:
            value = next(iter(value.values()))
        else:
            raise ValueError("stored user profile has ambiguous named vectors")
    return validate_embedding_vector(
        list(value), expected_size=VECTOR_DIMENSION, field_name="stored user embedding"
    )


def _has_feedback_state(payload: Mapping[str, Any]) -> bool:
    try:
        if int(payload.get("last_feedback_version") or 0) > 0:
            return True
    except (TypeError, ValueError) as exc:
        raise ValueError("last_feedback_version must be a non-negative integer") from exc
    return any(
        key in payload
        and key not in {"last_feedback_version", "preference_accumulator"}
        for key in payload
        if _is_feedback_payload_key(key)
    )


def _string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, set):
        values = sorted(value, key=str)
    else:
        values = value if isinstance(value, (list, tuple)) else [value]
    return [
        text
        for item in values
        if item is not None and (text := str(item).strip())
    ]


def _synthesize_user_context_impl(user_data: dict[str, Any]) -> str:
    """Flatten supported profile fields into deterministic embedding text."""
    if not isinstance(user_data, dict):
        raise ValueError("user_data must be a dictionary.")

    context_parts: list[str] = []
    for label, field_name in (
        ("Skills", "skills"),
        ("Tech Stack", "tech_stack"),
    ):
        values = _string_values(user_data.get(field_name))
        if values:
            context_parts.append(f"{label}: {', '.join(values)}")

    # Backend v2 calls these values ``topics``; the existing ML/API contract
    # called them ``interests``.  Both map to one semantic interest sentence.
    interest_values = _string_values(user_data.get("interests"))
    interest_values.extend(_string_values(user_data.get("topics")))
    unique_interests = list(dict.fromkeys(interest_values))
    if unique_interests:
        context_parts.append(f"Interests: {', '.join(unique_interests)}")

    bio = user_data.get("bio")
    if bio is not None and str(bio).strip():
        context_parts.append(f"Bio: {str(bio).strip()}")

    if not context_parts:
        raise ValueError("User data is empty or missing all embedding fields.")
    return ". ".join(context_parts)


def _validate_backend_onboarding_metadata(user_data: dict[str, Any]) -> None:
    if not isinstance(user_data, dict):
        raise ValueError("user_data must be a dictionary.")
    if "job_id" in user_data:
        canonical_backend_uuid(user_data["job_id"], field_name="job_id")
    if "profile_version" in user_data:
        profile_version = user_data["profile_version"]
        if (
            isinstance(profile_version, bool)
            or not isinstance(profile_version, int)
            or profile_version < 1
        ):
            raise ValueError("profile_version must be a positive integer")


class UserOnboardingPipeline:
    """Generate and store a contract-compatible Day-1 user interest vector."""

    def __init__(
        self,
        embedding_model: str | None = None,
        *,
        model: Any | None = None,
        store: QdrantUserProfileStore | None = None,
    ) -> None:
        self.model_name = embedding_model or EMBEDDING_MODEL
        expected_dimension = SUPPORTED_REPOSITORY_EMBEDDING_DIMS.get(self.model_name)
        if expected_dimension != VECTOR_DIMENSION:
            supported = ", ".join(sorted(SUPPORTED_REPOSITORY_EMBEDDING_DIMS))
            raise ValueError(
                f"Unsupported onboarding embedding model {self.model_name!r}. "
                f"The user collection requires {VECTOR_DIMENSION} dimensions. "
                f"Supported models: {supported}."
            )
        self._model = model
        self.store = store
        self._baseline_cache: TowerEmbeddingCache | None = None
        self._baseline_cache_lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for user onboarding. "
                    "Run 'uv sync' to install project dependencies."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
            logger.info("Loaded onboarding embedding model: %s", self.model_name)
        return self._model

    @property
    def model(self):
        """Return the lazily loaded SentenceTransformer model."""
        return self._get_model()

    def synthesize_user_context(self, user_data: dict[str, Any]) -> str:
        return _synthesize_user_context_impl(user_data)

    def _cache_for_model(self, model: Any) -> TowerEmbeddingCache:
        revision = getattr(model, "query_embedding_revision", EMBEDDING_MODEL_REVISION)
        with self._baseline_cache_lock:
            cache = self._baseline_cache
            if cache is None or not cache.owns(model) or cache.model_revision != revision:
                # Separate from repository towers: at most 128 packed 384D
                # vectors (384 KiB), plus bounded digest/entry overhead. No text,
                # user identity, payload or in-flight inference is retained.
                cache = TowerEmbeddingCache(
                    model=model, model_revision=revision,
                    max_entries=128, max_bytes=128 * VECTOR_DIMENSION * 8,
                )
                self._baseline_cache = cache
            return cache

    def generate_interest_vector(self, user_data: dict[str, Any]) -> list[float]:
        """Create one finite, normalized 384-dimensional interest vector."""
        _validate_backend_onboarding_metadata(user_data)
        context = self.synthesize_user_context(user_data)
        check_deadline()
        model = self._get_model()
        cache = self._cache_for_model(model)
        key = hashlib.sha256(json.dumps({
            "context": context, "model": self.model_name,
            "model_revision": cache.model_revision, "dimension": VECTOR_DIMENSION,
            "preprocessing": USER_BASELINE_PREPROCESSING_VERSION,
            "max_seq_length": getattr(model, "max_seq_length", 256),
            "normalize_embeddings": True, "convert_to_numpy": True,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).digest()
        cached = cache.get(key, expected_size=VECTOR_DIMENSION)
        if cached is not None and math.isclose(
            sum(value * value for value in cached), 1.0, rel_tol=1e-6, abs_tol=1e-6,
        ):
            return cached
        embedding = model.encode(
            context,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if embedding is None:
            raise ValueError("The onboarding embedding model returned no vector.")
        raw_vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        validated = validate_embedding_vector(
            raw_vector,
            expected_size=VECTOR_DIMENSION,
            field_name="generated user embedding",
        )
        # Normalize again at our boundary rather than relying on model-specific
        # behavior.  This also rejects an unusable all-zero embedding.
        normalized = aggregate_vectors([validated])
        cache.put(key, normalized, expected_size=VECTOR_DIMENSION)
        return normalized

    def save_to_qdrant(
        self,
        user_id: str,
        vector: list[float],
        payload: dict[str, Any] | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
        *,
        expected_point: Any = _EXPECTED_USER_NOT_SUPPLIED,
    ) -> bool:
        """Validate and conditionally persist one unnamed user vector.

        When ``expected_point`` is supplied by the API, the write is fenced by
        that exact profile revision and feedback cursor. Direct/offline callers
        take one snapshot here and receive ``UserProfileWriteConflict`` if it
        changes before the write.
        """
        canonical_user_id = canonical_backend_uuid(user_id, field_name="user_id")
        validated = validate_embedding_vector(
            vector,
            expected_size=VECTOR_DIMENSION,
            field_name=f"user embedding for {canonical_user_id}",
        )
        normalized = aggregate_vectors([validated])
        if payload is not None and not isinstance(payload, dict):
            raise TypeError("payload must be a dictionary or None")
        user_payload = copy.deepcopy(payload or {})
        _validate_backend_onboarding_metadata(user_payload)
        if "job_id" in user_payload:
            user_payload["job_id"] = canonical_backend_uuid(
                user_payload["job_id"], field_name="job_id"
            )
        active_store = self.store or QdrantUserProfileStore(
            url=qdrant_url or QDRANT_URL,
            api_key=qdrant_api_key or QDRANT_API_KEY,
        )
        active_store.ensure_collection()
        if expected_point is _EXPECTED_USER_NOT_SUPPLIED:
            existing = (
                active_store.retrieve_user(canonical_user_id)
                if hasattr(active_store, "retrieve_user")
                else None
            )
        else:
            existing = expected_point
        existing_payload = copy.deepcopy(dict(existing.payload or {})) if existing else {}
        existing_profile_version = int(existing_payload.get("profile_version") or 0)
        requested_profile_version = int(user_payload.get("profile_version") or 0)
        if requested_profile_version and requested_profile_version < existing_profile_version:
            raise ValueError(
                f"profile_version {requested_profile_version} is older than stored "
                f"version {existing_profile_version}"
            )

        stored_vector = normalized
        if existing and _has_feedback_state(existing_payload):
            stored_vector = _stored_vector(existing)
            for key in existing_payload:
                if _is_feedback_payload_key(key):
                    user_payload[key] = copy.deepcopy(existing_payload[key])
            # Keep the new profile baseline for a deliberate future replay or
            # reconciliation without replacing the live learned state.
            user_payload["profile_baseline_vector"] = list(normalized)
        else:
            user_payload["preference_accumulator"] = list(normalized)
            user_payload.setdefault("last_feedback_version", 0)

        user_payload.update(
            {
                "user_id": canonical_user_id,
                "embedding_dim": VECTOR_DIMENSION,
                "embedding_model": self.model_name,
                "embedding_model_revision": EMBEDDING_MODEL_REVISION,
            }
        )
        if hasattr(active_store, "compare_and_set_user"):
            applied = active_store.compare_and_set_user(
                canonical_user_id,
                stored_vector,
                payload=user_payload,
                expected_point=existing,
            )
            if not applied:
                raise UserProfileWriteConflict(
                    "user profile or feedback state changed before storage"
                )
        else:
            # Lightweight custom stores used by offline callers retain their
            # old interface. Production Qdrant stores always take the CAS path.
            active_store.upsert_user(
                canonical_user_id,
                stored_vector,
                payload=user_payload,
            )
        # V2 writes only the canonical point identity. Any pre-cutover identity
        # cleanup is an offline administrative operation.
        return True

    def onboard_user(
        self,
        user_id: str,
        user_data: dict[str, Any],
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ) -> bool:
        """Generate and persist a user's Day-1 interest vector."""
        try:
            canonical_user_id = canonical_backend_uuid(user_id, field_name="user_id")
            vector = self.generate_interest_vector(user_data)
            return self.save_to_qdrant(
                user_id=canonical_user_id,
                vector=vector,
                payload=user_data,
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
            )
        except (TypeError, ValueError):
            raise
        except Exception as exc:
            logger.error("User onboarding failed for %r: %s", user_id, exc)
            return False


def synthesize_user_context(user_data: dict[str, Any]) -> str:
    return _synthesize_user_context_impl(user_data)


def generate_interest_vector(user_data: dict[str, Any]) -> list[float]:
    return UserOnboardingPipeline().generate_interest_vector(user_data)


def save_user_vector_to_qdrant(
    user_id: str,
    vector: list[float],
    payload: dict[str, Any] | None = None,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
) -> bool:
    return UserOnboardingPipeline().save_to_qdrant(
        user_id=user_id,
        vector=vector,
        payload=payload,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
    )


def onboard_user(
    user_id: str,
    user_data: dict[str, Any],
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
) -> bool:
    try:
        return UserOnboardingPipeline().onboard_user(
            user_id=user_id,
            user_data=user_data,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
        )
    except Exception as exc:
        logger.error("User onboarding failed for %r: %s", user_id, exc)
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    example_user = {
        "user_id": "00000000-0000-4000-8000-000000000101",
        "skills": ["Python", "Machine Learning", "Data Engineering"],
        "tech_stack": ["PyTorch", "FastAPI", "PostgreSQL", "Docker"],
        "interests": ["AI/ML", "Open Source", "Cloud Computing", "MLOps"],
        "bio": "ML engineer building scalable recommendation pipelines.",
    }
    success = onboard_user(
        user_id=example_user["user_id"],
        user_data=example_user,
    )
    print("User onboarded successfully." if success else "User onboarding failed.")
