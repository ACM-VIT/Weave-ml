"""Qdrant storage adapter for Day-1 user profile embeddings."""

from __future__ import annotations

import copy
import threading
from collections.abc import Mapping, Sequence
from typing import Any

from config import QDRANT_API_KEY, QDRANT_URL
from .embeddings import aggregate_vectors
from .qdrant_cas import payload_matches, payload_snapshot_filter
from .vector_contract import (
    FEEDBACK_STATE_REVISION_FIELD,
    USER_PROFILE_COLLECTION_CONTRACT,
    canonical_backend_uuid,
    user_point_id,
    user_point_id,
    validate_embedding_vector,
)


class QdrantUserProfileStore:
    """Create, validate, and write the frozen unnamed user-vector collection."""

    def __init__(
        self,
        *,
        url: str = QDRANT_URL,
        api_key: str | None = QDRANT_API_KEY,
        timeout: int = 30,
        client: Any | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is required for user-vector storage. "
                "Run 'uv sync' to install project dependencies."
            ) from exc

        self.models = models
        self.contract = USER_PROFILE_COLLECTION_CONTRACT
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise ValueError("timeout must be a positive integer number of seconds")
        self.timeout = timeout
        self._ready = False
        self._ensure_lock = threading.Lock()
        self.client = (
            client
            if client is not None
            else QdrantClient(url=url, api_key=api_key, timeout=timeout)
        )

    def ensure_collection(self) -> None:
        """Create the user collection when absent, then validate its contract."""
        if self._ready:
            return
        with self._ensure_lock:
            if self._ready:
                return
            if not self._collection_exists():
                try:
                    self.client.create_collection(
                        collection_name=self.contract.collection_name,
                        vectors_config=self.models.VectorParams(
                            size=self.contract.vector_size,
                            distance=self._distance(),
                        ),
                    )
                except Exception as exc:
                    # Two simultaneous first-time onboarding requests may both
                    # see an absent collection; validate the winner's schema.
                    if "already exists" not in str(exc).lower():
                        raise
            self.validate_collection()
            self._ready = True

    def validate_collection(self) -> None:
        """Reject named, wrong-sized, or wrong-distance user collections."""
        if not self._collection_exists():
            raise ValueError(
                f"Qdrant collection {self.contract.collection_name!r} does not exist."
            )
        info = self.client.get_collection(self.contract.collection_name)
        vectors = info.config.params.vectors
        if isinstance(vectors, Mapping):
            raise ValueError(
                f"Qdrant collection {self.contract.collection_name!r} must use one "
                "unnamed user vector."
            )
        if vectors is None:
            raise ValueError(
                f"Qdrant collection {self.contract.collection_name!r} has no vector configuration."
            )
        if int(vectors.size) != self.contract.vector_size:
            raise ValueError(
                f"Qdrant collection {self.contract.collection_name!r} has vector size "
                f"{vectors.size}, expected {self.contract.vector_size}."
            )
        if vectors.distance != self._distance():
            raise ValueError(
                f"Qdrant collection {self.contract.collection_name!r} uses distance "
                f"{vectors.distance}, expected {self._distance()}."
            )

    def upsert_user(
        self,
        user_id: str,
        vector: Sequence[float],
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Normalize and upsert one user using its deterministic point ID."""
        canonical_user_id = canonical_backend_uuid(user_id, field_name="user_id")
        point_id = user_point_id(canonical_user_id)
        self.client.upsert(
            collection_name=self.contract.collection_name,
            points=[
                self._validated_user_point(
                    canonical_user_id,
                    point_id,
                    vector,
                    payload,
                )
            ],
            wait=True,
        )

    def compare_and_set_user(
        self,
        user_id: str,
        vector: Sequence[float],
        *,
        payload: Mapping[str, Any],
        expected_point: Any | None,
    ) -> bool:
        """Atomically replace a profile only if profile and feedback state match.

        Existing canonical points are updated in place so profile and feedback
        revisions remain protected by one compare-and-set operation.
        """

        canonical_user_id = canonical_backend_uuid(user_id, field_name="user_id")
        target_id = (
            user_point_id(canonical_user_id)
            if expected_point is None
            else str(expected_point.id)
        )
        point = self._validated_user_point(
            canonical_user_id,
            target_id,
            vector,
            payload,
        )
        compare_fields = (
            "profile_version",
            "job_id",
            "last_feedback_version",
            "last_feedback_event_id",
            FEEDBACK_STATE_REVISION_FIELD,
        )
        kwargs: dict[str, Any] = {
            "collection_name": self.contract.collection_name,
            "points": [point],
            "wait": True,
            "ordering": self.models.WriteOrdering.STRONG,
        }
        if expected_point is None:
            kwargs["update_mode"] = self.models.UpdateMode.INSERT_ONLY
        else:
            expected_payload = dict(expected_point.payload or {})
            desired_payload = dict(point.payload or {})
            current_version = self._stored_version(
                expected_payload,
                "profile_version",
            )
            requested_version = self._stored_version(
                desired_payload,
                "profile_version",
            )
            if requested_version < current_version:
                raise ValueError("profile_version cannot move backwards")
            if (
                requested_version == current_version
                and str(desired_payload.get("job_id") or "")
                != str(expected_payload.get("job_id") or "")
            ):
                raise ValueError(
                    "a profile revision cannot be replaced by a different job_id"
                )
            cursor_matches = payload_matches(
                desired_payload,
                expected_payload,
                (
                    "last_feedback_version",
                    "last_feedback_event_id",
                    FEEDBACK_STATE_REVISION_FIELD,
                ),
            )
            initializes_zero_cursor = (
                self._stored_version(desired_payload, "last_feedback_version") == 0
                and self._stored_version(expected_payload, "last_feedback_version") == 0
                and self._stored_version(
                    desired_payload, FEEDBACK_STATE_REVISION_FIELD
                )
                == 0
                and self._stored_version(
                    expected_payload, FEEDBACK_STATE_REVISION_FIELD
                )
                == 0
                and desired_payload.get("last_feedback_event_id") is None
                and expected_payload.get("last_feedback_event_id") is None
            )
            if not cursor_matches and not initializes_zero_cursor:
                raise ValueError(
                    "onboarding must preserve the observed feedback cursor"
                )
            kwargs.update(
                update_mode=self.models.UpdateMode.UPDATE_ONLY,
                update_filter=payload_snapshot_filter(
                    self.models,
                    point_id=expected_point.id,
                    payload=expected_payload,
                    fields=compare_fields,
                ),
            )
        self.client.upsert(**kwargs)

        records = self.client.retrieve(
            collection_name=self.contract.collection_name,
            ids=[target_id],
            with_payload=list(compare_fields),
            with_vectors=False,
        )
        if not records:
            return False
        stored_payload = dict(records[0].payload or {})
        desired_payload = dict(point.payload or {})
        # The job/version prove which profile won. Matching the cursor proves
        # the write did not claim success after a newer feedback write won.
        return payload_matches(
            stored_payload,
            desired_payload,
            compare_fields,
        )

    @staticmethod
    def _stored_version(payload: Mapping[str, Any], field: str) -> int:
        raw = payload.get(field, 0)
        if raw is None:
            return 0
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"stored {field} must be a non-negative integer")
        return raw

    def retrieve_user(self, user_id: str) -> Any | None:
        """Read canonical V2 user state."""
        canonical = user_point_id(user_id)
        points = self.client.retrieve(
            collection_name=self.contract.collection_name,
            ids=[canonical],
            with_payload=True,
            with_vectors=True,
        )
        by_id = {str(point.id): point for point in points}
        return by_id.get(canonical)

    def _validated_user_point(
        self,
        canonical_user_id: str,
        point_id: str,
        vector: Sequence[float],
        payload: Mapping[str, Any] | None,
    ) -> Any:
        validated = validate_embedding_vector(
            vector,
            expected_size=self.contract.vector_size,
            field_name=f"user embedding for {canonical_user_id}",
        )
        normalized = aggregate_vectors([validated])
        if payload is not None and not isinstance(payload, Mapping):
            raise TypeError("user payload must be a mapping or None")
        user_payload = copy.deepcopy(dict(payload or {}))
        user_payload["user_id"] = canonical_user_id
        return self.models.PointStruct(
            id=point_id,
            vector=normalized,
            payload=user_payload,
        )

    def _collection_exists(self) -> bool:
        if hasattr(self.client, "collection_exists"):
            return bool(self.client.collection_exists(self.contract.collection_name))
        try:
            self.client.get_collection(self.contract.collection_name)
            return True
        except Exception:
            return False

    def _distance(self):
        try:
            return getattr(self.models.Distance, self.contract.distance.upper())
        except AttributeError as exc:
            raise ValueError(
                f"Unsupported user collection distance {self.contract.distance!r}"
            ) from exc
