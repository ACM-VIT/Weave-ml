"""Qdrant storage adapter for repository embeddings."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from numbers import Integral, Real
from typing import Any

from summarization.contracts import CardSummaryArtifact, card_summary_payload

from config import (
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_DISTANCE,
    QDRANT_PAYLOAD_INDEX_FIELDS,
    QDRANT_PAYLOAD_INDEX_SCHEMA,
    QDRANT_URL,
    QDRANT_VECTOR_NAME,
    REPOSITORY_EMBEDDING_DIM,
)
from .qdrant_cas import payload_matches, payload_snapshot_filter
from .repository_embedding import RepositoryEmbeddingResult
from .vector_contract import (
    REPOSITORY_DISCOVERY_CHANNELS,
    REPOSITORY_SERVING_ELIGIBILITY_FIELD,
    REPOSITORY_SERVING_ELIGIBILITY_VERSION,
    canonical_backend_uuid,
    repository_point_id,
    validate_embedding_vector,
    validate_repository_payload,
)

logger = logging.getLogger(__name__)


class QdrantRepositoryStore:
    """Own the public repository-vector interface to Qdrant."""

    def __init__(
        self,
        *,
        url: str = QDRANT_URL,
        api_key: str | None = QDRANT_API_KEY,
        collection_name: str = QDRANT_COLLECTION_NAME,
        vector_name: str = QDRANT_VECTOR_NAME,
        vector_size: int = REPOSITORY_EMBEDDING_DIM,
        distance: str = QDRANT_DISTANCE,
        client: Any | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is required for vector storage. "
                "Run 'uv sync' to install project dependencies."
            ) from exc

        self.models = models
        self.collection_name = collection_name
        self.vector_name = vector_name
        self.vector_size = vector_size
        self.distance = distance
        self._validate_store_config()
        self.client = client if client is not None else QdrantClient(url=url, api_key=api_key)

    def ensure_collection(self) -> None:
        """Create or validate the repository collection and payload indexes."""
        # The below check is for safe startup: existing collections are
        # validated instead of recreated, so stored vectors are not dropped.
        if not self._collection_exists():
            distance = self._distance()
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.vector_name: self.models.VectorParams(
                        size=self.vector_size,
                        distance=distance,
                    )
                },
            )
            logger.info("Created Qdrant collection: %s", self.collection_name)
        self._validate_collection()

        for field_name in QDRANT_PAYLOAD_INDEX_FIELDS:
            self._create_payload_index(field_name)

    def validate_collection(self) -> None:
        """Validate the configured collection without creating indexes."""
        if not self._collection_exists():
            raise ValueError(f"Qdrant collection {self.collection_name!r} does not exist.")
        self._validate_collection()

    def upsert(self, results: Iterable[RepositoryEmbeddingResult]) -> None:
        """Upsert embedding results into Qdrant."""
        points = []
        for result in results:
            # The below deterministic ID is for safe re-runs; the same repo is
            # updated instead of inserted as a duplicate vector.
            points.append(
                self._validated_point(
                    result,
                    point_id=self._point_id(result.repo_id),
                )
            )
        if not points:
            return
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        logger.info("Upserted %d repository vectors into %s", len(points), self.collection_name)

    def compare_and_set_content(
        self,
        result: RepositoryEmbeddingResult,
        *,
        expected_point: Any | None,
    ) -> Any | None:
        """Write content only when the observed content and features remain current.

        Qdrant evaluates the conditional filter atomically. Including the
        independent feature revision prevents an embedding worker that lost
        its Redis lease from erasing a newer feature refresh. Canonical point
        IDs are updated in place under the V2 identity contract.

        Qdrant reports a filtered no-op as a completed operation, so this
        method returns a post-write read of the exact target for verification.
        """

        target_id = (
            self._point_id(result.repo_id)
            if expected_point is None
            else str(expected_point.id)
        )
        point = self._validated_point(result, point_id=target_id)
        kwargs: dict[str, Any] = {
            "collection_name": self.collection_name,
            "points": [point],
            "wait": True,
            "ordering": self.models.WriteOrdering.STRONG,
        }
        if expected_point is None:
            kwargs["update_mode"] = self.models.UpdateMode.INSERT_ONLY
        else:
            desired_payload = dict(point.payload or {})
            expected_payload = dict(expected_point.payload or {})
            current_version = self._stored_version(
                expected_payload,
                "content_version",
            )
            requested_version = self._stored_version(
                desired_payload,
                "content_version",
            )
            if requested_version < current_version:
                raise ValueError("content_version cannot move backwards")
            if (
                requested_version == current_version
                and str(desired_payload.get("content_job_id") or "")
                != str(expected_payload.get("content_job_id") or "")
            ):
                raise ValueError(
                    "a content revision cannot be replaced by a different job_id"
                )
            if not payload_matches(
                desired_payload,
                expected_payload,
                ("feature_version", "feature_job_id"),
            ):
                raise ValueError(
                    "content upsert must preserve the observed feature revision"
                )
            kwargs.update(
                update_mode=self.models.UpdateMode.UPDATE_ONLY,
                update_filter=payload_snapshot_filter(
                    self.models,
                    point_id=expected_point.id,
                    payload=expected_payload,
                    fields=(
                        "content_version",
                        "content_job_id",
                        "feature_version",
                        "feature_job_id",
                    ),
                ),
            )
        self.client.upsert(**kwargs)
        return self._retrieve_point(target_id)

    def compare_and_set_features(
        self,
        *,
        expected_point: Any,
        feature_payload: Mapping[str, Any],
    ) -> Any | None:
        """Apply a feature patch only to the exact feature revision read."""

        if expected_point is None:
            raise ValueError("expected_point is required for a feature refresh")
        if not isinstance(feature_payload, Mapping) or not feature_payload:
            raise ValueError("feature_payload must be a non-empty mapping")
        expected_payload = dict(expected_point.payload or {})
        current_version = self._stored_version(expected_payload, "feature_version")
        requested_version = self._stored_version(feature_payload, "feature_version")
        if requested_version < current_version:
            raise ValueError("feature_version cannot move backwards")
        if (
            requested_version == current_version
            and str(feature_payload.get("feature_job_id") or "")
            != str(expected_payload.get("feature_job_id") or "")
        ):
            raise ValueError(
                "a feature revision cannot be replaced by a different job_id"
            )
        selector = payload_snapshot_filter(
            self.models,
            point_id=expected_point.id,
            payload=expected_payload,
            fields=("feature_version", "feature_job_id"),
        )
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=dict(feature_payload),
            points=selector,
            wait=True,
            ordering=self.models.WriteOrdering.STRONG,
        )
        return self._retrieve_point(str(expected_point.id))

    def compare_and_set_card_summary(
        self,
        *,
        expected_point: Any,
        artifact: CardSummaryArtifact,
    ) -> Any | None:
        """Attach/replace only the summary artifact for an unchanged content snapshot."""

        if expected_point is None:
            raise ValueError("expected_point is required for a card-summary backfill")
        expected_payload = dict(expected_point.payload or {})
        selector = payload_snapshot_filter(
            self.models,
            point_id=expected_point.id,
            payload=expected_payload,
            fields=(
                "content_version",
                "content_job_id",
                "card_summary_artifact_hash",
            ),
        )
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=card_summary_payload(artifact),
            points=selector,
            wait=True,
            ordering=self.models.WriteOrdering.STRONG,
        )
        return self._retrieve_point(str(expected_point.id))

    def _validated_point(
        self,
        result: RepositoryEmbeddingResult,
        *,
        point_id: str,
    ) -> Any:
        vector = validate_embedding_vector(
            result.final_embedding,
            expected_size=self.vector_size,
            field_name=f"embedding for {result.repo_id}",
        )
        validate_repository_payload(
            result.payload,
            require_serving_eligibility=False,
        )
        if result.payload["repo_id"].strip() != result.repo_id.strip():
            raise ValueError(
                "embedding result repo_id does not match its repository payload"
            )
        serving_payload = dict(result.payload)
        serving_payload[REPOSITORY_SERVING_ELIGIBILITY_FIELD] = (
            REPOSITORY_SERVING_ELIGIBILITY_VERSION
        )
        validate_repository_payload(serving_payload)
        return self.models.PointStruct(
            id=point_id,
            vector={self.vector_name: vector},
            payload=serving_payload,
        )

    def _retrieve_point(self, point_id: str) -> Any | None:
        records = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        return records[0] if records else None

    @staticmethod
    def _stored_version(payload: Mapping[str, Any], field: str) -> int:
        raw = payload.get(field, 0)
        if raw is None:
            return 0
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"stored {field} must be a non-negative integer")
        return raw

    def search(
        self,
        vector: list[float],
        *,
        limit: int = 5,
        with_vectors: bool = False,
        exact: bool = True,
        score_threshold: float | None = None,
        query_filter: Any | None = None,
    ) -> list[dict]:
        """Search Qdrant by final repository embedding vector."""
        # The below query uses the named vector configured for repository
        # embeddings, so search targets the final repo embedding field. Exact
        # search is the default for evaluation-grade nearest-neighbor results.
        query_vector = validate_embedding_vector(
            vector,
            expected_size=self.vector_size,
            field_name="repository search vector",
        )
        self._validate_limit(limit)
        if not isinstance(exact, bool):
            raise TypeError("exact must be a boolean")
        if score_threshold is not None:
            if isinstance(score_threshold, bool) or not isinstance(score_threshold, Real):
                raise TypeError("score_threshold must be a real number or None")
            score_threshold = float(score_threshold)
            if not math.isfinite(score_threshold):
                raise ValueError("score_threshold must be finite")

        search_params = self.models.SearchParams(exact=exact)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=self.vector_name,
            query_filter=query_filter,
            search_params=search_params,
            limit=limit,
            with_payload=True,
            with_vectors=with_vectors,
            score_threshold=score_threshold,
        )
        return [self._format_point(point, with_vectors=with_vectors) for point in response.points]

    def semantic_search(
        self,
        vector: list[float],
        *,
        limit: int = 50,
        with_vectors: bool = True,
        score_threshold: float | None = None,
        query_filter: Any | None = None,
        exact: bool = False,
    ) -> list[dict]:
        """Run semantic retrieval with approximate search as the production default."""
        return self.search(
            vector,
            limit=limit,
            with_vectors=with_vectors,
            exact=exact,
            score_threshold=score_threshold,
            query_filter=query_filter,
        )

    def discover(
        self,
        channel: str,
        *,
        limit: int = 50,
        with_vectors: bool = True,
        query_filter: Any | None = None,
    ) -> list[dict]:
        """Return repositories ordered by a frozen discovery channel."""
        if not isinstance(channel, str):
            raise TypeError("channel must be a string")
        canonical_channel = channel.strip().lower()
        order_field = REPOSITORY_DISCOVERY_CHANNELS.get(canonical_channel)
        if order_field is None:
            allowed = ", ".join(REPOSITORY_DISCOVERY_CHANNELS)
            raise ValueError(f"Unsupported discovery channel {channel!r}. Use one of: {allowed}")
        self._validate_limit(limit)

        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            order_by=self.models.OrderBy(
                key=order_field,
                direction=self.models.Direction.DESC,
            ),
            with_payload=True,
            with_vectors=[self.vector_name] if with_vectors else False,
        )
        return [self._format_point(record, with_vectors=with_vectors) for record in records]

    def retrieve_batch(
        self,
        repo_ids: Iterable[str],
        *,
        with_vectors: bool = True,
    ) -> list[dict]:
        """Retrieve repositories by stable repo ID in one Qdrant request.

        Duplicate IDs are requested once, and results follow the caller's
        original ID order even if Qdrant returns records in another order.
        """
        canonical_repo_ids = self._canonical_repo_ids(repo_ids)
        if not canonical_repo_ids:
            return []
        point_ids = [repository_point_id(repo_id) for repo_id in canonical_repo_ids]
        records = self.client.retrieve(
            collection_name=self.collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=[self.vector_name] if with_vectors else False,
        )
        by_point_id = {
            str(record.id): self._format_point(record, with_vectors=with_vectors)
            for record in records
        }
        return [by_point_id[point_id] for point_id in point_ids if point_id in by_point_id]

    def list_points(self, *, limit: int = 100, with_vectors: bool = True) -> list[dict]:
        """Load repository points from Qdrant for offline evaluation."""
        self._validate_limit(limit)
        points: list[dict] = []
        offset = None
        while len(points) < limit:
            # The below scroll call is for evaluation/reporting workflows that
            # need existing payloads and vectors from Qdrant without a query.
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=min(100, limit - len(points)),
                offset=offset,
                with_payload=True,
                with_vectors=[self.vector_name] if with_vectors else False,
            )
            if not records:
                break
            points.extend(
                self._format_point(record, with_vectors=with_vectors) for record in records
            )
            if offset is None:
                break
        return points

    def _format_point(self, point: Any, *, with_vectors: bool) -> dict:
        payload = point.payload or {}
        vector = self._extract_vector(point.vector) if with_vectors else None
        if vector is not None:
            vector = validate_embedding_vector(
                vector,
                expected_size=self.vector_size,
                field_name=f"stored embedding for point {point.id}",
            )
        raw_score = getattr(point, "score", None)
        return {
            "id": str(point.id),
            "score": float(raw_score) if raw_score is not None else None,
            "repo_id": payload.get("repo_id"),
            "full_name": payload.get("full_name"),
            "payload": payload,
            "vector": vector,
        }

    def _extract_vector(self, vector_data) -> list[float] | None:
        if vector_data is None:
            return None
        if isinstance(vector_data, dict):
            vector = vector_data.get(self.vector_name)
            return list(vector) if vector is not None else None
        return list(vector_data)

    def _collection_exists(self) -> bool:
        if hasattr(self.client, "collection_exists"):
            return bool(self.client.collection_exists(self.collection_name))
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def _validate_collection(self) -> None:
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        vector_config = vectors.get(self.vector_name) if isinstance(vectors, Mapping) else None
        if vector_config is None:
            raise ValueError(
                f"Qdrant collection {self.collection_name!r} does not define vector "
                f"{self.vector_name!r}."
            )
        if int(vector_config.size) != int(self.vector_size):
            raise ValueError(
                f"Qdrant collection {self.collection_name!r} vector {self.vector_name!r} "
                f"has size {vector_config.size}, expected {self.vector_size}."
            )
        expected_distance = self._distance()
        if vector_config.distance != expected_distance:
            raise ValueError(
                f"Qdrant collection {self.collection_name!r} vector {self.vector_name!r} "
                f"uses distance {vector_config.distance}, expected {expected_distance}."
            )

    def _validate_store_config(self) -> None:
        for field_name, value in (
            ("collection_name", self.collection_name),
            ("vector_name", self.vector_name),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if (
            isinstance(self.vector_size, bool)
            or not isinstance(self.vector_size, Integral)
            or self.vector_size <= 0
        ):
            raise ValueError("vector_size must be a positive integer")
        if not isinstance(self.distance, str) or not self.distance.strip():
            raise ValueError("distance must be a non-empty string")
        self._distance()

    def _create_payload_index(self, field_name: str) -> None:
        # The below schema selection is for keeping payload indexes aligned with
        # the payload fields emitted by build_vector_payload.
        schema_name = QDRANT_PAYLOAD_INDEX_SCHEMA[field_name]
        schema = getattr(self.models.PayloadSchemaType, schema_name.upper())
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to create Qdrant payload index {field_name!r}") from exc

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, Integral) or limit <= 0:
            raise ValueError("limit must be a positive integer")

    @staticmethod
    def _canonical_repo_ids(repo_ids: Iterable[str]) -> list[str]:
        if isinstance(repo_ids, (str, bytes)):
            raise TypeError("repo_ids must be an iterable of repository ID strings")
        canonical: list[str] = []
        seen: set[str] = set()
        for repo_id in repo_ids:
            normalized = canonical_backend_uuid(repo_id, field_name="repo_id")
            if normalized not in seen:
                seen.add(normalized)
                canonical.append(normalized)
        return canonical

    @staticmethod
    def _point_id(repo_id: str) -> str:
        return repository_point_id(repo_id)

    def _distance(self):
        try:
            return getattr(self.models.Distance, self.distance.upper())
        except AttributeError as exc:
            allowed = ", ".join(item.name for item in self.models.Distance)
            raise ValueError(
                f"Unsupported Qdrant distance {self.distance!r}. Use one of: {allowed}"
            ) from exc
