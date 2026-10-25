"""Qdrant storage adapter for repository embeddings."""

from __future__ import annotations

from ingestion.lifecycle import LIFECYCLE_FIELDS

import logging
import math
from collections.abc import Iterable, Iterator, Mapping

from utils.deadline import check_deadline
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
from .batch_limits import check_field_budget
from .corpus_revision import (
    CorpusAuthorityUnavailable, CorpusRevisionAuthority, MaintenanceProof,
    authority_from_environment, operation_completed,
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
_AUTHORITY_FROM_ENV = object()


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
        corpus_authority: CorpusRevisionAuthority | None | object = _AUTHORITY_FROM_ENV,
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
        if corpus_authority is _AUTHORITY_FROM_ENV:
            corpus_authority = authority_from_environment(
                collection=collection_name, vector=vector_name, endpoint=url,
            )
            if corpus_authority is not None and client is not None:
                corpus_authority.close()
                raise ValueError("enabled authority requires an explicit matching authority for a custom Qdrant client")
        elif corpus_authority is None:
            # An explicitly omitted authority must not silently bypass an enabled
            # protocol while another process can still expose a READY revision.
            configured = authority_from_environment(collection=collection_name, vector=vector_name, endpoint=url)
            if configured is not None:
                configured.close()
                raise ValueError("an enabled corpus authority cannot be omitted by a writer")
        if corpus_authority is not None and not isinstance(corpus_authority, CorpusRevisionAuthority):
            raise TypeError("corpus_authority must be a CorpusRevisionAuthority")
        self.corpus_authority = corpus_authority
        self.client = client if client is not None else QdrantClient(url=url, api_key=api_key)

    def ensure_collection(self) -> None:
        """Create or validate the repository collection and payload indexes."""
        # The below check is for safe startup: existing collections are
        # validated instead of recreated, so stored vectors are not dropped.
        if not self._collection_exists():
            def create():
                from qdrant_client.http.exceptions import UnexpectedResponse

                try:
                    return self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config={
                            self.vector_name: self.models.VectorParams(
                                size=self.vector_size,
                                distance=self._distance(),
                            )
                        },
                    )
                except UnexpectedResponse as exc:
                    if exc.status_code != 409:
                        raise
                    # A concurrent creator won. Only an explicit conflict is
                    # known not to have dispatched our creation; lost replies
                    # and other errors still retain the authority writer token.
                    self._validate_collection()
                    return True
            self._mutate(create, synchronous_boolean=True)
        info = self._validate_collection()
        missing = self._missing_payload_indexes(info)
        for field_name in missing:
            # Another bootstrap may have provisioned this field, or an admin
            # may have replaced the collection since the initial inspection.
            current_missing = self._missing_payload_indexes(self._validate_collection())
            if field_name not in current_missing:
                continue
            self._create_payload_index(field_name)
        if missing:
            remaining = self._missing_payload_indexes(self._validate_collection())
            if remaining:
                raise RuntimeError(
                    f"Qdrant payload indexes are missing after provisioning: {remaining}"
                )

    def validate_collection(self) -> None:
        """Validate the configured collection without creating indexes."""
        if not self._collection_exists():
            raise ValueError(f"Qdrant collection {self.collection_name!r} does not exist.")
        missing = self._missing_payload_indexes(self._validate_collection())
        if missing:
            raise ValueError(f"Qdrant payload indexes are missing: {missing}")

    def iter_upsert(
        self,
        results: Iterable[RepositoryEmbeddingResult],
        *,
        max_points: int = 64,
        max_request_bytes: int = 1024 * 1024,
    ) -> Iterator[RepositoryEmbeddingResult]:
        """Yield acknowledged offline writes bounded by points and REST bytes.

        Earlier chunks stay committed when validation, iteration or a write
        fails. The failed/ambiguous chunk is never yielded. Restart by replaying
        deterministic IDs; this is deliberately separate from online CAS.
        """
        self._validate_limit(max_points)
        self._validate_limit(max_request_bytes)
        # Use the client's REST serializer, including UTF-8, escaping and the
        # request envelope. Character counts/estimated vector sizes undercount
        # multilingual payloads and float representations.
        from qdrant_client.http.api.points_api import jsonable_encoder

        envelope_bytes = len(jsonable_encoder(self.models.PointsList(points=[])).encode("utf-8"))
        points, pending = [], []
        request_bytes = envelope_bytes

        def flush():
            check_deadline()
            response = self._mutate(lambda: self.client.upsert(
                collection_name=self.collection_name, points=points, wait=True,
            ))
            if not operation_completed(response):
                raise CorpusAuthorityUnavailable(
                    "Qdrant chunk completion is not proven; replay before recording progress"
                )
            logger.info("Upserted %d repository vectors into %s (%d REST bytes)",
                        len(points), self.collection_name, request_bytes)

        for result in results:
            check_deadline()
            check_field_budget(
                [result.payload], max_bytes=max_request_bytes, label="Qdrant point payload",
            )
            point = self._validated_point(result, point_id=self._point_id(result.repo_id))
            size = len(jsonable_encoder(point).encode("utf-8"))
            if size + envelope_bytes > max_request_bytes:
                raise ValueError(f"repository {result.repo_id} exceeds the Qdrant request byte budget")
            if points and (len(points) >= max_points or request_bytes + 1 + size > max_request_bytes):
                flush()
                yield from pending
                points, pending = [], []
                request_bytes = envelope_bytes
            request_bytes += size + int(bool(points))
            points.append(point)
            pending.append(result)
            if len(points) == max_points:
                flush()
                yield from pending
                points, pending = [], []
                request_bytes = envelope_bytes
        if points:
            flush()
            yield from pending

    def upsert(
        self,
        results: Iterable[RepositoryEmbeddingResult],
        *,
        max_points: int = 64,
        max_request_bytes: int = 1024 * 1024,
    ) -> None:
        """Consume bounded offline writes; use iter_upsert for progress."""
        for _ in self.iter_upsert(results, max_points=max_points, max_request_bytes=max_request_bytes):
            pass

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
                        *LIFECYCLE_FIELDS,
                    ),
                ),
            )
        self._mutate(lambda: self.client.upsert(**kwargs))
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
            fields=("feature_version", "feature_job_id", *LIFECYCLE_FIELDS),
        )
        self._mutate(lambda: self.client.set_payload(
            collection_name=self.collection_name,
            payload=dict(feature_payload),
            points=selector,
            wait=True,
            ordering=self.models.WriteOrdering.STRONG,
        ))
        return self._retrieve_point(str(expected_point.id))

    def compare_and_set_lifecycle(self, *, expected_point: Any, lifecycle_payload: Mapping[str, Any]) -> Any | None:
        """Status observations can arrive after newer unrelated feature jobs.

        Fence both the independent status snapshot and feature revision; never
        rewind feature versions or overwrite counts while applying delayed status.
        """
        if not lifecycle_payload or any(field not in LIFECYCLE_FIELDS for field in lifecycle_payload):
            raise ValueError("lifecycle CAS accepts only lifecycle observation fields")
        expected_payload = dict(expected_point.payload or {})
        selector = payload_snapshot_filter(
            self.models, point_id=expected_point.id, payload=expected_payload,
            fields=("content_version", "content_job_id", "feature_version", "feature_job_id", *LIFECYCLE_FIELDS),
        )
        self._mutate(lambda: self.client.set_payload(
            collection_name=self.collection_name, payload=dict(lifecycle_payload),
            points=selector, wait=True, ordering=self.models.WriteOrdering.STRONG,
        ))
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
        self._mutate(lambda: self.client.set_payload(
            collection_name=self.collection_name,
            payload=card_summary_payload(artifact),
            points=selector,
            wait=True,
            ordering=self.models.WriteOrdering.STRONG,
        ))
        return self._retrieve_point(str(expected_point.id))

    def _mutate(self, operation, *, synchronous_boolean: bool = False):
        if self.corpus_authority is None:
            return operation()
        return self.corpus_authority.mutate(operation, synchronous_boolean=synchronous_boolean)

    def delete_repositories(self, repo_ids: Iterable[str]) -> None:
        """Managed bounded canonical deletion; no public product deletion API."""
        ids = []
        for value in repo_ids:
            if len(ids) >= 100:
                raise ValueError("managed deletion accepts at most 100 repository IDs")
            ids.append(canonical_backend_uuid(value, field_name="repo_id"))
        ids = list(dict.fromkeys(ids))
        if ids:
            self._mutate(lambda: self.client.delete(
                collection_name=self.collection_name,
                points_selector=self.models.PointIdsList(points=ids),
                wait=True, ordering=self.models.WriteOrdering.STRONG,
            ))

    def replace_empty_collection(self, *, proof: MaintenanceProof) -> None:
        """Controlled empty replacement after proved retirement/completion.

        This is a maintenance primitive, never automatic recovery. It does not
        clear unknown authority tokens or bootstrap a missing authority record.
        """
        if not isinstance(proof, MaintenanceProof):
            raise ValueError("explicit maintenance proof is required for replacement")
        def replace():
            if self.client.delete_collection(self.collection_name) is not True:
                raise CorpusAuthorityUnavailable("collection deletion completion not proven")
            if self.client.create_collection(self.collection_name, vectors_config={
                self.vector_name: self.models.VectorParams(
                    size=self.vector_size, distance=self._distance(),
                ),
            }) is not True:
                raise CorpusAuthorityUnavailable("collection creation completion not proven")
            return True
        self._mutate(replace, synchronous_boolean=True)
        self.ensure_collection()

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
        except Exception as exc:
            from qdrant_client.http.exceptions import UnexpectedResponse

            if isinstance(exc, UnexpectedResponse) and exc.status_code == 404:
                return False
            raise

    def _validate_collection(self) -> Any:
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
        return info

    def _missing_payload_indexes(self, info: Any) -> list[str]:
        """Validate authoritative metadata before issuing any schema mutation."""
        schema = getattr(info, "payload_schema", None)
        if not isinstance(schema, Mapping):
            raise ValueError("Qdrant repository payload index schema is unavailable")
        missing = []
        for field in QDRANT_PAYLOAD_INDEX_FIELDS:
            if field not in schema:
                missing.append(field)
                continue
            actual = schema[field]
            if isinstance(actual, Mapping):
                actual = actual.get("data_type")
            else:
                actual = getattr(actual, "data_type", actual)
            actual = getattr(actual, "value", actual)
            expected = QDRANT_PAYLOAD_INDEX_SCHEMA[field]
            if not isinstance(actual, str) or actual.casefold() != expected:
                raise ValueError(
                    f"Qdrant payload index {field!r} has type {actual!r}, "
                    f"expected {expected!r}; explicit schema migration is required"
                )
        return missing

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
            self._mutate(lambda: self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            ))
        except CorpusAuthorityUnavailable:
            raise
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
