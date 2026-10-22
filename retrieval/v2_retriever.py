from __future__ import annotations

import hashlib
import logging
import math
import os
import random
import re
import threading
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import (
    EMBEDDING_MODEL_REVISION,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_DISTANCE,
    QDRANT_PAYLOAD_INDEX_SCHEMA,
    QDRANT_URL,
    QDRANT_VECTOR_NAME,
    REPOSITORY_EMBEDDING_DIM,
    REPOSITORY_EMBEDDING_MODEL,
    REPOSITORY_EMBEDDING_VERSION,
    REPOSITORY_FEATURE_SPEC_VERSION,
)
from embedding.vector_contract import (
    REPOSITORY_DISCOVERY_CHANNELS,
    REPOSITORY_COLLECTION_CONTRACT,
    REPOSITORY_SERVING_ELIGIBILITY_FIELD,
    REPOSITORY_SERVING_ELIGIBILITY_VERSION,
    USER_PROFILE_COLLECTION_CONTRACT,
    user_point_id,
    validate_embedding_vector,
)
from inference.feed_assembly import FeedAssemblySystem
from inference.feature_spec import FEATURE_ORDER
from inference.ranker_service import RankerService
from inference.runtime import RecommendationDeadlineExceeded, recommendation_timeout_seconds
from retrieval.metrics import record_candidate_read
from retrieval.config import TOTAL_CANDIDATE_POOL
from retrieval.runtime import RetrievalWork, validate_retrieval_runtime
from utils.deadline import CooperativeDeadlineExceeded, check_deadline, current_deadline, qdrant_timeout
from scripts.user_onboarding import TARGET_VECTOR_NAME, USER_PROFILES_COLLECTION

logger = logging.getLogger("pipeline.v2_retrieval")

_COUNTER_LIMIT = (2**63) - 1
_MINIMUM_QDRANT_SERVER_VERSION = (1, 18, 0)
_MAX_RETRIEVAL_ROUNDS = 3  # initial selection plus at most two refills
_CANDIDATE_READ_MULTIPLIER = 6
_FALLBACK_MESSAGES = {
    "HEAVY_UNAVAILABLE": "heavy ranker is unavailable",
    "USER_VECTOR_UNAVAILABLE": "user profile vector is unavailable",
    "NO_ELIGIBLE_CANDIDATES": "no eligible candidates were retrieved",
    "INVALID_HEAVY_OUTPUT": "heavy ranker returned an incomplete or invalid candidate set",
    "HEAVY_SCORING_FAILED": "heavy ranker scoring failed",
}

_DISCOVERY_SOURCES = {
    "trend": "trending",
    "activity": "active",
    "popularity": "popular",
    "freshness": "fresh",
}

SOCIAL_BOOST_FRACTION = 0.05
SOCIAL_BOOST_MIN = 0.005
SOCIAL_BOOST_MAX = 0.03
SOCIAL_BOOSTED_BATCH_FRACTION = 0.25

# Keep the recommendation hot path independent of the potentially larger
# feedback ledger stored on the same user point. These are the only profile
# fields retrieval/ranking consumes.
_USER_RETRIEVAL_PAYLOAD_FIELDS = [
    "skills",
    "tech_stack",
    "interests",
    "topics",
    "embedding_dim",
    "embedding_model",
    "embedding_model_revision",
]

_REPOSITORY_RETRIEVAL_PAYLOAD_FIELDS = [
    "repo_id",
    REPOSITORY_SERVING_ELIGIBILITY_FIELD,
    "content_version",
    "embedding_model",
    "embedding_model_revision",
    "embedding_version",
    "embedding_dim",
    "model_version",
    "feature_spec_version",
    "doc_quality",
    "code_health",
    "readme_length",
    "star_count",
    "fork_count",
    "open_issues_count",
    "pushed_days_ago",
    "activity_score",
    "trend_velocity",
    "delta_7d",
    "primary_language",
    "created_at",
]
_HEAVY_RANKER_PAYLOAD_FIELDS = ["languages", "topics", "tags"]


class RetrievalDependencyError(RuntimeError):
    """Raised when no bounded retrieval channel can reach Qdrant."""


@dataclass(frozen=True, slots=True)
class RankedRepository:
    repo_id: str
    score: float
    source: str


@dataclass(frozen=True, slots=True)
class RecommendationBatch:
    items: list[RankedRepository]
    model_version: str
    embedding_version: str
    ranker_applied: bool
    fallback_reason: str | None = None
    embedding_versions: tuple[str, ...] = ()
    served_ranker: str = "hybrid"
    heavy_ranker_selected: bool = False
    fallback_code: str | None = None
    retrieval_mode: str = "personalized"


@dataclass(frozen=True, slots=True)
class _EligibleRepositoryPoint:
    point: Any
    repo_id: str
    payload: dict[str, Any]
    vector: list[float] | None
    embedding_version: str


@dataclass(slots=True)
class _CandidateReadBudget:
    remaining: int
    deadline: float

    def reserve(self, requested: int) -> int:
        check_deadline()
        if time.monotonic() >= self.deadline:
            return 0
        count = min(requested, self.remaining)
        self.remaining -= count
        return count


def _read_point_ids(points: list) -> set[str | int]:
    """Exclude rejected point identities too, without trusting payload IDs."""
    ids: set[str | int] = set()
    for point in points:
        value = getattr(point, "id", None)
        if type(value) is int and 0 <= value < 2**64:
            ids.add(value)
        else:
            try:
                ids.add(str(uuid.UUID(str(value))))
            except ValueError:
                continue
    return ids


class QdrantV2Retriever:
    """V2 candidate retrieval plus fail-safe heavy-ranker serving."""

    def __init__(
        self,
        *,
        client: QdrantClient | None = None,
        repository_collection: str = QDRANT_COLLECTION_NAME,
        user_collection: str = USER_PROFILES_COLLECTION,
        max_candidates: int = TOTAL_CANDIDATE_POOL,
        assembler: FeedAssemblySystem | None = None,
        ranker: RankerService | None = None,
        heavy_ranker_enabled: bool | None = None,
        heavy_ranker_required: bool | None = None,
        heavy_ranker_traffic_percent: float | None = None,
        compatible_embedding_versions: set[str] | None = None,
        minimum_eligible_repositories: int | None = None,
        allow_unqualified_ranker: bool | None = None,
    ) -> None:
        validate_retrieval_runtime()
        self.qdrant_timeout_seconds = self._float_env(
            "QDRANT_TIMEOUT_SECONDS", default=10.0, minimum=1.0, maximum=120.0,
        )
        self.client = client or QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=self.qdrant_timeout_seconds,
        )
        self.repository_collection = repository_collection
        self.user_collection = user_collection
        self.max_candidates = max(50, min(max_candidates, 2_000))
        self.app_env = os.getenv("APP_ENV", "development").strip().casefold()
        if assembler is None:
            assembler = FeedAssemblySystem(
                explore_fraction=self._float_env(
                    "V2_EXPLORATION_FRACTION", default=1 / 3, minimum=0.0, maximum=0.5
                ),
                max_same_language=self._int_env(
                    "V2_MAX_SAME_LANGUAGE", default=5, minimum=1
                ),
            )
        self.assembler = assembler
        self.hybrid_model_version = os.getenv("ML_MODEL_VERSION", "qdrant-hybrid-v2")
        self.embedding_version = REPOSITORY_EMBEDDING_VERSION
        self.embedding_model = REPOSITORY_EMBEDDING_MODEL
        self.embedding_model_revision = os.getenv(
            "EMBEDDING_MODEL_REVISION", EMBEDDING_MODEL_REVISION
        ).strip()
        if not self.embedding_model_revision:
            raise ValueError("EMBEDDING_MODEL_REVISION must be a non-empty string")
        self.embedding_dim = REPOSITORY_EMBEDDING_DIM
        self.required_content_version = self._int_env(
            "V2_REQUIRED_CONTENT_VERSION", default=1, minimum=1
        )
        self.required_feature_spec_version = (
            os.getenv(
                "V2_REQUIRED_FEATURE_SPEC_VERSION",
                REPOSITORY_FEATURE_SPEC_VERSION,
            ).strip()
        )
        if not self.required_feature_spec_version:
            raise ValueError(
                "V2_REQUIRED_FEATURE_SPEC_VERSION must be a non-empty string"
            )
        if compatible_embedding_versions is None:
            compatible_embedding_versions = self._csv_env(
                "V2_COMPATIBLE_EMBEDDING_VERSIONS",
                default={self.embedding_version},
            )
        normalized_embedding_versions = {
            str(version).strip() for version in compatible_embedding_versions
        }
        if (
            any(
                not version or len(version) > 128
                for version in normalized_embedding_versions
            )
            or len(normalized_embedding_versions) > 8
        ):
            raise ValueError(
                "compatible embedding versions must contain 1 to 8 values "
                "of at most 128 characters"
            )
        self.compatible_embedding_versions = frozenset(
            normalized_embedding_versions
        )
        if not self.compatible_embedding_versions:
            raise ValueError("At least one compatible embedding version is required")
        if self.embedding_version not in self.compatible_embedding_versions:
            raise ValueError(
                "The configured repository embedding version must be included in "
                "V2_COMPATIBLE_EMBEDDING_VERSIONS"
            )
        self.allow_missing_embedding_revision = self._boolean_env(
            "V2_ALLOW_MISSING_EMBEDDING_REVISION", default=False
        )
        if minimum_eligible_repositories is None:
            minimum_eligible_repositories = self._int_env(
                "MIN_ELIGIBLE_REPOSITORIES",
                default=1 if self.app_env == "production" else 0,
                minimum=0,
            )
        if minimum_eligible_repositories < 0:
            raise ValueError("minimum_eligible_repositories must be non-negative")
        self.minimum_eligible_repositories = minimum_eligible_repositories
        self.user_collection_required = self._boolean_env(
            "V2_USER_COLLECTION_REQUIRED",
            default=True,
        )

        heavy_ranker_explicitly_disabled = heavy_ranker_enabled is False
        if heavy_ranker_enabled is None:
            heavy_ranker_enabled = self._boolean_env(
                "V2_HEAVY_RANKER_ENABLED", default=True
            )
        self.heavy_ranker_enabled = heavy_ranker_enabled
        if heavy_ranker_traffic_percent is None:
            heavy_ranker_traffic_percent = (
                self._float_env(
                    "V2_HEAVY_RANKER_TRAFFIC_PERCENT",
                    default=100.0,
                    minimum=0.0,
                    maximum=100.0,
                )
                if self.heavy_ranker_enabled
                else 0.0
            )
        if not 0.0 <= float(heavy_ranker_traffic_percent) <= 100.0:
            raise ValueError("heavy_ranker_traffic_percent must be between 0 and 100")
        self.heavy_ranker_traffic_percent = float(heavy_ranker_traffic_percent)
        if not self.heavy_ranker_enabled and self.heavy_ranker_traffic_percent > 0:
            raise ValueError(
                "V2_HEAVY_RANKER_TRAFFIC_PERCENT must be 0 when the heavy ranker is disabled"
            )
        if heavy_ranker_required is None:
            heavy_ranker_required = (
                False
                if heavy_ranker_explicitly_disabled
                else self._boolean_env(
                    "V2_HEAVY_RANKER_REQUIRED", default=self.heavy_ranker_enabled
                )
            )
        self.heavy_ranker_required = heavy_ranker_required
        if self.heavy_ranker_required and not self.heavy_ranker_enabled:
            raise ValueError(
                "V2_HEAVY_RANKER_REQUIRED cannot be true while the heavy ranker is disabled"
            )
        if self.heavy_ranker_required and self.heavy_ranker_traffic_percent != 100.0:
            raise ValueError(
                "V2_HEAVY_RANKER_TRAFFIC_PERCENT must be 100 when the heavy ranker is required"
            )
        if allow_unqualified_ranker is None:
            allow_unqualified_ranker = self._boolean_env(
                "V2_ALLOW_UNQUALIFIED_HEAVY_RANKER", default=True
            )
        if allow_unqualified_ranker and self.app_env == "production":
            raise ValueError(
                "V2_ALLOW_UNQUALIFIED_HEAVY_RANKER is a development-only override"
            )
        self.allow_unqualified_ranker = allow_unqualified_ranker
        self._counter_lock = threading.Lock()
        self._ranking_counters = {
            "requests_total": 0,
            "heavy_selected": 0,
            "heavy_served": 0,
            "hybrid_served": 0,
            **{f"fallback_{code.casefold()}": 0 for code in _FALLBACK_MESSAGES},
        }
        self.ranker: RankerService | None = None
        self.ranker_error: str | None = None
        self.heavy_model_version: str | None = None
        self.heavy_ranker_production_qualified = False
        if self.heavy_ranker_enabled:
            try:
                self.ranker = ranker or self._load_ranker()
                if not getattr(self.ranker, "ready", True):
                    raise RuntimeError("heavy ranker artifact is not loaded")
                self.heavy_ranker_production_qualified = bool(
                    getattr(self.ranker, "production_qualified", False)
                )
                if (
                    not self.heavy_ranker_production_qualified
                    and not self.allow_unqualified_ranker
                ):
                    raise RuntimeError(
                        "heavy ranker manifest is not production-qualified"
                    )
                ranker_version = str(
                    getattr(self.ranker, "model_version", "heavy-ranker-unknown")
                )
                self.heavy_model_version = ranker_version
            except Exception as exc:
                self.ranker = None
                self.ranker_error = "heavy ranker initialization failed"
                logger.error(
                    "V2 heavy ranker initialization failed "
                    "error_type=%s",
                    type(exc).__name__,
                )

        # Compatibility for callers that inspect the configured primary model.
        self.model_version = self.heavy_model_version or self.hybrid_model_version

    @staticmethod
    def _boolean_env(name: str, *, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        normalized = raw.strip().casefold()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{name} must be a boolean; got {raw!r}")

    @staticmethod
    def _int_env(name: str, *, default: int, minimum: int) -> int:
        raw = os.getenv(name)
        try:
            value = default if raw is None else int(raw.strip())
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        return value

    @staticmethod
    def _float_env(
        name: str,
        *,
        default: float,
        minimum: float,
        maximum: float,
    ) -> float:
        raw = os.getenv(name)
        try:
            value = default if raw is None else float(raw.strip())
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number") from exc
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")
        return value

    @staticmethod
    def _csv_env(name: str, *, default: set[str]) -> set[str]:
        raw = os.getenv(name)
        if raw is None:
            return set(default)
        values = {item.strip() for item in raw.split(",") if item.strip()}
        if not values:
            raise ValueError(f"{name} must contain at least one value")
        return values

    def _load_ranker(self) -> RankerService:
        inference_dir = Path(__file__).resolve().parents[1] / "inference"
        return RankerService(
            model_path=str(inference_dir / "heavy_ranker.pt"),
            scaler_path=str(inference_dir / "feature_scaler.json"),
            manifest_path=str(inference_dir / "model_manifest.json"),
            expected_embedding_versions=self.compatible_embedding_versions,
            expected_embedding_model=self.embedding_model,
            expected_embedding_model_revision=self.embedding_model_revision,
            require_production_manifest=not self.allow_unqualified_ranker,
        )

    @staticmethod
    def _canonical_id(point: Any) -> str | None:
        payload = point.payload or {}
        candidate = str(payload.get("repo_id") or point.id)
        try:
            canonical = str(uuid.UUID(candidate))
        except (ValueError, AttributeError):
            return None
        has_canonical_payload = str(payload.get("repo_id")) == canonical
        return canonical if str(point.id) == canonical and has_canonical_payload else None

    @staticmethod
    def _vector(value: Any, preferred: str | None = None) -> list[float] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            if preferred and preferred in value:
                return list(value[preferred])
            if len(value) == 1:
                return list(next(iter(value.values())))
            return None
        return list(value)

    def _eligible_filter(
        self, *, exclude_repo_ids: set[str | int] | None = None
    ) -> models.Filter:
        # Filtering in Qdrant happens before the retrieval limit is applied, so
        # stale pre-contract points cannot crowd valid backend repositories out.
        must: list[Any] = [
            models.FieldCondition(
                key=REPOSITORY_SERVING_ELIGIBILITY_FIELD,
                match=models.MatchValue(
                    value=REPOSITORY_SERVING_ELIGIBILITY_VERSION
                ),
            ),
            models.FieldCondition(
                key="content_version",
                range=models.Range(gte=self.required_content_version),
            ),
            models.FieldCondition(
                key="embedding_dim",
                match=models.MatchValue(value=self.embedding_dim),
            ),
            models.FieldCondition(
                key="embedding_model",
                match=models.MatchValue(value=self.embedding_model),
            ),
            models.FieldCondition(
                key="embedding_version",
                match=models.MatchAny(
                    any=sorted(self.compatible_embedding_versions)
                ),
            ),
            models.FieldCondition(
                key="model_version",
                match=models.MatchValue(value=self.embedding_model),
            ),
            models.FieldCondition(
                key="feature_spec_version",
                match=models.MatchValue(value=self.required_feature_spec_version),
            ),
        ]
        if not self.allow_missing_embedding_revision:
            must.append(
                models.FieldCondition(
                    key="embedding_model_revision",
                    match=models.MatchValue(value=self.embedding_model_revision),
                )
            )
        # The store uses canonical backend UUIDs as point IDs, and
        # _canonical_id rejects points whose payload and point ID disagree.
        # Exclude history before *every* channel's bounded selection so a
        # viewed top pool cannot hide eligible unseen repositories behind it.
        return models.Filter(
            must=must,
            must_not=(
                [models.HasIdCondition(has_id=sorted(exclude_repo_ids, key=str))]
                if exclude_repo_ids else None
            ),
        )

    def _eligible_repository_point(
        self,
        point: Any,
        cache: dict[tuple[str, bool], _EligibleRepositoryPoint | None] | None = None,
        *,
        require_vector: bool = True,
    ) -> _EligibleRepositoryPoint | None:
        point_id = str(getattr(point, "id", ""))
        cache_key = (point_id, require_vector)
        if cache is not None and cache_key in cache:
            return cache[cache_key]

        eligible: _EligibleRepositoryPoint | None = None
        try:
            repo_id = self._canonical_id(point)
            if repo_id is None:
                raise ValueError("repository identity is not canonical")
            payload = dict(point.payload or {})
            revision = payload.get("embedding_model_revision")
            if (
                payload.get(REPOSITORY_SERVING_ELIGIBILITY_FIELD)
                != REPOSITORY_SERVING_ELIGIBILITY_VERSION
            ):
                raise ValueError("repository serving eligibility is missing")
            if int(payload.get("content_version", 0)) < self.required_content_version:
                raise ValueError("content version is below the serving minimum")
            if int(payload.get("embedding_dim", 0)) != self.embedding_dim:
                raise ValueError("embedding dimension is incompatible")
            if str(payload.get("embedding_model")) != self.embedding_model:
                raise ValueError("embedding model is incompatible")
            if revision is None and self.allow_missing_embedding_revision:
                pass
            elif str(revision) != self.embedding_model_revision:
                raise ValueError("embedding model revision is incompatible")
            embedding_version = str(payload.get("embedding_version") or "")
            if embedding_version not in self.compatible_embedding_versions:
                raise ValueError("embedding version is incompatible")
            if str(payload.get("model_version")) != self.embedding_model:
                raise ValueError("repository model version is incompatible")

            if (
                str(payload.get("feature_spec_version"))
                != self.required_feature_spec_version
            ):
                raise ValueError("ranker feature specification is incompatible")
            for feature_name in FEATURE_ORDER:
                if feature_name == "skill_match_score":
                    continue
                if feature_name not in payload:
                    raise ValueError(f"ranker feature {feature_name} is missing")
                value = payload[feature_name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(f"ranker feature {feature_name} is invalid")

            vector = None
            if require_vector:
                vector = validate_embedding_vector(
                    self._vector(getattr(point, "vector", None), QDRANT_VECTOR_NAME),
                    expected_size=self.embedding_dim,
                    field_name=f"repository vector {repo_id}",
                )
            eligible = _EligibleRepositoryPoint(
                point=point,
                repo_id=repo_id,
                payload=payload,
                vector=vector,
                embedding_version=embedding_version,
            )
        except (TypeError, ValueError) as exc:
            logger.debug(
                "Rejected ineligible V2 repository point %s: %s",
                point_id or "<unknown>",
                exc,
            )

        if cache is not None:
            cache[cache_key] = eligible
        return eligible

    def _fetch_user_profile(self, user_id: str) -> list[Any]:
        return self.client.retrieve(
            collection_name=self.user_collection,
            ids=[user_point_id(user_id)],
            with_vectors=True,
            with_payload=_USER_RETRIEVAL_PAYLOAD_FIELDS,
            **qdrant_timeout(self.qdrant_timeout_seconds),
        )

    def _profile_from_points(
        self, user_id: str, points: list[Any],
    ) -> tuple[list[float] | None, dict[str, Any]]:
        canonical = user_point_id(user_id)
        if not points:
            return None, {}
        by_id = {str(point.id): point for point in points}
        point = by_id.get(canonical)
        if not point:
            return None, {}
        payload = dict(point.payload or {})
        try:
            vector = validate_embedding_vector(
                self._vector(point.vector, TARGET_VECTOR_NAME),
                expected_size=self.embedding_dim,
                field_name="user profile vector",
            )
            payload_dim = payload.get("embedding_dim")
            if payload_dim is not None and int(payload_dim) != self.embedding_dim:
                raise ValueError("user profile embedding dimension is incompatible")
            payload_model = payload.get("embedding_model")
            if payload_model is not None and str(payload_model) != self.embedding_model:
                raise ValueError("user profile embedding model is incompatible")
            payload_revision = payload.get("embedding_model_revision")
            if (
                payload_revision is None
                and self.allow_missing_embedding_revision
            ):
                pass
            elif str(payload_revision) != self.embedding_model_revision:
                raise ValueError("user profile embedding revision is incompatible")
            return vector, payload
        except (TypeError, ValueError) as exc:
            logger.warning("Ignoring incompatible user profile vector: %s", exc)
            return None, payload

    def _user_profile(self, user_id: str) -> tuple[list[float] | None, dict[str, Any]]:
        with RetrievalWork() as work:
            points = work.result(work.submit("profile", self._fetch_user_profile, user_id))
        return self._profile_from_points(user_id, points)

    def _user_vector(self, user_id: str) -> list[float] | None:
        """Compatibility wrapper retained for existing callers and tests."""
        vector, _ = self._user_profile(user_id)
        return vector

    @staticmethod
    def _profile_terms(payload: dict[str, Any]) -> list[str]:
        terms: list[str] = []
        for field_name in ("skills", "tech_stack", "interests", "topics"):
            raw = payload.get(field_name)
            values = raw if isinstance(raw, (list, tuple, set)) else [raw]
            for value in values:
                if value is not None and (text := str(value).strip()):
                    terms.append(text)
        return list(dict.fromkeys(terms))

    def _candidate_read_budget(self) -> _CandidateReadBudget:
        deadline = current_deadline()
        seconds = (
            deadline.remaining()
            if deadline is not None else recommendation_timeout_seconds()
        )
        return _CandidateReadBudget(
            self.max_candidates * _CANDIDATE_READ_MULTIPLIER,
            time.monotonic() + seconds,
        )

    def _fetch_semantic(
        self,
        vector: list[float],
        limit: int,
        *,
        include_vectors: bool,
        exclude_repo_ids: set[str | int] | None,
    ) -> Any:
        return self.client.query_points(
            collection_name=self.repository_collection,
            query=vector,
            using=QDRANT_VECTOR_NAME,
            query_filter=self._eligible_filter(exclude_repo_ids=exclude_repo_ids),
            limit=min(limit, self.max_candidates),
            with_payload=(
                _REPOSITORY_RETRIEVAL_PAYLOAD_FIELDS
                + (_HEAVY_RANKER_PAYLOAD_FIELDS if include_vectors else [])
            ),
            with_vectors=[QDRANT_VECTOR_NAME] if include_vectors else False,
            **qdrant_timeout(self.qdrant_timeout_seconds),
        )

    def _collect_semantic(
        self, response: Any,
        cache: dict[tuple[str, bool], _EligibleRepositoryPoint | None] | None,
        *, include_vectors: bool,
    ) -> list[tuple[_EligibleRepositoryPoint, float]]:
        eligible: list[tuple[_EligibleRepositoryPoint, float]] = []
        for point in response.points:
            candidate = self._eligible_repository_point(
                point,
                cache,
                require_vector=include_vectors,
            )
            if candidate is not None:
                eligible.append((candidate, float(point.score)))
        return eligible

    def _semantic(
        self,
        vector: list[float],
        limit: int,
        cache: dict[tuple[str, bool], _EligibleRepositoryPoint | None] | None = None,
        *,
        include_vectors: bool = True,
        exclude_repo_ids: set[str] | None = None,
        minimum_candidates: int | None = None,
        budget: _CandidateReadBudget | None = None,
    ) -> list[tuple[_EligibleRepositoryPoint, float]]:
        target = min(limit, self.max_candidates)
        minimum = min(
            target, minimum_candidates if minimum_candidates is not None else target
        )
        budget = budget or self._candidate_read_budget()
        excluded: set[str | int] = set(exclude_repo_ids or ())
        eligible: dict[str, tuple[_EligibleRepositoryPoint, float]] = {}
        with RetrievalWork() as work:
            for _ in range(_MAX_RETRIEVAL_ROUNDS):
                read_limit = budget.reserve(target - len(eligible))
                if not read_limit:
                    break
                try:
                    response = work.result(work.submit(
                        "semantic", self._fetch_semantic, vector, read_limit,
                        include_vectors=include_vectors, exclude_repo_ids=set(excluded),
                    ))
                except CooperativeDeadlineExceeded:
                    raise
                except Exception:
                    if not eligible:
                        raise
                    # A failed optional refill must not discard a successful read.
                    logger.warning("V2 semantic refill failed; retaining partial candidates")
                    break
                points = response.points
                before = len(eligible)
                for candidate, score in self._collect_semantic(
                    response, cache, include_vectors=include_vectors,
                ):
                    if candidate.repo_id not in excluded and math.isfinite(score):
                        eligible.setdefault(candidate.repo_id, (candidate, score))
                record_candidate_read("semantic", points, len(eligible) - before)
                if len(eligible) >= minimum or len(points) < read_limit:
                    break
                new_ids = _read_point_ids(points) - excluded
                if not new_ids:
                    break
                excluded.update(new_ids)
        return list(eligible.values())[:target]

    def _fetch_discovery_channel(
        self,
        channel: str,
        per_channel: int,
        *,
        include_vectors: bool,
        exclude_repo_ids: set[str | int] | None,
    ) -> list[Any]:
        records, _ = self.client.scroll(
            collection_name=self.repository_collection,
            scroll_filter=self._eligible_filter(exclude_repo_ids=exclude_repo_ids),
            limit=per_channel,
            order_by=models.OrderBy(
                key=REPOSITORY_DISCOVERY_CHANNELS[channel],
                direction=models.Direction.DESC,
            ),
            with_payload=(
                _REPOSITORY_RETRIEVAL_PAYLOAD_FIELDS
                + (_HEAVY_RANKER_PAYLOAD_FIELDS if include_vectors else [])
            ),
            with_vectors=[QDRANT_VECTOR_NAME] if include_vectors else False,
            **qdrant_timeout(self.qdrant_timeout_seconds),
        )
        return records

    def _start_discovery(
        self, work: RetrievalWork, limit: int, *, include_vectors: bool,
        exclude_repo_ids: set[str | int] | None,
        budget: _CandidateReadBudget,
    ) -> list[tuple[str, Future]]:
        target = min(limit, self.max_candidates)
        per_channel = min(self.max_candidates, max(25, math.ceil(target / 2)))
        futures = []
        for channel in _DISCOVERY_SOURCES:
            read_limit = budget.reserve(per_channel)
            if not read_limit:
                break
            futures.append((channel, work.submit(
                channel, self._fetch_discovery_channel, channel, read_limit,
                include_vectors=include_vectors, exclude_repo_ids=exclude_repo_ids,
            )))
        return futures

    def _collect_discovery(
        self,
        work: RetrievalWork,
        futures: list[tuple[str, Future]],
        limit: int,
        cache: dict[tuple[str, bool], _EligibleRepositoryPoint | None] | None = None,
        *,
        include_vectors: bool = True,
        exclude_repo_ids: set[str] | None = None,
        minimum_candidates: int | None = None,
        existing_repo_ids: set[str] | None = None,
        budget: _CandidateReadBudget | None = None,
    ) -> list[tuple[_EligibleRepositoryPoint, str]]:
        target = min(limit, self.max_candidates)
        minimum = min(
            target, minimum_candidates if minimum_candidates is not None else target
        )
        budget = budget or self._candidate_read_budget()
        excluded: set[str | int] = set(exclude_repo_ids or ())
        existing = set(existing_repo_ids or ())
        per_channel = min(self.max_candidates, max(25, math.ceil(target / 2)))
        pending = futures
        read_limits: dict[str, int] = {}
        combined: dict[str, tuple[_EligibleRepositoryPoint, str]] = {}
        successful_channels = 0
        for round_index in range(_MAX_RETRIEVAL_ROUNDS):
            channel_points: list[list[tuple[_EligibleRepositoryPoint, str]]] = []
            next_channels: list[str] = []
            round_records: list = []
            round_fresh_ids: set[str] = set()
            for channel, future in pending:
                read_limit = read_limits.get(channel, per_channel)
                try:
                    records = work.result(future)
                except CooperativeDeadlineExceeded:
                    raise
                except Exception as exc:
                    logger.error(
                        "V2 discovery channel failed",
                        extra={"retrieval_context": {
                            "channel": channel, "error_type": type(exc).__name__,
                        }},
                    )
                    continue
                successful_channels += 1
                round_records.extend(records)
                eligible_points = []
                fresh = set()
                for point in records:
                    candidate = self._eligible_repository_point(
                        point, cache, require_vector=include_vectors
                    )
                    if candidate is not None and candidate.repo_id not in excluded:
                        eligible_points.append((candidate, _DISCOVERY_SOURCES[channel]))
                        if (
                            candidate.repo_id not in combined
                            and candidate.repo_id not in existing
                            and candidate.repo_id not in round_fresh_ids
                        ):
                            fresh.add(candidate.repo_id)
                round_fresh_ids.update(fresh)
                record_candidate_read(channel, records, len(fresh))
                channel_points.append(eligible_points)
                if len(records) >= read_limit:
                    next_channels.append(channel)

            # Retain the original round-robin source balance and blending with
            # semantic candidates. Exclusions are request-local, never cached.
            round_size = max((len(points) for points in channel_points), default=0)
            for index in range(round_size):
                for points in channel_points:
                    if index < len(points) and len(combined) < target:
                        point, source = points[index]
                        combined.setdefault(point.repo_id, (point, source))
            if (
                len(existing | combined.keys()) >= minimum
                or len(combined) >= target
                or not next_channels
                or round_index + 1 == _MAX_RETRIEVAL_ROUNDS
            ):
                break
            new_ids = _read_point_ids(round_records) - excluded
            if not new_ids:
                break
            excluded.update(new_ids)
            pending = []
            read_limits = {}
            for channel in next_channels:
                read_limit = budget.reserve(per_channel)
                if not read_limit:
                    break
                read_limits[channel] = read_limit
                pending.append((channel, work.submit(
                    channel, self._fetch_discovery_channel, channel, read_limit,
                    include_vectors=include_vectors, exclude_repo_ids=set(excluded),
                )))

        if successful_channels == 0:
            if time.monotonic() >= budget.deadline:
                raise RecommendationDeadlineExceeded("candidate retrieval deadline exceeded")
            raise RetrievalDependencyError(
                "all bounded Qdrant discovery channels were unavailable"
            )
        return list(combined.values())

    def _discovery(
        self, limit: int,
        cache: dict[tuple[str, bool], _EligibleRepositoryPoint | None] | None = None,
        *, include_vectors: bool = True, exclude_repo_ids: set[str] | None = None,
        minimum_candidates: int | None = None,
        existing_repo_ids: set[str] | None = None,
        budget: _CandidateReadBudget | None = None,
    ) -> list[tuple[_EligibleRepositoryPoint, str]]:
        budget = budget or self._candidate_read_budget()
        with RetrievalWork() as work:
            futures = self._start_discovery(
                work, limit, include_vectors=include_vectors,
                exclude_repo_ids=exclude_repo_ids, budget=budget,
            )
            return self._collect_discovery(
                work, futures, limit, cache, include_vectors=include_vectors,
                exclude_repo_ids=exclude_repo_ids, minimum_candidates=minimum_candidates,
                existing_repo_ids=existing_repo_ids, budget=budget,
            )

    @staticmethod
    def _number(value: Any, *, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if math.isfinite(number) else default

    @staticmethod
    def _discovery_score(payload: dict[str, Any]) -> tuple[float, str]:
        stars = max(0, int(QdrantV2Retriever._number(payload.get("star_count"))))
        velocity = max(
            0.0,
            QdrantV2Retriever._number(
                payload.get("trend_velocity") or payload.get("delta_7d")
            ),
        )
        activity = max(
            0.0, QdrantV2Retriever._number(payload.get("activity_score"))
        )
        raw_pushed_days = payload.get("pushed_days_ago")
        pushed_days = (
            999
            if raw_pushed_days is None
            else max(0, int(QdrantV2Retriever._number(raw_pushed_days, default=999)))
        )
        freshness = math.exp(-pushed_days / 60)
        score = (
            0.35 * math.log1p(stars)
            + 0.35 * math.log1p(velocity)
            + 0.2 * activity
            + 0.1 * freshness
        )
        source = (
            "trending"
            if velocity > 0
            else "fresh"
            if pushed_days <= 30
            else "popular"
        )
        return score, source

    def _heavy_rank(
        self,
        *,
        user_vector: list[float] | None,
        user_profile: dict[str, Any],
        ranked: list[RankedRepository],
        metadata: dict[str, dict[str, Any]],
        vectors: dict[str, list[float] | None],
    ) -> tuple[list[RankedRepository] | None, str | None, str | None]:
        if self.ranker is None:
            code = "HEAVY_UNAVAILABLE"
            return None, code, _FALLBACK_MESSAGES[code]
        if user_vector is None:
            code = "USER_VECTOR_UNAVAILABLE"
            return None, code, _FALLBACK_MESSAGES[code]
        if not ranked:
            code = "NO_ELIGIBLE_CANDIDATES"
            return None, code, _FALLBACK_MESSAGES[code]

        ranker_candidates: list[dict[str, Any]] = []
        for item in ranked:
            payload = metadata.get(item.repo_id, {})
            ranker_candidates.append(
                {
                    **payload,
                    "id": item.repo_id,
                    "embedding": vectors.get(item.repo_id),
                    "languages": payload.get("languages")
                    or ([payload["primary_language"]] if payload.get("primary_language") else []),
                    "topics": payload.get("topics") or [],
                    "tags": payload.get("tags") or [],
                }
            )

        try:
            # Prefer compact serving results, retaining compatibility with
            # older/custom rankers that expose only the diagnostic API.
            score_batch = getattr(self.ranker, "score_batch_compact", None)
            if not callable(score_batch):
                score_batch = self.ranker.score_batch
            check_deadline()
            results = score_batch(
                user_vector,
                self._profile_terms(user_profile),
                ranker_candidates,
            )
            expected_ids = {item.repo_id for item in ranked}
            returned_ids = [str(result.get("repo_id")) for result in results]
            if (
                len(results) != len(ranked)
                or len(set(returned_ids)) != len(returned_ids)
                or set(returned_ids) != expected_ids
                or any(
                    not math.isfinite(self._number(result.get("final_score"), default=math.nan))
                    for result in results
                )
            ):
                code = "INVALID_HEAVY_OUTPUT"
                return None, code, _FALLBACK_MESSAGES[code]
            sources = {item.repo_id: item.source for item in ranked}
            return (
                [
                    RankedRepository(
                        repo_id=str(result["repo_id"]),
                        score=float(result["final_score"]),
                        source=sources[str(result["repo_id"])],
                    )
                    for result in results
                ],
                None,
                None,
            )
        except CooperativeDeadlineExceeded:
            raise
        except Exception as exc:
            logger.error(
                "V2 heavy ranking failed for one request",
                extra={
                    "ranking_context": {"error_type": type(exc).__name__}
                },
            )
            code = "HEAVY_SCORING_FAILED"
            return None, code, _FALLBACK_MESSAGES[code]

    def _heavy_ranker_selected(self, user_id: str) -> bool:
        if not self.heavy_ranker_enabled or self.heavy_ranker_traffic_percent <= 0:
            return False
        if self.heavy_ranker_traffic_percent >= 100:
            return True
        salt = os.getenv("V2_HEAVY_RANKER_CANARY_SALT", "v2-heavy-ranker")
        digest = hashlib.blake2b(
            f"{salt}:{user_id}".encode("utf-8"), digest_size=8
        ).digest()
        basis_point = int.from_bytes(digest, "big") % 10_000
        return basis_point < round(self.heavy_ranker_traffic_percent * 100)

    def _record_ranking_result(
        self,
        *,
        heavy_selected: bool,
        ranker_applied: bool,
        fallback_code: str | None,
        served: bool = True,
    ) -> None:
        keys = ["requests_total"]
        if heavy_selected:
            keys.append("heavy_selected")
        if served:
            keys.append("heavy_served" if ranker_applied else "hybrid_served")
        if fallback_code in _FALLBACK_MESSAGES:
            keys.append(f"fallback_{fallback_code.casefold()}")
        with self._counter_lock:
            for key in keys:
                self._ranking_counters[key] = min(
                    _COUNTER_LIMIT, self._ranking_counters[key] + 1
                )

    def _ranking_counter_snapshot(self) -> dict[str, int]:
        with self._counter_lock:
            return dict(self._ranking_counters)

    @staticmethod
    def _served_embedding_versions(
        items: list[RankedRepository],
        metadata: dict[str, dict[str, Any]],
    ) -> tuple[str, tuple[str, ...]]:
        versions = tuple(
            sorted(
                {
                    str(metadata[item.repo_id]["embedding_version"])
                    for item in items
                    if item.repo_id in metadata
                    and metadata[item.repo_id].get("embedding_version")
                }
            )
        )
        if not versions:
            return "none", ()
        if len(versions) == 1:
            return versions[0], versions
        return f"compatible-mixed:{','.join(versions)}", versions

    @staticmethod
    def _apply_social_boost(
        ranked: list[RankedRepository],
        social_repo_ids: set[str],
        limit: int,
    ) -> list[RankedRepository]:
        """Apply a small, capped boost after ranking without injecting candidates."""
        if not ranked or not social_repo_ids or limit <= 0:
            return ranked
        eligible = [item for item in ranked if item.repo_id in social_repo_ids]
        if not eligible:
            return ranked

        output_size = min(limit, len(ranked))
        boost_cap = max(1, int(output_size * SOCIAL_BOOSTED_BATCH_FRACTION))
        boosted_ids = {item.repo_id for item in eligible[:boost_cap]}
        scores = [item.score for item in ranked]
        score_span = max(scores) - min(scores)
        boost = min(
            SOCIAL_BOOST_MAX,
            max(SOCIAL_BOOST_MIN, score_span * SOCIAL_BOOST_FRACTION),
        )
        positions = {item.repo_id: index for index, item in enumerate(ranked)}
        adjusted = [
            RankedRepository(
                repo_id=item.repo_id,
                score=item.score + boost if item.repo_id in boosted_ids else item.score,
                source=item.source,
            )
            for item in ranked
        ]
        return sorted(
            adjusted,
            key=lambda item: (-item.score, positions[item.repo_id]),
        )

    def recommend_batch(
        self,
        user_id: str,
        limit: int,
        exclude_repo_ids: list[str],
        generation_seed: str | None = None,
        cold_start: bool = False,
        social_repo_ids: list[str] | None = None,
    ) -> RecommendationBatch:
        check_deadline()
        excluded = {str(uuid.UUID(item)) for item in exclude_repo_ids}
        candidates: dict[str, RankedRepository] = {}
        metadata: dict[str, dict[str, Any]] = {}
        vectors: dict[str, list[float] | None] = {}
        eligibility_cache: dict[
            tuple[str, bool], _EligibleRepositoryPoint | None
        ] = {}
        budget = self._candidate_read_budget()
        heavy_selected = self._heavy_ranker_selected(user_id)
        with RetrievalWork() as work:
            profile_future = work.submit("profile", self._fetch_user_profile, user_id)
            discovery_futures = self._start_discovery(
                work, max(limit * 8, 200),
                # We cannot know profile validity yet. Fetch the heavy projection
                # speculatively, but preserve post-profile vector eligibility.
                include_vectors=bool(heavy_selected and self.ranker is not None),
                exclude_repo_ids=excluded, budget=budget,
            )
            user_vector, user_profile = self._profile_from_points(
                user_id, work.result(profile_future),
            )
            if self.heavy_ranker_required and not heavy_selected:
                raise RetrievalDependencyError(
                    "the required V2 heavy ranker was not selected"
                )
            include_candidate_vectors = bool(
                heavy_selected and self.ranker is not None and user_vector is not None
            )
            retrieval_mode = (
                "cold_start_profile"
                if cold_start and user_vector is not None
                else "cold_start_discovery"
                if cold_start
                else "personalized"
                if user_vector is not None
                else "profile_missing_discovery"
            )
            if user_vector:
                try:
                    semantic = self._semantic(
                        user_vector, max(limit * 5, 100), eligibility_cache,
                        include_vectors=include_candidate_vectors,
                        exclude_repo_ids=excluded, minimum_candidates=limit,
                        budget=budget,
                    )
                except CooperativeDeadlineExceeded:
                    raise
                except Exception as exc:
                    logger.error(
                        "V2 semantic retrieval failed; continuing with discovery",
                        extra={
                            "retrieval_context": {
                                "channel": "semantic",
                                "error_type": type(exc).__name__,
                            }
                        },
                    )
                    semantic = []
                for point, score in semantic:
                    repo_id = point.repo_id
                    if repo_id not in excluded and math.isfinite(score):
                        candidates[repo_id] = RankedRepository(repo_id, score, "semantic")
                        metadata[repo_id] = point.payload
                        vectors[repo_id] = point.vector

            discovery: list[tuple[str, float, str, dict[str, Any], list[float] | None]] = []
            try:
                discovery_points = self._collect_discovery(
                    work, discovery_futures, max(limit * 8, 200), eligibility_cache,
                    include_vectors=include_candidate_vectors,
                    exclude_repo_ids=excluded, minimum_candidates=limit,
                    existing_repo_ids=set(candidates), budget=budget,
                )
            except RetrievalDependencyError:
                if not candidates:
                    raise
                discovery_points = []
        for point, channel_source in discovery_points:
            repo_id = point.repo_id
            if repo_id in excluded:
                continue
            payload = point.payload
            score, inferred_source = self._discovery_score(payload)
            if math.isfinite(score):
                discovery.append(
                    (
                        repo_id,
                        score,
                        channel_source or inferred_source,
                        payload,
                        point.vector,
                    )
                )

        maximum_discovery = max((item[1] for item in discovery), default=0.0)
        for repo_id, score, source, payload, vector in discovery:
            normalized_discovery = (
                score / maximum_discovery if maximum_discovery > 0 else 0.0
            )
            current = candidates.get(repo_id)
            discovery_weight = 0.35 if cold_start else 0.15
            combined = discovery_weight * normalized_discovery
            if current is not None:
                combined += current.score
            if math.isfinite(combined):
                candidates[repo_id] = RankedRepository(
                    repo_id, combined, current.source if current else source
                )
                metadata[repo_id] = payload
                vectors[repo_id] = vector or vectors.get(repo_id)

        check_deadline()
        hybrid_ranked = sorted(
            candidates.values(), key=lambda item: (-item.score, item.repo_id)
        )[: self.max_candidates]
        heavy_ranked: list[RankedRepository] | None = None
        fallback_code: str | None = None
        fallback_reason: str | None = None
        if heavy_selected:
            heavy_ranked, fallback_code, fallback_reason = self._heavy_rank(
                user_vector=user_vector,
                user_profile=user_profile,
                ranked=hybrid_ranked,
                metadata=metadata,
                vectors=vectors,
            )
        if self.heavy_ranker_required and heavy_ranked is None:
            self._record_ranking_result(
                heavy_selected=heavy_selected,
                ranker_applied=False,
                fallback_code=fallback_code,
                served=False,
            )
            raise RetrievalDependencyError(
                "the required V2 heavy ranker could not serve this request: "
                f"{fallback_reason or 'unknown ranker failure'}"
            )
        ranker_applied = heavy_ranked is not None
        ranked = heavy_ranked if heavy_ranked is not None else hybrid_ranked
        ranked = self._apply_social_boost(
            ranked,
            {str(uuid.UUID(repo_id)) for repo_id in (social_repo_ids or [])},
            limit,
        )
        model_version = (
            self.heavy_model_version
            if ranker_applied and self.heavy_model_version
            else self.hybrid_model_version
        )

        assembly_input = [
            {
                "repo_id": item.repo_id,
                "score": item.score,
                "final_score": item.score,
                "source": item.source,
                "primary_language": metadata.get(item.repo_id, {}).get(
                    "primary_language"
                ),
                "created_at": metadata.get(item.repo_id, {}).get("created_at"),
            }
            for item in ranked
        ]
        check_deadline()
        shaped = self.assembler.shape_batch(
            assembly_input,
            seen_repo_ids=excluded,
            randomizer=random.Random(generation_seed),
            generation_id=generation_seed,
            target_size=limit,
            input_is_unique=True,
        )
        items = [
            RankedRepository(
                repo_id=str(item["repo_id"]),
                score=round(float(item["final_score"]), 6),
                source=str(item["source"]),
            )
            for item in shaped
        ]
        served_embedding_version, served_embedding_versions = (
            self._served_embedding_versions(items, metadata)
        )
        self._record_ranking_result(
            heavy_selected=heavy_selected,
            ranker_applied=ranker_applied,
            fallback_code=fallback_code,
        )
        return RecommendationBatch(
            items=items,
            model_version=model_version,
            embedding_version=served_embedding_version,
            ranker_applied=ranker_applied,
            fallback_reason=fallback_reason,
            embedding_versions=served_embedding_versions,
            served_ranker="heavy" if ranker_applied else "hybrid",
            heavy_ranker_selected=heavy_selected,
            fallback_code=fallback_code,
            retrieval_mode=retrieval_mode,
        )

    def recommend(
        self,
        user_id: str,
        limit: int,
        exclude_repo_ids: list[str],
        generation_seed: str | None = None,
        cold_start: bool = False,
        social_repo_ids: list[str] | None = None,
    ) -> list[RankedRepository]:
        """Compatibility wrapper returning only the recommendation items."""
        return self.recommend_batch(
            user_id,
            limit,
            exclude_repo_ids,
            generation_seed,
            cold_start,
            social_repo_ids,
        ).items

    @staticmethod
    def _qdrant_server_version(raw_version: Any) -> tuple[int, int, int]:
        match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", str(raw_version))
        if match is None:
            raise RuntimeError("Qdrant server returned an invalid version")
        return tuple(int(part) for part in match.groups())

    @staticmethod
    def _validate_collection_contract(
        info: Any,
        *,
        collection_name: str,
        vector_name: str | None,
        expected_size: int,
        expected_distance: str,
    ) -> None:
        try:
            vectors = info.config.params.vectors
        except AttributeError as exc:
            raise RuntimeError(
                f"Qdrant collection {collection_name!r} has no vector configuration"
            ) from exc

        if vector_name is None:
            if isinstance(vectors, Mapping):
                raise RuntimeError(
                    f"Qdrant collection {collection_name!r} must use an unnamed vector"
                )
            vector_config = vectors
        else:
            if not isinstance(vectors, Mapping) or vector_name not in vectors:
                raise RuntimeError(
                    f"Qdrant collection {collection_name!r} does not define "
                    f"vector {vector_name!r}"
                )
            vector_config = vectors[vector_name]

        if int(getattr(vector_config, "size", 0)) != expected_size:
            raise RuntimeError(
                f"Qdrant collection {collection_name!r} has an incompatible vector size"
            )
        actual_distance = getattr(
            getattr(vector_config, "distance", None),
            "value",
            getattr(vector_config, "distance", None),
        )
        if str(actual_distance).casefold() != str(expected_distance).casefold():
            raise RuntimeError(
                f"Qdrant collection {collection_name!r} has an incompatible distance"
            )

    @staticmethod
    def _validate_repository_payload_indexes(info: Any) -> None:
        payload_schema = getattr(info, "payload_schema", None)
        if not isinstance(payload_schema, Mapping):
            raise RuntimeError("Qdrant repository payload index schema is unavailable")
        for field, expected_type in QDRANT_PAYLOAD_INDEX_SCHEMA.items():
            actual = payload_schema.get(field)
            if actual is None:
                raise RuntimeError(
                    f"Qdrant repository payload index {field!r} is missing"
                )
            if isinstance(actual, Mapping):
                actual = actual.get("data_type")
            else:
                actual = getattr(actual, "data_type", actual)
            actual = getattr(actual, "value", actual)
            if str(actual).casefold() != expected_type:
                raise RuntimeError(
                    f"Qdrant repository payload index {field!r} has an incompatible type"
                )

    def health(self) -> dict[str, Any]:
        if self.heavy_ranker_required and self.ranker is None:
            raise RuntimeError(
                "V2 heavy ranker is required but unavailable: "
                f"{self.ranker_error or 'unknown initialization failure'}"
            )
        version_info = self.client.info()
        qdrant_server_version = str(getattr(version_info, "version", ""))
        if (
            self._qdrant_server_version(qdrant_server_version)
            < _MINIMUM_QDRANT_SERVER_VERSION
        ):
            raise RuntimeError(
                "Qdrant server 1.18.0 or newer is required for conditional writes"
            )
        info = self.client.get_collection(self.repository_collection)
        self._validate_collection_contract(
            info,
            collection_name=self.repository_collection,
            vector_name=QDRANT_VECTOR_NAME,
            expected_size=REPOSITORY_COLLECTION_CONTRACT.vector_size,
            expected_distance=QDRANT_DISTANCE,
        )
        self._validate_repository_payload_indexes(info)
        user_collection_contract = "not_required"
        if self.user_collection_required:
            user_info = self.client.get_collection(self.user_collection)
            self._validate_collection_contract(
                user_info,
                collection_name=self.user_collection,
                vector_name=TARGET_VECTOR_NAME,
                expected_size=USER_PROFILE_COLLECTION_CONTRACT.vector_size,
                expected_distance=QDRANT_DISTANCE,
            )
            user_collection_contract = "healthy"

        health: dict[str, Any] = {
            "qdrant": "healthy",
            "qdrant_server_version": qdrant_server_version,
            "minimum_qdrant_server_version": "1.18.0",
            "repository_points": int(info.points_count or 0),
            "repository_collection_contract": "healthy",
            "user_collection_contract": user_collection_contract,
            "embedding_model": self.embedding_model,
            "embedding_model_revision": self.embedding_model_revision,
            "configured_embedding_version": self.embedding_version,
            "compatible_embedding_versions": sorted(
                self.compatible_embedding_versions
            ),
            "required_content_version": self.required_content_version,
            "required_feature_spec_version": self.required_feature_spec_version,
            "serving_eligibility_version": (
                REPOSITORY_SERVING_ELIGIBILITY_VERSION
            ),
            "serving_eligibility_evidence": "validated_vector_at_atomic_upsert",
            "allow_missing_embedding_revision": self.allow_missing_embedding_revision,
            "minimum_eligible_repository_points": self.minimum_eligible_repositories,
            "heavy_ranker_enabled": self.heavy_ranker_enabled,
            "heavy_ranker_required": self.heavy_ranker_required,
            "heavy_ranker_ready": self.ranker is not None,
            "heavy_ranker_production_qualified": self.heavy_ranker_production_qualified,
            "heavy_ranker_traffic_percent": self.heavy_ranker_traffic_percent,
            "heavy_ranker_model_version": self.heavy_model_version,
            "hybrid_fallback_model_version": self.hybrid_model_version,
            "ranking_counters": self._ranking_counter_snapshot(),
        }
        if self.ranker_error:
            health["heavy_ranker_error"] = self.ranker_error
        eligible = self.client.count(
            collection_name=self.repository_collection,
            count_filter=self._eligible_filter(),
            # Every contract field in this filter has a payload index. Exact
            # counting therefore gives the deployment gate a truthful corpus
            # minimum without scrolling repository payloads or vectors.
            exact=True,
        )
        eligible_count = int(eligible.count or 0)
        health["eligible_repository_points"] = eligible_count
        health["eligible_repository_count_exact"] = True
        if eligible_count < self.minimum_eligible_repositories:
            raise RuntimeError(
                "Eligible repository corpus is below the configured minimum: "
                f"{eligible_count} < {self.minimum_eligible_repositories}"
            )
        return health
