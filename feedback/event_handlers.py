"""Qdrant-only feedback state transitions and user-vector learning."""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Mapping, Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from embedding.vector_contract import repository_point_id, user_point_id

from .interactions import get_interaction, normalize_interaction
from .settings import FeedbackSettings

logger = logging.getLogger("pipeline.feedback")
LATENT_KEY = "feedback_latent_vector"
ADJUSTMENTS_KEY = "feedback_adjustments"
PROCESSED_KEY = "feedback_processed_events"
APPLIED_SIGNALS_KEY = "feedback_applied_signals"


def dwell_alpha(
    dwell_seconds: float,
    *,
    minimum_seconds: float = 3.0,
    full_credit_seconds: float = 300.0,
    maximum_alpha: float = 0.15,
) -> float | None:
    """Return a linear, bounded dwell alpha, or None below the threshold."""
    value = float(dwell_seconds)
    if not math.isfinite(value) or value < 0:
        raise ValueError("dwell_seconds must be finite and non-negative")
    if full_credit_seconds <= minimum_seconds:
        raise ValueError("full_credit_seconds must exceed minimum_seconds")
    if value < minimum_seconds:
        return None
    progress = (value - minimum_seconds) / (full_credit_seconds - minimum_seconds)
    return min(maximum_alpha, maximum_alpha * max(0.0, progress))


def _dwell_alpha(dwell_seconds: float) -> float | None:
    settings = FeedbackSettings.from_env()
    return dwell_alpha(
        dwell_seconds,
        minimum_seconds=settings.dwell_min_seconds,
        full_credit_seconds=settings.dwell_full_credit_seconds,
        maximum_alpha=settings.dwell_max_alpha,
    )


def _vector(value: Sequence[float], dimension: int, *, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or len(result) != dimension:
        raise ValueError(f"{label} must contain exactly {dimension} values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def normalize_vector(value: Sequence[float], dimension: int | None = None) -> list[float]:
    expected = dimension if dimension is not None else len(value)
    result = _vector(value, expected, label="vector")
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("vector must have a finite, non-zero L2 norm")
    return (result / norm).tolist()


def vector_delta(
    latent_user_vector: Sequence[float], repository_vector: Sequence[float], alpha: float
) -> list[float]:
    """Implement alpha * (repository - user)."""
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    dimension = len(latent_user_vector)
    user = _vector(latent_user_vector, dimension, label="latent user vector")
    repository = _vector(repository_vector, dimension, label="repository vector")
    return (alpha * (repository - user)).tolist()


def shift_vector(
    user_vec: Sequence[float], repo_vec: Sequence[float], alpha: float
) -> list[float]:
    delta = np.asarray(vector_delta(user_vec, repo_vec, alpha), dtype=np.float64)
    return normalize_vector(np.asarray(user_vec, dtype=np.float64) + delta)


def _point_vector(
    point: Any, configured_name: str | None, *, label: str
) -> tuple[list[float], str | None]:
    value = point.vector
    if isinstance(value, Mapping):
        if configured_name:
            if configured_name not in value:
                raise ValueError(f"{label} does not contain named vector {configured_name!r}")
            return list(value[configured_name]), configured_name
        if len(value) != 1:
            raise ValueError(f"{label} has ambiguous named vectors; configure the vector name")
        name, vector = next(iter(value.items()))
        return list(vector), str(name)
    return list(value), None


def _payload_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Return a mutable payload mapping or reject corrupted state."""
    value = payload.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return copy.deepcopy(dict(value))


def _processed_events(payload: Mapping[str, Any]) -> list[str]:
    value = payload.get(PROCESSED_KEY, [])
    if not isinstance(value, list) or any(
        not isinstance(event_id, str) or not event_id for event_id in value
    ):
        raise ValueError(f"{PROCESSED_KEY} must be a list of non-empty strings")
    return list(value)


def _feedback_adjustments(
    payload: Mapping[str, Any], dimension: int
) -> dict[str, dict[str, Any]]:
    adjustments = _payload_mapping(payload, ADJUSTMENTS_KEY)
    validated: dict[str, dict[str, Any]] = {}
    for repo_id, repo_state in adjustments.items():
        if not isinstance(repo_id, str) or not repo_id:
            raise ValueError(f"{ADJUSTMENTS_KEY} repository keys must be non-empty strings")
        if not isinstance(repo_state, Mapping):
            raise ValueError(f"{ADJUSTMENTS_KEY}[{repo_id!r}] must be an object")
        validated_state: dict[str, Any] = {}
        for family, stored in repo_state.items():
            if not isinstance(family, str) or not family:
                raise ValueError(f"{ADJUSTMENTS_KEY} family keys must be non-empty strings")
            if not isinstance(stored, Mapping):
                raise ValueError(
                    f"{ADJUSTMENTS_KEY}[{repo_id!r}][{family!r}] must be an object"
                )
            action = stored.get("action")
            if not isinstance(action, str) or not action:
                raise ValueError(
                    f"{ADJUSTMENTS_KEY}[{repo_id!r}][{family!r}].action "
                    "must be a non-empty string"
                )
            _vector(
                stored.get("delta", []),
                dimension,
                label=f"{ADJUSTMENTS_KEY}[{repo_id!r}][{family!r}].delta",
            )
            validated_state[family] = copy.deepcopy(dict(stored))
        validated[repo_id] = validated_state
    return validated


def _applied_signals(payload: Mapping[str, Any]) -> dict[str, list[str]]:
    signals = _payload_mapping(payload, APPLIED_SIGNALS_KEY)
    validated: dict[str, list[str]] = {}
    for repo_id, actions in signals.items():
        if not isinstance(repo_id, str) or not repo_id:
            raise ValueError(
                f"{APPLIED_SIGNALS_KEY} repository keys must be non-empty strings"
            )
        if not isinstance(actions, list) or any(
            not isinstance(action, str) or not action for action in actions
        ):
            raise ValueError(
                f"{APPLIED_SIGNALS_KEY}[{repo_id!r}] must be a list of non-empty strings"
            )
        validated[repo_id] = list(actions)
    return validated


class FeedbackHandler:
    """Apply one idempotent feedback event to an existing Qdrant user point.

    The point payload owns the unnormalized latent vector, reversible deltas,
    and a bounded replay guard. The consumer serializes updates per user.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient | None = None,
        settings: FeedbackSettings | None = None,
        *,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ) -> None:
        self.settings = settings or FeedbackSettings.from_env()
        self.qdrant = qdrant_client or QdrantClient(
            url=qdrant_url or self.settings.qdrant_url,
            api_key=qdrant_api_key or self.settings.qdrant_api_key,
            timeout=30.0,
        )

    def healthy(self) -> bool:
        self.qdrant.get_collections()
        return True

    def handle_feedback(
        self,
        user_id: str,
        repo_id: str,
        action: str,
        *,
        event_id: str | None = None,
        dwell_seconds: float | None = None,
        message_id: str | None = None,
    ) -> bool:
        logical_id = event_id or message_id
        if not logical_id:
            raise ValueError("event_id is required for idempotent processing")
        try:
            return self._handle(
                str(user_id), str(repo_id), normalize_interaction(action),
                str(logical_id), dwell_seconds,
            )
        except ValueError:
            raise
        except Exception as exc:
            logger.error(
                "Feedback event failed error_type=%s",
                type(exc).__name__,
            )
            return False

    def _handle(
        self,
        user_id: str,
        repo_id: str,
        action: str,
        event_id: str,
        dwell_seconds: float | None,
    ) -> bool:
        definition = get_interaction(action)
        canonical_user_id = user_point_id(user_id)
        user_points = self.qdrant.retrieve(
            collection_name=self.settings.user_collection,
            ids=[canonical_user_id],
            with_payload=True,
            with_vectors=True,
        )
        if not user_points:
            logger.warning("User profile %s is not available yet", user_id)
            return False

        points_by_id = {str(candidate.id): candidate for candidate in user_points}
        point = points_by_id.get(canonical_user_id)
        if point is None:
            logger.warning("User profile %s is not available yet", user_id)
            return False
        if isinstance(point.payload, Mapping) and "model_policy_sha256" in point.payload:
            raise ValueError("negotiated user model requires the ordered model-operation lane")
        search_vector, vector_name = _point_vector(
            point, self.settings.user_vector_name, label="user profile"
        )
        _vector(search_vector, self.settings.vector_dimension, label="user vector")
        if point.payload is not None and not isinstance(point.payload, Mapping):
            raise ValueError("user profile payload must be an object")
        payload = copy.deepcopy(dict(point.payload or {}))
        processed = _processed_events(payload)
        if event_id in processed:
            return True

        latent = _vector(
            payload.get(LATENT_KEY, search_vector),
            self.settings.vector_dimension,
            label="latent user vector",
        )
        adjustments = _feedback_adjustments(payload, self.settings.vector_dimension)
        repo_state = adjustments.setdefault(repo_id, {})
        applied_signals = _applied_signals(payload)

        if action == "dwell":
            if dwell_seconds is None:
                raise ValueError("dwell_seconds is required for dwell")
            alpha = dwell_alpha(
                dwell_seconds,
                minimum_seconds=self.settings.dwell_min_seconds,
                full_credit_seconds=self.settings.dwell_full_credit_seconds,
                maximum_alpha=self.settings.dwell_max_alpha,
            )
            if alpha is None:
                return True
        else:
            if dwell_seconds is not None:
                raise ValueError("dwell_seconds is only valid for dwell")
            alpha = definition.embedding_alpha

        changed = False
        if definition.reversal_of:
            family = definition.state_family or ""
            stored = repo_state.get(family)
            if stored and stored.get("action") == definition.reversal_of:
                latent -= _vector(
                    stored.get("delta", []), self.settings.vector_dimension,
                    label="stored feedback adjustment",
                )
                repo_state.pop(family, None)
                changed = True
        elif definition.state_family:
            family = definition.state_family
            stored = repo_state.get(family)
            if not stored or stored.get("action") != action:
                if stored:
                    latent -= _vector(
                        stored.get("delta", []), self.settings.vector_dimension,
                        label="stored feedback adjustment",
                    )
                repository = self._repository_vector(repo_id)
                if repository is None:
                    return False
                delta = _vector(
                    vector_delta(latent, repository, alpha),
                    self.settings.vector_dimension,
                    label="feedback delta",
                )
                latent += delta
                repo_state[family] = {
                    "action": action,
                    "delta": delta.tolist(),
                    "event_id": event_id,
                }
                changed = True
        elif definition.apply_once:
            repo_signals = applied_signals.setdefault(repo_id, [])
            if action not in repo_signals:
                repository = self._repository_vector(repo_id)
                if repository is None:
                    return False
                latent += _vector(
                    vector_delta(latent, repository, alpha),
                    self.settings.vector_dimension,
                    label="feedback delta",
                )
                repo_signals.append(action)
                changed = True
        elif alpha != 0.0:
            repository = self._repository_vector(repo_id)
            if repository is None:
                return False
            latent += _vector(
                vector_delta(latent, repository, alpha),
                self.settings.vector_dimension,
                label="feedback delta",
            )
            changed = True

        if not repo_state:
            adjustments.pop(repo_id, None)
        processed.append(event_id)
        payload[LATENT_KEY] = latent.tolist()
        payload[ADJUSTMENTS_KEY] = adjustments
        payload[APPLIED_SIGNALS_KEY] = applied_signals
        payload[PROCESSED_KEY] = processed[-self.settings.processed_event_history :]
        normalized = normalize_vector(latent, self.settings.vector_dimension)
        final_vector: Any = {vector_name: normalized} if vector_name else normalized
        self.qdrant.upsert(
            collection_name=self.settings.user_collection,
            points=[PointStruct(id=point.id, vector=final_vector, payload=payload)],
            wait=True,
        )
        logger.info(
            "Applied feedback event %s action=%s user=%s changed=%s",
            event_id, action, user_id, changed,
        )
        return True

    def _repository_vector(self, repo_id: str) -> list[float] | None:
        canonical_repo_id = repository_point_id(repo_id)
        points = self.qdrant.retrieve(
            collection_name=self.settings.repository_collection,
            ids=[canonical_repo_id],
            with_payload=False,
            with_vectors=True,
        )
        if not points:
            logger.warning("Repository vector %s is not available", repo_id)
            return None
        points_by_id = {str(point.id): point for point in points}
        point = points_by_id.get(canonical_repo_id)
        if point is None:
            logger.warning("Repository vector %s is not available", repo_id)
            return None
        vector, _ = _point_vector(
            point, self.settings.repository_vector_name, label="repository"
        )
        return _vector(
            vector, self.settings.vector_dimension, label="repository vector"
        ).tolist()
