"""Repository embedding data models and repo tower composition."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from numbers import Integral, Real
from typing import Any

from summarization.contracts import CardSummaryArtifact, card_summary_payload
from summarization.pipeline import description_fallback_artifact
from utils.readme_processor import clean_markdown_copy

def _parse_list_field(val: Any) -> list[str]:
    if not val:
        return []
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except Exception:
            return []
    if isinstance(val, dict):
        return [str(x) for x in val if x]
    if isinstance(val, list):
        return [str(x) for x in val if x]
    return []

from config import (
    EMBEDDING_MODEL_REVISION,
    README_CHUNK_CHARS,
    README_CHUNK_OVERLAP_CHARS,
    REPO_TOWER_WEIGHTS,
    REPOSITORY_EMBEDDING_DIM,
    REPOSITORY_EMBEDDING_MODEL,
    REPOSITORY_EMBEDDING_VERSION,
    REPOSITORY_FEATURE_SPEC_VERSION,
)
from .embeddings import Vector, aggregate_vectors
from .vector_contract import (
    resolve_repository_identity,
    validate_embedding_vector,
    validate_repository_payload,
)
from ingestion.features import (
    activity_score,
    extract_tags,
    score_code_health,
    score_documentation,
    trend_velocity,
)
from ingestion.classification import classify_category


SUPPORTED_REPOSITORY_EMBEDDING_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


@dataclass(slots=True)
class RepositoryEmbeddingConfig:
    """Configuration for repository embedding generation."""

    model_name: str = REPOSITORY_EMBEDDING_MODEL
    embedding_dim: int = REPOSITORY_EMBEDDING_DIM
    version: str = REPOSITORY_EMBEDDING_VERSION
    readme_chunk_chars: int = README_CHUNK_CHARS
    readme_chunk_overlap_chars: int = README_CHUNK_OVERLAP_CHARS
    tower_weights: dict[str, float] = field(default_factory=lambda: dict(REPO_TOWER_WEIGHTS))

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        expected_dim = SUPPORTED_REPOSITORY_EMBEDDING_DIMS.get(self.model_name)
        if expected_dim is None:
            supported = ", ".join(sorted(SUPPORTED_REPOSITORY_EMBEDDING_DIMS))
            raise ValueError(
                f"Unsupported repository embedding model {self.model_name!r}. "
                "The current Qdrant schema requires a known repository embedding dimension. "
                f"Supported models: {supported}."
            )
        if isinstance(self.embedding_dim, bool) or not isinstance(self.embedding_dim, Integral):
            raise TypeError("embedding_dim must be an integer")
        if self.embedding_dim != expected_dim:
            raise ValueError(
                f"Repository embedding model {self.model_name!r} produces {expected_dim} dimensions, "
                f"but embedding_dim is configured as {self.embedding_dim}."
            )
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be a non-empty string")
        if (
            isinstance(self.readme_chunk_chars, bool)
            or not isinstance(self.readme_chunk_chars, Integral)
            or self.readme_chunk_chars <= 0
        ):
            raise ValueError("readme_chunk_chars must be a positive integer")
        if (
            isinstance(self.readme_chunk_overlap_chars, bool)
            or not isinstance(self.readme_chunk_overlap_chars, Integral)
            or self.readme_chunk_overlap_chars < 0
        ):
            raise ValueError("readme_chunk_overlap_chars must be a non-negative integer")
        if self.readme_chunk_overlap_chars >= self.readme_chunk_chars:
            raise ValueError("readme_chunk_overlap_chars must be smaller than readme_chunk_chars")

        expected_towers = {"readme", "metadata", "topics"}
        if not isinstance(self.tower_weights, Mapping):
            raise TypeError("tower_weights must be a mapping")
        if set(self.tower_weights) != expected_towers:
            raise ValueError(
                "tower_weights must contain exactly: metadata, readme, topics"
            )
        validated_weights: dict[str, float] = {}
        for tower, raw_weight in self.tower_weights.items():
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, Real):
                raise TypeError(f"tower weight {tower!r} must be a real number")
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"tower weight {tower!r} must be finite and non-negative")
            validated_weights[tower] = weight
        if sum(validated_weights.values()) <= 0:
            raise ValueError("tower_weights must have a positive total")
        self.tower_weights = validated_weights


@dataclass(slots=True)
class RepositoryEmbeddingResult:
    """Embeddings and Qdrant payload generated for one repository."""

    repo_id: str
    final_embedding: Vector
    readme_embedding: Vector
    metadata_embedding: Vector
    topic_embedding: Vector
    payload: dict[str, Any]
    readme_chunks: int
    source_hash: str
    embedding_model: str
    embedding_version: str
    card_summary: CardSummaryArtifact | None = None


def build_readme_text(source: Any) -> str:
    """Return cleaned README text from an EnrichmentResult or payload dict."""
    # The below preference is for using the cleaned README from acquisition when
    # available; staged JSON payloads fall back to extracted paragraphs.
    readme = getattr(source, "readme", None)
    clean_text = getattr(readme, "clean_text", None)
    if clean_text:
        return str(clean_text)

    payload = coerce_payload(source)
    raw_markdown = payload.get("readme")
    if isinstance(raw_markdown, str) and raw_markdown.strip():
        return clean_markdown_copy(raw_markdown)
    paragraphs = payload.get("extracted_paragraphs") or []
    if isinstance(paragraphs, list):
        return "\n\n".join(str(item) for item in paragraphs if item)
    return str(paragraphs or "")


def coerce_payload(source: Any) -> dict[str, Any]:
    """Extract the repository payload from supported pipeline inputs."""
    if isinstance(source, Mapping):
        return dict(source)
    payload = getattr(source, "payload", None)
    if isinstance(payload, Mapping):
        return dict(payload)
    raise TypeError("repository source must be a payload dict or an object with a payload dict")


def build_metadata_text(repo: Mapping[str, Any]) -> str:
    """Project structured repository metadata into stable embedding text."""
    lines = [
        f"Repository: {repo.get('full_name') or repo.get('id', '')}",
        f"Description: {repo.get('description') or ''}",
        f"Primary language: {repo.get('primary_language') or 'Unknown'}",
        f"Stars: {int(repo.get('star_count') or 0)}",
        f"Forks: {int(repo.get('fork_count') or 0)}",
        f"Open issues: {int(repo.get('open_issues_count') or 0)}",
        f"Pushed days ago: {int(repo.get('pushed_days_ago') or 999)}",
        f"Star deltas: 3d={int(repo.get('delta_3d') or 0)}, 7d={int(repo.get('delta_7d') or 0)}, 30d={int(repo.get('delta_30d') or 0)}",
        f"Contributor signal: mentionable_users_count={int(repo.get('mentionable_users_count') or 0)}",
        f"Discovery category: {repo.get('discovery_category') or ''}",
        f"Discovery band: {repo.get('discovery_band') or ''}",
    ]
    return "\n".join(lines)


def build_topic_text(repo: Mapping[str, Any]) -> str:
    """Project topics and languages into embedding text."""
    topics = ", ".join(_parse_list_field(repo.get("topics")))
    languages = ", ".join(_parse_list_field(repo.get("languages")))
    return f"GitHub topics: {topics}\nLanguages: {languages}".strip()


def combine_repo_tower(
    *,
    readme_embedding: Sequence[float],
    metadata_embedding: Sequence[float],
    topic_embedding: Sequence[float],
    weights: Mapping[str, float],
) -> Vector:
    """Combine component embeddings with a simple weighted normalized tower."""
    # The below weighted average is for the first repo tower implementation; it
    # keeps the interface stable while leaving room for a learned tower later.
    vectors: list[Sequence[float]] = []
    vector_weights: list[float] = []
    for key, vector in (
        ("readme", readme_embedding),
        ("metadata", metadata_embedding),
        ("topics", topic_embedding),
    ):
        if vector:
            vectors.append(vector)
            vector_weights.append(float(weights.get(key, 0.0)))
    return aggregate_vectors(vectors, weights=vector_weights)


def build_vector_payload(
    repo: Mapping[str, Any],
    *,
    repo_id: str,
    final_embedding: Sequence[float],
    readme_chunks: int,
    source_hash: str,
    config: RepositoryEmbeddingConfig,
    card_summary: CardSummaryArtifact | None = None,
) -> dict[str, Any]:
    """Build the Qdrant payload schema for one repository vector."""
    normalized_repo = dict(repo)
    normalized_repo["repo_id"] = repo_id
    canonical_repo_id, full_name = resolve_repository_identity(normalized_repo)
    validated_embedding = validate_embedding_vector(
        final_embedding,
        expected_size=config.embedding_dim,
        field_name=f"embedding for {canonical_repo_id}",
    )
    normalized_repo.update(
        {
            "repo_id": canonical_repo_id,
            "full_name": full_name,
            "github_id": _optional_decimal_string_field(repo.get("github_id")),
            "description": _string_field(repo.get("description"), "description", default=""),
            "primary_language": _string_field(
                repo.get("primary_language"), "primary_language", default="Unknown"
            ),
            "languages": _parse_list_field(repo.get("languages")),
            "topics": _parse_list_field(repo.get("topics")),
            "star_count": _integer_field(repo.get("star_count"), "star_count", default=0),
            "fork_count": _integer_field(repo.get("fork_count"), "fork_count", default=0),
            "open_issues_count": _integer_field(
                repo.get("open_issues_count"), "open_issues_count", default=0
            ),
            "readme_length": _integer_field(
                repo.get("readme_length"), "readme_length", default=0
            ),
            "pushed_days_ago": _integer_field(
                repo.get("pushed_days_ago"), "pushed_days_ago", default=999
            ),
            "delta_3d": _integer_field(
                repo.get("delta_3d"), "delta_3d", default=0, non_negative=False
            ),
            "delta_7d": _integer_field(
                repo.get("delta_7d"), "delta_7d", default=0, non_negative=False
            ),
            "delta_30d": _integer_field(
                repo.get("delta_30d"), "delta_30d", default=0, non_negative=False
            ),
            "mentionable_users_count": _integer_field(
                repo.get("mentionable_users_count"),
                "mentionable_users_count",
                default=0,
            ),
            "content_version": _integer_field(
                repo.get("content_version"), "content_version", default=0
            ),
            "readme_to_codebase_ratio": _finite_number_field(
                repo.get("readme_to_codebase_ratio"),
                "readme_to_codebase_ratio",
                default=0.0,
            ),
            "extracted_paragraphs": _parse_list_field(repo.get("extracted_paragraphs")),
            "recent_commits": _parse_list_field(repo.get("recent_commits")),
        }
    )
    normalized_readme_chunks = _integer_field(
        readme_chunks, "readme_chunks", default=0
    )
    # The below payload fields are for Qdrant filtering and inspection without
    # fetching the original repository object again.
    tags = extract_tags(full_name, normalized_repo["extracted_paragraphs"])
    category = classify_category(normalized_repo, tags)
    documentation = score_documentation(normalized_repo)

    payload = {
        "repo_id": canonical_repo_id,
        "github_id": normalized_repo["github_id"],
        "full_name": full_name,
        "html_url": _optional_string_field(repo.get("html_url"), "html_url"),
        "description": normalized_repo["description"],
        "primary_language": normalized_repo["primary_language"],
        "languages": normalized_repo["languages"],
        "topics": normalized_repo["topics"],
        "star_count": normalized_repo["star_count"],
        "fork_count": normalized_repo["fork_count"],
        "open_issues_count": normalized_repo["open_issues_count"],
        "readme_length": normalized_repo["readme_length"],
        "readme_chunks": normalized_readme_chunks,
        "pushed_days_ago": normalized_repo["pushed_days_ago"],
        "delta_3d": normalized_repo["delta_3d"],
        "delta_7d": normalized_repo["delta_7d"],
        "delta_30d": normalized_repo["delta_30d"],
        "mentionable_users_count": normalized_repo["mentionable_users_count"],
        "created_at": _timestamp_field(repo.get("created_at"), "created_at"),
        "updated_at": _timestamp_field(repo.get("updated_at"), "updated_at"),
        "pushed_at": _timestamp_field(repo.get("pushed_at"), "pushed_at"),
        "discovery_category": _optional_string_field(
            repo.get("discovery_category"), "discovery_category"
        ),
        "discovery_band": _optional_string_field(
            repo.get("discovery_band"), "discovery_band"
        ),
        "category": category,
        "tags": tags,
        "doc_quality": _finite_number_field(documentation.score, "doc_quality"),
        "code_health": _finite_number_field(
            score_code_health(normalized_repo), "code_health"
        ),
        "activity_score": _finite_number_field(
            activity_score(normalized_repo), "activity_score"
        ),
        "trend_velocity": _finite_number_field(
            trend_velocity(normalized_repo), "trend_velocity"
        ),
        "embedding_dim": len(validated_embedding),
        "embedding_model": config.model_name,
        "embedding_model_revision": EMBEDDING_MODEL_REVISION,
        "embedding_version": config.version,
        "feature_spec_version": REPOSITORY_FEATURE_SPEC_VERSION,
        "content_version": normalized_repo["content_version"],
        "content_hash": _string_field(
            repo.get("content_hash") or source_hash, "content_hash"
        ),
        "model_version": _string_field(
            repo.get("model_version") or config.model_name, "model_version"
        ),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "source_hash": _string_field(source_hash, "source_hash"),
    }
    artifact = card_summary or description_fallback_artifact(
        normalized_repo,
    )
    payload.update(card_summary_payload(artifact))
    validate_repository_payload(payload, require_serving_eligibility=False)
    return payload


def _string_field(value: Any, field_name: str, *, default: str | None = None) -> str:
    if value is None or value == "":
        if default is not None:
            return default
        raise ValueError(f"{field_name} must be a non-empty string")
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized and default is None:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized or default or ""


def _optional_string_field(value: Any, field_name: str) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")
    return value.strip() or None


def _optional_decimal_string_field(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise TypeError("github_id must be a decimal string or None")
    normalized = value.strip()
    if not normalized.isdecimal():
        raise ValueError("github_id must be a decimal string")
    return normalized


def _integer_field(
    value: Any,
    field_name: str,
    *,
    default: int,
    non_negative: bool = True,
) -> int:
    if value is None or value == "":
        result = default
    elif isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    elif isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, Real) and math.isfinite(float(value)) and float(value).is_integer():
        result = int(value)
    elif isinstance(value, str):
        try:
            result = int(value.strip())
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer") from exc
    else:
        raise TypeError(f"{field_name} must be an integer")
    if non_negative and result < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return result


def _finite_number_field(value: Any, field_name: str, *, default: float = 0.0) -> float:
    if value is None or value == "":
        result = default
    elif isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
    else:
        result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _timestamp_field(value: Any, field_name: str) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
        normalized = value.isoformat()
        if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
            raise ValueError(f"{field_name} must be an ISO-8601 UTC timestamp")
        return normalized
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be an ISO-8601 UTC string or None")
    normalized = value.strip()
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be an ISO-8601 UTC timestamp")
    return normalized


def source_fingerprint(*parts: str) -> str:
    """Hash source text used for future re-embedding decisions."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()
