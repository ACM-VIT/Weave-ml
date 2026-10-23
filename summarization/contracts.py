"""Strict public contracts for concise repository card summaries."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Annotated, Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, StringConstraints, field_validator


CARD_SUMMARY_PROMPT_VERSION = "repo-card-summary-v1"
CARD_SUMMARY_FORMAT_VERSION = "repo-card-summary-json-v1"
CARD_SUMMARY_FALLBACK_MODEL_VERSION = "repository-description-fallback-v1"
# Lengths are Unicode codepoints (Python len, PostgreSQL char_length), not UTF-16.
CARD_SUMMARY_MAX_CHARS = 360
CARD_SUMMARY_MIN_GENERATED_CHARS = 180
CARD_SUMMARY_MAX_HIGHLIGHTS = 3

ShortHighlight = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=96),
]


class _StrictSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CardSummaryContent(_StrictSummaryModel):
    """The only content keys a model is allowed to return."""

    summary: str = Field(min_length=1, max_length=CARD_SUMMARY_MAX_CHARS)
    highlights: list[ShortHighlight] = Field(
        default_factory=list,
        max_length=CARD_SUMMARY_MAX_HIGHLIGHTS,
    )

    @field_validator("highlights")
    @classmethod
    def unique_highlights(cls, value: list[str]) -> list[str]:
        normalized = [item.casefold() for item in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("highlights must be unique")
        return value


class CardSummaryArtifact(CardSummaryContent):
    """Versioned summary artifact returned to the canonical backend."""

    prompt_version: str = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=256)
    format_version: str = Field(min_length=1, max_length=128)
    source: Literal["generated", "description_fallback"]
    # Internal Qdrant provenance, deliberately absent from the backend JSON.
    _reuse_key: str | None = PrivateAttr(default=None)


CARD_SUMMARY_PAYLOAD_FIELDS = (
    "card_summary",
    "card_summary_highlights",
    "card_summary_prompt_version",
    "card_summary_model_version",
    "card_summary_format_version",
    "card_summary_source",
    "card_summary_artifact_hash",
    "card_summary_reuse_key",
)


def card_summary_artifact_hash(artifact: CardSummaryArtifact) -> str:
    """Fingerprint every artifact field so summary-only CAS writes are complete."""

    content = artifact.model_dump(mode="json")
    if artifact._reuse_key is not None:
        content = {**content, "reuse_key": artifact._reuse_key}
    serialized = json.dumps(
        content,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def card_summary_payload(artifact: CardSummaryArtifact) -> dict[str, Any]:
    """Flatten an artifact into scalar/list Qdrant payload fields."""

    return {
        "card_summary": artifact.summary,
        "card_summary_highlights": list(artifact.highlights),
        "card_summary_prompt_version": artifact.prompt_version,
        "card_summary_model_version": artifact.model_version,
        "card_summary_format_version": artifact.format_version,
        "card_summary_source": artifact.source,
        "card_summary_artifact_hash": card_summary_artifact_hash(artifact),
        "card_summary_reuse_key": artifact._reuse_key,
    }


def card_summary_from_payload(
    payload: Mapping[str, Any],
) -> CardSummaryArtifact | None:
    """Read a complete artifact from Qdrant, treating partial legacy data as absent."""

    # Legacy payloads can still be parsed, but generated artifacts without
    # provenance require regeneration before current-source replay.
    required_fields = CARD_SUMMARY_PAYLOAD_FIELDS[:-1]
    if not all(field in payload for field in required_fields):
        return None
    if any(payload.get(field) is None for field in required_fields):
        return None
    try:
        artifact = CardSummaryArtifact.model_validate(
            {
                "summary": payload["card_summary"],
                "highlights": payload["card_summary_highlights"],
                "prompt_version": payload["card_summary_prompt_version"],
                "model_version": payload["card_summary_model_version"],
                "format_version": payload["card_summary_format_version"],
                "source": payload["card_summary_source"],
            }
        )
        reuse_key = payload.get("card_summary_reuse_key")
        if reuse_key is not None:
            if not isinstance(reuse_key, str) or len(reuse_key) != 64 or any(
                char not in "0123456789abcdef" for char in reuse_key
            ):
                return None
            artifact._reuse_key = reuse_key
        stored_hash = payload["card_summary_artifact_hash"]
        if not isinstance(stored_hash, str) or not hmac.compare_digest(
            stored_hash,
            card_summary_artifact_hash(artifact),
        ):
            return None
        return artifact
    except (TypeError, ValueError):
        return None
