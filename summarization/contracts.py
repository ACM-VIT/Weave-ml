"""Strict public contracts for concise repository card summaries."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Annotated, Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


CARD_SUMMARY_PROMPT_VERSION = "repo-card-summary-v1"
CARD_SUMMARY_FORMAT_VERSION = "repo-card-summary-json-v1"
CARD_SUMMARY_FALLBACK_MODEL_VERSION = "repository-description-fallback-v1"
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


CARD_SUMMARY_PAYLOAD_FIELDS = (
    "card_summary",
    "card_summary_highlights",
    "card_summary_prompt_version",
    "card_summary_model_version",
    "card_summary_format_version",
    "card_summary_source",
    "card_summary_artifact_hash",
)


def card_summary_artifact_hash(artifact: CardSummaryArtifact) -> str:
    """Fingerprint every artifact field so summary-only CAS writes are complete."""

    serialized = json.dumps(
        artifact.model_dump(mode="json"),
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
    }


def card_summary_from_payload(
    payload: Mapping[str, Any],
) -> CardSummaryArtifact | None:
    """Read a complete artifact from Qdrant, treating partial legacy data as absent."""

    if not all(field in payload for field in CARD_SUMMARY_PAYLOAD_FIELDS):
        return None
    if any(payload.get(field) is None for field in CARD_SUMMARY_PAYLOAD_FIELDS):
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
        stored_hash = payload["card_summary_artifact_hash"]
        if not isinstance(stored_hash, str) or not hmac.compare_digest(
            stored_hash,
            card_summary_artifact_hash(artifact),
        ):
            return None
        return artifact
    except (TypeError, ValueError):
        return None
