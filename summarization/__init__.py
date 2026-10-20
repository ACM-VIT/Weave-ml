"""Versioned, bounded repository card-summary generation."""

from .contracts import (
    CARD_SUMMARY_FORMAT_VERSION,
    CARD_SUMMARY_MAX_CHARS,
    CARD_SUMMARY_PROMPT_VERSION,
    CardSummaryArtifact,
    CardSummaryContent,
)
from .pipeline import CardSummaryPipeline

__all__ = [
    "CARD_SUMMARY_FORMAT_VERSION",
    "CARD_SUMMARY_MAX_CHARS",
    "CARD_SUMMARY_PROMPT_VERSION",
    "CardSummaryArtifact",
    "CardSummaryContent",
    "CardSummaryPipeline",
]
