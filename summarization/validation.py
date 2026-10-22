"""Deterministic parser and safety validation for model summary output."""

from __future__ import annotations

from difflib import SequenceMatcher
import json
import re
from typing import Iterable

from pydantic import ValidationError

from utils.readme_processor import MAX_PREPROCESS_CHARS, contains_inline_markdown_link

from .contracts import (
    CARD_SUMMARY_MAX_CHARS,
    CARD_SUMMARY_MIN_GENERATED_CHARS,
    CardSummaryContent,
)
from .sentences import has_two_or_three_complete_sentences, mask_summary_abbreviations


_MARKDOWN = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]|[-*+][ \t]|\d+[.)][ \t])|```|~~~|`",
    re.MULTILINE,
)
_COMMAND = re.compile(
    r"(?:\b(?:npm|pnpm|yarn|pipx?|uv|cargo|go)\s+(?:install|add|run|get)\b"
    r"|\b(?:brew|apt(?:-get)?|dnf|yum|pacman|choco|winget)\s+install\b"
    r"|\bgit\s+clone\b|\b(?:curl|wget)\s+https?://|(?:^|\s)\$\s*\w)",
    re.IGNORECASE,
)
_README_SECTION = re.compile(
    r"\b(?:installation|installing|getting started|quick ?start|changelog|"
    r"release notes?|licen[cs]e|contributors?|contributing|api\s+(?:reference|docs?)|"
    r"reference|table of contents|badges?)\b",
    re.IGNORECASE,
)
_LICENSE_BOILERPLATE = re.compile(
    r"\blicensed under\b|(?:^|[.!?][ \t]+|\n)[ \t]*GPL(?:[ \t-]*v?\d+(?:\.\d+)*(?:\+|[ \t]+or[ \t]+later)?|[ \t]*\+|[ \t]+or[ \t]+later)?[ \t]*(?:[.!?:]|$)",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)?", re.IGNORECASE)
_HTML = re.compile(r"<(?:/?[A-Za-z]|!)[^<>]*>")
_IDENTIFIER = re.compile(r"\w+")


class SummaryValidationError(ValueError):
    def __init__(self, reasons: Iterable[str]) -> None:
        self.reasons = tuple(dict.fromkeys(reasons))
        super().__init__("; ".join(self.reasons))


def normalize_summary(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def parse_summary_response(raw: str) -> CardSummaryContent:
    """Parse exactly one JSON object, allowing only an outer transport fence."""

    if not isinstance(raw, str) or not raw.strip():
        raise SummaryValidationError(("model response is empty",))
    if len(raw) > 8_192:
        raise SummaryValidationError(("model response exceeds the transport bound",))
    candidate = raw.strip()
    if candidate.startswith("```") and candidate.endswith("```") and len(candidate) >= 6:
        candidate = candidate[3:-3]
        if candidate[:4].casefold() == "json":
            candidate = candidate[4:]
        candidate = candidate.strip()
    try:
        decoded = json.loads(candidate)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SummaryValidationError(("model response is not valid JSON",)) from exc
    if not isinstance(decoded, dict):
        raise SummaryValidationError(("model response must be a JSON object",))
    try:
        return CardSummaryContent.model_validate(decoded)
    except ValidationError as exc:
        raise SummaryValidationError(("model response violates the summary schema",)) from exc


def _words(value: str) -> list[str]:
    return [match.group(0).casefold().replace("’", "'") for match in _WORD.finditer(value)]


def _near_copy(summary: str, source: str) -> bool:
    summary_words = _words(summary)
    source_words = _words(source[:MAX_PREPROCESS_CHARS])
    joined_summary = " ".join(summary_words)
    joined_source = " ".join(source_words)
    if len(joined_summary) >= 80 and joined_summary in joined_source:
        return True
    if len(summary_words) < 8 or len(source_words) < 8:
        return False

    contiguous = min(12, len(summary_words))
    if contiguous >= 8:
        summary_ngrams = {
            tuple(summary_words[index : index + contiguous])
            for index in range(len(summary_words) - contiguous + 1)
        }
        if any(
            tuple(source_words[index : index + contiguous]) in summary_ngrams
            for index in range(len(source_words) - contiguous + 1)
        ):
            return True

    window_size = len(summary_words)
    if len(source_words) < window_size:
        return SequenceMatcher(None, summary_words, source_words, autojunk=False).ratio() >= 0.84
    step = max(1, window_size // 4)
    last_start = len(source_words) - window_size
    starts = list(range(0, last_start + 1, step))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return any(
        SequenceMatcher(
            None,
            summary_words,
            source_words[start : start + window_size],
            autojunk=False,
        ).ratio()
        >= 0.84
        for start in starts
    )


def _contains_inline_formatting(value: str) -> bool:
    """Find paired emphasis/HTML with one scan, preserving Python identifiers."""
    value = value[:MAX_PREPROCESS_CHARS]
    if _HTML.search(value):
        return True
    # A standalone __identifier__ is valid Python prose. Protect complete word
    # tokens, while multiword __emphasized phrases__ remain formatting.
    identifiers: dict[int, int] = {}
    for match in _IDENTIFIER.finditer(value):
        token = match.group(0)
        if len(token) > 4 and token.startswith("__") and token.endswith("__") and token[2:-2].isidentifier():
            identifiers[match.start()] = match.end()

    openers: set[str] = set()
    index = 0
    while index < len(value):
        if index in identifiers:
            index = identifiers[index]
            continue
        marker = value[index]
        if marker == "\\":
            index += 2
            continue
        if marker not in "*_~":
            index += 1
            continue
        end = index + 1
        while end < len(value) and value[end] == marker:
            end += 1
        previous = value[index - 1] if index else " "
        following = value[end] if end < len(value) else " "
        can_open = not following.isspace()
        can_close = not previous.isspace()
        if marker == "_":
            can_open = can_open and not (previous.isalnum() or previous == "_")
            can_close = can_close and not (following.isalnum() or following == "_")
        if marker != "~" or end - index >= 2:
            if can_close and marker in openers:
                return True
            if can_open:
                openers.add(marker)
        index = end
    return False


def _unsafe_text(value: str) -> list[str]:
    reasons: list[str] = []
    if "\n" in value or "\r" in value:
        reasons.append("summary must be one paragraph")
    if _MARKDOWN.search(value) or contains_inline_markdown_link(value) or _contains_inline_formatting(value):
        reasons.append("summary contains Markdown structure or code")
    if _COMMAND.search(value):
        reasons.append("summary contains a command or installation instruction")
    if _README_SECTION.search(value) or _LICENSE_BOILERPLATE.search(mask_summary_abbreviations(value)):
        reasons.append("summary contains an excluded README section")
    return reasons


def validate_generated_summary(
    content: CardSummaryContent,
    *,
    comparison_texts: Iterable[str],
) -> CardSummaryContent:
    """Return normalized safe content or raise deterministic validation errors."""

    sources = tuple(source for source in comparison_texts if source)
    reasons = _unsafe_text(content.summary)
    normalized = normalize_summary(content.summary)
    if len(normalized) > CARD_SUMMARY_MAX_CHARS:
        reasons.append(f"summary exceeds {CARD_SUMMARY_MAX_CHARS} characters")
    if len(normalized) < CARD_SUMMARY_MIN_GENERATED_CHARS:
        reasons.append("summary is too short for a useful discovery card")
    if not has_two_or_three_complete_sentences(normalized):
        reasons.append("summary must contain exactly two or three complete sentences")

    normalized_highlights: list[str] = []
    for highlight in content.highlights:
        reasons.extend(_unsafe_text(highlight))
        normalized_highlight = normalize_summary(highlight)
        normalized_highlights.append(normalized_highlight)
        if any(_near_copy(normalized_highlight, source) for source in sources):
            reasons.append("highlight copies a long contiguous source passage")

    if any(_near_copy(normalized, source) for source in sources):
        reasons.append("summary copies a long contiguous source passage")
    if reasons:
        raise SummaryValidationError(reasons)
    return CardSummaryContent(summary=normalized, highlights=normalized_highlights)
