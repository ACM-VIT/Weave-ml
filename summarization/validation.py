"""Deterministic parser and safety validation for model summary output."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import chain
import json
import re
import unicodedata
from typing import Iterable

from pydantic import ValidationError

from utils.readme_processor import MAX_PREPROCESS_CHARS, contains_inline_markdown_link

from .contracts import (
    CARD_SUMMARY_MAX_CHARS,
    CARD_SUMMARY_MIN_GENERATED_CHARS,
    CardSummaryContent,
)
from .sentences import SENTENCE_END_PATTERN, has_two_or_three_complete_sentences, mask_summary_abbreviations


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
    rf"\blicensed under\b|(?:^|{SENTENCE_END_PATTERN}\s+|\n)[ \t]*GPL(?:[ \t-]*v?\d+(?:\.\d+)*(?:\+|[ \t]+or[ \t]+later)?|[ \t]*\+|[ \t]+or[ \t]+later)?[ \t]*(?:[.!?:]|$)",
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


@dataclass(frozen=True, slots=True)
class SourceAnalysis:
    """Complete source tokens and an eight-word index shared by all outputs."""

    words: tuple[str, ...]
    joined: str
    eightgrams: frozenset[tuple[str, ...]]
    normalized_prose: str


def _normalized_prose(value: str) -> str:
    """Match the backend's lowercase Unicode-letter/number normalization."""
    lowered = value.lower()
    if lowered.isascii():
        return re.sub(r"[^a-z0-9]+", " ", lowered).strip()
    return " ".join("".join(
        char if unicodedata.category(char)[0] in {"L", "N"} else " "
        for char in lowered
    ).split())


def _contains_partial_copy(prose: str, source_prose: str) -> bool:
    if prose.isascii() and source_prose.isascii():
        return any(prose[index:index + 80] in source_prose
                   for index in range(len(prose) - 79))
    # JavaScript length/slice use UTF-16 units, including windows whose ends
    # split surrogate pairs. Match its 80-unit gate without matching odd byte
    # offsets, which cannot represent the backend's string positions.
    candidate = prose.encode("utf-16-le", "surrogatepass")
    source = source_prose.encode("utf-16-le", "surrogatepass")
    for index in range(0, len(candidate) - 159, 2):
        passage = candidate[index:index + 160]
        match = source.find(passage)
        while match >= 0:
            if match % 2 == 0:
                return True
            match = source.find(passage, match + 1)
    return False


def analyze_comparison_texts(texts: Iterable[str]) -> tuple[SourceAnalysis, ...]:
    sources = tuple(source for source in texts if source)
    if any(len(source) > MAX_PREPROCESS_CHARS for source in sources):
        raise SummaryValidationError(("source comparison exceeds the preprocessing bound",))
    analyses = []
    for source in dict.fromkeys(sources):
        words = tuple(_words(source))
        analyses.append(SourceAnalysis(
            words, " " + " ".join(words) + " ",
            frozenset(words[index:index + 8] for index in range(len(words) - 7)),
            _normalized_prose(source),
        ))
    return tuple(analyses)


def _near_copy(summary: str, source: str | SourceAnalysis) -> bool:
    if isinstance(source, str):
        if not source:
            return False
        analysis = analyze_comparison_texts((source,))[0]
    else:
        analysis = source
    return _near_copy_words(tuple(_words(summary)), analysis)


def _near_copy_words(summary_words: tuple[str, ...], source: SourceAnalysis) -> bool:
    source_words = source.words
    joined_summary = " ".join(summary_words)
    if len(joined_summary) >= 80 and joined_summary in source.joined:
        return True
    if len(summary_words) < 8 or len(source_words) < 8:
        return False

    contiguous = min(12, len(summary_words))
    for index in range(len(summary_words) - contiguous + 1):
        if summary_words[index:index + 8] in source.eightgrams and (
            contiguous == 8
            or " " + " ".join(summary_words[index:index + contiguous]) + " " in source.joined
        ):
            return True

    window_size = len(summary_words)
    summary_counts = Counter(summary_words)
    if len(source_words) < window_size:
        overlap = sum((summary_counts & Counter(source_words)).values())
        if 200 * overlap < 84 * (window_size + len(source_words)):
            return False
        return SequenceMatcher(None, summary_words, source_words, autojunk=False).ratio() >= 0.84
    step = max(1, window_size // 4)
    last_start = len(source_words) - window_size
    starts = range(0, last_start + 1, step)
    # Preserve every original window, including the off-grid tail. The multiset
    # intersection is an upper bound on matching tokens, so it cannot discard a
    # window whose SequenceMatcher ratio reaches the existing 0.84 threshold.
    for start in chain(starts, () if last_start % step == 0 else (last_start,)):
        window = source_words[start:start + window_size]
        overlap = sum((summary_counts & Counter(window)).values())
        if 100 * overlap < 84 * window_size:
            continue
        if SequenceMatcher(None, summary_words, window, autojunk=False).ratio() >= 0.84:
            return True
    return False


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
    comparison_texts: Iterable[str] = (),
    source_analysis: tuple[SourceAnalysis, ...] | None = None,
    canonical_readme: str | None = None,
) -> CardSummaryContent:
    """Return normalized safe content or raise deterministic validation errors."""

    # Internal callers pass complete precomputed analysis across highlights and
    # repairs; direct callers retain the same fail-closed input bound.
    sources = source_analysis if source_analysis is not None else analyze_comparison_texts(comparison_texts)
    # The provider may see an approved prefix, but the acceptance boundary
    # checks the complete canonical README. This is the existing HTTP source
    # limit, not a larger prompt/markdown/tokenization budget. Do not retain the
    # full normalization in the process-local analysis cache.
    if canonical_readme is not None and (
        not isinstance(canonical_readme, str) or len(canonical_readme) > 1_000_000
    ):
        raise SummaryValidationError(("canonical README exceeds the source contract",))
    copy_sources = tuple(source.normalized_prose for source in sources)
    if canonical_readme:
        copy_sources += (_normalized_prose(canonical_readme),)
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
        highlight_words = tuple(_words(normalized_highlight))
        highlight_prose = _normalized_prose(normalized_highlight)
        if (any(_contains_partial_copy(highlight_prose, source) for source in copy_sources)
                or any(_near_copy_words(highlight_words, source) for source in sources)):
            reasons.append("highlight copies a long contiguous source passage")

    summary_words = tuple(_words(normalized))
    summary_prose = _normalized_prose(normalized)
    if (any(_contains_partial_copy(summary_prose, source) for source in copy_sources)
            or any(_near_copy_words(summary_words, source) for source in sources)):
        reasons.append("summary copies a long contiguous source passage")
    if reasons:
        raise SummaryValidationError(reasons)
    return CardSummaryContent(summary=normalized, highlights=normalized_highlights)
