"""Build bounded, high-signal model input without mutating canonical Markdown."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import chain
import re
from typing import Any

from utils.readme_processor import MAX_PREPROCESS_CHARS, clean_inline_markdown


_FENCE = re.compile(r"^[ \t]*(```|~~~)")
_HEADING = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(.+)$")
_SETEXT = re.compile(r"^ {0,3}(?:=+|-+)[ \t]*$")
_SETEXT_TEXT = re.compile(r"^ {0,3}[^ \t>]")
_URL = re.compile(r"https?://\S+")
_LIST_MARKER = re.compile(r"^[ \t]*(?:[-*+][ \t]+|\d+[.)][ \t]+)")
_TABLE_CELL = re.compile(r":?-{3,}:?")
_COMMAND = re.compile(
    r"(?:\b(?:npm|pnpm|yarn|pipx?|uv|cargo|go)\s+(?:install|add|run|get)\b"
    r"|\b(?:brew|apt(?:-get)?|dnf|yum|pacman|choco|winget)\s+install\b"
    r"|\bgit\s+clone\b|\b(?:curl|wget)\s+https?://|(?:^|\s)\$\s*\w)",
    re.IGNORECASE,
)
_BLOCKED_HEADING = re.compile(
    r"\b(?:install(?:ation)?|setup|getting started|quick ?start|api\s+(?:reference|docs?)|"
    r"reference|changelog|release notes?|licen[cs]e|contributors?|contributing|"
    r"development|build from source|table of contents|contents)\b",
    re.IGNORECASE,
)
_HIGH_SIGNAL_HEADING = re.compile(
    r"\b(?:overview|about|introduction|what is|why|features?|highlights?|"
    r"use cases?|who is|audience|benefits?|key concepts?|motivation)\b",
    re.IGNORECASE,
)
_LINE_BREAK = re.compile(r"\r\n|[\n\r\v\f\x1c-\x1e\x85\u2028\u2029]")

# These analysis limits are independent of the provider's prompt-length cap.
# Seven evidence variants (six Markdown/cleaned fields and joined paragraphs)
# plus one Markdown scan fit within the aggregate character-work allowance.
MAX_SOURCE_VARIANT_BYTES = 4 * MAX_PREPROCESS_CHARS
MAX_SOURCE_ANALYSIS_LINES = 4_096
MAX_SOURCE_ANALYSIS_WORK = 8 * MAX_PREPROCESS_CHARS
MAX_SOURCE_PARAGRAPHS = 80


class _SourceLimitExceeded(ValueError):
    pass


@dataclass(slots=True)
class _AnalysisBudget:
    work: int = 0

    def spend(self, characters: int) -> None:
        self.work += characters
        if self.work > MAX_SOURCE_ANALYSIS_WORK:
            raise _SourceLimitExceeded("source work budget exceeded")


@dataclass(frozen=True, slots=True)
class SummarySource:
    prompt_input: str
    comparison_texts: tuple[str, ...]
    analysis_truncated: bool = False
    analysis_limit: str | None = None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = (_clean_line(str(item)[:MAX_PREPROCESS_CHARS // 80]) for item in value[:80])
    return [item for item in cleaned if item and not _COMMAND.search(item)]


def _source_variants(
    source: Any, repo: Mapping[str, Any], budget: _AnalysisBudget,
) -> tuple[str, ...]:
    """Retain complete bounded evidence, never a silently truncated prefix."""

    variants: list[str] = []

    def add(value: Any) -> None:
        if not isinstance(value, str) or not value:
            return
        # len is constant-time, so oversized and all-whitespace inputs do not
        # trigger a full-file whitespace scan, encoding, or normalization first.
        if len(value) > MAX_PREPROCESS_CHARS:
            raise _SourceLimitExceeded("source character budget exceeded")
        budget.spend(len(value))
        if len(value.encode("utf-8", errors="surrogatepass")) > MAX_SOURCE_VARIANT_BYTES:
            raise _SourceLimitExceeded("source byte budget exceeded")
        if not value.isspace() and value not in variants:
            variants.append(value)

    for key in ("readme", "clean_text", "readme_md"):
        add(repo.get(key))
    readme = getattr(source, "readme", None)
    for key in ("raw_markdown", "clean_text", "readme_md"):
        add(getattr(readme, key, None))

    # Prefer Markdown when available so excluded section boundaries survive.
    # Paragraph-only jobs still supply both prompt context and copy-check text.
    paragraphs = repo.get("extracted_paragraphs")
    if isinstance(paragraphs, list):
        if len(paragraphs) > MAX_SOURCE_PARAGRAPHS:
            raise _SourceLimitExceeded("source paragraph budget exceeded")
        complete: list[str] = []
        used = 0
        for value in paragraphs:
            if not isinstance(value, str) or not value:
                continue
            used += len(value) + (2 if complete else 0)
            if used > MAX_PREPROCESS_CHARS:
                raise _SourceLimitExceeded("source paragraph character budget exceeded")
            # Do not scan a paragraph until its complete length is bounded.
            if not value.isspace():
                complete.append(value)
        add("\n\n".join(complete))
    return tuple(variants)


def _clean_line(line: str) -> str:
    line = clean_inline_markdown(line[:MAX_PREPROCESS_CHARS])
    line = _URL.sub(" ", line)
    line = _LIST_MARKER.sub("", line)
    line = line.replace("`", " ")
    return re.sub(r"\s+", " ", line).strip()


def _analysis_lines(markdown: str, budget: _AnalysisBudget) -> Iterator[tuple[int, int]]:
    """Iterate splitlines-compatible boundaries with one line of lookahead."""

    start = 0
    count = 0
    for match in _LINE_BREAK.finditer(markdown):
        count += 1
        if count > MAX_SOURCE_ANALYSIS_LINES:
            raise _SourceLimitExceeded("source line budget exceeded")
        budget.spend(match.end() - start)
        yield start, match.start()
        start = match.end()
    if start < len(markdown):
        if count >= MAX_SOURCE_ANALYSIS_LINES:
            raise _SourceLimitExceeded("source line budget exceeded")
        budget.spend(len(markdown) - start)
        yield start, len(markdown)


def _readme_sections(
    markdown: str, *, budget: _AnalysisBudget,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Classify bounded line offsets without retaining cleaned text copies."""

    high: list[tuple[int, int]] = []
    neutral: list[tuple[int, int]] = []
    current = neutral
    blocked_depth: int | None = None
    in_fence = False
    previous_was_heading = False
    lines = iter(_analysis_lines(markdown, budget))
    position = next(lines, None)
    pending_line = markdown[position[0]:position[1]] if position is not None else ""
    while position is not None:
        next_position = next(lines, None)
        line = pending_line
        next_line = (
            markdown[next_position[0]:next_position[1]]
            if next_position is not None else None
        )
        pending_line = next_line or ""
        current_position, position = position, next_position
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading_match = _HEADING.match(line)
        setext = (
            next_line is not None
            and _SETEXT_TEXT.match(line)
            and not _LIST_MARKER.match(line)
            and _SETEXT.match(next_line)
        )
        if heading_match or setext:
            if heading_match:
                depth = len(heading_match.group(1))
                heading = heading_match.group(2).rstrip()
            else:
                assert next_line is not None  # A setext heading has an underline.
                depth = 1 if next_line.lstrip().startswith("=") else 2
                heading = line.strip()
            if blocked_depth is not None and depth > blocked_depth:
                continue
            excluded = (
                _BLOCKED_HEADING.search(heading)
                or heading.casefold().strip(" #: ") == "api"
            )
            blocked_depth = depth if excluded else None
            current = high if _HIGH_SIGNAL_HEADING.search(heading) else neutral
            previous_was_heading = True
            continue
        if previous_was_heading and _SETEXT.match(line):
            previous_was_heading = False
            continue
        previous_was_heading = False
        if blocked_depth is not None:
            continue
        if line.count("|") >= 3:
            continue
        if "|" in line:
            cells = line.strip().strip("|").split("|")
            if all(_TABLE_CELL.fullmatch(cell.strip()) for cell in cells):
                continue
        current.append(current_position)
    return high, neutral


def build_summary_source(
    source: Any,
    repo: Mapping[str, Any],
    *,
    max_chars: int,
) -> SummarySource:
    """Build a metadata-led prompt input capped before provider transport."""

    if max_chars < 1_000:
        raise ValueError("max_chars must be at least 1000")
    budget = _AnalysisBudget()
    try:
        variants = _source_variants(source, repo, budget)
        raw_markdown = variants[0] if variants else ""
        high, neutral = _readme_sections(raw_markdown, budget=budget)
    except _SourceLimitExceeded as exc:
        # A shortened prompt is fine; shortened validation evidence is not.
        # Without complete copy-check coverage, use only metadata fallback.
        return SummarySource("", (), analysis_truncated=True, analysis_limit=str(exc))

    full_name = _clean_line(str(repo.get("full_name") or "")[:512])
    metadata: list[str] = [f"Repository: {full_name}"]
    description = _clean_line(str(repo.get("description") or "")[:MAX_PREPROCESS_CHARS])
    if description and not _COMMAND.search(description) and not _BLOCKED_HEADING.search(description):
        metadata.append(f"Description: {description}")
    topics = _string_list(repo.get("topics"))
    languages = _string_list(repo.get("languages"))
    if topics:
        metadata.append(f"Topics: {', '.join(topics[:20])}")
    if languages:
        metadata.append(f"Languages: {', '.join(languages[:12])}")

    # A name alone does not justify asking the provider to invent a summary.
    has_summary_evidence = bool(description or topics or languages)
    excerpts_truncated = False
    selected: list[str] = []
    used = 0
    chunks = chain(metadata, ("README high-signal excerpts:",), high, neutral)
    for chunk in chunks:
        remaining = max_chars - used - (2 if selected else 0)
        if remaining <= 0:
            excerpts_truncated = True
            break
        if isinstance(chunk, str):
            # Metadata was already normalized once.
            normalized = chunk.strip()
        else:
            # Only the final selector cleans README lines and accounts for
            # usable clipped text. Raw line length must not consume a second
            # allowance and suppress later high-signal candidates.
            start, end = chunk
            normalized = _clean_line(raw_markdown[start:end])
            if (
                len(normalized) < 24
                or _COMMAND.search(normalized)
                or _BLOCKED_HEADING.fullmatch(normalized.rstrip(":"))
            ):
                continue
            has_summary_evidence = True
        if not normalized:
            continue
        if len(normalized) > remaining:
            excerpts_truncated = True
            normalized = normalized[:remaining].rsplit(" ", 1)[0].rstrip(" ,;:-")
        if normalized:
            selected.append(normalized)
            used += len(normalized) + (2 if len(selected) > 1 else 0)

    prompt_input = "\n\n".join(selected) if has_summary_evidence else ""
    comparisons = tuple(dict.fromkeys((*variants, prompt_input)))
    return SummarySource(
        prompt_input=prompt_input,
        comparison_texts=comparisons,
        analysis_truncated=excerpts_truncated,
    )
