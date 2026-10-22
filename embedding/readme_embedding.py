"""Bounded Markdown section selection and tokenizer-aware README inputs.

Weights are budgets for entire section classes, not multipliers per chunk.
Adding installation/reference material therefore cannot dilute an overview
merely by producing more chunks or headings.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from utils.deadline import check_deadline
from utils.readme_processor import MAX_PREPROCESS_CHARS, clean_markdown_copy


README_STRATEGY_VERSION = "token-sections-v1"
SECTION_WEIGHTS = {"purpose": 3.0, "neutral": 1.0, "setup": 0.1}
# At most 64 README encoder inputs per repository, even for dense code/text.
SECTION_CHUNK_LIMITS = {"purpose": 32, "neutral": 24, "setup": 8}
_HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT = re.compile(r"^ {0,3}(=+|-+)\s*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_OMIT = re.compile(
    r"(?:licen[cs]es?(?: information)?|copyright(?: notice)?|sponsors?|sponsorship|"
    r"changelog|change log|release notes?|contributors?|contributing|"
    r"acknowledg(?:e)?ments?|table of contents|(?:generated )?api reference|"
    r"generated reference)(?:\s*[:(\-–].*)?", re.I,
)
_PURPOSE = re.compile(
    r"\b(?:overview|introduction|about|features?|highlights?|use cases?|"
    r"architecture|how it works|usage|what is|why|motivation)\b", re.I,
)
_SETUP = re.compile(r"\b(?:install(?:ation)?|setup|getting started|quick ?start|build)\b", re.I)


@dataclass(frozen=True, slots=True)
class ReadmeChunk:
    text: str
    section_class: str
    token_count: int
    weight: float


def readme_source(source: Any, payload: dict[str, Any]) -> str:
    """Prefer original Markdown; retain cleaned/paragraph-only compatibility."""
    document = getattr(source, "readme", None)
    for value in (
        getattr(document, "raw_markdown", None), payload.get("readme"),
        payload.get("readme_md"), getattr(document, "readme_md", None),
        getattr(document, "clean_text", None), payload.get("clean_text"),
    ):
        if isinstance(value, str) and value[:MAX_PREPROCESS_CHARS].strip():
            return value[:MAX_PREPROCESS_CHARS]
    paragraphs = payload.get("extracted_paragraphs")
    if isinstance(paragraphs, list):
        pieces, remaining = [], MAX_PREPROCESS_CHARS
        for value in paragraphs[:80]:
            if isinstance(value, str) and remaining > 0:
                piece = value[:remaining]
                pieces.append(piece)
                remaining -= len(piece) + 2
        return "\n\n".join(pieces)[:MAX_PREPROCESS_CHARS]
    return paragraphs[:MAX_PREPROCESS_CHARS] if isinstance(paragraphs, str) else ""


def _sections(markdown: str) -> list[tuple[str, str]]:
    """Recognize ATX/Setext headings, inheritance and fenced code boundaries."""
    lines = markdown[:MAX_PREPROCESS_CHARS].splitlines()
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []
    category = "neutral"
    fence: tuple[str, int] | None = None

    def flush() -> None:
        if category != "omit":
            text = clean_markdown_copy("\n".join(body)).strip()
            if text:
                sections.append((category, text))
        body.clear()

    index = 0
    while index < len(lines):
        check_deadline()
        line = lines[index]
        opening = _FENCE.match(line)
        if fence is not None:
            if opening and opening[1][0] == fence[0] and len(opening[1]) >= fence[1] and not line[opening.end():].strip():
                fence = None
            index += 1
            continue
        if opening:
            fence = (opening[1][0], len(opening[1]))
            index += 1
            continue
        heading = _HEADING.match(line)
        setext = (
            index + 1 < len(lines) and bool(line.strip())
            and not line.lstrip().startswith(("- ", "* ", ">", "|"))
            and _SETEXT.match(lines[index + 1])
        )
        if heading or setext:
            flush()
            depth = len(heading[1]) if heading else (1 if lines[index + 1].lstrip().startswith("=") else 2)
            title = heading[2] if heading else line.strip()
            # Emoji and inline markup commonly decorate README headings.
            title = re.sub(r"^[^\w]+", "", clean_markdown_copy(title)).strip()
            while stack and stack[-1][0] >= depth:
                stack.pop()
            inherited = stack[-1][1] if stack else "neutral"
            if inherited == "omit" or _OMIT.fullmatch(title):
                category = "omit"
            elif _SETUP.search(title):
                category = "setup"
            elif _PURPOSE.search(title):
                category = "purpose"
            else:
                category = inherited
            stack.append((depth, category))
            index += 1 if heading else 2
            continue
        body.append(line)
        index += 1
    flush()
    return sections


def token_chunks(text: str, tokenizer: Any, *, sequence_limit: int, overlap_tokens: int,
                 max_chunks: int | None = None) -> list[tuple[str, int]]:
    """Slice source offsets, then verify the exact text that the encoder sees."""
    if isinstance(sequence_limit, bool) or not isinstance(sequence_limit, int) or not 1 <= sequence_limit <= 2048:
        raise ValueError("encoder max_seq_length must be between 1 and 2048")
    special = tokenizer.num_special_tokens_to_add(pair=False)
    budget = sequence_limit - special
    if isinstance(overlap_tokens, bool) or not isinstance(overlap_tokens, int) or not 0 <= overlap_tokens < budget:
        raise ValueError("README token overlap must be smaller than the model content token budget")
    encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    chunks: list[tuple[str, int]] = []
    start = 0
    while start < len(offsets):
        check_deadline()
        end = min(start + budget, len(offsets))
        while end > start:
            chunk = text[offsets[start][0]:offsets[end - 1][1]]
            count = len(tokenizer(chunk, add_special_tokens=True, truncation=False)["input_ids"])
            if count <= sequence_limit:
                break
            end -= 1
        if end == start:
            raise ValueError("tokenizer cannot represent one source token within max_seq_length")
        if chunk.strip():
            chunks.append((chunk, count))
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
        if end == len(offsets):
            break
        start = max(start + 1, end - overlap_tokens)
    return chunks


def prepare_readme_chunks(markdown: str, tokenizer: Any, *, sequence_limit: int, overlap_tokens: int = 32) -> list[ReadmeChunk]:
    """Build deterministic inputs with class budgets independent of length."""
    selected: dict[str, list[tuple[str, int]]] = {key: [] for key in SECTION_WEIGHTS}
    for category, text in _sections(markdown):
        remaining = SECTION_CHUNK_LIMITS[category] - len(selected[category])
        if remaining <= 0:
            continue
        # Bound tokenization and emitted chunks before encoder admission.
        selected[category].extend(token_chunks(
            text, tokenizer, sequence_limit=sequence_limit, overlap_tokens=overlap_tokens,
            max_chunks=remaining,
        ))
    return [
        ReadmeChunk(text, category, count, SECTION_WEIGHTS[category] / len(chunks))
        for category, chunks in selected.items() for text, count in chunks
    ]
