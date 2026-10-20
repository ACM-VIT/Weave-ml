"""README fetching and markdown cleaning for Osiris ingestion payloads."""

from __future__ import annotations

import base64
import html
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


# Derived views are bounded before any decoding/cleanup; canonical Markdown is
# retained separately in ReadmeDocument and in backend ingestion payloads.
MAX_PREPROCESS_CHARS = 64_000
# Match only the link prefix here. A single stack scan supplies balanced target
# endings, so malformed or deeply nested targets cannot trigger suffix rescans.
_LINK_START_RE = re.compile(r"(!?)\[([^\[\]]*)\]\(")
_HTML_TAG_RE = re.compile(r"<[^<>]+>")
_FENCE_RE = re.compile(r"```.*?```", re.S)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*")
_BULLET_RE = re.compile(r"^[ \t]*[-*+][ \t]+")
_ORDERED_RE = re.compile(r"^[ \t]*\d+[.)][ \t]+")
_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*(table of contents|toc)\s*$", re.I),
    re.compile(r"^\s*(license|copyright)\s*$", re.I),
    re.compile(r"^\s*(built with|powered by)\s*$", re.I),
    re.compile(r"^\s*(npm install|pip install|cargo install|go get)\b", re.I),
]


@dataclass(slots=True)
class ReadmeDocument:
    raw_markdown: str
    clean_text: str
    extracted_paragraphs: list[str]
    readme_length: int
    readme_md: str = ""
    source_path: str | None = None
    default_branch: str | None = None
    base_url: str | None = None


def decode_readme_payload(payload: dict[str, Any] | None) -> str:
    """Decode a GitHub README API payload into markdown text."""
    if not payload:
        return ""
    content = payload.get("content") or ""
    encoding = (payload.get("encoding") or "").lower()
    if encoding == "base64":
        try:
            return base64.b64decode(content, validate=False).decode("utf-8", errors="replace")
        except Exception:
            return ""
    if isinstance(content, str):
        return content
    return ""


def process_readme_payload(payload: dict[str, Any] | None) -> ReadmeDocument:
    return process_markdown(decode_readme_payload(payload))


def process_markdown(
    markdown: str,
    *,
    source_path: str | None = None,
    default_branch: str | None = None,
    base_url: str | None = None,
) -> ReadmeDocument:
    """Clean markdown and extract paragraphs useful to Osiris semantic stages."""
    raw = markdown or ""
    text = clean_markdown_copy(raw)

    paragraphs: list[str] = []
    for block in text.split("\n\n"):
        if not _is_meaningful_paragraph(block):
            continue
        paragraphs.append(block)

    clean_text = "\n\n".join(paragraphs)
    return ReadmeDocument(
        raw_markdown=raw,
        clean_text=clean_text,
        extracted_paragraphs=paragraphs[:80],
        readme_length=len(raw),
        readme_md="",
        source_path=source_path,
        default_branch=default_branch,
        base_url=base_url,
    )


def clean_markdown_copy(markdown: str) -> str:
    """Return derived plain text while leaving canonical Markdown untouched."""

    text = html.unescape((markdown or "")[:MAX_PREPROCESS_CHARS])
    text = _FENCE_RE.sub(" ", text)
    text = clean_inline_markdown(text)
    text = _INLINE_CODE_RE.sub(r"\1", text)

    paragraphs: list[str] = []
    lines: list[str] = []
    # Process each line once. Multiline ^\s* patterns retry the complete
    # remaining blank-line suffix at every newline, even on bounded input.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            line = ""
        line = _ORDERED_RE.sub("", _BULLET_RE.sub("", _HEADING_RE.sub("", line)))
        normalized = " ".join(line.split())
        if normalized:
            lines.append(normalized)
            continue
        block = " ".join(lines).strip(" -*_\t\n")
        if block:
            paragraphs.append(block)
        lines.clear()
    block = " ".join(lines).strip(" -*_\t\n")
    if block:
        paragraphs.append(block)
    return "\n\n".join(paragraphs)


def clean_inline_markdown(text: str) -> str:
    """Remove images, badge links and HTML using bounded linear scans."""
    text = text[:MAX_PREPROCESS_CHARS]
    pieces: list[str] = []
    previous = 0
    for match, end in _inline_markdown_links(text):
        image, label = match.groups()
        target = text[match.end():end - 1].casefold()
        replacement = " " if image or "badge" in target or "shields.io" in target else label
        pieces.extend((text[previous:match.start()], replacement))
        previous = end
    pieces.append(text[previous:])
    return _HTML_TAG_RE.sub(" ", "".join(pieces))


def _inline_markdown_links(text: str) -> Iterator[tuple[re.Match[str], int]]:
    """Yield complete flat-label links, including balanced/escaped target parentheses."""
    endings: dict[int, int] = {}
    stack: list[int] = []
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "(":
            stack.append(index)
        elif char == ")" and stack:
            endings[stack.pop()] = index + 1

    consumed = 0
    for match in _LINK_START_RE.finditer(text):
        if match.start() < consumed:
            continue
        end = endings.get(match.end() - 1)
        if end is not None:
            yield match, end
            consumed = end


def contains_inline_markdown_link(text: str) -> bool:
    """Use the same bounded link grammar for cleanup and plain-text validation."""
    return next(_inline_markdown_links(text[:MAX_PREPROCESS_CHARS]), None) is not None


def _is_meaningful_paragraph(text: str) -> bool:
    if len(text) < 40:
        return False
    if len(text.split()) < 6:
        return False
    if any(pattern.search(text) for pattern in _BOILERPLATE_PATTERNS):
        return False
    alpha_chars = sum(1 for char in text if char.isalpha())
    if alpha_chars / max(len(text), 1) < 0.45:
        return False
    return True
