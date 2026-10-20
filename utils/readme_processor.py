"""README fetching and markdown cleaning for Osiris ingestion payloads."""

from __future__ import annotations

import base64
import html
import re
from dataclasses import dataclass
from typing import Any


# Derived views are bounded before any decoding/cleanup; canonical Markdown is
# retained separately in ReadmeDocument and in backend ingestion payloads.
MAX_PREPROCESS_CHARS = 64_000
# Excluding opening delimiters prevents each malformed opener from rescanning
# the complete suffix (e.g. '[' * 1_000_000 or '<' * 1_000_000).
_LINK_RE = re.compile(r"(!?)\[([^\[\]]*)\]\(([^()]*?)\)")
_HTML_TAG_RE = re.compile(r"<[^<>]+>")
_FENCE_RE = re.compile(r"```.*?```", re.S)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MARKDOWN_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$", re.M)
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
    for block in re.split(r"\n\s*\n+", text):
        block = re.sub(r"\s+", " ", block).strip(" -*_\t\n")
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
    text = _MARKDOWN_TABLE_RE.sub(" ", text)
    text = clean_inline_markdown(text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.M)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n+", text):
        block = re.sub(r"\s+", " ", block).strip(" -*_\t\n")
        if block:
            paragraphs.append(block)
    return "\n\n".join(paragraphs)


def clean_inline_markdown(text: str) -> str:
    """Remove images, badge links and HTML using delimiter-safe scans."""
    def link_text(match: re.Match[str]) -> str:
        image, label, target = match.groups()
        lowered = target.casefold()
        if image or any(token in lowered for token in ("badge", "shields.io")):
            return " "
        return label

    return _HTML_TAG_RE.sub(" ", _LINK_RE.sub(link_text, text[:MAX_PREPROCESS_CHARS]))


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
