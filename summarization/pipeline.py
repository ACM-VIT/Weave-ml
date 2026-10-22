"""Generate, repair, validate, and safely fall back for card summaries."""

from __future__ import annotations

from collections.abc import Mapping
import re
import time
from typing import Any

from utils.readme_processor import MAX_PREPROCESS_CHARS, clean_inline_markdown

from .contracts import (
    CARD_SUMMARY_FALLBACK_MODEL_VERSION,
    CARD_SUMMARY_FORMAT_VERSION,
    CARD_SUMMARY_MAX_CHARS,
    CARD_SUMMARY_PROMPT_VERSION,
    CardSummaryArtifact,
)
from .provider import SummaryProvider, SummaryProviderError
from .settings import SummarySettings
from .source import build_summary_source
from .sentences import (
    has_complete_sentence_ending,
    has_two_or_three_complete_sentences,
    mask_summary_abbreviations,
    split_summary_sentences,
)
from .validation import (
    _LICENSE_BOILERPLATE,
    _unsafe_text,
    SummaryValidationError,
    normalize_summary,
    parse_summary_response,
    validate_generated_summary,
)


_UNSAFE_SENTENCE = re.compile(
    r"(?:```|`|\b(?:npm|pnpm|yarn|pipx?|uv|cargo|go)\s+"
    r"(?:install|add|run|get)\b|\b(?:brew|apt(?:-get)?|dnf|yum|pacman|choco|"
    r"winget)\s+install\b|\bgit\s+clone\b|\b(?:curl|wget)\s+https?://|"
    r"\b(?:installation|installing|setup|getting started|quick ?start|changelog|"
    r"licen[cs]e|contributors?|contributing|api\s+(?:reference|docs?)|reference|"
    r"table of contents|badges?)\b)",
    re.IGNORECASE,
)
_MARKDOWN_STRUCTURE = re.compile(
    r"(?m)^[ \t]*(?:#{1,6}[ \t]+|[-*+][ \t]+|\d+[.)][ \t]+)|```|~~~|`"
)
_DUNDER_IDENTIFIER = re.compile(r"__([A-Za-z_][A-Za-z0-9_]*)__")
_INLINE_MARKDOWN_PUNCTUATION = re.compile(r"[*_~]+")
_URL = re.compile(r"https?://\S+")


def _bounded(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    candidate = value[: maximum + 1].rsplit(" ", 1)[0].rstrip(" ,;:-")
    if not candidate:
        candidate = value[:maximum].rstrip(" ,;:-")
    if candidate and not has_complete_sentence_ending(candidate):
        candidate = candidate[: maximum - 1].rstrip(" ,;:-") + "."
    return candidate[:maximum]


def _plain_description(value: Any) -> str:
    """Reduce a repository description to plain prose for the metadata fallback."""

    text = clean_inline_markdown(str(value or "")[:MAX_PREPROCESS_CHARS])
    text = _URL.sub("", text)
    text = _MARKDOWN_STRUCTURE.sub("", text)
    # Spell Python special-method identifiers as prose before stripping emphasis.
    # This keeps ``__getitem__`` distinguishable from ``getitem`` without leaking
    # double underscores that Markdown renderers interpret as emphasis markers.
    text = _DUNDER_IDENTIFIER.sub(
        lambda match: f"dunder {match.group(1).replace('_', ' ')}",
        text,
    )
    # Unbalanced emphasis markers, wildcards, and snake-case separators are not
    # useful formatting in a discovery card. Render their words as plain prose.
    text = _INLINE_MARKDOWN_PUNCTUATION.sub(" ", text)
    return normalize_summary(text)


def description_fallback_artifact(
    repo: Mapping[str, Any],
    *,
    prompt_version: str = CARD_SUMMARY_PROMPT_VERSION,
    format_version: str = CARD_SUMMARY_FORMAT_VERSION,
) -> CardSummaryArtifact:
    """Build a bounded metadata fallback that never reads README content."""

    raw_description = _plain_description(repo.get("description"))
    safe_sentences = [
        normalize_summary(sentence)
        for sentence in split_summary_sentences(raw_description)
        if sentence.strip() and not _UNSAFE_SENTENCE.search(sentence)
        and not _LICENSE_BOILERPLATE.search(mask_summary_abbreviations(sentence))
    ]
    if len(safe_sentences) >= 2:
        first = _bounded(safe_sentences[0], 175)
        second = _bounded(
            safe_sentences[1],
            CARD_SUMMARY_MAX_CHARS - len(first) - 1,
        )
        summary = f"{first} {second}".strip()
    else:
        summary = " ".join(safe_sentences).strip()
    if not summary:
        full_name = _plain_description(repo.get("full_name") or "This repository")
        language = _plain_description(repo.get("primary_language"))
        topics = repo.get("topics") if isinstance(repo.get("topics"), list) else []
        topic_text = ", ".join(
            item for item in (_plain_description(value) for value in topics[:3]) if item
        )
        summary = f"{full_name} is a GitHub repository."
        if language and language.casefold() != "unknown":
            summary += f" Its primary language is {language}."
        elif topic_text:
            summary += f" Its published topics include {topic_text}."
        else:
            summary += " Its description does not yet provide more project details."
    elif len(safe_sentences) == 1:
        language = _plain_description(repo.get("primary_language"))
        topics = repo.get("topics") if isinstance(repo.get("topics"), list) else []
        topic_text = ", ".join(
            item for item in (_plain_description(value) for value in topics[:3]) if item
        )
        full_name = _plain_description(repo.get("full_name") or "this repository")
        if language and language.casefold() != "unknown":
            supplement = f"Its primary language is {language}."
        elif topic_text:
            supplement = f"Its published topics include {topic_text}."
        else:
            supplement = f"It is published as {full_name}."
        summary = _bounded(
            summary,
            CARD_SUMMARY_MAX_CHARS - len(supplement) - 1,
        )
        if summary and not has_complete_sentence_ending(summary):
            summary = f"{summary}."
        summary = f"{summary} {supplement}"
    summary = _bounded(summary, CARD_SUMMARY_MAX_CHARS)
    if summary and not has_complete_sentence_ending(summary):
        summary = _bounded(summary + '.', CARD_SUMMARY_MAX_CHARS)
    # Truncation or unusual metadata must not create a permanently undeliverable
    # fallback. Keep the same sentence requirement as the canonical backend.
    if not has_two_or_three_complete_sentences(summary):
        summary = 'This is a GitHub repository. Its description does not yet provide more project details.'
    return CardSummaryArtifact(
        summary=summary,
        highlights=[],
        prompt_version=prompt_version,
        model_version=CARD_SUMMARY_FALLBACK_MODEL_VERSION,
        format_version=format_version,
        source="description_fallback",
    )


class CardSummaryPipeline:
    """One generation attempt, at most one repair, then safe fallback."""

    def __init__(
        self,
        *,
        settings: SummarySettings | None = None,
        provider: SummaryProvider | None = None,
    ) -> None:
        self.settings = settings or SummarySettings()
        self.provider = provider

    def is_current(self, artifact: CardSummaryArtifact, *, repo: Mapping[str, Any] | None = None) -> bool:
        if (
            artifact.prompt_version != self.settings.prompt_version
            or artifact.format_version != self.settings.format_version
            or not has_two_or_three_complete_sentences(artifact.summary)
        ):
            return False
        # Fingerprints establish integrity, not acceptance under today's rules.
        # Recheck stored content before duplicate/current replies bypass generation.
        for text in [artifact.summary, *artifact.highlights]:
            if _unsafe_text(text) or normalize_summary(text) != text:
                return False
        if artifact.source == "generated":
            if artifact.model_version != self.settings.model_id:
                return False
            material = build_summary_source(repo, repo, max_chars=self.settings.input_max_chars) if repo is not None else None
            try:
                validate_generated_summary(
                    artifact,
                    comparison_texts=material.comparison_texts if material is not None else (),
                )
            except SummaryValidationError:
                return False
            return True
        return artifact.model_version == CARD_SUMMARY_FALLBACK_MODEL_VERSION

    def should_upgrade(self, artifact: CardSummaryArtifact) -> bool:
        """Keep durable fallback replay independent from provider availability."""
        return artifact.source == "description_fallback" and self.provider is not None

    def summarize(self, source: Any, repo: Mapping[str, Any], *, deadline: float | None = None) -> CardSummaryArtifact:
        material = build_summary_source(
            source,
            repo,
            max_chars=self.settings.input_max_chars,
        )
        if self.provider is not None and material.prompt_input:
            repair_feedback: str | None = None
            for _attempt in range(2):
                if deadline is not None and time.monotonic() > deadline:
                    break
                try:
                    raw = self.provider.generate(
                        material.prompt_input,
                        repair_feedback=repair_feedback,
                        deadline=deadline,
                    )
                    content = validate_generated_summary(
                        parse_summary_response(raw),
                        comparison_texts=material.comparison_texts,
                    )
                    return CardSummaryArtifact(
                        **content.model_dump(),
                        prompt_version=self.settings.prompt_version,
                        model_version=self.settings.model_id,
                        format_version=self.settings.format_version,
                        source="generated",
                    )
                except SummaryValidationError as exc:
                    repair_feedback = str(exc)
                except (SummaryProviderError, ValueError, TypeError):
                    break
        return description_fallback_artifact(
            repo,
            prompt_version=self.settings.prompt_version,
            format_version=self.settings.format_version,
        )
