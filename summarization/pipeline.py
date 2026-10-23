"""Generate, repair, validate, and safely fall back for card summaries."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import TimeoutError
from dataclasses import replace
import hashlib
import json
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
from .provider import SUMMARY_COMPLETION_RESERVE_SECONDS, SummaryProvider, SummaryProviderError
from . import reuse
from .reuse import SummaryReuseCache, SummaryWork, source_identity
from .settings import SummarySettings
from .source import build_summary_source
from .sentences import (
    has_complete_sentence_ending,
    has_two_or_three_complete_sentences,
    mask_summary_abbreviations,
    split_summary_sentences,
)
from .validation import (
    analyze_comparison_texts,
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
    if not has_two_or_three_complete_sentences(summary) or _unsafe_text(summary):
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
        reuse_cache: SummaryReuseCache | None = None,
    ) -> None:
        self.settings = settings or SummarySettings()
        self.provider = provider
        self.reuse_cache = reuse_cache or SummaryReuseCache()

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
            key = self._reuse_key(repo, repo) if repo is not None else None
            if repo is not None and (key is None or artifact._reuse_key != key.hex()):
                return False
            work = self._analysis(repo, repo, key) if repo is not None else None
            material = work.material if work is not None else None
            if material is not None and material.analysis_limit is not None:
                # Do not replay generated content against only a prefix of its
                # source. A later attempt can still persist the safe fallback.
                return False
            try:
                validate_generated_summary(
                    artifact,
                    source_analysis=work.analysis if work is not None else (),
                )
            except SummaryValidationError:
                return False
            return True
        return artifact.model_version == CARD_SUMMARY_FALLBACK_MODEL_VERSION

    def should_upgrade(self, artifact: CardSummaryArtifact, *, repo: Mapping[str, Any] | None = None) -> bool:
        """Keep durable fallback replay independent from provider availability."""
        deferred = getattr(self.provider, "fallback_deferred", None)
        return (artifact.source == "description_fallback" and self.provider is not None
                and not (repo is not None and callable(deferred) and deferred(repo)))

    def _reuse_key(self, source: Any, repo: Mapping[str, Any]) -> bytes | None:
        identity = source_identity(source, repo)
        if identity is None:
            return None
        versions = (
            self.settings.provider, self.settings.api_url, self.settings.model_id,
            self.settings.prompt_version, self.settings.format_version,
            reuse.SUMMARY_VALIDATION_VERSION, reuse.SOURCE_ANALYSIS_VERSION,
            self.settings.input_max_chars, self.settings.temperature,
            self.settings.max_output_tokens,
        )
        return hashlib.sha256(identity + json.dumps(versions).encode("utf-8")).digest()

    def _analysis(self, source: Any, repo: Mapping[str, Any], key: bytes | None) -> SummaryWork:
        work = self.reuse_cache.get(key)
        if work is None:
            material = build_summary_source(source, repo, max_chars=self.settings.input_max_chars)
            work = SummaryWork(material, analyze_comparison_texts(material.comparison_texts))
            self.reuse_cache.put(key, work)
        return work

    def _fallback(self, repo: Mapping[str, Any]) -> CardSummaryArtifact:
        return description_fallback_artifact(
            repo, prompt_version=self.settings.prompt_version,
            format_version=self.settings.format_version,
        )

    def summarize(self, source: Any, repo: Mapping[str, Any], *, deadline: float | None = None) -> CardSummaryArtifact:
        deferred = getattr(self.provider, "fallback_deferred", None)
        if self.provider is None or (callable(deferred) and deferred(repo)):
            return self._fallback(repo)
        key = self._reuse_key(source, repo)
        if key is None:
            return self._generate(self._analysis(source, repo, None), repo, deadline)
        future, leader = self.reuse_cache.start(key)
        if future is None:
            return self._fallback(repo)
        if not leader:
            # A wait must leave the same completion reserve as a provider call.
            # Without a caller deadline, use one configured request timeout.
            timeout = self.settings.request_timeout_seconds if deadline is None else max(
                0.0, deadline - time.monotonic() - SUMMARY_COMPLETION_RESERVE_SECONDS,
            )
            try:
                artifact = future.result(timeout=timeout)
                return artifact.model_copy(deep=True)
            except TimeoutError:
                return self._fallback(repo)
        try:
            work = self._analysis(source, repo, key)
            if work.artifact is not None:
                self.reuse_cache.count("artifact_hits")
                artifact = work.artifact.model_copy(deep=True)
            else:
                artifact = self._generate(work, repo, deadline)
                if artifact.source == "generated":
                    artifact._reuse_key = key.hex()
                    self.reuse_cache.put(key, replace(work, artifact=artifact.model_copy(deep=True)))
            self.reuse_cache.finish(key, future, artifact=artifact.model_copy(deep=True))
            return artifact
        except BaseException as exc:
            self.reuse_cache.finish(key, future, error=exc)
            raise

    def _generate(self, work: SummaryWork, repo: Mapping[str, Any], deadline: float | None) -> CardSummaryArtifact:
        material = work.material
        if material.analysis_limit is None and material.prompt_input:
            repair_feedback: str | None = None
            for _attempt in range(2):
                if deadline is not None and time.monotonic() > deadline:
                    break
                try:
                    self.reuse_cache.count("provider_calls")
                    raw = self.provider.generate(
                        material.prompt_input,
                        repair_feedback=repair_feedback,
                        deadline=deadline,
                    )
                    content = validate_generated_summary(
                        parse_summary_response(raw), source_analysis=work.analysis,
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
                except SummaryProviderError as exc:
                    defer = getattr(self.provider, "defer_fallback", None)
                    if exc.defer_upgrade and callable(defer):
                        defer(repo)
                    break
                except (ValueError, TypeError):
                    break
        return self._fallback(repo)
