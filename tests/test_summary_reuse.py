"""Coverage equivalence, identity invalidation, memory and concurrency regressions."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from difflib import SequenceMatcher
import json
import random
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from summarization import reuse
from summarization.pipeline import CardSummaryPipeline
from summarization.provider import SummaryProviderError
from summarization.reuse import SummaryReuseCache, source_identity
from summarization.settings import SummarySettings
from summarization.validation import (
    _near_copy, _words, analyze_comparison_texts, parse_summary_response,
    validate_generated_summary,
)


SUMMARY = (
    "A focused workspace gives engineering teams a clear view of proposed software changes. "
    "It brings ownership decisions and review history together so maintainers can coordinate their work."
)
REPO = {"repo_id": "repo-88", "full_name": "weave/review", "description": "Tools for reviewing changes.",
        "readme": "# Overview\nA collaborative tool supports maintainers working across shared projects."}


class Provider:
    def __init__(self):
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        return json.dumps({"summary": SUMMARY, "highlights": ["Shared review tools"]})


def reference_near_copy(summary, source):
    """Pre-optimization algorithm kept as an independent detection oracle."""
    words, source_words = _words(summary), _words(source)
    joined = " ".join(words)
    if len(joined) >= 80 and joined in " ".join(source_words):
        return True
    if min(len(words), len(source_words)) < 8:
        return False
    n = min(12, len(words))
    grams = {tuple(words[i:i+n]) for i in range(len(words)-n+1)}
    if any(tuple(source_words[i:i+n]) in grams for i in range(len(source_words)-n+1)):
        return True
    if len(source_words) < len(words):
        return SequenceMatcher(None, words, source_words, autojunk=False).ratio() >= .84
    last = len(source_words) - len(words)
    starts = list(range(0, last+1, max(1, len(words)//4)))
    if starts[-1] != last:
        starts.append(last)
    return any(SequenceMatcher(None, words, source_words[i:i+len(words)], autojunk=False).ratio() >= .84
               for i in starts)


def test_window_shortlist_matches_original_on_randomized_and_adversarial_inputs():
    rng = random.Random(88)
    for _ in range(400):
        vocabulary = [f"token{i}" for i in range(rng.choice([3, 12, 80]))]
        source = [rng.choice(vocabulary) for _ in range(rng.randint(8, 240))]
        start = rng.randrange(len(source))
        candidate = source[start:start + rng.randint(8, 65)]
        for i in range(len(candidate)):
            if rng.random() < .15:
                candidate[i] = "replacement"
        source, candidate = " ".join(source), " ".join(candidate)
        assert _near_copy(candidate, source) == reference_near_copy(candidate, source)


@pytest.mark.parametrize("length", [8, 9, 10, 11, 12, 36, 53])
def test_tail_and_off_grid_copies_keep_detection_coverage(length):
    tokens = [f"word{i}" for i in range(length)]
    # Break every long exact ngram; force the SequenceMatcher branch for larger
    # outputs. The final window must still be checked when it is off-grid.
    source_tokens = list(tokens)
    if length > 12:
        for i in range(6, length, 9):
            source_tokens[i] = "altered"
    candidate = " ".join(tokens)
    source = "unrelated prefix material " * 1900 + " ".join(source_tokens)
    assert reference_near_copy(candidate, source)
    assert _near_copy(candidate, source)


def test_disjoint_source_skips_sequence_matcher():
    with patch("summarization.validation.SequenceMatcher", side_effect=AssertionError("impossible match")):
        assert not _near_copy(SUMMARY, "unrelated tokens only " * 2000)


def test_analysis_is_shared_by_summary_highlights_and_repair():
    provider = Provider()
    original = provider.generate
    def generate(*args, **kwargs):
        if not provider.calls:
            provider.calls += 1
            return json.dumps({"summary": "Invalid output."})
        return original(*args, **kwargs)
    provider.generate = generate
    pipeline = CardSummaryPipeline(provider=provider)
    with patch("summarization.pipeline.analyze_comparison_texts", wraps=analyze_comparison_texts) as analyze:
        result = pipeline.summarize(REPO, REPO)
        again = pipeline.summarize(REPO, REPO)
    assert result == again
    assert result.source == "generated"
    assert analyze.call_count == 1
    assert provider.calls == 2
    assert pipeline.reuse_cache.snapshot()["artifact_hits"] == 1


@pytest.mark.parametrize("field,value", [
    ("repo_id", "another-repo"), ("github_id", "456"), ("full_name", "weave/another"),
    ("content_hash", "changed"), ("readme", REPO["readme"] + " New evidence."),
    ("clean_text", "Another complete source variant."), ("readme_md", "More Markdown evidence."),
    ("extracted_paragraphs", ["Additional complete evidence for the project."]),
    ("description", "Different metadata."), ("topics", ["collaboration"]),
    ("languages", ["Python"]), ("primary_language", "Python"),
])
def test_source_changes_invalidate_artifact_reuse(field, value):
    provider = Provider()
    pipeline = CardSummaryPipeline(provider=provider)
    assert pipeline.summarize(REPO, REPO).source == "generated"
    changed = {**REPO, field: value}
    assert pipeline.summarize(changed, changed).source == "generated"
    assert provider.calls == 2


@pytest.mark.parametrize("field,value", [
    ("model_id", "another/model"), ("prompt_version", "next-prompt"),
    ("format_version", "next-format"), ("input_max_chars", 1000),
])
def test_model_prompt_and_format_changes_invalidate_reuse(field, value):
    provider = Provider()
    pipeline = CardSummaryPipeline(provider=provider)
    pipeline.summarize(REPO, REPO)
    # Future releases update these pinned constants together; bypass the settings
    # policy here to exercise cache version separation independently.
    settings = replace(pipeline.settings)
    object.__setattr__(settings, field, value)
    pipeline.settings = settings
    pipeline.summarize(REPO, REPO)
    assert provider.calls == 2


@pytest.mark.parametrize("version", ["SUMMARY_VALIDATION_VERSION", "SOURCE_ANALYSIS_VERSION"])
def test_policy_versions_invalidate_artifact_reuse(monkeypatch, version):
    provider = Provider()
    pipeline = CardSummaryPipeline(provider=provider)
    pipeline.summarize(REPO, REPO)
    monkeypatch.setattr(reuse, version, "next-version")
    pipeline.summarize(REPO, REPO)
    assert provider.calls == 2


def test_unprompted_tail_and_object_evidence_invalidate_reuse():
    provider = Provider()
    pipeline = CardSummaryPipeline(provider=provider, settings=SummarySettings(input_max_chars=1000))
    repo = {**REPO, "readme": "An introductory passage about shared tools. " * 1000}
    source = SimpleNamespace(readme=SimpleNamespace(raw_markdown="Original object evidence."))
    pipeline.summarize(source, repo)
    source.readme.raw_markdown += " Changed evidence."
    pipeline.summarize(source, repo)
    changed = {**repo, "readme": repo["readme"] + " Changed at the very end."}
    pipeline.summarize(source, changed)
    assert provider.calls == 3


def test_transient_fallback_is_regenerated_and_returned_artifacts_are_isolated():
    provider = Provider()
    original = provider.generate
    def generate(*args, **kwargs):
        if not provider.calls:
            provider.calls += 1
            raise SummaryProviderError("temporary outage")
        return original(*args, **kwargs)
    provider.generate = generate
    pipeline = CardSummaryPipeline(provider=provider)
    assert pipeline.summarize(REPO, REPO).source == "description_fallback"
    generated = pipeline.summarize(REPO, REPO)
    assert generated.source == "generated"
    generated.highlights.append("Mutated by caller")
    generated.summary = "Mutated by caller."
    cached = pipeline.summarize(REPO, REPO)
    assert cached.summary == SUMMARY
    assert cached.highlights == ["Shared review tools"]
    assert provider.calls == 2


def test_duplicate_work_coalesces_and_waiter_deadline_preserves_reserve():
    entered, release = threading.Event(), threading.Event()
    class Blocking(Provider):
        def generate(self, *args, **kwargs):
            entered.set()
            assert release.wait(3)
            return super().generate(*args, **kwargs)
    provider = Blocking()
    pipeline = CardSummaryPipeline(provider=provider)
    with ThreadPoolExecutor(max_workers=4) as executor:
        first = executor.submit(pipeline.summarize, REPO, REPO)
        assert entered.wait(1)
        second = executor.submit(pipeline.summarize, REPO, REPO)
        try:
            timed_out = pipeline.summarize(REPO, REPO, deadline=time.monotonic() + 1)
            assert timed_out.source == "description_fallback"
            assert pipeline.reuse_cache.snapshot()["inflight"] == 1
        finally:
            release.set()
        assert first.result().source == second.result().source == "generated"
    assert provider.calls == 1
    assert pipeline.reuse_cache.snapshot()["inflight"] == 0


def test_flight_failure_releases_slot_and_next_call_can_regenerate():
    provider = Provider()
    pipeline = CardSummaryPipeline(provider=provider)
    with patch.object(provider, "generate", side_effect=RuntimeError("unexpected")):
        with pytest.raises(RuntimeError, match="unexpected"):
            pipeline.summarize(REPO, REPO)
    assert pipeline.reuse_cache.snapshot()["inflight"] == 0
    assert pipeline.summarize(REPO, REPO).source == "generated"


def test_cache_entry_byte_and_inflight_limits():
    cache = SummaryReuseCache(max_entries=2, max_bytes=6000, max_inflight=1)
    provider = Provider()
    pipeline = CardSummaryPipeline(provider=provider, reuse_cache=cache)
    for i in range(12):
        repo = {**REPO, "repo_id": str(i)}
        pipeline.summarize(repo, repo)
        assert cache.snapshot()["entries"] <= 2
        assert cache.snapshot()["retained_bytes"] <= 6000
    first, leader = cache.start(b"first")
    assert leader
    assert cache.start(b"another") == (None, False)
    assert cache.start(b"first") == (first, False)
    cache.finish(b"first", first, artifact=pipeline._fallback(REPO))
    tiny = SummaryReuseCache(max_bytes=1)
    small_pipeline = CardSummaryPipeline(provider=Provider(), reuse_cache=tiny)
    small_pipeline.summarize(REPO, REPO)
    small_pipeline.summarize(REPO, REPO)
    assert tiny.snapshot()["entries"] == 0
    assert tiny.snapshot()["provider_calls"] == 2


def test_oversized_sources_bypass_cache_and_fail_closed():
    repo = {**REPO, "readme": "x" * 64001}
    provider = Provider()
    pipeline = CardSummaryPipeline(provider=provider)
    assert source_identity(repo, repo) is None
    assert pipeline.summarize(repo, repo).source == "description_fallback"
    assert pipeline.reuse_cache.snapshot()["entries"] == 0
    assert provider.calls == 0


def test_prepared_and_direct_validation_have_identical_results():
    content = parse_summary_response(json.dumps({"summary": SUMMARY}))
    texts = (REPO["readme"], "An extra source variant for validation.")
    assert validate_generated_summary(content, comparison_texts=texts) == validate_generated_summary(
        content, source_analysis=analyze_comparison_texts(texts),
    )


def test_durable_provenance_survives_restart_without_changing_public_json(monkeypatch):
    from summarization.contracts import card_summary_from_payload, card_summary_payload

    pipeline = CardSummaryPipeline(provider=Provider())
    artifact = pipeline.summarize(REPO, REPO)
    payload = card_summary_payload(artifact)
    restored = card_summary_from_payload(payload)
    restarted = CardSummaryPipeline(provider=Provider())
    assert restored is not None
    assert restarted.is_current(restored, repo=REPO)
    assert set(restored.model_dump()) == {
        "summary", "highlights", "prompt_version", "model_version", "format_version", "source",
    }
    assert not restarted.is_current(restored, repo={**REPO, "readme": REPO["readme"] + " Changed."})
    monkeypatch.setattr(reuse, "SUMMARY_VALIDATION_VERSION", "next-policy")
    assert not restarted.is_current(restored, repo=REPO)
    payload["card_summary_reuse_key"] = "0" * 64
    assert card_summary_from_payload(payload) is None


def test_legacy_generated_artifact_requires_source_key_before_durable_reuse():
    from summarization.contracts import CardSummaryArtifact, card_summary_from_payload, card_summary_payload

    pipeline = CardSummaryPipeline(provider=Provider())
    generated = pipeline.summarize(REPO, REPO)
    legacy = CardSummaryArtifact.model_validate(generated.model_dump())
    restored = card_summary_from_payload(card_summary_payload(legacy))
    assert restored is not None
    assert not pipeline.is_current(restored, repo=REPO)
    # Revalidation must still protect a matching-key artifact after tampering.
    generated.summary = "Run npm install to start. This is a useful tool."
    assert not pipeline.is_current(generated, repo=REPO)


def test_full_api_topic_and_language_limits_have_durable_provenance():
    provider = Provider()
    pipeline = CardSummaryPipeline(provider=provider)
    repo = {**REPO, "topics": [f"topic-{i}" for i in range(200)],
            "languages": [f"language-{i}" for i in range(100)]}
    artifact = pipeline.summarize(repo, repo)
    assert artifact.source == "generated"
    assert artifact._reuse_key is not None
    assert pipeline.is_current(artifact, repo=repo)
    changed = {**repo, "topics": [*repo["topics"][:-1], "last-topic-changed"]}
    assert not pipeline.is_current(artifact, repo=changed)
    pipeline.summarize(changed, changed)
    assert provider.calls == 2
