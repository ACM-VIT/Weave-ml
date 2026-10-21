"""Identical producer/consumer fixtures live in Weave's backend/tests/fixtures."""

import json
from pathlib import Path

import pytest

from summarization.contracts import CardSummaryContent
from summarization.pipeline import CardSummaryPipeline
from summarization.sentences import has_two_or_three_complete_sentences, split_summary_sentences
from summarization.validation import _unsafe_text, validate_generated_summary

CASES = json.loads((Path(__file__).parent / 'fixtures/summary_compatibility.json').read_text())


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['name'])
def test_shared_consumer_contract(case):
    def validate():
        content = CardSummaryContent(summary=case['summary'], highlights=case.get('highlights', []))
        assert has_two_or_three_complete_sentences(content.summary)
        assert not _unsafe_text(content.summary)
        for highlight in content.highlights:
            assert not _unsafe_text(highlight)
        return content

    if case['valid']:
        content = validate()
        # Short metadata fallbacks have no generated-content minimum.
        if len(content.summary) >= 180:
            assert validate_generated_summary(content, comparison_texts=[]).summary == content.summary
    else:
        with pytest.raises((ValueError, AssertionError)):
            validate()


@pytest.mark.parametrize('case', [case for case in CASES if case['valid']], ids=lambda case: case['name'])
def test_real_fallback_pipeline_preserves_semantic_sentences(case):
    repo = {'full_name': 'weave/compatibility', 'description': case['summary']}
    artifact = CardSummaryPipeline().summarize(repo, repo)
    assert has_two_or_three_complete_sentences(artifact.summary)
    assert not _unsafe_text(artifact.summary)
    assert len(artifact.summary) <= 360
    if len(case['summary']) < 175:
        assert artifact.summary == ' '.join(split_summary_sentences(case['summary'])[:2])


def test_generated_abbreviations_and_subject_matter_do_not_trigger_repair():
    summary = (
        'Tools detect GPL boundaries across projects, e.g. across shared parsers and developer utilities. '
        'Teams can inspect compatibility decisions, e.g. when evaluating new dependencies for their internal projects.'
    )
    class Provider:
        calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            return json.dumps({'summary': summary})

    provider = Provider()
    repo = {'full_name': 'weave/compatibility', 'description': 'A policy comparison tool.'}
    artifact = CardSummaryPipeline(provider=provider).summarize(repo, repo)
    assert artifact.source == 'generated'
    assert artifact.summary == summary
    assert provider.calls == 1
