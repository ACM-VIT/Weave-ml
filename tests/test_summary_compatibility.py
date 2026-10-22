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


@pytest.mark.parametrize('description', [
    '它支持多种格式e.g. JSON and YAML. Developers can add custom parsers.',
    '它支持多种格式i.e. JSON and YAML. Developers can add custom parsers.',
    'It provides a review workspace. Teams call it "easy to follow."',
    'It provides a review workspace. Teams call it “easy to follow.”',
    'It compares policies, e.g. GPL. Teams can review package choices.',
])
def test_fallback_keeps_both_semantic_sentences_and_their_punctuation(description):
    assert len(split_summary_sentences(description)) == 2
    repo = {'full_name': 'weave/compatibility', 'description': description}
    assert CardSummaryPipeline().summarize(repo, repo).summary == description


@pytest.mark.parametrize('metadata', [
    {'full_name': 'acme/license-checker', 'primary_language': 'Python'},
    {'full_name': 'acme/tool', 'primary_language': 'Unknown', 'topics': ['reference']},
    {'full_name': 'acme/tool', 'primary_language': 'reference'},
    {'full_name': 'acme/license-checker', 'description': 'A useful repository for teams.'},
])
def test_metadata_fallback_is_accepted_by_its_own_replay_validator(metadata):
    pipeline = CardSummaryPipeline()
    repo = {'full_name': 'acme/tool', 'description': '', **metadata}
    artifact = pipeline.summarize(repo, repo)
    assert artifact.source == 'description_fallback'
    assert not _unsafe_text(artifact.summary)
    assert pipeline.is_current(artifact, repo=repo)
