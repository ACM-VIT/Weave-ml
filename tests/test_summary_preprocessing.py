"""Issue #86: independently bounded analysis and metadata-only fast paths."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from summarization.pipeline import CardSummaryPipeline, description_fallback_artifact
from summarization.settings import SummarySettings
from summarization.source import MAX_SOURCE_ANALYSIS_LINES, build_summary_source
from utils.readme_processor import MAX_PREPROCESS_CHARS


@pytest.mark.parametrize('policy', ['reject', 'prefix'])
def test_provider_disabled_never_hashes_or_parses_readme_and_keeps_identical_fallback(policy):
    class UnreadableSource:
        @property
        def readme(self):
            raise AssertionError('metadata fallback must not inspect source')

    repo = {'full_name': 'weave/fast-path', 'description': 'Shared tools for reviewing changes.',
            'primary_language': 'Python', 'readme': '[' * 4_000_000,
            'content_hash': 'a' * 64}
    expected = description_fallback_artifact(repo)
    pipeline = CardSummaryPipeline(settings=SummarySettings(source_overflow_policy=policy))
    with patch('summarization.pipeline.source_identity', side_effect=AssertionError('hashing README')), \
         patch('summarization.pipeline.build_summary_source', side_effect=AssertionError('parsing README')):
        actual = pipeline.summarize(UnreadableSource(), repo)
    assert actual.model_dump() == expected.model_dump()
    assert pipeline.reuse_cache.snapshot()['entries'] == 0


@pytest.mark.parametrize('raw, expected_limit', [
    ('[' * (MAX_PREPROCESS_CHARS + 1), 'source character budget exceeded'),
    ('x\n' * (MAX_SOURCE_ANALYSIS_LINES + 1), 'source line budget exceeded'),
], ids=['character-overflow', 'line-overflow'])
def test_strict_analysis_rejects_incomplete_evidence_before_line_cleanup(raw, expected_limit):
    repo = {'full_name': 'weave/limits', 'readme': raw}
    with patch('summarization.source._clean_line', side_effect=AssertionError('cleaning incomplete evidence')):
        material = build_summary_source(repo, repo, max_chars=1_000)
    assert material.analysis_truncated
    assert material.analysis_limit == expected_limit
    assert material.prompt_input == '' and material.comparison_texts == ()
    assert repo['readme'] is raw


def test_small_prompt_retains_tail_and_derived_evidence_for_copy_detection():
    opening = 'A collaborative workspace helps maintainers coordinate proposed software changes. '
    tail = 'TAIL evidence remains available to detect copied output even after prompt clipping.'
    raw = '# Overview\n' + opening * 600 + '\n' + tail
    derived = 'A separate derived paragraph also remains part of complete validation evidence.'
    repo = {'full_name': 'weave/evidence', 'readme': raw, 'extracted_paragraphs': [derived]}
    source = SimpleNamespace(readme=SimpleNamespace(raw_markdown=raw))
    material = build_summary_source(source, repo, max_chars=1_000)
    assert material.analysis_truncated and material.analysis_limit is None
    assert len(material.prompt_input) <= 1_000
    assert tail not in material.prompt_input
    assert raw in material.comparison_texts and derived in material.comparison_texts
    assert repo['readme'] is raw and source.readme.raw_markdown is raw


@pytest.mark.parametrize('newline', ['\n', '\r\n', '\u2028'])
def test_line_budget_accepts_boundary_and_rejects_next_line(newline):
    raw = newline.join(['x'] * MAX_SOURCE_ANALYSIS_LINES)
    repo = {'full_name': 'weave/lines', 'readme': raw}
    accepted = build_summary_source(repo, repo, max_chars=1_000)
    assert accepted.analysis_limit is None
    rejected = build_summary_source(repo, {**repo, 'readme': raw + newline + 'extra'}, max_chars=1_000)
    assert rejected.analysis_limit == 'source line budget exceeded'
