"""Generated summaries obey the backend's normalized partial-copy boundary."""
import hashlib
import json

import pytest

from summarization.contracts import CardSummaryArtifact
from summarization.pipeline import CardSummaryPipeline, description_fallback_artifact
from summarization.settings import SummarySettings
from summarization.validation import SummaryValidationError, parse_summary_response, validate_generated_summary

KUBERNETES_SUMMARY = (
    'Kubernetes is an open-source system for managing containerized applications across multiple hosts, '
    'aimed at teams deploying and operating distributed workloads. It provides core mechanisms for '
    'application deployment, maintenance, and scaling, drawing on large-scale production experience '
    'and community practices.'
)
# Public README wording uses spaces where the provider used a hyphen. The
# existing whole-output similarity/12-word checks miss this partial passage.
KUBERNETES_SOURCE = (
    'Kubernetes is an open source system for managing containerized applications across multiple hosts.'
)
SAFE_SUMMARY = (
    'A focused workspace gives engineering teams a clear view of proposed software changes. '
    'Its focused review workflow brings ownership decisions and review history together so maintainers can coordinate changes across shared projects.'
)


def test_kubernetes_partial_passage_is_rejected_without_requiring_whole_summary_copy():
    content = parse_summary_response(json.dumps({'summary': KUBERNETES_SUMMARY}))
    with pytest.raises(SummaryValidationError, match='summary copies'):
        validate_generated_summary(content, comparison_texts=(KUBERNETES_SOURCE,))


@pytest.mark.parametrize('size,rejected', [(79, False), (80, True)])
def test_unicode_partial_copy_counts_normalized_characters(size, rejected):
    passage = '界' * size
    content = parse_summary_response(json.dumps({'summary': passage + '. ' + SAFE_SUMMARY.split('. ')[1]}))
    if rejected:
        with pytest.raises(SummaryValidationError, match='summary copies'):
            validate_generated_summary(content, comparison_texts=(passage,))
    else:
        assert validate_generated_summary(content, comparison_texts=(passage,)) == content


def test_highlights_enforce_the_same_partial_copy_gate():
    content = parse_summary_response(json.dumps({'summary': SAFE_SUMMARY, 'highlights': ['界' * 80]}))
    with pytest.raises(SummaryValidationError, match='highlight copies'):
        validate_generated_summary(content, comparison_texts=('界' * 80,))


def test_copy_normalization_uses_lowercase_without_casefold_expansion():
    content = parse_summary_response(json.dumps({'summary': 'S' * 80 + '. ' + SAFE_SUMMARY.split('. ')[1]}))
    assert validate_generated_summary(content, comparison_texts=('ß' * 80,)) == content


class CopiedProvider:
    def __init__(self):
        self.calls = 0

    def generate(self, source, *, repair_feedback=None, deadline=None):
        self.calls += 1
        return json.dumps({'summary': '界' * 80 + '. ' + SAFE_SUMMARY.split('. ')[1]})


def prefix_repo():
    raw = '# Overview\nA workspace helps software maintainers coordinate decisions.\n' + 'x\n' * 33_000 + '界' * 80
    return {'repo_id': '00000000-0000-4000-8000-000000000221', 'full_name': 'weave/bounded-copy',
            'description': 'A collaborative workspace.', 'primary_language': 'Python',
            'readme': raw, 'content_hash': hashlib.sha256(raw.encode()).hexdigest()}


def test_repair_validates_full_canonical_readme_beyond_the_approved_prompt_prefix():
    repo = prefix_repo()
    original = repo['readme']
    provider = CopiedProvider()
    pipeline = CardSummaryPipeline(settings=SummarySettings(source_overflow_policy='prefix'), provider=provider)
    artifact = pipeline.summarize(repo, repo)
    assert artifact.source == 'description_fallback'
    assert provider.calls == 2
    assert repo['readme'] == original


def test_same_content_replay_revalidates_stored_generated_artifact_against_full_readme():
    repo = prefix_repo()
    pipeline = CardSummaryPipeline(settings=SummarySettings(source_overflow_policy='prefix'), provider=CopiedProvider())
    content = parse_summary_response(json.dumps({'summary': '界' * 80 + '. ' + SAFE_SUMMARY.split('. ')[1]}))
    artifact = CardSummaryArtifact(**content.model_dump(), prompt_version=pipeline.settings.prompt_version,
        format_version=pipeline.settings.format_version, model_version=pipeline.settings.model_id, source='generated')
    artifact._reuse_key = pipeline._reuse_key(repo, repo).hex()
    assert not pipeline.is_current(artifact, repo=repo)


def test_long_description_fallback_remains_current_when_it_also_appears_in_readme():
    description = 'A collaborative workspace supports maintainers coordinating software ownership and review decisions.'
    repo = {'description': description, 'primary_language': 'Python', 'readme': description}
    pipeline = CardSummaryPipeline()
    artifact = description_fallback_artifact(repo)
    assert artifact.source == 'description_fallback'
    assert pipeline.is_current(artifact, repo=repo)


def test_full_canonical_copy_evidence_preserves_the_existing_http_source_bound():
    content = parse_summary_response(json.dumps({'summary': SAFE_SUMMARY}))
    with pytest.raises(SummaryValidationError, match='canonical README exceeds'):
        validate_generated_summary(content, canonical_readme='x' * 1_000_001)


@pytest.mark.parametrize('size,rejected', [(39, False), (40, True)])
def test_partial_copy_matches_backend_utf16_window_for_astral_letters(size, rejected):
    passage = '\U00020000' * size
    summary = 'This overview describes ' + passage + '. ' + SAFE_SUMMARY.split('. ')[1]
    content = parse_summary_response(json.dumps({'summary': summary}))
    if rejected:
        with pytest.raises(SummaryValidationError, match='summary copies'):
            validate_generated_summary(content, comparison_texts=(passage,))
    else:
        assert validate_generated_summary(content, comparison_texts=(passage,)) == content


def test_partial_copy_window_can_start_and_end_inside_surrogate_pairs():
    # JS slice(1,81) matches the low surrogate of the first letter, 39 complete
    # letters, and the high surrogate of the last. Only 39 codepoints match.
    shared = '\U00020800' * 39
    copied = '\U00020000' + shared + '\U00020001'
    source = '\U00020400' + shared + '\U00020002'
    summary = 'This overview describes ' + copied + '. ' + SAFE_SUMMARY.split('. ')[1]
    content = parse_summary_response(json.dumps({'summary': summary}))
    with pytest.raises(SummaryValidationError, match='summary copies'):
        validate_generated_summary(content, comparison_texts=(source,))


def test_adversarial_utf16_byte_alignment_finishes_in_a_bounded_process():
    import subprocess
    import sys

    # Reversed byte pairs previously produced many odd-offset byte matches,
    # making a negative copy check repeatedly scan the entire source.
    script = """
from summarization.validation import _contains_partial_copy, _normalized_prose
summary = _normalized_prose('\\u4142' * 360)
for size in (16_384, 1_000_000):
    source = _normalized_prose('\\u4241' * size)
    assert not _contains_partial_copy(summary, source)
"""
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True, timeout=8)


@pytest.mark.parametrize('source_mode', ['comparison', 'canonical'])
def test_new_unicode_source_cannot_bypass_known_lowercase_copy_check(source_mode):
    # Unicode 16 adds A7CB and lowercases it to the older assigned U+0264.
    # Python 3.12 cannot classify the new uppercase source, so fail closed.
    import unicodedata
    if unicodedata.category('\ua7cb') != 'Cn':
        pytest.skip('Runtime Unicode table already classifies the new source letter.')
    source = '\ua7cb' * 80
    content = parse_summary_response(json.dumps({'summary': '\u0264' * 80 + '. ' + SAFE_SUMMARY.split('. ')[1]}))
    arguments = {'comparison_texts': (source,)} if source_mode == 'comparison' else {'canonical_readme': source}
    with pytest.raises(SummaryValidationError, match='Unicode'):
        validate_generated_summary(content, **arguments)


def test_new_unicode_output_uses_repair_or_fallback_instead_of_dropping_letters():
    import unicodedata
    if unicodedata.category('\u1c89') != 'Cn':
        pytest.skip('Runtime Unicode table already classifies the new output letter.')
    content = parse_summary_response(json.dumps({'summary': '\u1c89' * 80 + '. ' + SAFE_SUMMARY.split('. ')[1]}))
    with pytest.raises(SummaryValidationError, match='Unicode'):
        validate_generated_summary(content, comparison_texts=('An unrelated known source.',))
