import hashlib
import json
from copy import deepcopy

import pytest
from summarization.pipeline import CardSummaryPipeline
from summarization.settings import SummarySettings
from tests.test_summary_reuse import Provider, REPO


def large_repo(tail='TAIL_NOT_FOR_PROVIDER'):
    raw = '# Overview\n' + 'A collaborative workspace supports software maintainers.\n' * 1400 + tail
    return {**REPO, 'readme': raw, 'clean_text': raw,
            'extracted_paragraphs': ['stale derived content'] * 100,
            'content_hash': hashlib.sha256(raw.encode()).hexdigest()}


def test_explicit_prefix_mode_generates_replays_and_keeps_canonical_readme_unchanged():
    repo = large_repo()
    original = deepcopy(repo)
    provider = Provider()
    pipeline = CardSummaryPipeline(settings=SummarySettings(source_overflow_policy='prefix'), provider=provider)
    artifact = pipeline.summarize(repo, repo)
    assert artifact.source == 'generated'
    assert pipeline.is_current(artifact, repo=repo)
    assert pipeline.summarize(repo, repo).model_dump() == artifact.model_dump()
    assert provider.calls == 1
    assert repo == original
    material = pipeline._analysis(repo, repo, pipeline._reuse_key(repo, repo)).material
    assert material.analysis_limit is None
    assert len(material.prompt_input) <= 12000
    assert all(len(value) <= 64000 for value in material.comparison_texts)
    assert 'TAIL_NOT_FOR_PROVIDER' not in material.prompt_input
    changed = large_repo('A DIFFERENT UNSEEN TAIL')
    assert not pipeline.is_current(artifact, repo=changed)
    pipeline.summarize(changed, changed)
    assert provider.calls == 2


def test_prefix_mode_requires_full_canonical_content_hash():
    provider = Provider()
    pipeline = CardSummaryPipeline(settings=SummarySettings(source_overflow_policy='prefix'), provider=provider)
    for content_hash in [None, '', 'not-a-canonical-hash']:
        assert pipeline.summarize(large_repo() | {'content_hash': content_hash}, large_repo() | {'content_hash': content_hash}).source == 'description_fallback'
    assert provider.calls == 0


def test_prefix_policy_is_part_of_reuse_identity():
    repo = REPO | {'content_hash': 'a' * 64}
    strict = CardSummaryPipeline(settings=SummarySettings())
    prefix = CardSummaryPipeline(settings=SummarySettings(source_overflow_policy='prefix'))
    assert strict._reuse_key(repo, repo) != prefix._reuse_key(repo, repo)


def test_prefix_mode_preserves_copy_detection():
    copied = ('A focused workspace gives engineering teams a clear view of proposed software changes. '
              'It brings ownership decisions and review history together so maintainers can coordinate their work.')
    repo = large_repo()
    repo['readme'] = copied + '\n' + repo['readme']
    repo['content_hash'] = hashlib.sha256(repo['readme'].encode()).hexdigest()
    provider = Provider()
    pipeline = CardSummaryPipeline(settings=SummarySettings(source_overflow_policy='prefix'), provider=provider)
    assert pipeline.summarize(repo, repo).source == 'description_fallback'
    assert provider.calls == 2


def test_prefix_setting_is_explicit_and_validated(monkeypatch):
    monkeypatch.setenv('SUMMARY_SOURCE_OVERFLOW_POLICY', 'prefix')
    assert SummarySettings.from_env().source_overflow_policy == 'prefix'
    with pytest.raises(ValueError, match='SUMMARY_SOURCE_OVERFLOW_POLICY'):
        SummarySettings(source_overflow_policy='ignore')


@pytest.mark.parametrize('policy,valid', [('prefix', True), ('reject', True), ('ignore', False)])
def test_production_preflight_checks_prefix_policy(policy, valid):
    from scripts.validate_production_config import validate_production_config
    from tests.test_production_config import valid_production_env
    env = valid_production_env() | {'SUMMARY_SOURCE_OVERFLOW_POLICY': policy}
    issues = validate_production_config(env)
    assert any(issue.name == 'SUMMARY_SOURCE_OVERFLOW_POLICY' for issue in issues) is not valid
