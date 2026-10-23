from __future__ import annotations

import copy

import pytest

from evaluation.benchmark_gates import Baseline, Policy, assess, qualification_manifest
from evaluation.recommendation_benchmark import digest
from tests.test_recommendation_benchmark import report


def policy_and_baseline(result=None):
    result = result or report()
    policy = Policy.model_validate(dict(schema_version=1, policy_id='synthetic-smoke-v1',
        reference='Synthetic correctness limits, not production quality thresholds', k=3, min_profiles=2,
        rules=[dict(stage='assembly', metric=metric, minimum=minimum, maximum=maximum, max_regression=delta)
               for metric, minimum, maximum, delta in [
                   ('judged_ndcg_at_k', .8, None, .02), ('judged_recall_at_k', .8, None, .02),
                   ('catalogue_coverage', .8, None, .01), ('intra_list_similarity', None, .9, .1),
                   ('repeat_rate', None, 0., 0.), ('excluded_rate', None, 0., 0.),
                   ('archived_rate', None, 0., 0.), ('fork_repeat_rate', None, 0., 0.),
                   ('latency_p95_ms', None, 20., .5), ('judgment_coverage', 1., None, 0.)]]))
    baseline = Baseline.model_validate(dict(schema_version=1, status='diagnostic', reference='synthetic only',
        policy_sha256=digest(policy), report_sha256=digest(result), report=result))
    return policy, baseline


def test_matching_diagnostic_passes_and_cannot_qualify():
    result = report()
    policy, baseline = policy_and_baseline(result)
    assert assess(result, baseline, policy)['passed']
    qualified = assess(result, baseline, policy, qualify=True)
    assert not qualified['passed']
    assert any('synthetic' in reason for reason in qualified['failures'])
    assert any('promoted baseline' in reason for reason in qualified['failures'])
    assert any('measured' in reason for reason in qualified['failures'])
    with pytest.raises(ValueError, match='passing qualification'):
        qualification_manifest(result, baseline, policy)


@pytest.mark.parametrize('metric,value', [('judged_ndcg_at_k', .79), ('latency_p95_ms', 4.),
    ('repeat_rate', .1), ('excluded_rate', .1), ('fork_repeat_rate', .1), ('catalogue_coverage', .6)])
def test_injected_quality_latency_safety_and_diversity_regressions_fail(metric, value):
    result = report()
    policy, baseline = policy_and_baseline(result)
    changed = copy.deepcopy(result)
    changed['stages']['assembly']['metrics'][metric] = value
    assessment = assess(changed, baseline, policy)
    assert not assessment['passed']
    assert any(metric in failure for failure in assessment['failures'])


@pytest.mark.parametrize('field,value', [('bundle_sha256', 'e'*64), ('embedding_sha256', 'f'*64),
                                        ('snapshot_id', 'other'), ('k', 4)])
def test_comparison_rejects_mismatched_evidence(field, value):
    result = report()
    policy, baseline = policy_and_baseline(result)
    result[field] = value
    assert not assess(result, baseline, policy)['passed']


def test_baseline_digest_and_policy_are_checked():
    result = report()
    policy, baseline = policy_and_baseline(result)
    raw = baseline.model_dump()
    raw['report']['stages']['assembly']['metrics']['judged_ndcg_at_k'] = .1
    with pytest.raises(ValueError, match='digest'): Baseline.model_validate(raw)
    different_policy = policy.model_copy(update={'policy_id': 'different'})
    assert not assess(result, baseline, different_policy)['passed']


def test_missing_metrics_stage_sample_counts_and_nonfinite_are_rejected():
    for mutation in ['metric', 'stage', 'count', 'nan', 'bool']:
        result = report()
        policy, baseline = policy_and_baseline(result)
        stage = result['stages']['assembly']
        if mutation == 'metric': del stage['metrics']['novelty_bits']
        if mutation == 'stage': del result['stages']['ranking']
        if mutation == 'count': stage['counts']['latency_samples'] = 1
        if mutation == 'nan': stage['metrics']['novelty_bits'] = float('nan')
        if mutation == 'bool': stage['metrics']['repeat_rate'] = True
        with pytest.raises(ValueError): assess(result, baseline, policy)


def test_unknown_metadata_does_not_look_like_proven_diversity():
    from tests.test_recommendation_benchmark import diagnostic
    b, r = diagnostic()
    b['repositories'][0]['owner'] = None
    b['repositories'][0]['topics'] = []
    b['repositories'][0]['fork_family'] = None
    result = report(b, r)
    metrics = result['stages']['assembly']['metrics']
    assert metrics['owner_metadata_coverage'] < 1
    assert metrics['topic_metadata_coverage'] < 1
    assert metrics['fork_metadata_coverage'] < 1


def test_holdout_temporal_boundaries_are_validated():
    from evaluation.recommendation_benchmark import Bundle
    from tests.test_recommendation_benchmark import diagnostic
    b, _ = diagnostic()
    b['provenance'] = dict(kind='event_holdout', reference='example external audit')
    with pytest.raises(ValueError, match='cutoff'): Bundle.model_validate(b)
    b['provenance'].update(observation_cutoff='2026-10-23T00:00:00Z',
        label_horizon_end='2026-10-24T00:00:00Z', ingestion_watermark='2026-10-23T12:00:00Z')
    with pytest.raises(ValueError, match='horizon'): Bundle.model_validate(b)


def qualification_fixture():
    """Invented declarations exercise gate mechanics only, never real qualification."""
    from evaluation.benchmark_gates import DIRECTIONS, STAGES
    result = report()
    result['provenance'].update(kind='human_judgment', reference='test-only invented declaration')
    result['artifact']['timing_kind'] = 'measured'
    policy = Policy.model_validate(dict(schema_version=1, policy_id='test-only-complete',
        reference='Invented gate mechanics fixture, not production thresholds', k=3, min_profiles=2,
        rules=[dict(stage=stage, metric=metric, max_regression=0.,
                    **({'minimum': 0.} if direction == 'higher' else {'maximum': 1000.}))
               for stage in STAGES for metric, direction in DIRECTIONS.items()]))
    baseline = Baseline.model_validate(dict(schema_version=1, status='promoted', reference='test-only',
        policy_sha256=digest(policy), report_sha256=digest(result), report=result))
    return result, policy, baseline


def test_test_only_complete_qualification_produces_bound_manifest():
    result, policy, baseline = qualification_fixture()
    assessment = assess(result, baseline, policy, qualify=True)
    assert assessment['passed'], assessment['failures']
    manifest = qualification_manifest(result, baseline, policy)
    assert manifest['report_sha256'] == digest(result)
    assert manifest['assessment_sha256'] == digest(assessment)
    assert manifest['data_snapshot_sha256'] == result['bundle_sha256']
    assert manifest['model_artifact'] == result['artifact']


@pytest.mark.parametrize('metric', ['judged_recall_at_k', 'latency_p95_ms'])
def test_qualification_rejects_required_gate_with_wrong_direction_only(metric):
    result, policy, baseline = qualification_fixture()
    raw = policy.model_dump()
    rule = next(r for r in raw['rules'] if r['stage'] == 'assembly' and r['metric'] == metric)
    if metric == 'latency_p95_ms':
        rule['maximum'], rule['minimum'] = None, 0.
        expected = 'maximum'
    else:
        rule['minimum'], rule['maximum'] = None, 1.
        expected = 'minimum'
    changed = Policy.model_validate(raw)
    baseline = baseline.model_copy(update={'policy_sha256': digest(changed)})
    assessment = assess(result, baseline, changed, qualify=True)
    assert assessment['failures'] == [f'assembly.{metric}: qualification requires {expected}']


def test_qualification_failure_bytes_are_independent_of_python_hash_seed():
    import os, subprocess, sys
    script="""
import json
from tests.test_benchmark_gates import report, policy_and_baseline
from evaluation.benchmark_gates import assess
result=report()
policy,baseline=policy_and_baseline(result)
for stage in result['stages'].values():
    for metric in ('repeat_rate','excluded_rate','archived_rate'):
        stage['metrics'][metric]=.5
print(json.dumps(assess(result,baseline,policy,qualify=True),sort_keys=True))
"""
    outputs=[subprocess.run([sys.executable,'-c',script],env={**os.environ,'PYTHONHASHSEED':str(seed)},
        capture_output=True,check=True,timeout=10).stdout for seed in (1,2,3,4)]
    assert len(set(outputs))==1
