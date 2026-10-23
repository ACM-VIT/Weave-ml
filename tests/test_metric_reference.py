"""Minimal diagnostic expectations reuse existing gates without promotion authority."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from evaluation.benchmark_gates import Baseline, MetricReference, Policy, assess, load_baseline, qualification_manifest
from evaluation.recommendation_benchmark import digest

FIXTURE=Path(__file__).parent/'fixtures/recommendation_actual_stage'

def fixture():
    baseline=Baseline.model_validate(json.loads((FIXTURE/'baseline.json').read_text()))
    policy=Policy.model_validate(json.loads((FIXTURE/'policy.json').read_text()))
    report=baseline.report
    reference=dict(format='curated-metric-reference-v1',schema_version=1,status='diagnostic',
        reference='Controlled adapter test over prior measured synthetic fixture; not new benchmark evidence',
        policy_sha256=digest(policy),snapshot_id=report.snapshot_id,k=report.k,
        bundle_sha256=report.bundle_sha256,evaluation_dataset_sha256=report.evaluation_dataset_sha256,
        embedding_sha256=report.embedding_sha256,provenance=report.provenance.model_dump(mode='json'),
        artifact=dict(model_sha256=report.artifact.model_sha256,scaler_sha256=report.artifact.scaler_sha256,
            feature_schema=report.artifact.feature_schema,timing_kind='measured',
            accepted_artifact_commit='c691c73998f401a74d15a044fb678bef621ff343',measurement_runner_commit=report.artifact.code_commit,measurement_source_manifest_sha256='a'*64,measurement_source_dirty=False),
        reference_run_sha256=report.run_sha256,stages={stage:{'metrics':row.metrics} for stage,row in report.stages.items()})
    return baseline,policy,reference


def test_expected_metrics_use_identical_existing_checks_and_deltas():
    baseline,policy,raw=fixture();reference=load_baseline(raw)
    assert isinstance(reference,MetricReference)
    before=assess(baseline.report.model_dump(mode='json'),baseline,policy)
    after=assess(baseline.report.model_dump(mode='json'),reference,policy)
    assert after['passed'] is True and before['checks']==after['checks']
    changed=baseline.report.model_dump(mode='json');changed['stages']['assembly']['metrics']['judged_ndcg_at_k']=0.
    assert not assess(changed,reference,policy)['passed']
    assert 'assembly.judged_ndcg_at_k: baseline regression' in assess(changed,reference,policy)['failures']


def test_minimal_references_cannot_qualify_even_if_diagnostic_checks_pass():
    baseline,policy,raw=fixture();reference=load_baseline(raw)
    result=assess(baseline.report.model_dump(mode='json'),reference,policy,qualify=True)
    assert result['passed'] is False
    assert 'summary-only diagnostic metric references cannot qualify a production model' in result['failures']
    with pytest.raises(ValueError,match='passing qualification'):qualification_manifest(baseline.report.model_dump(mode='json'),reference,policy)


@pytest.mark.parametrize('fault',['promoted','missing_metric','extra_report','invalid_ratio','missing_stage','wrong_format'])
def test_expected_reference_is_strict_and_does_not_accept_full_generated_reports(fault):
    _,_,raw=fixture();raw=deepcopy(raw)
    if fault=='promoted':raw['status']='promoted'
    elif fault=='missing_metric':del raw['stages']['assembly']['metrics']['judged_ndcg_at_k']
    elif fault=='extra_report':raw['outputs']=[{'private':'run trace'}]
    elif fault=='invalid_ratio':raw['stages']['assembly']['metrics']['judged_ndcg_at_k']=2.
    elif fault=='missing_stage':del raw['stages']['assembly']
    else:raw['format']='unrecognized-reference'
    with pytest.raises(ValueError):load_baseline(raw)
