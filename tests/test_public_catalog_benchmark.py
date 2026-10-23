"""Public-source-backed diagnostic tasks, not human labels or promotion evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pytest
from evaluation.benchmark_gates import Policy,MetricReference,assess,load_baseline
from evaluation.recommendation_benchmark import Run,digest,evaluate
from evaluation.stage_runner import FrozenSnapshot,execute_snapshot
from tests.test_tower_history import require_local_only_git

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'tests/fixtures/recommendation_public_catalog'
def read(name):return json.loads((FIXTURE/name).read_text())


def test_frozen_public_sources_and_complete_curated_intents_have_honest_provenance():
    snapshot=FrozenSnapshot.model_validate(read('snapshot.json'));notes=read('provenance.json')
    bundle=snapshot.bundle
    assert len(bundle.repositories)==32 and len(bundle.profiles)==10 and len(bundle.judgments)==320
    assert {(j.profile_id,j.repo_id) for j in bundle.judgments}=={(p.profile_id,r.repo_id) for p in bundle.profiles for r in bundle.repositories}
    assert {j.grade for j in bundle.judgments}=={0,1,2,3}
    assert bundle.provenance.kind=='synthetic' and notes['label_evidence']['human_labels'] is False
    assert all(r.fork_family is None for r in bundle.repositories)
    assert all(source.readme=='' for source in bundle.dataset.repositories)
    assert 'not empty-input encoding' in bundle.provenance.reference
    assert len(notes['source_notes'])==32 and all(len(row['readme_input_sha256'])==64 for row in notes['source_notes'])
    assert sum(row['readme_input_truncated'] for row in notes['source_notes'])==1
    assert sum(row['topics'] is None for row in notes['source_notes'])==5
    assert notes['unknowns']['fork_lineage'].startswith('unknown')
    assert all(p.seen and p.disliked for p in bundle.profiles)
    assert all(len(r.vector)==384 for r in bundle.repositories)
    reference=load_baseline(read('reference.json'))
    assert isinstance(reference,MetricReference) and reference.status=='diagnostic'
    assert reference.artifact.model_sha256==hashlib.sha256((ROOT/'inference/heavy_ranker.pt').read_bytes()).hexdigest()
    assert reference.artifact.scaler_sha256==hashlib.sha256((ROOT/'inference/feature_scaler.json').read_bytes()).hexdigest()
    assert reference.evaluation_dataset_sha256==bundle.dataset.sha256


@pytest.fixture(scope='module')
def measured():
    require_local_only_git()
    snapshot=FrozenSnapshot.model_validate(read('snapshot.json'))
    exported=execute_snapshot(snapshot,ROOT/'inference')
    policy=Policy.model_validate(read('policy.json'));reference=load_baseline(read('reference.json'))
    return snapshot,exported,policy,reference


def test_actual_current_stages_pass_fixed_accepted_artifact_scalar_reference(measured):
    snapshot,exported,policy,reference=measured
    report=evaluate(snapshot.bundle,exported.run,k=policy.k)
    result=assess(report,reference,policy)
    assert result['passed'],result['failures']
    for stage in report['stages'].values():
        assert stage['metrics']['judgment_coverage']==1
        assert stage['metrics']['excluded_rate']==stage['metrics']['repeat_rate']==stage['metrics']['archived_rate']==0
        assert stage['metrics']['fork_metadata_coverage']==0 # Unknown is reported, not invented complete lineage.
    assert not assess(report,reference,policy,qualify=True)['passed']


def test_real_ranked_rows_with_deliberately_worse_assembly_fail_quality_gates(measured):
    snapshot,exported,policy,reference=measured
    grades={(j.profile_id,j.repo_id):j.grade for j in snapshot.bundle.judgments}
    raw=deepcopy(exported.run.model_dump(mode='json'))
    for output in raw['outputs']:
        output['stages']['assembly']=sorted(output['stages']['ranking'],key=lambda row:(grades[output['profile_id'],row['repo_id']],row['repo_id']))[:policy.k]
    report=evaluate(snapshot.bundle,Run.model_validate(raw),k=policy.k)
    result=assess(report,reference,policy)
    assert result['passed'] is False
    assert 'assembly.judged_ndcg_at_k: baseline regression' in result['failures']


def test_existing_cli_accepts_curated_reference_and_refuses_qualification(measured,tmp_path):
    from scripts import evaluate_recommendations as cli
    snapshot,exported,_,_=measured
    bundle=tmp_path/'bundle.json';bundle.write_text(snapshot.bundle.model_dump_json())
    run=tmp_path/'run.json';run.write_text(exported.model_dump_json())
    output=tmp_path/'diagnostic.json'
    args=['--bundle',str(bundle),'--run',str(run),'--baseline',str(FIXTURE/'reference.json'),
          '--policy',str(FIXTURE/'policy.json'),'--output',str(output)]
    assert cli.main(args)==0
    assert cli.main([*args,'--qualify','--model-artifact',str(ROOT/'inference/heavy_ranker.pt'),
        '--scaler-artifact',str(ROOT/'inference/feature_scaler.json')])==1
    result=json.loads(output.read_text())
    assert result['qualification_manifest'] is None
    assert 'summary-only diagnostic metric references cannot qualify a production model' in result['assessment']['failures']
