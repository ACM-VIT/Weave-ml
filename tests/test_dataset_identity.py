"""Stable raw dataset binding and actual-stage regression baseline checks."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from evaluation.recommendation_benchmark import Bundle,Run,digest,evaluate
from evaluation.benchmark_gates import Baseline,Policy,assess
from evaluation.stage_runner import FrozenSnapshot,execute_snapshot
from tests.test_tower_history import require_local_only_git

ROOT=Path(__file__).resolve().parents[1]
FIXTURE=ROOT/'tests/fixtures/recommendation_actual_stage'
def read(name):
    if name=='bundle.json':return json.loads((FIXTURE/'snapshot.json').read_text())['bundle']
    return json.loads((FIXTURE/name).read_text())

def fixture_report():
    baseline=Baseline.model_validate(read('baseline.json'))
    return baseline,Policy.model_validate(read('policy.json'))


def test_legacy_bundle_policy_serialization_and_hashes_unchanged():
    root=ROOT/'tests/fixtures/recommendation_benchmark'
    baseline=Baseline.model_validate(json.loads((root/'baseline.json').read_text()))
    bundle=Bundle.model_validate(json.loads((root/'bundle.json').read_text()))
    policy=Policy.model_validate(json.loads((root/'policy.json').read_text()))
    assert digest(bundle)==baseline.report.bundle_sha256
    assert digest(policy)==baseline.policy_sha256
    assert 'dataset' not in bundle.model_dump(mode='json')
    assert 'embedding_comparison' not in policy.model_dump(mode='json')


def test_explicit_cross_encoder_policy_keeps_per_run_identity_and_metrics():
    old,policy=fixture_report();raw=read('bundle.json');bundle=Bundle.model_validate(raw)
    policy=Policy.model_validate({**policy.model_dump(mode='json'),'embedding_comparison':'cross_embedding'})
    baseline=Baseline.model_validate({**old.model_dump(mode='json'),'policy_sha256':digest(policy)})
    changed=deepcopy(old.report.model_dump(mode='json'))
    changed['bundle_sha256']='d'*64
    changed['embedding_identity']['revision']='e'*40
    changed['embedding_sha256']=digest(changed['embedding_identity'])
    assessment=assess(changed,baseline,policy)
    assert assessment['passed']
    changed['stages']['assembly']['metrics']['judged_ndcg_at_k']=0.
    assert 'assembly.judged_ndcg_at_k: below minimum' in assess(changed,baseline,policy)['failures']
    with pytest.raises(ValueError,match='identity digest'):
        assess({**changed,'embedding_sha256':'f'*64},baseline,policy)
    assert digest(bundle.dataset)==old.report.evaluation_dataset_sha256 # immutable historical v1 raw dump


@pytest.mark.parametrize('part',['readme','context','judgments','split','exclusions','intent','disliked'])
def test_every_raw_source_label_split_and_exclusion_change_is_bound(part):
    old,policy=fixture_report();raw=read('bundle.json')
    if part=='readme':raw['dataset']['repositories'][0]['readme']='different frozen source'
    elif part=='context':raw['dataset']['profiles'][0]['context_text']='different original profile context'
    elif part=='judgments':
        raw['dataset']['judgments'][0]['grade']=3
        raw['judgments']=deepcopy(raw['dataset']['judgments'])
    elif part=='intent':
        raw['dataset']['profiles'][0]['intent']='different declared intent';raw['profiles'][0]['intent']='different declared intent'
    elif part=='disliked':
        changed=[raw['repositories'][-1]['repo_id']]
        raw['dataset']['profiles'][0]['disliked']=changed;raw['profiles'][0]['disliked']=changed
    elif part=='split':raw['dataset']['split'][raw['profiles'][0]['profile_id']]='other'
    else:
        raw['dataset']['profiles'][0]['seen']=[];raw['profiles'][0]['seen']=[]
    changed=Bundle.model_validate(raw)
    assert changed.dataset.sha256!=old.report.evaluation_dataset_sha256
    policy=Policy.model_validate({**policy.model_dump(mode='json'),'embedding_comparison':'cross_embedding'})
    baseline=Baseline.model_validate({**old.model_dump(mode='json'),'policy_sha256':digest(policy)})
    report={**old.report.model_dump(mode='json'),'evaluation_dataset_sha256':changed.dataset.sha256}
    assert any('identical verified raw' in f for f in assess(report,baseline,policy)['failures'])


def test_dataset_cannot_include_vector_material_or_disagree_with_bundle():
    raw=read('bundle.json');raw['dataset']['repositories'][0]['ranking_material']['nested']={'vector':[1.,0.]}
    with pytest.raises(ValueError,match='model material'):Bundle.model_validate(raw)
    raw=read('bundle.json');raw['dataset']['profiles'][0]['seen']=[]
    with pytest.raises(ValueError,match='profile/exclusions'):Bundle.model_validate(raw)


def test_actual_snapshot_must_match_bound_non_model_material():
    raw=read('snapshot.json');raw['profiles'][0]['payload']['skills']=['other-context']
    with pytest.raises(ValueError,match='actual profile context'):FrozenSnapshot.model_validate(raw)
    raw=read('snapshot.json');raw['repositories'][0]['payload']['fork_count']+=1
    with pytest.raises(ValueError,match='actual ranking material'):FrozenSnapshot.model_validate(raw)


def test_actual_candidate_report_gates_against_fixed_dated_baseline():
    require_local_only_git()
    snapshot=FrozenSnapshot.model_validate(read('snapshot.json'))
    exported=execute_snapshot(snapshot,ROOT/'inference')
    baseline,policy=fixture_report();report=evaluate(snapshot.bundle,exported.run,k=policy.k)
    assert assess(report,baseline,policy)['passed']
    changed=exported.run.model_dump(mode='json')
    for output in changed['outputs']:
        positive={j.repo_id for j in snapshot.bundle.judgments if j.profile_id==output['profile_id'] and j.grade>0}
        output['stages']['assembly']=[r for r in output['stages']['ranking'] if r['repo_id'] not in positive][:policy.k]
    regression=evaluate(snapshot.bundle,Run.model_validate(changed),k=policy.k)
    assert not assess(regression,baseline,policy)['passed']
    with pytest.raises(ValueError,match='incompatible embedding identity'):
        evaluate(snapshot.bundle,Run.model_validate({**changed,'embedding':{**changed['embedding'],'revision':'e'*40}}),k=policy.k)


@pytest.mark.parametrize('field',['identity','dimension','feature_schema','weights','scaler','file_drift','unsupported_runtime','serving_model','serving_revision','serving_version','serving_compatible'])
def test_cross_encoder_cli_requires_actual_manifest_contract(tmp_path,field,monkeypatch):
    # Deliberately synthetic declarations test validation only. These tiny byte
    # artifacts are not models and can never yield qualification evidence.
    import hashlib
    from scripts import evaluate_recommendations as cli
    snapshot=read('snapshot.json');old,policy=fixture_report()
    policy=Policy.model_validate({**policy.model_dump(mode='json'),'embedding_comparison':'cross_embedding'})
    model=tmp_path/'model.pt';model.write_bytes(b'synthetic model bytes')
    scaler=tmp_path/'scaler.json';scaler.write_bytes(b'synthetic scaler bytes')
    # Preserve the independently recorded stage rows, with explicit synthetic
    # test artifact identities instead of pretending they came from inference.
    recorded=read('baseline.json')['report']['artifact']
    outputs=[]
    for profile in snapshot['bundle']['profiles']:
        rows=[{'repo_id':r['repo_id'],'source':'fixture','source_score':0.,'score':0.} for r in snapshot['bundle']['repositories'] if r['repo_id'] not in set(profile['seen'])|set(profile['disliked'])]
        outputs.append({'profile_id':profile['profile_id'],'stages':{stage:rows for stage in ('retrieval','ranking','assembly')},'latency_ms':{stage:1. for stage in ('retrieval','ranking','assembly')}})
    run={**recorded,'embedding':snapshot['bundle']['embedding'],'outputs':outputs,
         'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'scaler_sha256':hashlib.sha256(scaler.read_bytes()).hexdigest()}
    manifest={'embedding_identity':deepcopy(run['embedding']),'embedding_dim':run['embedding']['dimension'],
              'model_sha256':run['model_sha256'],'scaler_sha256':run['scaler_sha256'],'feature_spec_version':run['feature_schema'],'feature_count':10,'input_dim':778}
    manifest.update(embedding_model=run['embedding']['model'], embedding_model_revision=run['embedding']['revision'], embedding_version=run['embedding']['recipe'], compatible_embedding_versions=[run['embedding']['recipe']])
    good_manifest=deepcopy(manifest)
    if field=='identity':manifest['embedding_identity']['query_prompt']='wrong-role-prompt'
    elif field=='dimension':manifest['embedding_dim']=1024
    elif field=='feature_schema':manifest['feature_spec_version']='wrong-schema'
    elif field=='weights':manifest['model_sha256']='a'*64
    elif field=='scaler':manifest['scaler_sha256']='b'*64
    elif field=='file_drift':pass # Hash valid bytes, then swap at the actual read boundary.
    elif field=='serving_model':manifest['embedding_model']='stale-model'
    elif field=='serving_revision':manifest['embedding_model_revision']='a'*40
    elif field=='serving_version':manifest['embedding_version']='stale-version'
    elif field=='serving_compatible':manifest['compatible_embedding_versions']=['stale-version']
    elif field=='unsupported_runtime':
        run['embedding']['dimension']=1024
        snapshot['bundle']['embedding']['dimension']=1024
        for row in snapshot['bundle']['repositories']:row['vector'] += [0.]*(1024-len(row['vector']))
        manifest['embedding_identity']=deepcopy(run['embedding'])
        manifest['embedding_dim']=1024;manifest['input_dim']=2058
    manifest_path=tmp_path/'manifest.json';manifest_path.write_text(json.dumps(manifest));run['manifest_sha256']=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    swapped=[]
    if field=='file_drift':
        original=cli.read_regular
        def replace_before_read(path, limit):
            if path==manifest_path:
                changed=deepcopy(good_manifest)
                changed['embedding_identity']['query_prompt']='swapped wrong prompt'
                manifest_path.write_text(json.dumps(changed))
                swapped.append(True)
            return original(path,limit)
        monkeypatch.setattr(cli,'read_regular',replace_before_read)
    bundle=Bundle.model_validate(snapshot['bundle']);parsed=Run.model_validate(run);report=evaluate(bundle,parsed,k=policy.k)
    baseline={'schema_version':1,'status':'diagnostic','reference':'synthetic manifest-mechanics only',
        'policy_sha256':digest(policy),'report_sha256':digest(report),'report':report}
    argv=[]
    for name,value in [('bundle',snapshot['bundle']),('run',run),('baseline',baseline),('policy',policy.model_dump(mode='json'))]:
        path=tmp_path/(name+'.json');path.write_text(json.dumps(value));argv.extend(['--'+name,str(path)])
    output=tmp_path/'out.json'
    assert cli.main([*argv,'--output',str(output),'--qualify','--model-artifact',str(model),'--scaler-artifact',str(scaler),'--manifest-artifact',str(manifest_path)])==1
    payload=json.loads(output.read_text())
    expected='actual candidate manifest digest mismatch' if field=='file_drift' else 'candidate manifest embedding/feature/model compatibility mismatch'
    assert expected in payload['assessment']['failures']
    if field=='file_drift':assert swapped==[True]
    assert payload['qualification_manifest'] is None


@pytest.mark.parametrize('field',['source_hash','readme_chunks','fallback_content_hash'])
def test_common_dataset_identity_excludes_encoder_derived_run_metadata(field):
    from evaluation.dataset_identity import SourceDataset
    raw=read('bundle.json')['dataset']
    if field=='fallback_content_hash':
        raw['repositories'][0]['ranking_material']['content_hash']=raw['repositories'][0]['ranking_material']['source_hash']
        raw['repositories'][0]['content_hash_origin']='encoder_fallback'
        raw['repositories'][0]['readme_sha256']=__import__('hashlib').sha256(b'').hexdigest()
    before=SourceDataset.model_validate(raw)
    changed=deepcopy(raw);material=changed['repositories'][0]['ranking_material']
    if field=='source_hash':material['source_hash']='e'*64
    elif field=='readme_chunks':material['readme_chunks']+=1
    else:material.update(source_hash='e'*64,content_hash='e'*64)
    assert SourceDataset.model_validate(changed).sha256==before.sha256


def test_snapshot_accepts_different_run_fingerprint_without_changing_raw_sources():
    raw=read('snapshot.json');before=FrozenSnapshot.model_validate(raw)
    raw['repositories'][0]['payload']['source_hash']='e'*64
    raw['repositories'][0]['payload']['readme_chunks']+=1
    after=FrozenSnapshot.model_validate(raw)
    assert after.bundle.dataset.sha256==before.bundle.dataset.sha256


def test_backend_hash_is_preserved_and_ambiguous_fallback_requires_explicit_raw_identity():
    from evaluation.dataset_identity import SourceDataset
    raw=read('bundle.json')['dataset'];source=raw['repositories'][0]
    source['content_hash_origin']='backend';source['ranking_material']['content_hash']=source['ranking_material']['source_hash']
    before=SourceDataset.model_validate(raw)
    changed=deepcopy(raw);changed['repositories'][0]['ranking_material']['content_hash']='f'*64
    assert SourceDataset.model_validate(changed).sha256!=before.sha256
    source.pop('content_hash_origin')
    with pytest.raises(ValueError,match='explicit raw README'):SourceDataset.model_validate(raw)
    source['content_hash_origin']='encoder_fallback';source['readme_sha256']='a'*64
    before=SourceDataset.model_validate(raw)
    changed=deepcopy(raw);changed['repositories'][0]['readme_sha256']='b'*64
    assert SourceDataset.model_validate(changed).sha256!=before.sha256


def test_common_projection_keeps_raw_binding_across_encoder_and_recipe_variants():
    from evaluation.dataset_identity import common_ranking_material
    payload=read('snapshot.json')['repositories'][0]['payload']
    changed=deepcopy(payload)
    changed.update(embedding_model='other-controlled-384-encoder',embedding_version='different-recipe',
        embedding_model_revision='f'*40,source_hash='e'*64,readme_chunks=99)
    assert common_ranking_material(changed,content_hash_origin='backend')==common_ranking_material(payload,content_hash_origin='backend')
    changed['content_hash']='f'*64
    assert common_ranking_material(changed,content_hash_origin='backend')!=common_ranking_material(payload,content_hash_origin='backend')


def test_present_raw_readme_bytes_must_match_any_declared_hash():
    from evaluation.dataset_identity import SourceDataset
    raw=read('bundle.json')['dataset'];source=raw['repositories'][0]
    source.update(readme='actual frozen bytes',readme_sha256='a'*64)
    with pytest.raises(ValueError,match='bytes and declared hash differ'):SourceDataset.model_validate(raw)
