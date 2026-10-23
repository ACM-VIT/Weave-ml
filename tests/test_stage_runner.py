"""Actual frozen-component replay, with explicitly synthetic diagnostic labels."""
import copy
import hashlib
from pathlib import Path
import pytest
from evaluation.recommendation_benchmark import digest, evaluate
from evaluation.stage_runner import FrozenSnapshot, StageRunExport, execute_snapshot
from tests.test_tower_history import require_local_only_git
from embedding.vector_contract import repository_payload_defaults
from config import EMBEDDING_MODEL_REVISION, REPOSITORY_EMBEDDING_MODEL, REPOSITORY_EMBEDDING_VERSION
ROOT=Path(__file__).resolve().parents[1]


def snapshot_data():
    rid=lambda n:f'00000000-0000-4000-8000-{n:012d}'
    vector=lambda n:[float(i==n) for i in range(384)]
    embedding=dict(model=REPOSITORY_EMBEDDING_MODEL,revision=EMBEDDING_MODEL_REVISION,dimension=384,
        pooling='mean',normalized=True,query_prompt='',document_prompt='',recipe=REPOSITORY_EMBEDDING_VERSION)
    repositories=[];points=[]
    for i in range(8):
        repositories.append(dict(repo_id=rid(i+1),owner=str(i+1),topics=['python'],fork_family=str(i+1),
            vector=vector(i%3),popularity=i+10,archived=i==7))
        p=repository_payload_defaults()
        p.update(repo_id=rid(i+1),full_name=f'diagnostic/repo{i}',content_version=1,content_hash='a'*64,source_hash='b'*64,
            indexed_at='2026-10-24T00:00:00Z',github_id=str(i+1),owner_github_id=str(i+1),is_fork=False,fork_family_github_id=str(i+1),
            lineage_source='github-graphql-repository',lineage_observed_at='2026-10-24T00:00:00Z',star_count=i+10,topics=['python'],primary_language='Python',languages=['Python'],pushed_at='2026-10-23T00:00:00Z')
        points.append(dict(repo_id=rid(i+1),payload=p))
    profiles=[dict(profile_id=rid(100+i),intent='synthetic Python profile',seen=[rid(1)],disliked=[rid(2)]) for i in range(2)]
    bundle=dict(schema_version=1,snapshot_id='synthetic-actual-stages-v1',created_at='2026-10-24T00:00:00Z',
        provenance=dict(kind='synthetic',reference='generated unit diagnostic, no human relevance judgments'),embedding=embedding,
        repositories=repositories,profiles=profiles,judgments=[dict(profile_id=p['profile_id'],repo_id=rid(3),grade=1) for p in profiles])
    user_payload=dict(embedding_dim=384,embedding_model=embedding['model'],embedding_model_revision=embedding['revision'],
        embedding_version=embedding['recipe'],profile_version=1,last_feedback_version=0,feedback_state_revision=0,
        skills=['Python'],interests=['search'],tech_stack=[],topics=['python'])
    names={'model_sha256':'heavy_ranker.pt','scaler_sha256':'feature_scaler.json','manifest_sha256':'model_manifest.json'}
    return dict(schema_version=1,bundle=bundle,as_of='2026-10-24T00:00:00Z',limit=4,seed='diagnostic-fixed',
        artifacts={key:hashlib.sha256((ROOT/'inference'/name).read_bytes()).hexdigest() for key,name in names.items()},
        repositories=points,profiles=[dict(profile_id=p['profile_id'],vector=vector(i),payload=copy.deepcopy(user_payload)) for i,p in enumerate(profiles)])


@pytest.fixture(scope='module')
def real_export():
    require_local_only_git()
    snapshot=FrozenSnapshot.model_validate(snapshot_data())
    return snapshot,execute_snapshot(snapshot,ROOT/'inference')


def test_actual_production_components_export_separate_measured_stages(real_export):
    snapshot,exported=real_export
    assert exported.run.timing_kind=='measured'
    assert exported.bundle_sha256==digest(snapshot.bundle)
    assert exported.snapshot_sha256==digest(snapshot)
    assert exported.artifacts.model_dump()==snapshot.artifacts.model_dump()
    assert exported.execution_scope=='offline-local-qdrant-exact-cpu'
    assert {'retrieval/config.py','inference/feature_spec.py','inference/value_function.py',
            'inference/diversity.py','embedding/vector_contract.py'}<=set(exported.source_sha256)
    assert exported.configuration['retrieval']['heavy_ranker_required'] is True
    assert exported.configuration['runtime']['device']=='cpu'
    for output in exported.run.outputs:
        retrieval=output.stages['retrieval'];ranking=output.stages['ranking'];assembly=output.stages['assembly']
        assert len(retrieval)==len(ranking)==5 and len(assembly)==4
        assert {r.repo_id for r in retrieval}=={r.repo_id for r in ranking}
        assert all(r.repo_id not in {snapshot.bundle.repositories[i].repo_id for i in (0,1,7)} for r in assembly)
        assert [r.score for r in ranking]==sorted([r.score for r in ranking],reverse=True)
        assert any(a.score!=b.score for a,b in zip(retrieval,ranking))
        assert all(v>=0 for v in output.latency_ms.values())
        assert sum(output.latency_ms.values())<=exported.total_ms[output.profile_id]+1
    report=evaluate(snapshot.bundle,exported.run,k=4)
    assert report['provenance']['kind']=='synthetic'
    assert report['stages']['assembly']['metrics']['excluded_rate']==0
    assert report['stages']['assembly']['metrics']['archived_rate']==0
    assert StageRunExport.model_validate_json(exported.model_dump_json())==exported


def test_actual_runner_repeats_identical_stages_with_frozen_time(real_export):
    snapshot,first=real_export
    again=execute_snapshot(snapshot,ROOT/'inference')
    assert [o.stages for o in first.run.outputs]==[o.stages for o in again.run.outputs]
    assert first.run.model_sha256==again.run.model_sha256


@pytest.mark.parametrize('case',['vector','revision','payload','features','duplicate_profile','metadata','size','prompt','model-version'])
def test_snapshot_rejects_incoherent_or_unbounded_inputs(case):
    value=snapshot_data()
    if case=='vector':value['profiles'][0]['vector'][0]=float('nan')
    if case=='revision':value['profiles'][0]['payload']['profile_version']=True
    if case=='payload':value['repositories'][0]['payload']['repo_id']=value['repositories'][1]['repo_id']
    if case=='features':value['repositories'][0]['payload']['feature_version']=-1
    if case=='duplicate_profile':value['profiles'][1]=copy.deepcopy(value['profiles'][0])
    if case=='model-version':value['repositories'][0]['payload']['model_version']='stale-model'
    if case=='metadata':value['repositories'][0]['payload']['star_count']=999
    if case=='size':value['profiles']*=33
    if case=='prompt':value['bundle']['embedding']['query_prompt']='invented prompt'
    with pytest.raises((ValueError,TypeError)):FrozenSnapshot.model_validate(value)


def test_changed_actual_artifact_fails_before_model_loading(tmp_path):
    snapshot=FrozenSnapshot.model_validate(snapshot_data())
    for name in ('heavy_ranker.pt','feature_scaler.json','model_manifest.json'):
        (tmp_path/name).write_bytes(b'changed artifact')
    with pytest.raises(ValueError,match='artifact identity'):
        execute_snapshot(snapshot,tmp_path)


def test_evaluator_consumes_bound_stage_export_and_rejects_other_bundle(real_export,tmp_path):
    import json
    from evaluation.benchmark_gates import Policy
    from scripts import evaluate_recommendations as cli
    snapshot,exported=real_export
    policy=Policy.model_validate(dict(schema_version=1,policy_id='diagnostic',reference='synthetic test only',k=4,min_profiles=2,
        rules=[dict(stage='assembly',metric='excluded_rate',maximum=0.,max_regression=0.)]))
    report=evaluate(snapshot.bundle,exported.run,k=4)
    baseline=dict(schema_version=1,status='diagnostic',reference='synthetic actual-artifact test',policy_sha256=digest(policy),report_sha256=digest(report),report=report)
    values={'bundle':snapshot.bundle.model_dump(mode='json'),'run':exported.model_dump(mode='json'),
            'baseline':baseline,'policy':policy.model_dump(mode='json')}
    args=[]
    for name,value in values.items():
        path=tmp_path/f'{name}.json';path.write_text(json.dumps(value));args.extend([f'--{name}',str(path)])
    output=tmp_path/'output.json';args.extend(['--output',str(output)])
    assert cli.main(args)==0
    assert json.loads(output.read_text())['report']==report
    assert cli.main([*args,'--qualify','--model-artifact',str(ROOT/'inference/heavy_ranker.pt'),
                     '--scaler-artifact',str(ROOT/'inference/feature_scaler.json')])==1
    assert 'local exact-search stage replay is diagnostic, not production qualification evidence' in json.loads(output.read_text())['assessment']['failures']
    old=output.read_bytes();values['bundle']['snapshot_id']='different snapshot'
    (tmp_path/'bundle.json').write_text(json.dumps(values['bundle']))
    assert cli.main(args)==2
    assert output.read_bytes()==old


def test_real_cli_worker_is_offline_bounded_and_preserves_old_output(tmp_path):
    require_local_only_git()
    import json
    from scripts.run_recommendation_stages import main
    source=tmp_path/'snapshot.json';source.write_text(json.dumps(snapshot_data()))
    output=tmp_path/'run.json'
    assert main(['--snapshot',str(source),'--output',str(output)])==0
    exported=StageRunExport.model_validate_json(output.read_text())
    assert len(exported.run.outputs)==2
    old=output.read_bytes();value=snapshot_data();value['limit']=101
    source.write_text(json.dumps(value))
    assert main(['--snapshot',str(source),'--output',str(output)])==2
    assert output.read_bytes()==old


def test_cli_worker_timeout_and_snapshot_drift_preserve_output(tmp_path,monkeypatch):
    import json,subprocess
    from scripts import run_recommendation_stages as cli
    source=tmp_path/'snapshot.json';source.write_text(json.dumps(snapshot_data()))
    output=tmp_path/'run.json';output.write_bytes(b'previous successful output')
    def timeout(*args,**kwargs):
        if args[0][0]=='git':
            return subprocess.CompletedProcess(args[0],0,b'git version fixture',b'')
        assert kwargs['timeout']==180
        worker_output=Path(args[0][args[0].index('--output')+1])
        worker_output.write_bytes(b'completed worker evidence before cleanup timeout')
        raise subprocess.TimeoutExpired(args[0],180)
    monkeypatch.setattr(cli.subprocess,'run',timeout)
    assert cli.main(['--snapshot',str(source),'--output',str(output)])==2
    assert output.read_bytes()==b'previous successful output'
    assert cli.main(['--snapshot',str(source),'--output',str(output),'--worker','--expected-snapshot-sha','a'*64])==2
    assert output.read_bytes()==b'previous successful output'


def test_exported_final_stage_matches_actual_returned_batch_order_and_scores(monkeypatch):
    require_local_only_git()
    from retrieval.v2_retriever import QdrantV2Retriever
    original=QdrantV2Retriever.recommend_batch
    batches={}
    def observe(self,user_id,*args,**kwargs):
        batch=original(self,user_id,*args,**kwargs)
        batches[user_id]=[(item.repo_id,item.score,item.source,item.retrieval_scores[item.source]) for item in batch.items]
        return batch
    monkeypatch.setattr(QdrantV2Retriever,'recommend_batch',observe)
    exported=execute_snapshot(FrozenSnapshot.model_validate(snapshot_data()),ROOT/'inference')
    for output in exported.run.outputs:
        assert [(r.repo_id,r.score,r.source,r.source_score) for r in output.stages['assembly']]==batches[output.profile_id]


def test_partial_query_failure_cannot_export_discovery_only_success(monkeypatch):
    require_local_only_git()
    from qdrant_client import QdrantClient
    def unavailable(*args,**kwargs):
        raise RuntimeError('controlled local semantic query failure')
    monkeypatch.setattr(QdrantClient,'query_points',unavailable)
    with pytest.raises(ValueError,match='partial/fallback'):
        execute_snapshot(FrozenSnapshot.model_validate(snapshot_data()),ROOT/'inference')


def test_cli_oversized_input_rejected_before_starting_worker(tmp_path,monkeypatch):
    from scripts import run_recommendation_stages as cli
    source=tmp_path/'snapshot.json';source.write_bytes(b' '* (8*1024*1024+1))
    output=tmp_path/'output.json';output.write_bytes(b'prior evidence')
    monkeypatch.setattr(cli.subprocess,'run',lambda *a,**k:pytest.fail('oversized snapshot started worker'))
    assert cli.main(['--snapshot',str(source),'--output',str(output)])==2
    assert output.read_bytes()==b'prior evidence'


def test_replay_owns_deep_snapshot_despite_caller_aba_mutation(real_export,monkeypatch):
    from retrieval.v2_retriever import QdrantV2Retriever
    snapshot=FrozenSnapshot.model_validate(snapshot_data())
    original_digest=digest(snapshot)
    original=QdrantV2Retriever._fetch_semantic
    calls=0
    original_seed=snapshot.seed
    def caller_changes_then_restores(self,*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==1:
            snapshot.limit=1;snapshot.seed='caller changed during replay'
            snapshot.bundle.profiles[1].seen.append(snapshot.bundle.repositories[2].repo_id)
            snapshot.profiles[1].payload['skills'].append('caller changed nested data')
        elif calls==2:
            snapshot.limit=4;snapshot.seed=original_seed
            snapshot.bundle.profiles[1].seen.pop()
            snapshot.profiles[1].payload['skills'].pop()
        return original(self,*args,**kwargs)
    monkeypatch.setattr(QdrantV2Retriever,'_fetch_semantic',caller_changes_then_restores)
    exported=execute_snapshot(snapshot,ROOT/'inference')
    assert calls==2 and digest(snapshot)==original_digest
    assert exported.snapshot_sha256==original_digest and exported.seed==original_seed
    assert [row.stages for row in exported.run.outputs]==[row.stages for row in real_export[1].run.outputs]


def test_unsupported_git_fails_with_an_actionable_cli_result_before_worker(tmp_path, monkeypatch, capsys):
    import json, subprocess
    from evaluation.stage_runner import OfflineGitUnavailable, source_identity
    from scripts import run_recommendation_stages as cli
    source=tmp_path/'snapshot.json';source.write_text(json.dumps(snapshot_data()))
    output=tmp_path/'run.json';output.write_bytes(b'prior evidence')
    calls=[]
    def unsupported(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args,129,b'',b'unknown option: --no-lazy-fetch')
    monkeypatch.setattr(subprocess,'run',unsupported)
    with pytest.raises(OfflineGitUnavailable,match='requires Git supporting --no-lazy-fetch'):
        source_identity()
    assert cli.main(['--snapshot',str(source),'--output',str(output)])==2
    result=json.loads(capsys.readouterr().out)
    assert result['error']=='UnsupportedGit' and '--no-lazy-fetch' in result['detail']
    assert output.read_bytes()==b'prior evidence'
    assert calls==[['git','--no-lazy-fetch','--version']]*2


def test_replay_uses_the_production_readers_effective_vector_name(monkeypatch):
    require_local_only_git()
    monkeypatch.setattr('retrieval.v2_retriever.QDRANT_VECTOR_NAME','custom_repository_vector')
    exported=execute_snapshot(FrozenSnapshot.model_validate(snapshot_data()),ROOT/'inference')
    assert exported.configuration['runtime']['repository_vector_name']=='custom_repository_vector'
    assert all(output.stages['retrieval'] and output.stages['assembly'] for output in exported.run.outputs)
