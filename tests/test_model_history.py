"""Frozen synthetic mechanics use actual legacy transitions and serving stages."""
from copy import deepcopy
from dataclasses import asdict
import json
from datetime import datetime
from pathlib import Path

import pytest

from evaluation.model_history import FrozenModelHistory, derive_snapshots, compare_model_history, validate_model_comparison
from feedback.model_wire import canonical_json
from tests.test_model_operations import policy, uid


def experiment():
    snapshot=json.loads((Path(__file__).parent/'fixtures/recommendation_actual_stage/snapshot.json').read_text())
    p=policy();operations=[];sessions={}
    origin=int(datetime.fromisoformat(snapshot["as_of"].replace("Z","+00:00")).timestamp()*1000)-10
    for index,profile in enumerate(snapshot['profiles']):
        owner=profile['profile_id'];session=uid(10000+index);sessions[owner]={'session_id':session,'epoch':'1'}
        def operation(kind,version,payload,at):
            return canonical_json(dict(schema_version=1,operation_id=uid(11000+index*100+version),user_id=owner,
                model_operation_version=str(version),previous_model_operation_version=str(version-1),accepted_at_ms=str(origin+at),
                policy_version=p.version,policy_sha256=p.sha256,kind=kind,operation=payload))
        context={key:profile['payload'][key] for key in ('skills','tech_stack','interests','topics')}
        operations.append(operation('profile',1,dict(job_id=uid(12000+index),profile_version='1',allow_genesis=True,original_profile_context=context),1))
        operations.append(operation('begin',2,dict(session_id=session,epoch='1',epoch_started_ms=str(origin+2),witness_sha256='b'*64),2))
        for version,kind in [(3,'like'),(4,'like'),(5,'github_open'),(6,'save'),(7,'unsave')]:
            raw=canonical_json(dict(event_id=uid(13000+index*100+version),user_id=owner,repo_id=snapshot['repositories'][(index+1)%len(snapshot['repositories'])]['repo_id'],
                feedback_version=str(version-2),event_type=kind,occurred_at='2026-10-24T00:00:00Z'))
            operations.append(operation('feedback',version,dict(original_feedback_json=raw,canonical_attribution=dict(user_id=owner,
                session_id=session,epoch='1',epoch_started_ms=str(origin+2),active_at_acceptance=True,witness_sha256='b'*64)),version))
    return FrozenModelHistory.model_validate(dict(schema_version=1,snapshot=snapshot,policy=asdict(p),operations=operations,sessions=sessions,k=min(5,snapshot["limit"]),
        legacy_policy=dict(dwell_min_ms=3000,dwell_full_credit_ms=6000,dwell_max_alpha=.05,max_tracked_repositories=256,
                           max_user_state_bytes=1048576,vector_dimension=384,user_vector_name=None,repository_vector_name='repo_embedding')))


def test_same_frozen_history_has_distinct_exact_legacy_and_long_session_vectors():
    value=experiment();snapshots,storage=derive_snapshots(value)
    assert snapshots['single_vector'].bundle==snapshots['long_session'].bundle
    assert any(a.vector!=b.vector for a,b in zip(snapshots['single_vector'].profiles,snapshots['long_session'].profiles))
    assert all(row['feedback_version']=='5' and row['operation_version']=='7' for row in storage.values())
    assert derive_snapshots(value)==(snapshots,storage)
    assert value.snapshot.bundle.provenance.kind=='synthetic'


def test_actual_production_stages_compare_same_history_and_parent_refuses_forgery():
    value=experiment();result=compare_model_history(value,Path(__file__).parents[1]/'inference')
    assert result['qualified'] is False and set(result['results'])=={'single_vector','long_session'}
    assert validate_model_comparison(result,value)==result
    assert all(row['execution']['run']['timing_kind']=='measured' for row in result['results'].values())
    assert all(row['execution']['configuration']['runtime']['retrieval_executor']['stages']['profile']['calls_total']==len(value.snapshot.profiles)
               for row in result['results'].values())
    cumulative=deepcopy(result)
    cumulative['results']['long_session']['execution']['configuration']['runtime']['retrieval_executor']['stages']['profile']['calls_total']*=2
    with pytest.raises(ValueError):validate_model_comparison(cumulative,value)
    forged=deepcopy(result);forged['storage'][next(iter(forged['storage']))]['feedback_version']='9'
    with pytest.raises(ValueError):validate_model_comparison(forged,value)


@pytest.mark.parametrize('seen_count',[0,500])
def test_new_dislike_updates_both_common_source_and_bundle_exclusions(tmp_path,seen_count):
    value=experiment();raw=value.model_dump(mode='json')
    owner=raw['snapshot']['profiles'][0]['profile_id']
    if seen_count:
        existing=set(raw['snapshot']['bundle']['profiles'][0]['disliked'])
        seen=sorted(existing|{uid(500000+n) for n in range(seen_count-len(existing))})
        raw['snapshot']['bundle']['profiles'][0]['seen']=seen
        raw['snapshot']['bundle']['dataset']['profiles'][0]['seen']=list(seen)
    target=next(r['repo_id'] for r in raw['snapshot']['repositories'] if r['repo_id'] not in set(raw['snapshot']['bundle']['profiles'][0]['disliked'])|set(raw['snapshot']['bundle']['profiles'][0]['seen']))
    for index,original in enumerate(raw['operations']):
        operation=json.loads(original)
        if operation['user_id']==owner and operation['kind']=='feedback':
            event=json.loads(operation['operation']['original_feedback_json'])
            event.update(repo_id=target,event_type='dislike')
            operation['operation']['original_feedback_json']=canonical_json(event)
            raw['operations'][index]=canonical_json(operation)
            break
    snapshots,_=derive_snapshots(FrozenModelHistory.model_validate(raw))
    for snapshot in snapshots.values():
        assert target in snapshot.bundle.profiles[0].disliked
        assert snapshot.bundle.profiles[0].disliked==snapshot.bundle.dataset.profiles[0].disliked
        if seen_count:
            assert len(snapshot.request_exclusions[owner])==500
            assert target not in snapshot.request_exclusions[owner]
            assert target in snapshot.model_exclusions[owner]
    assert snapshots['single_vector'].bundle==snapshots['long_session'].bundle

    # Run the actual parent/worker CLI, preserving the immutable source history
    # and requiring the newly disliked repository to disappear in both models.
    import os,subprocess,sys
    changed=FrozenModelHistory.model_validate(raw);before=changed.model_dump(mode='json')
    source=tmp_path/'history.json';source.write_text(changed.model_dump_json())
    output=tmp_path/'comparison.json'
    result=subprocess.run([sys.executable,'-B','-m','scripts.compare_user_models',
        '--experiment',str(source),'--output',str(output)],cwd=Path(__file__).resolve().parents[1],
        env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
        capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    assert changed.model_dump(mode='json')==before
    report=json.loads(output.read_text());assert report['qualified'] is False
    for condition in report['results'].values():
        if seen_count:
            counts=condition['execution']['configuration']['runtime']['exclusion_channels_by_profile'][owner]
            assert counts=={'request_count':500,'model_count':1,'combined_count':501}
        for profile in condition['execution']['run']['outputs']:
            if profile['profile_id']==owner:
                assert all(target not in [row['repo_id'] for row in rows] for rows in profile['stages'].values())
    forged=deepcopy(report)
    forged['results']['single_vector']['execution']['configuration']['runtime']['exclusion_channels_by_profile'][owner]['request_count']+=1
    with pytest.raises(ValueError,match='exclusion channels'):validate_model_comparison(forged,changed)
    for wrong in (True,1.0):
        forged=deepcopy(report)
        forged['results']['single_vector']['execution']['configuration']['runtime']['exclusion_channels_by_profile'][owner]['model_count']=wrong
        with pytest.raises(ValueError,match='exclusion channels'):validate_model_comparison(forged,changed)


@pytest.mark.parametrize('mutation',['request_overflow','model_overflow','missing_owner','unbound_negative'])
def test_separate_replay_exclusion_channels_remain_bounded_and_source_bound(mutation):
    from evaluation.stage_runner import FrozenSnapshot
    snapshots,_=derive_snapshots(experiment())
    raw=snapshots['long_session'].model_dump(mode='json');owner=raw['profiles'][0]['profile_id']
    if mutation=='request_overflow':raw['request_exclusions'][owner]=[uid(600000+n) for n in range(501)]
    elif mutation=='model_overflow':raw['model_exclusions'][owner]=[uid(600000+n) for n in range(257)]
    elif mutation=='missing_owner':raw['model_exclusions'].pop(owner)
    else:raw['model_exclusions'][owner]=[uid(600000)]
    with pytest.raises(ValueError):FrozenSnapshot.model_validate(raw)


def test_plain_replay_request_cannot_exceed_500_or_invent_history_model_negatives():
    from evaluation.stage_runner import FrozenSnapshot
    raw=experiment().model_dump(mode='json');owner=raw['snapshot']['profiles'][0]['profile_id']
    raw['snapshot']['bundle']['profiles'][0]['seen']=[uid(600000+n) for n in range(501)]
    raw['snapshot']['bundle']['dataset']['profiles'][0]['seen']=list(raw['snapshot']['bundle']['profiles'][0]['seen'])
    with pytest.raises(ValueError,match='backend/ML contract'):FrozenSnapshot.model_validate(raw['snapshot'])
    original=experiment();snapshot=original.snapshot.model_dump(mode='json')
    snapshot['request_exclusions']={p.profile_id:sorted(set(p.seen)|set(p.disliked)) for p in original.snapshot.bundle.profiles}
    snapshot['model_exclusions']={p.profile_id:list(p.disliked) for p in original.snapshot.bundle.profiles}
    raw=original.model_dump(mode='json');raw['snapshot']=snapshot
    with pytest.raises(ValueError,match='unproved derived'):FrozenModelHistory.model_validate(raw)
