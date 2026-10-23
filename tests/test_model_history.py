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
