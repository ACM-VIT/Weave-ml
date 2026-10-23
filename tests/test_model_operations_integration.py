"""Opt-in real HTTP/Redis/Qdrant acceptance, confined to owned Docker.

No service URL or actual.env is accepted. Names, images, loopback ports and
ownership labels are checked on the pinned local Unix daemon before cleanup.
"""
from contextlib import contextmanager
from dataclasses import replace
import json
import os
import subprocess
import time
import uuid

import pytest
import redis
from qdrant_client import QdrantClient, models

from tests.test_model_operations import policy, wire, parse, uid, basis, profile_payload, begin_payload, feedback_payload, runtime_config
from feedback.model_runtime import ModelRuntime
from feedback.model_stream import OperationBytesConflict, OperationCapacity
from feedback.model_journal import JournalConflict, JournalMissing, CasConflict
from feedback import long_session as model
from feedback.model_serving import Context, owned_input
from feedback.v2_settings import V2FeedbackSettings
from feedback.user_lock import LockLostError
from config import REPOSITORY_EMBEDDING_VERSION, REPOSITORY_FEATURE_SPEC_VERSION, REPOSITORY_EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION
from embedding.vector_contract import REPOSITORY_SERVING_ELIGIBILITY_FIELD, REPOSITORY_SERVING_ELIGIBILITY_VERSION

pytestmark = pytest.mark.integration
DAEMON = 'unix:///var/run/docker.sock'


def docker(*args):
    env = {key: value for key, value in os.environ.items() if not key.startswith('DOCKER_')}
    return subprocess.check_output(['docker', '--host', DAEMON, *args], env=env, text=True, timeout=40).strip()


@contextmanager
def service(image, port):
    image_id = json.loads(docker('image', 'inspect', image))[0]['Id']
    owner = uuid.uuid4().hex
    name = 'weave-model-test-' + owner[:12]
    container = docker('run', '--pull=never', '-d', '--name', name, '--label', 'weave.model-test-owner='+owner,
                       '-p', f'127.0.0.1::{port}', image_id)
    try:
        state = json.loads(docker('inspect', container))[0]
        assert state['Image'] == image_id and state['Config']['Labels']['weave.model-test-owner'] == owner
        binding = state['NetworkSettings']['Ports'][f'{port}/tcp'][0]
        assert binding['HostIp'] == '127.0.0.1'
        yield '127.0.0.1:' + binding['HostPort']
    finally:
        state = json.loads(docker('inspect', container))[0]
        assert state['Image'] == image_id and state['Config']['Labels']['weave.model-test-owner'] == owner
        docker('rm', '-f', container)


@pytest.fixture
def runtime(request):
    if os.getenv('RUN_MODEL_OPERATION_INTEGRATION') != '1':
        pytest.skip('set RUN_MODEL_OPERATION_INTEGRATION=1 for owned local Docker acceptance')
    with service('redis:7.4.5-alpine', 6379) as redis_address, service('qdrant/qdrant:v1.18.2', 6333) as qdrant_address:
        r = redis.Redis.from_url('redis://'+redis_address, decode_responses=True)
        q = QdrantClient(url='http://'+qdrant_address, api_key=None, timeout=3)
        for attempt in range(100):
            try:
                r.ping(); q.get_collections(); break
            except Exception:
                if attempt == 99: raise
                time.sleep(.05)
        p = policy(**getattr(request, "param", {}))
        for name, vectors in [('fixture_model_users', models.VectorParams(size=384, distance=models.Distance.COSINE)),
                              ('fixture_model_journal', {}), ('fixture_model_repositories', {'repo_embedding': models.VectorParams(size=384, distance=models.Distance.COSINE)})]:
            q.create_collection(name, vectors_config=vectors)
        settings = replace(V2FeedbackSettings.from_env(), redis_url='redis://'+redis_address,
            qdrant_url='http://'+qdrant_address, qdrant_api_key=None,
            user_collection='fixture_model_users', repository_collection='fixture_model_repositories',
            user_lock_prefix='fixture-model-lock', user_lock_ttl_seconds=1.0,
            user_lock_renew_interval_seconds=.1, user_lock_wait_seconds=.1)
        q.upsert(settings.repository_collection, points=[models.PointStruct(id=uid(100), vector={'repo_embedding': basis(1)}, payload={
            'repo_id':uid(100), REPOSITORY_SERVING_ELIGIBILITY_FIELD:REPOSITORY_SERVING_ELIGIBILITY_VERSION,
            'content_version':1, 'feature_version':1, 'embedding_model':REPOSITORY_EMBEDDING_MODEL,
            'embedding_model_revision':EMBEDDING_MODEL_REVISION, 'embedding_version':REPOSITORY_EMBEDDING_VERSION,
            'embedding_dim':384, 'feature_spec_version':REPOSITORY_FEATURE_SPEC_VERSION})], wait=True)
        active = ModelRuntime(runtime_config(p), redis=r, qdrant=q, encoder=lambda context: basis(0), settings=settings)
        try:
            yield active
        finally:
            active.close()


def apply(runtime, raw):
    runtime.stream.admit(raw)
    assert runtime.consumer().run_once() == 1


def initialize(runtime):
    p=runtime.policy
    apply(runtime, wire(p, 'profile', 1, profile_payload()))
    apply(runtime, wire(p, 'begin', 2, begin_payload()))


def test_actual_genesis_begin_feedback_reset_end_and_exact_rebuild(runtime):
    initialize(runtime); p=runtime.policy
    checkpoint=runtime.processor.coordinator.checkpoint(uid(1))
    apply(runtime, wire(p, 'feedback', 3, feedback_payload(session=uid(50))))
    snapshot=runtime.users.read(uid(1))
    point=runtime.qdrant.retrieve(runtime.settings.user_collection, ids=[uid(1)], with_payload=True, with_vectors=True)[0]
    effective=owned_input(point,runtime.users,p,Context(uid(1),uid(50),1,p.sha256,3,1,3))
    assert effective.vector[1]>0 and snapshot.state.feedback_version==1
    apply(runtime,wire(p,'reset',4,dict(session_id=uid(50),old_epoch='1',new_epoch='2',new_epoch_started_ms='4',witness_sha256='c'*64)))
    state=runtime.users.read(uid(1)).state
    assert state.long_entries and not state.sessions[0].entries
    apply(runtime,wire(p,'end',5,dict(session_id=uid(50),epoch='2',epoch_started_ms='4',retired=True,witness_sha256='d'*64)))
    final=runtime.users.read(uid(1)); assert not final.state.sessions
    restored=runtime.processor.coordinator.rebuild(checkpoint,final.head,expected_user_id=uid(1))
    assert restored==final
    first=wire(p,'profile',1,profile_payload())
    assert runtime.processor(parse(first,p),lambda:None)=='duplicate'
    assert runtime.stream.admit(first)['status']=='duplicate'
    conflict=json.loads(first); conflict['operation']['original_profile_context']['interests']=['other']
    from feedback.model_wire import canonical_json
    with pytest.raises(OperationBytesConflict):runtime.stream.admit(canonical_json(conflict))


def test_applied_then_lost_ack_reclaims_by_reachable_head(runtime):
    initialize(runtime);p=runtime.policy
    raw=wire(p,'feedback',3,feedback_payload());runtime.stream.admit(raw)
    consumer=runtime.consumer()
    original=runtime.stream.acknowledge
    runtime.stream.acknowledge=lambda _:(_ for _ in ()).throw(RuntimeError('crash after CAS before ACK'))
    with pytest.raises(RuntimeError):consumer.run_once()
    before=runtime.users.read(uid(1));assert before.state.feedback_version==1
    runtime.stream.acknowledge=original
    time.sleep(.02)
    assert runtime.consumer().run_once()==1
    assert runtime.users.read(uid(1))==before
    assert runtime.redis.xlen(runtime.stream.stream)==0


def test_lock_loss_before_cas_leaves_prepared_only_pending(runtime):
    initialize(runtime);p=runtime.policy
    raw=wire(p,'feedback',3,feedback_payload());before=runtime.users.read(uid(1))
    calls=0
    def lose():
        nonlocal calls
        calls+=1
        if calls==2:raise LockLostError('fixture loss before CAS')
    with pytest.raises(LockLostError):runtime.processor(parse(raw,p),lose)
    assert runtime.users.read(uid(1))==before
    assert runtime.processor(parse(raw,p),lambda:None)=='applied'


def test_gap_missing_dependency_capacity_and_foreign_genesis_do_not_advance(runtime):
    initialize(runtime);p=runtime.policy;before=runtime.users.read(uid(1))
    with pytest.raises(model.OperationGap):runtime.processor(parse(wire(p,'feedback',4,feedback_payload()),p),lambda:None)
    assert runtime.users.read(uid(1))==before
    runtime.qdrant.delete(runtime.settings.repository_collection,points_selector=models.PointIdsList(points=[uid(100)]),wait=True)
    from feedback.v2 import MissingVectorError
    with pytest.raises(MissingVectorError):runtime.processor(parse(wire(p,'feedback',3,feedback_payload()),p),lambda:None)
    assert runtime.users.read(uid(1))==before
    runtime.stream.limits=replace(runtime.stream.limits,max_pending=1)
    runtime.stream.admit(wire(p,'feedback',3,feedback_payload(kind='unsave')))
    with pytest.raises(OperationCapacity):runtime.stream.admit(wire(p,'feedback',4,feedback_payload(2,kind='unsave')))
    assert runtime.users.read(uid(1))==before


def test_registered_http_provider_delivers_acceptance_without_claiming_application(runtime,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.v2 import router
    import feedback.model_runtime as provider
    monkeypatch.setattr(provider,'model_runtime',lambda:runtime)
    monkeypatch.setenv('INTERNAL_API_SECRET','fixture-secret')
    app=FastAPI();app.include_router(router);client=TestClient(app)
    raw=wire(runtime.policy,'profile',1,profile_payload())
    response=client.post('/api/v2/model-operations',headers={'x-internal-secret':'fixture-secret'},
        json={'schema_version':1,'operations':[{'operation_json':raw}]})
    assert response.status_code==200,response.text
    assert response.json()['operations'][0]['status']=='accepted'
    with pytest.raises(JournalMissing):runtime.users.read(uid(1))
    assert runtime.consumer().run_once()==1
    assert runtime.users.read(uid(1)).state.operation_version==1
    assert client.post('/api/v2/model-operations',json={'schema_version':1,'operations':[{'operation_json':raw}]}).status_code==401


def test_two_devices_profile_order_dislike_and_sequential_retirement(runtime):
    initialize(runtime);p=runtime.policy
    apply(runtime,wire(p,'begin',3,begin_payload(3,uid(51))))
    apply(runtime,wire(p,'feedback',4,feedback_payload(kind='dislike',session=uid(50))))
    state=runtime.users.read(uid(1)).state
    assert model.norm(model.recommendation_view(state,p,session_id=uid(51),epoch=1,as_of_ms=4).session)==0
    assert model.recommendation_view(state,p,session_id=uid(51),epoch=1,as_of_ms=4).explicit_negative_repo_ids==(uid(100),)
    apply(runtime,wire(p,'profile',5,profile_payload(False,2)))
    snapshot=runtime.users.read(uid(1))
    assert snapshot.state.feedback_version==1 and snapshot.state.baseline.content_version==2
    assert json.loads(snapshot.complete)['legacy']['payload']['last_feedback_version']==1
    # Retire both current sessions, then exceed max_live_sessions sequentially.
    version=5
    for session,start in [(uid(50),2),(uid(51),3)]:
        version+=1;apply(runtime,wire(p,'end',version,dict(session_id=session,epoch='1',epoch_started_ms=str(start),retired=True,witness_sha256='d'*64)))
    for index in range(p.max_sessions+2):
        version+=1;session=uid(60+index);start=version;apply(runtime,wire(p,'begin',version,begin_payload(start,session)))
        version+=1;apply(runtime,wire(p,'end',version,dict(session_id=session,epoch='1',epoch_started_ms=str(start),retired=True,witness_sha256='d'*64)))
    assert runtime.users.read(uid(1)).state.sessions==()
    assert runtime.users.read(uid(1)).state.long_entries


from tests.test_scoring_provider import trained_ranker


@pytest.mark.parametrize("capture", [True, False])
def test_actual_ann_and_captured_trained_tensor_use_identical_one_read_input(runtime,trained_ranker,monkeypatch,capture):
    from fastapi.testclient import TestClient
    from api import main,v2
    import feedback.model_runtime as provider
    from embedding.vector_contract import repository_payload_defaults
    from retrieval.v2_retriever import QdrantV2Retriever
    from inference.scoring_provenance import decode_owned_packet
    from api.contracts import CapturedRecommendationResponse
    initialize(runtime);p=runtime.policy
    apply(runtime,wire(p,'feedback',3,feedback_payload(session=uid(50))))
    for index in range(12):
        repo=uid(100+index);payload=repository_payload_defaults()
        payload.update(repo_id=repo,full_name=f'diagnostic/repo-{index}',content_version=1,content_hash='a'*64,
            source_hash='b'*64,indexed_at='2026-10-24T00:00:00Z',star_count=index*100,primary_language='Python',
            languages=['Python'],topics=['Python'],pushed_days_ago=1,delta_7d=1,activity_score=.5)
        runtime.qdrant.upsert(runtime.settings.repository_collection,points=[models.PointStruct(id=repo,
            vector={'repo_embedding':basis(index+1)},payload=payload)],wait=True)
    retriever=QdrantV2Retriever(client=runtime.qdrant,user_collection=runtime.settings.user_collection,
        repository_collection=runtime.settings.repository_collection,ranker=trained_ranker,heavy_ranker_enabled=True,
        heavy_ranker_required=True,heavy_ranker_traffic_percent=100.,minimum_eligible_repositories=1)
    monkeypatch.setattr(provider,'model_runtime',lambda:runtime);monkeypatch.setattr(v2,'retriever',lambda:retriever)
    for key,value in {'INTERNAL_API_SECRET':'fixture-secret','SCORING_CAPTURE_ENABLED':'true','SCORING_CAPTURE_CAPABILITY':'7'*64,
                      'SCORING_CAPTURE_CODE_COMMIT':'a'*40,'BAKED_ML_RELEASE_ID':'a'*40}.items():monkeypatch.setenv(key,value)
    reads=[];queries=[];matrices=[]
    original_read=runtime.qdrant.retrieve;original_query=runtime.qdrant.query_points
    def read(*args,**kwargs):
        rows=original_read(*args,**kwargs)
        if kwargs.get('collection_name')==runtime.settings.user_collection:reads.append(kwargs)
        return rows
    def query(*args,**kwargs):
        if 'query' in kwargs:queries.append(kwargs['query'])
        return original_query(*args,**kwargs)
    monkeypatch.setattr(runtime.qdrant,'retrieve',read);monkeypatch.setattr(runtime.qdrant,'query_points',query)
    import feedback.model_serving as serving
    original_owned = serving.owned_input
    def own_then_mutate_caller(point, *args, **kwargs):
        effective = original_owned(point, *args, **kwargs)
        point.payload['embedding_model'] = 'mutated-caller-alias'
        point.payload['interests'] = ['mutated-caller-alias']
        return effective
    monkeypatch.setattr(serving, 'owned_input', own_then_mutate_caller)
    hook=trained_ranker.model.register_forward_pre_hook(lambda _model,args:matrices.append(args[0].detach().cpu().numpy().copy()))
    try:
        response=TestClient(main.app).post('/api/v2/recommendations/generate',headers={
            'x-internal-secret':'fixture-secret','x-scoring-capture-capability':'7'*64},json={
            'schema_version':2,'user_id':uid(1),'generation_id':uid(3000),'feed_version':'9007199254740993','limit':5,
            'exclude_repo_ids':[],'social_repo_ids':[],'context':{'cold_start':False},
            **({'scoring_capture':{'schema_version':1,'encoding':'float32-le-v1'}} if capture else {}),'model_context':{
            'schema_version':1,'session_id':uid(50),'epoch':'1','policy_sha256':p.sha256,
            'minimum_applied_operation_version':'3','required_profile_version':'1','as_of_ms':'3'}})
    finally:hook.remove()
    assert response.status_code==200,response.text
    body=response.json();assert len(reads)==1 and reads[0]['with_payload'] is True
    import numpy as np,hashlib
    user_bytes=np.asarray(queries[0],dtype='<f4').tobytes()
    assert matrices and all(row[:384].astype('<f4').tobytes()==user_bytes for matrix in matrices for row in matrix)
    assert body['model_context']['effective_vector_sha256']==hashlib.sha256(user_bytes).hexdigest()
    assert body['feed_version']=='9007199254740993'
    assert body['model_context']['applied_operation_version']=='3'
    if capture:
        assert body['scoring_context']['user_vector_sha256']==hashlib.sha256(user_bytes).hexdigest()
        parsed=CapturedRecommendationResponse.model_validate_json(response.content)
        for item in parsed.items:
            assert decode_owned_packet(item.scored_input,parsed.scoring_context,owner_user_id=uid(1),generation_id=uid(3000),repo_id=str(item.repo_id))[:1536]==user_bytes
    else:
        assert 'scoring_context' not in body and 'scoring_capture' not in body
        from api.contracts import ModelRecommendationResponse
        assert ModelRecommendationResponse.model_validate_json(response.content).feed_version=='9007199254740993'


@pytest.mark.parametrize('runtime',[{'max_repositories':1}],indirect=True)
def test_real_prepared_rejection_is_not_authority_until_head_cas(runtime,monkeypatch):
    initialize(runtime);p=runtime.policy
    apply(runtime,wire(p,'feedback',3,feedback_payload()))
    checkpoint=runtime.processor.coordinator.checkpoint(uid(1));before=runtime.users.read(uid(1))
    point=runtime.qdrant.retrieve(runtime.settings.repository_collection,ids=[uid(100)],with_payload=True,with_vectors=True)[0]
    runtime.qdrant.upsert(runtime.settings.repository_collection,points=[models.PointStruct(id=uid(101),vector=point.vector,
        payload={**point.payload,'repo_id':uid(101)})],wait=True)
    from feedback.model_wire import canonical_json
    payload=feedback_payload(2);event=json.loads(payload['original_feedback_json']);event['repo_id']=uid(101)
    payload['original_feedback_json']=canonical_json(event);raw=wire(p,'feedback',4,payload)
    original=runtime.users.compare_and_set
    monkeypatch.setattr(runtime.users,'compare_and_set',lambda *_:(_ for _ in ()).throw(RuntimeError('crash after DLQ before CAS')))
    with pytest.raises(RuntimeError):runtime.processor(parse(raw,p),lambda:None)
    rows=runtime.redis.xrange(runtime.stream.limits.namespace+':dlq')
    assert len(rows)==1
    prepare=rows[0][1]['prepare_sha256']
    assert runtime.processor.coordinator.rejection_status(prepare)=='prepared_not_committed'
    assert runtime.users.read(uid(1))==before
    monkeypatch.setattr(runtime.users,'compare_and_set',original)
    assert runtime.processor(parse(raw,p),lambda:None)=='rejected'
    assert runtime.processor.coordinator.rejection_status(prepare)=='committed_rejection'
    after=runtime.users.read(uid(1));assert after.state.feedback_version==2 and after.state.long_entries==before.state.long_entries
    assert runtime.redis.xlen(runtime.stream.limits.namespace+':dlq')==1
    assert runtime.processor.coordinator.rebuild(checkpoint,after.head,expected_user_id=uid(1))==after


def test_rebuild_dryrun_staged_insert_only_and_stale_head_refusal(runtime):
    from scripts.rebuild_user_model import verify_and_stage
    initialize(runtime);p=runtime.policy
    checkpoint=runtime.processor.coordinator.checkpoint(uid(1));old_head=runtime.users.read(uid(1)).head
    apply(runtime,wire(p,'feedback',3,feedback_payload()))
    head=runtime.users.read(uid(1)).head
    target='fixture_model_restore_target';runtime.qdrant.create_collection(target,vectors_config=models.VectorParams(size=384,distance=models.Distance.COSINE))
    assert verify_and_stage(runtime,user_id=uid(1),checkpoint_sha256=checkpoint,expected_head=head)['status']=='verified_dry_run'
    with pytest.raises(JournalConflict):verify_and_stage(runtime,user_id=uid(1),checkpoint_sha256=checkpoint,expected_head=old_head,target_collection=target,stage_copy=True)
    assert runtime.qdrant.retrieve(target,ids=[uid(1)])==[]
    result=verify_and_stage(runtime,user_id=uid(1),checkpoint_sha256=checkpoint,expected_head=head,target_collection=target,stage_copy=True)
    assert result['status']=='verified_staged_copy' and result['serving_cutover'] is False
    with pytest.raises(ValueError):verify_and_stage(runtime,user_id=uid(1),checkpoint_sha256=checkpoint,expected_head=head,target_collection=runtime.settings.user_collection,stage_copy=True)


@pytest.mark.parametrize('stage',['before_prepare','before_cas','after_cas_before_ack'])
def test_abrupt_worker_process_death_keeps_pending_and_reuses_frozen_sources(runtime,tmp_path,stage):
    from dataclasses import asdict
    from pathlib import Path
    import sys
    initialize(runtime);p=runtime.policy;before=runtime.users.read(uid(1))
    raw=wire(p,'feedback',3,feedback_payload());runtime.stream.admit(raw)
    target=tmp_path/'owned-settings.json';target.write_text(json.dumps({'config':runtime_config(p),'settings':asdict(runtime.settings)}))
    result=subprocess.run([sys.executable,'-m','tests.fixtures.model_operation_worker',str(target),stage],
        cwd=Path(__file__).parents[1],capture_output=True,text=True,timeout=20)
    assert result.returncode==71,result.stderr
    observed=runtime.users.read(uid(1))
    if stage=='after_cas_before_ack':assert observed.state.feedback_version==1
    else:assert observed==before
    assert runtime.redis.xpending(runtime.stream.stream,runtime.stream.group)['pending']==1
    # Change the source after the dead worker pinned it. A retry must use the
    # old source, not this current vector or revision.
    point=runtime.qdrant.retrieve(runtime.settings.repository_collection,ids=[uid(100)],with_vectors=True,with_payload=True)[0]
    runtime.qdrant.upsert(runtime.settings.repository_collection,points=[models.PointStruct(id=uid(100),
        vector={'repo_embedding':basis(2)},payload={**point.payload,'content_version':2,'feature_version':2})],wait=True)
    time.sleep(1.05)  # Natural death of the owned worker's renewable one-second lock.
    assert runtime.consumer().run_once()==1
    final=runtime.users.read(uid(1))
    assert final.state.feedback_version==1 and final.state.operation_version==3
    source=final.state.long_entries[0].signals[0].source
    assert source.content_version==1 and source.feature_version==1 and source.vector==tuple(basis(1))
    assert runtime.redis.xlen(runtime.stream.stream)==0


@pytest.mark.parametrize('runtime',[{'max_sessions':1}],indirect=True)
def test_full_dlq_keeps_next_control_pending_and_rejected_begin_has_no_authority(runtime,monkeypatch):
    from dataclasses import replace
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.v2 import router
    import feedback.model_runtime as provider
    initialize(runtime);p=runtime.policy
    runtime.stream.limits=replace(runtime.stream.limits,max_dead_letters=1)
    assert runtime.processor(parse(wire(p,'begin',3,begin_payload(3,uid(51))),p),lambda:None)=='rejected'
    before=runtime.users.read(uid(1));assert before.state.operation_version==3 and before.state.feedback_version==0
    with pytest.raises(OperationCapacity):runtime.processor(parse(wire(p,'begin',4,begin_payload(4,uid(52))),p),lambda:None)
    assert runtime.users.read(uid(1))==before
    monkeypatch.setattr(provider,'model_runtime',lambda:runtime);monkeypatch.setenv('INTERNAL_API_SECRET','fixture-secret')
    app=FastAPI();app.include_router(router)
    response=TestClient(app).post('/api/v2/model-authority',headers={'x-internal-secret':'fixture-secret'},json={
        'schema_version':1,'user_id':uid(1),'model_context':{'schema_version':1,'session_id':uid(51),'epoch':'1',
        'policy_sha256':p.sha256,'minimum_applied_operation_version':'3','required_profile_version':'1','as_of_ms':'3'}})
    assert response.status_code==503,response.text


def test_late_cas_loser_cannot_overwrite_newer_complete_state_and_retries_as_duplicate(runtime,monkeypatch):
    import threading
    initialize(runtime);p=runtime.policy
    original_raw=wire(p,'feedback',3,feedback_payload());operation=parse(original_raw,p)
    runtime.journal.pin_input(operation.header,operation.kind,p,original_raw.encode(),lambda:runtime.processor._resolve(operation))
    reached=threading.Event();release=threading.Event();errors=[]
    original_cas=runtime.users.compare_and_set
    def cas(before,after):
        if threading.current_thread().name=='late-worker':
            reached.set();assert release.wait(10)
        return original_cas(before,after)
    monkeypatch.setattr(runtime.users,'compare_and_set',cas)
    def lose():
        try:runtime.processor(operation,lambda:None)
        except Exception as exc:errors.append(exc)
    thread=threading.Thread(target=lose,name='late-worker');thread.start()
    assert reached.wait(10)
    runtime.processor(operation,lambda:None)
    runtime.processor(parse(wire(p,'feedback',4,feedback_payload(2,kind='unsave')),p),lambda:None)
    winner=runtime.users.read(uid(1));release.set();thread.join(10);assert not thread.is_alive()
    assert len(errors)==1 and isinstance(errors[0],CasConflict)
    assert runtime.users.read(uid(1))==winner
    assert runtime.processor(operation,lambda:None)=='duplicate'


def test_old_direct_feedback_and_profile_writers_refuse_active_new_model(runtime):
    from embedding.user_profile_store import QdrantUserProfileStore
    from feedback.v2 import FeedbackStateError
    initialize(runtime);before=runtime.users.read(uid(1))
    with pytest.raises(FeedbackStateError):runtime.legacy.apply(json.loads(feedback_payload()['original_feedback_json']))
    # Use the real existing store write method against the actual owned point.
    store=QdrantUserProfileStore(client=runtime.qdrant)
    store.contract=replace(store.contract,collection_name=runtime.settings.user_collection)
    with pytest.raises(ValueError):store.upsert_user(uid(1),basis(2),payload={'profile_version':2,'job_id':uid(99)})
    assert runtime.users.read(uid(1))==before


def test_one_user_gap_does_not_starve_another_owner_or_later_predecessor(runtime):
    from feedback.model_wire import canonical_json
    initialize(runtime);p=runtime.policy
    # Missing common operation3 for owner1 remains pending while another owner
    # establishes its own independent profile and session using common1/2.
    runtime.stream.admit(wire(p,'feedback',4,feedback_payload()))
    consumer=runtime.consumer()
    with pytest.raises(model.OperationGap):consumer.run_once()
    for version,kind,payload in [(1,'profile',profile_payload()),(2,'begin',begin_payload(2,uid(52)))]:
        value=json.loads(wire(p,kind,version,payload,user=uid(2)))
        value['operation_id']=uid(21000+version)
        runtime.stream.admit(canonical_json(value))
    for _ in range(8):
        try:consumer.run_once()
        except model.OperationGap:pass
    other=runtime.users.read(uid(2));assert other.state.operation_version==2 and other.state.sessions[0].session_id==uid(52)
    assert runtime.users.read(uid(1)).state.operation_version==2
    runtime.stream.admit(wire(p,'feedback',3,feedback_payload(kind='unsave')))
    for _ in range(8):
        try:consumer.run_once()
        except model.OperationGap:pass
        except Exception as exc:
            # Operation4 deliberately has the same feedback version1 and must
            # remain pending after operation3 consumed it, never be skipped.
            assert isinstance(exc,model.ModelError)
    assert runtime.users.read(uid(1)).state.operation_version==3
    assert runtime.users.read(uid(2))==other


def test_applied_authority_is_exact_and_pending_profile_future_epoch_refused(runtime,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.v2 import router
    import feedback.model_runtime as provider
    initialize(runtime);p=runtime.policy
    monkeypatch.setattr(provider,'model_runtime',lambda:runtime);monkeypatch.setenv('INTERNAL_API_SECRET','fixture-secret')
    app=FastAPI();app.include_router(router);client=TestClient(app)
    context={'schema_version':1,'session_id':uid(50),'epoch':'1','policy_sha256':p.sha256,
             'minimum_applied_operation_version':'2','required_profile_version':'1','as_of_ms':'2'}
    def post(value):return client.post('/api/v2/model-authority',headers={'x-internal-secret':'fixture-secret'},
        json={'schema_version':1,'user_id':uid(1),'model_context':value})
    success=post(context);assert success.status_code==200,success.text
    body=success.json()['model_context'];assert body['applied_operation_version']=='2' and body['applied_feedback_version']=='0'
    for field,value in [('minimum_applied_operation_version','3'),('required_profile_version','2'),('epoch','2'),('as_of_ms','1')]:
        assert post({**context,field:value}).status_code==503


from tests.test_model_operations import withdrawal
from feedback.model_wire import parse_withdrawal


def parse_withdraw(raw,p):
    return parse_withdrawal(raw,maximum_bytes=131072,allowed_policies={p.sha256})


def test_withdraw_before_first_profile_keeps_unservable_prefix_then_applies_pending_like(runtime):
    p=runtime.policy;calls=[];runtime.processor.profile_encoder=lambda context:(calls.append(context) or basis(0))
    original=wire(p,'profile',1,profile_payload())
    attestation=withdrawal(p,original,revoked_at=2)
    runtime.stream.admit_withdrawal(attestation);assert runtime.consumer().run_once()==1
    prefix=runtime.users.read(uid(1));assert isinstance(prefix.state,model.BootstrapState)
    point=runtime.qdrant.retrieve(runtime.settings.user_collection,ids=[uid(1)],with_vectors=True,with_payload=True)[0]
    assert point.vector=={} and point.payload['model_basis_status']=='awaiting_profile'
    assert not calls
    with pytest.raises(model.CompatibilityError):model.recommendation_view(prefix.state,p,session_id=None,epoch=None,as_of_ms=2)
    assert runtime.stream.admit(original)['status']=='withdrawal_accepted'
    assert runtime.redis.xlen(runtime.stream.stream)==0 and not calls
    # Product feedback was frozen durably by the backend but not numbered until
    # the real profile basis became ready. Its original opaque JSON is retained.
    pending_feedback=feedback_payload()
    restored=wire(p,'profile',2,profile_payload(True,2),at=3)
    apply(runtime,restored);ready=runtime.users.read(uid(1));assert not isinstance(ready.state,model.BootstrapState)
    assert ready.state.baseline.content_version==2 and ready.state.feedback_version==0 and len(calls)==1
    feedback_raw=wire(p,'feedback',3,pending_feedback,at=1)
    apply(runtime,feedback_raw)
    assert runtime.users.read(uid(1)).state.feedback_version==1
    assert json.loads(feedback_raw)['operation']['original_feedback_json']==pending_feedback['original_feedback_json']
    assert runtime.processor.coordinator.rebuild(runtime.journal.get(prefix.head,'prepare')['genesis_checkpoint_sha256'],
        runtime.users.read(uid(1)).head,expected_user_id=uid(1))==runtime.users.read(uid(1))


def test_allow_genesis_corrective_profile_updates_existing_basis_and_preserves_feedback(runtime):
    initialize(runtime);p=runtime.policy
    apply(runtime,wire(p,'feedback',3,feedback_payload()))
    before=runtime.users.read(uid(1))
    apply(runtime,wire(p,'profile',4,profile_payload(True,2)))
    after=runtime.users.read(uid(1))
    assert after.state.long_entries==before.state.long_entries and after.state.sessions==before.state.sessions
    assert after.state.feedback_version==1 and after.state.baseline.content_version==2
    assert after.state.operation_version==4


def test_repeated_withdrawals_late_original_and_exact_rebuild_without_encoder(runtime):
    p=runtime.policy;calls=[];runtime.processor.profile_encoder=lambda context:(calls.append(context) or basis(0))
    originals=[wire(p,'profile',n,profile_payload(True,n)) for n in (1,2)]
    for n,raw in enumerate(originals,1):
        attestation=withdrawal(p,raw,revoked_at=n)
        runtime.stream.admit_withdrawal(attestation);assert runtime.consumer().run_once()==1
        assert runtime.processor(parse_withdraw(attestation,p),lambda:None)=='withdrawn_duplicate'
    assert not calls
    checkpoint=runtime.processor.coordinator.checkpoint(uid(1))
    for raw in originals:
        assert runtime.stream.admit(raw)['status']=='withdrawal_accepted'
        assert runtime.journal.records.get('operation:'+uid(1)+':'+json.loads(raw)['model_operation_version']) is None
    apply(runtime,wire(p,'profile',3,profile_payload(True,3)))
    final=runtime.users.read(uid(1));assert len(calls)==1 and final.state.baseline.content_version==3
    assert runtime.processor.coordinator.rebuild(checkpoint,final.head,expected_user_id=uid(1))==final
    assert runtime.stream.admit(originals[0])['status']=='withdrawal_accepted'
    assert len(calls)==1 and runtime.redis.xlen(runtime.stream.stream)==0


def test_initialized_guard_survives_point_loss_policy_and_transport_namespace_change(runtime):
    initialize(runtime);original=wire(runtime.policy,'profile',1,profile_payload())
    assert runtime.redis.hget(runtime.stream.guard_key,uid(1))=='initialized'
    runtime.qdrant.delete(runtime.settings.user_collection,points_selector=models.PointIdsList(points=[uid(1)]),wait=True)
    with pytest.raises(OperationBytesConflict):runtime.stream.admit(original)
    changed=policy(generation_freshness_ms=101);config=runtime_config(changed,namespace='another-transport-policy')
    other=ModelRuntime(config,redis=runtime.redis,qdrant=runtime.qdrant,settings=replace(runtime.settings,qdrant_url='http://127.0.0.1:9'),encoder=lambda _:basis(1))
    assert other.stream.guard_key==runtime.stream.guard_key
    with pytest.raises(OperationBytesConflict):other.stream.admit(wire(changed,'profile',1,profile_payload()))
    assert runtime.qdrant.retrieve(runtime.settings.user_collection,ids=[uid(1)])==[]


def test_registered_withdrawal_and_status_report_only_committed_proof(runtime,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.v2 import router
    import feedback.model_runtime as provider
    p=runtime.policy;original=wire(p,'profile',1,profile_payload());attestation=withdrawal(p,original,revoked_at=2)
    parsed=parse(original,p)
    monkeypatch.setattr(provider,'model_runtime',lambda:runtime);monkeypatch.setenv('INTERNAL_API_SECRET','fixture-secret')
    app=FastAPI();app.include_router(router);client=TestClient(app);headers={'x-internal-secret':'fixture-secret'}
    request={'schema_version':1,'user_id':uid(1),'operation_id':parsed.header.operation_id,
        'model_operation_version':'1','source_envelope_sha256':parsed.header.source_envelope_sha256}
    def status():return client.post('/api/v2/model-operation-status',headers=headers,json=request)
    assert status().json()['basis_status']=='missing'
    ack=client.post('/api/v2/model-operation-withdrawals',headers=headers,json={'schema_version':1,'withdrawals':[{'withdrawal_json':attestation}]})
    assert ack.status_code==200,ack.text
    assert ack.json()['withdrawals'][0]['status']=='accepted'
    assert status().json()['status']=='pending'
    assert runtime.consumer().run_once()==1
    committed=status();assert committed.status_code==200,committed.text
    value=committed.json();assert value['status']=='withdrawn_profile' and value['basis_status']=='awaiting_profile'
    assert value['operation_head']==value['current_head'] and 'applied_profile_version' not in value
    late=client.post('/api/v2/model-operations',headers=headers,json={'schema_version':1,'operations':[{'operation_json':original}]})
    assert late.status_code==200,late.text
    assert late.json()['operations'][0]['status']=='withdrawal_accepted'
    assert runtime.redis.xlen(runtime.stream.stream)==0
    wrong=client.post('/api/v2/model-operation-status',headers=headers,json={**request,'source_envelope_sha256':'f'*64})
    assert wrong.status_code==409
    assert client.post('/api/v2/model-operation-withdrawals',json={'schema_version':1,'withdrawals':[{'withdrawal_json':attestation}]}).status_code==401


def test_full_persisted_point_capacity_rejects_once_and_unblocks_next_feedback(runtime):
    from feedback.model_journal import Snapshot,_canonical,_pack
    initialize(runtime);p=runtime.policy;before=runtime.users.read(uid(1))
    raw=wire(p,'feedback',3,feedback_payload());incoming=parse(raw,p);operation=runtime.processor._resolve(incoming)
    after_state=model.reduce_operation(before.state,operation,p)
    legacy=json.loads(runtime.processor._legacy_prepare(incoming,before.complete,operation))
    complete=_canonical({'schema_version':1,'user_id':uid(1),'state':_pack(after_state),'legacy':legacy})
    probe=Snapshot(uid(1),after_state,'0'*64,complete)
    rejected=model.reject_capacity(before.state,operation,p)
    reject_complete=_canonical({'schema_version':1,'user_id':uid(1),'state':_pack(rejected),
        'legacy':json.loads(runtime.processor._legacy_reject(before.complete,operation))})
    def size(snapshot):
        vector,payload=runtime.users._point(snapshot)
        return len(_canonical({'vector':vector,'payload':payload}))
    prior_bytes,applied_bytes,reject_bytes=size(before),size(probe),size(Snapshot(uid(1),rejected,'0'*64,reject_complete))
    tight=max(prior_bytes,reject_bytes,len(complete))+128
    assert tight<applied_bytes and len(model.state_bytes(after_state))<p.max_state_bytes
    print({'prior_point_bytes':prior_bytes,'applied_complete_bytes':len(complete),'applied_point_bytes':applied_bytes,
           'rejected_point_bytes':reject_bytes,'configured_combined_limit':tight})
    runtime.users.limit=tight
    runtime.journal.limits=replace(runtime.journal.limits,combined_user_bytes=tight)
    runtime.stream.admit(raw)
    assert runtime.consumer().run_once()==1
    actual=runtime.users.read(uid(1))
    assert actual.state.feedback_version==1 and actual.state.long_entries==before.state.long_entries
    row=runtime.journal.get(actual.head,'prepare');assert row['disposition']=='rejected'
    assert runtime.processor.coordinator.rejection_status(actual.head)=='committed_rejection'
    assert runtime.redis.xlen(runtime.stream.limits.namespace+':dlq')==1
    assert size(actual)<=tight
    # A duplicate never mutates again, and the next valid small inverse moves
    # the lane forward instead of remaining behind the rejected large event.
    assert runtime.processor(incoming,lambda:None)=='rejected_duplicate'
    apply(runtime,wire(p,'feedback',4,feedback_payload(2,kind='unsave')))
    assert runtime.users.read(uid(1)).state.feedback_version==2


def test_rejection_that_cannot_fit_stays_pending_without_dlq_cas_or_ack(runtime):
    from feedback.model_journal import _canonical
    initialize(runtime);before=runtime.users.read(uid(1))
    vector,payload=runtime.users._point(before);tight=len(_canonical({'vector':vector,'payload':payload}))
    runtime.users.limit=tight;runtime.journal.limits=replace(runtime.journal.limits,combined_user_bytes=tight)
    raw=wire(runtime.policy,'feedback',3,feedback_payload());runtime.stream.admit(raw)
    with pytest.raises(model.StateOverflow):runtime.consumer().run_once()
    assert runtime.users.read(uid(1))==before
    assert runtime.redis.xlen(runtime.stream.stream)==1
    assert runtime.redis.xpending(runtime.stream.stream,runtime.stream.group)['pending']==1
    assert runtime.redis.xlen(runtime.stream.limits.namespace+':dlq')==0


def test_known_legacy_capacity_is_terminal_but_corruption_is_not_misclassified(runtime):
    initialize(runtime);before=runtime.users.read(uid(1));legacy=json.loads(before.complete)['legacy']
    runtime.legacy.settings=replace(runtime.legacy.settings,max_user_state_bytes=runtime.legacy._feedback_state_bytes(legacy['payload'])+1)
    apply(runtime,wire(runtime.policy,'feedback',3,feedback_payload()))
    after=runtime.users.read(uid(1));assert after.state.feedback_version==1 and not after.state.long_entries
    assert runtime.journal.get(after.head,'prepare')['disposition']=='rejected'
    # A corrupt persisted complete-point relationship is an unavailable
    # dependency, not another capacity rejection or an implicit cursor skip.
    runtime.qdrant.set_payload(runtime.settings.user_collection,payload={'feedback_latent_vector':['corrupt']},points=[uid(1)],wait=True)
    runtime.stream.admit(wire(runtime.policy,'feedback',4,feedback_payload(2,kind='unsave')))
    with pytest.raises(JournalConflict):runtime.consumer().run_once()
    assert runtime.redis.xlen(runtime.stream.limits.namespace+':dlq')==1
    raw=runtime.qdrant.retrieve(runtime.settings.user_collection,ids=[uid(1)],with_payload=True,with_vectors=False)[0]
    assert raw.payload['model_operation_version']==3


def test_guard_must_be_durable_before_ready_read_ack_and_ordinary_serving(runtime,monkeypatch):
    from fastapi.testclient import TestClient
    from api import main,v2
    import feedback.model_runtime as provider
    from retrieval.v2_retriever import QdrantV2Retriever,RetrievalDependencyError
    p=runtime.policy;raw=wire(p,'profile',1,profile_payload());incoming=parse(raw,p)
    good=runtime.users.ensure_initialized
    monkeypatch.setattr(runtime.users,'ensure_initialized',lambda _:(_ for _ in ()).throw(OperationCapacity('guard unavailable')))
    runtime.stream.admit(raw)
    with pytest.raises(OperationCapacity):runtime.consumer().run_once()
    point=runtime.qdrant.retrieve(runtime.settings.user_collection,ids=[uid(1)],with_payload=True,with_vectors=True)[0]
    assert point.vector==basis(0) and runtime.redis.hget(runtime.stream.guard_key,uid(1))=='pending'
    with pytest.raises(RetrievalDependencyError):QdrantV2Retriever._profile_from_points(None,uid(1),[point])
    monkeypatch.setattr(provider,'model_runtime',lambda:runtime);monkeypatch.setenv('INTERNAL_API_SECRET','fixture-secret')
    request={'schema_version':1,'user_id':uid(1),'operation_id':incoming.header.operation_id,'model_operation_version':'1',
        'source_envelope_sha256':incoming.header.source_envelope_sha256}
    response=TestClient(main.app).post('/api/v2/model-operation-status',headers={'x-internal-secret':'fixture-secret'},json=request)
    assert response.status_code==503
    assert runtime.redis.xpending(runtime.stream.stream,runtime.stream.group)['pending']==1
    monkeypatch.setattr(runtime.users,'ensure_initialized',good)
    time.sleep(.02);assert runtime.consumer().run_once()==1
    assert runtime.redis.hget(runtime.stream.guard_key,uid(1))=='initialized'
    response=TestClient(main.app).post('/api/v2/model-operation-status',headers={'x-internal-secret':'fixture-secret'},json=request)
    assert response.status_code==200 and response.json()['basis_status']=='ready'


def test_inflight_original_observes_withdrawal_before_genesis_cas_and_never_serves_it(runtime):
    import threading
    p=runtime.policy;raw=wire(p,'profile',1,profile_payload());entered=threading.Event();release=threading.Event();errors=[];calls=[]
    def encode(context):
        calls.append(context);entered.set();assert release.wait(10);return basis(0)
    runtime.processor.profile_encoder=encode
    runtime.stream.admit(raw);consumer=runtime.consumer()
    def original():
        try:consumer.run_once()
        except Exception as exc:errors.append(exc)
    thread=threading.Thread(target=original);thread.start();assert entered.wait(10)
    runtime.stream.admit_withdrawal(withdrawal(p,raw,revoked_at=2));release.set();thread.join(10);assert not thread.is_alive()
    assert len(errors)==1 and isinstance(errors[0],JournalConflict)
    assert runtime.users.read_optional(uid(1)) is None
    assert runtime.consumer().run_once()==1
    assert isinstance(runtime.users.read(uid(1)).state,model.BootstrapState)
    time.sleep(.02);assert runtime.consumer().run_once()==1
    assert runtime.redis.xlen(runtime.stream.stream)==0 and len(calls)==1
    assert runtime.stream.admit(raw)['status']=='withdrawal_accepted' and len(calls)==1


def test_already_applied_original_is_not_rewritten_and_corrective_profile_preserves_learning(runtime):
    initialize(runtime);p=runtime.policy
    apply(runtime,wire(p,'feedback',3,feedback_payload()))
    before=runtime.users.read(uid(1));raw=wire(p,'profile',1,profile_payload())
    attestation=withdrawal(p,raw,revoked_at=4)
    runtime.stream.admit_withdrawal(attestation);assert runtime.consumer().run_once()==1
    assert runtime.users.read(uid(1))==before
    assert runtime.processor(parse_withdraw(attestation,p),lambda:None)=='already_applied'
    proof=runtime.processor.coordinator.original_status(before,parse(raw,p).header)
    assert proof['status']=='applied'
    runtime.processor.profile_encoder=lambda _:basis(2)
    apply(runtime,wire(p,'profile',4,profile_payload(True,2),at=4))
    after=runtime.users.read(uid(1))
    assert after.state.baseline.vector==tuple(basis(2)) and after.state.long_entries==before.state.long_entries
    assert after.state.feedback_version==1 and after.state.sessions==before.state.sessions


def test_withdrawal_dlq_prepare_orphan_and_lost_cas_retry_have_distinct_authority(runtime,monkeypatch):
    initialize(runtime);p=runtime.policy;raw=wire(p,'profile',3,profile_payload(True,2));attestation=withdrawal(p,raw,revoked_at=4)
    runtime.stream.admit_withdrawal(attestation);before=runtime.users.read(uid(1));original=runtime.users.compare_and_set
    monkeypatch.setattr(runtime.users,'compare_and_set',lambda *_:(_ for _ in ()).throw(RuntimeError('after withdrawal prepare before CAS')))
    with pytest.raises(RuntimeError):runtime.consumer().run_once()
    prepare=runtime.redis.xrange(runtime.stream.limits.namespace+':dlq')[0][1]['prepare_sha256']
    assert runtime.processor.coordinator.rejection_status(prepare)=='prepared_not_committed'
    assert runtime.users.read(uid(1))==before
    assert runtime.stream.admit(raw)['status']=='withdrawal_accepted'
    monkeypatch.setattr(runtime.users,'compare_and_set',original)
    time.sleep(.02);assert runtime.consumer().run_once()==1
    assert runtime.processor.coordinator.rejection_status(prepare)=='committed_withdrawal'
    assert runtime.users.read(uid(1)).state.baseline.content_version==1
    assert runtime.users.read(uid(1)).state.operation_version==3
    assert runtime.redis.xlen(runtime.stream.limits.namespace+':dlq')==1


@pytest.mark.parametrize('stage',['before_prepare','before_cas','after_cas_before_ack'])
def test_withdrawal_prefix_recovers_after_abrupt_process_death_without_fake_basis(runtime,tmp_path,stage):
    from dataclasses import asdict
    from pathlib import Path
    import sys
    p=runtime.policy;raw=wire(p,'profile',1,profile_payload());runtime.stream.admit_withdrawal(withdrawal(p,raw,revoked_at=2))
    target=tmp_path/'owned-settings.json';target.write_text(json.dumps({'config':runtime_config(p),'settings':asdict(runtime.settings)}))
    result=subprocess.run([sys.executable,'-m','tests.fixtures.model_operation_worker',str(target),stage],
        cwd=Path(__file__).parents[1],capture_output=True,text=True,timeout=20)
    assert result.returncode==71,result.stderr
    observed=runtime.users.read_optional(uid(1))
    if stage=='after_cas_before_ack':assert isinstance(observed.state,model.BootstrapState)
    else:assert observed is None
    assert runtime.redis.hget(runtime.stream.guard_key,uid(1))=='pending'
    assert runtime.stream.admit(raw)['status']=='withdrawal_accepted'
    assert runtime.journal.records.get('operation:'+uid(1)+':1') is None
    time.sleep(1.05)
    assert runtime.consumer().run_once()==1
    final=runtime.users.read(uid(1));assert isinstance(final.state,model.BootstrapState)
    assert final.state.operation_version==1 and final.state.feedback_version==0
    assert runtime.redis.xlen(runtime.stream.stream)==0


def test_genuine_genesis_process_death_after_cas_has_guard_before_any_ack_and_denies_lost_state(runtime,tmp_path):
    from dataclasses import asdict
    from pathlib import Path
    import sys
    p=runtime.policy;raw=wire(p,'profile',1,profile_payload());runtime.stream.admit(raw)
    target=tmp_path/'owned-settings.json';target.write_text(json.dumps({'config':runtime_config(p),'settings':asdict(runtime.settings)}))
    result=subprocess.run([sys.executable,'-m','tests.fixtures.model_operation_worker',str(target),'after_cas_before_ack'],
        cwd=Path(__file__).parents[1],capture_output=True,text=True,timeout=20)
    assert result.returncode==71,result.stderr
    assert runtime.users.read(uid(1)).state.baseline.content_version==1
    assert runtime.redis.hget(runtime.stream.guard_key,uid(1))=='initialized'
    assert runtime.redis.xpending(runtime.stream.stream,runtime.stream.group)['pending']==1
    runtime.qdrant.delete(runtime.settings.user_collection,points_selector=models.PointIdsList(points=[uid(1)]),wait=True)
    time.sleep(1.05)
    with pytest.raises(OperationBytesConflict):runtime.consumer().run_once()
    assert runtime.qdrant.retrieve(runtime.settings.user_collection,ids=[uid(1)])==[]
    assert runtime.redis.xpending(runtime.stream.stream,runtime.stream.group)['pending']==1


def test_dense_qdrant_float_mirrors_keep_canonical_history_exact_and_refuse_corruption(runtime):
    import math,base64
    from feedback.model_journal import _json,_unpack
    values=[math.sin(index+1) for index in range(384)];length=math.sqrt(sum(x*x for x in values));dense=[x/length for x in values]
    runtime.processor.profile_encoder=lambda _:dense
    apply(runtime,wire(runtime.policy,'profile',1,profile_payload()))
    snapshot=runtime.users.read(uid(1))
    assert snapshot.state.baseline.vector==tuple(dense)
    point=runtime.qdrant.retrieve(runtime.settings.user_collection,ids=[uid(1)],with_payload=True,with_vectors=True)[0]
    assert base64.b64decode(point.payload['model_complete_base64'])==snapshot.complete
    frozen=_json(snapshot.complete,runtime.users.limit)
    expected=frozen['legacy']['payload']['preference_accumulator'];actual=point.payload['preference_accumulator']
    assert expected==dense
    assert any(a!=b for a,b in zip(actual,expected))  # Real server JSON roundtrip, not a fabricated tolerance test.
    assert all(a in {b,math.nextafter(b,math.inf),math.nextafter(b,-math.inf)} for a,b in zip(actual,expected))
    from copy import deepcopy
    two_ulps=deepcopy(point)
    two_ulps.payload['preference_accumulator'][0]=math.nextafter(math.nextafter(expected[0],math.inf),math.inf)
    with pytest.raises(JournalConflict):runtime.users.decode_point(two_ulps,uid(1))
    for key,value,error in [('profile_version',float(point.payload['profile_version']),JournalConflict),
                            ('model_operation_version',True,JournalConflict),('model_journal_head','f'*64,JournalMissing)]:
        altered_point=deepcopy(point);altered_point.payload[key]=value
        with pytest.raises(error):runtime.users.decode_point(altered_point,uid(1))
    altered=list(actual);altered[0]+=0.01
    runtime.qdrant.set_payload(runtime.settings.user_collection,payload={'preference_accumulator':altered},points=[uid(1)],wait=True)
    with pytest.raises(JournalConflict):runtime.users.read(uid(1))
