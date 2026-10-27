from dataclasses import asdict, replace
import hashlib
from types import SimpleNamespace
import uuid
import pytest
from feedback import long_session as model
from feedback.model_wire import WireError, canonical_json, parse_operation
from feedback.model_journal import Journal, Limits, QdrantAtomicUsers, JournalConflict
from feedback.model_processor import ModelProcessor
from feedback.model_stream import StreamLimits, OperationStream
from feedback.v2 import OrderedFeedbackApplier
from feedback.v2_settings import V2FeedbackSettings
from embedding.qdrant_cas import payload_snapshot_filter
from config import EMBEDDING_MODEL_REVISION, REPOSITORY_EMBEDDING_MODEL, REPOSITORY_EMBEDDING_VERSION
from copy import deepcopy
from dataclasses import asdict
import json
from datetime import datetime
from pathlib import Path
from evaluation.model_history import FrozenModelHistory, derive_snapshots, compare_model_history, validate_model_comparison
from feedback.model_wire import canonical_json
import asyncio
import threading
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api import v2, metrics


# Model operations

def uid(n):
    return str(uuid.UUID(int=n))

def basis(n):
    return [float(i == n) for i in range(384)]


def serving_point_fixture(repositories=16):
    from feedback.model_journal import Snapshot, _canonical, _pack
    p=policy();owner=uid(1);session=uid(50)
    baseline=model.PinnedVector.pin('profile',owner,1,0,p.baseline_recipe,p.embedding,basis(0))
    state=model.initial_state(owner,baseline,p,accepted_ms=0)
    def header(version):return model.Header(uid(1000+version),owner,version,version-1,version,'a'*64)
    state=model.reduce_operation(state,model.Begin(header(1),session,1,'b'*64),p)
    for index in range(repositories):
        repo=uid(100+index);source=model.PinnedVector.pin('repository',repo,1,0,p.repository_recipe,p.embedding,basis(index+1))
        binding=model.SessionBinding(owner,session,1,1,True,'b'*64)
        state=model.reduce_operation(state,model.Feedback(header(index+2),uid(2000+index),index+1,repo,
            'dislike' if index%4==0 else 'like',source,binding,None,None),p)
    class Records:
        def __init__(self):self.data={}
        def get(self,key):return self.data.get(key)
        def put_once(self,key,raw):return self.data.setdefault(key,raw)
    journal=Journal(Records(),Limits(131072,2097152,1048576,1000))
    users=QdrantAtomicUsers(None,None,None,collection='owned-serving-fixture',combined_user_bytes=1048576,policy=p,journal=journal)
    complete=_canonical({'schema_version':1,'user_id':owner,'state':_pack(state),'legacy':{'vector':basis(0),'payload':{
        'user_id':owner,'profile_version':1,'last_feedback_version':repositories,'feedback_state_revision':repositories+1,'interests':['Rust']}}})
    complete_sha=journal.put('complete',json.loads(complete))
    head=journal.put('prepare',{'user_id':owner,'operation_version':state.operation_version,'state_sha256':model.state_sha256(state),'complete_sha256':complete_sha})
    snapshot=Snapshot(owner,state,head,complete);vector,payload=users._point(snapshot)
    return SimpleNamespace(id=owner,vector=vector,payload=payload),users,p,snapshot


def test_owned_serving_reuses_validation_without_changing_float32_bytes_or_clocks(monkeypatch):
    import struct
    import feedback.model_journal as journal_module
    from feedback.model_serving import Context,owned_input
    point,users,p,snapshot=serving_point_fixture()
    for offset in (0,5000):
        clock=snapshot.state.effective_ms+offset
        expected=model.recommendation_view(snapshot.state,p,session_id=uid(50),epoch=1,as_of_ms=clock)
        calls={'state_checks':0,'state_hashes':0,'json_validations':0}
        def counted(name,function):
            def run(*args,**kwargs):calls[name]+=1;return function(*args,**kwargs)
            return run
        with monkeypatch.context() as patch:
            patch.setattr(model,'_check_state',counted('state_checks',model._check_state))
            patch.setattr(model,'state_sha256',counted('state_hashes',model.state_sha256))
            patch.setattr(journal_module,'_json',counted('json_validations',journal_module._json))
            effective=owned_input(point,users,p,Context(uid(1),uid(50),1,p.sha256,snapshot.state.operation_version,1,clock))
        assert effective.float32_le==struct.pack('<384f',*expected.blended)
        assert effective.explicit_negative_repo_ids==expected.explicit_negative_repo_ids
        assert effective.state_sha256==model.state_sha256(snapshot.state)
        assert effective.valid_until_ms==min(clock+p.generation_freshness_ms,snapshot.state.sessions[0].last_activity_ms+p.idle_expiry_ms)
        print({'repositories':16,**calls})
        assert calls=={'state_checks':1,'state_hashes':1,'json_validations':2}
    point.payload['interests'].append('caller mutation')
    point.vector[0]=0.
    assert effective.profile_payload['interests']==['Rust'],'owned output cannot alias caller payload'
    with pytest.raises(JournalConflict):owned_input(point,users,p,Context(uid(1),uid(50),1,p.sha256,snapshot.state.operation_version,1,clock))


def test_owned_serving_still_checks_corrupt_points_policy_heads_and_pending_contexts():
    from feedback.model_serving import Context,owned_input
    point,users,p,snapshot=serving_point_fixture(2)
    context=Context(uid(1),uid(50),1,p.sha256,snapshot.state.operation_version,1,snapshot.state.effective_ms)
    for mutate in (lambda row:setattr(row,'id',uid(2)),lambda row:row.payload.update(profile_version=True),
            lambda row:row.payload.update(model_state_sha256='f'*64),lambda row:row.payload.update(model_journal_head='f'*64),
            lambda row:row.payload.update(model_policy_sha256='f'*64),lambda row:row.vector.__setitem__(1,float('nan'))):
        bad=deepcopy(point);mutate(bad)
        from feedback.model_journal import JournalError
        with pytest.raises(JournalError):owned_input(bad,users,p,context)
    other=policy(generation_freshness_ms=101)
    with pytest.raises(model.CompatibilityError):owned_input(point,users,other,replace(context,policy_sha256=other.sha256))
    with pytest.raises(model.OperationGap):owned_input(point,users,p,replace(context,minimum_applied_operation_version=context.minimum_applied_operation_version+1))
    with pytest.raises(model.OperationGap):owned_input(point,users,p,replace(context,required_profile_version=2))
    with pytest.raises(model.ModelError):owned_input(point,users,p,replace(context,as_of_ms=0))
    with pytest.raises(model.EpochConflict):owned_input(point,users,p,replace(context,epoch=2))
    with pytest.raises(model.EpochConflict):owned_input(point,users,p,replace(context,as_of_ms=context.as_of_ms+p.idle_expiry_ms))
    assert owned_input(point,users,p,context).head==snapshot.head
    # A successful read cannot cache away the next request's committed-head proof.
    users.journal.records.data.pop('blob:'+snapshot.head)
    from feedback.model_journal import JournalMissing
    with pytest.raises(JournalMissing):owned_input(point,users,p,context)

def policy(**changes):
    # Controlled synthetic fixture numbers only; no production policy defaults.
    values = dict(version=model.VERSION, embedding=model.EmbeddingIdentity(REPOSITORY_EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION, 384),
        repository_recipe=REPOSITORY_EMBEDDING_VERSION, baseline_recipe='fixture-profile-v1',
        long_like=.4, long_dislike=.4, long_save=.4, session_like=.5, session_dislike=.5, session_save=.5,
        session_readme_open=.1, session_github_open=.2, session_share=.3, session_dwell=.3,
        long_repo_cap=.3, session_repo_cap=.25, session_total_cap=.4, session_blend=.25,
        generation_freshness_ms=100, half_life_ms=1000, idle_expiry_ms=10000, minimum_dwell_ms=3000,
        saturated_dwell_ms=6000, refresh_idle_on_noop=False, normalization_epsilon=1e-12, unit_tolerance=1e-6,
        max_repositories=256, max_sessions=4, max_state_bytes=1048576, max_replay_operations=1000,
        zero_norm_rule='long-term-then-baseline')
    values.update(changes)
    return model.Policy(**values)

def wire(p, kind, version, payload, *, user=uid(1), at=None):
    return canonical_json(dict(schema_version=1, operation_id=uid(1000+version), user_id=user,
        model_operation_version=str(version), previous_model_operation_version=str(version-1),
        accepted_at_ms=str(version if at is None else at), policy_version=model.VERSION,
        policy_sha256=p.sha256, kind=kind, operation=payload))

def profile_payload(initial=True, version=1):
    return dict(job_id=uid(20+version), profile_version=str(version), allow_genesis=initial,
                original_profile_context={'interests': ['Python', '検索']})

def begin_payload(at=2, session=uid(50)):
    return dict(session_id=session, epoch='1', epoch_started_ms=str(at), witness_sha256='b'*64)

def feedback_payload(version=1, kind='like', session=None):
    event = dict(event_id=uid(2000+version), user_id=uid(1), repo_id=uid(100), feedback_version=str(version),
                 event_type=kind, occurred_at='2026-10-24T00:00:00Z')
    if kind == 'dwell':
        event['dwell_ms'] = 4000
    payload = {'original_feedback_json': canonical_json(event)}
    if session is not None:
        payload['canonical_attribution'] = dict(user_id=uid(1), session_id=session, epoch='1', epoch_started_ms='2',
                                                 active_at_acceptance=True, witness_sha256='b'*64)
    return payload

def parse(text, p):
    return parse_operation(text, maximum_bytes=131072, allowed_policies={p.sha256})


def status_history(length,legacy_until=0,coordinator_type=None,**policy_changes):
    from feedback.model_journal import Coordinator, JournalMissing, _canonical
    class Records:
        def __init__(self):self.rows={};self.reads=0;self.writes=0
        def get(self,key):self.reads+=1;return self.rows.get(key)
        def put_once(self,key,raw):
            if key not in self.rows:self.writes+=1
            return self.rows.setdefault(key,raw)
    class HistoryJournal(Journal):
        legacy=False
        def put(self,kind,value):
            # An older writer's actual prepare format; the bytes are still
            # canonical/content-addressed and every chain validation is real.
            if self.legacy and kind=='prepare':value={k:v for k,v in value.items() if k!='status_index'}
            return super().put(kind,value)
    class Users:
        snapshot=None;fail_cas=False
        def read_optional(self,user):return self.snapshot
        def read(self,user):
            if self.snapshot is None:raise JournalMissing('fixture missing')
            return self.snapshot
        def insert_once(self,desired):
            if self.snapshot is not None:return False
            self.snapshot=desired;return True
        def compare_and_set(self,expected,desired):
            if self.fail_cas:self.fail_cas=False;return False
            if self.snapshot!=expected:return False
            self.snapshot=desired;return True
    p=policy(**policy_changes);records=Records();journal=HistoryJournal(records,Limits(131072,2097152,1048576,1000));users=Users()
    coordinator=(coordinator_type or Coordinator)(journal,users,p);snapshots={};headers={}
    def append():
        version=1 if users.snapshot is None else users.snapshot.state.operation_version+1
        journal.legacy=version<=legacy_until
        raw=wire(p,'profile',version,profile_payload(initial=version==1,version=version));operation_wire=parse(raw,p)
        source=model.PinnedVector.pin('profile',uid(1),version,0,p.baseline_recipe,p.embedding,basis(0))
        operation=model.Profile(operation_wire.header,source,version==1);headers[version]=operation.header
        legacy=_canonical({'vector':basis(0),'payload':{'user_id':uid(1),'profile_version':version,
            'last_feedback_version':0 if users.snapshot is None else users.snapshot.state.feedback_version,'feedback_state_revision':version}})
        if version==1:coordinator.initialize(operation.header,raw.encode(),lambda:operation,legacy,lambda:None)
        else:coordinator.apply(operation.header,'profile',raw.encode(),lambda:operation,lambda *_:legacy,lambda:None)
        snapshots[version]=users.snapshot
        return users.snapshot
    for _ in range(length):append()
    return SimpleNamespace(records=records,journal=journal,users=users,coordinator=coordinator,p=p,snapshots=snapshots,headers=headers,append=append)


@pytest.fixture(scope='module')
def indexed_history():return status_history(1000)


def test_historical_status_proofs_have_logarithmic_reads_and_exact_original_heads(indexed_history):
    import math
    f=indexed_history
    for length in (10,100,1000):
        targets=[1,length//2,length]
        if length==1000:targets+=[15,16,17,31,32,33,63,64,65,127,128,129,255,256,257,999]
        for target in targets:
            h=f.headers[target];f.records.reads=0
            result=f.coordinator.status_for_identity(f.snapshots[length],operation_id=h.operation_id,
                operation_version=target,source_sha256=h.source_envelope_sha256)
            reads=f.records.reads
            assert result['status']=='applied' and result['operation_head']==f.snapshots[target].head
            if target in (1,length):print({'history':length,'target':target,'journal_reads':reads})
            assert reads<=4+math.ceil(math.log2(max(1,length/16)))
    # Blocks plus binary carry nodes cost less than one new record per 8 ancestors.
    index_records=[raw for raw in f.records.rows.values() if json.loads(raw).get('kind')=='status_index']
    assert len(index_records)<1000/8


def test_legacy_status_stays_read_only_and_is_indexed_on_its_next_commit():
    f=status_history(40,legacy_until=40);h=f.headers[1];before=f.records.writes
    assert f.coordinator.original_status(f.users.snapshot,h)['status']=='applied'
    assert f.records.writes==before,'status lookup must never backfill or claim a write'
    f.append();f.records.reads=0
    assert f.coordinator.original_status(f.users.snapshot,h)['operation_head']==f.snapshots[1].head
    assert f.records.reads<=6
    # Rebuild still follows the original chain, including the new metadata.
    first=f.journal.get(f.snapshots[1].head,'prepare');checkpoint=first['genesis_checkpoint_sha256']
    assert f.coordinator.rebuild(checkpoint,f.users.snapshot.head,expected_user_id=uid(1))==f.users.snapshot


def test_raised_history_horizon_can_verify_retained_operations_before_the_index():
    f=status_history(100,legacy_until=100)
    f.journal.limits=replace(f.journal.limits,history_operations=10);f.append()
    frontier=f.journal.get(f.users.snapshot.head,'prepare')['status_index']
    assert frontier['first_version']==91
    f.journal.limits=replace(f.journal.limits,history_operations=100)
    for advance in (False,True):
        if advance:f.append()
        before=f.records.writes
        proof=f.coordinator.original_status(f.users.snapshot,f.headers[50])
        assert proof['status']=='applied' and proof['operation_head']==f.snapshots[50].head
        assert f.records.writes==before,'an expanded horizon must not turn status into a backfill writer'


def test_user_point_adapter_stops_initialization_guards_after_cancelled_dependencies(monkeypatch):
    from utils.deadline import RequestDeadline,deadline_scope,CooperativeDeadlineExceeded
    p=policy();guards=[];reads=[];deadline=RequestDeadline(10)
    def missing(**kwargs):reads.append(kwargs);deadline.cancel();return []
    users=QdrantAtomicUsers(SimpleNamespace(retrieve=missing),None,None,collection='cancelled-user-read',
        combined_user_bytes=1048576,policy=p,refuse_missing=lambda owner:guards.append(owner))
    with deadline_scope(deadline),pytest.raises(CooperativeDeadlineExceeded):users.read_optional(uid(1))
    assert len(reads)==1 and not guards
    point,users,p,_=serving_point_fixture(0);deadline=RequestDeadline(10)
    users.client=SimpleNamespace(retrieve=lambda **kwargs:[point]);users.ensure_initialized=lambda owner:guards.append(owner)
    original=users.journal.records.get;head_reads=[]
    def last_head_read(key):
        result=original(key);head_reads.append(key);deadline.cancel();return result
    monkeypatch.setattr(users.journal.records,'get',last_head_read)
    with deadline_scope(deadline),pytest.raises(CooperativeDeadlineExceeded):users.read_optional(uid(1))
    assert len(head_reads)==1 and not guards


def test_indexed_status_refuses_missing_corrupt_foreign_and_uncommitted_proofs(monkeypatch):
    from feedback.model_journal import JournalMissing,CasConflict,OriginalIdentityConflict,HistoryBudget
    from utils.deadline import RequestDeadline,deadline_scope,CooperativeDeadlineExceeded
    f=status_history(40);h=f.headers[1]
    with pytest.raises(OriginalIdentityConflict):f.coordinator.status_for_identity(f.users.snapshot,
        operation_id=uid(9999),operation_version=1,source_sha256=h.source_envelope_sha256)
    with pytest.raises(OriginalIdentityConflict):f.coordinator.status_for_identity(f.users.snapshot,
        operation_id=h.operation_id,operation_version=1,source_sha256='f'*64)
    root=f.journal.get(f.users.snapshot.head,'prepare')['status_index']['peaks'][0]['head'];key='blob:'+root;raw=f.records.rows.pop(key)
    try:
        with pytest.raises(JournalMissing):f.coordinator.original_status(f.users.snapshot,h)
    finally:f.records.rows[key]=raw
    # A self-consistent but foreign/malformed indexed node is not authority,
    # even when a faulty stored head points at its valid content address.
    committed=f.journal.get(f.users.snapshot.head,'prepare');node=f.journal.get(root,'status_index')
    for wrong in ({**node,'user_id':uid(2)},{**node,'policy_sha256':'f'*64},{**node,'first_version':2}):
        bad=deepcopy(committed);bad['status_index']['peaks'][0]['head']=f.journal.put('status_index',wrong)
        damaged=replace(f.users.snapshot,head=f.journal.put('prepare',bad))
        with pytest.raises(JournalConflict):f.coordinator.original_status(damaged,h)
    f.records.rows[key]=raw+b' '
    try:
        with pytest.raises(JournalConflict):f.coordinator.original_status(f.users.snapshot,h)
    finally:f.records.rows[key]=raw
    before=f.users.snapshot;f.users.fail_cas=True
    with pytest.raises(CasConflict):f.append()
    assert f.users.snapshot==before
    with pytest.raises(model.OperationGap):f.coordinator.original_status(f.users.snapshot,f.headers[41])
    assert f.coordinator.original_status(f.users.snapshot,h)['operation_head']==f.snapshots[1].head
    # Cancellation during a dependency read prevents the next history/index read.
    deadline=RequestDeadline(10);original=f.records.get;reads=0
    def cancelled(key):
        nonlocal reads
        result=original(key);reads+=1;deadline.cancel();return result
    with monkeypatch.context() as patch:
        patch.setattr(f.records,'get',cancelled)
        with deadline_scope(deadline),pytest.raises(CooperativeDeadlineExceeded):f.coordinator.original_status(f.users.snapshot,h)
    assert reads==1
    # The configured proof horizon remains a logical limit even with an index.
    old=f.journal.limits;f.journal.limits=replace(old,history_operations=10)
    try:
        with pytest.raises(HistoryBudget):f.coordinator.original_status(f.users.snapshot,h)
    finally:f.journal.limits=old


def test_registered_status_propagates_a_read_only_budget_into_the_feedback_executor(monkeypatch):
    import feedback.model_runtime as provider
    from utils.deadline import current_deadline
    f=status_history(20);h=f.headers[1];original_get=f.records.get;calls=[];finished=threading.Event()
    runtime=SimpleNamespace(users=f.users,processor=SimpleNamespace(coordinator=f.coordinator))
    monkeypatch.setattr(provider,'model_runtime',lambda:runtime);monkeypatch.setenv('INTERNAL_API_SECRET','fixture-secret')
    app=FastAPI();app.include_router(v2.router);client=TestClient(app)
    identity={'schema_version':1,'user_id':uid(1),'operation_id':h.operation_id,'model_operation_version':'1','source_envelope_sha256':h.source_envelope_sha256}
    headers={'x-internal-secret':'fixture-secret'}
    response=client.post('/api/v2/model-operation-status',headers=headers,json=identity)
    assert response.status_code==200 and response.json()['operation_head']==f.snapshots[1].head
    def cancel_after_read(key):
        deadline=current_deadline();assert deadline is not None,'the status worker must own the HTTP read budget'
        result=original_get(key);calls.append(key);deadline.cancel();return result
    original_status=f.coordinator.status_for_identity
    def observed(*args,**kwargs):
        try:return original_status(*args,**kwargs)
        finally:finished.set()
    monkeypatch.setattr(f.records,'get',cancel_after_read);monkeypatch.setattr(f.coordinator,'status_for_identity',observed)
    response=client.post('/api/v2/model-operation-status',headers=headers,json=identity)
    assert response.status_code==503 and finished.wait(2) and len(calls)==1
    calls.clear()
    response=client.post('/api/v2/model-operation-status',headers={**headers,'X-Request-Timeout-Ms':'nan'},json=identity)
    assert response.status_code==400 and not calls


def test_indexed_status_preserves_committed_rejections_withdrawals_and_orphan_boundaries():
    from feedback.model_journal import Coordinator,CasConflict,_canonical
    from feedback.model_wire import parse_withdrawal
    f=status_history(1,max_sessions=1);prepared=[]
    def compatibility(complete,operation):
        legacy=json.loads(complete)['legacy'];legacy['payload']['feedback_state_revision']+=1
        if isinstance(operation,model.Feedback):legacy['payload']['last_feedback_version']=operation.feedback_version
        return _canonical(legacy)
    coordinator=Coordinator(f.journal,f.users,f.p,reject_capacity=True,legacy_reject=compatibility,
        dead_letter_prepare=lambda header,input_sha,head,code:prepared.append((head,code)))
    def begin(version,session):
        raw=wire(f.p,'begin',version,begin_payload(version,session));header=parse(raw,f.p).header
        operation=model.Begin(header,session,version,'b'*64)
        return header,lambda:coordinator.apply(header,'begin',raw.encode(),lambda:operation,compatibility,lambda:None)
    _,first=begin(2,uid(50));first()
    rejected,second=begin(3,uid(51));f.users.fail_cas=True
    with pytest.raises(CasConflict):second()
    with pytest.raises(model.OperationGap):coordinator.original_status(f.users.snapshot,rejected)
    assert second()=='rejected';capacity_head=f.users.snapshot.head
    payload=feedback_payload(session=uid(51));payload['canonical_attribution']['epoch_started_ms']='3'
    raw=wire(f.p,'feedback',4,payload);feedback_header=parse(raw,f.p).header
    source=model.PinnedVector.pin('repository',uid(100),1,0,f.p.repository_recipe,f.p.embedding,basis(1))
    operation=model.Feedback(feedback_header,uid(2001),1,uid(100),'like',source,
        model.SessionBinding(uid(1),uid(51),1,3,True,'b'*64),None,None)
    assert coordinator.apply(feedback_header,'feedback',raw.encode(),lambda:operation,compatibility,lambda:None)=='rejected'
    dependency_head=f.users.snapshot.head;f.append()
    original=wire(f.p,'profile',6,profile_payload(initial=False,version=6))
    attestation=parse_withdrawal(withdrawal(f.p,original),maximum_bytes=131072,allowed_policies={f.p.sha256})
    assert coordinator.apply_withdrawal(attestation,lambda:None)=='withdrawn_profile'
    withdrawal_head=f.users.snapshot.head;f.append()
    for header,status,head in [(rejected,'rejected_capacity',capacity_head),
            (feedback_header,'rejected_session_dependency',dependency_head),(attestation.header,'withdrawn_profile',withdrawal_head)]:
        proof=coordinator.original_status(f.users.snapshot,header)
        assert proof['status']==status and proof['operation_head']==head

@pytest.mark.parametrize('mutate', [
    lambda raw: raw+' ', lambda raw: raw.replace('"schema_version":1', '"schema_version":true'),
    lambda raw: raw.replace('"schema_version":1', '"schema_version":1,"schema_version":1'),
    lambda raw: raw.replace('"model_operation_version":"1"', '"model_operation_version":"01"'),
    lambda raw: raw.replace('"accepted_at_ms":"1"', '"accepted_at_ms":1'),
    lambda raw: raw.replace('"allow_genesis":true', '"allow_genesis":1'),
    lambda raw: raw.replace('"kind":"profile"', '"kind":"reject"'),
    lambda raw: raw.replace('"検索"', '"\\u691c\\u7d22"'),
])
def test_wire_refuses_noncanonical_or_ambiguous_original_bytes(mutate):
    p = policy()
    with pytest.raises((WireError, model.ModelError)):
        parse(mutate(wire(p, 'profile', 1, profile_payload())), p)

def test_original_feedback_is_opaque_and_hashes_exact_delivered_utf8():
    p = policy(); payload = feedback_payload()
    payload['original_feedback_json'] = __import__('json').dumps(__import__('json').loads(payload['original_feedback_json']), indent=2)
    raw = wire(p, 'feedback', 2, payload)
    operation = parse(raw, p)
    assert operation.payload['original_feedback_json'] == payload['original_feedback_json']
    assert operation.header.source_envelope_sha256 == hashlib.sha256(raw.encode()).hexdigest()
    assert operation.feedback['feedback_version'] == '1'

@pytest.mark.parametrize('payload', [
    {'original_feedback_json': '{"event_id":"x","event_id":"y"}'},
    feedback_payload(kind='impression'),
    {**feedback_payload(), 'canonical_attribution': None},
])
def test_wire_does_not_admit_impressions_or_unknown_session_authority(payload):
    p = policy()
    with pytest.raises(WireError):
        parse(wire(p, 'feedback', 2, payload), p)

def test_every_policy_number_including_generation_freshness_changes_identity():
    assert policy().sha256 != policy(generation_freshness_ms=101).sha256
    with pytest.raises(model.ModelError):
        policy(generation_freshness_ms=0)

def runtime_config(p=None, namespace='fixture-model'):
    return dict(schema_version=1, policy=asdict(p or policy()),
        stream=asdict(StreamLimits(namespace, 100, 1000, 131072, 10, 10)),
        journal=dict(input_bytes=131072, record_bytes=2097152, combined_user_bytes=1048576, history_operations=1000),
        journal_collection='fixture_model_journal', max_journal_records=10000, max_journal_bytes=100000000,
        initialization_scope='fixture-weave-deployment-v1', max_initialization_guard_users=1000)

@pytest.mark.parametrize('bad_event', [
    lambda event:event.pop('occurred_at'),
    lambda event:event.update(occurred_at='2026-10-24T00:00:00+05:30'),
    lambda event:event.update(occurred_at='nonsense'),
    lambda event:event.update(event_type='dwell',dwell_ms=2**53-1),
])
def test_registered_route_refuses_invalid_feedback_before_any_admission(monkeypatch,bad_event):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.v2 import router
    import feedback.model_runtime as provider
    p=policy();event=__import__('json').loads(feedback_payload()['original_feedback_json']);bad_event(event)
    raw=wire(p,'feedback',2,{'original_feedback_json':canonical_json(event)})
    calls=[]
    stream=OperationStream(SimpleNamespace(eval=lambda *args:calls.append(args)),
        StreamLimits('invalid-batch',100,1000,131072,10,10),{p.sha256},
        initialization_scope='invalid-batch',max_guard_users=100)
    monkeypatch.setattr(provider,'model_runtime',lambda:SimpleNamespace(stream=stream))
    monkeypatch.setenv('INTERNAL_API_SECRET','fixture-secret')
    app=FastAPI();app.include_router(router)
    response=TestClient(app).post('/api/v2/model-operations',headers={'x-internal-secret':'fixture-secret'},
        json={'schema_version':1,'operations':[{'operation_json':raw}]})
    assert response.status_code==422,response.text
    assert not calls

def test_registered_route_rejects_nonfinite_numeric_json_before_admission(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.v2 import router
    import feedback.model_runtime as provider
    p=policy();raw=wire(p,'profile',1,profile_payload()).replace('"allow_genesis":true','"allow_genesis":1e999')
    with pytest.raises(WireError):parse(raw,p)
    calls=[];stream=OperationStream(SimpleNamespace(eval=lambda *args:calls.append(args)),
        StreamLimits('invalid-batch',100,1000,131072,10,10),{p.sha256},
        initialization_scope='invalid-batch',max_guard_users=100)
    monkeypatch.setattr(provider,'model_runtime',lambda:SimpleNamespace(stream=stream));monkeypatch.setenv('INTERNAL_API_SECRET','fixture-secret')
    app=FastAPI();app.include_router(router)
    response=TestClient(app).post('/api/v2/model-operations',headers={'x-internal-secret':'fixture-secret'},json={'schema_version':1,'operations':[{'operation_json':raw}]})
    assert response.status_code==422,response.text
    assert not calls

def test_reducer_repetition_decay_inverse_and_unit_normalization():
    p=policy();owner=uid(1);repo=uid(100);session=uid(50)
    base=model.PinnedVector.pin('profile',owner,1,0,p.baseline_recipe,p.embedding,basis(0))
    state=model.initial_state(owner,base,p,accepted_ms=0)
    def header(at):
        return model.Header(uid(5000+state.operation_version),owner,state.operation_version+1,state.operation_version,at,'a'*64)
    state=model.reduce_operation(state,model.Profile(header(0),base,True),p)
    state=model.reduce_operation(state,model.Begin(header(1),session,1,'b'*64),p)
    source=model.PinnedVector.pin('repository',repo,1,0,p.repository_recipe,p.embedding,basis(1))
    binding=model.SessionBinding(owner,session,1,1,True,'b'*64)
    def event(kind,at):
        return model.Feedback(header(at),uid(6000+state.feedback_version),state.feedback_version+1,repo,kind,
            None if kind in {'unlike','undislike','unsave'} else source,binding,None,None)
    state=model.reduce_operation(state,event('like',2),p)
    initial=model.recommendation_view(state,p,session_id=session,epoch=1,as_of_ms=2)
    for _ in range(20):state=model.reduce_operation(state,event('like',2),p)
    repeated=model.recommendation_view(state,p,session_id=session,epoch=1,as_of_ms=2)
    assert repeated.long_term==initial.long_term and repeated.session==initial.session
    later=model.recommendation_view(state,p,session_id=session,epoch=1,as_of_ms=1002)
    assert model.norm(later.session)==pytest.approx(model.norm(initial.session)/2)
    assert later.long_term==initial.long_term
    assert model.norm(later.blended)==pytest.approx(1)
    state=model.reduce_operation(state,event('unlike',1002),p)
    final=model.recommendation_view(state,p,session_id=session,epoch=1,as_of_ms=1002)
    assert final.blended==tuple(basis(0)) and final.explicit_negative_repo_ids==()

def test_valid_original_safe_integer_feedback_cursor_is_preserved_without_reencoding():
    p=policy();payload=feedback_payload();event=__import__('json').loads(payload['original_feedback_json'])
    event['feedback_version']=1;payload['original_feedback_json']=__import__('json').dumps(event,indent=2)
    parsed=parse(wire(p,'feedback',2,payload),p)
    assert parsed.feedback['feedback_version']==1
    assert parsed.payload['original_feedback_json']==payload['original_feedback_json']

def test_registry_builds_one_client_set_under_concurrent_first_use(monkeypatch,tmp_path):
    import feedback.model_runtime as provider
    from concurrent.futures import ThreadPoolExecutor
    import threading,time
    provider.model_runtime.cache_clear()
    path=tmp_path/'explicit-config.json';path.write_text(canonical_json(runtime_config()))
    monkeypatch.setenv('WEAVE_MODEL_LANE_CONFIG',str(path))
    calls=[];mutex=threading.Lock()
    def build(config):
        with mutex:calls.append(config)
        time.sleep(.01)
        return object()
    monkeypatch.setattr(provider,'ModelRuntime',build)
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:results=list(executor.map(lambda _:provider.model_runtime(),range(8)))
        assert len(calls)==1 and all(value is results[0] for value in results)
    finally:provider.model_runtime.cache_clear()

def test_journal_budget_includes_base64_original_bytes_and_complete_state():
    from feedback.model_journal import JournalError
    with pytest.raises(JournalError):Limits(131072,131072,1048576,1000)

def withdrawal(p, original_json, *, revoked_at=100, witness='e'*64):
    import json
    value=json.loads(original_json);payload=value['operation']
    fields={key:value[key] for key in ('schema_version','operation_id','user_id','model_operation_version',
        'previous_model_operation_version','accepted_at_ms','policy_version','policy_sha256')}
    fields.update(source_envelope_sha256=hashlib.sha256(original_json.encode()).hexdigest(),
        profile_job_id=payload['job_id'],profile_version=payload['profile_version'],allow_genesis=payload['allow_genesis'],
        revoked_at_ms=str(revoked_at),revocation_witness_sha256=witness,reason='github_consent_revoked')
    return canonical_json(fields)

def test_withdrawal_attestation_does_not_include_or_reconstruct_private_profile_context():
    from feedback.model_wire import parse_withdrawal
    p=policy();raw=wire(p,'profile',1,profile_payload())
    attestation=withdrawal(p,raw)
    parsed=parse_withdrawal(attestation,maximum_bytes=131072,allowed_policies={p.sha256})
    assert parsed.header.source_envelope_sha256==hashlib.sha256(raw.encode()).hexdigest()
    assert parsed.profile_version==1 and parsed.allow_genesis is True
    assert 'original_profile_context' not in attestation and '検索' not in attestation
    invalid=__import__('json').loads(attestation);invalid['original_profile_context']={'bio':'revoked'}
    with pytest.raises(WireError):parse_withdrawal(canonical_json(invalid),maximum_bytes=131072,allowed_policies={p.sha256})

def test_only_marked_legacy_float_vector_projection_uses_exact_hex_leaves():
    import math
    from copy import deepcopy
    from feedback.model_journal import _legacy_float_projection,_exact_payload_matches
    payload={'preference_accumulator':[7.541981512397263e-35,-0.0,1],
        'profile_baseline_vector':[.3], 'feedback_latent_vector':[.2],
        'feedback_adjustments':{uid(100):{'like':{'delta':[.4],'score':.7}}},
        'profile_version':1,'other_float':.3}
    original=deepcopy(payload);encoded=_legacy_float_projection(payload)
    assert payload==original and encoded['preference_accumulator']==[float.hex(7.541981512397263e-35),'-0x0.0p+0',1]
    assert encoded['feedback_adjustments'][uid(100)]['like']=={'delta':[float.hex(.4)],'score':.7}
    assert encoded['other_float']==.3 and _exact_payload_matches(encoded,_legacy_float_projection(original))
    wrong=deepcopy(encoded);wrong['profile_version']=1.0
    assert not _exact_payload_matches(wrong,encoded)
    for value in [math.inf,math.nan]:
        with pytest.raises(JournalConflict):_legacy_float_projection({'preference_accumulator':[value]})


# Model history

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


# Model review regressions

def application():
    app=FastAPI();app.include_router(v2.router)
    @app.middleware('http')
    async def request_id(request, call_next):
        request.state.request_id=uid(999)
        return await call_next(request)
    @app.get('/ping')
    async def ping():return {'live':True}
    return app

@pytest.mark.parametrize('error,expected',[(model.OperationGap('pending'),503),
    (model.CompatibilityError('corrupt'),500)])
def test_registered_uncaptured_pending_model_is_retryable_but_corruption_is_not(monkeypatch,error,expected):
    monkeypatch.setenv('INTERNAL_API_SECRET','fixture-secret')
    def unavailable(*args,**kwargs):raise error
    monkeypatch.setattr(v2,'retriever',lambda:SimpleNamespace(recommend_batch=unavailable))
    p=policy()
    result=TestClient(application()).post('/api/v2/recommendations/generate',headers={'x-internal-secret':'fixture-secret'},json={
        'schema_version':2,'user_id':uid(1),'generation_id':uid(3000),'feed_version':'1','limit':5,
        'exclude_repo_ids':[],'context':{'cold_start':False},'model_context':{
            'schema_version':1,'session_id':uid(50),'epoch':'1','policy_sha256':p.sha256,
            'minimum_applied_operation_version':'3','required_profile_version':'1','as_of_ms':'3'}})
    assert result.status_code==expected,result.text
    assert ('Retry-After' in result.headers)==(expected==503)

def model_requests():
    p=policy();raw=wire(p,'profile',1,profile_payload());incoming=parse(raw,p)
    identity={'schema_version':1,'user_id':uid(1),'operation_id':incoming.header.operation_id,
        'model_operation_version':'1','source_envelope_sha256':incoming.header.source_envelope_sha256}
    return [('/api/v2/model-operations',{'schema_version':1,'operations':[{'operation_json':raw}]}),
        ('/api/v2/model-operation-withdrawals',{'schema_version':1,'withdrawals':[{'withdrawal_json':withdrawal(p,raw,revoked_at=2)}]}),
        ('/api/v2/model-operation-status',identity),
        ('/api/v2/model-authority',{'schema_version':1,'user_id':uid(1),'model_context':{
            'schema_version':1,'session_id':uid(50),'epoch':'1','policy_sha256':p.sha256,
            'minimum_applied_operation_version':'2','required_profile_version':'1','as_of_ms':'2'}})]

@pytest.mark.parametrize('path,body',model_requests())
def test_first_runtime_construction_stays_off_event_loop(monkeypatch,path,body):
    import feedback.model_runtime as provider
    monkeypatch.setenv('INTERNAL_API_SECRET','fixture-secret')
    entered=threading.Event();release=threading.Event();threads=[]
    def blocked():
        threads.append(threading.get_ident());entered.set()
        assert release.wait(5)
        raise provider.ModelLaneUnavailable('fixture unavailable')
    monkeypatch.setattr(provider,'model_runtime',blocked)
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application()),base_url='http://fixture') as client:
            task=asyncio.create_task(client.post(path,headers={'x-internal-secret':'fixture-secret'},json=body))
            # A bounded watchdog lets the failing-before direct constructor exit.
            timer=threading.Timer(2,release.set);timer.start()
            try:
                assert await asyncio.to_thread(entered.wait,1)
                ping=await client.get('/ping')
                assert ping.status_code==200
                assert threads[0]!=threading.get_ident()
                assert not release.is_set(), "unrelated request must finish while runtime construction is still blocked"
            finally:
                release.set();timer.cancel()
                response=await task
            assert response.status_code==503,response.text
    asyncio.run(exercise())

def test_model_routes_have_distinct_fixed_metrics_and_unknown_paths_are_bounded():
    metrics.reset_metrics_for_tests()
    try:
        for path,_ in model_requests():
            metrics.record_api_request(path=path,method='POST',status_code=503,duration_seconds=.01)
        metrics.record_api_request(path='/api/v2/model-operation-status/private-user',method='POST',status_code=404,duration_seconds=.01)
        output=metrics.render_prometheus({})
        for path,_ in model_requests():assert f'route="{path}"' in output
        assert 'route="/api/v2/other"' in output and 'private-user' not in output
    finally:metrics.reset_metrics_for_tests()
