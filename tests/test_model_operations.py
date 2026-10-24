"""Negotiated source-byte, genesis and ordered-journal regression fixtures."""
from dataclasses import asdict, replace
import hashlib
from types import SimpleNamespace
import uuid

import pytest

from feedback import long_session as model
from feedback.model_wire import WireError, canonical_json, parse_operation
from feedback.model_journal import Journal, Limits, QdrantAtomicUsers, JournalConflict
from feedback.model_processor import ModelProcessor
from feedback.model_stream import StreamLimits
from feedback.v2 import OrderedFeedbackApplier
from feedback.v2_settings import V2FeedbackSettings
from embedding.qdrant_cas import payload_snapshot_filter
from config import EMBEDDING_MODEL_REVISION, REPOSITORY_EMBEDDING_MODEL, REPOSITORY_EMBEDDING_VERSION


def uid(n):
    return str(uuid.UUID(int=n))


def basis(n):
    return [float(i == n) for i in range(384)]


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
    stream=SimpleNamespace(limits=SimpleNamespace(input_bytes=131072),policies={p.sha256},admit=lambda raw:calls.append(raw))
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
    calls=[];stream=SimpleNamespace(limits=SimpleNamespace(input_bytes=131072),policies={p.sha256},admit=lambda raw:calls.append(raw))
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
