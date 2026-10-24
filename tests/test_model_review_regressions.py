"""Registered API responsiveness, pending state and bounded model telemetry."""
import asyncio
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import v2, metrics
from feedback import long_session as model
from tests.test_model_operations import policy, wire, uid, profile_payload, withdrawal, parse


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
