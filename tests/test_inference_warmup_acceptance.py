"""Execute existing warmup and actual lifespan/retirement/readiness boundaries."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from starlette.datastructures import State

from embedding import runtime as embeddings
from inference.feature_spec import INPUT_DIM
from tests.test_ranker_service import loaded_ranker_service, cpu_ranker_service


class CountingEncoder:
    max_seq_length=256
    def __init__(self,kind='valid',entered=None,release=None):
        self.kind,self.entered,self.release=kind,entered,release
        self.calls=[]
    def encode(self,texts,**kwargs):
        self.calls.append((list(texts),kwargs))
        if self.entered is not None: self.entered.set()
        if self.release is not None: assert self.release.wait(3)
        values=np.zeros((len(texts),384),dtype=np.float32)
        values[:,0]=1
        if self.kind=='shape': return values[:,:383]
        if self.kind=='nan': values[0,0]=np.nan
        if self.kind=='zero': values[0,:]=0
        return values


def test_concurrent_encoder_warmup_executes_one_encode_and_readiness_never_repeats_it():
    entered,release=threading.Event(),threading.Event()
    encoder=CountingEncoder(entered=entered,release=release)
    model=embeddings.BoundedEmbeddingModel(encoder,max_concurrency=1)
    with ThreadPoolExecutor(max_workers=4) as workers:
        futures=[workers.submit(model.warmup) for _ in range(4)]
        try:
            assert entered.wait(2)
        finally:
            release.set()
        assert all(future.result(2)['embedding_runtime_ready'] for future in futures)
    assert len(encoder.calls)==1 and len(encoder.calls[0][0])==2
    assert encoder.calls[0][1]['normalize_embeddings'] is True
    for _ in range(10):
        assert model.identity()['embedding_runtime_ready']
        assert model.warmup()['embedding_runtime_ready']
    assert len(encoder.calls)==1


@pytest.mark.parametrize('kind',['shape','nan','zero'])
def test_encoder_failure_is_sticky_and_never_declares_ready(kind):
    encoder=CountingEncoder(kind)
    model=embeddings.BoundedEmbeddingModel(encoder,max_concurrency=1)
    with pytest.raises(ValueError): model.warmup()
    assert model.identity()['embedding_warmup_state']=='failed'
    with pytest.raises(RuntimeError,match='previously failed'): model.warmup()
    assert len(encoder.calls)==1


def test_ranker_warmup_executes_one_real_torch_forward_in_inference_mode(cpu_ranker_service):
    service=cpu_ranker_service
    observed=[]
    hook=service.model.register_forward_pre_hook(
        lambda _module,args: observed.append((tuple(args[0].shape),torch.is_inference_mode_enabled())))
    try:
        with ThreadPoolExecutor(max_workers=3) as workers:
            futures=[workers.submit(service.warmup) for _ in range(3)]
            assert all(future.result(5)['heavy_ranker_warmup_ready'] for future in futures)
        assert observed==[((1,INPUT_DIM),True)]
        for _ in range(5): assert service.warmup()['heavy_ranker_warmup_ready']
        assert len(observed)==1
    finally:
        hook.remove()


class ControlledForward(torch.nn.Module):
    def __init__(self,kind='valid',entered=None,release=None):
        super().__init__(); self.kind=kind; self.calls=0
        self.entered,self.release=entered,release
    def forward(self,inputs):
        self.calls+=1
        if self.entered is not None: self.entered.set()
        if self.release is not None: assert self.release.wait(3)
        value=inputs[:,:1].reshape(-1)
        if self.kind=='nan': value=value*float('nan')
        if self.kind=='shape': value=value.repeat(2)
        count=4 if self.kind=='heads' else 5
        return tuple(value for _ in range(count))


@pytest.mark.parametrize('kind',['nan','shape','heads'])
def test_ranker_invalid_predictions_fail_readiness_and_do_not_retry_warmup(cpu_ranker_service,kind):
    service=cpu_ranker_service
    service.model=ControlledForward(kind)
    with pytest.raises(ValueError): service.warmup()
    assert service.warmup_status()['heavy_ranker_warmup_state']=='failed'
    with pytest.raises(RuntimeError,match='previously failed'): service.warmup()
    assert service.model.calls==1


def test_warm_runtime_loads_only_pinned_local_artifacts_and_reuses_pipeline_instances(monkeypatch):
    import sentence_transformers
    captured=[]; constructions={'repository':0,'user':0}
    encoder=CountingEncoder()
    def load(name,**kwargs): captured.append((name,kwargs)); return encoder
    def repository(**kwargs):
        constructions['repository']+=1
        return SimpleNamespace(config=SimpleNamespace(embedding_dim=384))
    def user(**kwargs):
        constructions['user']+=1
        return object()
    factories=[embeddings.repository_embedding_pipeline,embeddings.user_onboarding_pipeline,
               embeddings.shared_embedding_model]
    for factory in factories: factory.cache_clear()
    monkeypatch.setattr(sentence_transformers,'SentenceTransformer',load)
    monkeypatch.setattr(embeddings,'_configure_cpu_threads',lambda: 1)
    monkeypatch.setattr(embeddings,'RepositoryEmbeddingPipeline',repository)
    monkeypatch.setattr(embeddings,'UserOnboardingPipeline',user)
    monkeypatch.setattr(embeddings,'card_summary_pipeline',lambda: object())
    previous=getattr(embeddings._warmup_loading,'local_files_only',False)
    try:
        for _ in range(3): assert embeddings.warm_embedding_runtime()['embedding_runtime_ready']
        assert len(captured)==1 and captured[0][1]['local_files_only'] is True
        assert captured[0][1]['revision']==embeddings.embedding_model_revision()
        assert len(encoder.calls)==1
        assert constructions=={'repository':1,'user':1}
        assert all(factory.cache_info().misses==1 for factory in factories)
        assert getattr(embeddings._warmup_loading,'local_files_only',False)==previous
    finally:
        for factory in factories: factory.cache_clear()


@pytest.fixture
def startup(monkeypatch,cpu_ranker_service):
    from api import main as api_main, v2
    monkeypatch.setenv('APP_ENV','development')
    monkeypatch.setenv('EMBEDDING_WARMUP_ON_STARTUP','true')
    monkeypatch.setenv('INFERENCE_WARMUP_TIMEOUT_SECONDS','1')
    monkeypatch.setenv('INTERNAL_API_SECRET','warmup-test-only')
    monkeypatch.setattr(api_main.app,'state',State())
    monkeypatch.setattr(v2,'_runtime_retirement',None)
    closed=threading.Event()
    retirements=[]
    encoder=CountingEncoder()
    model=embeddings.BoundedEmbeddingModel(encoder,max_concurrency=1)
    @lru_cache(maxsize=1)
    def shared(): return model
    shared.peek=lambda: model
    monkeypatch.setattr(embeddings,'shared_embedding_model',shared)
    monkeypatch.setattr(embeddings,'repository_embedding_pipeline',lambda: SimpleNamespace(config=SimpleNamespace(embedding_dim=384)))
    monkeypatch.setattr(embeddings,'user_onboarding_pipeline',lambda: object())
    cpu_ranker_service.model=ControlledForward()
    runtime=SimpleNamespace(heavy_ranker_enabled=True,heavy_ranker_required=True,ranker=cpu_ranker_service)
    monkeypatch.setattr(v2,'retriever',lambda: runtime)
    settings=SimpleNamespace(production=False)
    monkeypatch.setattr(v2,'validate_v2_runtime_configuration',lambda: settings)
    class Diagnostics:
        async def run_periodic(self): await asyncio.Event().wait()
        def snapshot(self): return {'status':'healthy','fixture_dependencies':True}
    diagnostics=Diagnostics()
    monkeypatch.setattr(v2,'health_diagnostics',lambda: diagnostics)
    # Actual generation-owned retirement/cleanup implementations, with only
    # resource actions replaced by an observable test close. No datastore calls.
    def retire(*,after_startup=None):
        plan=v2._RuntimeCleanupPlan(lambda:[('fixture.close',closed.set)],lambda:[])
        generation=v2._RuntimeRetirement(plan,None,after_startup)
        retirements.append(generation)
        return generation
    monkeypatch.setattr(v2,'begin_v2_runtime_retirement',retire)
    for name in ['repository_store','user_profile_store','producer']:
        monkeypatch.setattr(v2,name,lambda: pytest.fail('warmup must not open user/corpus/feedback storage'))
    yield SimpleNamespace(app=api_main.app,encoder=encoder,model=model,runtime=runtime,
                          closed=closed,retirements=retirements,settings=settings)
    assert closed.is_set() or not retirements


def test_actual_lifespan_runs_warmup_once_before_repeated_readiness(startup):
    with TestClient(startup.app) as client:
        assert startup.app.state.inference_warmup_ready is True
        for _ in range(5):
            response=client.get('/api/v2/health/ready',headers={'x-internal-secret':'warmup-test-only'})
            assert response.status_code==200 and response.json()['inference_warmup_ready'] is True
        assert len(startup.encoder.calls)==startup.runtime.ranker.model.calls==1
    assert startup.closed.wait(2)


def test_required_failed_warmup_never_opens_readiness(startup):
    startup.runtime.ranker.model=ControlledForward('nan')
    with pytest.raises(ValueError,match='non-finite'):
        with TestClient(startup.app): pytest.fail('failed warmup must prevent startup')
    assert startup.app.state.inference_warmup_ready is False
    response=TestClient(startup.app).get('/api/v2/health/ready',headers={'x-internal-secret':'warmup-test-only'})
    assert response.status_code==503
    assert startup.closed.wait(2)


@pytest.mark.parametrize('stage',['encoder','ranker'])
def test_warmup_timeout_keeps_resources_until_real_worker_exit_and_never_becomes_ready(startup,monkeypatch,stage):
    entered,release=threading.Event(),threading.Event()
    if stage=='encoder':
        startup.encoder.entered,startup.encoder.release=entered,release
    else:
        startup.runtime.ranker.model=ControlledForward(entered=entered,release=release)
    monkeypatch.setenv('INFERENCE_WARMUP_TIMEOUT_SECONDS','0.1')
    try:
        with pytest.raises(RuntimeError,match='startup deadline'):
            with TestClient(startup.app): pytest.fail('timed-out startup must not open readiness')
        assert entered.is_set() and not startup.closed.is_set()
        assert startup.app.state.inference_warmup_ready is False
        assert startup.runtime.ranker.model.calls==(0 if stage=='encoder' else 1)
    finally:
        release.set()
        assert startup.closed.wait(2)
    assert startup.app.state.inference_warmup_ready is False
    if stage=='encoder':
        assert startup.model.identity()['embedding_warmup_state']=='failed'
    else:
        assert startup.model.identity()['embedding_runtime_ready'] is True
        assert startup.runtime.ranker.warmup_status()['heavy_ranker_warmup_state']=='failed'


def test_production_cannot_disable_mandatory_warmup(startup,monkeypatch):
    from scripts import validate_production_config
    monkeypatch.setenv('APP_ENV','production')
    monkeypatch.setenv('INTERNAL_API_SECRET','b'*64)
    monkeypatch.setenv('EMBEDDING_WARMUP_ON_STARTUP','false')
    monkeypatch.setattr(validate_production_config,'validate_production_config',lambda:[])
    startup.settings.production=True
    with pytest.raises(RuntimeError,match='must be enabled'):
        with TestClient(startup.app): pytest.fail('mandatory warmup must remain enforced')
    assert startup.encoder.calls==[] and startup.runtime.ranker.model.calls==0


def test_unloaded_ranker_artifact_cannot_be_marked_warm(cpu_ranker_service):
    service=cpu_ranker_service
    service.model=ControlledForward()
    service._model_loaded=False
    with pytest.raises(RuntimeError,match='artifact is not loaded'): service.warmup()
    assert service.model.calls==0
    assert service.warmup_status()['heavy_ranker_warmup_ready'] is False


def test_missing_local_encoder_is_not_retried_by_readiness_inspection(monkeypatch):
    calls=[]
    def missing():
        calls.append(getattr(embeddings._warmup_loading,'local_files_only',False))
        raise FileNotFoundError('isolated missing local artifact')
    factory=embeddings._single_model_cache(missing)
    monkeypatch.setattr(embeddings,'shared_embedding_model',factory)
    previous=getattr(embeddings._warmup_loading,'local_files_only',False)
    with pytest.raises(FileNotFoundError): embeddings.warm_embedding_runtime()
    for _ in range(10): assert embeddings.embedding_runtime_status()['embedding_runtime_ready'] is False
    assert calls==[True] and factory.peek() is None
    assert getattr(embeddings._warmup_loading,'local_files_only',False)==previous


def test_explicit_optional_ranker_failure_is_reported_without_claiming_heavy_readiness(startup):
    startup.runtime.heavy_ranker_required=False
    startup.runtime.ranker.model=ControlledForward('nan')
    with TestClient(startup.app):
        status=startup.app.state.inference_warmup
        assert status['heavy_ranker_warmup_ready'] is False
        assert status['heavy_ranker_warmup_state']=='failed'
        assert status['heavy_ranker_warmup_required'] is False
        assert startup.runtime.ranker is None
        assert startup.app.state.inference_warmup_ready is True
    assert startup.closed.wait(2)


def test_profile_fingerprint_includes_onboarding_and_detects_setup_only_changes(tmp_path, monkeypatch):
    from scripts import profile_inference_warmup as profile
    names = ['scripts/profile_inference_warmup.py','scripts/user_onboarding.py',
             'embedding/runtime.py','inference/ranker_service.py','api/main.py',
             'inference/feature_spec.py','inference/value_function.py',
             'scripts/benchmark_ranker_devices.py']
    for name in names:
        target = tmp_path/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text('original source\n')
    monkeypatch.setattr(profile,'ROOT',tmp_path)
    monkeypatch.setattr(profile,'__file__',str(tmp_path/names[0]))
    monkeypatch.setattr(profile.subprocess,'check_output',lambda *a,**kw:('\0'.join(names)+'\0').encode())
    before=profile.source_hashes()
    (tmp_path/'scripts/user_onboarding.py').write_text('changed timed constructor\n')
    assert 'scripts/user_onboarding.py' in before
    assert profile.source_hashes()!=before
    with pytest.raises(RuntimeError,match='source changed'):
        profile.require_sources(before)
