from __future__ import annotations
import copy
import inspect
import json
from pathlib import Path
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from fastapi.testclient import TestClient
import numpy as np
import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient, models
from api import main, v2, scoring_capture
from api.contracts import CapturedRecommendationResponse, UnscoredEmptyRecommendationResponse, RecommendationRequest
from config import EMBEDDING_MODEL_REVISION, REPOSITORY_EMBEDDING_MODEL
from embedding.vector_contract import repository_payload_defaults
from inference.ranker_service import RankerService
from inference.scoring_provenance import ScoredPacket, decode_owned_packet
from retrieval.v2_retriever import QdrantV2Retriever, _USER_RETRIEVAL_PAYLOAD_FIELDS
import traceback
from api import scoring_capture
from api.metrics import record_scoring_capture_failure, render_prometheus, reset_metrics_for_tests
from inference.scoring_runtime import CaptureOptions
from functools import lru_cache
from jsonschema import Draft202012Validator
from api import main
from api.contracts import RecommendationRequest
import os
import stat
import subprocess
from urllib.parse import urlsplit
from qdrant_client import QdrantClient
from api import v2
from api.contracts import CapturedRecommendationResponse
from inference.scoring_provenance import decode_owned_packet
SECRET = 'private-test-internal'
CAPABILITY = '7' * 64
COMMIT = 'a' * 40
USER = str(uuid.UUID(int=7001))

@pytest.fixture(scope='module')
def trained_ranker():
    folder = Path(inspect.getfile(RankerService)).parent
    return RankerService(model_path=str(folder / 'heavy_ranker.pt'), scaler_path=str(folder / 'feature_scaler.json'), manifest_path=str(folder / 'model_manifest.json'), require_production_manifest=True, device='cpu')

def make_retriever(client, ranker, *, prefix=None, count=12):
    suffix = prefix or uuid.uuid4().hex
    repositories, users = ('capture_repos_' + suffix, 'capture_users_' + suffix)
    client.create_collection(repositories, vectors_config={'repo_embedding': models.VectorParams(size=384, distance=models.Distance.COSINE)})
    client.create_collection(users, vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
    user = np.zeros(384, np.float32)
    user[0] = 1.0
    client.upsert(users, points=[models.PointStruct(id=USER, vector=user.tolist(), payload={'embedding_dim': 384, 'embedding_model': REPOSITORY_EMBEDDING_MODEL, 'embedding_model_revision': EMBEDDING_MODEL_REVISION, 'skills': ['python'], 'profile_version': 2, 'last_feedback_version': 3, 'feedback_state_revision': 4})], wait=True)
    for index in range(count):
        repo_id = str(uuid.UUID(int=8000 + index))
        vector = np.zeros(384, np.float32)
        vector[0], vector[index + 1] = (0.8, 0.6)
        payload = repository_payload_defaults()
        payload.update(repo_id=repo_id, full_name=f'diagnostic/repo-{index}', content_version=1, content_hash='a' * 64, source_hash='b' * 64, indexed_at='2026-10-24T00:00:00Z', star_count=index * 100, primary_language='Python', languages=['Python'], topics=['python'], pushed_days_ago=1, delta_7d=1, activity_score=0.5)
        payload.pop('feature_version', None)
        client.upsert(repositories, points=[models.PointStruct(id=repo_id, vector={'repo_embedding': vector.tolist()}, payload=payload)], wait=True)
    return (QdrantV2Retriever(client=client, repository_collection=repositories, user_collection=users, ranker=ranker, heavy_ranker_enabled=True, heavy_ranker_required=True, heavy_ranker_traffic_percent=100.0, minimum_eligible_repositories=1), repositories, users)

@pytest.fixture
def actual_pipeline(monkeypatch, trained_ranker):
    monkeypatch.setenv('INTERNAL_API_SECRET', SECRET)
    monkeypatch.setenv('APP_ENV', 'development')
    monkeypatch.setenv('SCORING_CAPTURE_ENABLED', 'true')
    monkeypatch.setenv('SCORING_CAPTURE_CAPABILITY', CAPABILITY)
    monkeypatch.setenv('SCORING_CAPTURE_CODE_COMMIT', COMMIT)
    monkeypatch.setenv('BAKED_ML_RELEASE_ID', COMMIT)
    client = QdrantClient(':memory:')
    retriever, repositories, users = make_retriever(client, trained_ranker)
    monkeypatch.setattr(v2, 'retriever', lambda: retriever)
    matrices = []
    hook = trained_ranker.model.register_forward_pre_hook(lambda _model, args: matrices.append(args[0].detach().cpu().numpy().copy()))
    try:
        yield (retriever, client, repositories, users, matrices)
    finally:
        hook.remove()
        client.close()

def payload(*, captured=True, generation=None, feed_version=None):
    result = {'schema_version': 2, 'user_id': USER, 'generation_id': generation or str(uuid.uuid4()), 'feed_version': feed_version if feed_version is not None else '9007199254740993' if captured else 1, 'limit': 5, 'exclude_repo_ids': [], 'social_repo_ids': [], 'context': {'cold_start': False}}
    if captured:
        result['scoring_capture'] = {'schema_version': 1, 'encoding': 'float32-le-v1'}
    return result

def headers(*, capability=CAPABILITY):
    return {'x-internal-secret': SECRET, 'x-scoring-capture-capability': capability}

def post(body, request_headers=None):
    return TestClient(main.app).post('/api/v2/recommendations/generate', json=body, headers=request_headers or headers())

@lru_cache(maxsize=1)
def _generated_request_validator():
    document = main.app.openapi()
    wire = document['paths']['/api/v2/recommendations/generate']['post']['requestBody']['content']['application/json']['schema']
    schema = {**wire, 'components': document['components']}
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)
IMAGE = 'qdrant/qdrant@sha256:75eab8c4ba42096724fdcfde8b4de0b5713d529dde32f285a1f86fdcb2c9e50c'

@contextmanager
def owned_server():
    target = os.getenv('DOCKER_HOST') or 'unix:///var/run/docker.sock'
    parsed = urlsplit(target)
    if parsed.scheme != 'unix' or parsed.netloc or parsed.query or parsed.fragment or (not parsed.path.startswith('/')):
        raise RuntimeError('owned capture fixture requires a local Unix Docker socket')
    if not stat.S_ISSOCK(Path(parsed.path).stat().st_mode):
        raise RuntimeError('local Docker socket is unavailable')
    environment = {key: value for key, value in os.environ.items() if not key.startswith('DOCKER_')}
    name = 'weave-scoring-provider-' + uuid.uuid4().hex[:16]
    owner = uuid.uuid4().hex

    def docker(*arguments, check=True):
        result = subprocess.run(['docker', '--host', target, *arguments], env=environment, check=False, capture_output=True, text=True, timeout=30)
        if check and result.returncode:
            raise RuntimeError('owned capture Docker fixture operation failed')
        return result
    primary_error = None
    try:
        docker('run', '--pull=never', '-d', '--name', name, '--label', 'weave.scoring-capture-owner=' + owner, '--memory', '1g', '--cpus', '2', '--pids-limit', '256', '-p', '127.0.0.1::6333', '-e', 'QDRANT__TELEMETRY_DISABLED=true', IMAGE)
        info = json.loads(docker('inspect', name).stdout)[0]
        binding = info['NetworkSettings']['Ports']['6333/tcp']
        assert len(binding) == 1 and binding[0]['HostIp'] == '127.0.0.1'
        url = 'http://127.0.0.1:' + binding[0]['HostPort']
        client = QdrantClient(url=url, api_key=None, timeout=2, check_compatibility=False)
        try:
            deadline = time.monotonic() + 20
            while True:
                try:
                    client.get_collections()
                    break
                except Exception:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.1)
            yield client
        finally:
            client.close()
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            result = docker('inspect', name, check=False)
            if result.returncode == 0:
                info = json.loads(result.stdout)[0]
                if info['Config']['Labels'].get('weave.scoring-capture-owner') != owner:
                    raise RuntimeError('owned capture fixture label mismatch; refusing removal')
                docker('rm', '-f', '-v', name)
            elif 'no such object' not in result.stderr.casefold() and 'no such container' not in result.stderr.casefold():
                raise RuntimeError('could not verify capture fixture cleanup')
        except Exception as cleanup_error:
            if primary_error is None:
                raise
            if hasattr(primary_error, 'add_note'):
                primary_error.add_note(str(cleanup_error))
