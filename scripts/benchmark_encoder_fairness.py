"""Offline CPU encoder fairness fixture; no network/storage/provider/service startup.

Uses the pinned cached real MiniLM and current production BoundedEmbeddingModel,
SentenceTransformerEmbedder grouping and query-cache miss path. Optional ASGI
requests exercise real api.main search admission with an empty fake Qdrant.
Neither mode measures live dependencies, retrieval quality or production capacity.
"""
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from hashlib import sha256
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from config import REPOSITORY_EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION, REPOSITORY_EMBEDDING_DIM
from embedding.runtime import BoundedEmbeddingModel
from embedding.embeddings import SentenceTransformerEmbedder, chunk_text
from embedding.query_cache import CachedQueryEmbedder

CLASS = ContextVar('fixture_work_class', default='warmup')
SUBMITTED = ContextVar('fixture_call_submission', default=None)

def quantiles(values):
    return {'count': len(values), 'mean_seconds': float(np.mean(values)),
            'p50_seconds': float(np.percentile(values, 50)),
            'p95_seconds': float(np.percentile(values, 95)),
            'p99_seconds': float(np.percentile(values, 99)),
            'max_seconds': max(values)} if values else {'count': 0}

class MeasuredRaw:
    def __init__(self, model):
        self.model, self.tokenizer = model, model.tokenizer
        self.max_seq_length = model.max_seq_length
        self.lock = threading.Lock()
        self.records = []
        self.active = self.max_active = 0

    def encode(self, texts, **kwargs):
        entered, operation = time.perf_counter(), CLASS.get()
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            return self.model.encode(texts, **kwargs)
        finally:
            ended = time.perf_counter()
            with self.lock:
                self.active -= 1
                self.records.append({'class': operation, 'texts': len(texts),
                    'entered': entered, 'ended': ended,
                    'wait': entered - SUBMITTED.get(), 'service': ended - entered})

class MeasuredBounded(BoundedEmbeddingModel):
    def encode(self, *args, **kwargs):
        token = SUBMITTED.set(time.perf_counter())
        try:
            return super().encode(*args, **kwargs)
        finally:
            SUBMITTED.reset(token)


def measure(encoder, *, grouped, args):
    raw = MeasuredRaw(encoder)
    bounded = MeasuredBounded(raw, max_concurrency=1)
    embedder = SentenceTransformerEmbedder(model=bounded)
    query = CachedQueryEmbedder(model=bounded, cache=bounded.query_embedding_cache,
        model_revision=EMBEDDING_MODEL_REVISION, embedding_version='local-fixture-v1',
        expected_size=REPOSITORY_EMBEDDING_DIM)
    # A long synthetic software README, with real current character chunking.
    readme = '\n\n'.join(f'## Section {i}\n' + ('This repository explains parsing, indexing, storage and testing. ' * 35)
                           for i in range(args.sections))
    chunks = [chunk.text for chunk in chunk_text(readme)]
    jobs = {'repository': [], 'search': [], 'onboard': []}
    job_lock = threading.Lock()
    barrier = threading.Barrier(args.repository_jobs + args.search_workers + 1)
    http_statuses = {}

    def run(operation, worker):
        token = CLASS.set(operation)
        try:
            barrier.wait(timeout=10)
            count = 1 if operation == 'repository' else (args.queries if operation == 'search' else args.profiles)
            for index in range(count):
                started = time.perf_counter()
                if operation == 'repository':
                    if grouped:
                        values = embedder.embed_texts(chunks)
                    else:
                        values = bounded.encode(chunks, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
                    assert len(values) == len(chunks)
                else:
                    text = (f'Find repository {worker}-{index} for parser tooling' if operation == 'search'
                            else f'Developer {worker}-{index} interested in Python testing and storage')
                    if operation == 'search' and args.transport == 'http':
                        response = http.post('/api/v2/search/repositories',
                            headers={'x-internal-secret': 'isolated-encoder-fixture'},
                            json={'schema_version': 2, 'search_id': str(uuid4()), 'query': text, 'limit': 5})
                        with job_lock:
                            http_statuses.setdefault(str(response.status_code), []).append(time.perf_counter() - started)
                        assert response.status_code in {200, 503}, response.text
                    else:
                        values = query.embed_text(text) if operation == 'search' else embedder.embed_text(text)
                        assert len(values) == REPOSITORY_EMBEDDING_DIM and np.isfinite(values).all()
                with job_lock:
                    jobs[operation].append(time.perf_counter() - started)
        finally:
            CLASS.reset(token)

    started = time.perf_counter()
    with http_search_fixture(bounded, enabled=args.transport == 'http') as http:
        with ThreadPoolExecutor(max_workers=barrier.parties) as pool:
            work = [('repository', index) for index in range(args.repository_jobs)]
            work += [('search', index) for index in range(args.search_workers)] + [('onboard', 0)]
            futures = [pool.submit(run, operation, index) for operation, index in work]
            for future in futures:
                future.result(timeout=180)
    elapsed = time.perf_counter() - started
    records = raw.records
    repo = [r for r in records if r['class'] == 'repository']
    first_repo = min(r['entered'] for r in repo)
    last_repo = max(r['ended'] for r in repo)
    report = {
        'duration_seconds': elapsed,
        'http_status_durations': {status: quantiles(values) for status, values in http_statuses.items()}, 'max_actual_encoder_concurrency': raw.max_active,
        'readme_chars': len(readme), 'readme_chunks_per_job': len(chunks),
        'completed_repository_jobs': len(jobs['repository']),
        'repository_encode_groups': len(repo), 'max_repository_group_texts': max(r['texts'] for r in repo),
        'query_cache': bounded.query_embedding_cache.snapshot(),
        'classes': {operation: {
            'encoder_wait': quantiles([r['wait'] for r in records if r['class'] == operation]),
            'encoder_service': quantiles([r['service'] for r in records if r['class'] == operation]),
            'embedding_job_duration': quantiles(durations),
            'calls_started_while_repository_active': sum(first_repo <= r['entered'] <= last_repo
                for r in records if r['class'] == operation),
        } for operation, durations in jobs.items()},
    }
    assert raw.max_active == 1 and len(jobs['repository']) == args.repository_jobs
    if grouped:
        assert report['max_repository_group_texts'] <= 8
        assert report['classes']['search']['calls_started_while_repository_active'] > 0
        assert report['classes']['onboard']['calls_started_while_repository_active'] > 0
    return report


@contextmanager
def http_search_fixture(model, *, enabled):
    if not enabled:
        yield None
        return
    from fastapi.testclient import TestClient
    from api import runtime, v2
    from api.main import app
    from retrieval.runtime import shutdown_retrieval_runtime
    from retrieval.v2_retriever import QdrantV2Retriever

    # Process-local fixture only. Never enter lifespan or construct a live client.
    runtime.shutdown_service_runtime()
    retriever = QdrantV2Retriever(
        client=SimpleNamespace(query_points=lambda **kwargs: SimpleNamespace(points=[])),
        heavy_ranker_enabled=False, heavy_ranker_required=False,
    )
    original_search_job = v2._search_repository_job
    def classified_search_job(request):
        token = CLASS.set('search')
        try:
            return original_search_job(request)
        finally:
            CLASS.reset(token)
    with patch.dict(os.environ, {'INTERNAL_API_SECRET': 'isolated-encoder-fixture',
                                 'INTERNAL_API_HEADER': 'x-internal-secret',
                                 'APP_ENV': 'development', 'V2_SEARCH_EXECUTOR_WORKERS': '1',
                                 'V2_SEARCH_MAX_OUTSTANDING': '4', 'V2_SEARCH_TIMEOUT_SECONDS': '4'}), \
         patch.object(v2, 'shared_embedding_model', return_value=model), \
         patch.object(v2, 'retriever', return_value=retriever), \
         patch.object(v2, '_search_repository_job', side_effect=classified_search_job):
        try:
            # No context manager: production lifespan must stay off.
            yield TestClient(app)
        finally:
            # HTTP cancellation cannot end a real encoder call. Join the
            # process-owned work before sampling or retiring the fixture.
            stop = time.monotonic() + 30
            while runtime.service_runtime_status()['search']['outstanding']:
                if time.monotonic() >= stop:
                    raise RuntimeError('search workers did not drain after the fixture')
                time.sleep(.005)
            runtime.shutdown_service_runtime()
            shutdown_retrieval_runtime()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sections', type=int, default=160)
    parser.add_argument('--queries', type=int, default=64, help='unique queries per client stream')
    parser.add_argument('--profiles', type=int, default=32)
    parser.add_argument('--repository-jobs', type=int, default=2)
    parser.add_argument('--search-workers', type=int, default=2, help='client streams, not HTTP service executor workers')
    parser.add_argument('--transport', choices=['encoder', 'http'], default='encoder')
    parser.add_argument('--only-grouped', action='store_true')
    args = parser.parse_args()
    for name, low, high in [('sections', 8, 320), ('queries', 8, 256), ('profiles', 8, 128),
                            ('repository_jobs', 1, 4), ('search_workers', 1, 4)]:
        if not low <= getattr(args, name) <= high:
            parser.error(f'{name} must be between {low} and {high}')
    torch.set_num_threads(1)
    model = SentenceTransformer(REPOSITORY_EMBEDDING_MODEL, revision=EMBEDDING_MODEL_REVISION,
                               local_files_only=True, device='cpu')
    model.encode(['Warm model for offline encoder fairness measurement.'], show_progress_bar=False)
    report = {
        'scope': __doc__, 'model': REPOSITORY_EMBEDDING_MODEL, 'revision': EMBEDDING_MODEL_REVISION,
        'device': 'cpu', 'torch_threads': torch.get_num_threads(), 'python': platform.python_version(),
        'processor': platform.processor(), 'cpu_count': os.cpu_count(),
        'source_sha256': {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in
            ['embedding/runtime.py', 'embedding/embeddings.py', 'embedding/query_cache.py']},
        'workload': vars(args),
        'limitations': ['Synthetic README/query/profile text, real cached pinned CPU model.',
            'Encoder transport excludes HTTP/Qdrant/admission. HTTP transport uses real api.main ASGI, auth, validation, bounded search execution and cached real encoder, with empty fake Qdrant; lifespan stays off.',
            'HTTP durations separate successful responses from 503 capacity/deadline failures; no live service latency claim.',
            'Ungrouped control is a counterfactual full-list encode, not a historical checked-out release.',
            'One process/run/order; scheduling and hardware influence percentiles; not a production bound.'],
        'ungrouped_control': None if args.only_grouped else measure(model, grouped=False, args=args),
        'current_grouped': measure(model, grouped=True, args=args),
    }
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
