"""Offline retrieval I/O/deadline acceptance with an injected dependency delay.

Run: uv run python -m scripts.benchmark_retrieval_io
No services, credentials, models or product data are used. Serial-worker control
and bounded fan-out use the same real retriever and deterministic eligible points.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

import numpy as np
from config import EMBEDDING_MODEL_REVISION, REPOSITORY_EMBEDDING_DIM, REPOSITORY_EMBEDDING_MODEL
from embedding.vector_contract import repository_payload_defaults, REPOSITORY_DISCOVERY_CHANNELS
from retrieval import runtime as io
from retrieval.v2_retriever import QdrantV2Retriever
from inference import runtime as recommendation
from api.request_budget import run_read_only_request
from utils.deadline import RequestDeadline, current_deadline, deadline_scope
from utils import runtime_lifecycle

VECTOR = [1.] + [0.] * (REPOSITORY_EMBEDDING_DIM - 1)
STAGES = ('profile', 'semantic', 'trend', 'activity', 'popularity', 'freshness')


def summary(values):
    return {'count': len(values), 'p50_ms': float(np.percentile(values, 50) * 1000),
            'p95_ms': float(np.percentile(values, 95) * 1000),
            'p99_ms': float(np.percentile(values, 99) * 1000),
            'max_ms': max(values) * 1000} if values else {'count': 0}


class FixtureQdrant:
    """Synthetic canonical fixtures, real eligibility checks, no external client."""

    def __init__(self, *, delay=.01, fail_stage=None, hook=None):
        self.delay, self.fail_stage, self.hook = delay, fail_stage, hook
        self.user_id = str(UUID(int=100))
        self.repo_ids = [str(UUID(int=index)) for index in range(1, 5)]
        self.lock = threading.Lock()
        self.calls = []
        self.active = self.peak = 0
        self.started = threading.Event()

    def call(self, stage, function):
        began = time.perf_counter()
        deadline = current_deadline()
        after_stop = bool(deadline and (deadline.cancelled.is_set() or deadline.clock() >= deadline.expires_at))
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            record = {'stage': stage, 'started': began, 'after_stop': after_stop}
            self.calls.append(record)
        self.started.set()
        try:
            if self.hook:
                self.hook(stage)
            else:
                time.sleep(self.delay)
            if stage == self.fail_stage:
                raise TimeoutError('injected isolated dependency timeout')
            return function()
        finally:
            with self.lock:
                self.active -= 1
                record['service_seconds'] = time.perf_counter() - began

    def point(self, repo_id, score=None):
        payload = repository_payload_defaults()
        payload.update(repo_id=repo_id, full_name=f'fixture/{repo_id}', content_version=1,
            content_hash=f'content-{repo_id}', source_hash=f'source-{repo_id}',
            indexed_at=datetime(2026, 10, 23, tzinfo=timezone.utc).isoformat(), star_count=10)
        return SimpleNamespace(id=repo_id, payload=payload, vector=None, score=score)

    def retrieve(self, **kwargs):
        return self.call('profile', lambda: [SimpleNamespace(
            id=self.user_id, vector=VECTOR, payload={
                'embedding_dim': REPOSITORY_EMBEDDING_DIM, 'embedding_model': REPOSITORY_EMBEDDING_MODEL,
                'embedding_model_revision': EMBEDDING_MODEL_REVISION, 'last_feedback_version': 0,
            })])

    def query_points(self, **kwargs):
        return self.call('semantic', lambda: SimpleNamespace(points=[
            self.point(self.repo_ids[0], .9), self.point(self.repo_ids[1], .8)]))

    def scroll(self, **kwargs):
        stage = next(name for name, key in REPOSITORY_DISCOVERY_CHANNELS.items()
                     if key == kwargs['order_by'].key)
        return self.call(stage, lambda: ([self.point(repo_id) for repo_id in self.repo_ids], None))


def retriever(client):
    # The optional hybrid fixture avoids loading any ranker/model. It does not
    # qualify or alter production required-heavy behavior.
    return QdrantV2Retriever(client=client, heavy_ranker_enabled=False,
                             heavy_ranker_required=False, minimum_eligible_repositories=0)


def until(predicate, seconds=5):
    stop = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= stop:
            raise AssertionError('isolated work did not reach its expected state')
        time.sleep(.001)


@contextmanager
def isolated_runtimes(*, workers=4, capacity=8):
    previous = io._pool
    if previous is not None:
        previous.drain()
    io._pool = None
    recommendation.shutdown_recommendation_runtime()
    with patch.dict(os.environ, {
        'V2_RETRIEVAL_IO_WORKERS': str(workers), 'V2_RETRIEVAL_IO_MAX_OUTSTANDING': str(capacity),
        'V2_RECOMMENDATION_EXECUTOR_WORKERS': '1', 'V2_RECOMMENDATION_MAX_OUTSTANDING': '1',
    }), patch.object(runtime_lifecycle, '_retiring', threading.Event()):
        try:
            yield
        finally:
            until(lambda: recommendation.recommendation_runtime_status()['outstanding'] == 0)
            if io._pool is not None:
                io._pool.drain()
                io._pool = None
            recommendation.shutdown_recommendation_runtime()


def measure_requests(*, workers, requests, clients, delay, fail_stage=None):
    client = FixtureQdrant(delay=delay, fail_stage=fail_stage)
    durations, outputs, errors = [], [], []
    with isolated_runtimes(workers=workers):
        engine = retriever(client)
        def run(index):
            began = time.perf_counter()
            try:
                with deadline_scope(RequestDeadline(5)):
                    batch = engine.recommend_batch(client.user_id, 4, [], f'fixture-{index}')
                return time.perf_counter() - began, tuple((r.repo_id, r.score, r.source) for r in batch.items), None
            except Exception as exc:
                return time.perf_counter() - began, (), type(exc).__name__
        began = time.perf_counter()
        with ThreadPoolExecutor(max_workers=clients) as pool:
            for duration, output, error in pool.map(run, range(requests)):
                durations.append(duration)
                outputs.append(output)
                if error:
                    errors.append(error)
        elapsed = time.perf_counter() - began
        snapshot = io.retrieval_runtime_status()
    return {'request_duration': summary(durations), 'elapsed_seconds': elapsed,
        'requests_per_second': requests / elapsed, 'request_errors': errors,
        'dependency_calls': len(client.calls), 'dependency_peak': client.peak,
        'dependency_timeout_calls': sum(r['stage'] == fail_stage for r in client.calls),
        'post_stop_dependency_calls': sum(r['after_stop'] for r in client.calls),
        'runtime': snapshot,
        'dependency_service': {stage: summary([r['service_seconds'] for r in client.calls
                                             if r['stage'] == stage]) for stage in STAGES},
        'outputs': outputs}


async def measure_abandonment(kind):
    """Real request-budget watcher/executors, fake disconnected request/dependency."""
    release, disconnected = threading.Event(), threading.Event()
    client = FixtureQdrant(hook=lambda stage: release.wait(5))
    async def is_disconnected():
        return disconnected.is_set()
    request = SimpleNamespace(headers={}, is_disconnected=is_disconnected)
    with isolated_runtimes(workers=2, capacity=3):
        engine = retriever(client)
        budget = 1.1 if kind == 'timeout' else 5
        began = time.perf_counter()
        task = asyncio.create_task(run_read_only_request(request, budget, lambda:
            recommendation.run_recommendation_job(engine.recommend_batch, client.user_id, 4, [])))
        try:
            assert await asyncio.to_thread(client.started.wait, 2)
            if kind == 'disconnect':
                disconnected.set()
            try:
                await task
                raise AssertionError('abandoned request unexpectedly completed')
            except (asyncio.CancelledError, recommendation.RecommendationDeadlineExceeded):
                pass
            returned = time.perf_counter()
            calls_at_return = len(client.calls)
            held = recommendation.recommendation_runtime_status()['outstanding']
            try:
                await recommendation.run_recommendation_job(lambda: None)
                rejected = False
            except recommendation.RecommendationCapacityError:
                rejected = True
            # Intentional controlled dependency tail after the caller has gone.
            await asyncio.sleep(.05)
            release.set()
            await asyncio.to_thread(until, lambda: recommendation.recommendation_runtime_status()['outstanding'] == 0)
            ended = time.perf_counter()
            return {'kind': kind, 'caller_duration_ms': (returned - began) * 1000,
                'caller_return_to_worker_exit_ms': (ended - returned) * 1000,
                'intentional_dependency_tail_ms': 50, 'admission_held_at_return': held,
                'new_request_rejected_until_exit': rejected, 'calls_at_return': calls_at_return,
                'calls_after_caller_return': sum(r['started'] > returned for r in client.calls),
                'post_stop_dependency_calls': sum(r['after_stop'] for r in client.calls),
                'settled_runtime': io.retrieval_runtime_status(),
                'deadline_metrics': recommendation.recommendation_runtime_status()}
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.to_thread(until, lambda: recommendation.recommendation_runtime_status()['outstanding'] == 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=int, default=24)
    parser.add_argument('--clients', type=int, default=4)
    parser.add_argument('--delay-ms', type=float, default=10)
    args = parser.parse_args()
    if not 4 <= args.requests <= 200 or not 1 <= args.clients <= 8 or not 1 <= args.delay_ms <= 50:
        parser.error('requests4..200, clients1..8, delay-ms1..50 required')
    common = dict(requests=args.requests, clients=args.clients, delay=args.delay_ms / 1000)
    serial = measure_requests(workers=1, **common)
    fanout = measure_requests(workers=4, **common)
    partial = measure_requests(workers=4, fail_stage='trend', **common)
    assert serial.pop('outputs') == fanout.pop('outputs'), 'worker-budget controls changed ranking'
    assert all(partial.pop('outputs')), 'one failed channel lost supported partial results'
    report = {'scope': __doc__, 'workload': vars(args),
        'limits': 'Synthetic dependency delay and canonical fixtures; no network, heavy ranker/model, live load or production speedup measured. Worker-count comparison is a control, not a historical release.',
        'serial_worker_control': serial, 'shared_fanout': fanout, 'one_channel_timeout': partial,
        'abandonment': [asyncio.run(measure_abandonment(kind)) for kind in ('disconnect', 'timeout')]}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
