"""Offline JSON-vector transport comparison on the actual required-heavy retriever.

Both explicit production transport modes run the same actual retriever; only
the datastore responses and heavy scoring are deterministic local fixtures.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

from config import (EMBEDDING_MODEL_REVISION, REPOSITORY_EMBEDDING_DIM,
                    REPOSITORY_EMBEDDING_MODEL, QDRANT_VECTOR_NAME)
from embedding.vector_contract import repository_payload_defaults, REPOSITORY_DISCOVERY_CHANNELS
from retrieval import runtime as io
from retrieval.v2_retriever import QdrantV2Retriever

ROOT = Path(__file__).resolve().parents[1]
USER_VECTOR = [1.] + [0.] * (REPOSITORY_EMBEDDING_DIM - 1)


class FixtureRanker:
    ready = True
    production_qualified = True  # A fixture interface flag, not model qualification.
    model_version = 'fixture-vector-transport-v1'

    def __init__(self):
        self.candidates = []

    def score_batch(self, user_vector, skills, candidates):
        self.candidates = candidates
        return [{'repo_id': item['id'], 'final_score': 1 - int(UUID(item['id'])) / 10000}
                for item in candidates]


class WireQdrant:
    """Serialize/decode complete fixture responses and inject per-call RTT."""

    def __init__(self, *, overlap='high', rtt_ms=0, points=20, change=None):
        self.overlap, self.rtt = overlap, rtt_ms / 1000
        self.points, self.change = points, change
        self.user_id = str(UUID(int=10000))
        self.records = []
        self.changed_id = None
        self.lock = threading.Lock()
        raw = [(index + 1) / 1000 for index in range(REPOSITORY_EMBEDDING_DIM)]
        norm = math.sqrt(sum(value * value for value in raw))
        self.vector = [value / norm for value in raw]

    def repo(self, index):
        repo_id = str(UUID(int=index + 1))
        payload = repository_payload_defaults()
        payload.update(repo_id=repo_id, full_name=f'fixture/{repo_id}', content_version=1,
            feature_version=1, content_hash=f'content-{repo_id}', source_hash=f'source-{repo_id}',
            indexed_at=datetime(2026, 10, 23, tzinfo=timezone.utc).isoformat(),
            primary_language=['Python', 'Rust', 'JavaScript'][index % 3], star_count=100 + index)
        return {'id': repo_id, 'payload': payload, 'vector': {QDRANT_VECTOR_NAME: self.vector}}

    def project(self, row, kwargs):
        row = dict(row)
        if not kwargs.get('with_vectors'):
            row['vector'] = None
        fields = kwargs.get('with_payload')
        if isinstance(fields, list):
            row['payload'] = {name: value for name, value in row['payload'].items() if name in fields}
        return row

    def exchange(self, stage, rows):
        encoded_at = time.thread_time()
        wire = json.dumps(rows, separators=(',', ':'), allow_nan=False).encode()
        encode_cpu = time.thread_time() - encoded_at
        if self.rtt:
            time.sleep(self.rtt)
        decoded_at = time.thread_time()
        decoded = json.loads(wire)
        decode_cpu = time.thread_time() - decoded_at
        result = [SimpleNamespace(**row) for row in decoded]
        with self.lock:
            self.records.append({'stage': stage, 'bytes': len(wire), 'encode_cpu_seconds': encode_cpu,
                'decode_cpu_seconds': decode_cpu,
                'ids': [row['id'] for row in decoded],
                'vector_ids': [row['id'] for row in decoded if row.get('vector') and stage != 'profile']})
        return result

    @staticmethod
    def exclusions(condition):
        return {str(value) for clause in (getattr(condition, 'must_not', None) or [])
                for value in (getattr(clause, 'has_id', None) or [])}

    def channel(self, stage, kwargs):
        offset = 0 if self.overlap == 'high' or stage == 'semantic' else (
            list(REPOSITORY_DISCOVERY_CHANNELS).index(stage) + 1) * self.points
        excluded = self.exclusions(kwargs.get('query_filter') or kwargs.get('scroll_filter'))
        rows = [self.repo(index) for index in range(offset, offset + self.points)]
        rows = [row for row in rows if row['id'] not in excluded][:kwargs['limit']]
        for row in rows:
            if stage == 'semantic':
                row['score'] = .9
        return self.exchange(stage, [self.project(row, kwargs) for row in rows])

    def query_points(self, **kwargs):
        return SimpleNamespace(points=self.channel('semantic', kwargs))

    def scroll(self, **kwargs):
        stage = next(name for name, field in REPOSITORY_DISCOVERY_CHANNELS.items()
                     if field == kwargs['order_by'].key)
        return self.channel(stage, kwargs), None

    def retrieve(self, **kwargs):
        if kwargs['ids'] == [self.user_id]:
            payload = {'embedding_dim': REPOSITORY_EMBEDDING_DIM,
                'embedding_model': REPOSITORY_EMBEDDING_MODEL,
                'embedding_model_revision': EMBEDDING_MODEL_REVISION, 'last_feedback_version': 0}
            return self.exchange('profile', [{'id': self.user_id, 'vector': USER_VECTOR, 'payload': payload}])
        rows = [self.repo(int(UUID(repo_id)) - 1) for repo_id in kwargs['ids']]
        if self.change and rows:
            bad = rows[0]
            self.changed_id = bad['id']
            if self.change == 'deleted':
                rows.pop(0)
            elif self.change == 'ineligible':
                bad['payload'].pop('serving_eligibility_version')
            elif self.change == 'version':
                bad['payload']['content_version'] += 1
            elif self.change == 'shape':
                bad['vector'] = {QDRANT_VECTOR_NAME: [1.]}
        return self.exchange('vectors', [self.project(row, kwargs) for row in rows])


@contextmanager
def isolated_io():
    if io._pool is not None:
        io._pool.drain()
        io._pool = None
    with patch.dict(os.environ, {'V2_RETRIEVAL_IO_WORKERS': '4', 'V2_RETRIEVAL_IO_MAX_OUTSTANDING': '8'}):
        try:
            yield
        finally:
            if io._pool is not None:
                io._pool.drain()
                io._pool = None


def run_case(*, overlap, inline=False, rtt_ms=0, max_candidates=150, heavy=True, change=None):
    client = WireQdrant(overlap=overlap, rtt_ms=rtt_ms, change=change)
    ranker = FixtureRanker()
    with isolated_io():
        engine = QdrantV2Retriever(candidate_vector_mode="inline" if inline else "deferred", client=client, ranker=ranker if heavy else None, max_candidates=max_candidates,
            heavy_ranker_enabled=heavy, heavy_ranker_required=heavy,
            heavy_ranker_traffic_percent=100 if heavy else 0, minimum_eligible_repositories=0)
        began = time.perf_counter()
        batch = engine.recommend_batch(client.user_id, 10, [], 'wire-parity-seed')
        elapsed = time.perf_counter() - began
        snapshot = io.retrieval_runtime_status()
    output = [(item.repo_id, item.score, item.source) for item in batch.items]
    return {'output': output, 'elapsed_ms': elapsed * 1000, 'wire_bytes': sum(row['bytes'] for row in client.records),
        'json_encode_cpu_ms': sum(row['encode_cpu_seconds'] for row in client.records) * 1000,
        'json_decode_cpu_ms': sum(row['decode_cpu_seconds'] for row in client.records) * 1000,
        'transport_calls': len(client.records), 'ranker_candidate_ids': [row['id'] for row in ranker.candidates],
        'repository_vector_ids': [repo_id for row in client.records for repo_id in row['vector_ids']],
        'records': client.records, 'runtime': snapshot, 'ranker_applied': batch.ranker_applied,
        'changed_id': client.changed_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--high-rtt-ms', type=float, default=50)
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.high_rtt_ms <= 500 or not 2 <= args.runs <= 10:
        parser.error('--high-rtt-ms must be 1..500 and --runs must be 2..10')
    reports = {}
    for overlap in ('low', 'high'):
        reports[overlap] = {}
        for rtt in (0, args.high_rtt_ms):
            samples = {'validated_inline': [], 'current_deferred': []}
            for index in range(args.runs):
                pair = {}
                # Alternate order to expose cold/order effects rather than
                # attributing one first-run sample to transport policy.
                modes = (True, False) if index % 2 == 0 else (False, True)
                for inline in modes:
                    name = 'validated_inline' if inline else 'current_deferred'
                    pair[name] = run_case(overlap=overlap, inline=inline, rtt_ms=rtt)
                assert pair['validated_inline']['output'] == pair['current_deferred']['output']
                for name, report in pair.items():
                    samples[name].append(report)
            reports[overlap][str(rtt)] = {
                name: {
                    'samples': args.runs,
                    'wire_bytes': rows[0]['wire_bytes'], 'transport_calls': rows[0]['transport_calls'],
                    'elapsed_ms_median': statistics.median(row['elapsed_ms'] for row in rows),
                    'elapsed_ms_range': [min(row['elapsed_ms'] for row in rows), max(row['elapsed_ms'] for row in rows)],
                    'json_decode_cpu_ms_median': statistics.median(row['json_decode_cpu_ms'] for row in rows),
                    'json_encode_cpu_ms_median': statistics.median(row['json_encode_cpu_ms'] for row in rows),
                } for name, rows in samples.items()
            }
    print(json.dumps({'scope': __doc__, 'source_sha256': hashlib.sha256((ROOT / 'retrieval/v2_retriever.py').read_bytes()).hexdigest(),
        'limits': 'Synthetic immutable canonical points and deterministic fixture ranker, real JSON encode/decode CPU, injected per-call RTT. No live services, heavy model quality or production speedup measured. No live Qdrant or production workload measured.',
        'cases': reports}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
