"""Compare pinned encoders offline, in serial fresh CPU worker processes.

No production configuration, Qdrant collection or ranker is changed. The default
fixture has synthetic labels; measured encoder costs do not qualify model quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone

import numpy as np

from evaluation.embedding_registry import Registry, compatible_scores, encode_role
from evaluation.retrieval_metrics import retrieval_metrics

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / 'tests/fixtures/embedding_models.json'
FIXTURE = ROOT / 'tests/fixtures/readme_retrieval.json'


def read_json(path):
    with Path(path).open('rb') as source:
        raw = source.read(32 * 1024 * 1024 + 1)
    if len(raw) > 32 * 1024 * 1024:
        raise ValueError('offline input exceeds byte budget')
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def validate_fixture(fixture):
    if not isinstance(fixture.get('fixture_version'), str) or not fixture['fixture_version'].strip():
        raise ValueError('fixture requires an explicit version')
    repos, queries = fixture['repositories'], fixture['queries']
    if not 1 <= len(repos) <= 2000 or not 1 <= len(queries) <= 1000:
        raise ValueError('fixture counts exceed offline evaluation bounds')
    ids = [repo['repo_id'] for repo in repos]
    for repo in repos:
        if not isinstance(repo['repo_id'], str) or str(uuid.UUID(repo['repo_id'])) != repo['repo_id']:
            raise ValueError('fixture requires canonical repository UUIDs')
        if not isinstance(repo.get('full_name'), str) or not repo['full_name'].strip():
            raise ValueError('repository full_name is required')
        for field in ('full_name', 'description', 'readme'):
            if not isinstance(repo.get(field, ''), str):
                raise ValueError('repository text fields must contain strings')
        if not isinstance(repo.get('topics', []), list) or any(not isinstance(topic, str) for topic in repo.get('topics', [])):
            raise ValueError('repository topics must be strings')
    if len(ids) != len(set(ids)):
        raise ValueError('fixture repository IDs must be unique')
    for query in queries:
        if not isinstance(query['query'], str) or not query['query'].strip() or len(query['query']) > 4096:
            raise ValueError('fixture query must contain text')
        relevant = query['relevant_repo_ids']
        if not relevant or len(set(relevant)) != len(relevant) or not set(relevant) <= set(ids):
            raise ValueError('fixture relevance must resolve to unique repository IDs')
    if not isinstance(fixture['provenance'], str) or not fixture['provenance'].strip():
        raise ValueError('fixture requires explicit label provenance')
    return repos, queries


def atomic_json(path, payload):
    target = Path(path)
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=f'.{target.name}.')
        with os.fdopen(descriptor, 'w') as output:
            json.dump(payload, output, indent=2, sort_keys=True, allow_nan=False)
            output.write('\n')
        os.replace(temporary, target)
    finally:
        if temporary and os.path.exists(temporary): os.unlink(temporary)


def _percentile(samples, fraction):
    import math
    return sorted(samples)[max(0, math.ceil(len(samples) * fraction) - 1)]


def _peak_process_rss_bytes():
    # resource is Unix-only. Keep import/help/collection portable and report an
    # unavailable measurement as unknown rather than fabricating a zero peak.
    try:
        import resource
    except ImportError:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)


def run_worker(args):
    import torch
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    import transformers

    registry, registry_hash = read_json(args.registry)
    config = Registry.model_validate(registry).get(args.models[0])
    fixture, fixture_hash = read_json(args.fixture)
    repos, queries = validate_fixture(fixture)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    load_start = time.perf_counter()
    model = SentenceTransformer(config.model_id, revision=config.revision, device='cpu',
        local_files_only=not args.allow_download, token=False, trust_remote_code=False,
        model_kwargs={'torch_dtype': torch.float32})
    load_seconds = time.perf_counter() - load_start
    native_dimension = model.get_embedding_dimension() if hasattr(model, 'get_embedding_dimension') else model.get_sentence_embedding_dimension()
    if native_dimension != config.native_dimension:
        raise ValueError('loaded model native dimension differs from pinned registry')
    if config.max_tokens > model.max_seq_length:
        raise ValueError('configured token budget exceeds loaded model context')
    poolings = [module.pooling_mode if hasattr(module, 'pooling_mode') else module.get_pooling_mode_str()
                for module in model.modules() if type(module).__name__ == 'Pooling']
    expected_pooling = 'lasttoken' if config.pooling == 'last_token' else config.pooling
    if poolings != [expected_pooling]:
        raise ValueError(f'loaded model pooling differs from registry: {poolings}')
    model.max_seq_length = config.max_tokens
    documents = ['\n'.join((repo['full_name'], repo.get('description', ''),
                  ', '.join(repo.get('topics', [])), repo.get('readme', '')))[:config.document_char_limit]
                 for repo in repos]
    query_texts = [query['query'] for query in queries]
    # Explicit warmup is excluded from measured encoding samples.
    encode_role(model, ['warmup'], config, role='document', batch_size=1)
    document_samples, query_samples = [], []
    for _ in range(args.repeats):
        started = time.perf_counter()
        documents_batch = encode_role(model, documents, config, role='document', batch_size=args.batch_size)
        document_samples.append(time.perf_counter() - started)
    query_rows = []
    for _ in range(args.repeats):
        query_rows = []
        for query in query_texts:
            started = time.perf_counter()
            encoded = encode_role(model, [query], config, role='query', batch_size=1)
            query_samples.append(time.perf_counter() - started)
            query_rows.append(encoded.values[0])
    from evaluation.embedding_registry import VectorBatch
    queries_batch = VectorBatch.create(np.asarray(query_rows), config, role='query')
    scores = compatible_scores(queries_batch, documents_batch)
    metrics = retrieval_metrics(scores, queries, [repo['repo_id'] for repo in repos], k=args.k)
    output_path = Path(args.output)
    shadow = output_path.with_name(f'{output_path.stem}-{uuid.uuid4().hex}.vectors.npz')
    metadata = {'schema_version': 1, 'embedding_version': config.version, 'configuration': config.model_dump(),
                'fixture_sha256': fixture_hash, 'repository_ids': [r['repo_id'] for r in repos],
                'queries': query_texts, 'roles': ['query', 'document']}
    with shadow.open('wb') as output:
        np.savez_compressed(output, queries=queries_batch.values, documents=documents_batch.values,
                            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    peak_rss = _peak_process_rss_bytes()
    mean_documents = sum(document_samples) / len(document_samples)
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (Path(__file__), ROOT / 'evaluation/embedding_registry.py', ROOT / 'evaluation/retrieval_metrics.py')}
    return {'schema_version': 1, 'configuration': config.model_dump(), 'embedding_version': config.version,
        'registry_sha256': registry_hash, 'fixture_sha256': fixture_hash, 'fixture_version': fixture['fixture_version'],
        'label_provenance': fixture['provenance'], 'quality_scope': 'supplied diagnostic labels; no production qualification',
        'created_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': source_hashes,
        'libraries': {'torch': torch.__version__, 'sentence_transformers': sentence_transformers.__version__, 'transformers': transformers.__version__},
        'hardware': {'device': 'cpu', 'platform': platform.system(), 'machine': platform.machine(), 'threads': args.threads,
                     'dtype': str(next(model.parameters()).dtype)},
        'sample_counts': {'repositories': len(repos), 'queries': len(queries), 'document_batches': args.repeats,
                          'query_samples': len(query_samples), 'batch_size': args.batch_size, 'warmup': 1},
        'retrieval': metrics,
        'cost': {'load_seconds': load_seconds, 'corpus_encode_seconds_mean': mean_documents,
                 'documents_per_second': len(documents) / mean_documents, 'peak_process_rss_bytes': peak_rss,
                 'query_p50_seconds': _percentile(query_samples, .5), 'query_p95_seconds': _percentile(query_samples, .95),
                 'document_encode_samples_seconds': document_samples, 'query_encode_samples_seconds': query_samples,
                 'raw_repository_vector_bytes': documents_batch.values.nbytes,
                 'shadow_artifact_bytes': shadow.stat().st_size, 'shadow_artifact': shadow.name,
                 'ann_index_bytes': None, 'index_scope': 'versioned in-memory/NPZ shadow; no ANN index built',
                 'reindex_scope': 'measured encoding of this frozen corpus only; no full catalogue extrapolation'},
        'promotion_decision': 'not promoted; genuine judged quality and serving/index migration acceptance required'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=REGISTRY)
    parser.add_argument('--fixture', type=Path, default=FIXTURE)
    parser.add_argument('--models', nargs='+', default=['minilm-384', 'qwen3-384', 'qwen3-1024'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.threads <= 8 or not 1 <= args.batch_size <= 32 or not 1 <= args.repeats <= 10 or not 1 <= args.k <= 100:
            raise ValueError('worker budget out of bounds')
        registry_payload, registry_hash = read_json(args.registry)
        registry = Registry.model_validate(registry_payload)
        for name in args.models: registry.get(name)
        if len(set(args.models)) != len(args.models): raise ValueError('model selection must be unique')
        if {args.output.resolve(), args.output.with_suffix('.vectors.npz').resolve()} & {args.registry.resolve(), args.fixture.resolve()}:
            raise ValueError('output cannot replace input evidence')
        fixture_payload, fixture_hash = read_json(args.fixture)
        validate_fixture(fixture_payload)
        if args.worker:
            if len(args.models) != 1: raise ValueError('each fresh worker loads exactly one model')
            atomic_json(args.output, run_worker(args))
        else:
            reports = []
            run_id = uuid.uuid4().hex
            for index, name in enumerate(args.models):
                child_output = args.output.with_name(f'{args.output.stem}-{run_id}-{index}.json')
                command = [sys.executable, '-m', 'scripts.compare_embedding_models', '--worker', '--registry', str(args.registry),
                    '--fixture', str(args.fixture), '--models', name, '--output', str(child_output),
                    '--threads', str(args.threads), '--batch-size', str(args.batch_size), '--repeats', str(args.repeats), '--k', str(args.k)]
                if args.allow_download: command.append('--allow-download')
                result = subprocess.run(command, timeout=600, capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError(f'{name}: worker failed; inspect isolated stderr: {result.stderr[-3000:]}')
                report = read_json(child_output)[0]
                if report.get('registry_sha256') != registry_hash or report.get('fixture_sha256') != fixture_hash:
                    raise ValueError('worker input hashes differ from validated parent inputs')
                reports.append(report)
                print(f'{name}: measured CPU worker complete', flush=True)
            atomic_json(args.output, {'schema_version': 1, 'execution': 'serial fresh CPU processes', 'models': reports,
                'promotion_decision': 'none; offline comparison cannot activate serving models'})
        return 0
    except (ValueError, OSError, KeyError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
