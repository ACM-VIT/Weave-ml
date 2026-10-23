"""Replay current v2 README-tower reuse on bounded genuine local git snapshots.

No checkout, network, GitHub metadata guesses, summary provider or storage writes.
The ratio concerns README inputs in this exact cohort, not production job mix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time

import numpy as np
from config import EMBEDDING_MODEL_REVISION, REPOSITORY_EMBEDDING_DIM, REPOSITORY_EMBEDDING_MODEL
from embedding.embeddings import SentenceTransformerEmbedder, aggregate_vectors, chunk_text
from embedding.repository_embedding import RepositoryEmbeddingConfig, build_readme_text
from embedding.tower_cache import TowerEmbeddingCache


class ControlEncoder:
    """Deterministic vectors for cache equivalence/counting, not model quality."""
    max_seq_length = 256
    def __init__(self):
        self.texts = 0
    def encode(self, texts, **kwargs):
        self.texts += len(texts)
        rows = []
        for text in texts:
            seed = hashlib.sha256(text.encode()).digest()
            vector = np.asarray([seed[index % len(seed)] + 1 for index in range(REPOSITORY_EMBEDDING_DIM)], dtype=np.float32)
            rows.append(vector / np.linalg.norm(vector))
        return np.asarray(rows)


def git(repository, *args, input=None):
    # Git subprocess transports bypass Python socket guards. Missing promisor
    # objects must fail locally, not trigger a hidden partial-clone fetch.
    # Unsupported Git versions reject the option rather than trying a fetch.
    try:
        return subprocess.run(['git', '--no-lazy-fetch', '--no-pager', '-C', str(repository), *args], input=input,
            env={**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_TERMINAL_PROMPT': '0'},
            check=True, capture_output=True, timeout=15).stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError('local-only Git read failed; required objects must already exist (lazy fetching disabled)') from exc


def snapshots(repository, *, ref, limit=40, path='README.md', changes_only=False):
    if type(limit) is not int or not 1 <= limit <= 200:
        raise ValueError('history limit must be between 1 and 200')
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or '..' in candidate.parts or not candidate.name.lower().startswith('readme'):
        raise ValueError('only relative README paths may be sampled')
    repository = Path(repository).resolve()
    frozen = git(repository, 'rev-parse', '--verify', '--end-of-options', f'{ref}^{{commit}}').decode().strip()
    args = ['log', '--first-parent', f'-{limit}', '--format=%H', frozen]
    if changes_only:
        args += ['--', path]
    commits = git(repository, *args).decode().splitlines()[::-1]
    rows, skipped = [], []
    for commit in commits:
        entries = git(repository, 'ls-tree', '-z', commit, '--', path).split(b'\0')
        entry = next((value for value in entries
                      if value and value.split(b'\t', 1)[1].decode('utf-8') == path), None)
        if entry is None:
            skipped.append({'commit': commit, 'reason': 'README missing'})
            continue
        mode, kind, object_id = entry.split(b'\t', 1)[0].split()
        if mode not in {b'100644', b'100755'} or kind != b'blob':
            skipped.append({'commit': commit, 'reason': 'README is not a regular file'})
            continue
        blob = object_id.decode('ascii')
        size = int(git(repository, 'cat-file', '-s', blob))
        if size > 1_000_000:
            skipped.append({'commit': commit, 'reason': 'README exceeds bounded input'})
            continue
        try:
            text = git(repository, 'cat-file', 'blob', blob).decode('utf-8')
        except UnicodeDecodeError:
            skipped.append({'commit': commit, 'reason': 'README is not UTF-8'})
            continue
        rows.append({'commit': commit, 'blob': blob, 'path': path, 'raw_bytes': size,
                     'commit_time': git(repository, 'show', '-s', '--format=%cI', commit).decode().strip(),
                     'text': text})
    return frozen, rows, skipped


def replay(rows, *, model=None, max_entries=768):
    model = model if model is not None else ControlEncoder()
    config = RepositoryEmbeddingConfig(version='repo-embedding-v2')
    embedder = SentenceTransformerEmbedder(model=model)
    cache = TowerEmbeddingCache(model=model, model_revision=EMBEDDING_MODEL_REVISION,
                               max_entries=max_entries)
    records, max_error, baseline_texts, reused_texts = [], 0., 0, 0
    started = time.perf_counter()
    for row in rows:
        texts = [part.text for part in chunk_text(build_readme_text({'readme': row['text']}),
                    max_chars=config.readme_chunk_chars, overlap_chars=config.readme_chunk_overlap_chars)]
        record = {name: value for name, value in row.items() if name != 'text'}
        if not texts:
            record.update(empty=True, reused=False, chunk_count=0)
            records.append(record)
            continue
        key = cache.fingerprint(tower='readme', texts=texts, config=config)
        # Cold recomputation is an independent numerical reference, not a
        # fabricated cache outcome. Neither path modifies canonical content.
        cold = aggregate_vectors(embedder.embed_texts(texts))
        baseline_texts += len(texts)
        cached = cache.get(key, expected_size=config.embedding_dim)
        reused = cached is not None
        if cached is None:
            cached = aggregate_vectors(embedder.embed_texts(texts))
            cache.put(key, cached, expected_size=config.embedding_dim)
        else:
            reused_texts += len(texts)
        error = float(np.max(np.abs(np.asarray(cold) - np.asarray(cached))))
        max_error = max(max_error, error)
        if not np.allclose(cold, cached, rtol=1e-6, atol=1e-6):
            raise AssertionError('reused README tower differs from current cold encoding')
        record.update(empty=False, reused=reused, chunk_count=len(texts), fingerprint=key.hex())
        records.append(record)
    nonempty = sum(not row['empty'] for row in records)
    hits = sum(row['reused'] for row in records)
    return {'snapshots': len(records), 'nonempty_snapshots': nonempty,
        'readme_tower_reuses': hits, 'readme_tower_reuse_ratio': hits / nonempty if nonempty else None,
        'baseline_chunk_encodes': baseline_texts, 'chunk_encodes_avoided_by_reuse': reused_texts,
        'cold_vs_reused_max_absolute_error': max_error, 'elapsed_seconds': time.perf_counter() - started,
        'cache': cache.snapshot(), 'records': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--ref', required=True)
    parser.add_argument('--limit', type=int, default=40)
    parser.add_argument('--path', default='README.md')
    parser.add_argument('--changes-only', action='store_true')
    parser.add_argument('--cached-real-model', action='store_true')
    args = parser.parse_args()
    frozen, rows, skipped = snapshots(args.repo, ref=args.ref, limit=args.limit,
                                     path=args.path, changes_only=args.changes_only)
    model = None
    if args.cached_real_model:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(1)
        model = SentenceTransformer(REPOSITORY_EMBEDDING_MODEL, revision=EMBEDDING_MODEL_REVISION,
                                    local_files_only=True, device='cpu')
    report = replay(rows, model=model)
    report.update(scope=__doc__, frozen_commit=frozen, ref=args.ref, path=args.path,
        cohort='README-changing first-parent commits' if args.changes_only else 'consecutive first-parent commits',
        encoder='cached pinned MiniLM CPU' if model is not None else 'deterministic control encoder',
        model_contract={'model': REPOSITORY_EMBEDDING_MODEL, 'revision': EMBEDDING_MODEL_REVISION,
                        'dimensions': REPOSITORY_EMBEDDING_DIM, 'projection': 'current reserved v2'},
        skipped=skipped, unknown_historical_fields=['GitHub topics', 'GitHub stars', 'GitHub description',
            'backend content versions', 'actual ingestion-job selection', 'deployed model/cache lifetime'],
        limits='Current v2 replay on genuine README snapshots only. No full-payload, historical runtime or representative production skip ratio. No historical GitHub facts are inferred; metadata/topics/summary are outside this sample.')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
