"""Local synthetic query-cache cost/reuse profile, not a production speedup.

Run with: python -m scripts.profile_query_embeddings --requests 400
No services, model downloads, credentials or real user queries are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
import tracemalloc

from config import REPOSITORY_EMBEDDING_DIM
from embedding.embeddings import SentenceTransformerEmbedder
from embedding.query_cache import CachedQueryEmbedder
from embedding.runtime import BoundedEmbeddingModel
from embedding.vector_contract import validate_embedding_vector


class SyntheticEncoder:
    """Deterministic CPU work plus vector construction, not a transformer."""

    def __init__(self) -> None:
        self.calls = 0
        self.seconds = 0.

    def encode(self, texts, **kwargs):
        began = time.perf_counter()
        vectors = []
        for text in texts:
            seed = hashlib.pbkdf2_hmac('sha256', text.encode(), b'local-query-cache-profile', 2_000)
            vectors.append([seed[i % len(seed)] / 255. for i in range(REPOSITORY_EMBEDDING_DIM)])
            self.calls += 1
        self.seconds += time.perf_counter() - began
        return vectors


def measure(queries: list[str], *, cached: bool) -> dict:
    tracemalloc.start()
    encoder = SyntheticEncoder()
    model = BoundedEmbeddingModel(encoder, max_concurrency=1)
    embedder = CachedQueryEmbedder(
        model=model, cache=model.query_embedding_cache,
        model_revision=model.query_embedding_revision,
        embedding_version='synthetic-profile-v1', expected_size=REPOSITORY_EMBEDDING_DIM,
    ) if cached else SentenceTransformerEmbedder(model=model)
    began = time.perf_counter()
    checksum = hashlib.sha256()
    for query in queries:
        vector = validate_embedding_vector(
            embedder.embed_text(query), expected_size=REPOSITORY_EMBEDDING_DIM,
        )
        # Frame and hash values in request/coordinate order; scalar sums
        # cannot detect swapped requests or reordered coordinates.
        checksum.update(struct.pack('!I', len(vector)))
        checksum.update(struct.pack(f'!{len(vector)}d', *vector))
    seconds = time.perf_counter() - began
    snapshot = model.query_embedding_cache.snapshot()
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        'requests': len(queries), 'encoder_calls': encoder.calls,
        'cache_hit_rate': snapshot['hits'] / len(queries) if cached else 0.,
        'workload_seconds': seconds, 'encoder_seconds': encoder.seconds,
        'tracemalloc_retained_bytes': retained, 'tracemalloc_peak_bytes': peak,
        'cache': snapshot, 'vector_checksum': checksum.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=int, default=400)
    args = parser.parse_args()
    if not 10 <= args.requests <= 100_000:
        parser.error('--requests must be between 10 and 100000')
    output = {'scope': 'Local deterministic synthetic CPU encoder; no production inference or speedup claim.',
              'dimension': REPOSITORY_EMBEDDING_DIM,
              'vector_checksum_algorithm': 'sha256-length-framed-big-endian-float64',
              'workloads': {}}
    for name, queries in (
        ('repeat', [f'synthetic-query-{i % max(1, args.requests // 10)}' for i in range(args.requests)]),
        ('unique', [f'synthetic-query-{i}' for i in range(args.requests)]),
    ):
        baseline = measure(queries, cached=False)
        cached = measure(queries, cached=True)
        if baseline['vector_checksum'] != cached['vector_checksum']:
            raise AssertionError('cached and baseline vectors differ')
        output['workloads'][name] = {
            'baseline': baseline, 'cached': cached,
            'encoder_seconds_saved_measured': baseline['encoder_seconds'] - cached['encoder_seconds'],
            'workload_seconds_delta': cached['workload_seconds'] - baseline['workload_seconds'],
            'tracemalloc_peak_bytes_delta': cached['tracemalloc_peak_bytes'] - baseline['tracemalloc_peak_bytes'],
        }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
