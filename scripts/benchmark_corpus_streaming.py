"""Offline synthetic comparison: real orchestration/encoding, fake GitHub/HTTP.

Run each mode in a fresh process, from the repository root:
  uv run python scripts/benchmark_corpus_streaming.py --mode materialized
  uv run python scripts/benchmark_corpus_streaming.py --mode stream
No credentials, external services, shared checkpoints or network are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import tempfile
import time
from types import SimpleNamespace
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from acquisition.backend_client import BackendIngestionClient
from acquisition.config import CorpusPipelineSettings
from acquisition.corpus_pipeline import CorpusPipeline
from acquisition.pipeline import enrich_repositories
from acquisition.repository_enricher import RepositoryEnricher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['materialized', 'stream'], required=True)
    parser.add_argument('--records', type=int, default=600)
    parser.add_argument('--readme-chars', type=int, default=60_000)
    args = parser.parse_args()
    if args.records < 1 or not 1 <= args.readme_chars <= 1_000_000:
        parser.error('records must be positive and readme chars within 1..1000000')
    names = [f'fixture/repo{i}' for i in range(args.records)]
    calls = 0
    peak_body_bytes = 0
    first_persist = None
    acquired_at_first_persist = None
    enriched = 0
    started = time.perf_counter()

    class Response:
        status_code = 200
        def __init__(self, body): self.body = body
        def raise_for_status(self): pass
        def json(self): return self.body

    class Session:
        def post(self, _url, *, data, **_kwargs):
            nonlocal calls, peak_body_bytes, first_persist, acquired_at_first_persist
            calls += 1
            peak_body_bytes = max(peak_body_bytes, len(data))
            records = json.loads(data)['repositories']
            time.sleep(.005)  # Same deterministic HTTP-boundary delay in both modes.
            if first_persist is None:
                first_persist = time.perf_counter() - started
                acquired_at_first_persist = enriched
            return Response({'mappings': [dict(github_id=r['github_id'],
                repo_id=str(uuid.uuid5(uuid.NAMESPACE_URL, r['full_name'])), content_version=1)
                for r in records]})

    def enrich(_self, batch):
        nonlocal enriched
        time.sleep(.003 * len(batch))
        result = []
        for name in batch:
            number = int(name.removeprefix('fixture/repo')) + 1
            markdown = f'# {name}\n' + 'A' * args.readme_chars
            result.append(SimpleNamespace(repo_id=name, warnings=[],
                payload=dict(full_name=name, github_id=str(number), github_node_id=f'R_{number}',
                             owner_github_id='1', readme=markdown)))
        enriched += len(result)
        return result

    RepositoryEnricher.get_repositories_batch = enrich
    def acquire(*, on_batch=None, on_targets=None, **kwargs):
        if on_targets: on_targets(names)
        return enrich_repositories('fixture-token', names, batch_size=kwargs['batch_size'],
                                   workers=kwargs['workers'], on_batch=on_batch)
    backend = BackendIngestionClient(base_url='http://fixture.invalid', internal_secret='fixture', session=Session())
    backend.manifest_status = 'empty'
    with tempfile.TemporaryDirectory(prefix='weave-corpus-benchmark-') as directory:
        pipeline = CorpusPipeline(database=backend, acquire=acquire, acquire_retries=lambda _: [],
            quality_filter=lambda values, **_: (values, []), indexer=lambda _: [],
            settings=CorpusPipelineSettings(target_count=args.records, max_cycles=1,
                                             checkpoint_path=Path(directory) / 'state.json'),
            indexing_enabled=False, streaming=args.mode == 'stream')
        report = pipeline.run(limit=args.records, batch_size=15, workers=4, min_readme_chars=1)
    elapsed = time.perf_counter() - started
    assert report.persisted == args.records
    print(json.dumps(dict(mode=args.mode, records=args.records, readme_chars=args.readme_chars,
        first_persist_seconds=first_persist, enriched_at_first_persist=acquired_at_first_persist,
        duration_seconds=elapsed, records_per_second=args.records / elapsed,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        http_calls=calls, peak_http_body_bytes=peak_body_bytes), sort_keys=True))


if __name__ == '__main__':
    main()
