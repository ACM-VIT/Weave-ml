"""Network-free issue #86 profile: python -m scripts.profile_summary_preprocessing."""
from __future__ import annotations

import argparse
import gc
import json
import time
import tracemalloc

from summarization.pipeline import CardSummaryPipeline, description_fallback_artifact
from summarization.source import build_summary_source


def measure(action, repeats):
    # Input generation and imports are deliberately outside both measurements.
    gc.collect()
    # Windows process CPU clocks have coarse ticks. Collect at least 100 ms
    # of CPU so fast rejection does not misleadingly report zero cost.
    started = time.process_time()
    iterations = 0
    batch_size = repeats
    while iterations < 100_000:
        batch_size = min(batch_size, 100_000 - iterations)
        for _ in range(batch_size):
            action()
        iterations += batch_size
        # Read the clock between batches, amortizing clock overhead even for
        # microsecond rejection paths. Keep a hard cap for coarse/stalled clocks.
        elapsed = time.process_time() - started
        if elapsed >= 0.1:
            break
        batch_size = min(batch_size * 2, 4_096)
    cpu = elapsed / iterations
    gc.collect()
    tracemalloc.start()
    try:
        result = action()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, {'cpu_seconds_per_call': cpu, 'cpu_iterations': iterations,
                    'cpu_target_reached': elapsed >= 0.1, 'traced_peak_bytes': peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100:
        parser.error('--repeats must be between 1 and 100')
    rows = []
    for shape in ('prose', 'unclosed-markup', 'many-lines'):
        unit = {'prose': 'A workspace helps maintainers coordinate shared projects. ',
                'unclosed-markup': '[label]((', 'many-lines': 'x\n'}[shape]
        for size in (16_000, 64_000, 1_000_000, 4_000_000):
            raw = (unit * (size // len(unit) + 1))[:size]
            repo = {'full_name': 'weave/profile', 'description': 'Shared review tools.', 'readme': raw}
            material, analysis = measure(lambda: build_summary_source(repo, repo, max_chars=1_000), args.repeats)
            pipeline = CardSummaryPipeline()
            artifact, disabled = measure(lambda: pipeline.summarize(repo, repo), args.repeats)
            assert artifact.model_dump() == description_fallback_artifact(repo).model_dump()
            assert repo['readme'] is raw
            rows.append({'shape': shape, 'source_chars': size, 'analysis': analysis,
                         'analysis_limit': material.analysis_limit,
                         'prompt_chars': len(material.prompt_input), 'provider_disabled': disabled})
    print(json.dumps({'repeats': args.repeats, 'measurements': rows}, indent=2))


if __name__ == '__main__':
    main()
