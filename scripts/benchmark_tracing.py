"""Local tracing/scrape microbenchmark; no model, network, or private fixtures.

Run: python -m scripts.benchmark_tracing --iterations 20000 --scrapes 100
Baseline executes the same synthetic arithmetic and empty /metrics renderer.
Enabled runs add real INTERNAL spans under one unsampled/sampled SERVER scope.
The scrape phase populates the full finite stage-label space. Each mode runs in
an isolated child process; Linux ru_maxrss is peak KiB, including imports.
These synthetic costs do not predict service throughput or production latency.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import resource
import subprocess
import sys
import time


def measure(mode: str, iterations: int, scrapes: int) -> dict:
    from api.metrics import render_prometheus
    from utils import observability as obs

    enabled = mode != "baseline"
    flags = "01" if mode == "sampled" else "00"
    parent = obs.span_scope(
        "search", server=True,
        authenticated_traceparent="00-1234567890abcdef1234567890abcdef-1234567890abcdef-" + flags,
    ) if enabled else nullcontext()
    cpu = time.process_time()
    wall = time.perf_counter()
    checksum = 0
    with parent:
        for index in range(iterations):
            scope = obs.span_scope("search", stage="qdrant") if enabled else nullcontext()
            with scope:
                checksum += index % 17
    trace_cpu = time.process_time() - cpu
    trace_wall = time.perf_counter() - wall
    obs.reset_stage_histograms_for_tests()
    if enabled:
        # The maximum series space is a conservative scrape-size fixture,
        # including valid combinations not normally emitted by real boundaries.
        for operation in sorted(obs._OPERATIONS):
            for stage in sorted(obs._STAGES):
                for outcome in sorted(obs._OUTCOMES):
                    obs.observe_duration(operation, stage, outcome, 0.1)
    cpu = time.process_time()
    wall = time.perf_counter()
    for _ in range(scrapes):
        rendered = render_prometheus({})
    return {
        "mode": mode, "iterations": iterations, "scrapes": scrapes,
        "trace_wall_seconds": trace_wall, "trace_cpu_seconds": trace_cpu,
        "scrape_wall_seconds": time.perf_counter() - wall,
        "scrape_cpu_seconds": time.process_time() - cpu,
        "scrape_bytes": len(rendered.encode()),
        "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "stage_label_combinations": len(obs._OPERATIONS) * len(obs._STAGES) * len(obs._OUTCOMES),
        "max_stage_series": len(obs._OPERATIONS) * len(obs._STAGES) * len(obs._OUTCOMES) * (len(obs._BUCKETS) + 3),
        "checksum": checksum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--scrapes", type=int, default=100)
    parser.add_argument("--worker", choices=("baseline", "unsampled", "sampled"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 1000000 or not 1 <= args.scrapes <= 10000:
        parser.error("iterations must be 1..1000000 and scrapes 1..10000")
    if args.worker:
        print(json.dumps(measure(args.worker, args.iterations, args.scrapes)))
        return
    results = []
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for mode in ("baseline", "unsampled", "sampled"):
        output = subprocess.check_output([
            sys.executable, "-m", "scripts.benchmark_tracing", "--worker", mode,
            "--iterations", str(args.iterations), "--scrapes", str(args.scrapes),
        ], text=True, env=environment)
        results.append(json.loads(output))
    print(json.dumps({"scope": "synthetic_local_microbenchmark_not_production", "native_thread_limit": 1, "results": results}, indent=2))


if __name__ == "__main__":
    main()
