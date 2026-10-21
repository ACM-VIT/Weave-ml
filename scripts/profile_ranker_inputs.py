"""Network-free input-assembly CPU, Python allocations and process RSS (#66).

Run: python -m scripts.profile_ranker_inputs --repeats 100
Each mode/size runs in a fresh process. Dense feature computation is common and
excluded; both paths retain embedding/assembled-input validation. Tracemalloc
excludes Torch-native storage, and retained blocks are not total allocation
traffic. RSS is whole-process high water, including imports and fixtures.
"""
from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc

import numpy as np
import torch

from inference.feature_spec import EMBEDDING_DIM, FEATURE_COUNT, INPUT_DIM
from inference.ranker_service import RankerService


def baseline(service, user, candidates, dense):
    users = np.tile(user, (len(candidates), 1))
    repos = np.vstack([service._embedding(repo['embedding'], field_name='repo')
                       for repo in candidates])
    combined = np.hstack((users, repos, dense))
    if combined.shape != (len(candidates), service.input_dim):
        raise ValueError('wrong shape')
    if not np.all(np.isfinite(combined)):
        raise ValueError('nonfinite input')
    return torch.tensor(combined, dtype=torch.float32).to(service.device)


def measure(mode, count, repeats):
    service = RankerService.__new__(RankerService)
    service.emb_dim = EMBEDDING_DIM
    service.input_dim = INPUT_DIM
    service.device = torch.device('cpu')
    user = np.arange(EMBEDDING_DIM * 2, dtype=np.float32)[::2] / 1000
    candidates = [{'id': str(i), 'embedding': np.roll(user, i)[::-1]}
                  for i in range(count)]
    dense = np.full((count, FEATURE_COUNT), 0.25, dtype=np.float32)
    build = (lambda: baseline(service, user, candidates, dense)) if mode == 'baseline' else (
        lambda: service._input_tensor(user, candidates, dense))
    actual = build()
    del actual
    samples = []
    for _ in range(repeats):
        start = time.process_time_ns()
        build()
        samples.append(time.process_time_ns() - start)
    tracemalloc.start()
    tensor = build()
    _, peak = tracemalloc.get_traced_memory()
    blocks = sum(stat.count for stat in tracemalloc.take_snapshot().statistics('filename'))
    tracemalloc.stop()
    result = {'mode': mode, 'candidates': count,
            'median_cpu_us': statistics.median(samples) / 1000,
            'peak_python_bytes': peak, 'retained_python_blocks': blocks,
            'input_tensor_bytes': tensor.numel() * tensor.element_size(),
            'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    # Parity checking follows all measurements, so reference allocations cannot
    # inflate the contiguous path's RSS high water.
    expected = baseline(service, user, candidates, dense)
    torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--mode', choices=('baseline', 'contiguous'))
    parser.add_argument('--count', type=int, default=150)
    args = parser.parse_args()
    if args.repeats < 1 or args.count < 1:
        parser.error('repeats and count must be positive')
    if args.mode:
        print(json.dumps(measure(args.mode, args.count, args.repeats)))
        return
    rows = []
    for count in (15, 150, 2000):
        for mode in ('baseline', 'contiguous'):
            output = subprocess.check_output([
                sys.executable, '-m', 'scripts.profile_ranker_inputs', '--mode', mode,
                '--count', str(count), '--repeats', str(args.repeats)], text=True)
            rows.append(json.loads(output))
    print(json.dumps({'repeats': args.repeats, 'results': rows}, indent=2))


if __name__ == '__main__':
    main()
