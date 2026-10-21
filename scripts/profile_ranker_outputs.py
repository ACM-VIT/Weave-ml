"""Network-free real-model output CPU/transfer/allocation profile (#68).

Run: OMP_NUM_THREADS=1 python -m scripts.profile_ranker_outputs --repeats 50
Includes model forward, output conversion, weighted scoring, result creation and
sorting; common input preparation is excluded. Python peaks exclude Torch-native
memory, and retained blocks are not cumulative allocation traffic. GPU is tested
only when CUDA is actually available; no production latency claim is implied.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from unittest.mock import patch

import numpy as np
import torch

from inference.feature_spec import INPUT_DIM
from inference.ranker_service import MMoEHeavyRanker, RankerService
from inference.value_function import compute_value_score

_FIELDS = ('p_ctr', 'p_save', 'p_gh', 'pred_dwell_fraction', 'p_follow')


def baseline(service, inputs, candidates, skills):
    with torch.no_grad():
        outputs = service.model(inputs)
        columns = [np.clip(value.reshape(-1).cpu().numpy(), 0, 1) for value in outputs]
    if not np.all(np.isfinite(np.column_stack(columns))):
        raise ValueError('nonfinite predictions')
    results = []
    for index, repo in enumerate(candidates):
        predictions = dict(zip(_FIELDS, (float(column[index]) for column in columns)))
        score = float(compute_value_score(predictions))
        if not np.isfinite(score):
            raise ValueError('nonfinite score')
        results.append({'repo_id': repo['id'], 'final_score': score,
                        'skill_match': float(skills[index]), 'predictions': predictions})
    return sorted(results, key=lambda result: result['final_score'], reverse=True)


def measure(build, repeats):
    for _ in range(3):
        build()
    wall, cpu = [], []
    for _ in range(repeats):
        wall_start, cpu_start = time.perf_counter_ns(), time.process_time_ns()
        build()  # .cpu() synchronizes output completion on CUDA.
        cpu.append(time.process_time_ns() - cpu_start)
        wall.append(time.perf_counter_ns() - wall_start)
    original_cpu = torch.Tensor.cpu
    transfers = []
    def transfer(tensor, *args, **kwargs):
        transfers.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)
    with patch.object(torch.Tensor, 'cpu', transfer):
        build()
    tracemalloc.start()
    result = build()
    _, peak = tracemalloc.get_traced_memory()
    blocks = sum(stat.count for stat in tracemalloc.take_snapshot().statistics('filename'))
    tracemalloc.stop()
    return {'median_wall_us': statistics.median(wall) / 1000,
            'median_cpu_us': statistics.median(cpu) / 1000,
            'output_cpu_calls': len(transfers), 'output_transfer_shapes': transfers,
            'peak_python_bytes': peak, 'retained_python_blocks': blocks}, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=50)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    torch.manual_seed(68)
    rows = []
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    for device in devices:
        service = RankerService.__new__(RankerService)
        service.device = torch.device(device)
        service.model = MMoEHeavyRanker(INPUT_DIM).to(device).eval()
        for count in (1, 150, 2000):
            inputs = torch.randn(count, INPUT_DIM, device=device)
            candidates = [{'id': str(index)} for index in range(count)]
            skills = np.full(count, 0.5, dtype=np.float32)
            before, expected = measure(lambda: baseline(service, inputs, candidates, skills), args.repeats)
            diagnostic, detailed = measure(lambda: service._rank_results(inputs, candidates, skills), args.repeats)
            compact, actual = measure(lambda: service._rank_results(inputs, candidates), args.repeats)
            assert detailed == expected
            assert actual == [{'repo_id': row['repo_id'], 'final_score': row['final_score']}
                              for row in expected]
            rows.append({'device': device, 'candidates': count, 'baseline': before,
                         'diagnostic': diagnostic, 'compact': compact})
    print(json.dumps({'repeats': args.repeats, 'cuda_available': torch.cuda.is_available(),
                     'gpu_limit': None if torch.cuda.is_available() else 'No GPU available; GPU timing/transfer behavior not measured.',
                     'results': rows}, indent=2))


if __name__ == '__main__':
    main()
