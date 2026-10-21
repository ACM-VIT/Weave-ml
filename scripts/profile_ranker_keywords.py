"""Network-free feature CPU/allocation comparison for Weave-ml #65.

Run with: python -m scripts.profile_ranker_keywords --repeats 100
Comparable median CPU times exclude tracemalloc. Cold CPU is explicitly traced;
traced peaks include output lists and cold-cache entries but not fixtures.
Allocation counts are retained Python blocks, not cumulative allocation traffic.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc

from inference.ranker_service import (
    _cached_repository_keywords,
    _keyword_match_score,
    _keywords,
    _repository_keywords,
)


def baseline(skills, candidates):
    scores = []
    for repo in candidates:
        user = _keywords(skills)
        keywords = (_keywords(repo['languages']) | _keywords(repo['topics'])
                    | _keywords(repo['tags']))
        scores.append(len(user & keywords) / len(user) if user else 0.0)
    return scores


def optimized(skills, candidates):
    user = _keywords(skills)
    return [
        _keyword_match_score(user, _repository_keywords(
            repo['languages'], repo['topics'], repo['tags'])) if user else 0.0
        for repo in candidates
    ]


def measure(build, skills, candidates, repeats):
    samples = []
    for _ in range(repeats):
        start = time.process_time_ns()
        build(skills, candidates)
        samples.append(time.process_time_ns() - start)
    tracemalloc.start()
    scores = build(skills, candidates)
    _, peak = tracemalloc.get_traced_memory()
    snapshot = tracemalloc.take_snapshot()
    retained_blocks = sum(stat.count for stat in snapshot.statistics('filename'))
    tracemalloc.stop()
    return {'median_cpu_us': statistics.median(samples) / 1000,
            'peak_python_bytes': peak, 'retained_python_blocks': retained_blocks}, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=100)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    skills = [f' Skill-{index} ' for index in range(30)] + ['STRASSE', 'Σ']
    rows = []
    for count in (15, 150, 500):
        candidates = [
            {'languages': [' Python ', f'Skill-{index % 30}'],
             'topics': ['Straße', 'ς'] + [f'topic-{i}' for i in range(12)],
             'tags': [f'tag-{i}' for i in range(8)]}
            for index in range(count)
        ]
        _cached_repository_keywords.cache_clear()
        before, expected = measure(baseline, skills, candidates, args.repeats)
        # Warmed content reuse is measured separately from the first cold batch.
        tracemalloc.start()
        start = time.process_time_ns()
        actual = optimized(skills, candidates)
        cold_cpu_us = (time.process_time_ns() - start) / 1000
        _, cold_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        after, warm = measure(optimized, skills, candidates, args.repeats)
        assert actual == expected == warm
        rows.append({'candidates': count, 'baseline': before, 'warm_cache': after,
                     'cold_cache_traced_cpu_us': cold_cpu_us,
                     'cold_cache_peak_python_bytes': cold_peak,
                     'cache': _cached_repository_keywords.cache_info()._asdict()})
    print(json.dumps({'repeats': args.repeats, 'results': rows}, indent=2))


if __name__ == '__main__':
    main()
