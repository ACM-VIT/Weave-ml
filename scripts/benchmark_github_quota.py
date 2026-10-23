"""Injected-throttle acceptance for the actual GitHub client and README scheduler.

uv run python -m scripts.benchmark_github_quota
The clock and HTTP session are fixtures. Reported virtual throughput is not live
GitHub throughput; no credentials, network, repository writes or models are used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import weakref

from acquisition import github_rate_limit as quota
from acquisition import github_graphql_client as graphql
from acquisition import readme_scheduler as scheduling


class Clock:
    def __init__(self):
        self.lock = threading.Lock()
        self.now = 0.
        self.worker_sleep_calls = 0

    def monotonic(self):
        with self.lock:
            return self.now

    def wall(self):
        return 1000 + self.monotonic()

    def advance(self, seconds):
        with self.lock:
            self.now += seconds

    def forbidden_worker_sleep(self, seconds):
        self.worker_sleep_calls += 1
        raise AssertionError('scheduler retry must not sleep in its transport worker')


@contextmanager
def fixture_clock(clock):
    fake_time = SimpleNamespace(monotonic=clock.monotonic, time=clock.wall,
                                sleep=clock.forbidden_worker_sleep)
    with ExitStack() as stack:
        for module in (quota, graphql, scheduling):
            stack.enter_context(patch.object(module, 'time', fake_time))
        stack.enter_context(patch.object(quota.random, 'uniform', return_value=.1))
        stack.enter_context(patch.object(quota, '_registry', weakref.WeakValueDictionary()))
        stack.enter_context(patch.object(quota, '_retained_scopes', {}))
        yield


class Transport:
    """Two-stage metadata/blob reads; one injected blob429 or permanent errors."""

    def __init__(self, clock, *, mode='once'):
        self.clock, self.mode = clock, mode
        self.lock = threading.Lock()
        self.records = []
        self.throttled = False
        self.coordinators = []
        self.sessions = []

    def post(self, url, *, json, timeout):
        stage, name = json['query'], json['variables']['name']
        with self.lock:
            status = 404 if self.mode == 'permanent' else 200
            if stage == 'blob' and (self.mode == 'always' or not self.throttled):
                status = 429
                self.throttled = True
            self.records.append({'stage': stage, 'name': name, 'at': self.clock.monotonic(), 'status': status})
        headers = {'X-RateLimit-Remaining': '100', 'X-RateLimit-Reset': '1060'}
        if status == 429:
            headers = {'Retry-After': '5', 'X-RateLimit-Remaining': '0',
                       'X-RateLimit-Reset': str(self.clock.wall() + 5)}
        return SimpleNamespace(status_code=status, headers=headers, text='fixture response',
            json=lambda: {'data': {'text': '# Canonical fixture README'}})

    def session(self):
        transport = self
        class Session:
            def __init__(self):
                self.headers, self.auth, self.closed = {}, None, False
            def post(self, *args, **kwargs):
                return transport.post(*args, **kwargs)
            def close(self):
                self.closed = True
        session = Session()
        self.sessions.append(session)
        return session

    def client(self):
        transport = self
        class Client(graphql.GitHubGraphQLClient):
            def get_readme(self, owner, name):
                self.execute('metadata', {'name': name})
                result = self.execute('blob', {'name': name})
                return graphql.FetchedReadme(result['data']['text'], source_path='README.md', default_branch='main')
        client = Client(token='isolated-fixture-token', base_url='https://fixture.invalid/graphql',
                        session=transport.session(), max_retries=2)
        self.coordinators.append(client.rate_limits)
        return client


def until(predicate, *, seconds=3):
    stop = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= stop:
            raise AssertionError('fixture did not reach expected scheduling state')
        time.sleep(.001)


def advance_until(future, clock, scheduler, *, seconds=5):
    samples = []
    stop = time.monotonic() + seconds
    while not future.done():
        if time.monotonic() >= stop:
            raise AssertionError('injected workload did not settle')
        samples.append(scheduler.snapshot())
        clock.advance(.05)
        with scheduler._condition:
            scheduler._condition.notify_all()
        time.sleep(.001)  # The driver yields; transport workers never sleep.
    return samples


def measure(*, jobs=8, workers=2):
    clock = Clock()
    transport = Transport(clock)
    began = time.perf_counter()
    with fixture_clock(clock):
        with scheduling.ReadmeScheduler(transport.client, workers) as scheduler:
            with ThreadPoolExecutor(max_workers=1) as producer:
                def consume():
                    return [future.result() for future in scheduler.ordered(
                        ('fixture', f'repository-{index}') for index in range(jobs))]
                result = producer.submit(consume)
                samples = advance_until(result, clock, scheduler)
                values = result.result()
            final = scheduler.snapshot()
        coordinator = transport.coordinators[0].snapshot()
        shared = all(value is transport.coordinators[0] for value in transport.coordinators)
    elapsed = time.perf_counter() - began
    throttles = [r for r in transport.records if r['status'] == 429]
    first_reset = throttles[0]['at'] + 5
    after_reset = sorted(r['at'] for r in transport.records if r['at'] >= first_reset)
    metadata_counts = {name: sum(r['name'] == name and r['stage'] == 'metadata' for r in transport.records)
                       for name in {r['name'] for r in transport.records}}
    return {'scope': __doc__, 'jobs': jobs, 'workers': workers, 'completed_jobs': len(values),
        'wall_seconds': elapsed, 'virtual_seconds': clock.monotonic(),
        'virtual_useful_jobs_per_second': len(values) / clock.monotonic(),
        'transport_requests': len(transport.records), 'injected_429s': len(throttles),
        'max_metadata_attempts_per_job': max(metadata_counts.values()),
        'maximum_admitted_jobs': max(s['admitted'] for s in samples),
        'maximum_queued_jobs': max(s['queued'] for s in samples),
        'maximum_active_workers': max(s['active'] for s in samples),
        'observed_queued_with_no_active_worker': any(s['queued'] and not s['active'] for s in samples),
        'transport_worker_sleep_calls': clock.worker_sleep_calls,
        'minimum_post_reset_request_gap_seconds': min((b - a for a, b in zip(after_reset, after_reset[1:])), default=None),
        'all_worker_clients_share_quota': shared, 'coordinator': coordinator,
        'settled_scheduler': final, 'all_worker_sessions_closed': all(s.closed for s in transport.sessions),
        'requests': transport.records,
        'limits': 'One ingestion process, synthetic time and stubbed HTTP. Other processes/accounts/consumers may consume the same real quota. Virtual throughput is a scheduler diagnostic, not API or production performance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if not 4 <= args.jobs <= 100 or not 1 <= args.workers <= 4:
        parser.error('jobs4..100 and workers1..4 required')
    print(json.dumps(measure(jobs=args.jobs, workers=args.workers), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
