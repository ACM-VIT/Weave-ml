"""Shared retrieval I/O, real candidate path, and actual worker ownership."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from retrieval import runtime as io
from scripts.benchmark_retrieval_io import (
    FixtureQdrant, VECTOR, isolated_runtimes, measure_abandonment,
    measure_requests, retriever, until, validate_control,
)
from utils.deadline import CooperativeDeadlineExceeded, RequestDeadline, deadline_scope


def test_concurrent_recommendations_share_one_aggregate_rpc_budget():
    release = threading.Event()
    client = FixtureQdrant(hook=lambda stage: release.wait(5))
    with isolated_runtimes(workers=2, capacity=3):
        engine = retriever(client)
        with ThreadPoolExecutor(max_workers=4) as callers:
            futures = []
            try:
                futures = [callers.submit(engine.recommend_batch, client.user_id, 4, [], str(i))
                           for i in range(4)]
                until(lambda: io.retrieval_runtime_status()['running'] == 2
                      and io.retrieval_runtime_status()['outstanding'] == 3
                      and io.retrieval_runtime_status()['waiting_submitters'] > 0)
                until(lambda: len(client.calls) == client.active == 2)
                assert len(client.calls) == client.active == client.peak == 2
                release.set()
                for future in futures:
                    assert {item.repo_id for item in future.result(timeout=3).items} == set(client.repo_ids)
                until(lambda: io.retrieval_runtime_status()['outstanding'] == 0)
                state = io.retrieval_runtime_status()
                assert state['peak_running'] == client.peak == 2
                assert state['outstanding'] == state['running'] == 0
                assert len(client.calls) == 24
                assert all(state['stages'][stage]['calls_total'] == 4 for stage in
                           ('profile', 'semantic', 'trend', 'activity', 'popularity', 'freshness'))
            finally:
                release.set()


@pytest.mark.parametrize('location', ['admission', 'executor_queue'])
def test_fake_clock_expiry_prevents_admission_and_queued_rpc_start(location):
    release, started = threading.Event(), threading.Event()
    now = [0.0]
    deadline = RequestDeadline(1, clock=lambda: now[0])
    invoked = []

    def running():
        started.set()
        assert release.wait(5)

    with isolated_runtimes(workers=1, capacity=1 if location == 'admission' else 2):
        pool = io._runtime()
        first = pool.submit('profile', running)
        assert started.wait(2)
        with ThreadPoolExecutor(max_workers=1) as callers:
            def submit_expiring():
                with deadline_scope(deadline):
                    return pool.submit('semantic', lambda: invoked.append('must-not-start'))
            try:
                submission = callers.submit(submit_expiring)
                if location == 'admission':
                    until(lambda: pool.snapshot()['waiting_submitters'] == 1)
                else:
                    queued = submission.result(timeout=2)
                    assert pool.snapshot()['outstanding'] == 2
                now[0] = 2
                with pool.condition:
                    pool.condition.notify_all()
                if location == 'admission':
                    with pytest.raises(CooperativeDeadlineExceeded):
                        submission.result(timeout=2)
                    assert pool.snapshot()['outstanding'] == 1
                release.set()
                first.result(timeout=2)
                if location == 'executor_queue':
                    with pytest.raises(CooperativeDeadlineExceeded):
                        queued.result(timeout=2)
                until(lambda: pool.snapshot()['outstanding'] == 0)
                assert invoked == []
                assert pool.snapshot()['stages']['semantic'].get('calls_total', 0) == 0
            finally:
                release.set()


def test_expiry_between_channels_bounds_timeout_and_starts_no_more_rpcs():
    now = [0.0]
    deadline = RequestDeadline(3, clock=lambda: now[0])
    timeouts = []
    client = FixtureQdrant(hook=lambda stage: now.__setitem__(0, 4.0))
    original = client.query_points
    def capture_timeout(**kwargs):
        timeouts.append(kwargs['timeout'])
        return original(**kwargs)
    client.query_points = capture_timeout
    with isolated_runtimes():
        engine = retriever(client)
        now[0] = .5
        with deadline_scope(deadline):
            # Already completed valid channels may remain usable. Regardless of
            # that race, expiry must prevent the next dependency stage entirely.
            try:
                engine._semantic(VECTOR, 100, minimum_candidates=4)
            except CooperativeDeadlineExceeded:
                pass
            with pytest.raises(CooperativeDeadlineExceeded):
                engine._discovery(150)
        assert timeouts == [2]  # floor(remaining2.5), never rounded beyond budget
        assert [call['stage'] for call in client.calls] == ['semantic']
        assert not any(call['after_stop'] for call in client.calls)
        until(lambda: io.retrieval_runtime_status()['outstanding'] == 0)
        assert io.retrieval_runtime_status()['outstanding'] == 0


def test_one_timed_out_discovery_channel_preserves_actual_retriever_results():
    result = measure_requests(workers=4, requests=4, clients=2, delay=.001, fail_stage='trend')
    assert result['request_errors'] == []
    assert all(result['outputs'])
    assert result['runtime']['stages']['trend']['errors_total'] == 4
    assert result['runtime']['outstanding'] == result['runtime']['running'] == 0
    assert result['post_stop_dependency_calls'] == 0


@pytest.mark.anyio
@pytest.mark.parametrize('kind', ['disconnect', 'timeout'])
async def test_abandoned_request_keeps_outer_slot_and_drains_actual_rpcs(kind):
    result = await measure_abandonment(kind)
    assert result['admission_held_at_return'] == 1
    assert result['new_request_rejected_until_exit'] is True
    assert 1 <= result['calls_at_return'] <= 2
    assert result['calls_after_caller_return'] == result['post_stop_dependency_calls'] == 0
    assert result['settled_runtime']['outstanding'] == result['settled_runtime']['running'] == 0
    assert result['deadline_metrics']['outstanding'] == 0


def test_search_starts_no_qdrant_rpc_when_encoding_finishes_after_budget():
    from api.contracts import RepositorySearchRequest
    from api.search import search_repositories
    from types import SimpleNamespace
    from uuid import UUID

    now = [0.0]
    def late_embedding(text):
        now[0] = 4
        return VECTOR
    client = FixtureQdrant()
    request = RepositorySearchRequest(schema_version=2, search_id=UUID(int=99), query='parser tools', limit=5)
    with isolated_runtimes(), deadline_scope(RequestDeadline(3, clock=lambda: now[0])):
        with pytest.raises(CooperativeDeadlineExceeded):
            search_repositories(request, retriever(client), SimpleNamespace(embed_text=late_embedding))
    assert client.calls == []


@pytest.mark.anyio
@pytest.mark.parametrize('milliseconds,expected_expiry', [('1000', 101), ('999999', 105)])
async def test_relative_request_budget_can_reduce_but_never_extend_local_bound(monkeypatch, milliseconds, expected_expiry):
    from api import request_budget
    from types import SimpleNamespace
    from utils.deadline import current_deadline

    monkeypatch.setattr(request_budget, 'RequestDeadline', lambda seconds:
                        RequestDeadline(seconds, clock=lambda: 100.0))
    async def connected():
        return False
    async def job():
        return current_deadline().expires_at
    request = SimpleNamespace(headers={'X-Request-Timeout-Ms': milliseconds}, is_disconnected=connected)
    assert await request_budget.run_read_only_request(request, 5, job) == expected_expiry



@pytest.mark.parametrize('failure', ['request_error', 'missing_id', 'duplicate_id', 'missing_sample'])
def test_equivalence_control_rejects_matching_failed_or_incomplete_outputs(failure):
    client = FixtureQdrant()
    result = {'request_errors': [], 'request_duration': {'count': 1},
              'outputs': [tuple((repo_id, .5, 'semantic') for repo_id in client.repo_ids)]}
    validate_control(result, client.repo_ids)
    if failure == 'request_error':
        result['request_errors'] = ['TimeoutError']
    elif failure == 'missing_id':
        result['outputs'] = [result['outputs'][0][:-1]]
    elif failure == 'duplicate_id':
        batch = list(result['outputs'][0])
        batch[-1] = batch[0]
        result['outputs'] = [tuple(batch)]
    else:
        result['outputs'] = []
    with pytest.raises(AssertionError):
        validate_control(result, client.repo_ids)
