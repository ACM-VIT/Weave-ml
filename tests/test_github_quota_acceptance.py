"""Concurrent quota windows and scheduler-owned retries with no network IO."""
from concurrent.futures import CancelledError, ThreadPoolExecutor
import threading

import pytest

from acquisition import github_rate_limit as quota
from acquisition.github_graphql_client import GitHubClientError
from acquisition.readme_scheduler import ReadmeScheduler
from scripts.benchmark_github_quota import (
    Clock, Transport, advance_until, fixture_clock, measure, until,
)


def test_concurrent_reset_admission_is_bounded_and_staggered():
    clock = Clock()
    original_admit = quota.GitHubRateLimitCoordinator.admit
    with fixture_clock(clock, observe_admissions=False):
        assert quota.GitHubRateLimitCoordinator.admit is original_admit
        coordinator = quota.GitHubRateLimitCoordinator()
        coordinator.observe({'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': '1005'})
        coordinator.throttle({'Retry-After': '5'})
        def concurrent_admissions(count):
            barrier = threading.Barrier(count)
            def call(_):
                barrier.wait(timeout=2)
                return coordinator.admit()
            with ThreadPoolExecutor(max_workers=count) as pool:
                return list(pool.map(call, range(count)))
        results = concurrent_admissions(8)
        assert all(ready == pytest.approx(5.1) for ready in results)
        assert coordinator.snapshot().get('requests', 0) == 0
        pending = 8
        while pending:
            ready = min(value for value in results if value is not None)
            clock.advance(ready - clock.monotonic())
            results = concurrent_admissions(pending)
            # All callers observe the same reset. Exactly one can reserve the
            # next staggered transport turn, even under simultaneous arrivals.
            assert results.count(None) == 1
            pending -= 1
        assert coordinator.snapshot()['requests'] == 8


def test_injected_429_frees_workers_and_replays_completed_stages_once():
    result = measure()
    assert result['completed_jobs'] == 8
    assert result['transport_requests'] == 17  # metadata+blob per job, one blob retry
    assert result['injected_429s'] == 1
    assert result['max_metadata_attempts_per_job'] == 1
    assert result['transport_worker_sleep_calls'] == 0
    assert result['maximum_admitted_jobs'] <= 4 and result['maximum_active_workers'] <= 2
    assert result['observed_queued_with_no_active_worker'] is True
    assert result['minimum_post_reset_admission_gap_seconds'] >= .099999
    assert result['all_worker_clients_share_quota'] and result['all_worker_sessions_closed']
    assert result['settled_scheduler'] == {'active': 0, 'queued': 0, 'admitted': 0}
    assert result['all_post_throttle_admissions_respect_reset']
    # Transport arrival can occur after a later response even when its quota
    # permit was obtained earlier. The quota decision is the correct boundary.
    events = result['quota_events']
    first_throttle = next(index for index, event in enumerate(events) if event['event'] == 'throttle')
    ready = events[first_throttle]['ready_at']
    assert all(event['at'] >= ready - 1e-9 for event in events[first_throttle + 1:]
               if event['event'] == 'admit' and event['granted'])


def test_cancelling_a_deferred_job_releases_it_without_another_request():
    clock = Clock()
    transport = Transport(clock)
    with fixture_clock(clock), ReadmeScheduler(transport.client, 1) as scheduler:
        future = scheduler.submit('fixture', 'cancelled')
        until(lambda: scheduler.snapshot()['queued'] == 1 and scheduler.snapshot()['active'] == 0)
        before = len(transport.records)
        scheduler.cancel()
        with pytest.raises(CancelledError):
            future.result(timeout=2)
        clock.advance(10)
        with scheduler._condition:
            scheduler._condition.notify_all()
        assert len(transport.records) == before == 2
        assert scheduler.snapshot() == {'active': 0, 'queued': 0, 'admitted': 0}


@pytest.mark.parametrize('expiry', ['retry_would_exceed_deadline', 'expire_while_queued'])
def test_deadline_stops_deferred_jobs_without_exceeding_run_budget(expiry):
    clock = Clock()
    transport = Transport(clock)
    with fixture_clock(clock), ReadmeScheduler(transport.client, 1,
            deadline=1 if expiry == 'retry_would_exceed_deadline' else 6) as scheduler:
        future = scheduler.submit('fixture', 'deadline')
        if expiry == 'expire_while_queued':
            until(lambda: scheduler.snapshot()['queued'] == 1)
            clock.advance(7)
            with scheduler._condition:
                scheduler._condition.notify_all()
        with pytest.raises(TimeoutError if expiry == 'retry_would_exceed_deadline' else CancelledError):
            future.result(timeout=2)
        assert len(transport.records) == 2
        assert clock.worker_sleep_calls == 0
        assert scheduler.snapshot() == {'active': 0, 'queued': 0, 'admitted': 0}


def test_permanent_http_failure_is_not_retried():
    clock = Clock()
    transport = Transport(clock, mode='permanent')
    with fixture_clock(clock):
        client = transport.client()
        with pytest.raises(GitHubClientError) as error:
            client.get_readme('fixture', 'missing')
        assert error.value.category == 'http' and error.value.status_code == 404
        assert len(transport.records) == 1
        assert client.rate_limits.snapshot().get('retries', 0) == 0


def test_attempt_budget_survives_all_scheduler_deferrals_without_amplification():
    clock = Clock()
    transport = Transport(clock, mode='always')
    with fixture_clock(clock), ReadmeScheduler(transport.client, 1) as scheduler:
        future = scheduler.submit('fixture', 'throttled')
        advance_until(future, clock, scheduler)
        with pytest.raises(GitHubClientError) as error:
            future.result()
        assert error.value.category == 'rate_limit'
        assert [row['stage'] for row in transport.records] == ['metadata', 'blob', 'blob', 'blob']
        assert clock.worker_sleep_calls == 0


def test_late_transport_success_is_not_parsed_cached_or_returned():
    clock = Clock()
    transport = Transport(clock)
    with fixture_clock(clock):
        client = transport.client()
        original = client.session.post
        def late_success(*args, **kwargs):
            response = original(*args, **kwargs)
            clock.advance(2)
            response.json = lambda: pytest.fail('expired success must not be parsed')
            return response
        client.session.post = late_success
        budget = quota.GitHubRequestBudget(deadline=1)
        with client.retry_scope(budget), pytest.raises(TimeoutError):
            client.execute('metadata', {'name': 'late'})
        assert budget.responses == {}
        assert len(transport.records) == 1


def test_auth_scope_keys_are_opaque_and_changed_auth_cannot_replay_a_response():
    clock = Clock()
    transport = Transport(clock)
    with fixture_clock(clock):
        client, peer = transport.client(), transport.client()
        assert client.rate_limits is peer.rate_limits
        assert len(client.coordination_key) == 64
        assert 'isolated-fixture-token' not in client.coordination_key
        budget = quota.GitHubRequestBudget()
        with client.retry_scope(budget):
            client.execute('metadata', {'name': 'scoped'})
            client.session.headers['Authorization'] = 'Bearer changed-fixture-token'
            with pytest.raises(GitHubClientError, match='authorization changed') as error:
                client.execute('metadata', {'name': 'scoped'})
        assert len(transport.records) == 1
        assert 'fixture-token' not in str(error.value)



def test_driver_failure_cancels_deferred_jobs_before_producer_join(monkeypatch):
    from scripts import benchmark_github_quota as benchmark
    schedulers, failures = [], []
    def failed_driver(future, clock, scheduler):
        schedulers.append(scheduler)
        until(lambda: scheduler.snapshot()['queued'] > 0 and scheduler.snapshot()['active'] == 0)
        raise AssertionError('intentional driver timeout')
    monkeypatch.setattr(benchmark, 'advance_until', failed_driver)
    def run():
        try:
            benchmark.measure()
        except BaseException as error:
            failures.append(error)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        thread.join(timeout=3)
        assert not thread.is_alive(), 'driver failure joined a blocked producer before cancelling jobs'
        assert len(failures) == 1 and str(failures[0]) == 'intentional driver timeout'
    finally:
        for scheduler in schedulers:
            scheduler.cancel()
        thread.join(timeout=3)


def test_overlapping_submitter_waits_after_all_four_job_permits_are_held():
    from concurrent.futures import TimeoutError as FutureTimeout
    from acquisition.github_graphql_client import FetchedReadme
    releases = threading.Event()
    entered = []
    lock = threading.Lock()
    class Client:
        def get_readme(self, owner, name):
            with lock:
                entered.append(name)
            assert releases.wait(3), 'test did not release active transport'
            return FetchedReadme('# Fixture')
        def close(self):
            pass
    with ReadmeScheduler(Client, 2) as scheduler, ThreadPoolExecutor(max_workers=1) as callers:
        try:
            futures = [scheduler.submit('fixture', str(i)) for i in range(4)]
            until(lambda: scheduler.snapshot() == {'active': 2, 'queued': 2, 'admitted': 4})
            started = threading.Event()
            def fifth():
                started.set()
                return scheduler.submit('fixture', 'fifth')
            pending = callers.submit(fifth)
            assert started.wait(1)
            def waiting_or_finished():
                # Observe the actual semaphore wait, not just a thread scheduled
                # immediately before submit. A removed ceiling finishes instead.
                with scheduler._slots._cond:
                    return bool(scheduler._slots._cond._waiters) or pending.done()
            until(waiting_or_finished)
            assert not pending.done(), 'fifth submit bypassed the admission ceiling'
            with pytest.raises(FutureTimeout):
                pending.result(timeout=.1)
            assert scheduler.snapshot()['admitted'] == 4
            releases.set()
            futures.append(pending.result(timeout=2))
            assert all(future.result(timeout=2) for future in futures)
        finally:
            releases.set()
            scheduler.cancel()
