"""Provider outages, bounded probes, shared quotas and fallback retry admission."""

import multiprocessing
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
from urllib.parse import urlsplit

import pytest

from summarization.admission import (
    AdmissionDenied,
    LocalSummaryAdmission,
    RedisSummaryAdmission,
)
from summarization.pipeline import CardSummaryPipeline
from summarization.provider import OpenRouterSummaryProvider, SummaryProviderError
from summarization.settings import SummarySettings


class NoWait:
    def wait(self):
        pass


class Session:
    def __init__(self, status=503):
        self.calls = 0
        self.status = status

    def post(self, *args, **kwargs):
        self.calls += 1
        return type(
            "Response",
            (),
            {
                "status_code": self.status,
                "headers": {},
                "json": lambda _: {
                    "choices": [{"message": {"content": "invalid model output"}}]
                },
            },
        )()


def settings(**kwargs):
    return SummarySettings(api_key="isolated-admission-fixture", **kwargs)


def test_sustained_outage_stops_after_one_retry_budget_and_allows_one_fenced_probe(
    record_property,
):
    clock = [100.0]
    config = settings()
    policy = LocalSummaryAdmission(config, clock=lambda: clock[0])
    session = Session()
    provider = OpenRouterSummaryProvider(
        config,
        session=session,
        limiter=NoWait(),
        sleeper=lambda _: None,
        admission=policy,
    )
    durations = []
    pipeline = CardSummaryPipeline(settings=config, provider=provider)
    fallbacks = 0
    for index in range(50):
        repo = {
            "repo_id": f"outage-job-{index}",
            "content_hash": f"content-{index}",
            "description": "A tool for reviewing repository changes.",
        }
        started = time.perf_counter()
        artifact = pipeline.summarize(repo, repo)
        fallbacks += artifact.source == "description_fallback"
        durations.append(time.perf_counter() - started)
    assert fallbacks == 50
    assert session.calls == 3
    assert policy.metrics["circuit_rejected"] == 49
    clock[0] += 31
    with ThreadPoolExecutor(max_workers=8) as pool:

        def acquire(_):
            try:
                return policy.acquire(1)
            except AdmissionDenied:
                return None

        admitted = [p for p in pool.map(acquire, range(8)) if p is not None]
    assert len(admitted) == 1 and admitted[0].probe
    policy.report(admitted[0], True)
    assert not policy.acquire(1).probe
    # Old failed probe reports cannot reopen a successfully recovered circuit.
    policy.report(admitted[0], False)
    assert not policy.acquire(1).probe
    record_property("outage_jobs", 50)
    record_property("provider_calls", session.calls)
    record_property("retry_amplification", session.calls / 50)
    record_property("fallback_rate", fallbacks / 50)
    record_property("outage_job_p95_seconds", sorted(durations)[47])
    assert sorted(durations)[47] < 0.1


def test_invalid_generated_content_repairs_without_opening_availability_circuit():
    config = settings(failure_threshold=1)
    policy = LocalSummaryAdmission(config)
    session = Session(200)
    provider = OpenRouterSummaryProvider(
        config, session=session, limiter=NoWait(), admission=policy
    )
    result = CardSummaryPipeline(settings=config, provider=provider).summarize(
        {"description": "A tool for reviewing changes."},
        {"description": "A tool for reviewing changes."},
    )
    assert result.source == "description_fallback"
    assert session.calls == 2
    assert policy.metrics["failed"] == 0
    assert policy.acquire(1)


def test_fallback_content_cooldown_is_temporary_and_source_version_sensitive():
    clock = [100.0]
    config = settings(max_retries=0)
    policy = LocalSummaryAdmission(config, clock=lambda: clock[0])
    session = Session()
    provider = OpenRouterSummaryProvider(
        config, session=session, limiter=NoWait(), admission=policy
    )
    pipeline = CardSummaryPipeline(settings=config, provider=provider)
    repo = {
        "repo_id": "canonical-id",
        "content_hash": "content-a",
        "description": "A tool for reviewing changes.",
    }
    fallback = pipeline.summarize(repo, repo)
    assert not pipeline.should_upgrade(fallback, repo=repo)
    assert pipeline.summarize(repo, repo) == fallback
    assert session.calls == 1
    changed = {**repo, "content_hash": "content-b"}
    assert pipeline.should_upgrade(fallback, repo=changed)
    clock[0] += 31
    assert pipeline.should_upgrade(fallback, repo=repo)


def test_quota_rejection_never_sleeps_and_charges_repairs_and_retries():
    config = settings(rpm_limit=1)
    policy = LocalSummaryAdmission(config)
    session = Session()
    sleeps = []
    provider = OpenRouterSummaryProvider(
        config,
        session=session,
        limiter=NoWait(),
        sleeper=sleeps.append,
        admission=policy,
    )
    with pytest.raises(SummaryProviderError):
        provider.generate("An overview.")
    assert session.calls == 1
    assert policy.metrics["quota_rejected"] == 1
    assert len(sleeps) == 1  # Existing transport retry backoff, never a quota wait.
    with pytest.raises(SummaryProviderError):
        provider.generate("Another overview.")
    assert len(sleeps) == 1


def test_coordination_failure_fails_closed_without_a_provider_request():
    config = settings()
    session = Session(200)
    provider = OpenRouterSummaryProvider(
        config,
        session=session,
        limiter=NoWait(),
        admission=RedisSummaryAdmission(config, None),
    )
    with pytest.raises(SummaryProviderError, match="coordination"):
        provider.generate("Overview.")
    assert session.calls == 0


@pytest.fixture
def isolated_redis():
    url = os.getenv("SUMMARY_TEST_REDIS_URL")
    if not url:
        pytest.skip("Dedicated local Redis required")
    parsed = urlsplit(url)
    if parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.port != 6397:
        pytest.fail("Summary tests require the dedicated loopback Redis port6397")
    import redis

    client = redis.Redis.from_url(url)
    yield client
    client.close()


def _quota_process(args):
    url, key = args
    import redis

    client = redis.Redis.from_url(url)
    policy = RedisSummaryAdmission(
        SummarySettings(api_key=key, rpm_limit=100, tpm_limit=5), client
    )
    accepted = 0
    for _ in range(8):
        try:
            policy.acquire(1)
            accepted += 1
        except AdmissionDenied:
            pass
    client.close()
    return accepted


@pytest.mark.integration
def test_shared_request_token_allowance_across_processes(isolated_redis):
    # Avoid a measurement straddling the explicitly documented fixed-minute window.
    remaining = 60 - int(isolated_redis.time()[0]) % 60
    if remaining < 5:
        time.sleep(remaining + 0.05)
    key = "shared-fixture-" + uuid.uuid4().hex
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        accepted = sum(
            pool.map(_quota_process, [(os.environ["SUMMARY_TEST_REDIS_URL"], key)] * 4)
        )
    assert accepted == 5


@pytest.mark.integration
def test_redis_probe_fencing_expiry_and_shared_fallback_cooldown(isolated_redis):
    config = settings(failure_threshold=1)
    config = replace(config, api_key="probe-fixture-" + uuid.uuid4().hex)
    first = RedisSummaryAdmission(config, isolated_redis)
    second = RedisSummaryAdmission(config, isolated_redis)
    initial = first.acquire(1)
    first.report(initial, False)
    with pytest.raises(AdmissionDenied, match="circuit"):
        second.acquire(1)
    isolated_redis.hset(first.state, "until", 1)
    probe = second.acquire(1)
    with pytest.raises(AdmissionDenied):
        first.acquire(1)
    # Expired owner cannot close the circuit or overwrite a newer probe.
    isolated_redis.hset(first.state, "probe_until", 1)
    newer = first.acquire(1)
    second.report(probe, True)
    with pytest.raises(AdmissionDenied):
        second.acquire(1)
    first.report(newer, True)
    first.report(initial, False)
    assert not second.acquire(1).probe
    first.defer("bounded-content-key")
    assert second.deferred("bounded-content-key")
    assert (
        0
        < isolated_redis.pttl(first.prefix + ":fallback:bounded-content-key")
        <= 30_000
    )
    assert 0 < isolated_redis.pttl(first.quota) <= 240_000


def test_metrics_scrape_has_fixed_outcomes_and_does_not_initialize_provider():
    from api.metrics import render_prometheus
    from summarization.runtime import card_summary_pipeline, summary_admission_metrics

    before = card_summary_pipeline.cache_info().currsize
    metrics = summary_admission_metrics()
    assert len(metrics) == 9
    assert set(metrics).issuperset({"probes", "circuit_rejected", "fallback_deferred"})
    assert card_summary_pipeline.cache_info().currsize == before
    assert 'ml_summary_provider_events_total{outcome="probes"}' in render_prometheus({})


def test_preflight_deadline_abandons_probe_without_reporting_provider_failure():
    clock = [100.0]
    config = settings(failure_threshold=1)
    policy = LocalSummaryAdmission(config, clock=lambda: clock[0])
    first = policy.acquire(1)
    policy.report(first, False)
    clock[0] += 31
    probe = policy.acquire(1)
    policy.report(probe, None)
    assert policy.acquire(1).probe
    assert policy.metrics["failed"] == 1 and policy.metrics["abandoned"] == 1


def test_caller_deadline_does_not_defer_other_healthy_upgrade_jobs():
    config = settings(max_retries=0)
    policy = LocalSummaryAdmission(config)
    session = Session()
    provider = OpenRouterSummaryProvider(
        config, session=session, limiter=NoWait(), admission=policy
    )
    pipeline = CardSummaryPipeline(settings=config, provider=provider)
    repo = {
        "repo_id": "canonical",
        "content_hash": "same",
        "description": "A tool for reviewing changes.",
    }
    fallback = pipeline.summarize(repo, repo, deadline=time.monotonic() + 1)
    assert fallback.source == "description_fallback" and session.calls == 0
    assert pipeline.should_upgrade(fallback, repo=repo)


@pytest.mark.parametrize("coordinator", ["missing", "unavailable"])
def test_pipeline_counts_unavailable_coordination_before_acquisition(coordinator):
    from redis.exceptions import ConnectionError

    class Unavailable:
        def exists(self, _):
            raise ConnectionError("isolated fixture")

    config = settings()
    admission = RedisSummaryAdmission(
        config, None if coordinator == "missing" else Unavailable()
    )
    session = Session(200)
    provider = OpenRouterSummaryProvider(
        config, session=session, admission=admission, limiter=NoWait()
    )
    repo = {
        "repo_id": "canonical",
        "content_hash": "same",
        "description": "A tool for reviewing changes.",
    }
    assert (
        CardSummaryPipeline(settings=config, provider=provider)
        .summarize(repo, repo)
        .source
        == "description_fallback"
    )
    assert session.calls == 0 and admission.metrics["coordination_unavailable"] == 1


def test_slow_stream_is_actually_closed_before_probe_lease_can_expire(monkeypatch):
    import asyncio

    import httpx
    import requests

    from summarization.transport import BoundedSummarySession

    closed = []

    class Drip(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(50):
                await asyncio.sleep(0.02)
                yield b" "

        async def aclose(self):
            closed.append(True)

    async def handler(_request):
        return httpx.Response(200, stream=Drip(), headers={"Content-Length": "50"})

    monkeypatch.delattr(asyncio, "timeout", raising=False)
    transport = BoundedSummarySession(transport=httpx.MockTransport(handler))
    started = time.monotonic()
    with pytest.raises(requests.RequestException, match="deadline"):
        transport.post("https://summary.invalid", json={}, headers={}, timeout=0.1)
    assert time.monotonic() - started < 0.4 and closed == [True]


def test_compressed_success_is_not_decoded_twice_or_classified_as_outage():
    import gzip
    import json

    import httpx

    from summarization.transport import BoundedSummarySession

    expected = '{"summary":"A concise summary.","highlights":[]}'
    encoded = gzip.compress(
        json.dumps({"choices": [{"message": {"content": expected}}]}).encode()
    )

    async def handler(_):
        return httpx.Response(
            200,
            stream=httpx.ByteStream(encoded),
            headers={
                "Content-Encoding": "gzip",
                "Content-Length": str(len(encoded)),
                "X-Fixture": "preserved",
            },
        )

    config = settings(api_url="https://summary.invalid")
    policy = LocalSummaryAdmission(config)
    provider = OpenRouterSummaryProvider(
        config,
        session=BoundedSummarySession(transport=httpx.MockTransport(handler)),
        limiter=NoWait(),
        admission=policy,
        sleeper=lambda _: None,
    )
    assert provider.generate("A tool for reviewing changes.") == expected
    assert policy.metrics["available"] == 1 and policy.metrics["failed"] == 0
