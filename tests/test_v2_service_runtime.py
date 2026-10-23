from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from api import runtime


@pytest.fixture(autouse=True)
def _clean_service_runtime():
    runtime.shutdown_service_runtime()
    yield
    runtime.shutdown_service_runtime()


def test_service_runtime_rejects_undersized_capacity(monkeypatch) -> None:
    monkeypatch.setenv("V2_FEEDBACK_EXECUTOR_WORKERS", "3")
    monkeypatch.setenv("V2_FEEDBACK_MAX_OUTSTANDING", "2")

    with pytest.raises(ValueError, match="must be at least"):
        runtime.validate_service_runtime()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_cancelled_service_job_retains_capacity_until_worker_exits(
    monkeypatch,
    anyio_backend,
) -> None:
    monkeypatch.setenv("V2_REFRESH_EXECUTOR_WORKERS", "1")
    monkeypatch.setenv("V2_REFRESH_MAX_OUTSTANDING", "1")
    monkeypatch.setenv("V2_REFRESH_TIMEOUT_SECONDS", "30")
    from utils import observability
    observations = []
    original_observe = observability.observe_duration

    def observe(operation, stage, outcome, elapsed):
        observations.append((operation, stage, outcome, elapsed))
        original_observe(operation, stage, outcome, elapsed)

    monkeypatch.setattr(observability, "observe_duration", observe)
    worker_context = []
    started = threading.Event()
    allow_finish = threading.Event()

    def blocking_job() -> str:
        worker_context.append(observability.current_correlation())
        started.set()
        assert allow_finish.wait(timeout=2)
        return "finished"

    with observability.span_scope("refresh", server=True) as _span:
        server_context = observability.current_correlation()
        task = asyncio.create_task(runtime.run_service_job("refresh", blocking_job))
    assert await asyncio.to_thread(started.wait, 2)
    assert worker_context[0].trace_id == server_context.trace_id
    assert worker_context[0].parent_span_id == server_context.span_id
    assert not any(stage == "service" for _, stage, _, _ in observations)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with pytest.raises(runtime.ServiceCapacityError):
        await runtime.run_service_job("refresh", lambda: "too-early")

    assert any(stage == "queue" and outcome == "rejected"
               for _, stage, outcome, _ in observations)
    assert not any(stage == "late_work" for _, stage, _, _ in observations)
    allow_finish.set()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if runtime.service_runtime_status()["refresh"]["outstanding"] == 0:
            break
        await asyncio.sleep(0.01)
    assert runtime.service_runtime_status()["refresh"]["outstanding"] == 0
    assert any(stage == "queue" for _, stage, _, _ in observations)
    assert any(stage == "service" for _, stage, _, _ in observations)
    assert any(stage == "late_work" and outcome == "cancelled"
               for _, stage, outcome, _ in observations)
    assert await runtime.run_service_job("refresh", lambda: "accepted") == "accepted"


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_timed_out_service_job_retains_capacity_until_worker_exits(
    monkeypatch,
    anyio_backend,
) -> None:
    from utils.observability import diagnostic_snapshot, reset_stage_histograms_for_tests
    reset_stage_histograms_for_tests()
    monkeypatch.setenv("V2_HEALTH_EXECUTOR_WORKERS", "1")
    monkeypatch.setenv("V2_HEALTH_MAX_OUTSTANDING", "1")
    monkeypatch.setenv("V2_HEALTH_TIMEOUT_SECONDS", "0.1")
    started = threading.Event()
    allow_finish = threading.Event()

    def blocking_job() -> str:
        started.set()
        assert allow_finish.wait(timeout=2)
        return "finished"

    with pytest.raises(runtime.ServiceDeadlineExceeded):
        await runtime.run_service_job("health", blocking_job)
    assert started.is_set()
    status = runtime.service_runtime_status()["health"]
    assert status["outstanding"] == 1
    assert status["timed_out_total"] == 1
    assert not any(row["stage"] in {"service", "late_work"}
                   for row in diagnostic_snapshot()["histograms"])

    with pytest.raises(runtime.ServiceCapacityError):
        await runtime.run_service_job("health", lambda: "too-early")

    allow_finish.set()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if runtime.service_runtime_status()["health"]["outstanding"] == 0:
            break
        await asyncio.sleep(0.01)
    assert runtime.service_runtime_status()["health"]["outstanding"] == 0

    rows = diagnostic_snapshot()["histograms"]
    assert any(row["stage"] == "service" and row["count"] == 1 for row in rows)
    assert any(row["stage"] == "late_work" and row["outcome"] == "timeout"
               and row["count"] == 1 for row in rows)
    reset_stage_histograms_for_tests()


def test_runtime_status_uses_immutable_pool_settings(monkeypatch) -> None:
    monkeypatch.setenv("V2_FEEDBACK_EXECUTOR_WORKERS", "1")
    monkeypatch.setenv("V2_FEEDBACK_MAX_OUTSTANDING", "2")
    runtime.service_runtime_status()

    monkeypatch.setenv("V2_FEEDBACK_EXECUTOR_WORKERS", "8")

    assert runtime.service_runtime_status()["feedback"]["workers"] == 1


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["search", "onboarding"])
async def test_interactive_jobs_run_while_single_embedding_worker_waits_on_summary_provider(
    monkeypatch, operation,
) -> None:
    import uuid
    import httpx
    from fastapi import FastAPI
    from api.v2 import router
    from embedding import runtime as embedding_runtime
    from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
    from embedding.embeddings import SentenceTransformerEmbedder
    from summarization.pipeline import CardSummaryPipeline
    from summarization.provider import SummaryProviderError

    embedding_runtime.reset_embedding_runtime_for_tests()
    monkeypatch.setenv("EMBEDDING_EXECUTOR_WORKERS", "1")
    monkeypatch.setenv("EMBEDDING_MAX_OUTSTANDING_JOBS", "2")
    started = threading.Event()
    allow_finish = threading.Event()

    class SlowProvider:
        def generate(self, source, *, repair_feedback=None, deadline=None):
            started.set()
            assert allow_finish.wait(timeout=5)
            raise SummaryProviderError("unavailable")

    class Encoder:
        def encode(self, texts, **kwargs):
            vector = [1.0] + [0.0] * 383
            return vector if isinstance(texts, str) else [vector for _ in texts]

    model = embedding_runtime.BoundedEmbeddingModel(Encoder(), max_concurrency=1)
    embedder = SentenceTransformerEmbedder(model=model)
    pipeline = RepositoryEmbeddingPipeline(
        embedder=embedder, card_summarizer=CardSummaryPipeline(provider=SlowProvider()),
    )
    repo = {"repo_id": "00000000-0000-4000-8000-000000000111", "full_name": "weave/slow-summary", "description": "A collaborative review workspace."}
    task = asyncio.create_task(embedding_runtime.run_embedding_job(pipeline.embed_repository, repo))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        # Both interactive paths use the exact encoder held by the blocked
        # repository pipeline, not a second model or repository worker.
        monkeypatch.setenv("INTERNAL_API_SECRET", "isolated-search-secret")
        monkeypatch.setattr("api.v2.shared_embedding_model", lambda: model)
        monkeypatch.setattr("api.v2.retriever", lambda: SimpleNamespace(embedding_version="repo-embedding-v2", embedding_dim=384))

        def search(request, retriever, query_embedder):
            assert query_embedder.model is model
            assert len(query_embedder.embed_text(request.query)) == 384
            return {"schema_version": 2, "search_id": str(request.search_id), "embedding_version": "repo-embedding-v2", "items": []}

        monkeypatch.setattr("api.search.search_repositories", search)
        if operation == "onboarding":
            from scripts.user_onboarding import UserOnboardingPipeline

            onboarding = UserOnboardingPipeline(model=model)

            def onboard(request):
                vector = onboarding.generate_interest_vector(request.profile.model_dump())
                assert len(vector) == 384
                return {"accepted": True, "status": "applied", "user_id": str(request.user_id),
                        "profile_version": request.profile_version}

            # Replace external lock/storage boundaries, retaining real profile
            # synthesis, vector validation, shared encoder and HTTP admission.
            monkeypatch.setattr("api.v2._onboard_user_job", onboard)
            path = "/api/v2/users/onboard"
            body = {
                "schema_version": 2, "job_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()), "profile_version": 1,
                "profile": {"topics": ["image understanding"]},
            }
        else:
            path = "/api/v2/search/repositories"
            body = {"schema_version": 2, "search_id": str(uuid.uuid4()), "query": "image understanding", "limit": 5}
        app = FastAPI()
        app.include_router(router)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://isolated.test") as client:
            for _ in range(2):
                response = await asyncio.wait_for(client.post(
                    path, headers={"X-Internal-Secret": "isolated-search-secret"},
                    json=body,
                ), timeout=1)
                assert response.status_code == 200
                assert not task.done(), "interactive work must finish before the summary provider responds"
    finally:
        allow_finish.set()
        await task
        embedding_runtime.reset_embedding_runtime_for_tests()


@pytest.mark.anyio
async def test_cancelled_search_keeps_reserved_capacity_until_worker_exits(monkeypatch) -> None:
    monkeypatch.setenv("V2_SEARCH_EXECUTOR_WORKERS", "1")
    monkeypatch.setenv("V2_SEARCH_MAX_OUTSTANDING", "1")
    started = threading.Event()
    allow_finish = threading.Event()

    def work():
        started.set()
        assert allow_finish.wait(timeout=5)

    task = asyncio.create_task(runtime.run_service_job("search", work))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(runtime.ServiceCapacityError):
            await runtime.run_service_job("search", lambda: None)
        assert runtime.service_runtime_status()["search"]["outstanding"] == 1
    finally:
        allow_finish.set()


@pytest.mark.anyio
async def test_search_deadline_retains_capacity_and_records_saturation(monkeypatch) -> None:
    monkeypatch.setenv("V2_SEARCH_EXECUTOR_WORKERS", "1")
    monkeypatch.setenv("V2_SEARCH_MAX_OUTSTANDING", "1")
    monkeypatch.setenv("V2_SEARCH_TIMEOUT_SECONDS", "0.1")
    allow_finish = threading.Event()
    try:
        with pytest.raises(runtime.ServiceDeadlineExceeded):
            await runtime.run_service_job("search", lambda: allow_finish.wait(timeout=5))
        assert runtime.service_runtime_status()["search"]["timed_out_total"] == 1
        with pytest.raises(runtime.ServiceCapacityError):
            await runtime.run_service_job("search", lambda: None)
    finally:
        allow_finish.set()


@pytest.mark.anyio
@pytest.mark.parametrize('interruption', ['cancel_owner', 'cancel_waiter', 'timeout', 'configured_timeout'])
async def test_shared_query_flight_preserves_search_admission_during_http_interruptions(monkeypatch, interruption):
    import uuid
    import httpx
    from fastapi import FastAPI
    from api.v2 import router
    from embedding.runtime import BoundedEmbeddingModel
    from config import REPOSITORY_EMBEDDING_DIM
    from retrieval.v2_retriever import QdrantV2Retriever

    monkeypatch.setenv('INTERNAL_API_SECRET', 'isolated-query-cache-secret')
    monkeypatch.setenv('V2_SEARCH_EXECUTOR_WORKERS', '2')
    monkeypatch.setenv('V2_SEARCH_MAX_OUTSTANDING', '2')
    timeout = '6' if interruption == 'configured_timeout' else '5'
    monkeypatch.setenv('V2_SEARCH_TIMEOUT_SECONDS', timeout)
    started, release = threading.Event(), threading.Event()
    calls = []

    class Encoder:
        def encode(self, texts, **kwargs):
            calls.extend(texts)
            started.set()
            assert release.wait(8)
            return [[1.] + [0.] * (REPOSITORY_EMBEDDING_DIM - 1)]

    model = BoundedEmbeddingModel(Encoder(), max_concurrency=1)
    retriever = QdrantV2Retriever(
        client=SimpleNamespace(query_points=lambda **kwargs: SimpleNamespace(points=[])),
        heavy_ranker_enabled=False,
    )
    monkeypatch.setattr('api.v2.shared_embedding_model', lambda: model)
    monkeypatch.setattr('api.v2.retriever', lambda: retriever)
    app = FastAPI()
    app.include_router(router)
    tasks = []

    async def until(predicate):
        deadline = time.monotonic() + 2
        while not predicate():
            assert time.monotonic() < deadline
            await asyncio.sleep(.001)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://isolated.test') as client:
        def submit(*, short_budget=False):
            headers = {'X-Internal-Secret': 'isolated-query-cache-secret'}
            if short_budget:
                headers['X-Request-Timeout-Ms'] = '100'
            return asyncio.create_task(client.post(
                '/api/v2/search/repositories', headers=headers,
                json={'schema_version': 2, 'search_id': str(uuid.uuid4()), 'query': 'private matching query', 'limit': 5},
            ))

        try:
            tasks.append(submit(short_budget=interruption == 'timeout'))
            assert await asyncio.to_thread(started.wait, 2)
            if interruption == 'configured_timeout':
                # The already-created pool keeps its six-second deadline even
                # if the process environment changes before a matching job.
                monkeypatch.setenv('V2_SEARCH_TIMEOUT_SECONDS', '.1')
            tasks.append(submit(short_budget=interruption == 'timeout'))
            await until(lambda: model.query_embedding_cache.snapshot()['waiters'] == 1)
            if interruption == 'configured_timeout':
                await asyncio.sleep(4.2)
                release.set()
                assert [response.status_code for response in await asyncio.gather(*tasks)] == [200, 200]
                assert calls == ['private matching query']
                assert model.query_embedding_cache.snapshot()['flights'] == 0
                assert model.query_embedding_cache.snapshot()['waiters'] == 0
                return
            if interruption == 'timeout':
                assert [response.status_code for response in await asyncio.gather(*tasks)] == [503, 503]
                await until(lambda: model.query_embedding_cache.snapshot()['waiters'] == 0
                            and runtime.service_runtime_status()['search']['outstanding'] == 1)
            else:
                tasks[0 if interruption == 'cancel_owner' else 1].cancel()
                with pytest.raises(asyncio.CancelledError):
                    await tasks[0 if interruption == 'cancel_owner' else 1]
            # A completed cache waiter may release its own worker. Cancellation
            # must retain every still-live worker; the owner always holds its
            # slot/flight until the real encoder completes.
            if interruption == 'timeout':
                assert runtime.service_runtime_status()['search']['outstanding'] == 1
                assert model.query_embedding_cache.snapshot()['waiters'] == 0
            else:
                assert runtime.service_runtime_status()['search']['outstanding'] == 2
                assert model.query_embedding_cache.snapshot()['waiters'] == 1
                assert (await submit()).status_code == 503
            assert model.query_embedding_cache.snapshot()['flights'] == 1
            release.set()
            if interruption != 'timeout':
                assert (await tasks[1 if interruption == 'cancel_owner' else 0]).status_code == 200
            await until(lambda: runtime.service_runtime_status()['search']['outstanding'] == 0)
            assert (await submit()).status_code == 200
            assert calls == ['private matching query']
            assert model.query_embedding_cache.snapshot()['waiters'] == 0
            assert model.query_embedding_cache.snapshot()['flights'] == 0
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await until(lambda: runtime.service_runtime_status()['search']['outstanding'] == 0)


@pytest.mark.anyio
@pytest.mark.parametrize("interruption", ["cancel", "timeout"])
async def test_onboarding_and_repository_share_actual_worker_admission(monkeypatch, interruption):
    from embedding import runtime as embedding_runtime

    embedding_runtime.reset_embedding_runtime_for_tests()
    monkeypatch.setenv("EMBEDDING_EXECUTOR_WORKERS", "1")
    monkeypatch.setenv("EMBEDDING_MAX_OUTSTANDING_JOBS", "2")
    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "0.1" if interruption == "timeout" else "30")
    repository_started, onboarding_started, release = (threading.Event() for _ in range(3))

    def work(started):
        started.set()
        assert release.wait(5)

    repository = asyncio.create_task(embedding_runtime.run_embedding_job(work, repository_started))
    onboarding = None
    try:
        assert await asyncio.to_thread(repository_started.wait, 1)
        onboarding = asyncio.create_task(embedding_runtime.run_onboarding_job(work, onboarding_started))
        assert await asyncio.to_thread(onboarding_started.wait, 1)
        if interruption == "cancel":
            onboarding.cancel()
            with pytest.raises(asyncio.CancelledError):
                await onboarding
        else:
            with pytest.raises(embedding_runtime.EmbeddingDeadlineExceeded):
                await onboarding
        for run_job in (embedding_runtime.run_embedding_job, embedding_runtime.run_onboarding_job):
            with pytest.raises(embedding_runtime.EmbeddingCapacityError):
                await run_job(lambda: None)
    finally:
        release.set()
        await asyncio.gather(*([repository, onboarding] if onboarding else [repository]), return_exceptions=True)
        embedding_runtime.reset_embedding_runtime_for_tests()


@pytest.mark.anyio
async def test_embedding_shutdown_cancels_onboarding_queue_but_retains_running_slots(monkeypatch):
    from embedding import runtime as embedding_runtime

    embedding_runtime.reset_embedding_runtime_for_tests()
    monkeypatch.setenv("EMBEDDING_EXECUTOR_WORKERS", "1")
    monkeypatch.setenv("EMBEDDING_MAX_OUTSTANDING_JOBS", "3")
    release, repository_started, onboarding_started, queued_started = (threading.Event() for _ in range(4))
    admission = embedding_runtime.embedding_admission()

    def work(started):
        started.set()
        assert release.wait(5)

    tasks = [
        asyncio.create_task(embedding_runtime.run_embedding_job(work, repository_started)),
        asyncio.create_task(embedding_runtime.run_onboarding_job(work, onboarding_started)),
    ]
    try:
        assert await asyncio.to_thread(repository_started.wait, 1)
        assert await asyncio.to_thread(onboarding_started.wait, 1)
        queued = asyncio.create_task(embedding_runtime.run_onboarding_job(work, queued_started))
        tasks.append(queued)
        await asyncio.sleep(0)  # Submit the queued job before closing the pools.
        embedding_runtime.shutdown_embedding_runtime()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(queued, 1)
        assert not queued_started.is_set()
        # Only the cancelled queued future released its global slot. Shutdown
        # does not pretend either running thread has stopped.
        assert admission.acquire(blocking=False)
        assert not admission.acquire(blocking=False)
        admission.release()
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        embedding_runtime.reset_embedding_runtime_for_tests()


def test_native_executor_queue_and_service_histograms_wait_for_physical_exit(monkeypatch):
    from utils.observability import diagnostic_snapshot, reset_stage_histograms_for_tests

    monkeypatch.setenv("V2_REFRESH_EXECUTOR_WORKERS", "1")
    monkeypatch.setenv("V2_REFRESH_MAX_OUTSTANDING", "2")
    reset_stage_histograms_for_tests()
    executor = runtime._BoundedExecutor("refresh")
    entered = threading.Event()
    release = threading.Event()
    def blocked():
        entered.set()
        assert release.wait(2)
    try:
        first = executor.submit(blocked)
        assert entered.wait(2)
        second = executor.submit(lambda: "queued")
        with pytest.raises(runtime.ServiceCapacityError):
            executor.submit(lambda: "rejected")
        time.sleep(0.02)
        pending = diagnostic_snapshot()["histograms"]
        assert not any(row["stage"] == "service" for row in pending)
        assert any(row["stage"] == "queue" and row["outcome"] == "rejected"
                   and row["count"] == 1 for row in pending)
        release.set()
        first.result(2)
        assert second.result(2) == "queued"
        executor.drain()
        rows = {row["stage"]: row for row in diagnostic_snapshot()["histograms"]
                if row["operation"] == "refresh" and row["outcome"] == "ok"}
        assert rows["queue"]["count"] == rows["service"]["count"] == 2
        assert rows["queue"]["sum_seconds"] >= 0.02
        assert rows["service"]["sum_seconds"] >= 0.02
    finally:
        release.set()
        executor.drain()
        reset_stage_histograms_for_tests()
