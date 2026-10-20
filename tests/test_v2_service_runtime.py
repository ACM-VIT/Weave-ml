from __future__ import annotations

import asyncio
import threading
import time

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
    started = threading.Event()
    allow_finish = threading.Event()

    def blocking_job() -> str:
        started.set()
        assert allow_finish.wait(timeout=2)
        return "finished"

    task = asyncio.create_task(runtime.run_service_job("refresh", blocking_job))
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with pytest.raises(runtime.ServiceCapacityError):
        await runtime.run_service_job("refresh", lambda: "too-early")

    allow_finish.set()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if runtime.service_runtime_status()["refresh"]["outstanding"] == 0:
            break
        await asyncio.sleep(0.01)
    assert runtime.service_runtime_status()["refresh"]["outstanding"] == 0
    assert await runtime.run_service_job("refresh", lambda: "accepted") == "accepted"


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_timed_out_service_job_retains_capacity_until_worker_exits(
    monkeypatch,
    anyio_backend,
) -> None:
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

    with pytest.raises(runtime.ServiceCapacityError):
        await runtime.run_service_job("health", lambda: "too-early")

    allow_finish.set()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if runtime.service_runtime_status()["health"]["outstanding"] == 0:
            break
        await asyncio.sleep(0.01)
    assert runtime.service_runtime_status()["health"]["outstanding"] == 0


def test_runtime_status_uses_immutable_pool_settings(monkeypatch) -> None:
    monkeypatch.setenv("V2_FEEDBACK_EXECUTOR_WORKERS", "1")
    monkeypatch.setenv("V2_FEEDBACK_MAX_OUTSTANDING", "2")
    runtime.service_runtime_status()

    monkeypatch.setenv("V2_FEEDBACK_EXECUTOR_WORKERS", "8")

    assert runtime.service_runtime_status()["feedback"]["workers"] == 1


@pytest.mark.anyio
async def test_search_runs_while_single_embedding_worker_waits_on_summary_provider(monkeypatch) -> None:
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
    monkeypatch.setenv("EMBEDDING_MAX_OUTSTANDING_JOBS", "1")
    started = threading.Event()
    allow_finish = threading.Event()

    class SlowProvider:
        def generate(self, source, *, repair_feedback=None, deadline=None):
            started.set()
            assert allow_finish.wait(timeout=5)
            raise SummaryProviderError("unavailable")

    class Encoder:
        def encode(self, texts, **kwargs):
            return [[1.0] + [0.0] * 383 for _ in texts]

    model = embedding_runtime.BoundedEmbeddingModel(Encoder(), max_concurrency=1)
    embedder = SentenceTransformerEmbedder(model=model)
    pipeline = RepositoryEmbeddingPipeline(
        embedder=embedder, card_summarizer=CardSummaryPipeline(provider=SlowProvider()),
    )
    repo = {"repo_id": "00000000-0000-4000-8000-000000000111", "full_name": "weave/slow-summary", "description": "A collaborative review workspace."}
    task = asyncio.create_task(embedding_runtime.run_embedding_job(pipeline.embed_repository, repo))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        # No second encoder and no second indexing worker: only a reserved
        # short-job queue, using the exact model held by the blocked pipeline.
        monkeypatch.setenv("INTERNAL_API_SECRET", "isolated-search-secret")
        monkeypatch.setattr("api.v2.shared_embedding_model", lambda: model)
        monkeypatch.setattr("api.v2.retriever", lambda: object())

        def search(request, retriever, query_embedder):
            assert query_embedder.model is model
            assert len(query_embedder.embed_text(request.query)) == 384
            return {"schema_version": 2, "search_id": str(request.search_id), "embedding_version": "repo-embedding-v2", "items": []}

        monkeypatch.setattr("api.search.search_repositories", search)
        app = FastAPI()
        app.include_router(router)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://isolated.test") as client:
            response = await asyncio.wait_for(client.post(
                "/api/v2/search/repositories", headers={"X-Internal-Secret": "isolated-search-secret"},
                json={"schema_version": 2, "search_id": str(uuid.uuid4()), "query": "image understanding", "limit": 5},
            ), timeout=1)
        assert response.status_code == 200
        assert not task.done(), "search must complete before the summary provider responds"
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
