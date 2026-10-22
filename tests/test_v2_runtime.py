from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import pytest

from inference import runtime


def test_recommendation_runtime_rejects_an_undersized_capacity(monkeypatch) -> None:
    monkeypatch.setenv("V2_RECOMMENDATION_EXECUTOR_WORKERS", "4")
    monkeypatch.setenv("V2_RECOMMENDATION_MAX_OUTSTANDING", "3")

    with pytest.raises(ValueError, match="must be at least"):
        runtime.validate_recommendation_runtime()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.parametrize("interruption", ["cancel", "timeout"])
async def test_interrupted_recommendation_keeps_admission_until_worker_exits(
    monkeypatch,
    anyio_backend,
    interruption,
) -> None:
    class RecordingAdmission:
        def __init__(self) -> None:
            self.held = False
            self.releases = 0

        def acquire(self, *, blocking: bool) -> bool:
            assert blocking is False
            if self.held:
                return False
            self.held = True
            return True

        def release(self) -> None:
            assert self.held
            self.held = False
            self.releases += 1

    started = threading.Event()
    allow_finish = threading.Event()
    finished = threading.Event()
    admission = RecordingAdmission()
    executor = ThreadPoolExecutor(max_workers=1)

    def blocking_job() -> str:
        started.set()
        assert allow_finish.wait(timeout=2)
        finished.set()
        return "finished"

    monkeypatch.setattr(runtime, "recommendation_admission", lambda: admission)
    monkeypatch.setattr(runtime, "recommendation_executor", lambda: executor)
    monkeypatch.setattr(runtime, "recommendation_timeout_seconds", lambda: 0.1 if interruption == "timeout" else 120.0)
    task = asyncio.create_task(runtime.run_recommendation_job(blocking_job))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        if interruption == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(runtime.RecommendationDeadlineExceeded):
                await task

        assert admission.releases == 0
        with pytest.raises(runtime.RecommendationCapacityError):
            await runtime.run_recommendation_job(lambda: "too-early")

        allow_finish.set()
        assert await asyncio.to_thread(finished.wait, 2)
        for _ in range(100):
            if admission.releases:
                break
            await asyncio.sleep(0)
        assert admission.releases == 1
        assert await runtime.run_recommendation_job(lambda: "accepted") == "accepted"
        assert admission.releases == 2
    finally:
        allow_finish.set()
        executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.anyio
async def test_recommendation_deadline_includes_queue_time_and_is_request_local(monkeypatch):
    executor = ThreadPoolExecutor(max_workers=1)
    admission = threading.BoundedSemaphore(2)
    started, release, queued = (threading.Event() for _ in range(3))
    clock = SimpleNamespace(now=100.0)
    submissions = []

    def submit(*args):
        future = executor.submit(*args)
        submissions.append(future)
        if len(submissions) == 2:
            queued.set()
        return future

    # Patch only the runtime's clock; asyncio retains its real event-loop clock.
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(runtime, "recommendation_admission", lambda: admission)
    monkeypatch.setattr(runtime, "recommendation_executor", lambda: SimpleNamespace(submit=submit))
    monkeypatch.setattr(runtime, "recommendation_timeout_seconds", lambda: 2.0)

    def block():
        started.set()
        assert release.wait(2)

    first = asyncio.create_task(runtime.run_recommendation_job(block))
    second = None
    try:
        assert await asyncio.to_thread(started.wait, 1)
        second = asyncio.create_task(runtime.run_recommendation_job(
            lambda: (runtime.current_recommendation_deadline(), clock.now)
        ))
        assert await asyncio.to_thread(queued.wait, 1)
        # Advance time only after the second job is queued behind the blocker.
        clock.now = 100.5
        release.set()
        deadline, worker_started = await second
        await first
        assert deadline == 102.0
        assert worker_started == 100.5
        assert deadline - worker_started == 1.5
        assert runtime.current_recommendation_deadline() is None
        assert executor.submit(runtime.current_recommendation_deadline).result(1) is None
    finally:
        release.set()
        await asyncio.gather(*([first, second] if second else [first]), return_exceptions=True)
        executor.shutdown(wait=True, cancel_futures=True)
