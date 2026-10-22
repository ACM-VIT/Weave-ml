"""Run-scoped bounded README admission with scheduler-owned retry waits."""

from __future__ import annotations

from collections import deque
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from collections.abc import Callable, Iterable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
import heapq
import itertools
import logging
import threading
import time

from .github_graphql_client import FetchedReadme, GitHubGraphQLClient
from .github_rate_limit import GitHubRequestBudget, GitHubRetryLater


@dataclass(eq=False)
class _ReadmeJob:
    owner: str
    name: str
    future: Future[FetchedReadme]
    budget: GitHubRequestBudget
    started: bool = False
    active: bool = False


class ReadmeScheduler:
    """Retain one job permit across retries, but free its transport worker.

    Worker-local clients/sessions remain owned by one executor thread. A live
    request retains its worker and job admission until it actually completes;
    cancellation cannot forcibly interrupt requests IO. Delayed retries keep
    the same future, identity and request budget, bounded by 2 * workers jobs.
    """

    def __init__(
        self,
        client_factory: Callable[[], GitHubGraphQLClient],
        workers: int,
        cancel_event: threading.Event | None = None,
        *,
        deadline: float | None = None,
    ) -> None:
        if type(workers) is not int or not 1 <= workers <= 16:
            raise ValueError("README workers must be between 1 and 16")
        self.workers = workers
        self.deadline = deadline
        self._cancel_event = threading.Event()
        self._external_cancel = cancel_event
        self._factory = client_factory
        self._local = threading.local()
        self._clients: list[GitHubGraphQLClient] = []
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._closed = False
        self._futures: set[Future[FetchedReadme]] = set()
        self._jobs: dict[Future[FetchedReadme], _ReadmeJob] = {}
        self._pending: list[tuple[float, int, _ReadmeJob]] = []
        self._sequence = itertools.count()
        self._active = 0
        self._slots = threading.BoundedSemaphore(2 * workers)
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="readme")
        self._dispatcher = threading.Thread(target=self._dispatch, name="readme-retries", daemon=True)
        self._dispatcher.start()

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set() or (
            self._external_cancel is not None and self._external_cancel.is_set()
        ) or (self.deadline is not None and time.monotonic() >= self.deadline)

    def _fetch(self, job: _ReadmeJob) -> FetchedReadme:
        if not job.started:
            if not job.future.set_running_or_notify_cancel():
                raise CancelledError("README enrichment cancelled")
            job.started = True
        job.budget.check()
        if not hasattr(self._local, "client"):
            client = self._factory()
            self._local.client = client
            with self._lock:
                self._clients.append(client)
        job.budget.check()
        client = self._local.client
        retry_scope = getattr(client, "retry_scope", None)
        with retry_scope(job.budget) if callable(retry_scope) else nullcontext():
            return client.get_readme(job.owner, job.name)

    def _finished(self, future: Future[FetchedReadme]) -> None:
        with self._condition:
            self._futures.discard(future)
            job = self._jobs.pop(future, None)
            if job is not None:
                job.budget.finish()
            self._condition.notify_all()
        self._slots.release()

    def _start(self, job: _ReadmeJob) -> None:
        # Called under the re-entrant condition. Even an immediate callback can
        # safely update accounting; executor occupancy never exceeds workers.
        job.active = True
        self._active += 1
        try:
            attempt = self._executor.submit(self._fetch, job)
        except BaseException:
            self._active -= 1
            job.active = False
            raise
        attempt.add_done_callback(lambda future: self._attempt_finished(job, future))

    def _attempt_finished(self, job: _ReadmeJob, attempt: Future[FetchedReadme]) -> None:
        with self._condition:
            self._active -= 1
            job.active = False
            try:
                value = attempt.result()
            except GitHubRetryLater as retry:
                try:
                    job.budget.check_wait(retry.not_before)
                except BaseException as exc:
                    if not job.future.done():
                        job.future.set_exception(exc)
                else:
                    heapq.heappush(self._pending, (retry.not_before, next(self._sequence), job))
            except BaseException as exc:
                if not job.future.done():
                    job.future.set_exception(exc)
            else:
                if not job.future.done():
                    job.future.set_result(value)
            self._condition.notify_all()

    def _dispatch(self) -> None:
        with self._condition:
            while True:
                if self.cancelled:
                    for _ready, _sequence, job in self._pending:
                        if not job.future.done():
                            if not job.future.cancel():
                                job.future.set_exception(CancelledError("README enrichment cancelled"))
                    self._pending.clear()
                if self._closed and not self._pending and self._active == 0:
                    return
                while self._pending and self._pending[0][2].future.done():
                    heapq.heappop(self._pending)
                if self._pending and self._active < self.workers and not self.cancelled:
                    ready_at, _sequence, job = self._pending[0]
                    if ready_at <= time.monotonic():
                        heapq.heappop(self._pending)
                        try:
                            self._start(job)
                        except BaseException as exc:
                            if not job.future.done():
                                job.future.set_exception(exc)
                        continue
                delay = 0.05
                if self._pending and self._active < self.workers:
                    delay = min(delay, max(0.001, self._pending[0][0] - time.monotonic()))
                self._condition.wait(delay)

    def submit(self, owner: str, name: str) -> Future[FetchedReadme]:
        while not self._slots.acquire(timeout=0.05):
            if self.cancelled:
                raise CancelledError("README enrichment cancelled")
        future: Future[FetchedReadme] = Future()
        try:
            with self._condition:
                if self._closed or self.cancelled:
                    raise CancelledError("README enrichment cancelled")
                job = _ReadmeJob(owner, name, future, GitHubRequestBudget(
                    deadline=self.deadline, cancelled=lambda: self.cancelled,
                ))
                self._futures.add(future)
                self._jobs[future] = job
                if self._active < self.workers:
                    self._start(job)
                else:
                    heapq.heappush(self._pending, (0.0, next(self._sequence), job))
                future.add_done_callback(self._finished)
                self._condition.notify_all()
        except BaseException:
            with self._condition:
                self._futures.discard(future)
                self._jobs.pop(future, None)
            self._slots.release()
            raise
        return future

    def ordered(self, repositories: Iterable[tuple[str, str]]) -> Iterator[Future[FetchedReadme]]:
        """Keep a small rolling window while yielding in repository order."""
        source = iter(repositories)
        pending: deque[Future[FetchedReadme]] = deque()
        try:
            for _ in range(self.workers):
                item = next(source, None)
                if item is None:
                    break
                pending.append(self.submit(*item))
            while pending:
                yield pending.popleft()
                item = next(source, None)
                if item is not None:
                    pending.append(self.submit(*item))
        finally:
            for future in pending:
                future.cancel()

    def cancel(self) -> None:
        self._cancel_event.set()
        with self._condition:
            for future, job in tuple(self._jobs.items()):
                # Never complete/release admission for IO that is still live.
                if not job.active and not future.done():
                    if not future.cancel():
                        future.set_exception(CancelledError("README enrichment cancelled"))
            self._condition.notify_all()

    def snapshot(self) -> dict[str, int]:
        """Bounded queue/active counters for injected-throttle measurement."""
        with self._lock:
            return {
                "admitted": len(self._futures),
                "active": self._active,
                "queued": sum(not job.future.done() for _, _, job in self._pending),
            }

    def close(self) -> None:
        with self._condition:
            self._closed = True
        self.cancel()
        self._dispatcher.join()
        self._executor.shutdown(wait=True, cancel_futures=True)
        for client in self._clients:
            try:
                client.session.close()
            except Exception:
                logging.getLogger(__name__).warning("Failed to close README worker session")

    def __enter__(self) -> ReadmeScheduler:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
