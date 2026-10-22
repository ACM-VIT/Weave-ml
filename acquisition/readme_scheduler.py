"""Run-scoped README concurrency with bounded admission and owned clients."""

from __future__ import annotations

from collections import deque
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from collections.abc import Callable, Iterable, Iterator
import threading
import logging

from .github_graphql_client import FetchedReadme, GitHubGraphQLClient


class ReadmeScheduler:
    """Share one limit across metadata batches, never their mutable sessions.

    Cancellation prevents new work and cancels queued work. Running requests
    retain admission until completion and obey the client's transport timeouts;
    Python cannot forcibly interrupt an active requests call.
    """

    def __init__(
        self,
        client_factory: Callable[[], GitHubGraphQLClient],
        workers: int,
        cancel_event: threading.Event | None = None,
    ) -> None:
        if type(workers) is not int or not 1 <= workers <= 16:
            raise ValueError("README workers must be between 1 and 16")
        self.workers = workers
        self._cancel_event = threading.Event()
        self._external_cancel = cancel_event
        self._factory = client_factory
        self._local = threading.local()
        self._clients: list[GitHubGraphQLClient] = []
        self._lock = threading.Lock()
        self._closed = False
        self._futures: set[Future[FetchedReadme]] = set()
        self._slots = threading.BoundedSemaphore(2 * workers)
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="readme")

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set() or (
            self._external_cancel is not None and self._external_cancel.is_set()
        )

    def _fetch(self, owner: str, name: str) -> FetchedReadme:
        if self.cancelled:
            raise CancelledError("README enrichment cancelled")
        if not hasattr(self._local, "client"):
            client = self._factory()
            self._local.client = client
            with self._lock:
                self._clients.append(client)
        if self.cancelled:
            raise CancelledError("README enrichment cancelled")
        return self._local.client.get_readme(owner, name)

    def _finished(self, future: Future[FetchedReadme]) -> None:
        with self._lock:
            self._futures.discard(future)
        self._slots.release()

    def submit(self, owner: str, name: str) -> Future[FetchedReadme]:
        while not self._slots.acquire(timeout=0.05):
            if self.cancelled:
                raise CancelledError("README enrichment cancelled")
        try:
            with self._lock:
                if self._closed or self.cancelled:
                    raise CancelledError("README enrichment cancelled")
                future = self._executor.submit(self._fetch, owner, name)
                self._futures.add(future)
        except BaseException:
            self._slots.release()
            raise
        # A completed future invokes callbacks immediately, so register outside
        # the lock used by _finished. It owns this permit from this point on.
        future.add_done_callback(self._finished)
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
        with self._lock:
            futures = tuple(self._futures)
        for future in futures:
            future.cancel()

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        # All users have finished before another thread closes their sessions.
        for client in self._clients:
            try:
                client.session.close()
            except Exception:
                logging.getLogger(__name__).warning("Failed to close README worker session")

    def __enter__(self) -> ReadmeScheduler:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
