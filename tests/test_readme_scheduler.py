"""Focused README scheduling bounds and ownership checks, without network IO."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock
import threading
import time

import pytest

from acquisition.github_graphql_client import FetchedReadme
from acquisition.pipeline import enrich_repositories
from acquisition.readme_scheduler import ReadmeScheduler


def test_shared_scheduler_overlaps_batches_without_sharing_clients():
    lock = threading.Lock()
    active = peak = 0
    clients = []

    class Client:
        def __init__(self):
            self.thread = threading.get_ident()
            self.session = MagicMock()
            clients.append(self)

        def get_readme(self, owner, name):
            nonlocal active, peak
            assert threading.get_ident() == self.thread
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.01 if int(name) % 2 else 0.03)
            with lock:
                active -= 1
            return FetchedReadme(name, source_path=".github/README.md", default_branch="main")

    def batch(start):
        return [future.result() for future in scheduler.ordered(("owner", str(i)) for i in range(start, start + 4))]

    event = threading.Event()
    with ReadmeScheduler(Client, 2, event) as scheduler:
        with ThreadPoolExecutor(max_workers=3) as outer:
            results = list(outer.map(batch, [0, 4, 8]))
    assert 1 < peak <= 2
    assert [[str(value) for value in batch] for batch in results] == [list(map(str, range(i, i + 4))) for i in [0, 4, 8]]
    assert all(value.source_path == ".github/README.md" for batch in results for value in batch)
    assert len(clients) == 2
    assert all(client.session.close.call_count == 1 for client in clients)
    assert not event.is_set()  # Closing never mutates the caller's signal.


def test_cancel_retains_running_admission_and_releases_queued_work():
    entered, release = threading.Event(), threading.Event()
    client = SimpleNamespace(session=MagicMock())
    def fetch(*_args):
        entered.set()
        assert release.wait(2)
        return FetchedReadme("done")
    client.get_readme = fetch
    scheduler = ReadmeScheduler(lambda: client, 1)
    try:
        running = scheduler.submit("owner", "running")
        assert entered.wait(1)
        queued = scheduler.submit("owner", "queued")
        scheduler.cancel()
        assert queued.cancelled()
        assert not running.done()
        # Only the queued slot is free; running IO still owns its permit.
        assert scheduler._slots.acquire(blocking=False)
        assert not scheduler._slots.acquire(blocking=False)
        scheduler._slots.release()
        with pytest.raises(CancelledError):
            scheduler.submit("owner", "rejected")
        release.set()
        assert running.result(timeout=1) == "done"
    finally:
        release.set()
        scheduler.close()


def test_submit_and_factory_failures_do_not_leak_admission(monkeypatch):
    def fail():
        raise RuntimeError("factory failed")
    with ReadmeScheduler(fail, 1) as scheduler:
        with pytest.raises(RuntimeError, match="factory failed"):
            scheduler.submit("owner", "repo").result(timeout=1)
        with monkeypatch.context() as patch:
            patch.setattr(scheduler._executor, "submit", MagicMock(side_effect=RuntimeError("submit failed")))
            with pytest.raises(RuntimeError, match="submit failed"):
                scheduler.submit("owner", "repo")
        assert scheduler._slots.acquire(timeout=1)
        assert scheduler._slots.acquire(timeout=1)
        scheduler._slots.release()
        scheduler._slots.release()


def test_pipeline_keeps_order_missing_readme_and_transient_failures(monkeypatch):
    class Client:
        def __init__(self, **_kwargs):
            self.session = MagicMock()
        def get_repositories_batch(self, repos, **_kwargs):
            return {f"{owner}/{name}": {"nameWithOwner": f"{owner}/{name}", "description": "project"} for owner, name in repos}
        def get_readme(self, owner, name):
            if name == "failed":
                raise TimeoutError("transient")
            if name == "first":
                time.sleep(0.02)
            return FetchedReadme("" if name == "missing" else "# Canonical", source_path="docs/README.md", default_branch="main")
    monkeypatch.setattr("acquisition.github_graphql_client.GitHubGraphQLClient", Client)
    monkeypatch.setenv("GITHUB_README_WORKERS", "2")
    result = enrich_repositories("fake", ["owner/first", "owner/missing", "owner/failed", "owner/last"], batch_size=2, workers=2)
    assert [r.repo_id for r in result.repositories] == ["owner/first", "owner/missing", "owner/last"]
    assert result.repositories[0].readme.source_path == "docs/README.md"
    assert result.repositories[1].readme.raw_markdown == ""
    assert [(f.repository, f.stage) for f in result.failures] == [("owner/failed", "readme_enrichment")]
