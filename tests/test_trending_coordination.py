"""Singleton durability, concurrent startup and snapshot retry acceptance."""

from datetime import datetime, timedelta, timezone
import json
import multiprocessing
from types import SimpleNamespace
import uuid

import pytest

from acquisition.models import AcquisitionRunResult
from trending.backend_storage import BackendTrendingStorage
from trending.coordination import SingletonTrendingCoordinator
from trending.scheduler import TrendingScheduler
from trending_service import parse_args


NOW = datetime(2026, 10, 23, tzinfo=timezone.utc)


def _competing_start(path, now, queue):
    with SingletonTrendingCoordinator(path, clock=lambda: now).refresh_session(force=False, refresh_hours=24) as acquired:
        queue.put(acquired)


def _source():
    return SimpleNamespace(
        repo_id="owner/repo", warnings=[], raw_repository={}, readme=None,
        payload={
            "full_name": "owner/repo", "github_id": "123", "github_node_id": "node",
            "owner_github_id": "456", "readme": "canonical source", "observed_at": NOW.isoformat(),
        },
    )


def test_singleton_restart_respects_durable_freshness_and_force(tmp_path):
    path = tmp_path / "trending.json"
    now = [NOW]
    coordinator = SingletonTrendingCoordinator(path, clock=lambda: now[0])
    with coordinator.refresh_session(force=False, refresh_hours=24) as acquired:
        assert acquired
        coordinator.mark_success(snapshot_id=str(uuid.uuid4()), observed_at=NOW)
    restarted = SingletonTrendingCoordinator(path, clock=lambda: now[0])
    with restarted.refresh_session(force=False, refresh_hours=24) as acquired:
        assert not acquired
    with restarted.refresh_session(force=True, refresh_hours=24) as acquired:
        assert acquired
        # Force never grants another owner permission to acquire in parallel.
        with coordinator.refresh_session(force=True, refresh_hours=24) as second:
            assert not second
    now[0] += timedelta(hours=24)
    with restarted.refresh_session(force=False, refresh_hours=24) as acquired:
        assert acquired


def test_concurrent_startup_has_one_owner_across_processes(tmp_path):
    path = tmp_path / "trending.json"
    coordinator = SingletonTrendingCoordinator(path, clock=lambda: NOW)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    with coordinator.refresh_session(force=False, refresh_hours=24) as acquired:
        assert acquired
        process = context.Process(target=_competing_start, args=(path, NOW, queue))
        process.start()
        try:
            assert queue.get(timeout=10) is False
            process.join(timeout=10)
            assert process.exitcode == 0
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
        coordinator.mark_success(snapshot_id=str(uuid.uuid4()), observed_at=NOW)
    queue.close()
    with SingletonTrendingCoordinator(path, clock=lambda: NOW).refresh_session(force=False, refresh_hours=24) as acquired:
        assert not acquired


def test_failed_or_abandoned_refresh_never_advances_freshness(tmp_path):
    coordinator = SingletonTrendingCoordinator(tmp_path / "trending.json", clock=lambda: NOW)
    with pytest.raises(RuntimeError, match="acquisition failed"):
        with coordinator.refresh_session(force=False, refresh_hours=24):
            raise RuntimeError("acquisition failed")
    assert coordinator.last_refresh() is None
    with coordinator.refresh_session(force=False, refresh_hours=24) as acquired:
        assert acquired
    with pytest.raises(RuntimeError, match="active singleton"):
        coordinator.mark_success(snapshot_id=str(uuid.uuid4()), observed_at=NOW)


@pytest.mark.parametrize("raw", ["broken json", '{"version": 2}', '{"version": 1}', '{"version": 1, "last_success_observed_at": "invalid"}'])
def test_invalid_durable_state_never_becomes_an_automatic_refresh(tmp_path, raw):
    path = tmp_path / "trending.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        with SingletonTrendingCoordinator(path).refresh_session(force=True, refresh_hours=24):
            pytest.fail("invalid freshness was bypassed")
    assert path.read_text() == raw


def test_stale_freshness_emits_48_hour_warning(tmp_path, caplog):
    now = [NOW]
    coordinator = SingletonTrendingCoordinator(tmp_path / "trending.json", clock=lambda: now[0])
    with coordinator.refresh_session(force=False, refresh_hours=24):
        coordinator.mark_success(snapshot_id=str(uuid.uuid4()), observed_at=NOW)
    now[0] += timedelta(hours=48)
    with coordinator.refresh_session(force=False, refresh_hours=24) as acquired:
        assert acquired
    assert "freshness target missed" in caplog.text


def test_publish_retry_reuses_exact_snapshot_id_payload_and_successful_enrichment():
    published, enrich_calls = [], []

    def publish(payload):
        published.append(json.dumps(payload, sort_keys=True))
        if len(published) == 1:
            raise RuntimeError("lost backend acknowledgement")

    storage = BackendTrendingStorage(
        backend=SimpleNamespace(publish_trending_snapshot=publish), github_token="fixture",
        clock=lambda: NOW, monotonic=lambda: 0,
    )
    storage._enrich_candidates = lambda candidates: enrich_calls.append(candidates) or AcquisitionRunResult(repositories=[_source()])
    candidates = [{"full_name": "owner/repo", "daily_stars": 5}]
    with pytest.raises(RuntimeError, match="acknowledgement"):
        storage.upsert_repositories(candidates)
    assert storage.get_last_refresh_time() is None
    assert storage.upsert_repositories(candidates) == 1
    assert len(enrich_calls) == 1
    assert published[0] == published[1]


def test_observation_window_is_checked_after_slow_enrichment():
    now, published = [NOW], []
    storage = BackendTrendingStorage(
        backend=SimpleNamespace(publish_trending_snapshot=published.append), github_token="fixture",
        clock=lambda: now[0],
    )

    def slow_enrich(_):
        now[0] += timedelta(minutes=30)
        return AcquisitionRunResult(repositories=[_source()])

    storage._enrich_candidates = slow_enrich
    with pytest.raises(ValueError, match="stale trending"):
        storage.upsert_repositories([{"full_name": "owner/repo"}])
    assert published == []


def test_scheduler_fails_closed_without_explicit_coordination(monkeypatch):
    monkeypatch.setattr("trending.config.validate_config", lambda: [])
    fetcher = SimpleNamespace(fetch_trending_repositories=lambda: pytest.fail("uncoordinated fetch"))
    storage = BackendTrendingStorage(backend=SimpleNamespace(), github_token="fixture")
    assert TrendingScheduler(fetcher=fetcher, storage=storage).start_once(force=True) is False


def test_scheduled_startup_respects_freshness_instead_of_forcing(monkeypatch):
    scheduler = TrendingScheduler(fetcher=SimpleNamespace(), storage=SimpleNamespace())
    calls = []

    def refresh(*, force):
        calls.append(force)
        scheduler.running = False
        return True

    monkeypatch.setattr(scheduler, "refresh_trending_repositories", refresh)
    scheduler.start_scheduled()
    assert calls == [False]


def test_cli_force_is_explicit_and_only_applies_to_once():
    assert parse_args(["--once"]).force is False
    assert parse_args(["--once", "--force"]).force is True
    with pytest.raises(SystemExit):
        parse_args(["--scheduled", "--force"])


def test_scheduler_restart_publishes_once_then_skips_without_source_work(tmp_path, monkeypatch):
    monkeypatch.setattr("trending.config.validate_config", lambda: [])
    monkeypatch.setattr("trending.config.TRENDING_REFRESH_HOURS", 24)
    published, fetched = [], []
    path = tmp_path / "trending.json"
    for _ in range(2):
        storage = BackendTrendingStorage(
            backend=SimpleNamespace(publish_trending_snapshot=published.append), github_token="fixture",
            coordinator=SingletonTrendingCoordinator(path, clock=lambda: NOW), clock=lambda: NOW,
        )
        storage._enrich_candidates = lambda _: AcquisitionRunResult(repositories=[_source()])
        fetcher = SimpleNamespace(fetch_trending_repositories=lambda: fetched.append(True) or [{"full_name": "owner/repo"}])
        scheduler = TrendingScheduler(fetcher=fetcher, storage=storage, clock=lambda: NOW)
        assert scheduler.start_once() is True
    assert len(fetched) == len(published) == 1
    state = json.loads(path.read_text())
    assert state["last_success_observed_at"] == NOW.isoformat()
    assert set(state) == {"version", "last_success_observed_at", "published_at", "snapshot_id"}


def test_cli_preflight_checks_coordination_without_creating_state(tmp_path, monkeypatch):
    import trending_service

    monkeypatch.setattr(trending_service, "validate_config", lambda: [])
    for name in ("GITHUB_TOKEN", "BACKEND_URL", "INTERNAL_API_SECRET"):
        monkeypatch.setenv(name, "fixture")
    monkeypatch.delenv("TRENDING_COORDINATION_MODE", raising=False)
    monkeypatch.delenv("TRENDING_STATE_PATH", raising=False)
    with pytest.raises(SystemExit) as failed:
        trending_service.main(["--validate-config"])
    assert failed.value.code == 1
    path = tmp_path / "state.json"
    monkeypatch.setenv("TRENDING_COORDINATION_MODE", "singleton_file")
    monkeypatch.setenv("TRENDING_STATE_PATH", str(path))
    with pytest.raises(SystemExit) as passed:
        trending_service.main(["--validate-config"])
    assert passed.value.code == 0
    assert not path.exists()
