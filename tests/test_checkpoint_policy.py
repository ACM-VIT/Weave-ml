"""Fake-clock acceptance for approved retry, rejection and retention rules."""

from datetime import datetime, timedelta, timezone

import pytest

from acquisition.checkpoint import CorpusCheckpoint
from acquisition.checkpoint_policy import CheckpointPolicy


def _fixture(tmp_path, **policy):
    now = [datetime(2026, 10, 23, tzinfo=timezone.utc)]
    checkpoint = CorpusCheckpoint(
        tmp_path / "checkpoint.json", clock=lambda: now[0], jitter=lambda: 1,
        policy=CheckpointPolicy(**policy),
    )
    return checkpoint, now


def test_temporary_failure_obeys_backoff_and_resumes_across_restart(tmp_path):
    checkpoint, now = _fixture(tmp_path)
    checkpoint.add_pending_persistence(["owner/repo"])
    checkpoint.record_failure("owner/repo", "enrichment", "offline")
    checkpoint.save()
    assert checkpoint.due(checkpoint.pending_persistence, 100) == []
    now[0] += timedelta(seconds=59)
    assert checkpoint.due(checkpoint.pending_persistence, 100) == []
    now[0] += timedelta(seconds=1)
    reloaded = CorpusCheckpoint(checkpoint.path, clock=lambda: now[0])
    assert reloaded.due(reloaded.pending_persistence, 100) == ["owner/repo"]
    assert reloaded.data["failures"]["owner/repo"]["attempts"] == 1


def test_attempt_exhaustion_quarantines_without_discarding_pending_identity(tmp_path):
    checkpoint, now = _fixture(tmp_path)
    checkpoint.add_pending_persistence(["owner/repo"])
    for attempt in range(8):
        checkpoint.record_failure("Owner/Repo", "enrichment", "offline")
        now[0] += timedelta(hours=1)
        assert bool(checkpoint.due(checkpoint.pending_persistence, 1)) == (attempt < 7)
    assert checkpoint.pending_persistence == ["owner/repo"]
    assert "Owner/Repo" in checkpoint.excluded_names()
    checkpoint.replay_quarantined("OWNER/REPO")
    assert checkpoint.due(checkpoint.pending_persistence, 1) == ["owner/repo"]


def test_age_exhaustion_applies_without_an_additional_external_attempt(tmp_path):
    checkpoint, now = _fixture(tmp_path)
    checkpoint.add_pending_persistence(["owner/repo"])
    checkpoint.record_failure("owner/repo", "enrichment", "offline")
    now[0] += timedelta(days=7)
    assert checkpoint.due(checkpoint.pending_persistence, 1) == []
    assert checkpoint.data["failures"]["owner/repo"]["disposition"] == "quarantined"
    assert checkpoint.data["failures"]["owner/repo"]["attempts"] == 1


def test_permanent_failure_never_enters_an_automatic_retry_loop(tmp_path):
    checkpoint, _ = _fixture(tmp_path)
    checkpoint.add_pending_persistence(["owner/repo"])
    checkpoint.record_failure("owner/repo", "persistence", "invalid record", disposition="permanent")
    assert checkpoint.due(checkpoint.pending_persistence, 1) == []


def test_lowered_attempt_policy_is_enforced_before_fetching_again(tmp_path):
    checkpoint, now = _fixture(tmp_path)
    checkpoint.add_pending_persistence(["owner/repo"])
    checkpoint.record_failure("owner/repo", "enrichment", "offline")
    now[0] += timedelta(hours=1)
    checkpoint.policy = CheckpointPolicy(max_attempts=1)
    assert checkpoint.due(checkpoint.pending_persistence, 1) == []


def test_rejection_cooldown_and_policy_change_reconsider_repository(tmp_path):
    checkpoint, now = _fixture(tmp_path)
    checkpoint.record_rejection("Owner/Repo", ["no README"])
    assert "Owner/Repo" in checkpoint.excluded_names()
    now[0] += timedelta(days=7)
    assert "Owner/Repo" not in checkpoint.excluded_names()
    checkpoint.record_rejection("Owner/Repo", ["no README"])
    checkpoint.policy = CheckpointPolicy(version="new-quality-policy")
    assert "Owner/Repo" not in checkpoint.excluded_names()


def test_history_cleanup_preserves_unresolved_and_legacy_work(tmp_path):
    checkpoint, now = _fixture(tmp_path)
    checkpoint.record_rejection("owner/rejected", ["no README"])
    checkpoint.add_pending_persistence(["owner/failed"])
    checkpoint.record_failure("owner/failed", "enrichment", "offline")
    checkpoint.data["legacy_pending_index"] = ["owner/legacy"]
    now[0] += timedelta(days=30)
    checkpoint.prune_history()
    assert checkpoint.data["rejected"] == {}
    assert checkpoint.pending_persistence == ["owner/failed"]
    assert "owner/failed" in checkpoint.data["failures"]
    assert checkpoint.legacy_pending_index == ["owner/legacy"]


def test_active_capacity_backpressures_instead_of_evicting_unresolved_work(tmp_path):
    checkpoint, _ = _fixture(tmp_path, max_active_entries=1)
    checkpoint.add_pending_persistence(["owner/first"])
    with pytest.raises(RuntimeError, match="capacity"):
        checkpoint.add_pending_persistence(["owner/second"])
    assert checkpoint.pending_persistence == ["owner/first"]


def test_history_capacity_preserves_cooldowns_then_reclaims_old_terminal_entries(tmp_path):
    checkpoint, now = _fixture(tmp_path, max_history_entries=1)
    checkpoint.record_rejection("owner/first", ["no README"])
    with pytest.raises(RuntimeError, match="capacity"):
        checkpoint.record_rejection("owner/second", ["no README"])
    now[0] += timedelta(days=7)
    checkpoint.record_rejection("owner/second", ["no README"])
    assert set(checkpoint.data["rejected"]) == {"owner/second"}


def test_unknown_legacy_attempt_history_gets_a_grace_window(tmp_path):
    checkpoint, now = _fixture(tmp_path)
    checkpoint.add_pending_persistence(["owner/repo"])
    checkpoint.data["failures"]["owner/repo"] = {
        "stage": "enrichment", "error": "offline", "timestamp": "2020-01-01T00:00:00Z",
    }
    assert checkpoint.due(checkpoint.pending_persistence, 1) == ["owner/repo"]
    checkpoint.record_failure("owner/repo", "enrichment", "offline")
    failure = checkpoint.data["failures"]["owner/repo"]
    assert failure["first_failed_at"] == now[0].isoformat()
    assert failure["attempts"] == 1


@pytest.mark.parametrize("jitter", [0, 0.5, 1])
def test_retry_delays_stay_bounded(jitter):
    policy = CheckpointPolicy()
    assert 30 <= policy.retry_delay(1, jitter) <= 60
    assert 1_800 <= policy.retry_delay(10_000, jitter) <= 3_600


def test_checkpoint_byte_limit_never_overwrites_recoverable_disk_state(tmp_path):
    checkpoint, _ = _fixture(tmp_path, max_checkpoint_bytes=1_000)
    checkpoint.save()
    original = checkpoint.path.read_bytes()
    checkpoint.record_run({"failure": "x" * 1_001})
    with pytest.raises(RuntimeError, match="byte capacity"):
        checkpoint.save()
    assert checkpoint.path.read_bytes() == original
