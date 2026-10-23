"""Checkpoint migration and crash-safe single-writer recovery tests."""

import json
import multiprocessing

import pytest

from acquisition.checkpoint import CorpusCheckpoint


def _try_competing_session(path, result):
    try:
        with CorpusCheckpoint(path).session():
            result.put("acquired")
    except RuntimeError:
        result.put("locked")


def _v1(path):
    original = {
        "version": 1,
        "pending_persistence": ["Owner/Delivery"],
        "pending_index": ["Owner/Index", "owner/index"],
        "failures": {"Owner/Delivery": {"stage": "persistence", "error": "failed"}},
        "rejected": {"Owner/Rejected": {"reasons": ["no README"]}},
        "last_run": {"persisted": 1},
        "updated_at": None,
    }
    raw = json.dumps(original).encode()
    path.write_bytes(raw)
    return raw


def test_v1_migration_preserves_unknown_ownership_and_exact_backup(tmp_path):
    path = tmp_path / "checkpoint.json"
    raw = _v1(path)
    checkpoint = CorpusCheckpoint(path)
    # Reading alone does not mutate the on-disk checkpoint.
    assert path.read_bytes() == raw
    assert checkpoint.pending_persistence == ["Owner/Delivery"]
    assert checkpoint.pending_index == []
    assert checkpoint.legacy_pending_index == ["Owner/Index"]
    checkpoint.save()

    loaded = CorpusCheckpoint(path)
    assert loaded.data["version"] == 2
    assert loaded.legacy_pending_index == ["Owner/Index"]
    assert loaded.data["failures"] == checkpoint.data["failures"]
    assert loaded.data["rejected"] == checkpoint.data["rejected"]
    backups = list(tmp_path.glob("*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == raw
    assert backups[0].stat().st_mode & 0o777 == 0o600
    loaded.save()
    assert list(tmp_path.glob("*.bak")) == backups


def test_interrupted_migration_preserves_original_and_can_restart(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.json"
    raw = _v1(path)
    checkpoint = CorpusCheckpoint(path)
    with monkeypatch.context() as patch:
        def fail_replace(*_):
            raise OSError("disk failure")

        patch.setattr("acquisition.checkpoint.os.replace", fail_replace)
        with pytest.raises(OSError, match="disk failure"):
            checkpoint.save()
    assert path.read_bytes() == raw
    assert not list(tmp_path.glob("*.tmp"))
    CorpusCheckpoint(path).save()
    assert CorpusCheckpoint(path).legacy_pending_index == ["Owner/Index"]
    assert len(list(tmp_path.glob("*.bak"))) == 1


@pytest.mark.parametrize("value", [
    [], {"version": True}, {"version": 3},
    {"version": 2, "pending_persistence": "Owner/Repo"},
    {"version": 2, "pending_index": ["invalid"]},
    {"version": 2, "failures": []},
    {"version": 2, "rejected": {"Owner/Repo": "bad"}},
])
def test_invalid_or_future_schema_is_never_overwritten(tmp_path, value):
    path = tmp_path / "checkpoint.json"
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="checkpoint"):
        CorpusCheckpoint(path)
    assert path.read_bytes() == raw


def test_stale_standalone_save_cannot_overwrite_new_work(tmp_path):
    path = tmp_path / "checkpoint.json"
    stale = CorpusCheckpoint(path)
    latest = CorpusCheckpoint(path)
    latest.add_pending_persistence(["Owner/New"])
    latest.save()
    stale.add_pending_persistence(["Owner/Old"])
    with pytest.raises(RuntimeError, match="changed"):
        stale.save()
    assert CorpusCheckpoint(path).pending_persistence == ["Owner/New"]


def test_session_reloads_state_before_recovery_and_releases_on_error(tmp_path):
    path = tmp_path / "checkpoint.json"
    stale = CorpusCheckpoint(path)
    latest = CorpusCheckpoint(path)
    latest.add_pending_persistence(["Owner/New"])
    latest.save()
    with pytest.raises(ValueError, match="worker failed"):
        with stale.session():
            assert stale.pending_persistence == ["Owner/New"]
            with pytest.raises(RuntimeError, match="active writer"):
                with CorpusCheckpoint(path).session():
                    pytest.fail("second writer entered")
            raise ValueError("worker failed")
    with latest.session():
        latest.save()


def test_writer_lock_excludes_another_process_across_atomic_replacement(tmp_path):
    path = tmp_path / "checkpoint.json"
    context = multiprocessing.get_context("spawn")
    result = context.Queue()
    with CorpusCheckpoint(path).session() as checkpoint:
        checkpoint.save()
        process = context.Process(target=_try_competing_session, args=(path, result))
        process.start()
        try:
            assert result.get(timeout=10) == "locked"
            process.join(timeout=10)
            assert process.exitcode == 0
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
    result.close()


def test_case_changes_replace_diagnostics_instead_of_growing_history(tmp_path):
    checkpoint = CorpusCheckpoint(tmp_path / "checkpoint.json")
    checkpoint.record_failure("Owner/Repo", "enrichment", "old")
    checkpoint.record_failure("owner/repo", "persistence", "new")
    checkpoint.record_rejection("Owner/Repo", ["old policy"])
    checkpoint.record_rejection("owner/repo", ["new policy"])
    assert list(checkpoint.data["failures"]) == ["owner/repo"]
    assert list(checkpoint.data["rejected"]) == ["owner/repo"]


def test_unchanged_save_does_not_rewrite_or_fsync_checkpoint(tmp_path, monkeypatch):
    checkpoint = CorpusCheckpoint(tmp_path / "checkpoint.json")
    checkpoint.save()
    original = checkpoint.path.read_bytes()
    monkeypatch.setattr("acquisition.checkpoint.os.fsync", lambda *_: pytest.fail("redundant fsync"))
    checkpoint.save()
    assert checkpoint.path.read_bytes() == original


@pytest.mark.parametrize('partial', [b'', b'{"version":'])
def test_partial_old_migration_backup_is_repaired_from_valid_original(tmp_path, partial):
    import hashlib
    path = tmp_path / 'checkpoint.json'
    raw = _v1(path)
    backup = path.with_name(f'{path.name}.v1.{hashlib.sha256(raw).hexdigest()}.bak')
    backup.write_bytes(partial)
    CorpusCheckpoint(path).save()
    assert backup.read_bytes() == raw
    assert CorpusCheckpoint(path).legacy_pending_index == ['Owner/Index']


def test_backup_fsync_failure_leaves_no_partial_final_backup(tmp_path, monkeypatch):
    path = tmp_path / 'checkpoint.json'
    raw = _v1(path)
    with monkeypatch.context() as patch:
        def fail(_):
            raise OSError('disk full during backup')
        patch.setattr('acquisition.checkpoint.os.fsync', fail)
        with pytest.raises(OSError, match='disk full'):
            CorpusCheckpoint(path).save()
    assert path.read_bytes() == raw
    assert not list(tmp_path.glob('*.bak'))
    CorpusCheckpoint(path).save()
    assert next(tmp_path.glob('*.bak')).read_bytes() == raw
