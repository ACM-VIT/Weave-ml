"""Preserve published replay evidence when post-worker publication fails."""
import json
from pathlib import Path
import subprocess

import pytest

from evaluation.stage_runner import FrozenSnapshot, execute_snapshot
from scripts import run_recommendation_stages as cli

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "tests/fixtures/recommendation_actual_stage/snapshot.json"


@pytest.fixture
def published_output(tmp_path):
    output = tmp_path / "run.json"
    output.write_bytes(b"previous successful output")
    return output


def successful_worker(report):
    def run(command, **kwargs):
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, "git version fixture", "")
        private_output = Path(command[command.index("--output") + 1])
        private_output.write_text(json.dumps(report))
        return subprocess.CompletedProcess(command, 0, "worker exited successfully", "")
    return run


def test_cli_rejects_invalid_worker_report_without_replacing_output(published_output, monkeypatch):
    monkeypatch.setattr(cli.subprocess, "run", successful_worker({}))
    assert cli.main(["--snapshot", str(SNAPSHOT), "--output", str(published_output)]) == 2
    assert published_output.read_bytes() == b"previous successful output"


def test_cli_cleanup_failure_does_not_publish_valid_worker_report(published_output, monkeypatch):
    snapshot = FrozenSnapshot.model_validate_json(SNAPSHOT.read_text())
    report = execute_snapshot(snapshot, ROOT / "inference").model_dump(mode="json")
    original_temporary_directory = cli.tempfile.TemporaryDirectory
    cleanup_reached = False

    class FailsAfterCleanup:
        def __init__(self, *args, **kwargs):
            self.manager = original_temporary_directory(*args, **kwargs)

        def __enter__(self):
            return self.manager.__enter__()

        def __exit__(self, *exc_info):
            nonlocal cleanup_reached
            self.manager.__exit__(*exc_info)
            assert exc_info[0] is None  # The real report validator succeeded.
            cleanup_reached = True
            raise OSError("controlled temporary-directory cleanup failure")

    monkeypatch.setattr(cli.subprocess, "run", successful_worker(report))
    monkeypatch.setattr(cli.tempfile, "TemporaryDirectory", FailsAfterCleanup)
    assert cli.main(["--snapshot", str(SNAPSHOT), "--output", str(published_output)]) == 2
    assert cleanup_reached
    assert published_output.read_bytes() == b"previous successful output"
