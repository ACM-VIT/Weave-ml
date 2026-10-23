"""Operational CLI contracts for offline workers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from main import _parse_args, main as corpus_main
from trending_service import main as trending_main


def test_corpus_help_renders_even_when_environment_defaults_are_invalid(
    monkeypatch, capsys
) -> None:
    monkeypatch.setenv("CORPUS_TARGET_COUNT", "invalid")

    with pytest.raises(SystemExit) as exit_info:
        _parse_args(["--help"])

    assert exit_info.value.code == 0
    assert "Corpus pipeline" in capsys.readouterr().out


def test_corpus_cli_value_overrides_invalid_environment_default(monkeypatch) -> None:
    monkeypatch.setenv("CORPUS_TARGET_COUNT", "invalid")

    args = _parse_args(["--corpus-target", "25"])

    assert args.corpus_target == 25


@pytest.mark.parametrize("metadata_size, expected", [("10", 0), ("11", 1)])
def test_corpus_config_validation_opens_no_network_connections(monkeypatch, metadata_size, expected) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("BACKEND_URL", "http://backend.test")
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-secret")
    monkeypatch.setenv("CORPUS_TARGET_COUNT", "50000")
    monkeypatch.setenv("ACQUISITION_MAX_CYCLES", "1")
    monkeypatch.setenv("GITHUB_METADATA_BATCH_SIZE", metadata_size)

    with patch(
        "database.connector.PostgreSQLConnector.connect",
        side_effect=AssertionError("Postgres connection attempted"),
    ), patch(
        "requests.sessions.Session.request",
        side_effect=AssertionError("HTTP request attempted"),
    ):
        result = corpus_main(["--validate-config"])

    assert result == expected


def test_corpus_config_validation_reports_missing_required_values(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("BACKEND_URL", raising=False)
    monkeypatch.delenv("INTERNAL_API_SECRET", raising=False)
    monkeypatch.setenv("CORPUS_TARGET_COUNT", "50000")

    assert corpus_main(["--validate-config"]) == 1


def test_trending_config_validation_does_not_start_worker(monkeypatch, tmp_path) -> None:
    import trending.config as config

    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("BACKEND_URL", "http://backend.test")
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-secret")
    monkeypatch.setenv("TRENDING_COORDINATION_MODE", "singleton_file")
    monkeypatch.setenv("TRENDING_STATE_PATH", str(tmp_path / "trending.json"))
    with patch(
        "trending_service.run_once",
        side_effect=AssertionError("trending worker started"),
    ), patch(
        "trending_service.run_scheduler",
        side_effect=AssertionError("trending scheduler started"),
    ):
        with pytest.raises(SystemExit) as exit_info:
            trending_main(["--validate-config"])

    assert exit_info.value.code == 0
