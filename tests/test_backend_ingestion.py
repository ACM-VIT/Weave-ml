"""Backend v2 ingestion boundary tests."""

from __future__ import annotations

from importlib.util import find_spec
from types import SimpleNamespace
import uuid

import pytest

from acquisition.backend_client import BackendIngestionClient, repository_upsert_record
from trending.backend_storage import BackendTrendingStorage
from trending_service import parse_args


class _Response:
    status_code = 200

    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


class _Session:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response(
            {
                "mappings": [
                    {
                        "github_id": "123",
                        "repo_id": str(uuid.uuid4()),
                        "content_version": 1,
                        "changed": True,
                    }
                ]
            }
        )


def _source(
    *,
    full_name: str = "owner/repo",
    github_id: str = "123",
    readme: str | None = None,
    warnings: list[str] | None = None,
):
    owner, name = full_name.split("/", 1)
    return SimpleNamespace(
        repo_id=full_name,
        payload={
            "id": full_name,
            "full_name": full_name,
            "github_id": github_id,
            "github_node_id": "R_kg_test",
            "owner_github_id": "456",
            "html_url": f"https://github.com/{full_name}",
            "description": "demo",
            "primary_language": "Python",
            "languages": ["Python"],
            "topics": ["ml"],
            "star_count": 10,
            "fork_count": 2,
            "open_issues_count": 1,
            "pushed_at": "2026-07-15T12:00:00Z",
            "observed_at": "2026-07-15T12:05:00Z",
        },
        raw_repository={},
        readme=SimpleNamespace(
            raw_markdown=readme
            if readme is not None
            else "# README\n\n![Preview](images/preview.png)\n\n```sh\nnpm install\n```",
            clean_text="README",
            source_path="README.md",
            default_branch="main",
            base_url=(
                f"https://raw.githubusercontent.com/{owner}/{name}/refs/heads/main/"
            ),
        ),
        warnings=list(warnings or []),
    )


def test_repository_record_keeps_source_and_backend_identity_fields_separate():
    record = repository_upsert_record(_source())

    assert record["github_id"] == "123"
    assert record["github_node_id"] == "R_kg_test"
    assert record["owner_github_id"] == "456"
    assert record["full_name"] == "owner/repo"
    assert "repo_id" not in record
    assert record["url"] == "https://github.com/owner/repo"
    assert record["readme"] == (
        "# README\n\n![Preview](images/preview.png)\n\n```sh\nnpm install\n```"
    )
    assert record["readme"] != _source().readme.clean_text
    assert record["readme_length"] == len(record["readme"])
    assert record["readme_source_path"] == "README.md"
    assert record["readme_default_branch"] == "main"
    assert record["readme_base_url"] == (
        "https://raw.githubusercontent.com/owner/repo/refs/heads/main/"
    )


def test_repository_record_normalizes_confirmed_readme_absence_to_null():
    source = _source(readme="")

    record = repository_upsert_record(source)

    assert record["readme"] is None
    assert record["readme_length"] == 0


def test_backend_mapping_is_validated_and_retained_for_the_run():
    session = _Session()
    client = BackendIngestionClient(
        base_url="http://backend.test",
        internal_secret="secret",
        session=session,
    )

    result = client.upsert_repositories_detailed([_source()])

    assert result.succeeded == ["owner/repo"]
    assert result.mappings["123"]["content_version"] == 1
    assert session.calls[0][0].endswith(
        "/api/internal/v2/ingestion/repositories/upsert"
    )
    assert session.calls[0][1]["headers"]["x-internal-secret"] == "secret"


def test_backend_transport_accepts_one_million_character_canonical_readme():
    session = _Session()
    client = BackendIngestionClient(
        base_url="http://backend.test",
        internal_secret="secret",
        session=session,
    )
    source = _source()
    source.readme.raw_markdown = "é" * 1_000_000

    result = client.upsert_repositories_detailed([source])

    assert result.succeeded == ["owner/repo"]
    encoded = session.calls[0][1]["data"]
    assert len(encoded) > 256 * 1024
    assert len(encoded) < 8 * 1024 * 1024


def test_backend_transport_rejects_warning_result_before_publish():
    session = _Session()
    client = BackendIngestionClient(
        base_url="http://backend.test",
        internal_secret="secret",
        session=session,
    )
    source = _source(warnings=["README fetch failed: TimeoutError"])

    result = client.upsert_repositories_detailed([source])

    assert result.succeeded == []
    assert "owner/repo" in result.failed
    assert "incomplete" in result.failed["owner/repo"]
    assert session.calls == []


def test_trending_snapshot_is_enriched_and_published_atomically():
    backend = SimpleNamespace(publish_trending_snapshot=lambda payload: captured.append(payload))
    storage = BackendTrendingStorage.__new__(BackendTrendingStorage)
    storage.backend = backend
    storage.enricher = SimpleNamespace(get_repositories_batch=lambda _repos: [_source()])
    storage._last_refresh = None
    captured = []

    count = storage.upsert_repositories(
        [{"full_name": "owner/repo", "daily_stars": 7}]
    )

    assert count == 1
    assert captured[0]["period"] == "daily"
    assert captured[0]["repositories"][0]["rank"] == 1
    assert captured[0]["repositories"][0]["score"] == 7.0
    assert "repo_id" not in captured[0]["repositories"][0]


def test_trending_snapshot_preserves_the_complete_canonical_readme():
    canonical_readme = "# Full README\n\n" + ("architecture detail " * 300)
    captured = []
    backend = SimpleNamespace(
        publish_trending_snapshot=lambda payload: captured.append(payload)
    )
    storage = BackendTrendingStorage.__new__(BackendTrendingStorage)
    storage.backend = backend
    storage.enricher = SimpleNamespace(
        get_repositories_batch=lambda _repos: [_source(readme=canonical_readme)]
    )
    storage._last_refresh = None

    count = storage.upsert_repositories([{"full_name": "owner/repo"}])

    assert count == 1
    assert len(canonical_readme) > 4_000
    assert captured[0]["repositories"][0]["readme"] == canonical_readme


def test_trending_snapshot_never_publishes_a_warning_result():
    captured = []
    backend = SimpleNamespace(
        publish_trending_snapshot=lambda payload: captured.append(payload)
    )
    storage = BackendTrendingStorage.__new__(BackendTrendingStorage)
    storage.backend = backend
    storage.enricher = SimpleNamespace(
        get_repositories_batch=lambda _repos: [
            _source(warnings=["README fetch failed: TimeoutError"])
        ]
    )
    storage._last_refresh = None

    with pytest.raises(RuntimeError, match="README acquisition failed"):
        storage.upsert_repositories([{"full_name": "owner/repo"}])

    assert captured == []
    assert storage.get_last_refresh_time() is None


def test_trending_snapshot_fails_atomically_when_full_readmes_exceed_transport():
    captured = []
    backend = SimpleNamespace(
        publish_trending_snapshot=lambda payload: captured.append(payload)
    )
    sources = [
        _source(
            full_name=f"owner/repo-{index}",
            github_id=str(1_000 + index),
            readme="x" * 1_000_000,
        )
        for index in range(9)
    ]
    storage = BackendTrendingStorage.__new__(BackendTrendingStorage)
    storage.backend = backend
    storage.enricher = SimpleNamespace(get_repositories_batch=lambda _repos: sources)
    storage._last_refresh = None

    with pytest.raises(ValueError, match="8 MiB"):
        storage.upsert_repositories(
            [{"full_name": source.repo_id} for source in sources]
        )

    assert captured == []
    assert storage.get_last_refresh_time() is None


def test_trending_worker_has_no_direct_qdrant_delivery_path():
    import trending.config as config

    assert not hasattr(config, "TRENDING_QDRANT_SYNC_ENABLED")
    assert not hasattr(config, "TRENDING_QDRANT_SYNC_STR")
    assert find_spec("trending.qdrant_sync") is None


def test_trending_cli_rejects_retired_direct_qdrant_flags():
    with pytest.raises(SystemExit):
        parse_args(["--once", "--sync-qdrant"])
    with pytest.raises(SystemExit):
        parse_args(["--once", "--no-sync-qdrant"])
