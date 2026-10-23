"""Backend v2 ingestion boundary tests."""

from __future__ import annotations

from importlib.util import find_spec
import json
from types import SimpleNamespace
import uuid

import pytest

from acquisition.models import AcquisitionFailure, AcquisitionRunResult
from acquisition import backend_client
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


class _WireSession:
    """HTTP boundary double that maps the actual received JSON records."""

    def __init__(self, statuses=()):
        self.calls = []
        self.statuses = iter(statuses)

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        records = json.loads(kwargs["data"])["repositories"]
        response = _Response(
            {
                "mappings": [
                    {
                        "github_id": record["github_id"],
                        "repo_id": str(uuid.UUID(int=int(record["github_id"]))),
                        "content_version": int(record["github_id"]),
                    }
                    for record in reversed(records)
                ]
            }
        )
        response.status_code = next(self.statuses, 200)
        return response


def _wire_client(session):
    return BackendIngestionClient(
        base_url="http://backend.test", internal_secret="secret", session=session
    )


def _wire_body(sources):
    return json.dumps(
        {"repositories": [repository_upsert_record(source) for source in sources]},
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


@pytest.mark.parametrize("spare_bytes", [0, -1])
def test_backend_batches_use_exact_utf8_envelope_size(monkeypatch, spare_bytes):
    sources = [
        _source(
            full_name=f"owner/repo-{i}",
            github_id=str(100 + i),
            readme='雪😀"\n\\\x00' * 10,
        )
        for i in range(2)
    ]
    expected = _wire_body(sources)
    limit = len(expected) + spare_bytes
    monkeypatch.setattr(backend_client, "MAX_BODY_BYTES", limit)
    session = _WireSession()

    result = _wire_client(session).upsert_repositories_detailed(sources)

    assert sorted(result.succeeded) == [source.repo_id for source in sources]
    assert not result.failed
    bodies = [call[1]["data"] for call in session.calls]
    assert all(len(body) <= limit for body in bodies)
    if spare_bytes == 0:
        assert bodies == [expected]
    else:
        assert bodies == [_wire_body([source]) for source in sources]


def test_backend_serializes_each_record_once_across_count_bounded_batches(monkeypatch):
    sources = [
        _source(full_name=f"owner/repo-{i}", github_id=str(1000 + i))
        for i in range(101)
    ]
    dumps = json.dumps
    encoded_values = []

    def capture(value, **kwargs):
        encoded_values.append(value)
        return dumps(value, **kwargs)

    monkeypatch.setattr(backend_client.json, "dumps", capture)
    session = _WireSession()
    result = _wire_client(session).upsert_repositories_detailed(sources)

    assert not result.failed
    assert sorted(result.succeeded) == sorted(source.repo_id for source in sources)
    assert [
        len(json.loads(call[1]["data"])["repositories"])
        for call in session.calls
    ] == [100, 1]
    assert len(encoded_values) == len(sources)
    assert [value["github_id"] for value in encoded_values] == [
        source.payload["github_id"] for source in sources
    ]
    assert all(
        mapping["content_version"] == int(github_id)
        for github_id, mapping in result.mappings.items()
    )


@pytest.mark.parametrize("oversized_index", [0, 1])
def test_backend_isolates_oversized_record_without_losing_valid_neighbors(
    monkeypatch, oversized_index
):
    valid = _source(full_name="owner/valid", github_id="100", readme="small")
    oversized = _source(full_name="owner/oversized", github_id="101", readme="x" * 4000)
    sources = [valid]
    sources.insert(oversized_index, oversized)
    monkeypatch.setattr(backend_client, "MAX_BODY_BYTES", len(_wire_body([valid])))
    session = _WireSession()

    result = _wire_client(session).upsert_repositories_detailed(sources)

    assert result.succeeded == ["owner/valid"]
    assert set(result.failed) == {"owner/oversized"}
    assert "body limit" in result.failed["owner/oversized"]
    assert len(session.calls) == 1
    assert json.loads(session.calls[0][1]["data"])["repositories"][0]["github_id"] == "100"


def test_backend_retry_reuses_identical_bytes_and_request_identity(monkeypatch):
    session = _WireSession(statuses=[503, 200])
    monkeypatch.setattr(backend_client.time, "sleep", lambda _delay: None)
    sources = [
        _source(full_name=f"owner/repo-{i}", github_id=str(100 + i), readme='é"\n' * 10)
        for i in range(2)
    ]
    expected = _wire_body(sources)

    result = _wire_client(session).upsert_repositories_detailed(sources)

    assert not result.failed
    assert result.succeeded == ["owner/repo-1", "owner/repo-0"]
    first, retry = [call[1] for call in session.calls]
    assert first["data"] == retry["data"] == expected
    assert first["data"] is retry["data"]
    assert first["headers"]["x-request-id"] == retry["headers"]["x-request-id"]


def test_backend_empty_delivery_does_not_publish_an_empty_batch():
    session = _WireSession()
    result = _wire_client(session).upsert_repositories_detailed([])

    assert not result.succeeded
    assert not result.failed
    assert session.calls == []


def test_trending_snapshot_is_enriched_and_published_atomically():
    backend = SimpleNamespace(publish_trending_snapshot=lambda payload: captured.append(payload))
    storage = BackendTrendingStorage(backend=backend, github_token="fixture-token")
    storage._enrich_candidates = lambda _repos: AcquisitionRunResult(repositories=[_source()])
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
    storage = BackendTrendingStorage(backend=backend, github_token="fixture-token")
    storage._enrich_candidates = lambda _repos: AcquisitionRunResult(
        repositories=[_source(readme=canonical_readme)]
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
    storage = BackendTrendingStorage(backend=backend, github_token="fixture-token")
    storage._enrich_candidates = lambda _repos: AcquisitionRunResult(
        repositories=[_source(warnings=["README fetch failed: TimeoutError"])]
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
    storage = BackendTrendingStorage(backend=backend, github_token="fixture-token")
    storage._enrich_candidates = lambda _repos: AcquisitionRunResult(repositories=sources)
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


def test_trending_retry_reuses_successes_and_restores_original_rank_order():
    captured, calls = [], []
    backend = SimpleNamespace(publish_trending_snapshot=captured.append)
    storage = BackendTrendingStorage(backend=backend, github_token="fixture-token")
    first = _source(full_name="owner/first", github_id="101")
    second = _source(full_name="owner/second", github_id="102")
    def enrich(candidates):
        calls.append([candidate["full_name"] for candidate in candidates])
        if len(calls) == 1:
            return AcquisitionRunResult(repositories=[second], failures=[
                AcquisitionFailure("owner/first", "readme_enrichment", "transient")
            ])
        return AcquisitionRunResult(repositories=[first])
    storage._enrich_candidates = enrich
    candidates = [{"full_name": "owner/first", "daily_stars": 9}, {"full_name": "owner/second", "daily_stars": 4}]
    with pytest.raises(RuntimeError, match="incomplete trending snapshot"):
        storage.upsert_repositories(candidates)
    assert captured == []
    assert storage.get_last_refresh_time() is None
    assert storage.upsert_repositories(candidates) == 2
    assert calls == [["owner/first", "owner/second"], ["owner/first"]]
    assert [(r["full_name"], r["rank"], r["score"], r["description"]) for r in captured[0]["repositories"]] == [
        ("owner/first", 1, 9.0, "demo"), ("owner/second", 2, 4.0, "demo")
    ]
    assert storage.last_enrichment_counts == {"fetched": 1, "reused": 1, "failed": 0, "fallback": 0}
    assert storage._retry_records == {}


def test_backend_malformed_mapping_suffix_does_not_acknowledge_a_valid_prefix():
    session = _Session()
    valid = {"github_id": "123", "repo_id": str(uuid.uuid4()), "content_version": 1}
    session.post = lambda *_args, **_kwargs: _Response({"mappings": [valid, {"github_id": "999"}]})
    result = BackendIngestionClient(
        base_url="https://backend.test", internal_secret="test", session=session,
    ).upsert_repositories_detailed([_source()])
    assert result.succeeded == []
    assert result.mappings == {}
    assert set(result.failed) == {"owner/repo"}


def test_invalid_delivery_record_is_classified_as_permanent():
    source = _source()
    source.payload["github_id"] = "invalid"
    result = BackendIngestionClient(
        base_url="https://backend.test", internal_secret="test", session=_Session(),
    ).upsert_repositories_detailed([source])
    assert result.failure_dispositions == {"owner/repo": "permanent"}


def test_backend_auth_failure_pauses_worker_without_exhausting_repository_attempts():
    import requests
    from acquisition.backend_client import BackendConfigurationError

    class Unauthorized(_Response):
        status_code = 401

        def raise_for_status(self):
            raise requests.HTTPError("credential-bearing message must not be persisted", response=self)

    session = _Session()
    session.post = lambda *_args, **_kwargs: Unauthorized({})
    client = BackendIngestionClient(
        base_url="https://backend.test", internal_secret="test", session=session,
    )
    with pytest.raises(BackendConfigurationError, match="configuration failed"):
        client.upsert_repositories_detailed([_source()])


@pytest.mark.parametrize('invalid', [
    {'github_id': 'unexpected', 'repo_id': 'invalid', 'content_version': 1},
    {'github_id': '123', 'repo_id': str(uuid.uuid4()), 'content_version': 1},
    {'github_id': '124', 'repo_id': 'invalid', 'content_version': 1},
])
def test_ambiguous_mapping_suffix_does_not_acknowledge_a_valid_prefix(invalid):
    class Session:
        def post(self, *_args, **_kwargs):
            return _Response({'mappings': [
                {'github_id': '123', 'repo_id': str(uuid.uuid4()), 'content_version': 1}, invalid,
            ]})
    client = BackendIngestionClient(base_url='http://backend.test', internal_secret='test', session=Session())
    result = client.upsert_repositories_detailed([
        _source(full_name='owner/one', github_id='123'),
        _source(full_name='owner/two', github_id='124'),
    ])
    assert result.succeeded == [] and result.mappings == {}
    assert set(result.failed) == {'owner/one', 'owner/two'}
    assert client.get_repo_count() == 0 and client._accepted_names == set()
