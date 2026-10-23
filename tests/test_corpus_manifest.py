"""Inventory contract and actual acquisition selection, without product SQL."""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import requests

from acquisition.backend_client import BackendIngestionClient
from acquisition.manifest import ManifestUnavailable, load_manifest
from acquisition.models import AcquisitionRunResult
from acquisition.pipeline import run_acquisition_detailed

NOW = datetime.now(timezone.utc)


def row(number, name=None, *, stale=False):
    return {
        "repo_id": str(uuid.UUID(int=number)),
        "github_id": str(number),
        "full_name": name or f"owner/repo{number}",
        "content_version": 1,
        "content_hash": "sha256-a",
        "source_updated_at": None,
        "content_updated_at": (NOW - timedelta(days=2 if stale else 0.001)).isoformat(),
    }


def page(items, next_cursor=None, **kwargs):
    return dict(
        version=1, as_of=NOW.isoformat(), items=items, next_cursor=next_cursor, **kwargs
    )


class Response:
    def __init__(self, payload, *, status=200):
        self.payload, self.status = payload, status

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def raise_for_status(self):
        if self.status != 200:
            raise requests.HTTPError("test backend error")

    def iter_content(self, _):
        yield json.dumps(self.payload).encode()


class Session:
    def __init__(self, pages):
        self.pages, self.calls, self.bytes = list(pages), [], 0

    def get(self, url, **kwargs):
        self.calls.append(kwargs)
        payload = self.pages.pop(0)
        self.bytes += len(json.dumps(payload).encode())
        return Response(payload)


def client(pages):
    session = Session(pages)
    return BackendIngestionClient(
        base_url="http://127.0.0.1", internal_secret="test", session=session
    ), session


def test_paginated_manifest_case_normalization_freshness_and_one_scan_per_invocation():
    db, session = client(
        [page([row(1, "Owner/Fresh")], "signed-page2"), page([row(2, stale=True)])]
    )
    assert db.get_existing_repository_names() == {"owner/fresh"}
    assert db.get_existing_repository_names() == {"owner/fresh"}
    assert db.manifest_status == "available"
    assert len(session.calls) == 2
    assert session.calls[1]["params"]["cursor"] == "signed-page2"
    assert db.get_repo_count() == 0  # Inventory count is never invocation progress.


@pytest.mark.parametrize(
    "pages",
    [
        [page([row(1)], "same"), page([row(2)], "same")],
        [page([row(1, "Owner/Repo")], "next"), page([row(2, "owner/repo")])],
        [page([row(2)], "next"), page([row(1)])],
        [
            page([row(1)], "next"),
            {**page([row(2)]), "as_of": (NOW - timedelta(seconds=1)).isoformat()},
        ],
        [{**page([]), "version": 2}],
        [{**page([]), "items": [row(1)] * 251}],
        [page([{**row(1), "content_version": True}])],
        [page([{**row(1), "github_id": 1}])],
        [page([{**row(1), "content_updated_at": "not-a-date"}])],
    ],
)
def test_invalid_or_partial_manifest_is_not_committed(pages):
    db, _ = client(pages)
    assert db.get_existing_repository_names() == set()
    assert db.manifest_status == "unavailable" and db._manifest is None


def test_empty_and_unavailable_are_distinct_with_backwards_compatible_enrichment():
    empty, _ = client([page([])])
    unavailable, _ = client([])  # Old backend/transport failure -> no inventory.
    assert (
        empty.get_existing_repository_names()
        == unavailable.get_existing_repository_names()
        == set()
    )
    assert empty.manifest_status == "empty"
    assert unavailable.manifest_status == "unavailable"


def test_oversize_response_fails_before_inventory_adoption():
    db, _ = client([page([row(1, "o/" + "r" * 600_000)])])
    with pytest.raises(ManifestUnavailable):
        load_manifest(db)


def test_second_run_avoids_enrichment_but_stale_candidate_is_refreshable(monkeypatch):
    from acquisition import pipeline
    from acquisition.github_discovery import GitHubDiscoveryEngine

    monkeypatch.setattr(
        GitHubDiscoveryEngine,
        "discover",
        lambda *_a, **_kw: ["Owner/Fresh", "owner/stale", "OWNER/FRESH"],
    )
    calls = []

    def enrich(_token, targets, **_kwargs):
        calls.append(list(targets))
        return AcquisitionRunResult(repositories=list(targets))

    monkeypatch.setattr(pipeline, "enrich_repositories", enrich)
    # First run with verified empty corpus enriches both unique candidates.
    first, _ = client([page([])])
    run_acquisition_detailed(
        "fixture", existing_repos=first.get_existing_repository_names()
    )
    # Second unchanged inventory excludes fresh content, but not a stale record.
    second, _ = client(
        [page([row(1, "owner/fresh"), row(2, "owner/stale", stale=True)])]
    )
    result = run_acquisition_detailed(
        "fixture", existing_repos=second.get_existing_repository_names()
    )
    assert calls == [["Owner/Fresh", "owner/stale"], ["owner/stale"]]
    assert result.skipped_existing_count == 1 and result.duplicates_removed == 1


@pytest.mark.parametrize("count", [250, 1000, 5000])
def test_manifest_transfer_scales_by_bounded_pages_without_readme(
    count, record_property
):
    items = [row(i + 1) for i in range(count)]
    pages = [
        page(items[i : i + 250], f"page-{i + 250}" if i + 250 < count else None)
        for i in range(0, count, 250)
    ]
    db, session = client(pages)
    assert len(db.get_existing_repository_names()) == count
    assert len(session.calls) == count // 250
    record_property("rows", count)
    record_property("calls", len(session.calls))
    record_property("manifest_bytes", session.bytes)
    record_property("avoided_enrichments", count)
    assert session.bytes < count * 400
