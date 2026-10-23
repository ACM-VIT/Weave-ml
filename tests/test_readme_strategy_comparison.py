"""Offline protocol checks; no GitHub credentials, model or service required."""
import hashlib
import json

import pytest

from acquisition.graphql_queries import README_CANDIDATES
from scripts.compare_readme_strategies import (
    MAX_CALLS, Measurement, compare_repository, load_sample, main,
)

COMMIT = "a" * 40


def blob(text):
    raw = text.encode("utf-8")
    return {"text": text, "oid": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest(),
            "byteSize": len(raw), "isBinary": False}


class Fixture:
    def __init__(self, files=None, *, errors=False, visibility="PUBLIC", branch="release/next"):
        self.files = files or {}
        self.errors = errors
        self.visibility = visibility
        self.branch = branch
        self.queries = []

    def __call__(self, query, variables):
        self.queries.append(query)
        assert variables["owner"] == "Example" and variables["name"] == "project"
        if "query GetReadmeRef(" in query:
            repo = {"id": "R_fixture", "visibility": self.visibility,
                    "defaultBranchRef": {"name": self.branch, "target": {"oid": COMMIT}}}
        elif "query GetReadmeBlob(" in query:
            repo = {"object": next(blob(text) for text in self.files.values() if blob(text)["oid"] == variables["oid"])}
        else:
            assert "HEAD:" not in query and COMMIT + ":" in query
            metadata = "query GetReadmeManifest(" in query
            repo = {alias: (blob(self.files[path]) if path in self.files else None) for alias, path in README_CANDIDATES}
            if metadata:
                for item in repo.values():
                    if item is not None:
                        item.pop("text")
            else:
                repo["defaultBranchRef"] = {"name": self.branch}
            if self.errors:
                repo.pop(README_CANDIDATES[0][0])
        response = {"data": {"repository": repo, "rateLimit": {"cost": 1, "remaining": 1000}}}
        if self.errors and "query GetReadmeRef(" not in query and "query GetReadmeBlob(" not in query:
            response["errors"] = [{"message": "sensitive-provider-detail-do-not-record"}]
        return response


@pytest.mark.parametrize("files,path", [
    ({".github/README.md": "preferred", "README.md": "other"}, ".github/README.md"),
    ({"README.md": "", "docs/readme.md": "nested ☃"}, "docs/readme.md"),
    ({"README.rst": "canonical\n======="}, "README.rst"),
    ({}, None),
])
def test_real_parsers_preserve_precedence_and_content(files, path):
    transport = Fixture(files)
    measured = Measurement(transport)
    result = compare_repository("Example/project", measured)
    assert result["content_path_branch_equal"] is True
    assert result["strategies"]["metadata"]["source_path"] == path
    assert len(measured.calls) == (4 if files else 3)
    assert all(call["cost"] == 1 for call in measured.calls)
    if files:
        assert result["media_base_equal"] is False  # Deliberately reported branch-vs-commit media URL.
        assert COMMIT in result["strategies"]["metadata"]["base_url"]
    encoded = json.dumps(result)
    assert "sensitive-provider-detail" not in encoded
    for text in files.values():
        if len(text) > 2:
            assert text not in encoded


def test_partial_missing_is_failure_and_never_confirmed_empty():
    result = compare_repository("Example/project", Measurement(Fixture(errors=True)))
    assert result["status"] == "inconclusive"
    assert all(value["status"] == "failed" for value in result["strategies"].values())
    assert "sensitive-provider-detail" not in json.dumps(result)


def test_usable_lower_precedence_partial_stays_explicit_transport_evidence():
    result = compare_repository("Example/project", Measurement(Fixture({"README.md": "usable"}, errors=True)))
    assert result["content_path_branch_equal"] is True
    # Partial usable output is source-compatible, not proof the response was complete.
    assert result["strategies"]["aliases"]["requests"][0]["graphql_errors"] is True


@pytest.mark.parametrize("visibility", ["PRIVATE", "INTERNAL", None])
def test_nonpublic_never_fetches_readme(visibility):
    transport = Fixture(visibility=visibility)
    result = compare_repository("Example/project", Measurement(transport))
    assert result["status"] == "unresolved_or_nonpublic"
    assert len(transport.queries) == 1


def test_missing_commit_stays_inconclusive():
    fixture = Fixture()
    def transport(query, variables):
        response = fixture(query, variables)
        response["data"]["repository"]["defaultBranchRef"] = None
        return response
    result = compare_repository("Example/project", Measurement(transport))
    assert result["status"] == "unresolved_commit"
    assert len(fixture.queries) == 1


def test_blob_identity_failure_not_reported_as_parity():
    fixture = Fixture({"README.md": "original"})
    def transport(query, variables):
        response = fixture(query, variables)
        if "query GetReadmeBlob(" in query:
            response["data"]["repository"]["object"]["text"] = "modified"
        return response
    result = compare_repository("Example/project", Measurement(transport))
    assert result["status"] == "inconclusive"
    assert result["strategies"]["metadata"]["status"] == "failed"


def test_budget_counts_failures_and_never_retries():
    def fail(query, variables):
        raise OSError("do-not-record-this")
    measured = Measurement(fail)
    for _ in range(MAX_CALLS):
        with pytest.raises(OSError):
            measured.execute("query", {})
    with pytest.raises(RuntimeError, match="budget"):
        measured.execute("query", {})
    assert len(measured.calls) == MAX_CALLS
    assert "do-not-record-this" not in json.dumps(measured.calls)


def test_rate_reserve_and_unknown_cost():
    measured = Measurement(lambda *_: {"data": {"rateLimit": {"remaining": 5}}})
    with pytest.raises(RuntimeError, match="reserve"):
        measured.execute("query", {})
    assert measured.calls[0]["cost"] is None and measured.calls[0]["ok"] is False


def test_expired_deadline_prevents_transport_call():
    measured = Measurement(lambda *_: pytest.fail("Expired transport called"))
    measured.deadline = 0
    with pytest.raises(RuntimeError, match="budget"):
        measured.execute("query", {})
    assert measured.calls == []


def test_response_bound_rejects_oversized_envelope(monkeypatch):
    monkeypatch.setattr("scripts.compare_readme_strategies.MAX_RESPONSE_BYTES", 32)
    measured = Measurement(lambda *_: {"data": {"content": "x" * 64}})
    with pytest.raises(ValueError, match="bound"):
        measured.execute("query", {})
    assert measured.calls[0]["ok"] is False


def test_dry_run_ignores_tokens_and_never_opens_transport(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_TOKEN", "never-read-service-token")
    monkeypatch.setattr("requests.Session.request", lambda *a, **k: pytest.fail("Network called"))
    assert main([]) == 0
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["mode"] == "plan_only" and report["max_graphql_requests"] == 40
    assert "never-read-service-token" not in output


@pytest.mark.parametrize("repos", [["../bad"], ["a/b", "A/B"], [], ["a/b"] * 11])
def test_invalid_samples_rejected(tmp_path, repos):
    path = tmp_path / "sample.json"
    path.write_text(json.dumps({"repositories": repos}))
    with pytest.raises(ValueError):
        load_sample(path)
