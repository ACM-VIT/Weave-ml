"""Query bounds, cursor safety and metadata projection contract fixtures."""

import re
from unittest.mock import MagicMock

import pytest

from acquisition.github_discovery import DiscoveryConfig, GitHubDiscoveryEngine
from acquisition.github_graphql_client import GitHubGraphQLClient, GitHubGraphQLClientError
from acquisition.graphql_queries import README_CANDIDATES, build_batch_metadata_query
from acquisition.pipeline import enrich_repositories
from acquisition.repository_enricher import RepositoryEnricher


@pytest.mark.parametrize("size", [1, 100])
def test_search_passes_effective_server_side_first(size):
    client = MagicMock()
    client.execute.return_value = {"data": {"search": {"nodes": []}}}
    GitHubDiscoveryEngine(client, config=DiscoveryConfig(per_query=size))._search_graphql("topic:test", size)
    query, variables = client.execute.call_args.args
    assert "$first: Int!" in query and "first: $first" in query
    assert variables == {"query": "topic:test", "first": size, "after": None}
    assert "databaseId" not in query


@pytest.mark.parametrize("field,value", [
    ("per_query", 0), ("per_query", 101), ("per_query", True), ("per_query", 1.5),
    ("pages_per_query", 0), ("pages_per_query", 11), ("pages_per_query", False),
])
def test_invalid_search_configuration_is_rejected(field, value):
    with pytest.raises(ValueError):
        DiscoveryConfig(**{field: value})


@pytest.mark.parametrize("partial", [False, True])
def test_search_pages_follow_cursor_but_stop_after_partial_response(partial):
    client = MagicMock()
    page = {"data": {"search": {
        "nodes": [{"nameWithOwner": "owner/first"}, None],
        "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"},
    }}}
    if partial:
        page["errors"] = [{"message": "partial page", "path": ["search", "nodes", 1]}]
    client.execute.side_effect = [page, {"data": {"search": {
        "nodes": [{"nameWithOwner": "owner/second"}],
        "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"},
    }}}]
    results = GitHubDiscoveryEngine(client, config=DiscoveryConfig(pages_per_query=3))._search_graphql("test", 2)
    assert [item["full_name"] for item in results] == (["owner/first"] if partial else ["owner/first", "owner/second"])
    assert client.execute.call_count == (1 if partial else 2)
    if not partial:
        assert client.execute.call_args_list[1].args[1]["after"] == "cursor-1"


def test_later_search_page_failure_keeps_completed_page(caplog):
    client = MagicMock()
    client.execute.side_effect = [{"data": {"search": {
        "nodes": [{"nameWithOwner": "owner/first"}],
        "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"},
    }}}, GitHubGraphQLClientError("private response body", category="server", status_code=502)]
    results = GitHubDiscoveryEngine(client, config=DiscoveryConfig(pages_per_query=3))._search_graphql("test", 1)
    assert [item["full_name"] for item in results] == ["owner/first"]
    assert client.execute.call_count == 2
    assert "server" in caplog.text and "502" in caplog.text
    assert "private response body" not in caplog.text


def test_invalid_json_on_later_page_keeps_completed_page():
    first = MagicMock(status_code=200)
    first.json.return_value = {"data": {"search": {
        "nodes": [{"nameWithOwner": "owner/first"}],
        "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"},
    }}}
    invalid = MagicMock(status_code=200)
    invalid.json.side_effect = ValueError("malformed JSON")
    session = MagicMock()
    session.post.side_effect = [first, invalid]
    client = GitHubGraphQLClient(token="test-token", session=session)

    results = GitHubDiscoveryEngine(client, config=DiscoveryConfig(pages_per_query=3))._search_graphql("test", 1)

    assert [item["full_name"] for item in results] == ["owner/first"]
    assert session.post.call_count == 2


def test_invalid_json_on_first_page_is_still_a_failure():
    invalid = MagicMock(status_code=200)
    invalid.json.side_effect = ValueError("malformed JSON")
    session = MagicMock()
    session.post.return_value = invalid
    client = GitHubGraphQLClient(token="test-token", session=session)

    with pytest.raises(GitHubGraphQLClientError, match="invalid JSON"):
        GitHubDiscoveryEngine(client)._search_graphql("test", 1)


def test_later_page_http_diagnostic_does_not_log_response_body(caplog):
    first = MagicMock(status_code=200)
    first.json.return_value = {"data": {"search": {
        "nodes": [{"nameWithOwner": "owner/first"}],
        "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"},
    }}}
    failure = MagicMock(status_code=401, text="Authorization: Bearer private-secret https://user:password@example.test")
    session = MagicMock()
    session.post.side_effect = [first, failure]
    client = GitHubGraphQLClient(token="test-token", session=session)

    results = GitHubDiscoveryEngine(client, config=DiscoveryConfig(pages_per_query=3))._search_graphql("test", 1)

    assert [item["full_name"] for item in results] == ["owner/first"]
    assert "http" in caplog.text and "401" in caplog.text
    assert "private-secret" not in caplog.text
    assert "password" not in caplog.text


@pytest.mark.parametrize("envelope", [None, [], 1, {"data": None}, {"data": []}, {"data": "invalid"}, {"errors": [None]}, {"errors": "invalid"}])
def test_invalid_response_envelope_preserves_only_completed_pages(envelope):
    first = MagicMock(status_code=200)
    first.json.return_value = {"data": {"search": {
        "nodes": [{"nameWithOwner": "owner/first"}],
        "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"},
    }}}
    invalid = MagicMock(status_code=200)
    invalid.json.return_value = envelope
    session = MagicMock()
    session.post.side_effect = [first, invalid]
    client = GitHubGraphQLClient(token="test-token", session=session)

    results = GitHubDiscoveryEngine(client, config=DiscoveryConfig(pages_per_query=3))._search_graphql("test", 1)
    assert [item["full_name"] for item in results] == ["owner/first"]

    session.post.side_effect = [invalid]
    with pytest.raises(GitHubGraphQLClientError):
        GitHubDiscoveryEngine(client)._search_graphql("test", 1)


def test_unusable_rate_limit_metadata_does_not_drop_search_results():
    response = MagicMock(status_code=200)
    response.json.return_value = {"data": {"rateLimit": None, "search": {"nodes": [{"nameWithOwner": "owner/repo"}]}}}
    session = MagicMock()
    session.post.return_value = response
    client = GitHubGraphQLClient(token="test-token", session=session)
    assert GitHubDiscoveryEngine(client)._search_graphql("test", 1)[0]["full_name"] == "owner/repo"


class MetadataFixture(GitHubGraphQLClient):
    def __init__(self):
        super().__init__(token="test-token")
        self.metadata_queries = []

    def execute(self, query, variables=None):
        if "query GetReadme" in query:
            return {"data": {"repository": {
                "defaultBranchRef": {"name": "main"},
                **{alias: None for alias, _path in README_CANDIDATES},
            }}}
        self.metadata_queries.append(query)
        data = {}
        errors = []
        for alias, owner, name in re.findall(r'(repo_\d+): repository\(owner: "([^"]+)", name: "([^"]+)"\)', query):
            data[alias] = {"nameWithOwner": f"{owner}/{name}", "description": "Healthy project."}
            if name == "failed":
                errors.append({"message": "timeout", "path": [alias, "description"]})
        return {"data": data, "errors": errors}


@pytest.mark.parametrize("size", [1, 3, 10])
def test_metadata_batch_setting_is_separate_from_scheduler_chunks(monkeypatch, size):
    client = MetadataFixture()
    monkeypatch.setenv("GITHUB_METADATA_BATCH_SIZE", str(size))
    monkeypatch.setattr("acquisition.github_graphql_client.GitHubGraphQLClient", lambda **kwargs: client)
    names = [f"owner/repo-{i}" for i in range(13)]
    result = enrich_repositories("test-token", names, batch_size=13, workers=1)
    assert [item.repo_id for item in result.repositories] == names
    assert len(client.metadata_queries) == (13 + size - 1) // size
    assert all(query.count(": repository(") <= size for query in client.metadata_queries)


@pytest.mark.parametrize("value", [0, 11, True, 1.5, "invalid"])
def test_invalid_metadata_batch_configuration_fails_before_requests(value):
    with pytest.raises(ValueError):
        RepositoryEnricher(MetadataFixture(), metadata_batch_size=value)


def test_delivery_projection_omits_only_delivery_unused_fields():
    repos = [("owner", "repo")]
    legacy = build_batch_metadata_query(repos)
    delivery = build_batch_metadata_query(repos, projection="delivery")
    for field in ("homepageUrl", "pullRequests"):
        assert field in legacy and field not in delivery
    for field in ("databaseId", "description", "repositoryTopics", "languages", "history", "issues", "watchers"):
        assert field in delivery
    assert "readme" not in delivery.lower()


def test_delivery_projection_preserves_partial_retry_boundary(monkeypatch):
    client = MetadataFixture()
    monkeypatch.setattr("acquisition.github_graphql_client.GitHubGraphQLClient", lambda **kwargs: client)
    result = enrich_repositories("test-token", ["owner/failed", "owner/healthy"], workers=1, metadata_projection="delivery")
    assert [item.repo_id for item in result.repositories] == ["owner/healthy"]
    assert [failure.repository for failure in result.failures] == ["owner/failed"]
    assert "pullRequests" not in client.metadata_queries[0]


def test_invalid_projection_is_rejected():
    with pytest.raises(ValueError, match="projection"):
        build_batch_metadata_query([("owner", "repo")], projection="unknown")


@pytest.mark.parametrize("size", [0, 11])
def test_metadata_query_builder_rejects_unsafe_alias_counts(size):
    with pytest.raises(ValueError, match="between 1 and 10"):
        build_batch_metadata_query([("owner", f"repo-{i}") for i in range(size)])


def test_delivery_projection_does_not_change_backend_record():
    from acquisition.backend_client import repository_upsert_record

    metadata = {
        "id": "R_123", "databaseId": 123, "nameWithOwner": "owner/repo", "name": "repo",
        "description": "A useful project.", "url": "https://github.com/owner/repo",
        "owner": {"login": "owner", "databaseId": 456}, "stargazerCount": 100,
        "homepageUrl": "https://example.test", "pullRequests": {"totalCount": 9},
    }
    enricher = RepositoryEnricher(MetadataFixture())
    legacy = enricher._process_graphql_data(metadata, None, None)
    delivery = enricher._process_graphql_data({key: value for key, value in metadata.items() if key not in {"homepageUrl", "pullRequests"}}, None, None)
    before, after = repository_upsert_record(legacy), repository_upsert_record(delivery)
    before.pop("observed_at", None)
    after.pop("observed_at", None)
    assert before == after


@pytest.mark.parametrize("projection", ["legacy", "delivery"])
@pytest.mark.parametrize("readme_state", ["present", "missing", "failed"])
def test_metadata_fallback_fetches_readme_only_in_second_pass(projection, readme_state):
    class FallbackFixture(MetadataFixture):
        def __init__(self):
            super().__init__()
            self.readme_calls = []

        def execute(self, query, variables=None):
            if "query GetReadme" in query:
                self.readme_calls.append(variables["name"])
                if readme_state == "failed":
                    raise GitHubGraphQLClientError("temporary README failure")
                response = super().execute(query, variables)
                if readme_state == "present":
                    response["data"]["repository"].update({
                        "readmeGithub1": {"text": "# Canonical community README"},
                        "readme1": {"text": "# Lower priority root README"},
                    })
                return response
            if query.count(": repository(") > 1:
                self.metadata_queries.append(query)
                raise GitHubGraphQLClientError("batch too large", category="server", status_code=502)
            # MetadataFixture marks only the failed identity unresolved, so the
            # healthy singleton must survive and receive exactly one README.
            return super().execute(query, variables)

    client = FallbackFixture()
    results = RepositoryEnricher(client, metadata_projection=projection).get_repositories_batch(
        ["owner/failed", "owner/healthy"]
    )
    assert [result.repo_id for result in results] == ["owner/healthy"]
    assert len(client.metadata_queries) == 3
    assert all("readme" not in query.lower() for query in client.metadata_queries)
    assert all(("pullRequests" in query) == (projection == "legacy") for query in client.metadata_queries)
    assert client.readme_calls == ["healthy"]
    result = results[0]
    assert result.raw_repository["description"] == "Healthy project."
    if readme_state == "present":
        assert result.readme.raw_markdown == "# Canonical community README"
        assert result.readme.source_path == ".github/README.md"
        assert result.readme.default_branch == "main"
        assert result.readme.base_url.endswith("/refs/heads/main/.github/")
    else:
        assert result.readme.raw_markdown == ""
    assert bool(result.warnings) == (readme_state == "failed")
