"""Canonical README acquisition and source-metadata regression tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from acquisition.github_graphql_client import (
    FetchedReadme,
    GitHubGraphQLClient,
    GitHubGraphQLClientError,
    build_readme_base_url,
)
from acquisition.graphql_queries import (
    GET_README_QUERY,
    GET_README_REF_QUERY,
    README_CANDIDATES,
    build_batch_metadata_query,
)
from acquisition.repository_enricher import RepositoryEnricher


def _complete_readme_response(
    *, branch: str | None = "main", errors: list[dict] | None = None, **aliases
) -> dict:
    repository = {
        "defaultBranchRef": None if branch is None else {"name": branch},
        **{alias: None for alias, _path in README_CANDIDATES},
    }
    repository.update(aliases)
    response: dict = {"data": {"repository": repository}}
    if errors is not None:
        response["errors"] = errors
    return response


@pytest.mark.unit
def test_get_readme_retains_selected_path_branch_and_media_base_url():
    markdown = "# Project\n\n![Architecture](docs/architecture.png)"
    client = GitHubGraphQLClient(token="test-token")
    client.execute = MagicMock(
        return_value=_complete_readme_response(
            branch="release/v2", readme2={"text": markdown}
        )
    )

    readme = client.get_readme("weave-org", "weave")

    assert isinstance(readme, str)
    assert readme == markdown
    assert readme.raw_markdown == markdown
    assert readme.source_path == "readme.md"
    assert readme.default_branch == "release/v2"
    assert readme.base_url == (
        "https://raw.githubusercontent.com/weave-org/weave/"
        "refs/heads/release/v2/"
    )


@pytest.mark.unit
def test_readme_queries_request_default_branch_metadata():
    assert "defaultBranchRef { name }" in GET_README_QUERY
    assert "defaultBranchRef {" in build_batch_metadata_query([("owner", "repo")])
    assert "name" in build_batch_metadata_query([("owner", "repo")])
    assert 'HEAD:.github/README.md' in GET_README_QUERY
    assert 'HEAD:docs/README.md' in GET_README_QUERY


@pytest.mark.unit
def test_get_readme_uses_github_precedence_before_root_and_docs():
    client = GitHubGraphQLClient(token="test-token")
    client.execute = MagicMock(
        return_value=_complete_readme_response(
            readmeGithub1={"text": "# Community README"},
            readme1={"text": "# Root README"},
            readmeDocs1={"text": "# Docs README"},
        )
    )

    readme = client.get_readme("owner", "repo")

    assert readme.raw_markdown == "# Community README"
    assert readme.source_path == ".github/README.md"
    assert readme.base_url.endswith("/refs/heads/main/.github/")


@pytest.mark.unit
def test_get_readme_uses_docs_after_root_candidates_are_missing():
    client = GitHubGraphQLClient(token="test-token")
    client.execute = MagicMock(
        return_value=_complete_readme_response(
            readmeDocs1={"text": "# Docs README"}
        )
    )

    readme = client.get_readme("owner", "repo")

    assert readme.raw_markdown == "# Docs README"
    assert readme.source_path == "docs/README.md"
    assert readme.base_url.endswith("/refs/heads/main/docs/")


@pytest.mark.unit
def test_get_readme_tolerates_partial_graphql_errors_when_valid_data_present(caplog):
    """GitHub can return field-level errors alongside a valid README alias.

    In that case the pipeline must NOT abort — it should log a warning and
    return the README that successfully resolved.  Aborting here prevents the
    entire trending snapshot from being published (the P1 reported by greptile).
    """
    import logging

    client = GitHubGraphQLClient(token="test-token")
    client.execute = MagicMock(
        return_value=_complete_readme_response(
            readme1={"text": "# README"},
            errors=[{"message": "README object resolution timed out"}],
        )
    )

    with caplog.at_level(logging.WARNING, logger="acquisition.github_graphql_client"):
        readme = client.get_readme("owner", "repo")

    assert readme.raw_markdown == "# README", (
        "A valid README alias should be returned even when field-level errors are present"
    )
    assert any("field-level errors" in record.message for record in caplog.records), (
        "A warning should be emitted when partial errors accompany valid data"
    )


@pytest.mark.unit
def test_get_readme_rejects_graphql_errors_when_no_usable_data():
    """When errors arrive with NO usable repository data at all, the call must raise."""
    client = GitHubGraphQLClient(token="test-token")
    client.execute = MagicMock(
        return_value={
            "errors": [{"message": "Resource not accessible by integration"}],
            "data": None,
        }
    )

    with pytest.raises(GitHubGraphQLClientError, match="GraphQL errors"):
        client.get_readme("owner", "repo")


@pytest.mark.unit
@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {"data": {}},
        {"data": {"repository": None}},
    ],
)
def test_get_readme_rejects_unresolved_or_missing_repository(response):
    client = GitHubGraphQLClient(token="test-token")
    client.execute = MagicMock(return_value=response)

    with pytest.raises(GitHubGraphQLClientError):
        client.get_readme("owner", "repo")


@pytest.mark.unit
def test_get_readme_accepts_an_incomplete_alias_set():
    response = _complete_readme_response(readme1={"text": "# Valid"})
    del response["data"]["repository"][README_CANDIDATES[-1][0]]
    client = GitHubGraphQLClient(token="test-token")
    client.execute = MagicMock(return_value=response)

    # Should not raise an error now, but return successfully using available aliases
    result = client.get_readme("owner", "repo")
    assert result == "# Valid"


@pytest.mark.unit
def test_get_readme_uses_valid_alias_when_partial_errors_omit_branch_metadata():
    response = _complete_readme_response(readme1={"text": "# Valid"})
    del response["data"]["repository"]["defaultBranchRef"]
    response["errors"] = [{"message": "defaultBranchRef timed out"}]
    client = GitHubGraphQLClient(token="test-token")
    client.execute = MagicMock(return_value=response)

    result = client.get_readme("owner", "repo")

    assert result == "# Valid"
    assert result.default_branch is None
    assert result.source_path == "README.md"
    assert result.base_url == "https://raw.githubusercontent.com/owner/repo/HEAD/"


@pytest.mark.unit
def test_get_readme_skips_an_incomplete_alias_object():
    client = GitHubGraphQLClient(token="test-token")
    client.execute = MagicMock(
        return_value=_complete_readme_response(readme1={}, readme2={"text": "# Valid fallback"})
    )

    readme = client.get_readme("owner", "repo")
    assert readme == "# Valid fallback"


@pytest.mark.unit
def test_get_readme_returns_empty_only_after_every_alias_resolves_null():
    client = GitHubGraphQLClient(token="test-token")
    client.execute = MagicMock(return_value=_complete_readme_response())

    readme = client.get_readme("owner", "repo")

    assert readme == ""
    assert readme.source_path is None


@pytest.mark.parametrize('incomplete', ['failed_alias', 'missing_alias', 'missing_text'])
def test_get_readme_retries_unknown_text_instead_of_confirming_empty(incomplete):
    response = _complete_readme_response()
    if incomplete == 'failed_alias':
        response['errors'] = [{'message': 'timeout', 'path': ['repository', 'readme1', 'text']}]
    elif incomplete == 'missing_alias':
        del response['data']['repository']['readme1']
    else:
        response['data']['repository']['readme1'] = {}
    client = GitHubGraphQLClient(token='test-token')
    client.execute = MagicMock(return_value=response)

    with pytest.raises(GitHubGraphQLClientError):
        client.get_readme('owner', 'repo')


@pytest.mark.parametrize('description', [None, 'still usable text'])
def test_failed_metadata_fields_retry_only_the_affected_repository(description):
    client = GitHubGraphQLClient(token='test-token')
    healthy = {'nameWithOwner': 'owner/healthy', 'description': 'A healthy repository.'}
    client.execute = MagicMock(return_value={
        'data': {'repo_0': {'nameWithOwner': 'owner/failed', 'description': description}, 'repo_1': healthy},
        'errors': [{'message': 'timeout', 'path': ['repo_0', 'description']}],
    })

    assert client.get_repositories_batch([('owner', 'failed'), ('owner', 'healthy')]) == {
        'owner/healthy': healthy,
    }


def test_single_repository_retries_failed_description_instead_of_replacing_text():
    client = GitHubGraphQLClient(token='test-token')
    client.execute = MagicMock(return_value={
        'data': {'repository': {'nameWithOwner': 'owner/repo', 'description': None}},
        'errors': [{'message': 'timeout', 'path': ['repository', 'description']}],
    })

    with pytest.raises(GitHubGraphQLClientError):
        client.get_repository('owner', 'repo')


@pytest.mark.parametrize('alias', ['readmeGithub1', 'readme1', 'readmeDocs1'])
def test_single_repository_keeps_metadata_when_only_a_readme_alias_failed(alias):
    client = GitHubGraphQLClient(token='test-token')
    alternate = next(name for name, _ in README_CANDIDATES if name != alias)
    repo = {'nameWithOwner': 'owner/repo', 'description': 'Resolved project description.',
            alias: None, alternate: {'text': 'Usable canonical README.'}}
    client.execute = MagicMock(return_value={
        'data': {'repository': repo},
        'errors': [{'message': 'README timeout', 'path': ['repository', alias, 'text']}],
    })
    assert client.get_repository('owner', 'repo') == repo


def test_single_repository_retries_unresolved_readme_without_erasing_canonical_text():
    client = GitHubGraphQLClient(token='test-token')
    client.execute = MagicMock(return_value={
        'data': {'repository': {'nameWithOwner': 'owner/repo', 'description': 'Usable metadata.',
                                **{alias: None for alias, _ in README_CANDIDATES}}},
        'errors': [{'message': 'README timeout', 'path': ['repository', 'readme1', 'text']}],
    })
    with pytest.raises(GitHubGraphQLClientError, match='README'):
        client.get_repository('owner', 'repo')


def test_readme_error_does_not_hide_a_simultaneous_metadata_error():
    client = GitHubGraphQLClient(token='test-token')
    client.execute = MagicMock(return_value={
        'data': {'repository': {'nameWithOwner': 'owner/repo', 'description': None}},
        'errors': [
            {'message': 'README timeout', 'path': ['repository', 'readme1']},
            {'message': 'description timeout', 'path': ['repository', 'description']},
        ],
    })
    with pytest.raises(GitHubGraphQLClientError):
        client.get_repository('owner', 'repo')


@pytest.mark.parametrize('errors', [
    [{'message': 'unlocated timeout'}],
    [{'message': 'unrecognized field', 'path': ['unknown']}],
])
def test_unlocated_metadata_errors_retry_instead_of_publishing_partial_rows(errors):
    client = GitHubGraphQLClient(token='test-token')
    client.execute = MagicMock(return_value={
        'data': {'repo_0': {'nameWithOwner': 'owner/repo', 'description': None}},
        'errors': errors,
    })
    with pytest.raises(GitHubGraphQLClientError):
        client.get_repositories_batch([('owner', 'repo')])


def test_partial_metadata_failure_reaches_acquisition_retry_without_losing_healthy_rows(monkeypatch):
    from acquisition.pipeline import enrich_repositories

    client = GitHubGraphQLClient(token='test-token')
    client.execute = MagicMock(side_effect=lambda query, *args: (
        _complete_readme_response() if query in {GET_README_QUERY, GET_README_REF_QUERY} else {
            'data': {
                'repo_0': {'nameWithOwner': 'owner/failed', 'description': None},
                'repo_1': {'nameWithOwner': 'owner/healthy', 'description': 'Healthy project.'},
            },
            'errors': [{'message': 'timeout', 'path': ['repo_0', 'description']}],
        }
    ))
    monkeypatch.setattr('acquisition.github_graphql_client.GitHubGraphQLClient', lambda **kwargs: client)

    result = enrich_repositories('test-token', ['owner/failed', 'owner/healthy'], workers=1)

    assert [repo.repo_id for repo in result.repositories] == ['owner/healthy']
    assert [failure.repository for failure in result.failures] == ['owner/failed']


@pytest.mark.unit
def test_batch_metadata_preserves_usable_partial_response_envelopes(caplog):
    import logging

    client = GitHubGraphQLClient(token="test-token")
    client.execute = MagicMock(
        return_value={
            "data": {"repo_0": {"nameWithOwner": "owner/repo", "description": None}},
            "errors": [{"message": "rate limit metadata timed out", "path": ["rateLimit"]}],
        }
    )

    with caplog.at_level(logging.WARNING, logger="acquisition.github_graphql_client"):
        results = client.get_repositories_batch([("owner", "repo")])

    assert results.get("owner/repo") == {"nameWithOwner": "owner/repo", "description": None}


@pytest.mark.unit
def test_graphql_transport_preserves_partial_error_data_envelope(caplog):
    import logging

    response = MagicMock()
    response.status_code = 200
    response.headers = {}
    response.json.return_value = {
        "data": {"repository": {"name": "repo"}},
        "errors": [{"message": "description field timed out"}],
    }
    session = MagicMock()
    session.post.return_value = response
    client = GitHubGraphQLClient(
        token="test-token", session=session, max_retries=0
    )

    with caplog.at_level(logging.WARNING, logger="acquisition.github_graphql_client"):
        result = client.execute("query { repository { name } }")

    assert result == {
        "data": {"repository": {"name": "repo"}},
        "errors": [{"message": "description field timed out"}],
    }
    assert any("partial errors" in record.message for record in caplog.records)


@pytest.mark.unit
def test_transient_readme_failure_propagates_for_acquisition_retry():
    client = GitHubGraphQLClient(token="test-token")
    client.execute = MagicMock(side_effect=TimeoutError("temporary GitHub timeout"))

    with pytest.raises(TimeoutError, match="temporary GitHub timeout"):
        client.get_readme("weave-org", "weave")


@pytest.mark.unit
def test_batch_enrichment_surfaces_readme_failure_as_a_retry_warning():
    graphql_client = MagicMock()
    graphql_client.get_repositories_batch.return_value = {
        "owner/repo": {
            "id": "R_test",
            "databaseId": 123,
            "nameWithOwner": "owner/repo",
            "name": "repo",
            "description": "demo",
            "url": "https://github.com/owner/repo",
            "owner": {"login": "owner", "databaseId": 456},
            "defaultBranchRef": {"name": "main"},
            "repositoryTopics": {"nodes": []},
            "languages": {"edges": []},
            "stargazers": {"edges": []},
        }
    }
    graphql_client.get_readme.side_effect = TimeoutError("temporary README timeout")

    results = RepositoryEnricher(
        graphql_client=graphql_client
    ).get_repositories_batch([{"full_name": "owner/repo"}])

    assert len(results) == 1
    assert results[0].readme.raw_markdown == ""
    assert results[0].warnings
    assert "temporary README timeout" in results[0].warnings[0]


@pytest.mark.unit
def test_readme_base_url_uses_readme_directory_and_https_only():
    assert build_readme_base_url(
        "owner", "repo", "main", "docs/README.md"
    ) == (
        "https://raw.githubusercontent.com/owner/repo/refs/heads/main/docs/"
    )


@pytest.mark.unit
def test_inline_readme_is_preserved_without_model_rewrite():
    raw_markdown = (
        "# Project\n\n![Preview](assets/preview.png)\n\n"
        "A sufficiently detailed project overview for developers and teams."
    )
    data = {
        "id": "R_test",
        "databaseId": 123,
        "nameWithOwner": "owner/repo",
        "name": "repo",
        "description": "demo",
        "url": "https://github.com/owner/repo",
        "owner": {"login": "owner", "databaseId": 456},
        "defaultBranchRef": {"name": "main", "target": {"history": {"nodes": []}}},
        "readme1": {"text": raw_markdown},
        "repositoryTopics": {"nodes": []},
        "languages": {"edges": []},
        "stargazers": {"edges": []},
    }

    result = RepositoryEnricher(graphql_client=MagicMock())._process_graphql_data(
        data, None, None
    )

    assert result is not None
    assert result.readme.raw_markdown == raw_markdown
    assert result.readme.clean_text != raw_markdown
    assert result.readme.readme_md == ""
    assert result.readme.source_path == "README.md"
    assert result.readme.default_branch == "main"
    assert result.readme.base_url == (
        "https://raw.githubusercontent.com/owner/repo/refs/heads/main/"
    )
