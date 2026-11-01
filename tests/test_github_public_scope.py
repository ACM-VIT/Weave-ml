"""Public-only authorization boundaries for offline GitHub acquisition."""

from __future__ import annotations

from urllib.parse import urlsplit

import pytest

from acquisition.github_discovery import DiscoveryConfig, GitHubDiscoveryEngine
from acquisition.github_graphql_client import GitHubClientError, GitHubGraphQLClient


class _Response:
    def __init__(self, status_code: int = 200, *, scopes: str | None = None) -> None:
        self.status_code = status_code
        self.headers = {} if scopes is None else {"X-OAuth-Scopes": scopes}
        self.closed = False

    def json(self):
        return {"data": {"viewer": {"login": "public-only"}}}

    def close(self):
        self.closed = True


class _Session:
    def __init__(self, token: str, scopes: str | None) -> None:
        self.headers = {"Authorization": f"Bearer {token}"}
        self.auth = None
        self.scopes = scopes
        self.get_calls = []
        self.post_calls = []
        self.scope_response = None

    def get(self, url: str, **kwargs):
        self.get_calls.append((url, kwargs, self.headers.get("Authorization")))
        self.scope_response = _Response(scopes=self.scopes)
        return self.scope_response

    def post(self, url: str, **kwargs):
        self.post_calls.append((url, kwargs, self.headers.get("Authorization")))
        return _Response()


@pytest.mark.parametrize(
    ("token", "scopes"),
    [
        ("ghp_public_only", "public_repo"),
        ("gho_public_only", "public_repo, read:user, user:email"),
        ("ghp_no_scope", ""),
    ],
)
def test_graphql_requires_and_caches_verified_public_repository_scope(token, scopes):
    session = _Session(token, scopes)
    client = GitHubGraphQLClient(token=token, session=session)

    client.execute("query First { viewer { login } }")
    client.execute("query Second { viewer { login } }")

    assert len(session.get_calls) == 1
    rate_url, options, authorization = session.get_calls[0]
    assert urlsplit(rate_url).path == "/rate_limit"
    assert options["allow_redirects"] is False
    assert 0 < options["timeout"] <= client.timeout_seconds
    assert authorization == f"Bearer {token}"
    assert session.scope_response.closed
    assert len(session.post_calls) == 2


@pytest.mark.parametrize(
    ("token", "scopes", "scope_checks"),
    [
        ("ghp_privileged", "repo, public_repo", 1),
        ("ghp_private_status", "repo:status", 1),
        ("ghp_unknown_scope", "public_repo, read:org", 1),
        ("ghp_missing_scope", None, 1),
        ("github_pat_fine_grained", "public_repo", 0),
        ("ghs_app_installation", "public_repo", 0),
        ("opaque_ambiguous_empty", "", 0),
        ("opaque_unverified_public_scope", "public_repo", 0),
    ],
)
def test_unverified_or_private_capable_credentials_never_reach_repository_graphql(token, scopes, scope_checks):
    session = _Session(token, scopes)
    client = GitHubGraphQLClient(token=token, session=session)

    with pytest.raises(GitHubClientError):
        client.execute(
            "query PrivateLookup($owner: String!, $name: String!) "
            "{ repository(owner: $owner, name: $name) { name description } }",
            {"owner": "owner", "name": "repo"},
        )

    assert len(session.get_calls) == scope_checks
    assert session.post_calls == []


def test_authentication_change_invalidates_the_public_scope_check():
    session = _Session("ghp_public_only", "public_repo")
    client = GitHubGraphQLClient(token="ghp_public_only", session=session)
    client.execute("query First { viewer { login } }")

    session.headers["Authorization"] = "Bearer ghp_broad_after_rotation"
    session.scopes = "repo"

    with pytest.raises(GitHubClientError):
        client.execute("query PrivateLookup { viewer { login } }")

    assert len(session.get_calls) == 2
    assert len(session.post_calls) == 1


def test_opaque_session_auth_fails_before_repository_graphql():
    session = _Session("ghp_public_only", "public_repo")
    session.auth = object()
    client = GitHubGraphQLClient(token="ghp_public_only", session=session)

    with pytest.raises(GitHubClientError):
        client.execute("query PrivateLookup { viewer { login } }")

    assert session.get_calls == []
    assert session.post_calls == []


def test_repository_discovery_qualifies_every_search_as_public():
    engine = GitHubDiscoveryEngine(object(), config=DiscoveryConfig(random_seed=7))

    queries = [
        engine._query_for_band("frameworks", ["python"], band)
        for band in ("high_star", "recently_active", "mid_sized", "emerging")
    ]

    assert all("is:public" in query for query in queries)
    assert all("is:private" not in query for query in queries)
