"""Public-only authorization boundaries for offline GitHub acquisition."""

from __future__ import annotations

from urllib.parse import urlsplit
from types import SimpleNamespace
import time

import pytest
import requests

import main
from acquisition.github_discovery import DiscoveryConfig, GitHubDiscoveryEngine
from acquisition.github_rate_limit import GitHubRequestBudget, GitHubRetryLater
from acquisition.github_graphql_client import FetchedReadme, GitHubClientError, GitHubGraphQLClient
from acquisition.models import github_failure_disposition


class _Response:
    def __init__(self, status_code: int = 200, *, scopes: str | None = None, headers=None) -> None:
        self.status_code = status_code
        self.headers = dict(headers or {})
        if scopes is not None:
            self.headers["X-OAuth-Scopes"] = scopes
        self.text = ""
        self.closed = False

    def json(self):
        return {"data": {"viewer": {"login": "public-only"}}}

    def close(self):
        self.closed = True


class _Session:
    def __init__(self, token: str, scopes: str | None, scope_responses=None) -> None:
        self.headers = {"Authorization": f"Bearer {token}"}
        self.auth = None
        self.scopes = scopes
        self.scope_responses = list(scope_responses or [])
        self.scope_responses_seen = []
        self.get_calls = []
        self.post_calls = []
        self.scope_response = None

    def get(self, url: str, **kwargs):
        self.get_calls.append((url, kwargs, self.headers.get("Authorization")))
        if self.scope_responses:
            response = self.scope_responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            self.scope_response = response
        else:
            self.scope_response = _Response(scopes=self.scopes)
        self.scope_responses_seen.append(self.scope_response)
        return self.scope_response

    def post(self, url: str, **kwargs):
        self.post_calls.append((url, kwargs, self.headers.get("Authorization")))
        return _Response()


@pytest.mark.parametrize(
    ("token", "scopes"),
    [
        ("ghp_public_only", "public_repo"),
        ("gho_public_only", "public_repo, read:user, user:email"),
        ("ghp_user_profile_only", "read:user"),
        ("gho_email_only", "user:email"),
        ("ghp_no_scope", ""),
    ],
)
def test_scope_verification_is_cached_only_inside_one_logical_job(token, scopes):
    session = _Session(token, scopes)
    client = GitHubGraphQLClient(token=token, session=session)
    first_job = GitHubRequestBudget()

    with client.retry_scope(first_job):
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
    first_job.finish()

    # The same token can gain permissions while the client remains alive. A
    # separate lookup job must fetch its current scopes before repository I/O.
    session.scopes = "repo, public_repo"
    next_job = GitHubRequestBudget()
    with client.retry_scope(next_job), pytest.raises(GitHubClientError):
        client.execute("query Later { viewer { login } }")
    next_job.finish()

    assert len(session.get_calls) == 2
    assert len(session.post_calls) == 2


def test_direct_repository_lookup_refreshes_scope_proof_for_each_lookup():
    session = _Session("ghp_direct_lookup", "public_repo")
    client = GitHubGraphQLClient(token="ghp_direct_lookup", session=session)

    assert client.get_repository("owner", "repo") is None
    session.scopes = "repo"

    with pytest.raises(GitHubClientError):
        client.get_repository("owner", "private-repo")

    assert len(session.get_calls) == 2
    assert len(session.post_calls) == 1


def test_readme_job_verifies_current_scopes_before_a_cached_readme_can_be_returned(monkeypatch):
    session = _Session("ghp_warm_readme", "public_repo")
    client = GitHubGraphQLClient(token="ghp_warm_readme", session=session)
    monkeypatch.setattr(
        client,
        "_get_readme_cached",
        lambda _owner, _name, _budget: FetchedReadme("# cached public README"),
    )

    assert str(client.get_readme("owner", "repo")) == "# cached public README"
    session.scopes = "repo, public_repo"

    with pytest.raises(GitHubClientError):
        client.get_readme("owner", "repo")

    assert len(session.get_calls) == 2
    assert session.post_calls == []


def test_rotated_readme_token_does_not_inherit_previous_token_cooldown(monkeypatch):
    session = _Session("ghp_old_readme_quota", "public_repo")
    client = GitHubGraphQLClient(
        token="ghp_old_readme_quota", session=session, sleep_on_rate_limit=False,
    )
    old_coordinator = client.rate_limits
    old_coordinator.throttle({"Retry-After": "3600"})
    session.headers["Authorization"] = "Bearer ghp_new_readme_quota"
    original_post = session.post

    def public_readme(*args, **kwargs):
        response = original_post(*args, **kwargs)
        response.json = lambda: {"data": {"repository": {"readme1": {"text": "# Public README"}}}}
        return response

    monkeypatch.setattr(session, "post", public_readme)

    assert str(client.get_readme("owner", "repo")) == "# Public README"
    assert client.rate_limits is not old_coordinator
    assert len(session.get_calls) == len(session.post_calls) == 1
    assert session.get_calls[0][2] == "Bearer ghp_new_readme_quota"
    assert session.post_calls[0][2] == "Bearer ghp_new_readme_quota"
    assert old_coordinator.admit() is not None


def test_rest_probe_reset_does_not_extend_shared_secondary_cooldown(monkeypatch):
    from acquisition import github_rate_limit as quota

    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(quota, "time", SimpleNamespace(
        monotonic=lambda: clock.now, time=lambda: clock.now + 1000,
    ))
    coordinator = quota.GitHubRateLimitCoordinator()
    coordinator.observe({"X-RateLimit-Remaining": "12", "X-RateLimit-Reset": "5000"})

    retry_at = coordinator.scope_probe_retry_at({
        "Retry-After": "60", "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "4700",
    }, share_secondary=True)

    assert 3700.0 <= retry_at <= 3700.3, "the probe job must honor its own REST reset"
    assert coordinator.admit() is not None, "GraphQL must honor the secondary delay"
    clock.now = 161.0
    assert coordinator.admit() is None, "verified GraphQL work can resume before the REST reset"
    assert coordinator.snapshot()["remaining"] == 11


def test_scope_probe_never_replaces_the_graphql_primary_quota():
    session = _Session(
        "ghp_quota_isolation",
        "public_repo",
        scope_responses=[_Response(
            scopes="public_repo",
            headers={
                "X-RateLimit-Remaining": "5000",
                "X-RateLimit-Reset": str(time.time() + 3600),
            },
        )],
    )
    client = GitHubGraphQLClient(token="ghp_quota_isolation", session=session)
    client.rate_limits.observe({
        "X-RateLimit-Remaining": "12",
        "X-RateLimit-Reset": str(time.time() + 60),
    })

    client.execute("query QuotaIsolation { viewer { login } }")

    assert client.rate_limits.snapshot()["remaining"] == 11


@pytest.mark.parametrize(
    "temporary_failure",
    [
        _Response(502),
        _Response(429, headers={"Retry-After": "0"}),
        requests.ConnectionError("temporary fixture failure"),
    ],
    ids=["server-error", "rate-limit", "transport-error"],
)
def test_temporary_scope_probe_failures_retry_before_repository_graphql(monkeypatch, temporary_failure):
    retry_response = _Response(scopes="public_repo")
    session = _Session(
        "ghp_probe_retry",
        "public_repo",
        scope_responses=[temporary_failure, retry_response],
    )
    client = GitHubGraphQLClient(token="ghp_probe_retry", session=session, max_retries=1)

    if isinstance(temporary_failure, _Response) and temporary_failure.status_code == 429:
        original_retry_at = client.rate_limits.scope_probe_retry_at

        def immediate_probe_retry(headers, *, share_secondary):
            original_retry_at(headers, share_secondary=share_secondary)
            with client.rate_limits._lock:
                client.rate_limits._blocked_until = 0.0
                client.rate_limits._next_admission = 0.0
                client.rate_limits._stagger_until = 0.0
            return 0.0

        monkeypatch.setattr(client.rate_limits, "scope_probe_retry_at", immediate_probe_retry)
    else:
        monkeypatch.setattr(client, "_wait_or_defer", lambda *_args, **_kwargs: None)

    client.execute("query AfterScopeCheck { viewer { login } }")

    assert len(session.get_calls) == 2
    assert retry_response.closed
    assert len(session.post_calls) == 1


def test_scope_probe_503_retry_after_is_deferred_resumed_and_deadline_bounded():
    server_error = _Response(503, headers={"Retry-After": "0.04"})
    success = _Response(scopes="public_repo")
    session = _Session(
        "ghp_probe_retry_after",
        "public_repo",
        scope_responses=[server_error, success],
    )
    client = GitHubGraphQLClient(token="ghp_probe_retry_after", session=session, max_retries=1)
    budget = GitHubRequestBudget(deadline=time.monotonic() + 1)
    identity = f"public-scope:{client._authorization_key()}"
    started = time.monotonic()

    with client.retry_scope(budget), pytest.raises(GitHubRetryLater) as deferred:
        client.execute("query RetryAfter { viewer { login } }")

    ready_at = deferred.value.not_before
    assert started + 0.035 <= ready_at <= started + 0.3
    assert budget.not_before[identity] == ready_at
    assert server_error.closed
    assert session.post_calls == []

    time.sleep(max(0.0, ready_at - time.monotonic() + 0.01))
    with client.retry_scope(budget):
        client.execute("query RetryAfter { viewer { login } }")

    assert len(session.get_calls) == 2
    assert success.closed
    assert len(session.post_calls) == 1
    budget.finish()

    # A server-directed delay beyond this new job's deadline never retries or
    # reaches repository GraphQL.
    timed_response = _Response(503, headers={"Retry-After": "1"})
    timed_session = _Session(
        "ghp_probe_retry_after_deadline",
        "public_repo",
        scope_responses=[timed_response],
    )
    timed_client = GitHubGraphQLClient(
        token="ghp_probe_retry_after_deadline", session=timed_session, max_retries=1,
    )
    timed_budget = GitHubRequestBudget(deadline=time.monotonic() + 0.05)
    with timed_client.retry_scope(timed_budget), pytest.raises(TimeoutError):
        timed_client.execute("query RetryAfterDeadline { viewer { login } }")

    assert timed_response.closed
    assert len(timed_session.get_calls) == 1
    assert timed_session.post_calls == []
    timed_budget.finish()


@pytest.mark.parametrize(
    ("category", "status_code", "expected"),
    [
        ("authorization", None, "configuration"),
        ("authorization", 401, "configuration"),
        ("transport", None, "retryable"),
        ("server", 502, "retryable"),
        ("rate_limit", 429, "retryable"),
    ],
)
def test_public_scope_failures_pause_only_for_deterministic_authorization_errors(category, status_code, expected):
    error = GitHubClientError("controlled fixture error", category=category, status_code=status_code)

    assert github_failure_disposition(error) == expected


def test_expired_github_token_is_classified_as_global_configuration_failure():
    session = _Session(
        "ghp_expired",
        None,
        scope_responses=[_Response(401)],
    )
    client = GitHubGraphQLClient(token="ghp_expired", session=session)

    with pytest.raises(GitHubClientError) as captured:
        client.execute("query Expired { viewer { login } }")

    assert captured.value.category == "authorization"
    assert github_failure_disposition(captured.value) == "configuration"
    assert session.post_calls == []


def test_network_free_cli_rejects_fine_grained_token_without_echoing_it(monkeypatch):
    secret = "github_pat_never-log-this-fixture"
    monkeypatch.setenv("GITHUB_TOKEN", secret)
    monkeypatch.setenv("BACKEND_URL", "https://backend.invalid")
    monkeypatch.setenv("INTERNAL_API_SECRET", "fixture-secret")

    errors = main._configuration_errors(SimpleNamespace())

    message = "\n".join(errors)
    assert "classic or OAuth" in message
    assert "public_repo" in message
    assert secret not in message


def test_acquisition_cli_help_explains_token_scope_requirements(capsys):
    with pytest.raises(SystemExit) as captured:
        main._parse_args(["--help"])

    output = capsys.readouterr().out
    assert captured.value.code == 0
    assert "ghp_" in output
    assert "gho_" in output
    assert "public_repo" in output
    assert "checked at runtime" in output


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
