"""GitHub GraphQL API client for repository discovery and enrichment."""

from __future__ import annotations

import os
import random
import hashlib
import json
import math
import threading
from contextlib import contextmanager
from concurrent.futures import CancelledError
from dataclasses import dataclass
from collections.abc import Callable, Iterator
import time
from datetime import datetime, timezone
from typing import Any
import logging
from pathlib import PurePosixPath
from urllib.parse import quote

import requests

from .github_rate_limit import (
    GitHubRequestBudget, GitHubRetryLater, github_coordination_key, shared_github_coordinator,
)

from .graphql_queries import (
    GET_README_QUERY,
    GET_README_REF_QUERY,
    GET_README_BLOB_QUERY,
    build_readme_manifest_query,
    GET_REPOSITORY_QUERY,
    README_CANDIDATES,
    build_batch_metadata_query,
)

logger = logging.getLogger(__name__)


class GitHubGraphQLClientError(RuntimeError):
    """Raised when the GitHub GraphQL API returns an unrecoverable error."""

    def __init__(self, message: str, *, category: str = "response", status_code: int | None = None) -> None:
        super().__init__(message)
        # Log only controlled diagnostic fields, never the exception message:
        # legacy messages may contain response bodies or transport URLs.
        self.category = category if category in {
            "response", "transport", "rate_limit", "server", "http", "invalid_json", "graphql",
        } else "response"
        self.status_code = status_code if type(status_code) is int and 100 <= status_code <= 599 else None

# Backwards-compatible alias so existing callers don't break
GitHubClientError = GitHubGraphQLClientError


def _failed_repository_aliases(
    errors: Any,
    aliases: set[str],
    *,
    ignored_child_fields: set[str] | None = None,
) -> set[str]:
    """Keep unaffected rows only when GraphQL identifies the failed field."""
    failed: set[str] = set()
    if not isinstance(errors, list):
        raise GitHubClientError("GraphQL errors have no usable field paths")
    for error in errors:
        path = error.get("path") if isinstance(error, dict) else None
        if not isinstance(path, list) or not path or not isinstance(path[0], str):
            raise GitHubClientError("GraphQL errors have no usable field paths")
        if path[0] == "rateLimit":
            continue
        if path[0] not in aliases:
            raise GitHubClientError("GraphQL error identifies an unknown repository field")
        if (
            ignored_child_fields
            and len(path) > 1
            and isinstance(path[1], str)
            and path[1] in ignored_child_fields
        ):
            continue
        failed.add(path[0])
    return failed


class FetchedReadme(str):
    """Canonical README Markdown with the metadata needed to resolve media.

    This remains a ``str`` subclass so existing acquisition callers that only
    consume README text continue to work while V2 ingestion can retain the
    source metadata as a separate artifact.
    """

    raw_markdown: str
    source_path: str | None
    default_branch: str | None
    base_url: str | None
    source_commit: str | None
    blob_oid: str | None
    authorization_scope: str | None
    source_repository: str | None

    def __new__(
        cls,
        raw_markdown: str = "",
        *,
        source_path: str | None = None,
        default_branch: str | None = None,
        base_url: str | None = None,
        source_commit: str | None = None,
        blob_oid: str | None = None,
        authorization_scope: str | None = None,
        source_repository: str | None = None,
    ) -> "FetchedReadme":
        value = str(raw_markdown or "")
        instance = super().__new__(cls, value)
        instance.raw_markdown = value
        instance.source_path = source_path
        instance.default_branch = default_branch
        instance.base_url = base_url
        instance.source_commit = source_commit
        instance.blob_oid = blob_oid
        instance.authorization_scope = authorization_scope
        instance.source_repository = source_repository
        return instance


def build_readme_base_url(
    owner: str,
    name: str,
    default_branch: str | None,
    source_path: str | None,
    *,
    source_commit: str | None = None,
) -> str | None:
    """Build an HTTPS raw-content directory URL for relative README media."""
    if not source_path:
        return None
    ref = (source_commit or default_branch or "HEAD").strip()
    if not ref:
        ref = "HEAD"
    owner_part = quote(owner.strip(), safe="")
    name_part = quote(name.strip(), safe="")
    ref_part = "/".join(quote(part, safe="") for part in ref.split("/"))
    parent = PurePosixPath(source_path).parent
    parent_part = "" if str(parent) == "." else "/".join(
        quote(part, safe="") for part in parent.parts if part not in {"", "."}
    )
    suffix = f"{parent_part}/" if parent_part else ""
    if default_branch and not source_commit:
        return (
            f"https://raw.githubusercontent.com/{owner_part}/{name_part}/"
            f"refs/heads/{ref_part}/{suffix}"
        )
    return (
        f"https://raw.githubusercontent.com/{owner_part}/{name_part}/"
        f"{ref_part}/{suffix}"
    )


@dataclass(frozen=True)
class _ReadmeManifest:
    source_path: str | None
    blob_oid: str | None
    byte_size: int = 0
    cacheable: bool = True


def _git_oid(value: Any) -> str | None:
    if isinstance(value, str) and len(value) in {40, 64} and all(c in "0123456789abcdef" for c in value):
        return value
    return None


class GitHubGraphQLClient:
    def __init__(
        self,
        *,
        token: str | None = None,
        base_url: str = "https://api.github.com/graphql",
        timeout_seconds: float = 30.0,
        max_retries: int = 4,
        sleep_on_rate_limit: bool = True,
        session: requests.Session | None = None,
        deadline: float | None = None,
        cancel_event: threading.Event | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        if type(max_retries) is not int or not 0 <= max_retries <= 10:
            raise ValueError("GitHub max_retries must be between 0 and 10")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("GitHub timeout_seconds must be positive")
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("GitHub deadline must be a finite monotonic time")
        self.url = base_url
        self.deadline = deadline
        self.cancel_event = cancel_event
        self.cancel_check = cancel_check
        self._retry_local = threading.local()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.sleep_on_rate_limit = sleep_on_rate_limit
        self.session = session or requests.Session()
        token = token if token is not None else os.getenv("GITHUB_TOKEN")
        self.session.headers.update(
            {
                "User-Agent": "osiris-repository-ingestion-pipeline",
                "Content-Type": "application/json",
            }
        )
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.coordination_key = self._authorization_key()
        self.rate_limits = shared_github_coordinator(self.coordination_key)

    def _authorization_key(self) -> str:
        credential = self.session.headers.get("Authorization")
        # Custom session authentication is opaque. Object replacement changes
        # the coordination scope; in-place mutation cannot be fingerprinted
        # safely, so custom-auth content/query replay is disabled below.
        auth = self.session.auth
        if auth is not None:
            credential = f"{credential or ''}\0session-auth:{id(self.session)}:{id(auth)}"
        return github_coordination_key(self.url, credential)

    def get_repository(self, owner: str, name: str) -> dict[str, Any] | None:
        """Fetches a single repository using GraphQL."""
        variables = {"owner": owner, "name": name}
        response = self.execute(GET_REPOSITORY_QUERY, variables)
        if not response:
            return None
        # This legacy single-repository query also selects README aliases.
        # Their failures cannot invalidate resolved metadata, but metadata or
        # unlocated errors must still retry rather than erase saved fields.
        if _failed_repository_aliases(
            response.get("errors") or [],
            {"repository"},
            ignored_child_fields={alias for alias, _path in README_CANDIDATES},
        ):
            raise GitHubClientError("repository metadata fields failed; retry acquisition")
        data = response.get("data", {})
        repo = data.get("repository") if isinstance(data, dict) else None
        if repo is not None and (
            not isinstance(repo, dict) or "description" not in repo
        ):
            raise GitHubClientError("repository description was not resolved")
        if _failed_repository_aliases(response.get("errors") or [], {"repository"}) and not (
            isinstance(repo, dict)
            and any(
                isinstance(repo.get(alias), dict)
                and isinstance(repo[alias].get("text"), str)
                and repo[alias]["text"]
                for alias, _path in README_CANDIDATES
            )
        ):
            # Direct enrichment consumes these inline aliases. An unresolved
            # README cannot become an empty canonical value even when metadata
            # succeeded; another usable alias is required to continue safely.
            raise GitHubClientError("README candidates failed; retry acquisition")
        return repo

    def get_readme(self, owner: str, name: str) -> FetchedReadme:
        """Revalidate HEAD, then reuse only immutable, authorization-scoped blobs."""
        existing = getattr(self._retry_local, "budget", None)
        budget = existing or GitHubRequestBudget(
            deadline=self.deadline, defer_retries=False,
            cancelled=lambda: bool(
                (self.cancel_event is not None and self.cancel_event.is_set())
                or (self.cancel_check is not None and self.cancel_check())
            ),
        )
        current_scope = self._authorization_key()
        if budget.authorization_scope is not None and budget.authorization_scope != current_scope:
            raise GitHubClientError("GitHub authorization changed during acquisition", category="http")
        budget.authorization_scope = current_scope
        previous_defer_retries = budget.defer_retries
        if self.session.auth is not None:
            # Opaque auth cannot safely replay successful earlier stages.
            # Keep this invocation (and its local ref/manifest state) alive
            # while waiting instead of restarting it through the scheduler.
            # Genuine HTTP attempts still consume the original job budget.
            budget.defer_retries = False
        try:
            with self.retry_scope(budget):
                result = self._get_readme_cached(owner, name, budget)
                budget.check()
                if budget.authorization_scope != self._authorization_key():
                    raise GitHubClientError("GitHub authorization changed during README acquisition", category="http")
                return result
        finally:
            budget.defer_retries = previous_defer_retries
            if existing is None:
                budget.finish()

    def _get_readme_cached(self, owner: str, name: str, budget: GitHubRequestBudget) -> FetchedReadme:
        from .readme_cache import README_BLOB_CACHE, README_MANIFEST_CACHE

        response = self.execute(GET_README_REF_QUERY, {"owner": owner, "name": name})
        data = response.get("data") if isinstance(response, dict) else None
        repo = data.get("repository") if isinstance(data, dict) else None
        if not isinstance(repo, dict):
            if isinstance(response, dict) and response.get("errors"):
                raise GitHubClientError("README query returned GraphQL errors")
            raise GitHubClientError("README reference query could not resolve repository data")
        # Legacy/partial-query clients can still return usable canonical text.
        # Without a verified immutable identity it is never cached.
        if any(alias in repo for alias, _path in README_CANDIDATES):
            return self._get_readme_legacy(owner, name, response=response)
        branch = repo.get("defaultBranchRef")
        target = branch.get("target") if isinstance(branch, dict) else None
        commit_oid = _git_oid(target.get("oid")) if isinstance(target, dict) else None
        branch_name = branch.get("name") if isinstance(branch, dict) else None
        branch_name = branch_name if isinstance(branch_name, str) else None
        if not commit_oid:
            # Null/failed ref resolution alone is not proof of missing README.
            return self._get_readme_legacy(owner, name)
        repository = (repo.get("id"), f"{owner}/{name}".casefold())
        scope = self.coordination_key
        # Arbitrary session.auth implementations may mutate credentials in
        # place. Their effective access identity cannot be proven from object
        # identity; do not share/cache content or in-flight work for them.
        allow_shared_cache = self.session.auth is None
        manifest_key = (scope, repository, commit_oid)
        if allow_shared_cache:
            manifest = README_MANIFEST_CACHE.get_or_load(
                manifest_key,
                lambda: self._readme_manifest(owner, name, commit_oid),
                budget, size=lambda value: 128 + len(value.source_path or "") + len(value.blob_oid or ""),
                cacheable=lambda value: value.cacheable,
            )
        else:
            manifest = self._readme_manifest(owner, name, commit_oid)
        if manifest.blob_oid is None:
            return FetchedReadme(
                default_branch=branch_name, source_commit=commit_oid,
                authorization_scope=scope if allow_shared_cache else None,
                source_repository=f"{owner}/{name}".casefold(),
            )
        blob_key = (scope, repository, manifest.source_path, manifest.blob_oid)
        if allow_shared_cache:
            markdown = README_BLOB_CACHE.get_or_load(
                blob_key,
                lambda: self._readme_blob(owner, name, manifest.blob_oid, manifest.byte_size),
                budget, size=lambda value: len(value.encode("utf-8")),
            )
        else:
            markdown = self._readme_blob(owner, name, manifest.blob_oid, manifest.byte_size)
        return FetchedReadme(
            markdown, source_path=manifest.source_path, default_branch=branch_name,
            source_commit=commit_oid, blob_oid=manifest.blob_oid,
            authorization_scope=scope if allow_shared_cache else None,
            source_repository=f"{owner}/{name}".casefold(),
            base_url=build_readme_base_url(
                owner, name, branch_name, manifest.source_path, source_commit=commit_oid,
            ),
        )

    def _readme_manifest(self, owner: str, name: str, commit_oid: str) -> _ReadmeManifest:
        response = self.execute(build_readme_manifest_query(commit_oid), {"owner": owner, "name": name})
        data = response.get("data") if isinstance(response, dict) else None
        repo = data.get("repository") if isinstance(data, dict) else None
        if not isinstance(repo, dict):
            raise GitHubClientError("README manifest query could not resolve repository data")
        complete = not response.get("errors")
        for alias, path in README_CANDIDATES:
            if alias not in repo:
                complete = False
                continue
            blob = repo[alias]
            if blob is None:
                continue
            if not isinstance(blob, dict):
                complete = False
                continue
            oid, byte_size = _git_oid(blob.get("oid")), blob.get("byteSize")
            if oid is None or type(byte_size) is not int or byte_size < 0 or blob.get("isBinary") is not False:
                complete = False
                continue
            if byte_size:
                return _ReadmeManifest(path, oid, byte_size, complete)
        if not complete:
            raise GitHubClientError("README candidates were not fully resolved; retry acquisition")
        return _ReadmeManifest(None, None)

    def _readme_blob(self, owner: str, name: str, oid: str, byte_size: int) -> str:
        response = self.execute(GET_README_BLOB_QUERY, {"owner": owner, "name": name, "oid": oid})
        data = response.get("data") if isinstance(response, dict) else None
        repo = data.get("repository") if isinstance(data, dict) else None
        blob = repo.get("object") if isinstance(repo, dict) else None
        if not isinstance(blob, dict) or blob.get("oid") != oid or blob.get("isBinary") is not False:
            raise GitHubClientError("README blob identity was not resolved")
        markdown = blob.get("text")
        if not isinstance(markdown, str) or blob.get("byteSize") != byte_size:
            raise GitHubClientError("README blob content was not fully resolved")
        raw = markdown.encode("utf-8")
        digest = hashlib.sha1 if len(oid) == 40 else hashlib.sha256
        if len(raw) != byte_size or digest(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != oid:
            # Never cache a truncated/rewritten text view as a canonical blob.
            raise GitHubClientError("README canonical blob content did not match its immutable identity")
        return markdown

    def _get_readme_legacy(
        self, owner: str, name: str, *, response: dict[str, Any] | None = None,
    ) -> FetchedReadme:
        """Fetch canonical README Markdown and its source metadata.

        ``FetchedReadme`` is string-compatible for older text-only callers.
        """
        try:
            if response is None:
                response = self.execute(GET_README_QUERY, {"owner": owner, "name": name})
            if not isinstance(response, dict):
                raise GitHubClientError("README query returned no response envelope")
            gql_errors = response.get("errors")
            data = response.get("data")
            # GitHub can return a partial response: usable repository data (and
            # valid README aliases) alongside a field-level error for a different
            # alias. Keep usable text, but never confirm absence from failed or
            # incomplete aliases: that would erase previously saved content.
            if gql_errors:
                if not isinstance(data, dict) or "repository" not in data:
                    raise GitHubClientError(
                        "README query returned GraphQL errors"
                    )
                logger.warning("README query returned partial field-level errors for %s/%s", owner, name)
            if not isinstance(data, dict) or "repository" not in data:
                raise GitHubClientError("README query response is missing repository data")
            repo = data["repository"]
            if not isinstance(repo, dict):
                raise GitHubClientError("README query could not resolve the repository")
            # A field-level GraphQL error may omit defaultBranchRef while a
            # README alias still resolves successfully.  The alias is usable;
            # build_readme_base_url deliberately falls back to GitHub's HEAD
            # reference when branch metadata is unavailable.
            branch_ref = repo.get("defaultBranchRef")
            default_branch = (
                branch_ref.get("name") if isinstance(branch_ref, dict) else None
            )
            complete = not gql_errors
            for key, source_path in README_CANDIDATES:
                if key not in repo:
                    complete = False
                blob = repo.get(key)
                if blob is None:
                    continue
                if not isinstance(blob, dict) or "text" not in blob:
                    complete = False
                    continue
                text = blob["text"]
                if not isinstance(text, str):
                    complete = False
                    continue
                if text:
                    return FetchedReadme(
                        text,
                        source_path=source_path,
                        default_branch=default_branch,
                        base_url=build_readme_base_url(
                            owner, name, default_branch, source_path
                        ),
                    )
            if not complete:
                raise GitHubClientError("README candidates were not fully resolved; retry acquisition")
            logger.info(
                "README not found for %s/%s: every candidate resolved empty or null.",
                owner,
                name,
            )
        except (GitHubRetryLater, CancelledError, TimeoutError):
            raise
        except Exception as exc:
            logger.warning("README fetch failed for %s/%s (type=%s)", owner, name, type(exc).__name__)
            # Missing READMEs are represented by an empty FetchedReadme above.
            # Transport/API failures must propagate so acquisition marks the
            # repository for retry instead of overwriting canonical content
            # with an accidental empty README.
            raise
        return FetchedReadme()

    def get_repositories_batch(self, repos: list[tuple[str, str]], *, projection: str = "legacy") -> dict[str, dict[str, Any]]:
        """Fetches multiple repositories using a lean metadata-only batch query."""
        if not repos:
            return {}

        query = build_batch_metadata_query(repos, projection=projection)
        response = self.execute(query)
        if not response:
            return {}
        if response.get("errors") and not response.get("data"):
            raise GitHubClientError(
                "batch metadata query returned GraphQL errors"
            )

        data = response.get("data", {})
        if not isinstance(data, dict):
            raise GitHubClientError("batch metadata query returned invalid data")
        failed = _failed_repository_aliases(
            response.get("errors") or [],
            {f"repo_{i}" for i in range(len(repos))},
        )
        results = {}
        for i, (owner, name) in enumerate(repos):
            alias = f"repo_{i}"
            repo = data.get(alias)
            if alias in failed or (isinstance(repo, dict) and "description" not in repo):
                # Omitted rows become structured acquisition failures upstream;
                # healthy rows can still be published without replacing text
                # from an unresolved field with an accidental empty value.
                logger.warning("Metadata fields unresolved for %s/%s; retry acquisition", owner, name)
                continue
            if isinstance(repo, dict) and repo:
                results[f"{owner}/{name}"] = repo
        return results

    @contextmanager
    def retry_scope(self, budget: GitHubRequestBudget) -> Iterator[None]:
        """Let the scheduler own waiting and preserve a logical job's budget."""
        previous = getattr(self._retry_local, "budget", None)
        self._retry_local.budget = budget
        try:
            yield
        finally:
            self._retry_local.budget = previous

    def _wait_or_defer(self, ready_at: float, budget: GitHubRequestBudget, deferred: bool) -> None:
        budget.check_wait(ready_at)
        if deferred:
            raise GitHubRetryLater(ready_at)
        self.rate_limits.count("waiting")
        while time.monotonic() < ready_at:
            budget.check()
            delay = min(0.05, ready_at - time.monotonic())
            if delay > 0:
                if self.cancel_event is not None:
                    self.cancel_event.wait(delay)
                else:
                    time.sleep(delay)
        budget.check()

    def execute(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """One bounded retry policy, shared quota admission and interruptible waits.

        README scheduler scopes defer instead of sleeping in transport workers.
        Standalone/metadata callers use the same admission and attempt policy,
        but wait synchronously with cancellation/deadline checks every 50ms.
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        scheduled = getattr(self._retry_local, "budget", None)
        budget = scheduled or GitHubRequestBudget(
            deadline=self.deadline,
            cancelled=lambda: bool(
                (self.cancel_event is not None and self.cancel_event.is_set())
                or (self.cancel_check is not None and self.cancel_check())
            ),
        )
        current_scope = self._authorization_key()
        if budget.authorization_scope is not None and budget.authorization_scope != current_scope:
            raise GitHubClientError("GitHub authorization changed during acquisition", category="http")
        budget.authorization_scope = current_scope
        if self.coordination_key != current_scope:
            self.coordination_key = current_scope
            self.rate_limits = shared_github_coordinator(current_scope)
        deferred = scheduled is not None and budget.defer_retries
        budget.check()
        # Opaque auth can change in place without changing its object ID. Do
        # not replay earlier successful access/content under that new state.
        allow_response_replay = self.session.auth is None
        if allow_response_replay and identity in budget.responses:
            return budget.responses[identity]
        while True:
            budget.check()
            ready_at = budget.not_before.get(identity, 0.0)
            if ready_at > time.monotonic():
                self._wait_or_defer(ready_at, budget, deferred)
            ready_at = self.rate_limits.admit()
            if ready_at is not None:
                if not self.sleep_on_rate_limit:
                    raise GitHubClientError("GitHub GraphQL quota admission deferred", category="rate_limit")
                self._wait_or_defer(ready_at, budget, deferred)
                continue
            attempt = budget.attempts.get(identity, 0)
            if attempt > self.max_retries:
                raise GitHubClientError("GitHub GraphQL request exhausted retries")
            budget.attempts[identity] = attempt + 1
            timeout = self.timeout_seconds
            if budget.deadline is not None:
                timeout = min(timeout, max(0.001, budget.deadline - time.monotonic()))
            try:
                response = self.session.post(self.url, json=payload, timeout=timeout)
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise GitHubClientError("GitHub GraphQL transport retries exhausted", category="transport") from exc
                self._retry(identity, attempt, budget, deferred)
                continue

            self.rate_limits.observe(response.headers)
            # IO has settled, so its worker/admission can now be released. A
            # requests timeout is not a total deadline: reject late success
            # before parsing, caching or returning canonical data.
            budget.check()
            if response.status_code == 429 or (response.status_code == 403 and self._is_rate_limited(response)):
                ready_at = self.rate_limits.throttle(response.headers)
                if attempt >= self.max_retries or not self.sleep_on_rate_limit:
                    raise GitHubClientError("GitHub GraphQL rate limit exceeded", category="rate_limit", status_code=response.status_code)
                self._retry(identity, attempt, budget, deferred, ready_at=ready_at)
                continue
            if response.status_code in {500, 502, 503, 504}:
                if attempt >= self.max_retries:
                    raise GitHubClientError("GitHub GraphQL server retries exhausted", category="server", status_code=response.status_code)
                self._retry(identity, attempt, budget, deferred)
                continue
            if response.status_code >= 400:
                raise GitHubClientError("GitHub GraphQL HTTP failure", category="http", status_code=response.status_code)
            try:
                result = response.json()
            except ValueError as exc:
                raise GitHubClientError("GitHub GraphQL returned invalid JSON", category="invalid_json", status_code=response.status_code) from exc
            if not isinstance(result, dict):
                raise GitHubClientError("GitHub GraphQL returned an invalid response envelope", status_code=response.status_code)
            errors = result.get("errors")
            if "errors" in result and (
                not isinstance(errors, list) or any(not isinstance(error, dict) for error in errors)
            ):
                raise GitHubClientError("GitHub GraphQL returned invalid error details", status_code=response.status_code)
            data_field = result.get("data")
            if (data_field is not None and not isinstance(data_field, dict)) or (
                not isinstance(data_field, dict) and not errors
            ):
                raise GitHubClientError("GitHub GraphQL returned an invalid data envelope", status_code=response.status_code)
            rl = data_field.get("rateLimit") if isinstance(data_field, dict) else None
            self.rate_limits.observe(response.headers, rl)
            rate_limited = any(
                error.get("type") == "RATE_LIMITED"
                or (isinstance(error.get("extensions"), dict) and error["extensions"].get("code") == "RATE_LIMITED")
                for error in errors or []
            )
            if rate_limited:
                ready_at = self.rate_limits.throttle(response.headers)
                if not data_field:
                    if attempt >= self.max_retries or not self.sleep_on_rate_limit:
                        raise GitHubClientError("GitHub GraphQL rate limit exceeded", category="rate_limit", status_code=response.status_code)
                    self._retry(identity, attempt, budget, deferred, ready_at=ready_at)
                    continue
            if errors:
                if data_field:
                    logger.warning("GitHub GraphQL returned partial errors (count=%d)", len(errors))
                elif any(
                    isinstance(error.get("message"), str)
                    and "Could not resolve to a Repository" in error["message"]
                    for error in errors
                ):
                    if allow_response_replay:
                        budget.responses[identity] = None
                    return None
                else:
                    raise GitHubClientError("GitHub GraphQL returned errors", category="graphql", status_code=response.status_code)
            budget.check()
            if allow_response_replay:
                budget.responses[identity] = result
            return result

    def _retry(
        self, identity: str, attempt: int, budget: GitHubRequestBudget,
        deferred: bool, *, ready_at: float | None = None,
    ) -> None:
        self.rate_limits.count("retries")
        if ready_at is None:
            ready_at = time.monotonic() + min((2 ** attempt) + random.random(), 30.0)
        budget.not_before[identity] = ready_at
        self._wait_or_defer(ready_at, budget, deferred)

    @staticmethod
    def _is_rate_limited(response: requests.Response) -> bool:
        remaining = response.headers.get("X-RateLimit-Remaining")
        return (
            remaining == "0"
            or "retry-after" in {key.lower() for key in response.headers}
            or "rate limit exceeded" in response.text.lower()
            or "secondary rate limit" in response.text.lower()
        )
