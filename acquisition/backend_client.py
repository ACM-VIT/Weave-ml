"""Backend v2 delivery adapter for trusted repository ingestion workers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
import json
import logging
import os
from datetime import datetime, timezone
import time
from typing import Any
import uuid

import requests

from .manifest import ManifestUnavailable, load_manifest, timestamp


MAX_RECORDS = 100
# The ingestion contract permits a canonical README of up to one million
# characters. Keep transport bounded while allowing UTF-8 and JSON escaping
# overhead for a single maximum-size repository record.
MAX_BODY_BYTES = 8 * 1024 * 1024
_REPOSITORIES_PREFIX = b'{"repositories":['
_REPOSITORIES_SUFFIX = b"]}"
_REPOSITORIES_ENVELOPE_BYTES = len(_REPOSITORIES_PREFIX) + len(_REPOSITORIES_SUFFIX)


@dataclass(slots=True)
class RepositoryDeliveryResult:
    succeeded: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    mappings: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.succeeded)


@dataclass(slots=True)
class _EncodedRepositoryBatch:
    identities: list[tuple[str, str]]
    body: bytes


class BackendIngestionClient:
    """Publish source data without accessing backend PostgreSQL or Qdrant."""

    enabled = True

    def __init__(
        self,
        *,
        base_url: str,
        internal_secret: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        session: requests.Session | None = None,
        refresh_after_seconds: int | None = None,
        manifest_transport: Any | None = None,
    ) -> None:
        self.base_url = str(base_url).strip().rstrip("/")
        self.internal_secret = str(internal_secret).strip()
        if not self.base_url:
            raise ValueError("BACKEND_URL is required")
        if not self.internal_secret:
            raise ValueError("INTERNAL_API_SECRET is required")
        if timeout <= 0 or max_retries < 1:
            raise ValueError("timeout and max_retries must be positive")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self.manifest_transport = manifest_transport
        self._accepted_names: set[str] = set()
        self._accepted_count = 0
        self.refresh_after_seconds = (
            int(os.getenv("ACQUISITION_REFRESH_AFTER_SECONDS", "86400"))
            if refresh_after_seconds is None else refresh_after_seconds
        )
        if type(self.refresh_after_seconds) is not int or not 1 <= self.refresh_after_seconds <= 2_592_000:
            raise ValueError("refresh_after_seconds must be between 1 and 2592000")
        self._manifest: dict[str, dict[str, Any]] | None = None
        self.manifest_status = "not_requested"

    def verify_connection(self) -> bool:
        # Configuration validation is enough here; the first authenticated write
        # is the authoritative readiness check for this narrowly scoped worker.
        return True

    def init_db(self) -> None:
        """Compatibility no-op: schema ownership belongs to the backend."""

    def get_repo_count(self) -> int:
        """Return progress accepted during this invocation only."""
        return self._accepted_count

    def get_existing_repository_names(self) -> set[str]:
        if self.manifest_status == "not_requested":
            try:
                self._manifest = load_manifest(self)
                self.manifest_status = "available" if self._manifest else "empty"
            except ManifestUnavailable:
                self.manifest_status = "unavailable"
                logging.getLogger(__name__).warning(
                    "Backend corpus manifest unavailable; proceeding without inventory deduplication"
                )
        now = datetime.now(timezone.utc)
        return self._accepted_names | {
            item["full_name"] for item in (self._manifest or {}).values()
            if 0 <= (now - timestamp(item["content_updated_at"])).total_seconds() < self.refresh_after_seconds
        }

    def upsert_repositories_detailed(
        self, sources: list[Any]
    ) -> RepositoryDeliveryResult:
        result = RepositoryDeliveryResult()

        def encoded_records() -> Iterator[tuple[str, str, bytes]]:
            for source in sources:
                try:
                    record = repository_upsert_record(source)
                    name, github_id = record["full_name"], record["github_id"]
                    encoded = json.dumps(
                        record, separators=(",", ":"), ensure_ascii=False
                    ).encode("utf-8")
                    # Do not retain the full normalized record during delivery.
                    del record
                    if len(encoded) + _REPOSITORIES_ENVELOPE_BYTES > MAX_BODY_BYTES:
                        raise ValueError("backend request exceeds the 8 MiB body limit")
                    yield name, github_id, encoded
                except Exception as exc:
                    result.failed[_source_name(source)] = str(exc)[:500]

        for batch in _bounded_batches(encoded_records()):
            names = [name for name, _ in batch.identities]
            try:
                response = self._post_encoded(
                    "/api/internal/v2/ingestion/repositories/upsert",
                    batch.body,
                )
                mappings = response.get("mappings")
                if not isinstance(mappings, list):
                    raise ValueError("backend response is missing mappings")
                by_github_id = {github_id: name for name, github_id in batch.identities}
                mapped_names: set[str] = set()
                for mapping in mappings:
                    github_id = str(mapping.get("github_id") or "")
                    name = by_github_id.get(github_id)
                    if not name:
                        raise ValueError(f"unexpected github_id mapping: {github_id!r}")
                    uuid.UUID(str(mapping.get("repo_id") or ""))
                    content_version = mapping.get("content_version")
                    if isinstance(content_version, bool) or not isinstance(content_version, int):
                        raise ValueError("content_version must be an integer")
                    result.mappings[github_id] = dict(mapping)
                    result.succeeded.append(name)
                    self._accepted_names.add(name.casefold())
                    mapped_names.add(name)
                for name in names:
                    if name not in mapped_names:
                        result.failed[name] = "backend returned no repository mapping"
            except Exception as exc:
                for name in names:
                    result.failed[name] = str(exc)[:500]

        self._accepted_count += result.count
        return result

    def publish_trending_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post(
            "/api/internal/v2/ingestion/trending/snapshots", payload
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(
            payload, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return self._post_encoded(path, encoded)

    def _post_encoded(self, path: str, encoded: bytes) -> dict[str, Any]:
        if len(encoded) > MAX_BODY_BYTES:
            raise ValueError("backend request exceeds the 8 MiB body limit")
        headers = {
            "content-type": "application/json",
            "x-internal-secret": self.internal_secret,
            "x-request-id": str(uuid.uuid4()),
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}{path}",
                    data=encoded,
                    headers=headers,
                    timeout=self.timeout,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(
                        f"retryable backend response {response.status_code}",
                        response=response,
                    )
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict):
                    raise ValueError("backend response must be a JSON object")
                return body
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is not None and status < 500 and status != 429:
                    break
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2**attempt, 4))
        raise RuntimeError(f"backend ingestion failed: {last_error}") from last_error


def repository_upsert_record(source: Any) -> dict[str, Any]:
    raw_warnings = (
        source.get("warnings", [])
        if isinstance(source, dict)
        else getattr(source, "warnings", [])
    )
    warnings = (
        [raw_warnings]
        if isinstance(raw_warnings, str)
        else list(raw_warnings or [])
    )
    if warnings:
        raise ValueError(
            "repository enrichment is incomplete: " + "; ".join(map(str, warnings))[:500]
        )
    payload = source if isinstance(source, dict) else getattr(source, "payload", {})
    raw = source if isinstance(source, dict) else getattr(source, "raw_repository", {})
    full_name = str(payload.get("full_name") or payload.get("id") or raw.get("full_name") or "").strip()
    owner, separator, name = full_name.partition("/")
    if separator != "/" or not owner or not name or "/" in name:
        raise ValueError("full_name must use owner/repository format")
    github_id = str(payload.get("github_id") or raw.get("github_id") or "").strip()
    if not github_id.isdecimal():
        raise ValueError("github_id must be a decimal string")
    github_node_id = str(
        payload.get("github_node_id") or raw.get("github_node_id") or ""
    ).strip()
    if not github_node_id:
        raise ValueError("github_node_id is required")
    raw_owner = raw.get("owner") if isinstance(raw.get("owner"), dict) else {}
    owner_github_id = str(
        payload.get("owner_github_id") or raw_owner.get("github_id") or ""
    ).strip()
    if not owner_github_id.isdecimal() or int(owner_github_id) <= 0:
        raise ValueError("owner_github_id must be a positive decimal string")
    readme_document = getattr(source, "readme", None)
    canonical_readme = getattr(readme_document, "raw_markdown", None)
    if canonical_readme is None:
        canonical_readme = payload.get("readme")
    readme_source_path = (
        getattr(readme_document, "source_path", None)
        or payload.get("readme_source_path")
    )
    readme_default_branch = (
        getattr(readme_document, "default_branch", None)
        or payload.get("readme_default_branch")
        or raw.get("default_branch")
    )
    readme_base_url = (
        getattr(readme_document, "base_url", None)
        or payload.get("readme_base_url")
    )
    canonical_readme_text = (
        None
        if canonical_readme is None or canonical_readme == ""
        else str(canonical_readme)
    )
    return {
        "github_id": github_id,
        "github_node_id": github_node_id,
        "owner_github_id": owner_github_id,
        "full_name": full_name,
        "owner": owner,
        "name": name,
        "url": payload.get("html_url") or raw.get("html_url") or raw.get("url"),
        "description": str(payload.get("description") or ""),
        "readme": canonical_readme_text,
        "readme_length": len(canonical_readme_text or ""),
        "readme_source_path": readme_source_path,
        "readme_default_branch": readme_default_branch,
        "readme_base_url": readme_base_url,
        "primary_language": str(payload.get("primary_language") or "Unknown"),
        "languages": list(payload.get("languages") or []),
        "topics": list(payload.get("topics") or []),
        "star_count": max(int(payload.get("star_count") or 0), 0),
        "fork_count": max(int(payload.get("fork_count") or 0), 0),
        "open_issues_count": max(int(payload.get("open_issues_count") or 0), 0),
        "pushed_at": payload.get("pushed_at"),
        "observed_at": payload.get("observed_at"),
    }


def _bounded_batches(
    records: Iterable[tuple[str, str, bytes]],
) -> Iterator[_EncodedRepositoryBatch]:
    identities: list[tuple[str, str]] = []
    body = bytearray(_REPOSITORIES_PREFIX)
    for name, github_id, encoded in records:
        size = len(body) + bool(identities) + len(encoded) + len(_REPOSITORIES_SUFFIX)
        if identities and (size > MAX_BODY_BYTES or len(identities) == MAX_RECORDS):
            body.extend(_REPOSITORIES_SUFFIX)
            batch = _EncodedRepositoryBatch(identities, bytes(body))
            identities = []
            body = bytearray(_REPOSITORIES_PREFIX)
            yield batch
        if identities:
            body.extend(b",")
        body.extend(encoded)
        identities.append((name, github_id))
    if identities:
        body.extend(_REPOSITORIES_SUFFIX)
        batch = _EncodedRepositoryBatch(identities, bytes(body))
        body.clear()
        yield batch


def _source_name(source: Any) -> str:
    payload = source if isinstance(source, dict) else getattr(source, "payload", {})
    return str(payload.get("full_name") or payload.get("id") or "unknown/repository")
