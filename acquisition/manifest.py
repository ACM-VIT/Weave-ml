"""Bounded offline consumption of the backend-owned conservative inventory."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

from .identity import normalize_repository_name, repository_identity_key

MAX_PAGE_BYTES = 512 * 1024
MAX_MANIFEST_ROWS = 100_000


class ManifestUnavailable(RuntimeError):
    pass


def timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise TypeError("manifest timestamp must be a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("manifest timestamp must include timezone")
    return parsed


def load_manifest(client: Any) -> dict[str, dict[str, Any]]:
    """Discard partial scans on failure; an unavailable inventory is never empty."""
    records: dict[str, dict[str, Any]] = {}
    names: set[str] = set()
    github_ids: set[str] = set()
    cursor = None
    cursors: set[str] = set()
    as_of = None
    previous_id = ""
    deadline = time.monotonic() + 120
    for _ in range(MAX_MANIFEST_ROWS // 250):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ManifestUnavailable("manifest scan exceeded its time budget")
        try:
            page = _read_page(client, cursor, remaining)
            if (
                not isinstance(page, dict)
                or type(page.get("version")) is not int
                or page["version"] != 1
            ):
                raise ValueError("unsupported manifest version")
            cutoff = page.get("as_of")
            if timestamp(cutoff) > datetime.now(timezone.utc):
                raise ValueError("manifest cutoff is in the future")
            if as_of is not None and cutoff != as_of:
                raise ValueError("manifest cutoff changed between pages")
            as_of = cutoff
            items = page.get("items")
            if not isinstance(items, list) or len(items) > 250:
                raise ValueError("invalid manifest page size")
            for item in items:
                if not isinstance(item, dict):
                    raise TypeError("invalid manifest entry")
                if set(item) != {
                    "repo_id",
                    "github_id",
                    "full_name",
                    "content_hash",
                    "content_version",
                    "content_updated_at",
                    "source_updated_at",
                }:
                    raise ValueError("unexpected manifest entry fields")
                repo_id = item.get("repo_id")
                if (
                    not isinstance(repo_id, str)
                    or str(uuid.UUID(repo_id)) != repo_id
                    or repo_id <= previous_id
                ):
                    raise ValueError("noncanonical or unordered repository identity")
                previous_id = repo_id
                github_id = item.get("github_id")
                if (
                    not isinstance(github_id, str)
                    or not github_id.isascii()
                    or not github_id.isdecimal()
                    or not 0 < int(github_id)
                    or str(int(github_id)) != github_id
                    or len(github_id) > 30
                ):
                    raise ValueError("invalid GitHub identity")
                raw_name = item.get("full_name")
                name = normalize_repository_name(raw_name)
                if not name or name != raw_name or len(name) > 256:
                    raise ValueError("invalid repository name")
                key = repository_identity_key(name)
                if key in names or github_id in github_ids:
                    raise ValueError("duplicate canonical manifest identity")
                names.add(key)
                github_ids.add(github_id)
                if (
                    type(item.get("content_version")) is not int
                    or item["content_version"] < 1
                ):
                    raise ValueError("invalid content version")
                if (
                    not isinstance(item.get("content_hash"), str)
                    or not 1 <= len(item["content_hash"]) <= 128
                ):
                    raise ValueError("invalid content hash")
                if timestamp(item.get("content_updated_at")) > timestamp(as_of):
                    raise ValueError("entry exceeds manifest cutoff")
                if item.get("source_updated_at") is not None:
                    timestamp(item["source_updated_at"])
                records[repo_id] = {**item, "full_name": key}
            next_cursor = page["next_cursor"]
            if next_cursor is None:
                return records
            if (
                not items
                or not isinstance(next_cursor, str)
                or not 1 <= len(next_cursor) <= 2048
                or next_cursor in cursors
            ):
                raise ValueError("invalid manifest continuation")
            cursors.add(next_cursor)
            cursor = next_cursor
        except Exception as exc:
            # No response body, credentials, or request headers in the public error.
            raise ManifestUnavailable(
                "backend manifest unavailable or invalid"
            ) from exc
    raise ManifestUnavailable("manifest exceeds the bounded inventory size")


def _read_page(client: Any, cursor: str | None, remaining: float):
    async def read():
        async with (
            httpx.AsyncClient(
                transport=client.manifest_transport,
                timeout=min(client.timeout, 5, remaining),
            ) as session,
            session.stream(
                "GET",
                f"{client.base_url}/api/internal/v2/ingestion/repositories/manifest",
                params={"limit": 250, **({"cursor": cursor} if cursor else {})},
                headers={"x-internal-secret": client.internal_secret},
            ) as response,
        ):
            response.raise_for_status()
            body = bytearray()
            async for chunk in response.aiter_bytes(16384):
                body.extend(chunk)
                if len(body) > MAX_PAGE_BYTES:
                    raise ValueError("manifest response exceeded its byte budget")
            return json.loads(body)

    async def bounded_read():
        return await asyncio.wait_for(read(), timeout=remaining)

    return asyncio.run(bounded_read())
