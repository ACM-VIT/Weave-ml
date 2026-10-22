"""Backend v2 publisher implementing the scheduler's storage boundary."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import threading
import time
from typing import Any
import uuid

from acquisition.backend_client import (
    MAX_BODY_BYTES,
    BackendIngestionClient,
    repository_upsert_record,
)
from acquisition.identity import repository_identity_key
from acquisition.models import AcquisitionRunResult
from acquisition.pipeline import enrich_repositories

logger = logging.getLogger(__name__)
_RETRY_CACHE_SECONDS = 300


class BackendTrendingStorage:
    """Enrich and atomically publish one complete trending snapshot."""

    enabled = True

    def __init__(
        self,
        *,
        backend: BackendIngestionClient,
        github_token: str,
    ) -> None:
        self.backend = backend
        self._github_token = github_token
        self._last_refresh: datetime | None = None
        self._refresh_lock = threading.Lock()
        self._retry_key: str | None = None
        self._retry_expires = 0.0
        # Immutable encoded records cannot be mutated while adding rank/score.
        self._retry_records: dict[str, bytes] = {}
        self._retry_bytes = 0
        self.last_enrichment_counts: dict[str, int] = {}

    def init_schema(self) -> None:
        """Compatibility no-op: backend migrations own durable schemas."""

    def get_last_refresh_time(self) -> datetime | None:
        return self._last_refresh

    def _enrich_candidates(self, repositories: list[dict[str, Any]]) -> AcquisitionRunResult:
        return enrich_repositories(
            self._github_token, repositories, metadata_projection="delivery"
        )

    def _clear_retry(self) -> None:
        self._retry_records.clear()
        self._retry_bytes = 0
        self._retry_key = None

    def upsert_repositories(
        self,
        repositories: list[dict[str, Any]],
        refresh_timestamp: datetime | None = None,
    ) -> int:
        # One storage instance owns one bounded pending snapshot. Concurrent
        # refreshes must not mix records or overwrite each other's retry state.
        with self._refresh_lock:
            return self._publish_complete_snapshot(repositories, refresh_timestamp)

    def _publish_complete_snapshot(
        self,
        repositories: list[dict[str, Any]],
        refresh_timestamp: datetime | None,
    ) -> int:
        computed_at = refresh_timestamp or datetime.now(timezone.utc)
        identities = [repository_identity_key(repo.get("full_name")) for repo in repositories]
        if not identities or not all(identities) or len(set(identities)) != len(identities):
            raise ValueError("trending snapshot requires unique valid repository identities")
        # Include every input field and its ranked position, not just names:
        # changed daily stars, descriptions, versions or limits invalidate reuse.
        snapshot_key = hashlib.sha256(json.dumps(
            repositories, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")).hexdigest()
        now = time.monotonic()
        if snapshot_key != self._retry_key or now >= self._retry_expires:
            self._clear_retry()
            self._retry_key = snapshot_key
            self._retry_expires = now + _RETRY_CACHE_SECONDS

        pending = [repo for key, repo in zip(identities, repositories) if key not in self._retry_records]
        reused = len(identities) - len(pending)
        if pending:
            result = self._enrich_candidates(pending)
            pending_keys = {repository_identity_key(repo["full_name"]) for repo in pending}
            failed_keys = {repository_identity_key(failure.repository) for failure in result.failures}
            if not failed_keys.issubset(pending_keys):
                raise RuntimeError("refusing trending enrichment with unexpected failure identities")
            seen: set[str] = set()
            for source in result.repositories:
                key = repository_identity_key(source.repo_id)
                if key not in pending_keys or key in seen:
                    raise RuntimeError("refusing trending enrichment with unexpected or duplicate identities")
                seen.add(key)
                if key in failed_keys or list(getattr(source, "warnings", []) or []):
                    continue
                try:
                    record = repository_upsert_record(source)
                    if repository_identity_key(record["full_name"]) != key:
                        raise ValueError("enrichment identity mismatch")
                    encoded = json.dumps(
                        record, separators=(",", ":"), ensure_ascii=False, allow_nan=False
                    ).encode("utf-8")
                except (TypeError, ValueError, KeyError):
                    # Invalid records are missing work, never fallback content.
                    continue
                if self._retry_bytes + len(encoded) > MAX_BODY_BYTES:
                    self._clear_retry()
                    raise ValueError("refusing trending snapshot larger than the 8 MiB transport limit")
                self._retry_records[key] = encoded
                self._retry_bytes += len(encoded)

        missing = [key for key in identities if key not in self._retry_records]
        self.last_enrichment_counts = {
            "fetched": len(pending), "reused": reused,
            "failed": len(missing), "fallback": 0,
        }
        logger.info(
            "Trending enrichment: requested=%d fetched=%d reused=%d failed=%d fallback=0",
            len(identities), len(pending), reused, len(missing),
        )
        if missing:
            raise RuntimeError(
                f"refusing incomplete trending snapshot: enriched {len(identities) - len(missing)}/"
                f"{len(identities)} repositories; metadata or README acquisition failed"
            )

        records: list[dict[str, Any]] = []
        for rank, (key, trend) in enumerate(zip(identities, repositories), start=1):
            record = json.loads(self._retry_records[key])
            record["rank"] = rank
            if trend.get("daily_stars") is not None:
                record["score"] = float(trend.get("daily_stars") or 0)
            records.append(record)
        if len({record["github_id"] for record in records}) != len(records):
            self._clear_retry()
            raise ValueError("trending snapshot requires unique GitHub repository IDs")
        snapshot = {
            "snapshot_id": str(uuid.uuid4()),
            "period": "daily",
            "computed_at": computed_at.astimezone(timezone.utc).isoformat(),
            "source": "github-trending",
            "repositories": records,
        }
        encoded_size = len(
            json.dumps(snapshot, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode(
                "utf-8"
            )
        )
        if encoded_size > MAX_BODY_BYTES:
            self._clear_retry()
            raise ValueError(
                "refusing trending snapshot larger than the 8 MiB transport limit"
            )
        self.backend.publish_trending_snapshot(snapshot)
        self._last_refresh = computed_at
        self._clear_retry()
        return len(records)
