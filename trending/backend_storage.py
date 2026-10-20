"""Backend v2 publisher implementing the scheduler's storage boundary."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any
import uuid

from acquisition.backend_client import (
    MAX_BODY_BYTES,
    BackendIngestionClient,
    repository_upsert_record,
)
from acquisition.github_graphql_client import GitHubGraphQLClient
from acquisition.repository_enricher import RepositoryEnricher


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
        self.enricher = RepositoryEnricher(GitHubGraphQLClient(token=github_token))
        self._last_refresh: datetime | None = None

    def init_schema(self) -> None:
        """Compatibility no-op: backend migrations own durable schemas."""

    def get_last_refresh_time(self) -> datetime | None:
        return self._last_refresh

    def upsert_repositories(
        self,
        repositories: list[dict[str, Any]],
        refresh_timestamp: datetime | None = None,
    ) -> int:
        computed_at = refresh_timestamp or datetime.now(timezone.utc)
        enriched = self.enricher.get_repositories_batch(repositories)
        if len(enriched) != len(repositories):
            raise RuntimeError(
                f"refusing incomplete trending snapshot: enriched {len(enriched)}/"
                f"{len(repositories)} repositories"
            )
        incomplete = [
            source.repo_id
            for source in enriched
            if list(getattr(source, "warnings", []) or [])
        ]
        if incomplete:
            raise RuntimeError(
                "refusing incomplete trending snapshot: README acquisition failed for "
                + ", ".join(incomplete)
            )
        trending_by_name = {
            str(repository.get("full_name")): repository
            for repository in repositories
        }
        records: list[dict[str, Any]] = []
        for rank, source in enumerate(enriched, start=1):
            record = repository_upsert_record(source)
            trend = trending_by_name.get(record["full_name"], {})
            record["rank"] = rank
            if trend.get("daily_stars") is not None:
                record["score"] = float(trend.get("daily_stars") or 0)
            records.append(record)
        snapshot = {
            "snapshot_id": str(uuid.uuid4()),
            "period": "daily",
            "computed_at": computed_at.astimezone(timezone.utc).isoformat(),
            "source": "github-trending",
            "repositories": records,
        }
        encoded_size = len(
            json.dumps(snapshot, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        )
        if encoded_size > MAX_BODY_BYTES:
            raise ValueError(
                "refusing trending snapshot larger than the 8 MiB transport limit"
            )
        self.backend.publish_trending_snapshot(snapshot)
        self._last_refresh = computed_at
        return len(records)
