"""Acquisition pipeline logic for discovering and enriching GitHub repositories."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from .identity import (
    deduplicate_candidates,
    repository_identity_key,
    repository_name_from_candidate,
)
from .models import AcquisitionFailure, AcquisitionRunResult

logger = logging.getLogger("pipeline.acquisition")

if TYPE_CHECKING:
    from acquisition.repository_enricher import RepositoryEnricher


def run_acquisition(
    token: str,
    *,
    limit: int = 150,
    batch_size: int = 15,
    workers: int = 4,
    existing_repos: set[str] | None = None,
) -> list[Any]:
    """
    Discover and enrich GitHub repositories via GraphQL only.

    Returns a list of EnrichmentResult objects. Each carries:
      .repo_id          — "owner/repo"
      .payload          — Osiris-compatible dict (star_count, language, topics, …)
      .raw_repository   — raw GraphQL response fields
      .readme           — ReadmeDocument (canonical raw Markdown, source
                          metadata, and separately derived clean text)
      .topics           — list[str]
      .languages        — dict[str, int]  (language → bytes)
    """
    return run_acquisition_detailed(
        token,
        limit=limit,
        batch_size=batch_size,
        workers=workers,
        existing_repos=existing_repos,
    ).repositories


def run_acquisition_detailed(
    token: str,
    *,
    limit: int = 150,
    batch_size: int = 15,
    workers: int = 4,
    existing_repos: set[str] | None = None,
) -> AcquisitionRunResult:
    """Discover and enrich repositories with structured audit information."""
    _validate_inputs(token=token, limit=limit, batch_size=batch_size, workers=workers)

    from acquisition.github_graphql_client import GitHubGraphQLClient
    from acquisition.github_discovery import GitHubDiscoveryEngine, DiscoveryConfig

    client   = GitHubGraphQLClient(token=token)
    # Fetch a larger buffer of candidate repositories to account for filtering duplicates
    discovery_limit = limit + 50 if existing_repos else limit + 20
    config   = DiscoveryConfig(total_limit=discovery_limit)
    discovery = GitHubDiscoveryEngine(client, config=config)

    # ── Step 1: Discovery ─────────────────────────────────────────────────────
    logger.info("Discovering repositories …")
    discovered = discovery.discover(limit=discovery_limit)
    logger.info("Discovered %d candidate repos", len(discovered))

    unique_discovered, duplicates_removed = deduplicate_candidates(discovered)
    existing_keys = {repository_identity_key(name) for name in (existing_repos or set())}
    new_candidates = [
        candidate
        for candidate in unique_discovered
        if repository_identity_key(repository_name_from_candidate(candidate)) not in existing_keys
    ]
    targets = new_candidates[:limit]
    skipped_existing = len(unique_discovered) - len(new_candidates)
    logger.info(
        "Discovery audit: %d raw, %d duplicate/invalid, %d existing, %d selected.",
        len(discovered), duplicates_removed, skipped_existing, len(targets),
    )

    enriched = enrich_repositories(
        token,
        targets,
        batch_size=batch_size,
        workers=workers,
    )
    enriched.discovered_count = len(discovered)
    enriched.skipped_existing_count = skipped_existing
    enriched.duplicates_removed = duplicates_removed
    logger.info("Acquisition complete — %d / %d repos enriched", len(enriched.repositories), limit)
    return enriched


def enrich_repository_ids(
    token: str,
    repositories: list[str],
    *,
    batch_size: int = 15,
    workers: int = 4,
) -> AcquisitionRunResult:
    """Enrich explicit repository names, primarily for checkpoint retries."""
    _validate_inputs(token=token, limit=max(len(repositories), 1), batch_size=batch_size, workers=workers)
    return enrich_repositories(token, repositories, batch_size=batch_size, workers=workers)


def enrich_repositories(
    token: str,
    repositories: list[Any],
    *,
    batch_size: int = 15,
    workers: int = 4,
) -> AcquisitionRunResult:
    """Enrich an explicit candidate list concurrently and return stable ordering."""
    _validate_inputs(token=token, limit=max(len(repositories), 1), batch_size=batch_size, workers=workers)
    targets, duplicates_removed = deduplicate_candidates(repositories)
    batches = [targets[i : i + batch_size] for i in range(0, len(targets), batch_size)]
    logger.info(
        "Enriching %d repos in %d batch(es) of up to %d with %d concurrent worker(s) …",
        len(targets), len(batches), batch_size, workers,
    )
    failures: list[AcquisitionFailure] = []
    results_by_batch: dict[int, list[Any]] = {}

    # Each worker thread gets its own GitHubGraphQLClient (and requests.Session)
    # so concurrent threads never race on a shared Session object.
    _thread_local = threading.local()

    def _get_thread_enricher() -> "RepositoryEnricher":
        if not hasattr(_thread_local, "enricher"):
            from acquisition.github_graphql_client import GitHubGraphQLClient
            from acquisition.repository_enricher import RepositoryEnricher
            _thread_local.enricher = RepositoryEnricher(
                graphql_client=GitHubGraphQLClient(token=token)
            )
        return _thread_local.enricher

    def _enrich_batch(batch: list[Any]) -> list[Any]:
        # get_repositories_batch() uses a two-pass approach: lean metadata first,
        # README fetched separately. Transient README failures are surfaced as
        # retryable structured failures after the batch returns.
        return _get_thread_enricher().get_repositories_batch(batch)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Each future processes one batch of batch_size repos. workers controls
        # how many batches run concurrently; batch_size controls the GraphQL
        # request payload size. Both levers are now effective.
        futures = {executor.submit(_enrich_batch, batch): (i, batch) for i, batch in enumerate(batches)}
        for future in as_completed(futures):
            batch_index, batch = futures[future]
            try:
                results = future.result()
                clean_results = []
                for result in results:
                    warnings = list(getattr(result, "warnings", []))
                    if warnings:
                        failures.append(
                            AcquisitionFailure(
                                result.repo_id,
                                "readme_enrichment",
                                "; ".join(warnings)[:500],
                            )
                        )
                    else:
                        clean_results.append(result)
                results_by_batch[batch_index] = clean_results
                returned = {repository_identity_key(result.repo_id) for result in results}
                for candidate in batch:
                    name = repository_name_from_candidate(candidate)
                    if repository_identity_key(name) not in returned:
                        failures.append(
                            AcquisitionFailure(name, "enrichment", "repository was not returned by enrichment")
                        )
                logger.info(
                    "  Batch %d/%d → +%d enriched  (total: %d)",
                    batch_index + 1,
                    len(batches),
                    len(clean_results),
                    sum(len(value) for value in results_by_batch.values()),
                )
            except Exception as exc:
                logger.warning("  Batch %d/%d failed: %s", batch_index + 1, len(batches), exc)
                results_by_batch[batch_index] = []
                failures.extend(
                    AcquisitionFailure(repository_name_from_candidate(candidate), "batch_enrichment", str(exc))
                    for candidate in batch
                )

    ordered_results = [
        result
        for batch_index in range(len(batches))
        for result in results_by_batch.get(batch_index, [])
    ]
    return AcquisitionRunResult(
        repositories=ordered_results,
        failures=failures,
        discovered_count=len(repositories),
        duplicates_removed=duplicates_removed,
    )


def _validate_inputs(*, token: str, limit: int, batch_size: int, workers: int) -> None:
    if not isinstance(token, str) or not token.strip():
        raise ValueError("GitHub token must not be empty")
    for name, value in (("limit", limit), ("batch_size", batch_size), ("workers", workers)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
