"""Production orchestration for bounded, resumable offline corpus ingestion."""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

from .checkpoint import CorpusCheckpoint
from .config import CorpusPipelineSettings
from .identity import normalize_repository_name, repository_identity_key
from .models import AcquisitionRunResult, CorpusRunReport

logger = logging.getLogger("pipeline.corpus")

AcquireCallable = Callable[..., AcquisitionRunResult | list[Any]]
FilterCallable = Callable[..., tuple[list[Any], list[tuple[Any, list[str]]]]]
IndexCallable = Callable[[list[Any]], list[Any]]


class CorpusPipeline:
    """Coordinate acquisition, filtering, Postgres persistence, and indexing."""

    def __init__(
        self,
        *,
        database: Any,
        acquire: AcquireCallable,
        acquire_retries: Callable[[list[str]], AcquisitionRunResult | list[Any]],
        quality_filter: FilterCallable,
        indexer: IndexCallable,
        settings: CorpusPipelineSettings,
        checkpoint: CorpusCheckpoint | None = None,
        allow_qdrant_without_postgres: bool = False,
        indexing_enabled: bool = True,
        streaming: bool = False,
    ) -> None:
        self.database = database
        self.acquire = acquire
        self.acquire_retries = acquire_retries
        self.quality_filter = quality_filter
        self.indexer = indexer
        self.settings = settings.validated()
        self.checkpoint = checkpoint or CorpusCheckpoint(self.settings.checkpoint_path)
        self.allow_without_postgres = allow_qdrant_without_postgres
        self.indexing_enabled = indexing_enabled
        self.streaming = streaming

    def run(
        self,
        *,
        limit: int,
        batch_size: int,
        workers: int,
        min_readme_chars: int,
    ) -> CorpusRunReport:
        for name, value in (
            ("limit", limit),
            ("batch_size", batch_size),
            ("workers", workers),
            ("min_readme_chars", min_readme_chars),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        database_ready = bool(getattr(self.database, "enabled", False))
        if database_ready and hasattr(self.database, "verify_connection"):
            database_ready = bool(self.database.verify_connection())
        if not database_ready and not self.allow_without_postgres:
            raise RuntimeError(
                "Production corpus ingestion requires a verified Postgres connection. "
                "Use --allow-qdrant-without-postgres only for explicit development runs."
            )
        if database_ready and hasattr(self.database, "init_db"):
            self.database.init_db()

        report = CorpusRunReport()
        try:
            if database_ready:
                self._resume_indexing(report)
                self._resume_persistence(
                    report,
                    min_readme_chars=min_readme_chars,
                    batch_size=batch_size,
                )

            current_count = self.database.get_repo_count() if database_ready else 0
            for _cycle in range(self.settings.max_cycles):
                if database_ready and current_count >= self.settings.target_count:
                    logger.info(
                        "Corpus target reached: %d/%d repositories.",
                        current_count,
                        self.settings.target_count,
                    )
                    break

                cycle_limit = (
                    min(limit, self.settings.target_count - current_count)
                    if database_ready
                    else limit
                )
                existing = (
                    self.database.get_existing_repository_names()
                    if database_ready
                    else set()
                )
                report.manifest_status = getattr(self.database, "manifest_status", "not_supported")
                before_count = current_count
                enriched_count = 0
                persisted_count = 0
                discovery_published = False

                def discovered(audit: AcquisitionRunResult) -> None:
                    nonlocal discovery_published
                    if discovery_published:
                        raise ValueError("Discovery totals were already published for this cycle")
                    report.discovered += audit.discovered_count
                    report.skipped_existing += audit.skipped_existing_count
                    report.duplicates_removed += audit.duplicates_removed
                    discovery_published = True

                def consume(acquired: AcquisitionRunResult) -> None:
                    nonlocal enriched_count, persisted_count
                    enriched_count += len(acquired.repositories)
                    report.enriched += len(acquired.repositories)
                    for failure in acquired.failures:
                        report.failures[failure.repository] = f"{failure.stage}: {failure.error}"
                        self.checkpoint.record_failure(failure.repository, failure.stage, failure.error)
                        self.checkpoint.add_pending_persistence([failure.repository])
                    approved, rejected = self.quality_filter(
                        acquired.repositories, min_readme_chars=min_readme_chars,
                    )
                    report.rejected += len(rejected)
                    for source, reasons in rejected:
                        name = _source_name(source)
                        self.checkpoint.record_rejection(name, reasons)
                        self.checkpoint.clear_pending_persistence([name])
                    persisted = self._persist_and_index(
                        approved, database_ready=database_ready, report=report,
                    )
                    persisted_count += len(persisted)
                    if not approved or not database_ready:
                        self.checkpoint.save()
                    if self.streaming and approved and not persisted:
                        raise RuntimeError("Backend delivery made no progress; remaining identities stay retryable")

                def selected(names: list[str]) -> None:
                    # Identities only; no README/token staging. Save before workers
                    # start so interruption retains even not-yet-enriched work.
                    self.checkpoint.add_pending_persistence(names)
                    self.checkpoint.save()

                kwargs = dict(limit=cycle_limit, batch_size=batch_size,
                              workers=workers, existing_repos=existing)
                try:
                    if self.streaming:
                        acquired = self._as_acquisition_result(self.acquire(
                            **kwargs, on_batch=consume, on_targets=selected, on_discovery=discovered,
                        ))
                    else:
                        acquired = self._as_acquisition_result(self.acquire(**kwargs))
                        consume(acquired)
                    if not discovery_published:
                        # Compatibility acquire callables may not expose the early
                        # audit callback; successful returns still retain totals.
                        discovered(acquired)
                finally:
                    # Includes cancellation/BaseException; accepted subsets were
                    # cleared batch-wise, all remaining identities stay retryable.
                    self.checkpoint.record_run(report.as_dict())
                    self.checkpoint.save()

                if not enriched_count:
                    logger.warning("Acquisition made no progress; ending this bounded run.")
                    break
                if database_ready:
                    current_count = self.database.get_repo_count()
                    if not persisted_count or current_count <= before_count:
                        logger.warning(
                            "Corpus persistence made no forward progress; stopping safely."
                        )
                        break
                else:
                    break

            return report
        finally:
            # Resume can fail/cancel before discovery starts. Always replace the
            # previous invocation's audit with the progress made in this run.
            self.checkpoint.record_run(report.as_dict())
            self.checkpoint.save()

    def _resume_indexing(self, report: CorpusRunReport) -> None:
        pending = self.checkpoint.pending_index
        if not pending or not self.indexing_enabled:
            return
        payloads = self.database.get_repositories_by_full_names(pending)
        found = {_source_key(source) for source in payloads}
        for name in pending:
            if repository_identity_key(name) not in found:
                message = "persisted repository could not be reconstructed for indexing"
                report.failures[name] = message
                self.checkpoint.record_failure(name, "resume_index", message)
        succeeded, failed = self._index_sources(payloads)
        self.checkpoint.clear_pending_index(succeeded)
        for name, error in failed.items():
            report.failures[name] = error
            self.checkpoint.record_failure(name, "resume_index", error)
        report.indexed += len(succeeded)
        report.resumed_indexing += len(succeeded)
        self.checkpoint.save()

    def _resume_persistence(
        self, report: CorpusRunReport, *, min_readme_chars: int, batch_size: int
    ) -> None:
        pending = self.checkpoint.pending_persistence
        if not pending:
            return
        size = batch_size if self.streaming else len(pending)
        for offset in range(0, len(pending), size):
            names = pending[offset:offset + size]
            acquired = self._as_acquisition_result(self.acquire_retries(names))
            for failure in acquired.failures:
                report.failures[failure.repository] = f"{failure.stage}: {failure.error}"
                self.checkpoint.record_failure(
                    failure.repository, failure.stage, failure.error
                )
            approved, rejected = self.quality_filter(
                acquired.repositories,
                min_readme_chars=min_readme_chars,
            )
            for source, reasons in rejected:
                name = _source_name(source)
                self.checkpoint.record_rejection(name, reasons)
                self.checkpoint.clear_pending_persistence([name])
            before = report.persisted
            persisted = self._persist_and_index(approved, database_ready=True, report=report)
            report.resumed_persistence += report.persisted - before
            if not approved:
                self.checkpoint.save()
            if self.streaming and approved and not persisted:
                raise RuntimeError("Backend delivery made no progress; remaining identities stay retryable")

    def _persist_and_index(
        self,
        sources: list[Any],
        *,
        database_ready: bool,
        report: CorpusRunReport,
    ) -> list[Any]:
        if not sources:
            return []
        if database_ready:
            self.checkpoint.add_pending_persistence([_source_name(source) for source in sources])
            self.checkpoint.save()
            outcome = self.database.upsert_repositories_detailed(sources)
            succeeded_keys = {
                repository_identity_key(name) for name in outcome.succeeded
            }
            persisted = [
                source for source in sources if _source_key(source) in succeeded_keys
            ]
            failed_names = list(outcome.failed)
            self.checkpoint.add_pending_persistence(failed_names)
            self.checkpoint.clear_pending_persistence(outcome.succeeded)
            for name, error in outcome.failed.items():
                report.failures[name] = f"persistence: {error}"
                self.checkpoint.record_failure(name, "persistence", error)
            report.persisted += len(persisted)
        else:
            persisted = list(sources)

        persisted_names = [_source_name(source) for source in persisted]
        if not self.indexing_enabled:
            if database_ready:
                self.checkpoint.add_pending_index(persisted_names)
                self.checkpoint.save()
            return persisted

        if database_ready:
            self.checkpoint.add_pending_index(persisted_names)
            self.checkpoint.save()
        succeeded, failed = self._index_sources(persisted)
        self.checkpoint.clear_pending_index(succeeded)
        for name in succeeded:
            self.checkpoint.clear_failure(name)
        for name, error in failed.items():
            report.failures[name] = f"indexing: {error}"
            self.checkpoint.record_failure(name, "indexing", error)
        report.indexed += len(succeeded)
        self.checkpoint.save()
        return persisted

    def _index_sources(self, sources: list[Any]) -> tuple[list[str], dict[str, str]]:
        if not sources:
            return [], {}
        names = [_source_name(source) for source in sources]
        try:
            self.indexer(sources)
            return names, {}
        except Exception as batch_error:
            logger.warning(
                "Batch indexing failed; retrying repositories individually: %s",
                batch_error,
            )
        succeeded: list[str] = []
        failed: dict[str, str] = {}
        for source in sources:
            name = _source_name(source)
            try:
                self.indexer([source])
                succeeded.append(name)
            except Exception as exc:
                failed[name] = str(exc)[:500]
        return succeeded, failed

    @staticmethod
    def _as_acquisition_result(
        value: AcquisitionRunResult | list[Any],
    ) -> AcquisitionRunResult:
        if isinstance(value, AcquisitionRunResult):
            return value
        return AcquisitionRunResult(
            repositories=list(value), discovered_count=len(value)
        )


def _source_name(source: Any) -> str:
    if isinstance(source, dict):
        value = source.get("id") or source.get("full_name")
    else:
        value = getattr(source, "repo_id", "")
    return normalize_repository_name(value) or str(value or "unknown/repository")


def _source_key(source: Any) -> str:
    return repository_identity_key(_source_name(source))
