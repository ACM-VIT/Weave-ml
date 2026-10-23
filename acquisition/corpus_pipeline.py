"""Production orchestration for bounded, resumable offline corpus ingestion."""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any, Literal

from .checkpoint import CorpusCheckpoint
from .config import CorpusPipelineSettings
from .identity import normalize_repository_name, repository_identity_key
from .models import AcquisitionConfigurationError, AcquisitionRunResult, CorpusRunReport

logger = logging.getLogger("pipeline.corpus")

AcquireCallable = Callable[..., AcquisitionRunResult | list[Any]]
FilterCallable = Callable[..., tuple[list[Any], list[tuple[Any, list[str]]]]]
IndexCallable = Callable[[list[Any]], list[Any]]
IndexingOwnership = Literal["backend_outbox", "local_immediate", "local_deferred"]


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
        indexing_enabled: bool | None = None,
        indexing_ownership: IndexingOwnership | None = None,
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
        if indexing_ownership is None:
            # Preserve legacy callers: disabled local indexing is deferred,
            # never evidence of a durable handoff to the backend.
            indexing_ownership = (
                "local_deferred" if indexing_enabled is False else "local_immediate"
            )
        if indexing_ownership not in ("backend_outbox", "local_immediate", "local_deferred"):
            raise ValueError("Invalid indexing_ownership")
        if indexing_enabled is not None and indexing_enabled != (indexing_ownership == "local_immediate"):
            raise ValueError("Conflicting indexing_enabled and indexing_ownership")
        if (
            getattr(database, "indexing_ownership", None) == "backend_outbox"
            and indexing_ownership != "backend_outbox"
        ):
            raise ValueError("Backend delivery requires backend_outbox indexing ownership")
        if indexing_ownership == "backend_outbox" and allow_qdrant_without_postgres:
            raise ValueError("Backend indexing ownership requires acknowledged persistence")
        self.indexing_ownership = indexing_ownership
        self.indexing_enabled = indexing_ownership == "local_immediate"
        self.streaming = streaming

    def run(
        self,
        *,
        limit: int,
        batch_size: int,
        workers: int,
        min_readme_chars: int,
    ) -> CorpusRunReport:
        with self.checkpoint.session():
            return self._run(
                limit=limit, batch_size=batch_size, workers=workers,
                min_readme_chars=min_readme_chars,
            )

    def _run(
        self,
        *,
        limit: int,
        batch_size: int,
        workers: int,
        min_readme_chars: int,
    ) -> CorpusRunReport:
        self._configuration_paused = False
        self.checkpoint.prune_history()
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
        report.legacy_pending_index = len(self.checkpoint.legacy_pending_index)
        if report.legacy_pending_index:
            logger.warning("%d legacy index entries require ownership reconciliation; preserving them.", report.legacy_pending_index)
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
                capacity = self.checkpoint.remaining_capacity()
                if capacity == 0:
                    raise RuntimeError("Acquisition checkpoint capacity exhausted; reconcile pending work")
                cycle_limit = min(cycle_limit, capacity)
                existing = (
                    self.database.get_existing_repository_names()
                    if database_ready
                    else set()
                )
                existing = set(existing) | self.checkpoint.excluded_names()
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
                    self.checkpoint.add_pending_persistence([
                        *[_source_name(source) for source in acquired.repositories],
                        *[failure.repository for failure in acquired.failures],
                    ])
                    self.checkpoint.save()
                    self._pause_on_configuration_failure(acquired)
                    enriched_count += len(acquired.repositories)
                    report.enriched += len(acquired.repositories)
                    for failure in acquired.failures:
                        report.failures[failure.repository] = f"{failure.stage}: {failure.error}"
                        self.checkpoint.record_failure(failure.repository, failure.stage, failure.error, disposition=failure.disposition)
                        self.checkpoint.add_pending_persistence([failure.repository])
                    self.checkpoint.save()
                    approved, rejected = self.quality_filter(
                        acquired.repositories, min_readme_chars=min_readme_chars,
                    )
                    report.rejected += len(rejected)
                    for source, reasons in rejected:
                        name = _source_name(source)
                        self._record_rejection(name, reasons)
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
            report.pending_delivery = len(self.checkpoint.pending_persistence)
            report.pending_local_index = len(self.checkpoint.pending_index)
            report.quarantined = sum(failure.get("disposition") == "quarantined" for failure in self.checkpoint.data["failures"].values())
            result = report.as_dict()
            if self._configuration_paused:
                result.update(status="paused", reason="github_configuration")
            self.checkpoint.record_run(result)
            self.checkpoint.save()

    def _resume_indexing(self, report: CorpusRunReport) -> None:
        if not self.indexing_enabled:
            return
        pending = self.checkpoint.due(self.checkpoint.pending_index, self.settings.recovery_batch_size)
        if not pending:
            return
        self.checkpoint.clear_pending_index(pending)
        self.checkpoint.add_pending_index(pending)
        self.checkpoint.save()
        payloads = self.database.get_repositories_by_full_names(pending)
        found = {_source_key(source) for source in payloads}
        for name in pending:
            if repository_identity_key(name) not in found:
                message = "persisted repository could not be reconstructed for indexing"
                report.failures[name] = message
                self.checkpoint.record_failure(name, "resume_index", message)
        succeeded, failed = self._index_sources(payloads)
        self.checkpoint.clear_pending_index(succeeded)
        for name in succeeded:
            self.checkpoint.clear_failure(name)
        for name, error in failed.items():
            report.failures[name] = error
            self.checkpoint.record_failure(name, "resume_index", error)
        report.indexed += len(succeeded)
        report.resumed_indexing += len(succeeded)
        self.checkpoint.save()

    def _resume_persistence(
        self, report: CorpusRunReport, *, min_readme_chars: int, batch_size: int
    ) -> None:
        pending = self.checkpoint.due(self.checkpoint.pending_persistence, self.settings.recovery_batch_size)
        if not pending:
            return
        selected_keys = {repository_identity_key(name) for name in pending}
        self.checkpoint.set_pending_persistence([
            *[name for name in self.checkpoint.pending_persistence if repository_identity_key(name) not in selected_keys],
            *pending,
        ])
        self.checkpoint.save()
        size = batch_size if self.streaming else len(pending)
        for offset in range(0, len(pending), size):
            names = pending[offset:offset + size]
            acquired = self._as_acquisition_result(self.acquire_retries(names))
            self._pause_on_configuration_failure(acquired)
            for failure in acquired.failures:
                report.failures[failure.repository] = f"{failure.stage}: {failure.error}"
                self.checkpoint.record_failure(
                    failure.repository, failure.stage, failure.error, disposition=failure.disposition
                )
            approved, rejected = self.quality_filter(
                acquired.repositories,
                min_readme_chars=min_readme_chars,
            )
            for source, reasons in rejected:
                name = _source_name(source)
                self._record_rejection(name, reasons)
            report.rejected += len(rejected)
            before = report.persisted
            persisted = self._persist_and_index(approved, database_ready=True, report=report)
            report.resumed_persistence += report.persisted - before
            if not approved:
                self.checkpoint.save()
            if self.streaming and approved and not persisted:
                raise RuntimeError("Backend delivery made no progress; remaining identities stay retryable")

    def _record_rejection(self, name: str, reasons: list[str]) -> None:
        self.checkpoint.record_rejection(name, reasons)
        self.checkpoint.clear_pending_persistence([name])
        self.checkpoint.clear_failure(name)

    def _pause_on_configuration_failure(self, acquired: AcquisitionRunResult) -> None:
        if not any(failure.disposition == "configuration" for failure in acquired.failures):
            return
        self.checkpoint.add_pending_persistence([
            *[failure.repository for failure in acquired.failures],
            *[_source_name(source) for source in acquired.repositories],
        ])
        self._configuration_paused = True
        self.checkpoint.record_run({"status": "paused", "reason": "github_configuration"})
        self.checkpoint.save()
        raise AcquisitionConfigurationError("GitHub configuration incident; repair credentials before retrying")

    def _persist_and_index(
        self,
        sources: list[Any],
        *,
        database_ready: bool,
        report: CorpusRunReport,
    ) -> list[Any]:
        if not sources:
            self.checkpoint.save()
            return []
        if database_ready:
            # Write identities before the external side effect. A crash after
            # acceptance but before local acknowledgement replays idempotently.
            self.checkpoint.add_pending_persistence([_source_name(source) for source in sources])
            self.checkpoint.save()
            outcome = self.database.upsert_repositories_detailed(sources)
            source_keys = {_source_key(source) for source in sources}
            succeeded_keys = {
                repository_identity_key(name) for name in outcome.succeeded
            }
            failed_keys = {repository_identity_key(name) for name in outcome.failed}
            if not (succeeded_keys | failed_keys).issubset(source_keys) or succeeded_keys & failed_keys:
                raise ValueError("Invalid or conflicting persistence acknowledgement identities")
            persisted = [
                source for source in sources if _source_key(source) in succeeded_keys
            ]
            failed_names = list(outcome.failed)
            self.checkpoint.add_pending_persistence(failed_names)
            if self.indexing_ownership != "backend_outbox":
                # Keep a local recovery owner before finally-save can release
                # persistence ownership, including a later indexing crash.
                self.checkpoint.add_pending_index([_source_name(source) for source in persisted])
            self.checkpoint.clear_pending_persistence(outcome.succeeded)
            for source in persisted:
                self.checkpoint.clear_rejection(_source_name(source))
            for name, error in outcome.failed.items():
                report.failures[name] = f"persistence: {error}"
                self.checkpoint.record_failure(
                    name, "persistence", error,
                    disposition=getattr(outcome, "failure_dispositions", {}).get(name, "retryable"),
                )
            for source in sources:
                if _source_key(source) not in succeeded_keys | failed_keys:
                    name = _source_name(source)
                    message = "persistence returned no acknowledgement"
                    report.failures[name] = message
                    self.checkpoint.record_failure(name, "persistence", message)
            report.persisted += len(persisted)
        else:
            persisted = list(sources)

        persisted_names = [_source_name(source) for source in persisted]
        if self.indexing_ownership == "backend_outbox":
            for name in persisted_names:
                self.checkpoint.clear_failure(name)
            report.backend_handoffs += len(persisted)
            self.checkpoint.save()
            return persisted
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
        value = source.get("full_name") or source.get("id")
    else:
        value = getattr(source, "repo_id", "")
    name = normalize_repository_name(value)
    if not name:
        raise ValueError("Corpus source requires a valid full_name identity")
    return name


def _source_key(source: Any) -> str:
    return repository_identity_key(_source_name(source))
