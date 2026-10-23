"""Failure-path tests for production corpus acquisition orchestration."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone
import threading
from types import SimpleNamespace

import pytest

from acquisition.checkpoint import CorpusCheckpoint
from acquisition.config import CorpusPipelineSettings
from acquisition.corpus_pipeline import CorpusPipeline
from acquisition.identity import deduplicate_candidates, normalize_repository_name
from acquisition.models import AcquisitionFailure, AcquisitionRunResult
from acquisition.pipeline import enrich_repositories
from acquisition.repository_enricher import RepositoryEnricher
from database.connector import RepositoryUpsertResult
from main import _parse_args


def _source(name: str):
    return SimpleNamespace(repo_id=name, payload={"id": name})


class _Database:
    enabled = True

    def __init__(self, outcome: RepositoryUpsertResult | None = None) -> None:
        self.count = 0
        self.outcome = outcome or RepositoryUpsertResult()

    def verify_connection(self) -> bool:
        return True

    def init_db(self) -> None:
        pass

    def get_repo_count(self) -> int:
        return self.count

    def get_existing_repository_names(self) -> set[str]:
        return set()

    def upsert_repositories_detailed(self, sources):
        self.count += len(self.outcome.succeeded)
        return self.outcome

    def get_repositories_by_full_names(self, names):
        return [{"id": name, "full_name": name} for name in names]


def _settings(
    path: Path, *, target: int = 50_000, max_cycles: int = 1
) -> CorpusPipelineSettings:
    return CorpusPipelineSettings(
        target_count=target,
        max_cycles=max_cycles,
        checkpoint_path=path,
    )


class _SuccessfulDatabase(_Database):
    def upsert_repositories_detailed(self, sources):
        names = [source.repo_id for source in sources]
        self.count += len(names)
        return RepositoryUpsertResult(succeeded=names)


def test_identity_normalization_and_case_insensitive_deduplication():
    assert (
        normalize_repository_name(" https://github.com/OpenAI/Codex/ ")
        == "OpenAI/Codex"
    )
    unique, removed = deduplicate_candidates(
        ["OpenAI/Codex", {"full_name": "openai/codex"}, "invalid"]
    )
    assert unique == ["OpenAI/Codex"]
    assert removed == 2


def test_transient_readme_failure_is_reported_for_retry(monkeypatch):
    def return_warning(_self, batch):
        name = batch[0]
        return [SimpleNamespace(repo_id=name, warnings=["README fetch failed"])]

    monkeypatch.setattr(RepositoryEnricher, "get_repositories_batch", return_warning)
    result = enrich_repositories("test-token", ["owner/repo"], batch_size=1, workers=1)

    assert result.repositories == []
    assert len(result.failures) == 1
    assert result.failures[0].stage == "readme_enrichment"


def test_concurrent_enrichment_preserves_discovery_order(monkeypatch):
    second_batch_completed = threading.Event()

    def finish_second_batch_first(_self, batch):
        name = batch[0]
        if name == "owner/first":
            assert second_batch_completed.wait(timeout=1)
        else:
            second_batch_completed.set()
        return [SimpleNamespace(repo_id=name, warnings=[])]

    monkeypatch.setattr(
        RepositoryEnricher,
        "get_repositories_batch",
        finish_second_batch_first,
    )
    result = enrich_repositories(
        "test-token",
        ["owner/first", "owner/second"],
        batch_size=1,
        workers=2,
    )

    assert [repository.repo_id for repository in result.repositories] == [
        "owner/first",
        "owner/second",
    ]


def test_pipeline_indexes_only_successfully_persisted_subset(tmp_path):
    path = tmp_path / "checkpoint.json"
    checkpoint = CorpusCheckpoint(path)
    database = _Database(
        RepositoryUpsertResult(
            succeeded=["owner/good"],
            failed={"owner/bad": "constraint failure"},
        )
    )
    indexed = []
    pipeline = CorpusPipeline(
        database=database,
        acquire=lambda **_: AcquisitionRunResult(
            repositories=[_source("owner/good"), _source("owner/bad")],
            discovered_count=2,
        ),
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (values, []),
        indexer=lambda values: indexed.extend(values) or values,
        settings=_settings(path),
        checkpoint=checkpoint,
    )

    report = pipeline.run(limit=2, batch_size=1, workers=1, min_readme_chars=1)

    assert [item.repo_id for item in indexed] == ["owner/good"]
    assert report.persisted == 1
    assert report.indexed == 1
    assert checkpoint.pending_persistence == ["owner/bad"]
    assert checkpoint.pending_index == []


def test_enrichment_failure_is_checkpointed_for_identity_only_retry(tmp_path):
    path = tmp_path / "checkpoint.json"
    checkpoint = CorpusCheckpoint(path)
    pipeline = CorpusPipeline(
        database=_Database(),
        acquire=lambda **_: AcquisitionRunResult(
            failures=[
                AcquisitionFailure(
                    "owner/retry", "readme_enrichment", "temporary timeout"
                )
            ],
            discovered_count=1,
        ),
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (values, []),
        indexer=lambda values: values,
        settings=_settings(path),
        checkpoint=checkpoint,
    )

    report = pipeline.run(limit=1, batch_size=1, workers=1, min_readme_chars=1)

    assert checkpoint.pending_persistence == ["owner/retry"]
    assert "owner/retry" in report.failures
    assert "temporary timeout" in path.read_text(encoding="utf-8")


def test_rejected_repository_never_reaches_postgres_or_qdrant(tmp_path):
    path = tmp_path / "checkpoint.json"
    database = _Database()
    database.upsert_repositories_detailed = lambda _: pytest.fail(
        "rejected repository reached Postgres"
    )
    indexed = []
    rejected_source = _source("owner/rejected")
    pipeline = CorpusPipeline(
        database=database,
        acquire=lambda **_: AcquisitionRunResult(repositories=[rejected_source]),
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (
            [],
            [(values[0], ["no README", "shell repository"])],
        ),
        indexer=lambda values: indexed.extend(values) or values,
        settings=_settings(path),
    )

    report = pipeline.run(limit=1, batch_size=1, workers=1, min_readme_chars=1)

    assert report.rejected == 1
    assert report.persisted == 0
    assert indexed == []
    checkpoint = CorpusCheckpoint(path)
    assert checkpoint.data["rejected"]["owner/rejected"]["reasons"] == [
        "no README",
        "shell repository",
    ]


def test_zero_database_progress_stops_additional_cycles(tmp_path):
    path = tmp_path / "checkpoint.json"
    database = _Database(RepositoryUpsertResult(succeeded=["owner/repo"]))
    database.upsert_repositories_detailed = lambda _: database.outcome
    acquisition_calls = 0

    def acquire(**_):
        nonlocal acquisition_calls
        acquisition_calls += 1
        return AcquisitionRunResult(repositories=[_source("owner/repo")])

    pipeline = CorpusPipeline(
        database=database,
        acquire=acquire,
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (values, []),
        indexer=lambda values: values,
        settings=_settings(path, target=10, max_cycles=5),
    )

    pipeline.run(limit=1, batch_size=1, workers=1, min_readme_chars=1)

    assert acquisition_calls == 1


def test_max_cycles_bounds_successful_acquisition(tmp_path):
    path = tmp_path / "checkpoint.json"
    database = _SuccessfulDatabase()
    acquisition_calls = 0

    def acquire(**_):
        nonlocal acquisition_calls
        acquisition_calls += 1
        return AcquisitionRunResult(
            repositories=[_source(f"owner/repo-{acquisition_calls}")]
        )

    pipeline = CorpusPipeline(
        database=database,
        acquire=acquire,
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (values, []),
        indexer=lambda values: values,
        settings=_settings(path, target=10, max_cycles=2),
    )

    report = pipeline.run(limit=1, batch_size=1, workers=1, min_readme_chars=1)

    assert acquisition_calls == 2
    assert report.persisted == 2
    assert report.indexed == 2


def test_batch_index_failure_isolates_repo_and_resume_clears_checkpoint(tmp_path):
    path = tmp_path / "checkpoint.json"
    database = _SuccessfulDatabase()
    sources = [_source("owner/good"), _source("owner/retry")]
    indexing_calls = []

    def partially_failing_indexer(values):
        names = [value.repo_id for value in values]
        indexing_calls.append(names)
        if len(values) > 1:
            raise RuntimeError("batch unavailable")
        if names == ["owner/retry"]:
            raise RuntimeError("point unavailable")
        return values

    first_run = CorpusPipeline(
        database=database,
        acquire=lambda **_: AcquisitionRunResult(repositories=sources),
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (values, []),
        indexer=partially_failing_indexer,
        settings=_settings(path, target=2),
        checkpoint=CorpusCheckpoint(path, clock=lambda: datetime(2026, 10, 23, tzinfo=timezone.utc)),
    )
    first_report = first_run.run(limit=2, batch_size=2, workers=1, min_readme_chars=1)

    assert indexing_calls == [
        ["owner/good", "owner/retry"],
        ["owner/good"],
        ["owner/retry"],
    ]
    assert first_report.indexed == 1
    assert CorpusCheckpoint(path).pending_index == ["owner/retry"]

    resumed = []
    second_run = CorpusPipeline(
        database=database,
        acquire=lambda **_: pytest.fail("resume should happen before discovery"),
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (values, []),
        indexer=lambda values: resumed.extend(values) or values,
        settings=_settings(path, target=2),
        checkpoint=CorpusCheckpoint(path, clock=lambda: datetime(2026, 10, 23, tzinfo=timezone.utc) + timedelta(minutes=2)),
    )
    second_report = second_run.run(limit=1, batch_size=1, workers=1, min_readme_chars=1)

    assert resumed == [{"id": "owner/retry", "full_name": "owner/retry"}]
    assert second_report.resumed_indexing == 1
    assert CorpusCheckpoint(path).pending_index == []


def test_pending_index_is_resumed_before_new_discovery(tmp_path):
    path = tmp_path / "checkpoint.json"
    checkpoint = CorpusCheckpoint(path)
    checkpoint.add_pending_index(["owner/repo"])
    checkpoint.save()
    database = _Database()
    database.count = 1
    indexed = []
    pipeline = CorpusPipeline(
        database=database,
        acquire=lambda **_: pytest.fail("discovery should not run after target"),
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (values, []),
        indexer=lambda values: indexed.extend(values) or values,
        settings=_settings(path, target=1),
        checkpoint=CorpusCheckpoint(path),
    )

    report = pipeline.run(limit=1, batch_size=1, workers=1, min_readme_chars=1)

    assert indexed == [{"id": "owner/repo", "full_name": "owner/repo"}]
    assert report.resumed_indexing == 1
    assert pipeline.checkpoint.pending_index == []


def test_pipeline_refuses_implicit_qdrant_only_production_run(tmp_path):
    pipeline = CorpusPipeline(
        database=SimpleNamespace(enabled=False),
        acquire=lambda **_: [],
        acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (values, []),
        indexer=lambda values: values,
        settings=_settings(tmp_path / "checkpoint.json"),
    )

    with pytest.raises(RuntimeError, match="requires a verified Postgres"):
        pipeline.run(limit=1, batch_size=1, workers=1, min_readme_chars=1)


def test_checkpoint_round_trip_is_atomic_and_contains_only_retry_state(tmp_path):
    path = tmp_path / "nested" / "checkpoint.json"
    checkpoint = CorpusCheckpoint(path)
    checkpoint.add_pending_persistence(["Owner/Repo", "owner/repo"])
    checkpoint.record_failure("Owner/Repo", "persistence", "failed")
    checkpoint.save()

    loaded = CorpusCheckpoint(path)
    assert loaded.pending_persistence == ["Owner/Repo"]
    assert "readme" not in path.read_text(encoding="utf-8").lower()


def test_invalid_environment_and_cli_values_fail_before_pipeline(monkeypatch):
    monkeypatch.setenv("CORPUS_TARGET_COUNT", "0")
    with pytest.raises(ValueError, match="CORPUS_TARGET_COUNT"):
        CorpusPipelineSettings.from_environment()

    monkeypatch.setenv("CORPUS_TARGET_COUNT", "50000")
    with pytest.raises(SystemExit):
        _parse_args(["--workers", "0"])
    with pytest.raises(SystemExit):
        _parse_args(["--min-readme-chars", "-1"])


def test_settings_validation_returns_runtime_safe_integer_values(tmp_path):
    settings = CorpusPipelineSettings(
        target_count="5",  # type: ignore[arg-type]
        max_cycles="2",  # type: ignore[arg-type]
        checkpoint_path=tmp_path / "checkpoint.json",
    ).validated()

    assert settings.target_count == 5
    assert settings.max_cycles == 2


def _ownership_pipeline(path, database=None, **kwargs):
    defaults = dict(
        database=database or _SuccessfulDatabase(),
        acquire=lambda **_: [],
        acquire_retries=lambda names: [_source(name) for name in names],
        quality_filter=lambda values, **_: (values, []),
        indexer=lambda _: pytest.fail("backend-owned indexing reached local indexer"),
        settings=_settings(path),
        indexing_ownership="backend_outbox",
    )
    defaults.update(kwargs)
    return CorpusPipeline(**defaults)


def _run_pipeline(pipeline):
    return pipeline.run(limit=2, batch_size=1, workers=1, min_readme_chars=1)


def test_backend_handoff_does_not_accumulate_local_work_or_stale_failures(tmp_path):
    path = tmp_path / "checkpoint.json"
    checkpoint = CorpusCheckpoint(path)
    checkpoint.add_pending_persistence(["Owner/Repo"])
    checkpoint.record_failure("Owner/Repo", "persistence", "offline")
    checkpoint.record_rejection("Owner/Repo", ["old rejection"])
    checkpoint.save()
    for _ in range(3):
        pipeline = _ownership_pipeline(path, acquire=lambda **_: [_source("owner/repo")])
        report = _run_pipeline(pipeline)
        loaded = CorpusCheckpoint(path)
        assert loaded.pending_persistence == loaded.pending_index == []
        assert loaded.data["failures"] == loaded.data["rejected"] == {}
        assert report.indexed == 0
        assert report.backend_handoffs == report.persisted
    assert len(path.read_bytes()) < 1_000


def test_backend_mode_preserves_legacy_and_genuinely_deferred_local_work(tmp_path):
    path = tmp_path / "checkpoint.json"
    checkpoint = CorpusCheckpoint(path)
    checkpoint.data["legacy_pending_index"] = ["owner/unknown"]
    checkpoint.add_pending_index(["owner/local"])
    checkpoint.record_failure("owner/local", "indexing", "previous local failure")
    checkpoint.data["failures"]["owner/local"]["first_failed_at"] = "2020-01-01T00:00:00Z"
    checkpoint.save()
    report = _run_pipeline(_ownership_pipeline(path, acquire=lambda **_: [_source("owner/new")]))
    loaded = CorpusCheckpoint(path)
    assert loaded.legacy_pending_index == ["owner/unknown"]
    assert loaded.pending_index == ["owner/local"]
    assert loaded.data["failures"]["owner/local"]["disposition"] == "retryable"
    assert report.legacy_pending_index == 1
    assert report.backend_handoffs == 1


def test_disabled_indexing_remains_deferred_for_legacy_callers(tmp_path):
    path = tmp_path / "checkpoint.json"
    pipeline = _ownership_pipeline(
        path, indexing_ownership=None, indexing_enabled=False,
        acquire=lambda **_: [_source("owner/repo")],
    )
    report = _run_pipeline(pipeline)
    assert pipeline.indexing_ownership == "local_deferred"
    assert CorpusCheckpoint(path).pending_index == ["owner/repo"]
    assert report.backend_handoffs == report.indexed == 0


def test_backend_client_cannot_silently_take_the_local_indexing_path(tmp_path):
    from acquisition.backend_client import BackendIngestionClient

    backend = BackendIngestionClient(base_url="https://backend.test", internal_secret="test")
    with pytest.raises(ValueError, match="backend_outbox"):
        _ownership_pipeline(tmp_path / "checkpoint.json", database=backend, indexing_ownership=None)
    with pytest.raises(ValueError, match="Conflicting"):
        _ownership_pipeline(tmp_path / "checkpoint.json", indexing_enabled=True)


def test_unacknowledged_delivery_survives_worker_crash_and_restart(tmp_path):
    path = tmp_path / "checkpoint.json"
    database = _SuccessfulDatabase()

    def crash_after_acceptance(_sources):
        assert CorpusCheckpoint(path).pending_persistence == ["owner/repo"]
        raise RuntimeError("response lost after commit")

    database.upsert_repositories_detailed = crash_after_acceptance
    with pytest.raises(RuntimeError, match="response lost"):
        _run_pipeline(_ownership_pipeline(path, database, acquire=lambda **_: [_source("owner/repo")]))
    assert CorpusCheckpoint(path).pending_persistence == ["owner/repo"]
    report = _run_pipeline(_ownership_pipeline(path))
    assert report.resumed_persistence == report.backend_handoffs == 1
    assert CorpusCheckpoint(path).pending_persistence == []


def test_partial_backend_acknowledgement_keeps_only_failed_work(tmp_path):
    path = tmp_path / "checkpoint.json"
    database = _Database(RepositoryUpsertResult(
        succeeded=["owner/good"], failed={"owner/bad": "temporary failure"},
    ))
    pipeline = _ownership_pipeline(path, database, acquire=lambda **_: [
        _source("owner/good"), _source("owner/bad"),
    ])
    report = _run_pipeline(pipeline)
    checkpoint = CorpusCheckpoint(path)
    assert report.backend_handoffs == 1
    assert checkpoint.pending_persistence == ["owner/bad"]
    assert checkpoint.pending_index == []


@pytest.mark.parametrize("outcome", [
    RepositoryUpsertResult(succeeded=["owner/unexpected"]),
    RepositoryUpsertResult(succeeded=["owner/repo"], failed={"OWNER/REPO": "conflict"}),
])
def test_conflicting_acknowledgements_preserve_work_for_reconciliation(tmp_path, outcome):
    path = tmp_path / "checkpoint.json"
    with pytest.raises(ValueError, match="acknowledgement"):
        _run_pipeline(_ownership_pipeline(path, _Database(outcome), acquire=lambda **_: [_source("owner/repo")]))
    assert CorpusCheckpoint(path).pending_persistence == ["owner/repo"]


def test_missing_acknowledgement_is_reported_and_remains_pending(tmp_path):
    path = tmp_path / "checkpoint.json"
    report = _run_pipeline(_ownership_pipeline(path, _Database(), acquire=lambda **_: [_source("owner/repo")]))
    assert report.backend_handoffs == 0
    assert "owner/repo" in report.failures
    assert CorpusCheckpoint(path).pending_persistence == ["owner/repo"]


def test_retry_selection_is_bounded_and_failures_rotate_without_losing_work(tmp_path):
    path = tmp_path / "checkpoint.json"
    checkpoint = CorpusCheckpoint(path)
    checkpoint.add_pending_persistence(["owner/first", "owner/second", "owner/third"])
    checkpoint.save()
    calls = []

    def fail(names):
        calls.append(names)
        return AcquisitionRunResult(failures=[AcquisitionFailure(names[0], "enrichment", "offline")])

    for _ in range(3):
        pipeline = _ownership_pipeline(path, acquire_retries=fail, settings=CorpusPipelineSettings(
            checkpoint_path=path, recovery_batch_size=1,
        ))
        _run_pipeline(pipeline)
    assert calls == [["owner/first"], ["owner/second"], ["owner/third"]]
    assert set(CorpusCheckpoint(path).pending_persistence) == {"owner/first", "owner/second", "owner/third"}


def test_resumed_quality_rejection_resolves_failure_and_is_counted(tmp_path):
    path = tmp_path / "checkpoint.json"
    now = datetime(2026, 10, 23, tzinfo=timezone.utc)
    checkpoint = CorpusCheckpoint(path, clock=lambda: now)
    checkpoint.add_pending_persistence(["owner/repo"])
    checkpoint.record_failure("owner/repo", "enrichment", "temporary failure")
    checkpoint.save()
    pipeline = _ownership_pipeline(path, checkpoint=CorpusCheckpoint(path, clock=lambda: now + timedelta(minutes=2)), quality_filter=lambda values, **_: (
        [], [(value, ["no README"]) for value in values],
    ))
    report = _run_pipeline(pipeline)
    checkpoint = CorpusCheckpoint(path)
    assert report.rejected == 1
    assert checkpoint.pending_persistence == []
    assert checkpoint.data["failures"] == {}
    assert "owner/repo" in checkpoint.data["rejected"]


def test_resumed_local_index_success_resolves_stale_failure(tmp_path):
    path = tmp_path / "checkpoint.json"
    now = datetime(2026, 10, 23, tzinfo=timezone.utc)
    checkpoint = CorpusCheckpoint(path, clock=lambda: now)
    checkpoint.add_pending_index(["owner/repo"])
    checkpoint.record_failure("OWNER/REPO", "indexing", "offline")
    checkpoint.save()
    report = _run_pipeline(_ownership_pipeline(
        path, indexing_ownership="local_immediate", indexer=lambda values: values,
        checkpoint=CorpusCheckpoint(path, clock=lambda: now + timedelta(minutes=2)),
    ))
    assert report.resumed_indexing == 1
    assert CorpusCheckpoint(path).data["failures"] == {}


def test_global_github_incident_preserves_partial_work_without_spending_repo_attempts(tmp_path):
    from acquisition.models import AcquisitionConfigurationError

    path = tmp_path / "checkpoint.json"
    acquired = AcquisitionRunResult(
        repositories=[_source("owner/resolved")],
        failures=[AcquisitionFailure("owner/auth", "metadata", "authorization failed", disposition="configuration")],
    )
    with pytest.raises(AcquisitionConfigurationError, match="configuration incident"):
        _run_pipeline(_ownership_pipeline(path, acquire=lambda **_: acquired))
    checkpoint = CorpusCheckpoint(path)
    assert set(checkpoint.pending_persistence) == {"owner/auth", "owner/resolved"}
    assert checkpoint.data["failures"] == {}
    assert checkpoint.data["last_run"]["status"] == "paused"


def test_full_name_takes_precedence_over_backend_uuid_for_recovery_identity(tmp_path):
    from acquisition.models import CorpusRunReport

    path = tmp_path / "checkpoint.json"
    database = _Database(RepositoryUpsertResult(succeeded=["owner/repo"]))
    pipeline = _ownership_pipeline(path, database)
    with pipeline.checkpoint.session():
        pipeline._persist_and_index([
            {"id": "00000000-0000-0000-0000-000000000001", "full_name": "owner/repo"},
        ], database_ready=True, report=CorpusRunReport())
    assert CorpusCheckpoint(path).pending_persistence == []


@pytest.mark.parametrize('streaming', [False, True])
def test_rejection_overflow_retains_every_received_batch_identity(tmp_path, streaming):
    from acquisition.checkpoint_policy import CheckpointPolicy
    path = tmp_path / 'checkpoint.json'
    checkpoint = CorpusCheckpoint(path, policy=CheckpointPolicy(max_history_entries=1))
    checkpoint.record_rejection('owner/previous', ['no README'])
    checkpoint.save()
    values = [_source('owner/approved'), _source('owner/rejected'), _source('owner/remaining')]
    def acquire(**kwargs):
        result = AcquisitionRunResult(repositories=values)
        if streaming:
            kwargs['on_batch'](result)
            return AcquisitionRunResult()
        return result
    pipeline = _ownership_pipeline(
        path, checkpoint=checkpoint, streaming=streaming, acquire=acquire,
        quality_filter=lambda values, **_: (values[:1], [(value, ['no README']) for value in values[1:]]),
    )
    with pytest.raises(RuntimeError, match='rejection history capacity'):
        _run_pipeline(pipeline)
    assert set(CorpusCheckpoint(path).pending_persistence) == {value.repo_id for value in values}


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('partial_failure', [False, True])
def test_local_only_indexing_releases_successful_work_and_preserves_failures(tmp_path, streaming, partial_failure):
    path = tmp_path / 'checkpoint.json'
    sources = [_source('owner/good'), _source('owner/retry')]
    def acquire(**kwargs):
        result = AcquisitionRunResult(repositories=sources)
        if streaming:
            kwargs['on_targets']([source.repo_id for source in sources])
            kwargs['on_batch'](result)
            return AcquisitionRunResult()
        return result
    def index(values):
        if partial_failure and any(source.repo_id == 'owner/retry' for source in values):
            raise RuntimeError('local index unavailable')
        return values
    pipeline = _ownership_pipeline(
        path, database=SimpleNamespace(enabled=False),
        indexing_ownership='local_immediate', allow_qdrant_without_postgres=True,
        acquire=acquire, indexer=index, streaming=streaming,
    )
    report = _run_pipeline(pipeline)
    checkpoint = CorpusCheckpoint(path)
    expected = ['owner/retry'] if partial_failure else []
    assert checkpoint.pending_persistence == expected
    assert checkpoint.pending_index == []
    assert set(checkpoint.data['failures']) == set(expected)
    assert checkpoint.excluded_names() == set(expected)
    assert report.indexed == (1 if partial_failure else 2)
    assert report.persisted == report.backend_handoffs == 0
