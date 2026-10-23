"""Backpressure and restart contracts for streaming offline acquisition."""
from concurrent.futures import CancelledError
import threading
from types import SimpleNamespace

import pytest

from acquisition.checkpoint import CorpusCheckpoint
from acquisition.config import CorpusPipelineSettings
from acquisition.corpus_pipeline import CorpusPipeline
from acquisition.models import AcquisitionRunResult
from acquisition.pipeline import enrich_repositories
from acquisition.repository_enricher import RepositoryEnricher
from acquisition.backend_client import RepositoryDeliveryResult


def source(name):
    return SimpleNamespace(repo_id=name, warnings=[], payload={})


def test_stream_sink_applies_backpressure_without_retaining_completed_results(monkeypatch):
    started = []
    received = []
    in_sink = threading.Event()
    release = threading.Event()
    def enrich(_self, names):
        started.extend(names)
        return [source(name) for name in names]
    monkeypatch.setattr(RepositoryEnricher, 'get_repositories_batch', enrich)
    def consume(batch):
        received.extend(item.repo_id for item in batch.repositories)
        if len(received) == 1:
            in_sink.set()
            assert release.wait(2)
    result = []
    thread = threading.Thread(target=lambda: result.append(enrich_repositories(
        'test', [f'owner/repo{i}' for i in range(40)], batch_size=1, workers=2, on_batch=consume,
    )))
    thread.start()
    try:
        assert in_sink.wait(2)
        assert len(started) <= 2, 'slow delivery must stop replenishing enrichment futures'
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert len(received) == 40 and len(set(received)) == 40
    assert result[0].repositories == []


def test_stream_sink_failure_is_propagated_and_no_more_work_is_admitted(monkeypatch):
    started = []
    def enrich(_self, names):
        started.extend(names)
        return [source(name) for name in names]
    monkeypatch.setattr(RepositoryEnricher, 'get_repositories_batch', enrich)
    def consume(_batch):
        raise RuntimeError('backend unavailable')
    with pytest.raises(RuntimeError, match='backend unavailable'):
        enrich_repositories('test', [f'owner/repo{i}' for i in range(50)],
                            batch_size=1, workers=2, on_batch=consume)
    assert len(started) <= 2


def pipeline(tmp_path, acquire, database, **kwargs):
    return CorpusPipeline(
        database=database, acquire=acquire, acquire_retries=lambda _: [],
        quality_filter=lambda values, **_: (values, []), indexer=lambda _: [],
        indexing_enabled=False, streaming=True,
        settings=CorpusPipelineSettings(target_count=100, max_cycles=1, checkpoint_path=tmp_path / 'state.json'),
        **kwargs,
    )


class Database:
    enabled = True
    count = 0
    def verify_connection(self): return True
    def get_repo_count(self): return self.count
    def get_existing_repository_names(self): return set()
    def upsert_repositories_detailed(self, sources):
        self.count += len(sources)
        return RepositoryDeliveryResult(succeeded=[s.repo_id for s in sources])


def test_cancellation_keeps_selected_but_unsent_identities_retryable(tmp_path):
    names = ['owner/first', 'owner/unsent']
    def acquire(*, on_batch, on_targets, **_):
        on_targets(names)
        on_batch(AcquisitionRunResult(repositories=[source(names[0])]))
        raise CancelledError('cancel after first acknowledgement')
    with pytest.raises(CancelledError):
        pipeline(tmp_path, acquire, Database()).run(limit=2, batch_size=1, workers=1, min_readme_chars=1)
    checkpoint = CorpusCheckpoint(tmp_path / 'state.json')
    assert checkpoint.pending_persistence == ['owner/unsent']


def test_ambiguous_delivery_keeps_current_and_unsent_identities(tmp_path):
    names = ['owner/first', 'owner/unsent']
    class FailedDatabase(Database):
        def upsert_repositories_detailed(self, sources):
            raise RuntimeError('ambiguous transport')
    def acquire(*, on_batch, on_targets, **_):
        on_targets(names)
        on_batch(AcquisitionRunResult(repositories=[source(names[0])]))
        pytest.fail('failure must stop source admission')
    with pytest.raises(RuntimeError, match='ambiguous transport'):
        pipeline(tmp_path, acquire, FailedDatabase()).run(limit=2, batch_size=1, workers=1, min_readme_chars=1)
    assert CorpusCheckpoint(tmp_path / 'state.json').pending_persistence == names


def test_stream_partial_ack_does_not_index_or_clear_unacknowledged_identities(tmp_path):
    names = ['owner/good', 'owner/retry', 'owner/missing']
    class PartialDatabase(Database):
        def upsert_repositories_detailed(self, sources):
            self.count += 1
            return RepositoryDeliveryResult(succeeded=[names[0]], failed={names[1]: 'retry'})
    def acquire(*, on_batch, on_targets, **_):
        on_targets(names)
        on_batch(AcquisitionRunResult(repositories=[source(n) for n in names]))
        return AcquisitionRunResult(discovered_count=3)
    indexed = []
    p = pipeline(tmp_path, acquire, PartialDatabase())
    p.indexing_enabled = True
    p.indexer = lambda values: indexed.extend(s.repo_id for s in values)
    report = p.run(limit=3, batch_size=3, workers=1, min_readme_chars=1)
    assert indexed == [names[0]]
    assert report.persisted == 1 and report.enriched == 3 and report.discovered == 3
    assert CorpusCheckpoint(tmp_path / 'state.json').pending_persistence == names[1:]


def test_stream_emits_enrichment_failures_without_valid_empty_success(monkeypatch):
    def enrich(_self, names):
        return [SimpleNamespace(repo_id=names[0], warnings=['temporary README failure'])]
    monkeypatch.setattr(RepositoryEnricher, 'get_repositories_batch', enrich)
    batches = []
    result = enrich_repositories('test', ['owner/retry', 'owner/missing'],
                                 batch_size=2, workers=1, on_batch=batches.append)
    assert result.repositories == [] and result.failures == []
    assert len(batches) == 1 and batches[0].repositories == []
    assert {f.repository for f in batches[0].failures} == {'owner/retry', 'owner/missing'}


def test_stream_cancellation_in_sink_stops_new_enrichment(monkeypatch):
    event = threading.Event()
    started = []
    def enrich(_self, names):
        started.extend(names)
        return [source(n) for n in names]
    monkeypatch.setattr(RepositoryEnricher, 'get_repositories_batch', enrich)
    with pytest.raises(CancelledError):
        enrich_repositories('test', [f'owner/repo{i}' for i in range(20)], batch_size=1,
                            workers=1, cancel_event=event, on_batch=lambda _: event.set())
    assert len(started) == 1


def test_restart_reenrichment_is_bounded_and_keeps_unacknowledged_work(tmp_path):
    checkpoint = CorpusCheckpoint(tmp_path / 'state.json')
    names = [f'owner/repo{i}' for i in range(23)]
    checkpoint.add_pending_persistence(names)
    checkpoint.save()
    sizes = []
    def retry(batch):
        sizes.append(len(batch))
        return AcquisitionRunResult(repositories=[source(n) for n in batch])
    p = pipeline(tmp_path, lambda **_: AcquisitionRunResult(), Database(), checkpoint=checkpoint)
    p.acquire_retries = retry
    p.run(limit=1, batch_size=4, workers=1, min_readme_chars=1)
    assert sizes == [4, 4, 4, 4, 4, 3]
    assert CorpusCheckpoint(tmp_path / 'state.json').pending_persistence == []


def test_discovery_saves_targets_before_source_work_and_streams_to_sink(monkeypatch):
    from acquisition.pipeline import run_acquisition_detailed
    from acquisition.github_discovery import GitHubDiscoveryEngine
    names = ['owner/one', 'owner/two', 'owner/three']
    selected = []
    emitted = []
    monkeypatch.setattr(GitHubDiscoveryEngine, 'discover', lambda *_, **__: names)
    def enrich(_self, batch):
        assert selected == names, 'retry identities must be recorded before enrichment'
        return [source(n) for n in batch]
    monkeypatch.setattr(RepositoryEnricher, 'get_repositories_batch', enrich)
    result = run_acquisition_detailed('test', limit=3, batch_size=1, workers=1,
        on_targets=selected.extend, on_batch=lambda batch: emitted.extend(s.repo_id for s in batch.repositories))
    assert emitted == names and result.repositories == [] and result.discovered_count == 3


def test_all_failed_delivery_stops_producer_and_preserves_unsent_names(tmp_path):
    names = ['owner/failed', 'owner/unsent']
    class Unavailable(Database):
        def upsert_repositories_detailed(self, sources):
            return RepositoryDeliveryResult(failed={sources[0].repo_id: 'HTTP503'})
    def acquire(*, on_batch, on_targets, **_):
        on_targets(names)
        on_batch(AcquisitionRunResult(repositories=[source(names[0])]))
        pytest.fail('outage must prevent accepting more source work')
    with pytest.raises(RuntimeError, match='no progress'):
        pipeline(tmp_path, acquire, Unavailable()).run(limit=2, batch_size=1, workers=1, min_readme_chars=1)
    assert CorpusCheckpoint(tmp_path / 'state.json').pending_persistence == names


def test_restart_stops_retry_admission_on_backend_outage(tmp_path):
    names = [f'owner/repo{i}' for i in range(12)]
    checkpoint = CorpusCheckpoint(tmp_path / 'state.json')
    checkpoint.add_pending_persistence(names)
    checkpoint.save()
    calls = []
    class Unavailable(Database):
        def upsert_repositories_detailed(self, sources):
            return RepositoryDeliveryResult(failed={s.repo_id: 'HTTP503' for s in sources})
    p = pipeline(tmp_path, lambda **_: pytest.fail('no new acquisition on outage'), Unavailable(), checkpoint=checkpoint)
    def retry(batch):
        calls.append(batch)
        return AcquisitionRunResult(repositories=[source(n) for n in batch])
    p.acquire_retries = retry
    with pytest.raises(RuntimeError, match='no progress'):
        p.run(limit=1, batch_size=4, workers=1, min_readme_chars=1)
    assert calls == [names[:4]]
    assert CorpusCheckpoint(tmp_path / 'state.json').pending_persistence == names


def test_interrupted_real_discovery_preserves_audit_totals(tmp_path, monkeypatch):
    from acquisition.pipeline import run_acquisition_detailed
    from acquisition.github_discovery import GitHubDiscoveryEngine
    event = threading.Event()
    monkeypatch.setattr(GitHubDiscoveryEngine, 'discover', lambda *_, **__: [
        'owner/one', 'owner/one', 'owner/existing', 'owner/two',
    ])
    monkeypatch.setattr(RepositoryEnricher, 'get_repositories_batch',
                        lambda _self, names: [source(n) for n in names])
    class Interrupted(Database):
        def get_existing_repository_names(self): return {'owner/existing'}
        def upsert_repositories_detailed(self, sources):
            result = super().upsert_repositories_detailed(sources)
            event.set()
            return result
    def acquire(**kwargs):
        return run_acquisition_detailed('test', cancel_event=event, **kwargs)
    with pytest.raises(CancelledError):
        pipeline(tmp_path, acquire, Interrupted()).run(limit=3, batch_size=1, workers=1, min_readme_chars=1)
    checkpoint = CorpusCheckpoint(tmp_path / 'state.json')
    report = checkpoint.data['last_run']
    assert report['discovered'] == 4 and report['skipped_existing'] == 1 and report['duplicates_removed'] == 1
    assert report['persisted'] == 1 and checkpoint.pending_persistence == ['owner/two']


def test_successful_batches_bound_checkpoint_fsyncs(tmp_path, monkeypatch):
    import os
    calls = []
    fsync = os.fsync
    def measured(fd):
        calls.append(fd)
        fsync(fd)
    monkeypatch.setattr('acquisition.checkpoint.os.fsync', measured)
    names = [f'owner/repo{i}' for i in range(10)]
    def acquire(*, on_batch, on_targets, **_):
        on_targets(names)
        for name in names:
            on_batch(AcquisitionRunResult(repositories=[source(name)]))
        return AcquisitionRunResult(discovered_count=len(names))
    pipeline(tmp_path, acquire, Database()).run(limit=10, batch_size=1, workers=1, min_readme_chars=1)
    # Per-batch stage+ack writes and three run-level writes. An additional
    # unchanged fsync after every acknowledgement violates this IO budget.
    assert len(calls) <= 2 * len(names) + 3
    assert CorpusCheckpoint(tmp_path / 'state.json').pending_persistence == []


def test_delivered_objects_are_released_before_replacement_enrichment(monkeypatch):
    import weakref
    class Source:
        def __init__(self, name):
            self.repo_id = name
            self.warnings = []
            self.payload = {'readme': name + 'x' * 60_000}
    prior = []
    def enrich(_self, names):
        assert all(ref() is None for ref in prior), 'previous batch escaped the bounded workset'
        items = [Source(name) for name in names]
        prior.extend(weakref.ref(item) for item in items)
        return items
    monkeypatch.setattr(RepositoryEnricher, 'get_repositories_batch', enrich)
    received = []
    enrich_repositories('test', [f'owner/repo{i}' for i in range(40)], batch_size=2, workers=1,
                        on_batch=lambda batch: received.extend(item.repo_id for item in batch.repositories))
    assert len(received) == 40 and all(ref() is None for ref in prior)
