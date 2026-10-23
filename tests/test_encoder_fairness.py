"""Shared encoder progress and cancellation, independent of model/network timing."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time

import numpy as np
import pytest

from config import REPOSITORY_EMBEDDING_DIM
from embedding.embeddings import SentenceTransformerEmbedder
from embedding.query_cache import CachedQueryEmbedder
from embedding.runtime import BoundedEmbeddingModel
from utils.deadline import CooperativeDeadlineExceeded, RequestDeadline, deadline_scope


class ControlledEncoder:
    """Every actual model call waits for a test-owned release event."""

    max_seq_length = 256

    def __init__(self):
        self.started = queue.Queue()
        self.lock = threading.Lock()
        self.releases = []
        self.closing = False
        self.active = self.max_active = 0
        self.calls = []

    def encode(self, texts, **kwargs):
        release = threading.Event()
        with self.lock:
            if self.closing:
                release.set()
            self.releases.append(release)
            self.calls.append(list(texts))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.started.put((list(texts), release))
        try:
            assert release.wait(5), 'test did not release an actual encode'
            return np.ones((len(texts), REPOSITORY_EMBEDDING_DIM), dtype=np.float32)
        finally:
            with self.lock:
                self.active -= 1

    def close(self):
        with self.lock:
            self.closing = True
            for release in self.releases:
                release.set()


def wait_for_model(model, *, waiting, active=1):
    # Observe actual arrival before releasing a barrier; no scheduling sleeps
    # determine which caller is expected to be ahead in the FIFO.
    stop = time.monotonic() + 2
    while time.monotonic() < stop:
        with model._condition:
            if len(model._waiting) == waiting and model._active == active:
                return
        time.sleep(.001)
    raise AssertionError(f'encoder did not reach waiting={waiting}, active={active}')


def test_grouped_readme_and_interactive_streams_make_fifo_progress():
    raw = ControlledEncoder()
    model = BoundedEmbeddingModel(raw, max_concurrency=1)
    embedder = SentenceTransformerEmbedder(model=model)
    search = CachedQueryEmbedder(
        model=model, cache=model.query_embedding_cache,
        model_revision='isolated-fairness-revision', embedding_version='isolated-v1',
        expected_size=REPOSITORY_EMBEDDING_DIM,
    )
    readme_chunks = [f'repository:{index}' for index in range(25)]
    order = []

    def interactive(operation):
        target = search if operation == 'search' else embedder
        return [target.embed_text(f'{operation}:{index}') for index in range(3)]

    with ThreadPoolExecutor(max_workers=3) as workers:
        try:
            repository = workers.submit(embedder.embed_texts, readme_chunks)
            first = raw.started.get(timeout=2)
            assert first[0] == readme_chunks[:8]
            queries = workers.submit(interactive, 'search')
            wait_for_model(model, waiting=1)
            profiles = workers.submit(interactive, 'onboard')
            wait_for_model(model, waiting=2)
            remaining = {'repository': 4, 'search': 3, 'onboard': 3}
            call = first
            for _ in range(sum(remaining.values())):
                texts, release = call
                operation = texts[0].split(':')[0]
                order.append(operation)
                remaining[operation] -= 1
                # Keep all other unfinished streams queued before allowing the
                # current encode to finish and its owner to request another turn.
                wait_for_model(model, waiting=sum(
                    count > 0 for name, count in remaining.items() if name != operation
                ))
                release.set()
                if any(remaining.values()):
                    call = raw.started.get(timeout=2)
            assert len(repository.result(timeout=2)) == 25
            assert len(queries.result(timeout=2)) == len(profiles.result(timeout=2)) == 3
            assert order == ['repository', 'search', 'onboard'] * 3 + ['repository']
            assert [text for call in raw.calls for text in call if text.startswith('repository:')] == readme_chunks
            assert raw.max_active == 1
            assert all(len(call) <= 8 for call in raw.calls)
        finally:
            raw.close()
    wait_for_model(model, waiting=0, active=0)


@pytest.mark.parametrize('interruption', ['expiry', 'cancellation'])
def test_expired_or_cancelled_queued_work_never_invokes_encoder(interruption):
    raw = ControlledEncoder()
    model = BoundedEmbeddingModel(raw, max_concurrency=1)
    now = [0.0]
    deadline = RequestDeadline(1, clock=lambda: now[0])

    def queued():
        with deadline_scope(deadline):
            return model.encode(['expired-search'])

    with ThreadPoolExecutor(max_workers=3) as workers:
        try:
            running = workers.submit(model.encode, ['repository'])
            _, release_running = raw.started.get(timeout=2)
            expired = workers.submit(queued)
            wait_for_model(model, waiting=1)
            if interruption == 'expiry':
                now[0] = 2
            else:
                deadline.cancel()
            with model._condition:
                model._condition.notify_all()
            with pytest.raises(CooperativeDeadlineExceeded):
                expired.result(timeout=2)
            wait_for_model(model, waiting=0)
            assert raw.calls == [['repository']]
            successor = workers.submit(model.encode, ['onboard'])
            wait_for_model(model, waiting=1)
            assert raw.started.empty()
            release_running.set()
            texts, release_successor = raw.started.get(timeout=2)
            assert texts == ['onboard']
            release_successor.set()
            running.result(timeout=2)
            successor.result(timeout=2)
            assert raw.max_active == 1
        finally:
            raw.close()
    wait_for_model(model, waiting=0, active=0)


@pytest.mark.anyio
async def test_cancelled_awaiter_retains_encoder_slot_until_actual_encode_exits():
    raw = ControlledEncoder()
    model = BoundedEmbeddingModel(raw, max_concurrency=1)
    owner = asyncio.create_task(asyncio.to_thread(model.encode, ['repository']))
    successor = None
    try:
        _, release_owner = await asyncio.to_thread(raw.started.get, True, 2)
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        successor = asyncio.create_task(asyncio.to_thread(model.encode, ['search']))
        await asyncio.to_thread(wait_for_model, model, waiting=1)
        assert raw.active == 1 and raw.started.empty()
        release_owner.set()
        texts, release_successor = await asyncio.to_thread(raw.started.get, True, 2)
        assert texts == ['search']
        release_successor.set()
        await successor
        assert raw.max_active == 1
    finally:
        raw.close()
        await asyncio.gather(*(task for task in (owner, successor) if task is not None), return_exceptions=True)
        await asyncio.to_thread(wait_for_model, model, waiting=0, active=0)
