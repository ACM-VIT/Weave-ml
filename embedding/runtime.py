"""Process-scoped, bounded runtime for online embedding work.

Repository and user embeddings deliberately share one frozen encoder.  This
module owns that encoder and bounded job-class executors so API request pools
cannot create one transformer per job. Repository provider/storage waits do not
occupy onboarding workers, while admission stays bounded across both classes.
"""

from __future__ import annotations

import asyncio
import atexit
import math
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import lru_cache, wraps
from typing import Any, Callable, TypeVar

from config import EMBEDDING_MODEL_REVISION, REPOSITORY_EMBEDDING_DIM, REPOSITORY_EMBEDDING_MODEL
from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
from embedding.tower_cache import TowerEmbeddingCache
from embedding.embeddings import SentenceTransformerEmbedder
from embedding.query_cache import QueryEmbeddingCache
from scripts.user_onboarding import UserOnboardingPipeline
from summarization.runtime import (
    card_summary_cleanup_actions, card_summary_pipeline, shutdown_card_summary_runtime,
)
from utils.deadline import check_deadline, remaining_seconds
from utils.runtime_lifecycle import runtime_admission
from utils.observability import WorkObservation, span_scope


T = TypeVar("T")
DEFAULT_EMBEDDING_MODEL_REVISION = EMBEDDING_MODEL_REVISION
_warmup_loading = threading.local()


class EmbeddingCapacityError(RuntimeError):
    """Raised when the bounded embedding executor has no admission capacity."""


class EmbeddingDeadlineExceeded(RuntimeError):
    """Raised when the bounded embedding deadline was exceeded."""


def embedding_timeout_seconds() -> float:
    raw = os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("EMBEDDING_TIMEOUT_SECONDS must be a number") from exc
    if not math.isfinite(value) or not 0.1 <= value <= 300:
        raise ValueError("EMBEDDING_TIMEOUT_SECONDS must be between 0.1 and 300")
    return value


def _positive_int(name: str, default: int, *, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def embedding_model_revision() -> str:
    return (
        os.getenv("EMBEDDING_MODEL_REVISION", DEFAULT_EMBEDDING_MODEL_REVISION).strip()
        or DEFAULT_EMBEDDING_MODEL_REVISION
    )


def _production() -> bool:
    return os.getenv("APP_ENV", "development").strip().casefold() in {
        "production",
        "prod",
    }


def embedding_warmup_enabled() -> bool:
    raw = os.getenv(
        "EMBEDDING_WARMUP_ON_STARTUP", "true" if _production() else "false"
    ).strip().casefold()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError("EMBEDDING_WARMUP_ON_STARTUP must be a boolean")


def inference_warmup_timeout_seconds() -> float:
    try:
        value = float(os.getenv("INFERENCE_WARMUP_TIMEOUT_SECONDS", "120").strip() or "120")
    except ValueError as exc:
        raise ValueError("INFERENCE_WARMUP_TIMEOUT_SECONDS must be a number") from exc
    if not math.isfinite(value) or not 0.1 <= value <= 300:
        raise ValueError("INFERENCE_WARMUP_TIMEOUT_SECONDS must be between 0.1 and 300")
    return value


def _configure_cpu_threads() -> int:
    threads = _positive_int("EMBEDDING_CPU_THREADS", 1, maximum=64)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, str(threads))
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass
    return threads


def embedding_max_outstanding_jobs() -> int:
    capacity = _positive_int("EMBEDDING_MAX_OUTSTANDING_JOBS", 4, maximum=64)
    workers = _positive_int("EMBEDDING_EXECUTOR_WORKERS", 1, maximum=8)
    if capacity < workers:
        raise ValueError(
            "EMBEDDING_MAX_OUTSTANDING_JOBS must be at least "
            "EMBEDDING_EXECUTOR_WORKERS"
        )
    return capacity


def _load_sentence_transformer() -> Any:
    _configure_cpu_threads()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required for embedding jobs") from exc

    kwargs: dict[str, Any] = {"revision": embedding_model_revision()}
    if _production() or getattr(_warmup_loading, "local_files_only", False):
        kwargs["local_files_only"] = True
    return SentenceTransformer(REPOSITORY_EMBEDDING_MODEL, **kwargs)


class BoundedEmbeddingModel:
    """FIFO admission into one shared, concurrency-bounded encoder."""

    def __init__(self, model: Any, *, max_concurrency: int) -> None:
        self._model = model
        # Cache lifetime follows this exact loaded encoder, not a reusable ID.
        self.query_embedding_cache = QueryEmbeddingCache()
        self.query_embedding_revision = embedding_model_revision()
        self.tower_embedding_cache = TowerEmbeddingCache(
            model=self, model_revision=self.query_embedding_revision,
        )
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self._max_concurrency = max_concurrency
        self._condition = threading.Condition()
        self._waiting: deque[object] = deque()
        self._active = 0
        self._warmup_lock = threading.Lock()
        self._warmup_status = {"state": "not_started", "duration_seconds": None}

    @property
    def max_seq_length(self) -> int:
        return getattr(self._model, "max_seq_length", 256)

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        ticket = object()
        with self._condition:
            self._waiting.append(ticket)
            try:
                while self._waiting[0] is not ticket or self._active >= self._max_concurrency:
                    # Preserve FIFO fairness while bounding queue waits. Polling
                    # observes this job's cancellation without stealing a turn.
                    self._condition.wait(timeout=remaining_seconds(0.05))
                check_deadline()
            except BaseException:
                self._waiting.remove(ticket)
                self._condition.notify_all()
                raise
            self._waiting.popleft()
            self._active += 1
            self._condition.notify_all()
        try:
            return self._model.encode(*args, **kwargs)
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def warmup(self, *, stop: threading.Event | None = None) -> dict[str, Any]:
        """Execute one tiny real encode per loaded model, without persisted data."""
        import numpy as np

        with self._warmup_lock:
            state = self._warmup_status["state"]
            if state == "ready":
                return self.identity()
            if state == "failed":
                raise RuntimeError("Embedding inference warmup previously failed")
            started = time.monotonic()
            self._warmup_status = {"state": "warming", "duration_seconds": None}
            try:
                if stop is not None and stop.is_set():
                    raise RuntimeError("Inference warmup was stopped")
                vectors = np.asarray(self.encode(
                    ["Python developer interested in tools.", "Repository for developer tools."],
                    convert_to_numpy=True, normalize_embeddings=True,
                    show_progress_bar=False,
                ), dtype=np.float32)
                if vectors.shape != (2, REPOSITORY_EMBEDDING_DIM):
                    raise ValueError("Encoder warmup returned incompatible dimensions")
                if not np.all(np.isfinite(vectors)) or np.any(np.linalg.norm(vectors, axis=1) <= 0):
                    raise ValueError("Encoder warmup returned invalid vectors")
                if stop is not None and stop.is_set():
                    raise RuntimeError("Inference warmup was stopped")
            except Exception:
                self._warmup_status = {
                    "state": "failed", "duration_seconds": time.monotonic() - started,
                }
                raise
            self._warmup_status = {
                "state": "ready", "duration_seconds": time.monotonic() - started,
            }
            return self.identity()

    def identity(self) -> dict[str, Any]:
        status = self._warmup_status
        return {
            "embedding_model": REPOSITORY_EMBEDDING_MODEL,
            "embedding_model_revision": embedding_model_revision(),
            "embedding_max_concurrency": _positive_int(
                "EMBEDDING_MAX_CONCURRENCY", 1, maximum=8
            ),
            "embedding_runtime_ready": status["state"] == "ready",
            "embedding_warmup_state": status["state"],
            "embedding_warmup_duration_seconds": status["duration_seconds"],
            "tower_embedding_cache": self.tower_embedding_cache.snapshot(),
            "embedding_cpu_threads": _positive_int(
                "EMBEDDING_CPU_THREADS", 1, maximum=64
            ),
            "embedding_max_outstanding_jobs": embedding_max_outstanding_jobs(),
        }


def _single_model_cache(factory):
    # lru_cache alone may run a cold factory concurrently. Serialize the cache
    # lookup as well as construction so first search jobs share one query cache.
    cached = lru_cache(maxsize=1)(factory)
    lock = threading.Lock()
    current = [None]

    @wraps(factory)
    def resolve():
        with lock:
            model = cached()
            current[0] = model
            return model

    def clear():
        with lock:
            cached.cache_clear()
            current[0] = None

    resolve.cache_info = cached.cache_info
    resolve.cache_clear = clear
    resolve.cache_parameters = cached.cache_parameters
    # Reading one published reference never waits behind model construction and
    # cannot turn an inspection/clear race into a model load on a probe thread.
    resolve.peek = lambda: current[0]
    return resolve


@_single_model_cache
def shared_embedding_model() -> BoundedEmbeddingModel:
    concurrency = _positive_int("EMBEDDING_MAX_CONCURRENCY", 1, maximum=8)
    return BoundedEmbeddingModel(
        _load_sentence_transformer(), max_concurrency=concurrency
    )


@lru_cache(maxsize=1)
def repository_embedding_pipeline() -> RepositoryEmbeddingPipeline:
    model = shared_embedding_model()
    return RepositoryEmbeddingPipeline(
        embedder=SentenceTransformerEmbedder(
            REPOSITORY_EMBEDDING_MODEL,
            model=model,
        ),
        card_summarizer=card_summary_pipeline(),
        tower_cache=model.tower_embedding_cache,
    )


@lru_cache(maxsize=1)
def user_onboarding_pipeline() -> UserOnboardingPipeline:
    return UserOnboardingPipeline(model=shared_embedding_model())


@lru_cache(maxsize=1)
def embedding_executor() -> ThreadPoolExecutor:
    with runtime_admission():
        workers = _positive_int("EMBEDDING_EXECUTOR_WORKERS", 1, maximum=8)
        return ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="ml-embedding",
        )


@lru_cache(maxsize=1)
def onboarding_executor() -> ThreadPoolExecutor:
    # Separate pipeline execution, not separate model capacity. Each queue is
    # bounded by the same process-wide admission semaphore below.
    with runtime_admission():
        workers = _positive_int("EMBEDDING_EXECUTOR_WORKERS", 1, maximum=8)
        return ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="ml-onboarding",
        )


@lru_cache(maxsize=1)
def embedding_admission() -> threading.BoundedSemaphore:
    return threading.BoundedSemaphore(embedding_max_outstanding_jobs())


async def run_embedding_job(function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    return await _run_embedding_job(embedding_executor, function, *args, **kwargs)


async def run_onboarding_job(function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run admitted onboarding independently of repository provider/storage waits.

    Both classes retain the existing total admission ceiling. A saturated global
    ceiling still rejects new work rather than reserving or overcommitting slots.
    """
    return await _run_embedding_job(onboarding_executor, function, *args, **kwargs)


async def _run_embedding_job(
    executor_factory: Callable[[], ThreadPoolExecutor],
    function: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    with runtime_admission():
        admission = embedding_admission()
        if not admission.acquire(blocking=False):
            raise EmbeddingCapacityError(
                "the bounded embedding executor is at capacity"
            )
        operation = "onboard" if executor_factory is onboarding_executor else "embedding"
        observation = WorkObservation(operation)
        try:
            context = copy_context()
            def invoke() -> T:
                observation.start()
                try:
                    with span_scope(operation):
                        return function(*args, **kwargs)
                finally:
                    observation.worker_stopped()
            future = executor_factory().submit(context.run, invoke)
        except BaseException:
            observation.finish("error")
            admission.release()
            raise
    # A cancelled HTTP request does not stop a function that is already running
    # in a ThreadPoolExecutor.  Tie admission to the worker future rather than
    # the awaiting coroutine so cancellation cannot free a slot prematurely and
    # allow the real number of outstanding embedding jobs to exceed the bound.
    def completed(_future) -> None:
        try:
            observation.complete_future(_future)
        finally:
            admission.release()
    future.add_done_callback(completed)
    try:
        return await asyncio.wait_for(
            asyncio.wrap_future(future),
            timeout=embedding_timeout_seconds(),
        )
    except asyncio.CancelledError:
        observation.abandon("cancelled")
        raise
    except asyncio.TimeoutError as exc:
        if future.done() and not future.cancelled():
            return future.result()
        observation.abandon("timeout")
        raise EmbeddingDeadlineExceeded("the bounded embedding deadline was exceeded") from exc


def warm_embedding_runtime(stop: threading.Event | None = None) -> dict[str, Any]:
    """Validate pipelines and execute real inference using local artifacts only."""
    previous = getattr(_warmup_loading, "local_files_only", False)
    _warmup_loading.local_files_only = True
    try:
        if stop is not None and stop.is_set():
            raise RuntimeError("Inference warmup was stopped")
        model = shared_embedding_model()
        if stop is not None and stop.is_set():
            raise RuntimeError("Inference warmup was stopped")
        # Construction validates contracts without opening Redis/Qdrant or
        # calling summary providers. No user or corpus data enters warmup.
        pipeline = repository_embedding_pipeline()
        user_onboarding_pipeline()
        if pipeline.config.embedding_dim != REPOSITORY_EMBEDDING_DIM:
            raise ValueError("Repository pipeline warmup dimensions are incompatible")
        return model.warmup(stop=stop)
    finally:
        _warmup_loading.local_files_only = previous


def embedding_runtime_status() -> dict[str, Any]:
    """Return model identity without loading a model on a health-check thread."""
    model = shared_embedding_model.peek()
    if model is not None:
        return model.identity()
    return {
        "embedding_model": REPOSITORY_EMBEDDING_MODEL,
        "embedding_model_revision": embedding_model_revision(),
        "embedding_max_concurrency": _positive_int(
            "EMBEDDING_MAX_CONCURRENCY", 1, maximum=8
        ),
        "embedding_runtime_ready": False,
        "embedding_warmup_state": "not_started",
        "embedding_warmup_duration_seconds": None,
        "embedding_cpu_threads": _positive_int(
            "EMBEDDING_CPU_THREADS", 1, maximum=64
        ),
        "embedding_max_outstanding_jobs": embedding_max_outstanding_jobs(),
    }


def embedding_runtime_drain_actions():
    """Retain real executor completion before providers/model caches can close."""
    actions = []
    for name, factory in (("embedding", embedding_executor), ("onboarding", onboarding_executor)):
        if factory.cache_info().currsize:
            executor = factory()
            actions.append((
                f"{name}.executor.drain",
                lambda executor=executor: executor.shutdown(wait=True, cancel_futures=True),
            ))
    return actions


def embedding_runtime_cleanup_actions():
    """Capture each executor/provider before independently clearing caches."""
    actions = []
    for name, factory in (("embedding", embedding_executor), ("onboarding", onboarding_executor)):
        if factory.cache_info().currsize:
            executor = factory()
            actions.append((
                f"{name}.executor.close",
                lambda executor=executor: executor.shutdown(wait=False, cancel_futures=True),
            ))
        actions.append((f"{name}.executor.cache.clear", factory.cache_clear))
    return [
        *actions,
        ("embedding.admission.cache.clear", embedding_admission.cache_clear),
        ("embedding.repository_pipeline.cache.clear", repository_embedding_pipeline.cache_clear),
        ("embedding.user_pipeline.cache.clear", user_onboarding_pipeline.cache_clear),
        ("embedding.model.cache.clear", shared_embedding_model.cache_clear),
        *card_summary_cleanup_actions(),
    ]


def shutdown_embedding_runtime() -> None:
    if embedding_executor.cache_info().currsize:
        embedding_executor().shutdown(wait=False, cancel_futures=True)
        embedding_executor.cache_clear()
    if onboarding_executor.cache_info().currsize:
        onboarding_executor().shutdown(wait=False, cancel_futures=True)
        onboarding_executor.cache_clear()
    embedding_admission.cache_clear()
    repository_embedding_pipeline.cache_clear()
    user_onboarding_pipeline.cache_clear()
    shared_embedding_model.cache_clear()
    shutdown_card_summary_runtime()


def reset_embedding_runtime_for_tests() -> None:
    """Clear process caches; intended for deterministic tests only."""
    shutdown_embedding_runtime()


atexit.register(shutdown_embedding_runtime)
