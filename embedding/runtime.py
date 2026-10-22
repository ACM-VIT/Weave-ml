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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps
from typing import Any, Callable, TypeVar

from config import EMBEDDING_MODEL_REVISION, REPOSITORY_EMBEDDING_MODEL
from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
from embedding.embeddings import SentenceTransformerEmbedder
from embedding.query_cache import QueryEmbeddingCache
from scripts.user_onboarding import UserOnboardingPipeline
from summarization.runtime import (
    card_summary_cleanup_actions, card_summary_pipeline, shutdown_card_summary_runtime,
)
from utils.deadline import check_deadline, remaining_seconds
from utils.runtime_lifecycle import runtime_admission


T = TypeVar("T")
DEFAULT_EMBEDDING_MODEL_REVISION = EMBEDDING_MODEL_REVISION


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
    if _production():
        kwargs["local_files_only"] = True
    return SentenceTransformer(REPOSITORY_EMBEDDING_MODEL, **kwargs)


class BoundedEmbeddingModel:
    """FIFO admission into one shared, concurrency-bounded encoder."""

    def __init__(self, model: Any, *, max_concurrency: int) -> None:
        self._model = model
        # Cache lifetime follows this exact loaded encoder, not a reusable ID.
        self.query_embedding_cache = QueryEmbeddingCache()
        self.query_embedding_revision = embedding_model_revision()
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self._max_concurrency = max_concurrency
        self._condition = threading.Condition()
        self._waiting: deque[object] = deque()
        self._active = 0

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

    def identity(self) -> dict[str, Any]:
        return {
            "embedding_model": REPOSITORY_EMBEDDING_MODEL,
            "embedding_model_revision": embedding_model_revision(),
            "embedding_max_concurrency": _positive_int(
                "EMBEDDING_MAX_CONCURRENCY", 1, maximum=8
            ),
            "embedding_runtime_ready": True,
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

    @wraps(factory)
    def resolve():
        with lock:
            return cached()

    # Preserve the existing runtime status/reset and introspection contract.
    resolve.cache_info = cached.cache_info
    resolve.cache_clear = cached.cache_clear
    resolve.cache_parameters = cached.cache_parameters
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
        try:
            future = executor_factory().submit(function, *args, **kwargs)
        except BaseException:
            admission.release()
            raise
    # A cancelled HTTP request does not stop a function that is already running
    # in a ThreadPoolExecutor.  Tie admission to the worker future rather than
    # the awaiting coroutine so cancellation cannot free a slot prematurely and
    # allow the real number of outstanding embedding jobs to exceed the bound.
    future.add_done_callback(lambda _future: admission.release())
    try:
        return await asyncio.wait_for(
            asyncio.wrap_future(future),
            timeout=embedding_timeout_seconds(),
        )
    except asyncio.TimeoutError as exc:
        if future.done() and not future.cancelled():
            return future.result()
        raise EmbeddingDeadlineExceeded("the bounded embedding deadline was exceeded") from exc


def warm_embedding_runtime() -> dict[str, Any]:
    """Load and validate both frozen online embedding pipelines network-free."""
    model = shared_embedding_model()
    # Construction validates model aliases, dimensions, chunk settings, and the
    # shared user/repository vector contract without opening Redis or Qdrant.
    repository_embedding_pipeline()
    user_onboarding_pipeline()
    return model.identity()


def embedding_runtime_status() -> dict[str, Any]:
    """Return model identity without loading a model on a health-check thread."""
    if shared_embedding_model.cache_info().currsize:
        return shared_embedding_model().identity()
    return {
        "embedding_model": REPOSITORY_EMBEDDING_MODEL,
        "embedding_model_revision": embedding_model_revision(),
        "embedding_max_concurrency": _positive_int(
            "EMBEDDING_MAX_CONCURRENCY", 1, maximum=8
        ),
        "embedding_runtime_ready": False,
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
