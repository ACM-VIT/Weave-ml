"""Admission gate controlled only by V2's existing runtime cleanup plan.

The gate covers brief submission/construction critical sections, never worker
execution or joins. Already-admitted durable work keeps its existing lifetime.
"""

from contextlib import contextmanager
import threading
from typing import Iterator


class RuntimeRetiringError(RuntimeError):
    """New work cannot reuse a runtime whose cleanup is incomplete."""


_lock = threading.RLock()
_retiring = threading.Event()


@contextmanager
def runtime_admission() -> Iterator[None]:
    with _lock:
        check_runtime_admission()
        yield


def check_runtime_admission() -> None:
    """Nonblocking observation for bounded queue waits; submission still locks."""
    if _retiring.is_set():
        raise RuntimeRetiringError("V2 runtime is retiring")


def stop_runtime_admission() -> None:
    with _lock:
        _retiring.set()


def resume_runtime_admission() -> None:
    """Called by startup only after the cleanup plan reports completion."""
    with _lock:
        _retiring.clear()
