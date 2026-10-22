"""Single-flight dependency snapshots, never refreshed by routine probes."""

from __future__ import annotations

import asyncio
import copy
import math
import logging
import os
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from api.runtime import record_service_job_timeout, submit_service_job
from utils.observability import WorkObservation


logger = logging.getLogger("pipeline.api.health")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class HealthSettings:
    refresh_seconds: float
    max_age_seconds: float
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "HealthSettings":
        def number(
            name: str, default: float, minimum: float, maximum: float,
            *, blank_uses_default: bool = False,
        ) -> float:
            raw = os.getenv(name, str(default)).strip()
            if blank_uses_default and not raw:
                raw = str(default)
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be a number") from exc
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
            return value

        refresh = number("V2_HEALTH_REFRESH_SECONDS", 15, 1, 300, blank_uses_default=True)
        max_age = number("V2_HEALTH_MAX_AGE_SECONDS", 45, 1, 600, blank_uses_default=True)
        timeout = number("V2_HEALTH_TIMEOUT_SECONDS", 5, 0.1, 30)
        if max_age < refresh + 2 * timeout:
            raise ValueError(
                "V2_HEALTH_MAX_AGE_SECONDS must cover refresh plus two diagnostic deadlines"
            )
        return cls(refresh, max_age, timeout)


class HealthDiagnostics:
    """Own one actual worker future, even after HTTP timeout/disconnection.

    The reserved health executor still provides a second admission bound. The
    cadence applies to explicit requests too, including failures. A stuck worker
    cannot accumulate replacements, and late success cannot erase its timeout.
    """

    def __init__(
        self,
        collect: Callable[[threading.Event], tuple[bool, dict[str, Any]]],
        settings: HealthSettings,
    ) -> None:
        self.collect = collect
        self.settings = settings
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._retirement_state = "active"
        self._retirement_failures: tuple[str, ...] = ()
        self._retirement_cleanup: Callable[[], tuple[str, ...]] | None = None
        self._future: Future | None = None
        self._timed_out_future: Future | None = None
        self._started = 0.0
        self._next_attempt = 0.0
        self._observed: float | None = None
        self._checked_at: str | None = None
        self._completed_at: str | None = None
        self._attempt_started_at: str | None = None
        self._healthy = False
        self._failure: str | None = "diagnostics_not_checked"
        self._payload: dict[str, Any] = {}
        self._runs = 0

    def _start(self) -> Future | None:
        with self._lock:
            if self._stop.is_set():
                return None
            if self._future is not None and not self._future.done():
                return self._future
            now = time.monotonic()
            if now < self._next_attempt:
                return None
            self._started = now
            started_at = _timestamp()
            self._attempt_started_at = started_at
            self._next_attempt = now + self.settings.refresh_seconds
            try:
                # One observation shares the coordinator's expiry across all
                # waiters. Native worker completion alone records late work;
                # this does not cancel the Future or release its admission.
                observation = WorkObservation(
                    "health", shared_deadline=now + self.settings.timeout_seconds,
                )
                future = submit_service_job(
                    "health", self.collect, self._stop, observation=observation,
                )
            except Exception:
                # Admission/configuration failure invalidates prior success.
                self._healthy = False
                self._failure = "diagnostic_admission_failed"
                return None
            self._runs += 1
            self._future = future
        # A completed future invokes callbacks synchronously. Register outside
        # the lock so very fast diagnostics cannot deadlock readiness.
        future.add_done_callback(
            lambda completed: self._complete(completed, now, started_at)
        )
        return future

    def _complete(self, future: Future, started: float, started_at: str) -> None:
        try:
            healthy, payload = future.result()
            failure = None if healthy else "dependency_diagnostics_failed"
        except Exception:
            healthy, payload = False, {}
            failure = "dependency_diagnostics_failed"
        completed = time.monotonic()
        if completed - started >= self.settings.timeout_seconds:
            healthy = False
            failure = "diagnostic_deadline_exceeded"
        with self._lock:
            # A worker can finish before its callback runs. Never let its late
            # callback overwrite the newer refresh's result.
            if self._future is not future:
                return
            self._observed = started
            self._checked_at = started_at
            self._completed_at = _timestamp()
            if failure == "diagnostic_deadline_exceeded":
                self._record_timeout(future)
            self._healthy = healthy
            self._failure = failure
            self._payload = payload

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            in_flight = self._future is not None and not self._future.done()
            if in_flight and now - self._started >= self.settings.timeout_seconds:
                self._healthy = False
                self._failure = "diagnostic_deadline_exceeded"
                self._record_timeout(self._future)
            if self._stop.is_set():
                self._healthy = False
                self._failure = "diagnostics_stopping"
            age = None if self._observed is None else max(0.0, now - self._observed)
            fresh = age is not None and age <= self.settings.max_age_seconds
            healthy = fresh and self._healthy
            return {
                **copy.deepcopy(self._payload),
                "status": "healthy" if healthy else "unhealthy",
                "diagnostics_fresh": fresh,
                "diagnostics_checked_at": self._checked_at,
                "diagnostics_completed_at": self._completed_at,
                "diagnostics_age_seconds": age,
                "diagnostics_max_age_seconds": self.settings.max_age_seconds,
                "diagnostics_refresh_seconds": self.settings.refresh_seconds,
                "diagnostics_in_flight": in_flight,
                "diagnostics_attempt_started_at": self._attempt_started_at,
                "diagnostics_runs_total": self._runs,
                "diagnostics_retirement_state": self._retirement_state,
                "diagnostics_cleanup_failures": list(self._retirement_failures),
                "diagnostics_failure": self._failure or (None if fresh else "diagnostics_stale"),
            }

    def _record_timeout(self, future: Future) -> None:
        # Called under the snapshot lock; count once per actual diagnostic job,
        # not once per timed-out HTTP waiter/readiness probe.
        if not self._stop.is_set() and self._timed_out_future is not future:
            record_service_job_timeout("health")
            self._timed_out_future = future

    def close(self) -> None:
        self._stop.set()

    def retire(self, on_drained: Callable[[], tuple[str, ...]]) -> None:
        """Stop new work and retry failed cleanup only after the real job exits."""
        with self._lock:
            self._stop.set()
            if self._retirement_state in {"scheduled", "running", "completed"}:
                return
            if self._retirement_cleanup is None:
                self._retirement_cleanup = on_drained
            self._retirement_state = "scheduled"
            future = self._future
        if future is None or future.done():
            self._run_retirement_cleanup()
        else:
            future.add_done_callback(lambda _future: self._run_retirement_cleanup())

    def _run_retirement_cleanup(self) -> None:
        with self._lock:
            self._retirement_state = "running"
            cleanup = self._retirement_cleanup
        try:
            failures = cleanup() if cleanup is not None else ("cleanup_not_configured",)
        except Exception:
            # Do not let Future callback logging expose exception messages.
            logger.error("Diagnostic retirement cleanup callback failed")
            failures = ("cleanup_callback_failed",)
        with self._lock:
            self._retirement_failures = failures
            self._retirement_state = "failed" if failures else "completed"

    async def refresh(self) -> dict[str, Any]:
        future = self._start()
        if future is not None:
            with self._lock:
                remaining = self.settings.timeout_seconds - (time.monotonic() - self._started)
            if remaining > 0:
                try:
                    # Shielding retains the actual worker/future when an HTTP
                    # waiter times out or disconnects; other waiters share it.
                    wrapped = asyncio.wrap_future(future)
                    wrapped.add_done_callback(
                        lambda done: None if done.cancelled() else done.exception()
                    )
                    await asyncio.wait_for(asyncio.shield(wrapped), timeout=remaining)
                except Exception:
                    # All failure details come from the sanitized snapshot.
                    pass
        return self.snapshot()

    async def run_periodic(self) -> None:
        while True:
            await self.refresh()
            await asyncio.sleep(self.settings.refresh_seconds)
