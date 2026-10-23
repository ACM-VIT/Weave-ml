"""Process-local GitHub quota admission without retaining credential material.

This is deliberately not a distributed quota service. Independent processes,
other GitHub consumers and tokens sharing one account can still exhaust quota.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import hmac
import math
import os
import random
import threading
import time
import weakref
from typing import Any

COORDINATOR_CAPACITY = 256
COORDINATOR_IDLE_SECONDS = 300.0


class GitHubRetryLater(Exception):
    """Resume this same logical operation after a monotonic admission time."""

    def __init__(self, not_before: float) -> None:
        super().__init__("GitHub request deferred")
        self.not_before = not_before


@dataclass
class GitHubRequestBudget:
    """Survives scheduler deferrals; replay never resets request attempts.

    Completed queries are reused only inside this one logical operation. This
    also avoids repeating a successful first stage after a later query retries.
    """

    deadline: float | None = None
    defer_retries: bool = True
    authorization_scope: str | None = None
    cancelled: Callable[[], bool] = lambda: False
    attempts: dict[str, int] = field(default_factory=dict)
    not_before: dict[str, float] = field(default_factory=dict)
    responses: dict[str, dict[str, Any] | None] = field(default_factory=dict)

    shared_flights: dict[Hashable, Any] = field(default_factory=dict)
    cleanups: dict[Hashable, Callable[[], None]] = field(default_factory=dict)

    def finish(self) -> None:
        callbacks = tuple(self.cleanups.values())
        self.cleanups.clear()
        for callback in callbacks:
            callback()
        self.shared_flights.clear()
        self.responses.clear()

    def check(self) -> None:
        if self.cancelled():
            raise CancelledError("GitHub acquisition cancelled")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("GitHub acquisition deadline exceeded")

    def check_wait(self, not_before: float) -> None:
        self.check()
        if self.deadline is not None and not_before >= self.deadline:
            raise TimeoutError("GitHub retry would exceed acquisition deadline")


class GitHubRateLimitCoordinator:
    """Coordinate known remaining quota, reset windows and staggered resumes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._scope_key: str | None = None
        self._last_used = time.monotonic()
        self._remaining: int | None = None
        self._reset_at = 0.0
        self._blocked_until = 0.0
        self._next_admission = 0.0
        self._stagger_until = 0.0
        self._cost = 1
        self._counters: Counter[str] = Counter()

    def admit(self) -> float | None:
        """Reserve known quota or return a delay; never wait in a worker."""
        now = time.monotonic()
        with _registry_lock, self._lock:
            self._retain_locked(now)
            if self._reset_at and now >= self._reset_at:
                self._remaining = None
                self._reset_at = 0.0
            if self._remaining is not None and self._remaining < self._cost:
                # Infer one fixed window when a positive header-only quota is
                # consumed locally. Extending now + 60 on every wake would
                # postpone admission forever without clearing remaining.
                if not self._reset_at:
                    self._reset_at = now + 60.0
                    self._block_locked(self._reset_at)
                self._blocked_until = max(self._blocked_until, self._reset_at)
            ready_at = max(self._blocked_until, self._next_admission)
            if now < ready_at:
                self._counters["admission_deferred"] += 1
                return ready_at
            if self._remaining is not None:
                self._remaining = max(0, self._remaining - self._cost)
            if now < self._stagger_until:
                self._next_admission = now + random.uniform(0.05, 0.25)
            self._counters["requests"] += 1
            return None

    def observe(self, headers: Any, rate_limit: Any = None) -> None:
        """Use conservative updates so out-of-order replies cannot refill quota."""
        now, wall_now = time.monotonic(), time.time()
        remaining = _nonnegative_int(headers.get("X-RateLimit-Remaining"))
        reset_wall = _number(headers.get("X-RateLimit-Reset"))
        cost = None
        if isinstance(rate_limit, dict):
            remaining = _nonnegative_int(rate_limit.get("remaining")) if remaining is None else remaining
            cost = _nonnegative_int(rate_limit.get("cost"))
            if reset_wall is None:
                reset_wall = _timestamp(rate_limit.get("resetAt"))
        reset_at = now + max(0.0, reset_wall - wall_now) if reset_wall is not None else 0.0
        with _registry_lock, self._lock:
            self._retain_locked(now)
            if cost is not None:
                self._cost = max(1, cost)
            if remaining is not None:
                # A response belonging to an older window must not poison a
                # newer one, even if network completion order is reversed.
                if reset_at and reset_at + 0.5 < self._reset_at:
                    return
                if reset_at > self._reset_at + 0.5:
                    self._remaining = remaining
                else:
                    self._remaining = remaining if self._remaining is None else min(self._remaining, remaining)
                self._reset_at = max(self._reset_at, reset_at)
                if remaining < self._cost:
                    if not self._reset_at:
                        self._reset_at = now + 60.0
                    self._block_locked(self._reset_at)

    def throttle(self, headers: Any) -> float:
        now, wall_now = time.monotonic(), time.time()
        delays = []
        reset = _number(headers.get("X-RateLimit-Reset"))
        # A secondary limit may leave primary quota available. Its reset is
        # not a retry instruction unless that primary quota is exhausted.
        remaining = _nonnegative_int(headers.get("X-RateLimit-Remaining"))
        if remaining == 0 and reset is not None:
            delays.append(max(0.0, reset - wall_now))
        retry_after = headers.get("Retry-After")
        seconds = _number(retry_after)
        if seconds is not None:
            delays.append(max(0.0, seconds))
        elif isinstance(retry_after, str):
            try:
                delays.append(max(0.0, parsedate_to_datetime(retry_after).timestamp() - wall_now))
            except (TypeError, ValueError, OverflowError):
                pass
        # Honor the whole server delay, not a capped sleep followed by early IO.
        ready_at = now + max(delays or [60.0])
        with _registry_lock, self._lock:
            self._retain_locked(now)
            self._counters["throttled"] += 1
            self._block_locked(ready_at)
            return self._blocked_until

    def _block_locked(self, ready_at: float) -> None:
        # First admission after reset is jittered too, not just its followers.
        self._blocked_until = max(self._blocked_until, ready_at + random.uniform(0.05, 0.25))
        self._stagger_until = max(self._stagger_until, self._blocked_until + 5.0)

    def _retain_locked(self, now: float) -> None:
        # Registry pruning and quota updates acquire registry -> coordinator.
        # Re-pin before changing a cooldown, including an idle live instance
        # whose old retention horizon was already pruned.
        self._last_used = now
        if self._scope_key is not None:
            _retained_scopes[self._scope_key] = self

    @property
    def retention_deadline(self) -> float:
        with self._lock:
            return max(
                self._last_used + COORDINATOR_IDLE_SECONDS, self._reset_at,
                self._blocked_until, self._next_admission, self._stagger_until,
            )

    def count(self, name: str) -> None:
        with self._lock:
            self._counters[name] += 1

    def snapshot(self) -> dict[str, int | None]:
        with self._lock:
            return {**self._counters, "remaining": self._remaining}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _nonnegative_int(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None


def _timestamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc).timestamp() if parsed.tzinfo is None else parsed.timestamp()
    except (ValueError, OverflowError):
        return None


_registry_lock = threading.RLock()
_registry: weakref.WeakValueDictionary[str, GitHubRateLimitCoordinator] = weakref.WeakValueDictionary()
_retained_scopes: dict[str, GitHubRateLimitCoordinator] = {}
_process_key = os.urandom(32)


def github_coordination_key(endpoint: str, credential: str | None) -> str:
    """Opaque per-process HMAC, never the token or an unhashed authorization header."""
    return hmac.new(_process_key, f"{endpoint}\0{credential or ''}".encode(), hashlib.sha256).hexdigest()


def shared_github_coordinator(key: str) -> GitHubRateLimitCoordinator:
    with _registry_lock:
        now = time.monotonic()
        # Only expired retention pins may leave. The weak live-instance index
        # still preserves any caller-owned coordinator, so it cannot fork into
        # two independent quota states while an older client remains alive.
        for scope in tuple(_retained_scopes):
            if _retained_scopes[scope].retention_deadline <= now:
                _retained_scopes.pop(scope)
        coordinator = _registry.get(key)
        if coordinator is None:
            if len(_registry) >= COORDINATOR_CAPACITY:
                # Never forget a live/cooling scope to admit a new credential.
                raise RuntimeError("GitHub quota coordinator capacity reached")
            coordinator = GitHubRateLimitCoordinator()
            coordinator._scope_key = key
            _registry[key] = coordinator
        _retained_scopes[key] = coordinator
        return coordinator
