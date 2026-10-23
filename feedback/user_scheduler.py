"""Bounded round-robin user lanes; Redis remains the durable backlog."""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Work:
    message_id: str
    user_id: str
    payload: Any
    snapshot: Any = None


class UserFeedbackScheduler:
    """Capacity counts queued AND executing work, until synchronous apply exits.

    No executor queue or per-user threads. A failed lane is released back to the
    Redis pending list, never acknowledged here or retried ahead of earlier work.
    """

    def __init__(self, process: Callable[[Work], bool], *, workers: int, capacity: int):
        if not 1 <= workers <= 16 or not workers <= capacity <= 1000:
            raise ValueError(
                "feedback scheduler requires 1-16 workers and workers<=capacity<=1000"
            )
        self.process, self.capacity = process, capacity
        self._condition = threading.Condition()
        self._queues: dict[str, deque[Work]] = {}
        self._ready: deque[str] = deque()
        self._active: set[str] = set()
        self._ids: set[str] = set()
        self._closed = False
        self._completed = self._reported = 0
        self.high_water = self.deferred = self.errors = 0
        self._threads = [
            threading.Thread(
                target=self._worker, name=f"feedback-user-{i}", daemon=True
            )
            for i in range(workers)
        ]
        for thread in self._threads:
            thread.start()

    @property
    def pending(self) -> int:
        with self._condition:
            return len(self._ids)

    def submit(self, work: Work) -> bool:
        with self._condition:
            if (
                self._closed
                or work.message_id in self._ids
                or len(self._ids) >= self.capacity
            ):
                return False
            queue = self._queues.get(work.user_id)
            if len(queue or ()) + int(work.user_id in self._active) >= 2:
                return False
            queue = self._queues.setdefault(work.user_id, deque())
            queue.append(work)
            self._ids.add(work.message_id)
            self.high_water = max(self.high_water, len(self._ids))
            if len(queue) == 1 and work.user_id not in self._active:
                self._ready.append(work.user_id)
            self._condition.notify_all()
            return True

    def wait_capacity(self, timeout: float = 0.1) -> bool:
        with self._condition:
            self._condition.wait_for(
                lambda: self._closed or len(self._ids) < self.capacity, timeout
            )
            return not self._closed and len(self._ids) < self.capacity

    def wait_user(self, user_id: str) -> None:
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._closed
                    or (
                        len(self._queues.get(user_id, ()))
                        + int(user_id in self._active)
                        < 2
                        and len(self._ids) < self.capacity
                    )
                )
            )

    def completed(self) -> int:
        with self._condition:
            value = self._completed - self._reported
            self._reported = self._completed
            return value

    def drain(self) -> None:
        with self._condition:
            self._condition.wait_for(lambda: not self._ids)

    def close(self) -> None:
        # Stop admission, drop only unstarted work to the durable pending list,
        # and retain running slots/clients/heartbeats until apply actually exits.
        with self._condition:
            self._closed = True
            for queue in self._queues.values():
                for work in queue:
                    self._ids.discard(work.message_id)
            self._queues.clear()
            self._ready.clear()
            self._condition.notify_all()
        for thread in self._threads:
            thread.join()

    def _worker(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._ready or self._closed)
                if not self._ready:
                    return
                user_id = self._ready.popleft()
                work = self._queues[user_id].popleft()
                self._active.add(user_id)
            try:
                finalized = bool(self.process(work))
            except Exception:  # noqa: BLE001 - isolate worker failures; Redis retains unacknowledged work
                # No raw payload/error in logs. Unacknowledged work remains in PEL.
                logger.error("feedback lane failed; retaining pending deliveries")
                finalized = False
                with self._condition:
                    self.errors += 1
            with self._condition:
                self._ids.discard(work.message_id)
                self._active.remove(user_id)
                if finalized:
                    self._completed += 1
                else:
                    self.deferred += 1
                    for queued in self._queues.pop(user_id, ()):
                        self._ids.discard(queued.message_id)
                queue = self._queues.get(user_id)
                if queue:
                    self._ready.append(
                        user_id
                    )  # One event/turn prevents hot-user monopoly.
                else:
                    self._queues.pop(user_id, None)
                self._condition.notify_all()
