"""Durable metadata and exclusion for an explicitly deployed singleton worker.

All scheduler processes must use the same persistent local filesystem/path.
Independent hosts/filesystems require backend lease coordination instead.
No repository payload or README is stored here (issue #84).
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
from typing import Callable
import uuid

from acquisition.checkpoint_policy import parse_time

logger = logging.getLogger(__name__)


class SingletonTrendingCoordinator:
    def __init__(self, path: str | Path, *, clock: Callable[[], datetime] | None = None):
        self.path = Path(path).resolve()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._owner: int | None = None
        self._guard = threading.Lock()

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_bytes())
        except FileNotFoundError:
            return {"version": 1, "last_success_observed_at": None}
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("Invalid trending singleton state") from exc
        if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
            raise ValueError("Unsupported trending singleton state")
        observed = data.get("last_success_observed_at")
        if parse_time(observed) is None or parse_time(data.get("published_at")) is None:
            raise ValueError("Invalid trending freshness metadata")
        uuid.UUID(str(data.get("snapshot_id")))
        return data

    def last_refresh(self) -> datetime | None:
        return parse_time(self._load().get("last_success_observed_at"))

    @contextmanager
    def refresh_session(self, *, force: bool, refresh_hours: int):
        if not self._guard.acquire(blocking=False):
            yield False
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.with_name(self.path.name + ".lock").open("a+b") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    logger.info("Skipping trending refresh: singleton already owns acquisition")
                    yield False
                    return
                try:
                    last = self.last_refresh()
                    age = (self._clock() - last).total_seconds() if last else None
                    if age is not None and age < 0:
                        raise ValueError("Trending freshness is in the future; check worker clocks")
                    if age is not None and age >= 48 * 3600:
                        logger.warning("Trending freshness target missed: age_seconds=%.0f", age)
                    if not force and age is not None and age < refresh_hours * 3600:
                        logger.info("Skipping trending refresh: durable snapshot is still fresh")
                        yield False
                        return
                    if force:
                        logger.warning("Operator forced trending refresh under singleton ownership")
                    self._owner = threading.get_ident()
                    try:
                        yield True
                    finally:
                        self._owner = None
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            self._guard.release()

    def mark_success(self, *, snapshot_id: str, observed_at: datetime):
        if self._owner != threading.get_ident():
            raise RuntimeError("Trending success requires the active singleton owner")
        uuid.UUID(snapshot_id)
        if observed_at.tzinfo is None or observed_at > self._clock():
            raise ValueError("Invalid trending observation time")
        last = self.last_refresh()
        if last is not None and observed_at < last:
            raise ValueError("Refusing to regress trending freshness")
        data = {
            "version": 1, "snapshot_id": snapshot_id,
            "last_success_observed_at": observed_at.astimezone(timezone.utc).isoformat(),
            "published_at": self._clock().isoformat(),
        }
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(data, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
