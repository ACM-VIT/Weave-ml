"""Atomic, non-sensitive checkpoint state for resumable corpus ingestion."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import random
from typing import Any

from .identity import repository_identity_key, normalize_repository_name
from .checkpoint_policy import CheckpointPolicy, parse_time


class CorpusCheckpoint:
    VERSION = 2

    def __init__(
        self, path: str | Path, *, policy: CheckpointPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.path = Path(path).resolve()
        self.policy = policy or CheckpointPolicy()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._jitter = jitter
        self._guard = threading.Lock()
        self._session_thread: int | None = None
        self._loaded_bytes: bytes | None = None
        self.data = self._load()

    def _empty(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "pending_persistence": [],
            "pending_index": [],
            # V1 did not identify who owned these entries. Never silently
            # discard them or route them to a different indexing owner.
            "legacy_pending_index": [],
            "failures": {},
            "rejected": {},
            "last_run": {},
            "updated_at": None,
        }

    def _load(self) -> dict[str, Any]:
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            self._loaded_bytes = None
            return self._empty()
        try:
            loaded = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid acquisition checkpoint {self.path}: {exc}"
            ) from exc
        if (
            not isinstance(loaded, dict)
            or type(loaded.get("version")) is not int
            or loaded["version"] not in (1, self.VERSION)
        ):
            raise ValueError(
                f"Unsupported acquisition checkpoint format in {self.path}"
            )
        defaults = self._empty()
        defaults.update(loaded)
        self._validate(defaults)
        if loaded["version"] == 1:
            defaults["legacy_pending_index"] = self._normalized_unique([
                *defaults["legacy_pending_index"], *defaults["pending_index"],
            ])
            defaults["pending_index"] = []
            defaults["version"] = self.VERSION
        for field in ("pending_persistence", "pending_index", "legacy_pending_index"):
            defaults[field] = self._normalized_unique(defaults[field])
        self._loaded_bytes = raw
        return defaults

    @staticmethod
    def _validate(data: dict[str, Any]) -> None:
        for field in ("pending_persistence", "pending_index", "legacy_pending_index"):
            values = data.get(field)
            if not isinstance(values, list) or any(
                not normalize_repository_name(name) for name in values
            ):
                raise ValueError(f"Invalid acquisition checkpoint field: {field}")
        for field in ("failures", "rejected", "last_run"):
            if not isinstance(data.get(field), dict):
                raise ValueError(f"Invalid acquisition checkpoint field: {field}")
        for field in ("failures", "rejected"):
            for name, record in data[field].items():
                if not normalize_repository_name(name) or not isinstance(record, dict):
                    raise ValueError(f"Invalid acquisition checkpoint field: {field}")
                if "attempts" in record and (type(record["attempts"]) is not int or record["attempts"] < 0):
                    raise ValueError("Invalid acquisition checkpoint attempt count")
                for timestamp in ("timestamp", "first_failed_at", "next_attempt_at"):
                    if timestamp in record and parse_time(record[timestamp]) is None:
                        raise ValueError("Invalid acquisition checkpoint retry timestamp")
                if field == "rejected" and "reasons" in record:
                    if not isinstance(record["reasons"], list) or any(not isinstance(reason, str) for reason in record["reasons"]):
                        raise ValueError("Invalid acquisition checkpoint rejection reasons")
                if "disposition" in record and record["disposition"] not in ("retryable", "quarantined"):
                    raise ValueError("Invalid acquisition checkpoint failure disposition")

    @contextmanager
    def _file_lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Lock a stable inode, not the checkpoint inode replaced by save().
        with self.path.with_name(self.path.name + ".lock").open("a+b") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Acquisition checkpoint already has an active writer") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    @contextmanager
    def session(self):
        """Own a complete run; reload under the lock before making decisions."""
        if not self._guard.acquire(blocking=False):
            raise RuntimeError("Acquisition checkpoint already has an active writer")
        try:
            with self._file_lock():
                self.data = self._load()
                self._session_thread = threading.get_ident()
                try:
                    yield self
                finally:
                    self._session_thread = None
        finally:
            self._guard.release()

    @property
    def pending_persistence(self) -> list[str]:
        return list(self.data["pending_persistence"])

    @property
    def pending_index(self) -> list[str]:
        return list(self.data["pending_index"])

    @property
    def legacy_pending_index(self) -> list[str]:
        return list(self.data["legacy_pending_index"])

    def set_pending_persistence(self, names: list[str]) -> None:
        names = self._normalized_unique(names)
        self._check_capacity(names)
        self.data["pending_persistence"] = names

    def add_pending_persistence(self, names: list[str]) -> None:
        self.set_pending_persistence([*self.pending_persistence, *names])

    def clear_pending_persistence(self, names: list[str]) -> None:
        self.data["pending_persistence"] = self._without(
            self.pending_persistence, names
        )

    def set_pending_index(self, names: list[str]) -> None:
        names = self._normalized_unique(names)
        self._check_capacity(names)
        self.data["pending_index"] = names

    def add_pending_index(self, names: list[str]) -> None:
        self.set_pending_index([*self.pending_index, *names])

    def clear_pending_index(self, names: list[str]) -> None:
        self.data["pending_index"] = self._without(self.pending_index, names)

    def record_failure(
        self, repository: str, stage: str, error: object, *, disposition: str = "retryable",
    ) -> None:
        if disposition not in ("retryable", "permanent"):
            raise ValueError("Invalid failure disposition")
        name = normalize_repository_name(repository) or str(repository)
        self._check_capacity([name])
        previous = self._failure(name)
        # V1 did not track attempts or first failure. Start a documented grace
        # window rather than inventing an attempt history during migration.
        first = parse_time(previous.get("first_failed_at")) or self._clock()
        attempts = previous.get("attempts", 0) + 1
        now = self._clock()
        exhausted = (
            disposition == "permanent" or attempts >= self.policy.max_attempts
            or (now - first).total_seconds() >= self.policy.max_failure_age_seconds
        )
        self.clear_failure(name)
        self.data["failures"][name] = {
            "stage": stage,
            "error": str(error)[:500],
            "timestamp": self._now(),
            "first_failed_at": first.isoformat(),
            "attempts": attempts,
            "disposition": "quarantined" if exhausted else "retryable",
            "policy_version": self.policy.version,
            "next_attempt_at": (now + timedelta(seconds=self.policy.retry_delay(attempts, self._jitter()))).isoformat(),
        }

    def _failure(self, repository: str) -> dict[str, Any]:
        key = repository_identity_key(repository)
        return next((record for name, record in self.data["failures"].items()
                     if repository_identity_key(name) == key), {})

    def due(self, names: list[str], limit: int) -> list[str]:
        if type(limit) is not int or limit < 1:
            raise ValueError("Recovery limit must be a positive integer")
        now = self._clock()
        selected = []
        failures = {repository_identity_key(name): record for name, record in self.data["failures"].items()}
        for name in names:
            failure = failures.get(repository_identity_key(name), {})
            first = parse_time(failure.get("first_failed_at"))
            if (
                failure.get("attempts", 0) >= self.policy.max_attempts
                or first is not None and (now - first).total_seconds() >= self.policy.max_failure_age_seconds
            ):
                failure["disposition"] = "quarantined"
            if failure.get("disposition") == "quarantined":
                continue
            eligible_at = parse_time(failure.get("next_attempt_at"))
            if eligible_at is None or now >= eligible_at:
                selected.append(name)
                if len(selected) >= limit:
                    break
        return selected

    def replay_quarantined(self, repository: str) -> None:
        """Explicit operator repair preserves the queue entry and resets policy."""
        if self._failure(repository).get("disposition") != "quarantined":
            raise ValueError("Repository is not quarantined")
        self.clear_failure(repository)

    def excluded_names(self) -> set[str]:
        """Retry owners and current rejection cooldowns cannot become new work."""
        now = self._clock()
        excluded = set(self.pending_persistence) | set(self.pending_index) | set(self.legacy_pending_index)
        excluded.update(self.data["failures"])
        for name, record in self.data["rejected"].items():
            timestamp = parse_time(record.get("timestamp"))
            if (
                record.get("policy_version") == self.policy.version
                and timestamp is not None
                and (now - timestamp).total_seconds() < self.policy.rejection_cooldown_seconds
            ):
                excluded.add(name)
        return excluded

    def _check_capacity(self, names: list[str]) -> None:
        current = self._active_keys()
        added = {repository_identity_key(name) for name in names} - current
        if added and len(current | added) > self.policy.max_active_entries:
            raise RuntimeError("Acquisition checkpoint capacity exhausted; reconcile pending work")

    def _active_keys(self) -> set[str]:
        return {repository_identity_key(name) for name in (
            *self.pending_persistence, *self.pending_index, *self.legacy_pending_index,
            *self.data["failures"],
        )}

    def remaining_capacity(self) -> int:
        return max(0, self.policy.max_active_entries - len(self._active_keys()))

    def prune_history(self, *, reserve: int = 0) -> None:
        """Remove only terminal rejection history; unresolved work is retained."""
        now = self._clock()
        records = self.data["rejected"]
        active = {repository_identity_key(name) for name in self.excluded_names()}
        removable = []
        for name, record in list(records.items()):
            if repository_identity_key(name) in active:
                continue
            timestamp = parse_time(record.get("timestamp"))
            if timestamp is not None and (now - timestamp).total_seconds() >= self.policy.history_retention_seconds:
                records.pop(name)
            else:
                removable.append(name)
        # Cooldowns are active suppression state and must never be evicted to
        # admit new records; overflow of that state is backpressure, below.
        removable.sort(key=lambda name: records[name].get("timestamp") or "")
        for name in removable[:max(0, len(records) - self.policy.max_history_entries + reserve)]:
            records.pop(name)

    def clear_failure(self, repository: str) -> None:
        key = repository_identity_key(repository)
        for existing in list(self.data["failures"]):
            if repository_identity_key(existing) == key:
                self.data["failures"].pop(existing, None)

    def record_rejection(self, repository: str, reasons: list[str]) -> None:
        name = normalize_repository_name(repository) or str(repository)
        exists = repository_identity_key(name) in {repository_identity_key(value) for value in self.data["rejected"]}
        self.prune_history(reserve=0 if exists else 1)
        if (
            repository_identity_key(name) not in {repository_identity_key(value) for value in self.data["rejected"]}
            and len(self.data["rejected"]) >= self.policy.max_history_entries
        ):
            raise RuntimeError("Acquisition rejection history capacity exhausted")
        self.clear_rejection(name)
        self.data["rejected"][name] = {
            "reasons": [str(reason)[:300] for reason in reasons[:10]],
            "timestamp": self._now(),
            "policy_version": self.policy.version,
        }

    def clear_rejection(self, repository: str) -> None:
        key = repository_identity_key(repository)
        for existing in list(self.data["rejected"]):
            if repository_identity_key(existing) == key:
                self.data["rejected"].pop(existing, None)

    def record_run(self, report: dict[str, Any]) -> None:
        self.data["last_run"] = {**report, "completed_at": self._now()}

    def save(self) -> None:
        if self._session_thread is not None:
            if self._session_thread != threading.get_ident():
                raise RuntimeError("Acquisition checkpoint already has an active writer")
            self._save()
            return
        if not self._guard.acquire(blocking=False):
            raise RuntimeError("Acquisition checkpoint already has an active writer")
        try:
            with self._file_lock():
                self._save()
        finally:
            self._guard.release()

    def _save(self) -> None:
        try:
            current = self.path.read_bytes()
        except FileNotFoundError:
            current = None
        if current != self._loaded_bytes:
            raise RuntimeError("Acquisition checkpoint changed; reload before saving")
        self._validate(self.data)
        if self.data.get("version") != self.VERSION:
            raise ValueError("Cannot save an unsupported acquisition checkpoint version")
        self.data["updated_at"] = self._now()
        encoded = (json.dumps(self.data, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if len(encoded) > self.policy.max_checkpoint_bytes:
            raise RuntimeError("Acquisition checkpoint byte capacity exhausted; reconcile pending work")
        if current is not None:
            stored = json.loads(current)
            if (
                {key: value for key, value in stored.items() if key != "updated_at"}
                == {key: value for key, value in self.data.items() if key != "updated_at"}
            ):
                return
        if current is not None and json.loads(current)["version"] == 1:
            # Keep the exact original for recovery. A digest makes interruption
            # and repeated migration safe without overwriting another backup.
            digest = hashlib.sha256(current).hexdigest()
            backup = self.path.with_name(f"{self.path.name}.v1.{digest}.bak")
            try:
                fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                if backup.read_bytes() != current:
                    raise ValueError("Acquisition checkpoint migration backup is corrupt")
            else:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(current)
                    handle.flush()
                    os.fsync(handle.fileno())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
            self._loaded_bytes = self.path.read_bytes()
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def _now(self) -> str:
        return self._clock().isoformat()

    @staticmethod
    def _normalized_unique(names: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in names:
            normalized = normalize_repository_name(value)
            key = repository_identity_key(normalized)
            if normalized and key not in seen:
                seen.add(key)
                unique.append(normalized)
        return unique

    @staticmethod
    def _without(existing: list[str], removed: list[str]) -> list[str]:
        removed_keys = {repository_identity_key(value) for value in removed}
        return [
            value
            for value in existing
            if repository_identity_key(value) not in removed_keys
        ]
