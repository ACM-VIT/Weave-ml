"""Request-local proof of freshly validated atomic repository vector snapshots.

This deliberately is not a cache: membership always misses so every channel
record is validated. Only bounded digests survive; conflicting/invalid canonical
IDs cannot contribute a vector or metadata from a different snapshot.
"""
from __future__ import annotations

from array import array
import hashlib
import json
from typing import Any, Sequence
from uuid import UUID


class InlineCandidateSnapshots(dict):
    def __init__(self, *, core_fields: Sequence[str], extra_fields: Sequence[str], max_entries: int):
        super().__init__()
        if type(max_entries) is not int or not 1 <= max_entries <= 20_000:
            raise ValueError('inline snapshot capacity must be between 1 and 20000')
        self.core_fields, self.extra_fields = tuple(core_fields), tuple(extra_fields)
        self.max_entries = max_entries
        self._core: dict[str, bytes | None] = {}
        self._pairs: dict[str, bytes] = {}
        self.conflicts: set[str] = set()
        self.overflowed = False

    def __contains__(self, key: object) -> bool:
        # The retriever's normal eligibility memo must never skip validation of
        # a later record merely because another channel returned the same ID.
        return False

    @staticmethod
    def _digest(payload: dict[str, Any], fields: Sequence[str]) -> bytes:
        return hashlib.sha256(json.dumps(
            {field: payload.get(field) for field in fields}, sort_keys=True,
            separators=(',', ':'), ensure_ascii=True, allow_nan=False,
        ).encode()).digest()

    def __setitem__(self, key: tuple[str, bool], point: Any) -> None:
        point_id, require_vector = key
        try:
            repo_id = str(UUID(point_id))
        except (ValueError, TypeError, AttributeError):
            return  # Such an identity cannot become a canonical candidate.
        if repo_id not in self._core and len(self._core) >= self.max_entries:
            self.overflowed = True
            return
        if point is None:
            self._core.setdefault(repo_id, None)
            self.conflicts.add(repo_id)
            return
        try:
            core = self._digest(point.payload, self.core_fields)
            pair = None
            if require_vector:
                # point.vector already passed the retriever's dimension/finite
                # validation; no older vector is ever filled in here.
                pair = hashlib.sha256(core + self._digest(point.payload, self.extra_fields)
                                      + array('d', point.vector).tobytes()).digest()
        except (TypeError, ValueError, OverflowError):
            self._core.setdefault(repo_id, None)
            self.conflicts.add(repo_id)
            return
        if repo_id in self._core and self._core[repo_id] != core:
            self.conflicts.add(repo_id)
        self._core.setdefault(repo_id, core)
        if pair is not None:
            if repo_id in self._pairs and self._pairs[repo_id] != pair:
                self.conflicts.add(repo_id)
            self._pairs.setdefault(repo_id, pair)

    def has_pair(self, repo_id: str) -> bool:
        return not self.overflowed and repo_id in self._pairs and repo_id not in self.conflicts

    @property
    def tracked_ids(self) -> int:
        return len(self._core)
