"""Bounded process-local summary analysis/artifact reuse and single flight."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import sys
import threading
from typing import Any, Mapping

from utils.readme_processor import MAX_PREPROCESS_CHARS
from .contracts import CardSummaryArtifact
from .source import MAX_SOURCE_ANALYSIS_WORK, SummarySource
from .validation import SourceAnalysis


# Bump when source projection or validation policy changes. These are local
# reuse identities; the public backend artifact schema stays unchanged.
SOURCE_ANALYSIS_VERSION = "summary-source-analysis-v1"
SUMMARY_VALIDATION_VERSION = "summary-validation-v1"


def source_identity(source: Any, repo: Mapping[str, Any]) -> bytes | None:
    """Hash exact bounded inputs, including evidence omitted from the prompt.

    Unsupported or oversized metadata bypasses reuse, preserving the source
    builder's existing fail-closed behavior without unbounded serialization.
    """
    remaining = MAX_SOURCE_ANALYSIS_WORK

    def bounded(value: Any) -> Any:
        nonlocal remaining
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            if len(value) > MAX_PREPROCESS_CHARS:
                raise ValueError("unbounded cache input")
            remaining -= len(value)
            if remaining < 0:
                raise ValueError("cache input work bound exceeded")
            return value
        if isinstance(value, list) and len(value) <= 80:
            # Source metadata/paragraph lists are flat. Reject nested or unusual
            # values instead of traversing arbitrary objects or cycles.
            if any(isinstance(item, (list, dict)) for item in value):
                raise ValueError("unsupported nested cache input")
            return [bounded(item) for item in value]
        raise ValueError("unsupported cache input")

    try:
        projection = {key: bounded(repo.get(key)) for key in (
            "repo_id", "github_id", "full_name", "content_hash", "description",
            "primary_language", "topics", "languages", "readme", "clean_text",
            "readme_md", "extracted_paragraphs",
        )}
        readme = getattr(source, "readme", None)
        projection["source_readme"] = [bounded(getattr(readme, key, None)) for key in (
            "raw_markdown", "clean_text", "readme_md",
        )]
        encoded = json.dumps(projection, ensure_ascii=True, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("ascii")
    except (TypeError, ValueError, OverflowError):
        return None
    return hashlib.sha256(encoded).digest()


@dataclass(frozen=True, slots=True)
class SummaryWork:
    material: SummarySource
    analysis: tuple[SourceAnalysis, ...]
    artifact: CardSummaryArtifact | None = None


def _retained_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Account Python containers and tokens, not just serialized text bytes."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_retained_bytes(k, seen) + _retained_bytes(v, seen) for k, v in value.items())
    elif isinstance(value, (tuple, list, set, frozenset)):
        size += sum(_retained_bytes(item, seen) for item in value)
    elif is_dataclass(value) and not isinstance(value, type):
        size += sum(_retained_bytes(getattr(value, field.name), seen) for field in fields(value))
    elif isinstance(value, CardSummaryArtifact):
        size += _retained_bytes(value.__dict__, seen)
        size += _retained_bytes(value.__pydantic_private__, seen)
    return size


class SummaryReuseCache:
    def __init__(self, *, max_entries: int = 32, max_bytes: int = 16 * 1024 * 1024,
                 max_inflight: int = 32) -> None:
        for value in (max_entries, max_bytes, max_inflight):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("summary cache limits must be positive integers")
        self.max_entries, self.max_bytes, self.max_inflight = max_entries, max_bytes, max_inflight
        self._lock = threading.Lock()
        self._entries: OrderedDict[bytes, tuple[SummaryWork, int]] = OrderedDict()
        self._flights: dict[bytes, Future] = {}
        self._bytes = 0
        self._counts = dict(hits=0, misses=0, artifact_hits=0, coalesced=0,
                            saturated=0, provider_calls=0)

    def get(self, key: bytes | None) -> SummaryWork | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._counts["misses"] += 1
                return None
            self._entries.move_to_end(key)
            self._counts["hits"] += 1
            return entry[0]

    def put(self, key: bytes | None, work: SummaryWork) -> None:
        if key is None:
            return
        size = _retained_bytes(work) + sys.getsizeof(key) + 256
        if size > self.max_bytes:
            return
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous[1]
            while self._entries and (len(self._entries) >= self.max_entries
                                     or self._bytes + size > self.max_bytes):
                _, (_, removed_size) = self._entries.popitem(last=False)
                self._bytes -= removed_size
            self._entries[key] = (work, size)
            self._bytes += size

    def start(self, key: bytes) -> tuple[Future | None, bool]:
        with self._lock:
            if key in self._flights:
                self._counts["coalesced"] += 1
                return self._flights[key], False
            if len(self._flights) >= self.max_inflight:
                self._counts["saturated"] += 1
                return None, False
            future = Future()
            self._flights[key] = future
            return future, True

    def finish(self, key: bytes, future: Future, *, artifact: CardSummaryArtifact | None = None,
               error: BaseException | None = None) -> None:
        # Complete before removal so existing waiters always see the result;
        # failed/fallback work is never retained as a generated result.
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(artifact)
        with self._lock:
            if self._flights.get(key) is future:
                del self._flights[key]

    def count(self, name: str) -> None:
        with self._lock:
            self._counts[name] += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {**self._counts, "entries": len(self._entries),
                    "retained_bytes": self._bytes, "inflight": len(self._flights)}
