"""Structured results for acquisition and corpus orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AcquisitionFailure:
    repository: str
    stage: str
    error: str


@dataclass(slots=True)
class AcquisitionRunResult:
    repositories: list[Any] = field(default_factory=list)
    failures: list[AcquisitionFailure] = field(default_factory=list)
    discovered_count: int = 0
    skipped_existing_count: int = 0
    duplicates_removed: int = 0


@dataclass(slots=True)
class CorpusRunReport:
    manifest_status: str = "not_requested"
    discovered: int = 0
    skipped_existing: int = 0
    duplicates_removed: int = 0
    enriched: int = 0
    rejected: int = 0
    persisted: int = 0
    indexed: int = 0
    resumed_persistence: int = 0
    resumed_indexing: int = 0
    failures: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_status": self.manifest_status,
            "discovered": self.discovered,
            "skipped_existing": self.skipped_existing,
            "duplicates_removed": self.duplicates_removed,
            "enriched": self.enriched,
            "rejected": self.rejected,
            "persisted": self.persisted,
            "indexed": self.indexed,
            "resumed_persistence": self.resumed_persistence,
            "resumed_indexing": self.resumed_indexing,
            "failures": dict(self.failures),
        }
