"""Validated configuration for bounded, resumable corpus ingestion."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def positive_int(value: object, *, name: str) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer, got {parsed}")
    return parsed


def metadata_batch_size(value: object | None = None) -> int:
    """Keep metadata aliases at the existing, conservative ten-repo ceiling.

    This is an application response-size guard, not a GitHub alias limit.
    Scheduler chunks can be larger and are split independently by the enricher.
    """
    parsed = positive_int(
        os.getenv("GITHUB_METADATA_BATCH_SIZE", "10") if value is None else value,
        name="GITHUB_METADATA_BATCH_SIZE",
    )
    if parsed > 10:
        raise ValueError("GITHUB_METADATA_BATCH_SIZE must be between 1 and 10")
    return parsed


def readme_workers(value: object | None = None) -> int:
    """One README concurrency bound shared by every batch in an acquisition run."""
    parsed = positive_int(
        os.getenv("GITHUB_README_WORKERS", "4") if value is None else value,
        name="GITHUB_README_WORKERS",
    )
    if parsed > 16:
        raise ValueError("GITHUB_README_WORKERS must be between 1 and 16")
    return parsed


@dataclass(frozen=True, slots=True)
class CorpusPipelineSettings:
    target_count: int = 50_000
    max_cycles: int = 1
    checkpoint_path: Path = Path(".cache/acquisition_checkpoint.json")
    recovery_batch_size: int = 100

    @classmethod
    def from_environment(cls) -> "CorpusPipelineSettings":
        return cls(
            target_count=positive_int(
                os.getenv("CORPUS_TARGET_COUNT", "50000"),
                name="CORPUS_TARGET_COUNT",
            ),
            max_cycles=positive_int(
                os.getenv("ACQUISITION_MAX_CYCLES", "1"),
                name="ACQUISITION_MAX_CYCLES",
            ),
            checkpoint_path=Path(
                os.getenv(
                    "ACQUISITION_CHECKPOINT_PATH", ".cache/acquisition_checkpoint.json"
                )
            ),
            recovery_batch_size=positive_int(
                os.getenv("ACQUISITION_RECOVERY_BATCH_SIZE", "100"),
                name="ACQUISITION_RECOVERY_BATCH_SIZE",
            ),
        )

    def validated(self) -> "CorpusPipelineSettings":
        target_count = positive_int(self.target_count, name="target_count")
        max_cycles = positive_int(self.max_cycles, name="max_cycles")
        if not str(self.checkpoint_path).strip():
            raise ValueError("checkpoint_path must not be empty")
        return CorpusPipelineSettings(
            target_count=target_count,
            max_cycles=max_cycles,
            checkpoint_path=Path(self.checkpoint_path),
            recovery_batch_size=positive_int(self.recovery_batch_size, name="recovery_batch_size"),
        )
