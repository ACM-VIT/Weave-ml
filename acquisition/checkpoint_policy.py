"""Approved issue-85 lifecycle policy; no source payloads are retained."""

from dataclasses import dataclass
from datetime import datetime
import math


@dataclass(frozen=True)
class CheckpointPolicy:
    version: str = "issue-85-v1"
    max_attempts: int = 8
    max_failure_age_seconds: int = 7 * 24 * 3600
    rejection_cooldown_seconds: int = 7 * 24 * 3600
    history_retention_seconds: int = 30 * 24 * 3600
    retry_base_seconds: int = 60
    retry_max_seconds: int = 3600
    max_active_entries: int = 10_000
    max_history_entries: int = 1_000
    max_checkpoint_bytes: int = 8 * 1024 * 1024

    def __post_init__(self):
        if not self.version:
            raise ValueError("Checkpoint policy requires a version")
        for field, value in vars(self).items():
            if field != "version" and (type(value) is not int or value < 1):
                raise ValueError(f"{field} must be a positive integer")

    def retry_delay(self, attempts: int, jitter: float) -> float:
        if not math.isfinite(jitter) or not 0 <= jitter <= 1:
            raise ValueError("Retry jitter must be between zero and one")
        ceiling = min(self.retry_max_seconds, self.retry_base_seconds * 2 ** min(attempts - 1, 16))
        return ceiling * (0.5 + jitter / 2)


def parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None
