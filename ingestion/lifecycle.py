"""Candidate verified-status-v1 policy; serving remains opt-in pending owner approval."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

POLICY_VERSION = "verified-status-v1"
LIFECYCLE_FIELDS = (
    "lifecycle_archived", "lifecycle_disabled", "lifecycle_observed_at",
    "lifecycle_source", "lifecycle_last_attempt_at", "lifecycle_last_attempt_status",
)


class RepositoryLifecycleObservation(BaseModel):
    """One backend-owned GitHub response, never inferred from push recency.

    Unavailable includes permission errors, timeouts and ambiguous not-found
    responses. It supplies no archive/disable evidence and cannot revive a repo.
    """
    model_config = ConfigDict(extra="forbid")
    status: Literal["verified", "unavailable"]
    observed_at: datetime
    source: Literal["github-graphql-repository", "github-rest-repository"]
    archived: Annotated[bool, Field(strict=True)] | None = None
    disabled: Annotated[bool, Field(strict=True)] | None = None

    @field_validator("observed_at")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("lifecycle observed_at must include a timezone")
        try:
            normalized = value.astimezone(timezone.utc)
        except OverflowError as exc:
            raise ValueError("lifecycle observed_at must normalize within years 1 through 9999") from exc
        if normalized > datetime.now(timezone.utc):
            raise ValueError("lifecycle observation cannot be in the future")
        return normalized

    @model_validator(mode="after")
    def complete_observation(self):
        flags = (self.archived, self.disabled)
        if self.status == "verified" and any(value is None for value in flags):
            raise ValueError("verified lifecycle requires both archived and disabled")
        if self.status == "unavailable" and any(value is not None for value in flags):
            raise ValueError("unavailable lifecycle cannot assert repository status")
        return self


def _time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("stored lifecycle timestamp must include a timezone")
    return result.astimezone(timezone.utc)


def merge_lifecycle(stored: dict[str, Any], observation: Any) -> dict[str, Any]:
    """Monotone verified evidence, independent of job ordering and content age.

    Failed attempts retain the last verified state indefinitely. Equal-time
    conflicting evidence is rejected rather than resolving a tie by arrival.
    """
    result = {field: stored[field] for field in LIFECYCLE_FIELDS if field in stored}
    if observation is None:
        return result
    item = RepositoryLifecycleObservation.model_validate(observation)
    if item.observed_at > datetime.now(timezone.utc):
        raise ValueError("lifecycle observation cannot be in the future")
    stamp = item.observed_at.isoformat()
    last_attempt = result.get("lifecycle_last_attempt_at")
    if (last_attempt is None or item.observed_at > _time(last_attempt)
            or (item.observed_at == _time(last_attempt) and item.status == "verified")):
        result.update(lifecycle_last_attempt_at=stamp, lifecycle_last_attempt_status=item.status)
    if item.status == "unavailable":
        return result
    last_verified = result.get("lifecycle_observed_at")
    if last_verified is not None:
        previous = _time(last_verified)
        if item.observed_at < previous:
            return result
        if item.observed_at == previous:
            if (result.get("lifecycle_archived"), result.get("lifecycle_disabled")) != (item.archived, item.disabled):
                raise ValueError("conflicting lifecycle evidence at the same observation time")
            return result
    result.update(lifecycle_archived=item.archived, lifecycle_disabled=item.disabled,
                  lifecycle_observed_at=stamp, lifecycle_source=item.source)
    return result


def lifecycle_excluded(payload: dict[str, Any]) -> bool:
    # Missing legacy observations remain eligible; flags originate only from a
    # validated complete authoritative observation at our authenticated boundary.
    return payload.get("lifecycle_archived") is True or payload.get("lifecycle_disabled") is True
