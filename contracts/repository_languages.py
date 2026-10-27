"""Optional sourced language observations, separate from semantic name lists."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

SafeBytes = Annotated[int, Field(strict=True, ge=0, le=2**53 - 1)]
Percentage = Annotated[float, Field(strict=True, ge=0, le=100, allow_inf_nan=False)]
LanguageName = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=128)]
PERCENTAGE_ROUNDING_TOLERANCE = 0.01
PERCENTAGE_EPSILON = 1e-8


class RepositoryLanguage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: LanguageName
    color: Annotated[str, StringConstraints(strict=True, pattern=r"^#[0-9a-fA-F]{6}$")] | None = None
    size: SafeBytes | None = None
    percentage: Percentage | None = None

    @field_validator("name")
    @classmethod
    def named_language(cls, value):
        if value.isdecimal():
            raise ValueError("language names cannot be numeric array indexes")
        return value


class RepositoryLanguageObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language_breakdown: list[RepositoryLanguage] | None = Field(default=None, max_length=100)
    language_total_size: SafeBytes | None = None
    language_breakdown_complete: Annotated[bool, Field(strict=True)] | None = None

    @model_validator(mode="after")
    def coherent_languages(self):
        rows = self.language_breakdown
        if rows is not None and len({row.name.casefold() for row in rows}) != len(rows):
            raise ValueError("language breakdown names must be unique")
        if rows is not None:
            # Each published row may be rounded independently to two decimals.
            percentages = [row.percentage for row in rows if row.percentage is not None]
            if sum(percentages) > 100 + len(percentages) * PERCENTAGE_ROUNDING_TOLERANCE + PERCENTAGE_EPSILON:
                raise ValueError("language percentages exceed the full repository")
            if self.language_total_size is not None and sum(row.size or 0 for row in rows) > self.language_total_size:
                raise ValueError("language byte sizes exceed the observed total")
            if self.language_total_size is not None and self.language_total_size > 0:
                for row in rows:
                    if row.size is not None and row.percentage is not None:
                        expected = 100 * row.size / self.language_total_size
                        if abs(row.percentage - expected) > PERCENTAGE_ROUNDING_TOLERANCE + PERCENTAGE_EPSILON:
                            raise ValueError("language percentage contradicts observed byte size and total")
        if self.language_breakdown_complete is True:
            if rows is None or self.language_total_size is None or any(row.size is None for row in rows):
                raise ValueError("complete language breakdown requires observed byte sizes and total")
            if sum(row.size for row in rows) != self.language_total_size:
                raise ValueError("complete language breakdown must cover the observed total")
        return self


LANGUAGE_OBSERVATION_FIELDS = tuple(RepositoryLanguageObservation.model_fields)


def language_observation(payload: Mapping) -> dict:
    """Validate/copy only supplied observations; absence is not empty evidence."""
    supplied = {key: payload[key] for key in LANGUAGE_OBSERVATION_FIELDS if key in payload}
    if not any(value is not None for value in supplied.values()):
        return {}
    return RepositoryLanguageObservation.model_validate(supplied).model_dump(mode="json", exclude_unset=True)
