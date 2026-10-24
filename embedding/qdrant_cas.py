"""Shared building blocks for Qdrant compare-and-set writes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def payload_snapshot_filter(
    models: Any,
    *,
    point_id: Any,
    payload: Mapping[str, Any],
    fields: Iterable[str],
) -> Any:
    """Match one point and the exact scalar payload snapshot for ``fields``.

    Missing and null values use ``IsEmptyCondition`` instead of being coerced
    to a default. That distinction is part of the fencing token: a point with
    no cursor must not match a cursor initialized after the read.
    """

    conditions: list[Any] = [models.HasIdCondition(has_id=[point_id])]
    for field in fields:
        if field not in payload or payload[field] is None:
            conditions.append(
                models.IsEmptyCondition(
                    is_empty=models.PayloadField(key=field),
                )
            )
            continue
        value = payload[field]
        if not isinstance(value, (str, int, bool)):
            raise TypeError(
                f"Qdrant CAS field {field!r} must contain a scalar string, boolean or number"
            )
        conditions.append(
            models.FieldCondition(
                key=field,
                match=models.MatchValue(value=value),
            )
        )
    return models.Filter(must=conditions)


def payload_matches(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    fields: Iterable[str],
) -> bool:
    """Return whether selected fields retain the same value and presence."""

    for field in fields:
        actual_present = field in payload and payload[field] is not None
        expected_present = field in expected and expected[field] is not None
        if actual_present != expected_present:
            return False
        if actual_present and payload[field] != expected[field]:
            return False
    return True
