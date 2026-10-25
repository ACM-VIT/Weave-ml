"""Bound structured offline inputs before copying or serializing their text."""

from collections.abc import Mapping
from datetime import date, datetime
import math
from typing import Any


def check_field_budget(values: list[Any], *, max_bytes: int, label: str) -> None:
    """Conservative UTF-8 field budget; no full-source JSON allocation.

    Delimiters count toward the budget and scalar metadata has a fixed allowance.
    The separate Qdrant serializer remains authoritative for wire request bytes.
    """
    remaining = max_bytes

    def measure(value: Any, depth: int = 0) -> None:
        nonlocal remaining
        if depth > 64:
            raise ValueError(f"{label} nesting exceeds 64 levels")
        # Check length before encoding strings: reject a giant README without
        # allocating its escaped JSON representation or another full copy.
        if isinstance(value, str):
            if len(value) > remaining:
                raise ValueError(f"{label} exceeds the {max_bytes} byte field budget")
            remaining -= len(value.encode("utf-8")) + 2
        elif isinstance(value, Mapping):
            remaining -= 2
            for key, item in value.items():
                measure(key, depth + 1)
                measure(item, depth + 1)
                remaining -= 2
        elif isinstance(value, (list, tuple)):
            remaining -= 2
            for item in value:
                measure(item, depth + 1)
                remaining -= 1
        elif value is None or type(value) in (bool, date, datetime):
            remaining -= 32
        elif type(value) is int:
            if value.bit_length() > 64:
                raise ValueError(f"{label} integer exceeds the 64-bit scalar budget")
            remaining -= 32
        elif type(value) is float and math.isfinite(value):
            remaining -= 32
        else:
            # Bytes and arbitrary objects can expand without bound when tower
            # preparation calls str(). Reject them before making any copies.
            raise TypeError(f"{label} contains an unsupported scalar type")
        if remaining < 0:
            raise ValueError(f"{label} exceeds the {max_bytes} byte field budget")

    for value in values:
        measure(value)
