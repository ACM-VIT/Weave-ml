"""Card sentence boundaries, mirrored by Weave's ML response validator.

Mask abbreviation periods without changing offsets so fallback extraction and
generated-output validation use exactly the same boundaries.
"""

import re
import unicodedata

_END = re.compile(r"[.!?](?:[\"')\]]+)?(?=\s|$)")


def _masked(value: str) -> str:
    value = re.sub(r"\b(?:e\.g|i\.e)\.", lambda m: m[0].replace(".", "·"), value, flags=re.I)
    value = re.sub(
        r"\b(?:Mr|Mrs|Ms|Dr|Prof|Jr|Sr|vs)\.(?=\s+([^\W\d_]))",
        lambda m: m[0].replace(".", "·") if unicodedata.category(m[1]).startswith("L") else m[0],
        value,
    )
    return re.sub(
        r"\b(?:etc\.|(?:[A-Z]\.){2,})(?=\s+([^\W\d_]))",
        lambda m: m[0].replace(".", "·") if unicodedata.category(m[1]) == "Ll" else m[0],
        value,
    )


def split_summary_sentences(value: str) -> list[str]:
    result: list[str] = []
    start = 0
    for match in _END.finditer(_masked(value)):
        result.append(value[start:match.end()].strip())
        start = match.end()
    if value[start:].strip():
        result.append(value[start:].strip())
    return result


def has_two_or_three_complete_sentences(value: str) -> bool:
    return len(_END.findall(_masked(value))) in {2, 3} and bool(_END.search(value[-4:]))
