"""Card sentence boundaries, mirrored by Weave's ML response validator.

Mask abbreviation periods without changing offsets so fallback extraction and
generated-output validation use exactly the same boundaries.
"""

import re
import unicodedata

SENTENCE_END_PATTERN = r"[.!?](?:[\"'”’)\]]+)?(?=\s|$)"
_END = re.compile(SENTENCE_END_PATTERN)


def mask_summary_abbreviations(value: str) -> str:
    # Use explicit ASCII token boundaries, matching JavaScript even when a
    # Latin abbreviation is adjacent to Chinese or other non-Latin text.
    value = re.sub(r"(?<![A-Za-z0-9_])(?:e\.g|i\.e)\.", lambda m: m[0].replace(".", "·"), value, flags=re.I | re.A)
    value = re.sub(
        r"(?<![A-Za-z0-9_])(?:Mr|Mrs|Ms|Dr|Prof|Jr|Sr|vs)\.(?=\s+(\S))",
        lambda m: m[0].replace(".", "·") if unicodedata.category(m[1]).startswith("L") else m[0],
        value,
    )
    return re.sub(
        r"(?<![A-Za-z0-9_])(?:etc\.|(?:[A-Z]\.){2,})(?=\s+(\S))",
        lambda m: m[0].replace(".", "·") if unicodedata.category(m[1]) == "Ll" else m[0],
        value,
    )


def split_summary_sentences(value: str) -> list[str]:
    result: list[str] = []
    start = 0
    for match in _END.finditer(mask_summary_abbreviations(value)):
        result.append(value[start:match.end()].strip())
        start = match.end()
    if value[start:].strip():
        result.append(value[start:].strip())
    return result


def has_two_or_three_complete_sentences(value: str) -> bool:
    endings = list(_END.finditer(mask_summary_abbreviations(value)))
    return len(endings) in {2, 3} and endings[-1].end() == len(value)


def has_complete_sentence_ending(value: str) -> bool:
    return any(match.end() == len(value) for match in _END.finditer(mask_summary_abbreviations(value)))
