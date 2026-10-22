"""Aggregate candidate-read metrics; never retain request or repository IDs."""

from collections import Counter
import threading

from pydantic_core import to_json


_CHANNELS = frozenset({"semantic", "trend", "activity", "popularity", "freshness"})
_LIMIT = 2**63 - 1
_lock = threading.Lock()
_counts: Counter[tuple[str, str]] = Counter()
# Two fixed modes per channel: payload-only or payload plus vectors. Retain
# only sizes/counts, never the sampled candidate or its contents.
_sample_sizes: dict[tuple[str, bool], int] = {}
_sample_reads: Counter[tuple[str, bool]] = Counter()
_BYTE_SAMPLE_INTERVAL = 16


def record_candidate_read(channel: str, points: list, fresh: int) -> None:
    if channel not in _CHANNELS:
        return
    fresh = max(0, min(len(points), fresh))
    with _lock:
        size = 0
        sampled = False
        if points:
            point = points[len(points) // 2]
            mode = (channel, bool(getattr(point, "vector", None)))
            reads = _sample_reads[mode]
            _sample_reads[mode] = (reads + 1) % _BYTE_SAMPLE_INTERVAL
            if reads == 0:
                # Decoded projected JSON, not wire/compressed response bytes.
                # Sample one point every 16 calls in each mode to bound overhead.
                sample = {
                    key: getattr(point, key, None)
                    for key in ("id", "payload", "vector", "score")
                }
                _sample_sizes[mode] = len(to_json(sample, fallback=str))
                sampled = True
            size = len(points) * (_sample_sizes[mode] + 1) + 1
        values = {
            "calls_total": 1,
            "candidates_total": len(points),
            "fresh_candidates_total": fresh,
            "rejected_candidates_total": len(points) - fresh,
            "candidate_bytes_estimate_total": size,
            "candidate_byte_samples_total": int(sampled),
        }
        for name, amount in values.items():
            key = (channel, name)
            _counts[key] = min(_LIMIT, _counts[key] + amount)


def render_candidate_metrics() -> list[str]:
    with _lock:
        counts = dict(_counts)
    descriptions = {
        "calls_total": "Successful Qdrant candidate reads.",
        "candidates_total": "Decoded repository candidates received.",
        "fresh_candidates_total": "Unique eligible unseen candidates per retrieval channel.",
        "rejected_candidates_total": "Seen, duplicate or invalid candidates rejected.",
        "candidate_bytes_estimate_total": "Sampled candidate JSON byte estimate, excluding HTTP framing and compression.",
        "candidate_byte_samples_total": "Points sampled for the candidate JSON size estimate.",
    }
    lines = []
    for name, description in descriptions.items():
        metric = f"ml_retrieval_{name}"
        lines.extend([f"# HELP {metric} {description}", f"# TYPE {metric} counter"])
        for channel in sorted(_CHANNELS):
            lines.append(f'{metric}{{channel="{channel}"}} {counts.get((channel, name), 0)}')
    return lines


def reset_candidate_metrics_for_tests() -> None:
    with _lock:
        _counts.clear()
        _sample_sizes.clear()
        _sample_reads.clear()
