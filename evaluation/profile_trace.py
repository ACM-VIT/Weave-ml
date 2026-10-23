"""File-driven profile-read opportunities, without a serving cache or policy.

Inputs contain pseudonyms and same-read APPLIED Qdrant revision observations,
never raw user IDs/vectors/text or backend accepted-work versions. Provenance and
clock domains are supplied declarations, not authenticated by this analyzer.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

MAX_BYTES = 32 * 1024 * 1024
MAX_RECORDS = 100000
_KEY = re.compile(r"[a-f0-9]{64}")
_REVISION = re.compile(r"0|[1-9][0-9]{0,18}")
_FIELDS = {"subject_key", "process_key", "clock_key", "at_ms", "profile_version", "feedback_version",
           "feedback_state_revision", "embedding_contract_key", "compatible", "fetch_duration_ms"}


def _invalid():
    raise ValueError("PROFILE_TRACE_INVALID")


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            _invalid()
        result[key] = value
    return result


def validate_trace(value):
    if not isinstance(value, dict) or set(value) != {"schema_version", "provenance", "records"}:
        _invalid()
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        _invalid()
    provenance = value["provenance"]
    if (not isinstance(provenance, dict) or set(provenance) != {"kind", "reference"}
            or not isinstance(provenance["kind"], str) or provenance["kind"] not in {"synthetic", "measured"}
            or not isinstance(provenance["reference"], str) or not provenance["reference"].strip()
            or not 1 <= len(provenance["reference"]) <= 512):
        _invalid()
    try:
        provenance["reference"].encode("utf8")
    except UnicodeError:
        _invalid()
    rows = value["records"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_RECORDS:
        _invalid()
    for row in rows:
        if not isinstance(row, dict) or not set(row) <= _FIELDS:
            _invalid()
        for name in ("subject_key", "process_key", "clock_key", "embedding_contract_key"):
            key = row.get(name)
            if key is not None and (not isinstance(key, str) or _KEY.fullmatch(key) is None):
                _invalid()
        timestamp = row.get("at_ms")
        if timestamp is not None and (type(timestamp) is not int or not 0 <= timestamp <= 2**53-1):
            _invalid()
        for name in ("profile_version", "feedback_version", "feedback_state_revision"):
            revision = row.get(name)
            if revision is not None and (not isinstance(revision, str) or _REVISION.fullmatch(revision) is None
                    or int(revision) > 2**63-1 or (name == "profile_version" and revision == "0")):
                _invalid()
        if row.get("compatible") is not None and type(row["compatible"]) is not bool:
            _invalid()
        duration = row.get("fetch_duration_ms")
        if duration is not None and (type(duration) not in {int, float} or not 0 <= duration <= 60000 or not math.isfinite(duration)):
            _invalid()
    return value


def load_trace(path: Path):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= MAX_BYTES:
        _invalid()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            _invalid()
        raw = stream.read(MAX_BYTES+1)
        after = os.fstat(stream.fileno())
    if (len(raw) != before.st_size or len(raw) > MAX_BYTES
            or (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
        _invalid()
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=lambda _: _invalid())
    except (ValueError, UnicodeError, RecursionError):
        _invalid()
    return validate_trace(value), hashlib.sha256(raw).hexdigest()


def _quantile(values, fraction):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values)*fraction)-1)]


def _version(row):
    names = ("profile_version", "feedback_version", "feedback_state_revision")
    return tuple(int(row[name]) for name in names) if all(row.get(name) is not None for name in names) else None


def _eligible(row):
    return (row.get("subject_key") is not None and row.get("compatible") is True
            and row.get("embedding_contract_key") is not None and _version(row) is not None)


def _fingerprint(row):
    return row["subject_key"], row["embedding_contract_key"], _version(row)


def analyze_trace(trace, *, max_age_ms=()):
    """Observed duplicate/version/time opportunity bounds, never actual hits."""
    validate_trace(trace)
    if (not isinstance(max_age_ms, (list, tuple)) or len(max_age_ms) > 5
            or any(type(age) is not int or not 1 <= age <= 2**53-1 for age in max_age_ms)
            or len(set(max_age_ms)) != len(max_age_ms)):
        _invalid()
    rows = trace["records"]
    coverage = {"records": len(rows), "eligible_versioned_compatible_reads": 0, "missing_subject": 0,
        "missing_process": 0, "missing_clock": 0, "missing_timestamp": 0, "missing_revision": 0,
        "missing_embedding_contract": 0, "unknown_compatibility": 0, "incompatible_reads": 0}
    global_keys, local_keys, process_subjects = set(), set(), {}
    global_duplicates = local_duplicates = local_reads = 0
    ordered, last_input_clock = [], {}
    out_of_order = 0
    for index, row in enumerate(rows):
        for field, counter in (("subject_key", "missing_subject"), ("process_key", "missing_process"),
                ("clock_key", "missing_clock"), ("at_ms", "missing_timestamp"),
                ("embedding_contract_key", "missing_embedding_contract"), ("compatible", "unknown_compatibility")):
            coverage[counter] += row.get(field) is None
        coverage["missing_revision"] += _version(row) is None
        coverage["incompatible_reads"] += row.get("compatible") is False
        if not _eligible(row):
            continue
        coverage["eligible_versioned_compatible_reads"] += 1
        key = _fingerprint(row)
        global_duplicates += key in global_keys
        global_keys.add(key)
        process = row.get("process_key")
        if process is not None:
            local_reads += 1
            local_key = (process, key)
            local_duplicates += local_key in local_keys
            local_keys.add(local_key)
            process_subjects.setdefault(row["subject_key"], set()).add(process)
        if process is not None and row.get("clock_key") is not None and row.get("at_ms") is not None:
            domain = process, row["clock_key"]
            previous = last_input_clock.get(domain)
            out_of_order += previous is not None and row["at_ms"] < previous
            last_input_clock[domain] = max(previous if previous is not None else row["at_ms"], row["at_ms"])
            ordered.append((domain, row["at_ms"], index, row))
    ordered.sort(key=lambda item: (item[0], item[1], item[2]))
    last_revision, last_request, temporal = {}, {}, []
    regressions, gaps = 0, []
    for domain, timestamp, _, row in ordered:
        subject = domain, row["subject_key"], row["embedding_contract_key"]
        previous = last_revision.get(subject)
        revision = _version(row)
        if previous is not None and any(left < right for left, right in zip(revision, previous)):
            regressions += 1
            continue
        last_revision[subject] = revision
        if subject in last_request:
            gaps.append(timestamp-last_request[subject])
        last_request[subject] = timestamp
        temporal.append((domain, timestamp, row))
    experiments = []
    for age in max_age_ms:
        fills, opportunities = {}, 0
        for domain, timestamp, row in temporal:
            key = domain, _fingerprint(row)
            filled = fills.get(key)
            if filled is not None and timestamp-filled < age:
                opportunities += 1
            else:
                fills[key] = timestamp
        experiments.append({"max_age_ms": age, "eligible_reads": len(temporal),
            "compatible_repeat_opportunities": opportunities,
            "fraction_of_temporally_eligible_reads": opportunities/len(temporal) if temporal else None})
    durations = [row["fetch_duration_ms"] for row in rows if row.get("fetch_duration_ms") is not None]
    eligible = coverage["eligible_versioned_compatible_reads"]
    provenance = {"kind": trace["provenance"]["kind"],
        "reference_sha256": hashlib.sha256(trace["provenance"]["reference"].encode()).hexdigest()}
    return {"schema": "profile-reuse-opportunity-v1", "provenance": provenance, "coverage": coverage,
        "observed_cost": {"samples": len(durations), "p50_ms": _quantile(durations,.5), "p95_ms": _quantile(durations,.95)},
        "unbounded_free_version_oracle": {"global_duplicate_version_keys": global_duplicates,
            "global_fraction_of_eligible_reads": global_duplicates/eligible if eligible else None,
            "known_process_reads": local_reads, "process_local_duplicate_version_keys": local_duplicates,
            "process_local_fraction": local_duplicates/local_reads if local_reads else None,
            "subjects_observed_on_multiple_processes": sum(len(processes)>1 for processes in process_subjects.values())},
        "clock_and_order": {"temporally_eligible_reads": len(temporal), "input_rows_out_of_timestamp_order": out_of_order,
            "chronological_revision_regressions_excluded_from_time_experiments": regressions,
            "comparable_same_subject_request_gaps": len(gaps), "gap_p50_ms": _quantile(gaps,.5), "gap_p95_ms": _quantile(gaps,.95)},
        "max_age_experiments": experiments,
        "scope": "supplied pseudonymous current-profile observations; observed version duplicates are free-oracle upper bounds, not actual cache hits or representative traffic claims",
        "clock_scope": "sort only within supplied process/clock domains; domain declarations are not authenticated; no cross-process clock/freshness guarantee",
        "experiment_scope": "explicit ages only; instantaneous fills, unlimited per-version keys, no invalidation/authority/network/coalescing cost; strict age<max_age; hit does not renew fill time",
        "freshness_scope": "applied Qdrant revisions supplied by producer; backend acceptance is not proof of applied learning; no invalidation or feedback-effect delay measured",
        "decision": "no cache installed; production go/no-go/staleness/outage policy not selected"}
