"""Explicitly invoked bounded README experiment; never a serving strategy flag.

The normal CLI is plan-only. Live calls belong only to its dedicated hard-timed
worker after the operator has authority for the token and public sample.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import ssl
import stat
import tempfile
import threading
import time
import zlib

MAX_CALLS = 40
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_REQUEST_BYTES = 64 * 1024
HOST = "api.github.com"


def start_worker_watchdog(seconds=180):
    """Only for a disposable experiment child; process exit bounds blocked I/O."""
    if type(seconds) not in {int, float} or not math.isfinite(seconds) or not 0 < seconds <= 180:
        raise ReadmeExperimentError("CONFIG")
    timer = threading.Timer(seconds, lambda: os._exit(124))
    timer.daemon = True
    timer.start()
    return timer


def read_sample_bytes(path: Path):
    path = Path(path)
    observed = path.lstat()
    if not stat.S_ISREG(observed.st_mode) or not 1 <= observed.st_size <= 65536:
        raise ReadmeExperimentError("BOUND")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or (observed.st_dev, observed.st_ino) != (opened.st_dev, opened.st_ino):
            raise ReadmeExperimentError("CONFIG")
        raw = stream.read(65537)
        after = os.fstat(stream.fileno())
    if (len(raw) != observed.st_size or len(raw) > 65536
            or (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
        raise ReadmeExperimentError("BOUND")
    return raw


class ReadmeExperimentError(RuntimeError):
    def __init__(self, reason="TRANSPORT"):
        if reason not in {"CONFIG", "IDENTITY", "HTTP", "BODY_LIMIT", "DEADLINE", "JSON", "TRANSPORT", "BOUND"}:
            reason = "TRANSPORT"
        super().__init__("README_EXPERIMENT_" + reason)


def _pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError("ambiguous JSON")
        value[key] = item
    return value


def decode_sample_bytes(raw):
    try:
        return json.loads(raw, object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (ValueError, UnicodeError, RecursionError):
        raise ReadmeExperimentError("JSON") from None


@dataclass(frozen=True)
class _HTTPOutcome:
    data: dict | None = field(default=None, repr=False)
    observation: dict = field(default_factory=dict)
    failure: str | None = None


class LiveGitHubTransport:
    """No configurable host, proxy, redirect, retry or ambient-token fallback."""
    __slots__ = ("_token", "deadline", "observations", "_verified")

    def __init__(self, token: str, *, seconds=180):
        if (not isinstance(token, str) or not 1 <= len(token) <= 4096
                or any(not 33 <= ord(character) <= 126 for character in token)
                or type(seconds) not in {int, float} or not math.isfinite(seconds) or not 0 < seconds <= 180):
            raise ReadmeExperimentError("CONFIG")
        self._token = token
        self.deadline = time.monotonic() + seconds
        self.observations = []
        self._verified = False

    @property
    def last_measurement(self):
        if not self.observations:
            return None
        return dict(self.observations[-1])

    def _request_private(self, method, path, data) -> _HTTPOutcome:
        # Raw headers/body/provider exceptions never leave this private frame
        # through a traceback/cause. The outer method raises a fresh safe error.
        observation = {"kind": "identity" if path == "/user" else "graphql", "http_status": None,
                       "wire_bytes": 0, "decoded_body_bytes": 0, "http_request_started": False}
        failure = "TRANSPORT"
        connection = None
        started = time.perf_counter()
        try:
            if (method, path) not in {("GET", "/user"), ("POST", "/graphql")}:
                return _HTTPOutcome(observation=observation, failure="CONFIG")
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                return _HTTPOutcome(observation=observation, failure="DEADLINE")
            payload = None if data is None else json.dumps(data, allow_nan=False, separators=(",", ":")).encode()
            if payload is not None and len(payload) > MAX_REQUEST_BYTES:
                return _HTTPOutcome(observation=observation, failure="BOUND")
            # HTTPSConnection connects directly. Environment proxy settings are
            # never consumed; status3xx is an error, never a followed location.
            connection = http.client.HTTPSConnection(HOST, timeout=min(5, remaining), context=ssl.create_default_context())
            observation["http_request_started"] = True
            connection.request(method, path, body=payload, headers={
                "Authorization": "Bearer " + self._token, "User-Agent": "weave-readme-experiment",
                "Accept": "application/vnd.github+json", "Content-Type": "application/json",
                "Accept-Encoding": "identity",
            })
            response = connection.getresponse()
            observation["http_status"] = response.status
            if response.status != 200:
                return _HTTPOutcome(observation=observation, failure="HTTP")
            encoding = (response.getheader("Content-Encoding") or "identity").strip().lower()
            if encoding not in {"identity", "gzip"}:
                return _HTTPOutcome(observation=observation, failure="HTTP")
            decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS) if encoding == "gzip" else None
            body = bytearray()
            while True:
                remaining = self.deadline - time.monotonic()
                if remaining <= 0:
                    return _HTTPOutcome(observation=observation, failure="DEADLINE")
                if connection.sock is not None:
                    connection.sock.settimeout(min(5, remaining))
                chunk = response.read1(min(65536, MAX_RESPONSE_BYTES-observation["wire_bytes"]+1))
                if not chunk:
                    break
                observation["wire_bytes"] += len(chunk)
                if observation["wire_bytes"] > MAX_RESPONSE_BYTES:
                    return _HTTPOutcome(observation=observation, failure="BODY_LIMIT")
                decoded = decompressor.decompress(chunk, MAX_RESPONSE_BYTES-len(body)+1) if decompressor else chunk
                body.extend(decoded)
                observation["decoded_body_bytes"] = len(body)
                if len(body) > MAX_RESPONSE_BYTES or (decompressor and decompressor.unconsumed_tail):
                    return _HTTPOutcome(observation=observation, failure="BODY_LIMIT")
            if decompressor:
                body.extend(decompressor.flush(MAX_RESPONSE_BYTES-len(body)+1))
                observation["decoded_body_bytes"] = len(body)
                if len(body) > MAX_RESPONSE_BYTES:
                    return _HTTPOutcome(observation=observation, failure="BODY_LIMIT")
                if not decompressor.eof or decompressor.unused_data:
                    return _HTTPOutcome(observation=observation, failure="JSON")
            failure = "JSON"
            parsed = json.loads(body, object_pairs_hook=_pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
            if not isinstance(parsed, dict):
                return _HTTPOutcome(observation=observation, failure="JSON")
            if time.monotonic() >= self.deadline:
                return _HTTPOutcome(observation=observation, failure="DEADLINE")
            return _HTTPOutcome(data=parsed, observation=observation)
        except Exception:
            return _HTTPOutcome(observation=observation, failure="DEADLINE" if time.monotonic() >= self.deadline else failure)
        finally:
            observation["transport_elapsed_ms"] = (time.perf_counter()-started)*1000
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def _request(self, method, path, data=None):
        if len(self.observations) >= MAX_CALLS + 1:
            raise ReadmeExperimentError("BOUND")
        outcome = self._request_private(method, path, data)
        self.observations.append(outcome.observation)
        if outcome.failure is not None:
            raise ReadmeExperimentError(outcome.failure)
        return outcome.data

    def verify_identity(self):
        if self._verified or self.observations:
            raise ReadmeExperimentError("IDENTITY")
        value = self._request("GET", "/user")
        if value.get("login") != "TarunRam-git":
            raise ReadmeExperimentError("IDENTITY")
        self._verified = True

    def __call__(self, query, variables):
        if not self._verified:
            raise ReadmeExperimentError("IDENTITY")
        return self._request("POST", "/graphql", {"query": query, "variables": variables})


def comparison_trials(repositories, repeats, measurement, compare):
    """Balanced repeated matched trials, sharing one fixed total-call budget."""
    if (type(repeats) is not int or not 1 <= repeats <= 5
            or not 1 <= len(repositories) <= 10 or len(repositories)*repeats > 10):
        raise ReadmeExperimentError("BOUND")
    trials = []
    for repeat in range(repeats):
        for position, repository in enumerate(repositories):
            order = ("aliases", "metadata") if (repeat+position) % 2 == 0 else ("metadata", "aliases")
            result = compare(repository, measurement, strategies=order)
            trials.append({"repeat": repeat, "strategy_order": list(order), **result})
    return trials


def _percentile(values, fraction):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values)*fraction)-1)]


def summarize_trials(trials):
    result = {}
    for strategy in ("aliases", "metadata"):
        counts = {key: 0 for key in ("content", "missing", "failed", "not_run")}
        calls, successful_ms, attempted_ms, pure_transport_ms, partial = [], [], [], [], 0
        for trial in trials:
            item = trial.get("strategies", {}).get(strategy)
            if item is None:
                counts["not_run"] += 1
                continue
            status = item.get("status")
            if status not in {"content", "missing", "failed"}:
                raise ReadmeExperimentError("CONFIG")
            counts[status] += 1
            observed = [*trial["shared_reference_requests"], *item["requests"]]
            calls.extend(observed)
            partial += int(any(call.get("graphql_errors") for call in observed))
            elapsed = trial["shared_reference_elapsed_ms"] + item["elapsed_ms"]
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ReadmeExperimentError("CONFIG")
            attempted_ms.append(elapsed)
            timings = [call.get("transport_elapsed_ms") for call in observed]
            if all(type(value) in {int,float} and math.isfinite(value) and value>=0 for value in timings):
                pure_transport_ms.append(sum(timings))
            if status in {"content", "missing"}:
                successful_ms.append(elapsed)
        def total(field):
            values = [call.get(field) for call in calls]
            return sum(values) if all(type(value) is int and value >= 0 for value in values) else None
        attempted = sum(counts[key] for key in ("content", "missing", "failed"))
        result[strategy] = {"outcomes": counts, "graphql_calls_including_shared_reference": len(calls),
            "http_request_attempts_including_shared_reference": sum(call.get("http_request_started") is True for call in calls),
            "rate_cost_including_shared_reference": total("cost"),
            "wire_bytes_including_shared_reference": total("wire_bytes"),
            "decoded_body_bytes_including_shared_reference": total("decoded_body_bytes"),
            "normalized_json_bytes_including_shared_reference": total("decoded_json_bytes"),
            "partial_response_trials": partial, "failure_rate_of_attempted": counts["failed"]/attempted if attempted else None,
            "not_run_trials": counts["not_run"], "successful_latency_samples": len(successful_ms),
            "successful_end_to_end_p50_ms": _percentile(successful_ms, .5),
            "successful_end_to_end_p95_ms": _percentile(successful_ms, .95),
            "all_attempted_end_to_end_p50_ms": _percentile(attempted_ms, .5),
            "all_attempted_end_to_end_p95_ms": _percentile(attempted_ms, .95)}
        result[strategy]["transport_latency_samples"]=len(pure_transport_ms)
        result[strategy]["transport_p50_ms"]=_percentile(pure_transport_ms,.5)
        result[strategy]["transport_p95_ms"]=_percentile(pure_transport_ms,.95)
    reference_calls=[call for trial in trials for call in trial.get("shared_reference_requests", [])]
    def reference_total(field):
        values=[call.get(field) for call in reference_calls]
        return sum(values) if all(type(value) is int and value>=0 for value in values) else None
    reference_totals={"graphql_calls":len(reference_calls),
        "http_request_attempts":sum(call.get("http_request_started") is True for call in reference_calls),
        "rate_cost":reference_total("cost"),"wire_bytes":reference_total("wire_bytes"),
        "decoded_body_bytes":reference_total("decoded_body_bytes"),
        "normalized_json_bytes":reference_total("decoded_json_bytes")}
    failed_trials = sum(trial.get("status") == "reference_failed" or any(
        item.get("status") == "failed" for item in trial.get("strategies", {}).values()) for trial in trials)
    return {"strategies": result,"shared_reference_totals":reference_totals,
        "reference_failed_trials": sum(trial.get("status") == "reference_failed" for trial in trials),
        "failed_trials": failed_trials,
        "failure_rate_of_all_trials": failed_trials/len(trials) if trials else None,
        "latency_scope": "shared reference plus full strategy, including diagnostic normalized-JSON accounting; separate transport timings exclude that extra accounting; failures/not-run/partial reported",
        "cost_scope": "each executed alternative includes its shared reference; standalone shared_reference_totals includes all reference requests even when neither strategy runs; do not sum these overlapping totals",
        "order_scope": "alternates across repositories and repeats; per-repository balance requires even repeats",
        "paired_comparable_trials": sum(trial.get("status") == "comparable" for trial in trials),
        "canonical_content_equal_trials": sum(trial.get("canonical_content_equal") is True for trial in trials),
        "content_path_branch_equal_trials": sum(trial.get("content_path_branch_equal") is True for trial in trials),
        "media_base_equal_trials": sum(trial.get("media_base_equal") is True for trial in trials)}


def publish_new_report(path: Path, value):
    """Atomic non-overwrite report; aborted experiments preserve prior evidence."""
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2).encode()
    if len(raw) > 1024*1024:
        raise ReadmeExperimentError("BOUND")
    target = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".readme-experiment-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
