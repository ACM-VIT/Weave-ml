"""Uncached, commit-pinned README comparison; default CLI only plans a sample."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def _source_hashes():
    root=Path(__file__).resolve().parents[1]
    names=subprocess.check_output(["git","--no-lazy-fetch","ls-files","-z","--","*.py","pyproject.toml","uv.lock"],
        cwd=root,env={**os.environ,"GIT_NO_LAZY_FETCH":"1","GIT_TERMINAL_PROMPT":"0"},text=True,timeout=15).split("\0")
    paths={name for name in names if name}
    paths.update(("scripts/compare_readme_strategies.py","evaluation/readme_experiment.py"))
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sorted(paths)}


# Worker source identity is captured before importing any project parser.
_IMPORTED_SOURCE_SHA256 = None
if __name__ == "__main__" and "--worker" in sys.argv:
    try:
        _IMPORTED_SOURCE_SHA256 = _source_hashes()
    except Exception:
        _IMPORTED_SOURCE_SHA256 = {}


def _source_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


from acquisition.github_graphql_client import (
    FetchedReadme, GitHubGraphQLClient, build_readme_base_url,
)
from acquisition.graphql_queries import GET_README_QUERY, GET_README_REF_QUERY
from evaluation.readme_experiment import (LiveGitHubTransport, ReadmeExperimentError, comparison_trials, summarize_trials, publish_new_report, read_sample_bytes, decode_sample_bytes, start_worker_watchdog)

MAX_REPOSITORIES = 10
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_CALLS = 40
REF_QUERY = GET_README_REF_QUERY.replace("    id\n", "    id\n    visibility\n")
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]


def _load_sample_snapshot(path: Path):
    raw=read_sample_bytes(path)
    value = decode_sample_bytes(raw)
    if not isinstance(value,dict) or not set(value)<={"status","repositories"}:
        raise ValueError("Sample contains unsupported fields")
    if "status" in value and (not isinstance(value["status"],str) or len(value["status"])>128):
        raise ValueError("Sample status is invalid")
    repos = value.get("repositories") if isinstance(value, dict) else None
    if not isinstance(repos, list) or not 1 <= len(repos) <= MAX_REPOSITORIES:
        raise ValueError("Sample requires 1 to 10 repository names")
    pattern = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9_.-]{1,100}"
    if any(not isinstance(repo, str) or re.fullmatch(pattern, repo) is None for repo in repos):
        raise ValueError("Sample contains an invalid repository name")
    if len({repo.casefold() for repo in repos}) != len(repos):
        raise ValueError("Sample contains duplicate repositories")
    return repos,hashlib.sha256(raw).hexdigest()


def load_sample(path: Path) -> list[str]:
    return _load_sample_snapshot(path)[0]


class Measurement:
    """Bound injected transports; reports never retain response bodies or errors."""

    def __init__(self, transport: Transport, *, seconds: float = 180.0) -> None:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("A finite positive deadline is required")
        self.transport = transport
        self.deadline = time.monotonic() + seconds
        self.calls: list[dict[str, Any]] = []
        self.stopped = False

    def execute(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        if self.stopped:
            raise RuntimeError("Comparison stopped after rate-limit reserve")
        if len(self.calls) >= MAX_CALLS or time.monotonic() >= self.deadline:
            raise RuntimeError("Comparison budget exhausted")
        record: dict[str, Any] = {"ok": False}
        self.calls.append(record)  # Failed attempts consume budget too.
        start = time.perf_counter()
        try:
            response = self.transport(query, variables)
            if not isinstance(response, dict):
                raise ValueError("Invalid GraphQL envelope")
            # This is normalized decoded JSON size, not compressed wire bytes.
            size = len(json.dumps(response, ensure_ascii=False).encode("utf-8"))
            if size > MAX_RESPONSE_BYTES:
                raise ValueError("Response exceeds comparison bound")
            data = response.get("data")
            rate = data.get("rateLimit") if isinstance(data, dict) else None
            cost = rate.get("cost") if isinstance(rate, dict) else None
            remaining = rate.get("remaining") if isinstance(rate, dict) else None
            record.update(decoded_json_bytes=size, graphql_errors=bool(response.get("errors")),
                          cost=cost if type(cost) is int and cost >= 0 else None)
            if type(remaining) is int and remaining < 100:
                self.stopped = True
                raise RuntimeError("Rate-limit reserve reached")
            if time.monotonic() >= self.deadline:
                raise RuntimeError("Comparison deadline exceeded")
            record["ok"] = True
            return response
        finally:
            observed = getattr(self.transport,"last_measurement",None)
            if isinstance(observed,dict):
                for key in ("wire_bytes","decoded_body_bytes","http_status"):
                    value=observed.get(key)
                    if type(value) is int and value>=0:
                        record[key]=value
                duration=observed.get("transport_elapsed_ms")
                if type(duration) in {int,float} and math.isfinite(duration) and duration>=0:
                    record["transport_elapsed_ms"]=duration
                record["http_request_started"]=observed.get("http_request_started") is True
            record["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 3)


def _artifact(value: FetchedReadme) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(str(value).encode("utf-8")).hexdigest(),
            "utf8_bytes": len(str(value).encode("utf-8")),
            "source_path": value.source_path, "default_branch": value.default_branch,
            "base_url": value.base_url}


def compare_repository(repository: str, measurement: Measurement, *, strategies=("aliases","metadata")) -> dict[str, Any]:
    """Exercise real source parsers, with no acquisition/content caches involved.

    A caller-provided transport must enforce its own streaming response limit,
    socket deadline, endpoint/credential policy and sample approval. The normal
    CLI remains plan-only; execution requires an explicit approved-sample flag.
    """
    if re.fullmatch(r"[A-Za-z0-9-]+/[A-Za-z0-9_.-]+", repository) is None:
        raise ValueError("Invalid repository name")
    if type(strategies) is not tuple or len(strategies)!=2 or set(strategies)!={"aliases","metadata"}:
        raise ValueError("comparison requires both strategies once")
    owner, name = repository.split("/")
    variables = {"owner": owner, "name": name}
    reference_start = len(measurement.calls)
    reference_started = time.perf_counter()
    try:
        response = measurement.execute(REF_QUERY, variables)
    except Exception:
        # Reference transport failures are measured trial outcomes too. Keep no
        # raw exception and never attempt content without a public pinned ref.
        return {"repository": repository, "status": "reference_failed",
            "shared_reference_requests": list(measurement.calls[reference_start:]),
            "shared_reference_elapsed_ms": (time.perf_counter()-reference_started)*1000}
    shared_reference_requests = list(measurement.calls[reference_start:])
    reference_elapsed_ms = (time.perf_counter()-reference_started)*1000
    data = response.get("data")
    repo = data.get("repository") if isinstance(data, dict) else None
    if response.get("errors"):
        return {"repository": repository, "status": "reference_failed",
                "shared_reference_requests": shared_reference_requests,"shared_reference_elapsed_ms":reference_elapsed_ms}
    if not isinstance(repo, dict) or repo.get("visibility") != "PUBLIC":
        return {"repository": repository, "status": "unresolved_or_nonpublic",
                "shared_reference_requests": shared_reference_requests,"shared_reference_elapsed_ms":reference_elapsed_ms}
    ref = repo.get("defaultBranchRef")
    target = ref.get("target") if isinstance(ref, dict) else None
    commit = target.get("oid") if isinstance(target, dict) else None
    branch = ref.get("name") if isinstance(ref, dict) else None
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None or not isinstance(branch, str) or not branch:
        return {"repository": repository, "status": "unresolved_commit",
                "shared_reference_requests": shared_reference_requests,"shared_reference_elapsed_ms":reference_elapsed_ms}

    client = GitHubGraphQLClient(token="", max_retries=0, sleep_on_rate_limit=False)
    client.execute = measurement.execute
    result: dict[str, Any] = {"repository": repository, "commit": commit,
                              "shared_reference_requests": shared_reference_requests, "shared_reference_elapsed_ms":reference_elapsed_ms,
                              "strategies": {}}
    try:
        for strategy in strategies:
            start = len(measurement.calls)
            strategy_started = time.perf_counter()
            try:
                if strategy == "aliases":
                    query = GET_README_QUERY.replace("HEAD:", f"{commit}:")
                    query = query.rstrip().removesuffix("}") + "\n rateLimit { cost remaining resetAt }\n}"
                    raw = measurement.execute(query, variables)
                    value = client._get_readme_legacy(owner, name, response=raw)
                    if value and value.default_branch != branch:
                        result["reference_changed"] = True
                        raise ValueError("Default branch changed during comparison")
                else:
                    manifest = client._readme_manifest(owner, name, commit)
                    text = client._readme_blob(owner, name, manifest.blob_oid, manifest.byte_size) if manifest.blob_oid else ""
                    value = FetchedReadme(text, source_path=manifest.source_path,
                                         default_branch=branch,
                                         base_url=build_readme_base_url(owner, name, branch, manifest.source_path, source_commit=commit) if text else None)
                result["strategies"][strategy] = {"status": "content" if value else "missing", **_artifact(value)}
            except Exception:
                # Provider exceptions can contain raw data; persist only outcome.
                result["strategies"][strategy] = {"status": "failed"}
            result["strategies"][strategy]["requests"] = list(measurement.calls[start:])
            result["strategies"][strategy]["elapsed_ms"] = (time.perf_counter()-strategy_started)*1000
    finally:
        client.session.close()
    left, right = (result["strategies"][key] for key in ("aliases", "metadata"))
    comparable = left["status"] != "failed" and right["status"] != "failed"
    result["status"] = "comparable" if comparable else "inconclusive"
    result["canonical_content_equal"] = comparable and all(left.get(key)==right.get(key) for key in ("status","sha256"))
    result["content_path_branch_equal"] = comparable and all(left.get(key) == right.get(key) for key in ("status", "sha256", "source_path", "default_branch"))
    result["media_base_equal"] = comparable and left.get("base_url") == right.get("base_url")
    return result




def _live_worker(args,repos,sample_sha256):
    # This process belongs only to one explicitly invoked experiment. Hard exit
    # bounds native DNS/TLS/header work that socket timeouts alone cannot prove
    # stopped (e.g. a slow peer sending partial headers). No report file is ever
    # written by the child. The parent also owns an independent process timeout.
    timer=start_worker_watchdog()
    try:
        before=_source_hashes()
        if (_IMPORTED_SOURCE_SHA256 is not None and _IMPORTED_SOURCE_SHA256 != before
                or getattr(args,"expected_source_sha256",None) is not None
                and _source_digest(before) != args.expected_source_sha256):
            raise ReadmeExperimentError("CONFIG")
        token=os.environ.get(args.token_env,"")
        transport=LiveGitHubTransport(token)
        transport.verify_identity()
        measurement=Measurement(transport)
        trials=comparison_trials(repos,args.repeats,measurement,compare_repository)
        if before!=_source_hashes():raise ReadmeExperimentError("CONFIG")
        report={"schema":"readme-strategy-comparison-v2","mode":"executed_approved_sample",
            "source_sha256":before,"sample_sha256":sample_sha256,
            "repeats":args.repeats,"physical_graphql_attempts":len(measurement.calls),
            "identity_http_checks":1,"transport_observations":transport.observations,
            "connection_policy":"new direct HTTPS connection per HTTP request; no connection pooling or retries",
            "trials":trials,"summary":summarize_trials(trials),
            "scope":"measured public sample only; no strategy activation, representative-quality or production speedup inference",
            "strategy_decision":"not selected; review actual sample evidence and rollback separately"}
        wire=json.dumps(report,allow_nan=False,separators=(",",":"))
        if len(wire.encode())>1024*1024:raise ReadmeExperimentError("BOUND")
        print(wire)
        return 0
    except Exception:
        print("README_EXPERIMENT_UNAVAILABLE",file=sys.stderr)
        return 2
    finally:
        timer.cancel()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=Path(__file__).resolve().parents[1] / "tests/fixtures/readme_strategy_sample.json")
    parser.add_argument("--repeats",type=int,default=1)
    parser.add_argument("--execute-approved",action="store_true",help="execute only with actual authority for this public sample/token; default plans only")
    parser.add_argument("--token-env",default="GITHUB_README_COMPARISON_TOKEN",help="explicit approved developer credential variable; no ambient service-token fallback")
    parser.add_argument("--output",type=Path,help="new private report path; existing evidence is never overwritten")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--sample-sha256",help=argparse.SUPPRESS)
    parser.add_argument("--expected-source-sha256",help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        repos,sample_sha256 = _load_sample_snapshot(args.sample)
    except (ValueError,OSError,ReadmeExperimentError):
        parser.error("invalid bounded sample")
    if not 1<=args.repeats<=5 or len(repos)*args.repeats>10:
        parser.error("at most ten complete trials across1..5repeats")
    if re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}",args.token_env) is None:
        parser.error("token variable name is invalid")
    if args.worker:
        if not args.execute_approved or args.sample_sha256!=sample_sha256 or not args.expected_source_sha256:
            parser.error("worker requires exact frozen sample execution")
        return _live_worker(args,repos,sample_sha256)
    if not args.execute_approved:
        print(json.dumps({"mode": "plan_only", "repositories": repos,"repeats":args.repeats,
            "max_graphql_requests": 4*len(repos)*args.repeats,"max_identity_checks":1,
            "max_decoded_response_bytes": MAX_RESPONSE_BYTES,"session_deadline_seconds":180,
            "live_validation":"pending actual authority for the explicit public sample/developer token"},indent=2))
        return 0
    command=[sys.executable,str(Path(__file__).resolve()),"--worker","--execute-approved","--sample",str(args.sample.absolute()),
        "--repeats",str(args.repeats),"--token-env",args.token_env,"--sample-sha256",sample_sha256]
    try:
        if args.output is not None:
            try:args.output.lstat()
            except FileNotFoundError:pass
            else:raise FileExistsError()
            if not args.output.parent.is_dir():raise ReadmeExperimentError("CONFIG")
        expected_source=_source_hashes()
        command.extend(["--expected-source-sha256",_source_digest(expected_source)])
        child=subprocess.run(command,capture_output=True,text=True,timeout=180,check=False)
        if child.returncode or len(child.stdout.encode())>1024*1024:
            raise ReadmeExperimentError()
        report=json.loads(child.stdout)
        if report.get("source_sha256") != expected_source:
            raise ReadmeExperimentError("CONFIG")
        if args.output:publish_new_report(args.output,report)
        else:print(json.dumps(report,indent=2,allow_nan=False))
        return 0
    except (Exception,KeyboardInterrupt):
        print("README_EXPERIMENT_UNAVAILABLE",file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
