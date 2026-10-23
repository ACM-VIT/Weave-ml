"""Uncached, commit-pinned README comparison. CLI plans only; see the runbook."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acquisition.github_graphql_client import (
    FetchedReadme, GitHubGraphQLClient, build_readme_base_url,
)
from acquisition.graphql_queries import GET_README_QUERY, GET_README_REF_QUERY

MAX_REPOSITORIES = 10
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_CALLS = 40
REF_QUERY = GET_README_REF_QUERY.replace("    id\n", "    id\n    visibility\n")
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]


def load_sample(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    repos = value.get("repositories") if isinstance(value, dict) else None
    if not isinstance(repos, list) or not 1 <= len(repos) <= MAX_REPOSITORIES:
        raise ValueError("Sample requires 1 to 10 repository names")
    pattern = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9_.-]{1,100}"
    if any(not isinstance(repo, str) or re.fullmatch(pattern, repo) is None for repo in repos):
        raise ValueError("Sample contains an invalid repository name")
    if len({repo.casefold() for repo in repos}) != len(repos):
        raise ValueError("Sample contains duplicate repositories")
    return repos


class Measurement:
    """Bound injected transports; reports never retain response bodies or errors."""

    def __init__(self, transport: Transport, *, seconds: float = 180.0) -> None:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("A finite positive deadline is required")
        self.transport = transport
        self.deadline = time.monotonic() + seconds
        self.calls: list[dict[str, Any]] = []

    def execute(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
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
                raise RuntimeError("Rate-limit reserve reached")
            if time.monotonic() >= self.deadline:
                raise RuntimeError("Comparison deadline exceeded")
            record["ok"] = True
            return response
        finally:
            record["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 3)


def _artifact(value: FetchedReadme) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(str(value).encode("utf-8")).hexdigest(),
            "utf8_bytes": len(str(value).encode("utf-8")),
            "source_path": value.source_path, "default_branch": value.default_branch,
            "base_url": value.base_url}


def compare_repository(repository: str, measurement: Measurement) -> dict[str, Any]:
    """Exercise real source parsers, with no acquisition/content caches involved.

    A caller-provided transport must enforce its own streaming response limit,
    socket deadline, endpoint/credential policy and sample approval. The CLI
    deliberately provides no live transport before human validation is approved.
    """
    if re.fullmatch(r"[A-Za-z0-9-]+/[A-Za-z0-9_.-]+", repository) is None:
        raise ValueError("Invalid repository name")
    owner, name = repository.split("/")
    variables = {"owner": owner, "name": name}
    reference_start = len(measurement.calls)
    response = measurement.execute(REF_QUERY, variables)
    data = response.get("data")
    repo = data.get("repository") if isinstance(data, dict) else None
    if response.get("errors") or not isinstance(repo, dict) or repo.get("visibility") != "PUBLIC":
        return {"repository": repository, "status": "unresolved_or_nonpublic"}
    ref = repo.get("defaultBranchRef")
    target = ref.get("target") if isinstance(ref, dict) else None
    commit = target.get("oid") if isinstance(target, dict) else None
    branch = ref.get("name") if isinstance(ref, dict) else None
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None or not isinstance(branch, str) or not branch:
        return {"repository": repository, "status": "unresolved_commit"}

    client = GitHubGraphQLClient(token="", max_retries=0, sleep_on_rate_limit=False)
    client.execute = measurement.execute
    result: dict[str, Any] = {"repository": repository, "commit": commit,
                              "shared_reference_requests": list(measurement.calls[reference_start:]),
                              "strategies": {}}
    try:
        for strategy in ("aliases", "metadata"):
            start = len(measurement.calls)
            try:
                if strategy == "aliases":
                    query = GET_README_QUERY.replace("HEAD:", f"{commit}:")
                    query = query.rstrip().removesuffix("}") + "\n rateLimit { cost remaining resetAt }\n}"
                    raw = measurement.execute(query, variables)
                    value = client._get_readme_legacy(owner, name, response=raw)
                else:
                    manifest = client._readme_manifest(owner, name, commit)
                    text = client._readme_blob(owner, name, manifest.blob_oid, manifest.byte_size) if manifest.blob_oid else ""
                    value = FetchedReadme(text, source_path=manifest.source_path,
                                         default_branch=branch if text else None,
                                         base_url=build_readme_base_url(owner, name, branch, manifest.source_path, source_commit=commit) if text else None)
                result["strategies"][strategy] = {"status": "content" if value else "missing", **_artifact(value)}
            except Exception:
                # Provider exceptions can contain raw data; persist only outcome.
                result["strategies"][strategy] = {"status": "failed"}
            result["strategies"][strategy]["requests"] = list(measurement.calls[start:])
    finally:
        client.session.close()
    left, right = (result["strategies"][key] for key in ("aliases", "metadata"))
    comparable = left["status"] != "failed" and right["status"] != "failed"
    result["status"] = "comparable" if comparable else "inconclusive"
    result["content_path_branch_equal"] = comparable and all(left.get(key) == right.get(key) for key in ("status", "sha256", "source_path", "default_branch"))
    result["media_base_equal"] = comparable and left.get("base_url") == right.get("base_url")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=Path(__file__).resolve().parents[1] / "tests/fixtures/readme_strategy_sample.json")
    args = parser.parse_args(argv)
    repos = load_sample(args.sample)
    print(json.dumps({"mode": "plan_only", "repositories": repos,
                      "max_graphql_requests": 4 * len(repos),
                      "max_decoded_response_bytes": MAX_RESPONSE_BYTES,
                      "session_deadline_seconds": 180,
                      "live_validation": "pending human-approved sample and transport"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
