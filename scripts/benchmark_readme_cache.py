"""Measure real loopback GraphQL/ref/blob cache work; never targets GitHub."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import statistics
import sys
import threading
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from acquisition.github_graphql_client import GitHubGraphQLClient
from acquisition.graphql_queries import README_CANDIDATES
import acquisition.readme_cache as caches
import acquisition.repository_enricher as enrichment


def blob_oid(text: str) -> str:
    raw = text.encode("utf-8")
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


@contextmanager
def isolated_caches():
    """Standalone diagnostic only; never call inside a live service process."""
    previous = {name: getattr(caches, name) for name in (
        "README_MANIFEST_CACHE", "README_BLOB_CACHE", "README_DOCUMENT_CACHE")}
    previous_document = enrichment.README_DOCUMENT_CACHE
    current = {
        "README_MANIFEST_CACHE": caches.BoundedReadmeCache(max_entries=512, max_bytes=2 * 1024**2),
        "README_BLOB_CACHE": caches.BoundedReadmeCache(max_entries=256, max_bytes=64 * 1024**2),
        "README_DOCUMENT_CACHE": caches.BoundedReadmeCache(max_entries=256, max_bytes=32 * 1024**2),
    }
    try:
        for name, value in current.items():
            setattr(caches, name, value)
        enrichment.README_DOCUMENT_CACHE = current["README_DOCUMENT_CACHE"]
        yield current
    finally:
        for name, value in previous.items():
            setattr(caches, name, value)
        enrichment.README_DOCUMENT_CACHE = previous_document


class GraphQLFixture:
    """Synthetic immutable Git objects with actual HTTP transfer accounting."""

    def __init__(self, markdown: str = "# Fixture\n\n![Preview](images/demo.png)\n\né界 canonical text.", *, delay: float = 0):
        self.lock = threading.RLock()
        self.delay = delay
        self.branch = "main"
        self.revision = 0
        self.commits: dict[str, dict[str, str]] = {}
        self.calls: Counter[str] = Counter()
        self.response_bytes: Counter[str] = Counter()
        self.request_bytes: Counter[str] = Counter()
        self.failures: Counter[str] = Counter()
        self.denied = False
        self.partial_manifest = False
        self.tamper_blob = False
        self.truncate_blob = False
        self.blob_started = threading.Event()
        self.release_blob = threading.Event()
        self.release_blob.set()
        self.set_files({"README.md": markdown})

    def set_files(self, files: dict[str, str]):
        with self.lock:
            self.revision += 1
            encoded = json.dumps([self.revision, sorted(files.items())], ensure_ascii=False).encode()
            self.commit = hashlib.sha1(encoded).hexdigest()
            self.commits[self.commit] = dict(files)

    def _response(self, payload):
        query = payload["query"]
        variables = payload.get("variables") or {}
        stage = "ref" if "GetReadmeRef(" in query else "blob" if "GetReadmeBlob(" in query else "manifest"
        with self.lock:
            self.calls[stage] += 1
            if self.denied:
                return stage, 403, {"message": "fixture access denied"}
            if self.failures[stage]:
                self.failures[stage] -= 1
                return stage, 503, {"message": "fixture transient failure"}
            if stage == "ref":
                identity = variables.get("owner", "") + "/" + variables.get("name", "")
                repository = {"id": "R_fixture_" + hashlib.sha1(identity.encode()).hexdigest(),
                    "defaultBranchRef": {"name": self.branch, "target": {"oid": self.commit}}}
            elif stage == "manifest":
                commit = re.search(r'object\(expression: "([a-f0-9]+):', query).group(1)
                files = self.commits[commit]
                repository = {alias: None if path not in files else {
                    "oid": blob_oid(files[path]), "byteSize": len(files[path].encode()), "isBinary": False,
                } for alias, path in README_CANDIDATES}
                if self.partial_manifest:
                    repository.pop(README_CANDIDATES[0][0])
            else:
                oid = variables["oid"]
                matches = [text for files in self.commits.values() for text in files.values() if blob_oid(text) == oid]
                assert matches, "client requested an unknown immutable blob"
                text = matches[0]
                returned = text[:-1] if self.truncate_blob else "!" + text[1:] if self.tamper_blob else text
                repository = {"object": {"oid": oid, "isBinary": False,
                    "byteSize": len(text.encode()), "text": returned}}
            result = {"data": {"repository": repository}}
            if stage == "manifest" and self.partial_manifest:
                result["errors"] = [{"message": "candidate unresolved", "path": ["repository", README_CANDIDATES[0][0]]}]
            return stage, 200, result

    def __enter__(self):
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def do_POST(self):
                length = int(self.headers.get("content-length", "0"))
                assert 0 < length < 64_000
                raw = self.rfile.read(length)
                stage, status, result = fixture._response(json.loads(raw))
                if stage == "blob":
                    fixture.blob_started.set()
                    assert fixture.release_blob.wait(5), "fixture blob gate was not released"
                if fixture.delay:
                    time.sleep(fixture.delay)
                encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode()
                with fixture.lock:
                    fixture.request_bytes[stage] += len(raw)
                    fixture.response_bytes[stage] += len(encoded)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()
        self.origin = f"http://127.0.0.1:{self.server.server_port}/graphql"
        return self

    def __exit__(self, *_args):
        self.release_blob.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        assert not self.thread.is_alive()

    def client(self, token="fixture-readme-token"):
        session = requests.Session()
        session.trust_env = False
        return GitHubGraphQLClient(token=token, base_url=self.origin, session=session, max_retries=0,
                                   timeout_seconds=3, deadline=time.monotonic() + 10)


def run_benchmark(*, iterations: int = 10, delay: float = .005):
    if not 2 <= iterations <= 100 or not 0 <= delay <= .1:
        raise ValueError("Use 2..100 iterations and 0..100ms injected delay")
    markdown = "# Synthetic cache workload\n\n![Diagram](assets/diagram.png)\n\n" + "Canonical Unicode é界 documentation and relative media.\n" * 1000
    reports = {}
    for mode in ("uncached", "cached"):
        durations = []
        with isolated_caches() as stores, GraphQLFixture(markdown, delay=delay) as fixture:
            for _ in range(iterations):
                # Clearing retention is the explicit uncached control. The
                # client, ref/manifest/blob protocol and actual HTTP are equal.
                if mode == "uncached":
                    for store in stores.values():
                        store.max_bytes = 0
                client = fixture.client()
                try:
                    started = time.perf_counter()
                    readme = client.get_readme("fixture", "readme-cache")
                    document = enrichment.RepositoryEnricher(client)._process_readme(readme,
                        source_path=readme.source_path, default_branch=readme.default_branch, base_url=readme.base_url)
                    durations.append((time.perf_counter() - started) * 1000)
                    assert document.raw_markdown == markdown
                finally:
                    client.session.close()
            snapshots = {}
            for name, store in stores.items():
                snapshot = store.snapshot()
                lookups = sum(snapshot.get(key, 0) for key in ("hits", "misses", "coalesced"))
                snapshots[name] = {**snapshot, "stored_value_hit_ratio": snapshot.get("hits", 0) / lookups if lookups else None}
            reports[mode] = {"requests": dict(fixture.calls), "response_body_bytes": sum(fixture.response_bytes.values()),
                "request_body_bytes": sum(fixture.request_bytes.values()), "cold_ms": durations[0],
                "warm_median_ms": statistics.median(durations[1:]),
                "accounted_cache_payload_bytes": sum(store.snapshot()["bytes"] for store in stores.values()),
                "cache": snapshots}
    saved_calls = sum(reports["uncached"]["requests"].values()) - sum(reports["cached"]["requests"].values())
    return {"scope": "synthetic loopback HTTP; no GitHub or production speedup claim", "iterations": iterations,
        "injected_delay_ms": delay * 1000, "readme_utf8_bytes": len(markdown.encode()), "results": reports,
        "avoided_requests": saved_calls,
        "avoided_response_body_bytes": reports["uncached"]["response_body_bytes"] - reports["cached"]["response_body_bytes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--delay-ms", type=float, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(iterations=args.iterations, delay=args.delay_ms / 1000)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
