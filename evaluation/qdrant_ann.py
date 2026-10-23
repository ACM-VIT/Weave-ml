"""Offline HNSW measurement using only a pinned, newly owned local server."""
from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import uuid

import numpy as np

from evaluation import embedding_registry as registry_module
from evaluation import embedding_artifacts as artifact_module
from evaluation.embedding_artifacts import Artifact, JSON_LIMIT, decode_json, digest, encode_json, load_artifact, publish_new_json, read_regular

IMAGE = "qdrant/qdrant@sha256:75eab8c4ba42096724fdcfde8b4de0b5713d529dde32f285a1f86fdcb2c9e50c"
RPC_LIMIT = 4 * 1024 * 1024
STORAGE_LIMIT = 256 * 1024 * 1024
VECTOR_NAME = "repo_embedding"
LABEL = "weave.embedding-ann-owner"
INDEX_FILES = {"graph.bin", "links.bin", "links_compressed.bin", "links_comp_vec.bin", "hnsw_config.json"}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise ValueError("owned server redirected a request")


class OwnedQdrant(AbstractContextManager):
    """One pinned container and private storage; external target is impossible."""
    def __init__(self, *, total_seconds: float = 180):
        if not 10 <= total_seconds <= 300:
            raise ValueError("owned server deadline out of bounds")
        # Resolve once and pin every daemon operation to a local Unix socket.
        # Docker context/SSH/TLS environment must never redirect this experiment.
        host = os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock")
        parsed = urlsplit(host)
        if (parsed.scheme != "unix" or parsed.netloc or not parsed.path.startswith("/")
                or parsed.query or parsed.fragment or host != "unix://" + parsed.path
                or any(ord(character) < 32 for character in host)):
            raise ValueError("ANN measurement requires an explicit local Docker Unix socket")
        self._docker_prefix = ["docker", "--host", host]
        self._docker_env = {key: value for key, value in os.environ.items()
                            if key not in {"DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"}}
        self.run_id = uuid.uuid4().hex
        self.name = "weave-embedding-ann-" + self.run_id[:16]
        self.deadline = time.monotonic() + total_seconds
        self.root: Path | None = None
        self.storage: Path | None = None
        self.url: str | None = None
        self.stopped = False
        self.created = False
        self.server = {}
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def remaining(self, maximum=40):
        seconds = self.deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("owned ANN experiment deadline exceeded")
        return min(maximum, seconds)

    def _command(self, arguments: list[str], timeout=40, *, check=True):
        result = subprocess.run([*self._docker_prefix, *arguments], timeout=timeout,
                                capture_output=True, text=True, env=self._docker_env)
        if check and result.returncode:
            raise RuntimeError("owned Docker operation failed: " + arguments[0])
        return result

    def _owned_info(self):
        result = self._command(["inspect", self.name], timeout=10, check=False)
        if result.returncode:
            if "no such object" in result.stderr.casefold() or "no such container" in result.stderr.casefold():
                return None
            raise RuntimeError("cannot prove owned container cleanup state")
        info = decode_json(result.stdout.encode())[0]
        if info["Config"].get("Labels", {}).get(LABEL) != self.run_id:
            raise RuntimeError("container ownership label mismatch")
        return info

    def __enter__(self):
        try:
            if not hasattr(os, "getuid") or not hasattr(os.stat_result, "st_blocks"):
                raise ValueError("owned ANN storage measurements require POSIX uid and st_blocks")
            self._command(["image", "inspect", IMAGE], timeout=self.remaining())
            self.root = Path(tempfile.mkdtemp(prefix="weave-embedding-ann-"))
            self.storage = self.root / "storage"
            self.storage.mkdir()
            (self.root / "owner").write_text(self.run_id)
            # Mark before starting: an interrupted docker client can still have
            # created the uniquely labelled daemon-side container.
            self.created = True
            self._command(["run", "--pull=never", "-d", "--name", self.name,
                "--label", f"{LABEL}={self.run_id}", "--user", f"{os.getuid()}:{os.getgid()}",
                "--cpus", "2", "--memory", "1g", "--pids-limit", "256",
                "-p", "127.0.0.1::6333", "--mount", f"type=bind,src={self.storage},dst=/qdrant/storage",
                "-e", "QDRANT__TELEMETRY_DISABLED=true",
                "-e", "QDRANT__STORAGE__SNAPSHOTS_PATH=/qdrant/storage/snapshots",
                "-e", "QDRANT__STORAGE__TEMP_PATH=/qdrant/storage/tmp", IMAGE], timeout=self.remaining())
            info = self._owned_info()
            bindings = info["NetworkSettings"]["Ports"]["6333/tcp"]
            if len(bindings) != 1 or bindings[0]["HostIp"] != "127.0.0.1":
                raise ValueError("owned server is not bound only to loopback")
            port = bindings[0]["HostPort"]
            if not port.isdigit() or not 1 <= int(port) <= 65535:
                raise ValueError("owned server port is invalid")
            self.url = f"http://127.0.0.1:{port}"
            startup_deadline = time.monotonic() + self.remaining(30)
            while True:
                try:
                    self.server = self.api("/")
                    if self.server.get("version") != "1.18.2":
                        raise ValueError("owned server version mismatch")
                    return self
                except (urllib.error.URLError, TimeoutError, OSError):
                    if time.monotonic() >= startup_deadline:
                        raise TimeoutError("owned Qdrant startup deadline exceeded")
                    time.sleep(.1)
        except BaseException as error:
            try:
                self.cleanup()
            except Exception as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note(str(cleanup_error))
            raise

    def api(self, path: str, body=None, method=None):
        if self.url is None or not path.startswith("/") or path.startswith("//"):
            raise ValueError("invalid owned server path")
        raw = None if body is None else encode_json(body)
        if raw is not None and len(raw) > RPC_LIMIT:
            raise ValueError("ANN request byte limit exceeded")
        request = urllib.request.Request(self.url + path, data=raw, method=method,
                                        headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=self.remaining(5)) as response:
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdigit() or int(length) > RPC_LIMIT):
                raise ValueError("ANN response byte limit exceeded")
            pieces, size = [], 0
            while True:
                self.remaining()
                piece = response.read1(min(65536, RPC_LIMIT + 1 - size))
                if not piece:
                    break
                size += len(piece)
                if size > RPC_LIMIT:
                    raise ValueError("ANN response byte limit exceeded")
                pieces.append(piece)
            result = b"".join(pieces)
        return decode_json(result)

    def stop(self):
        if not self.stopped:
            info = self._owned_info()
            if info is None:
                raise RuntimeError("owned server disappeared before storage measurement")
            self._command(["stop", "--time", "10", self.name], timeout=20)
            if self._owned_info()["State"]["Running"]:
                raise RuntimeError("owned server is still running")
            self.stopped = True

    def cleanup(self):
        if self.created:
            info = self._owned_info()
            if info is not None:
                self._command(["rm", "-f", self.name], timeout=20)
            self.created = False
        if self.root is not None:
            if self.root.is_symlink() or read_regular(self.root / "owner", 128).decode() != self.run_id:
                raise RuntimeError("private storage ownership witness mismatch")
            shutil.rmtree(self.root)
            self.root = None

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.cleanup()
        except Exception as cleanup_error:
            if exc is None:
                raise
            if hasattr(exc, "add_note"):
                exc.add_note(str(cleanup_error))
        return False


def telemetry(telemetry_json: dict, collection: str) -> dict:
    collections = telemetry_json.get("result", {}).get("collections", {}).get("collections", [])
    matches = [row for row in collections if row.get("id") == collection]
    if len(matches) != 1:
        raise ValueError("owned collection telemetry is missing/ambiguous")
    counters = {"unfiltered_hnsw": 0, "unfiltered_plain": 0, "unfiltered_exact": 0}
    indexed_vectors = 0
    segments = 0

    def visit(value):
        nonlocal indexed_vectors, segments
        if isinstance(value, dict):
            if "vector_index_searches" in value:
                segments += 1
                details = value.get("info", {}).get("vector_data", {}).get(VECTOR_NAME, {})
                indexed = details.get("num_indexed_vectors", 0)
                if type(indexed) is not int or indexed < 0:
                    raise ValueError("invalid indexed vector telemetry")
                indexed_vectors += indexed
                for index in value["vector_index_searches"]:
                    if index.get("index_name") != VECTOR_NAME:
                        continue
                    for key in counters:
                        count = index.get(key, {}).get("count", 0)
                        if type(count) is not int or count < 0:
                            raise ValueError("invalid search telemetry count")
                        counters[key] += count
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(matches[0])
    if segments == 0:
        raise ValueError("detailed segment telemetry unavailable; details_level10 required")
    return {**counters, "indexed_vectors": indexed_vectors, "segments": segments}


def storage_inventory(storage: Path, collection: str) -> dict:
    """Call only after a graceful stop, on one owned collection directory."""
    root = storage / "collections" / collection
    if root.is_symlink() or not root.is_dir():
        raise ValueError("owned collection storage missing")
    rows = []
    seen = set()
    total = 0
    indexed = 0
    graphs = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        if any((Path(directory) / name).is_symlink() for name in [*dirs, *files]):
            raise ValueError("symlink inside measured storage")
        if len(rows) + len(files) > 4096:
            raise ValueError("storage file count budget exceeded")
        for name in sorted(files):
            path = Path(directory) / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("storage contains a nonregular file")
            if (info.st_dev, info.st_ino) in seen:
                raise ValueError("storage contains duplicate hardlinked files")
            seen.add((info.st_dev, info.st_ino))
            total += info.st_size
            if total > STORAGE_LIMIT:
                raise ValueError("storage logical byte budget exceeded")
            relative = path.relative_to(root)
            if f"vector_index-{VECTOR_NAME}" in relative.parts:
                if name not in INDEX_FILES:
                    raise ValueError("unknown pinned HNSW index file")
                category = "hnsw_config" if name == "hnsw_config.json" else "hnsw_graph"
                if name == "graph.bin":
                    graphs += 1
                if name == "hnsw_config.json":
                    config = decode_json(read_regular(path, 16384))
                    count = config.get("indexed_vector_count")
                    if type(count) is not int or count < 0 or config.get("m", 0) <= 0:
                        raise ValueError("invalid HNSW storage configuration")
                    indexed += count
            elif "wal" in relative.parts:
                category = "wal"
            elif f"vector_storage-{VECTOR_NAME}" in relative.parts:
                category = "vectors"
            elif "payload_storage" in relative.parts or "payload_index" in relative.parts:
                category = "payload"
            else:
                category = "other"
            row = {"path": relative.as_posix(), "category": category, "logical_bytes": info.st_size,
                   "allocated_bytes": info.st_blocks * 512}
            if category.startswith("hnsw"):
                row["sha256"] = digest(read_regular(path, 16 * 1024 * 1024))
            rows.append(row)
    if graphs == 0 or indexed == 0:
        raise ValueError("no actual HNSW graph/configuration in stopped server storage")
    categories = {}
    for category in ["hnsw_graph", "hnsw_config", "vectors", "wal", "payload", "other"]:
        categories[category] = {key: sum(row[key] for row in rows if row["category"] == category)
                                for key in ("logical_bytes", "allocated_bytes")}
    return {"files": sorted(rows, key=lambda row: row["path"]), "categories": categories,
            "hnsw_files_logical_bytes": sum(categories[c]["logical_bytes"] for c in ("hnsw_graph", "hnsw_config")),
            "total_logical_bytes": total, "total_allocated_bytes": sum(row["allocated_bytes"] for row in rows),
            "indexed_vectors_from_hnsw_configs": indexed,
            "scope": "gracefully stopped collection; logical and st_blocks allocation, not exclusive physical disk cost"}


def _percentile(samples, fraction):
    return sorted(samples)[max(0, math.ceil(len(samples) * fraction) - 1)]


def _results(response: dict, allowed_ids: set[str], k: int):
    rows = response["result"]["points"]
    if not isinstance(rows, list) or len(rows) != k:
        raise ValueError("incomplete ANN query result")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids) or not set(ids) <= allowed_ids:
        raise ValueError("query result changed canonical repository identities")
    if any(type(row["score"]) not in (int, float) or not math.isfinite(row["score"]) for row in rows):
        raise ValueError("nonfinite query score")
    return rows


def measure_artifact(artifact: Artifact, *, k: int = 3, repeats: int = 2, total_seconds: float = 180) -> dict:
    if type(k) is not int or not 1 <= k <= 100 or type(repeats) is not int or not 1 <= repeats <= 5:
        raise ValueError("query budget out of bounds")
    root = Path(__file__).resolve().parents[1]
    def source_identity():
        paths = [Path(__file__), Path(registry_module.__file__), Path(artifact_module.__file__),
                 root / 'scripts/measure_embedding_ann.py', root / 'scripts/compare_embedding_models.py']
        return {str(path.relative_to(root)): digest(read_regular(path, JSON_LIMIT)) for path in paths}
    source_hashes = source_identity()
    started = time.monotonic()
    with OwnedQdrant(total_seconds=total_seconds) as owned:
        startup_seconds = time.monotonic() - started
        collection = "offline_embedding_ann_" + owned.run_id[:16]
        endpoint = "/collections/" + collection
        config = {"vectors": {VECTOR_NAME: {"size": artifact.configuration.output_dimension,
                    "distance": "Cosine", "on_disk": True}}, "shard_number": 1,
                  "hnsw_config": {"m": 16, "ef_construct": 100, "full_scan_threshold": 10,
                                  "on_disk": True, "max_indexing_threads": 1},
                  "optimizers_config": {"indexing_threshold": 0, "default_segment_number": 1,
                                         "max_optimization_threads": 1, "flush_interval_sec": 1},
                  "metadata": {"offline_embedding_version": artifact.configuration.version, "owner": owned.run_id}}
        owned.api(endpoint, config, "PUT")
        upload_start = time.monotonic()
        batch, batch_bytes, upload_bytes, batches = [], 0, 0, []

        def upload():
            nonlocal batch, batch_bytes, upload_bytes
            if batch:
                body = {"points": batch}
                count = len(encode_json(body))
                before = time.monotonic()
                result = owned.api(endpoint + "/points?wait=true", body, "PUT")
                if result.get("result", {}).get("status") != "completed":
                    raise ValueError("owned ingest did not complete")
                batches.append(time.monotonic() - before)
                upload_bytes += count
                batch, batch_bytes = [], 0

        for repo_id, vector in zip(artifact.repository_ids, artifact.documents):
            point = {"id": repo_id, "vector": {VECTOR_NAME: vector.tolist()},
                     "payload": {"offline_embedding_version": artifact.configuration.version}}
            size = len(encode_json(point)) + 1
            if batch and (len(batch) >= 64 or batch_bytes + size > RPC_LIMIT // 2):
                upload()
            batch.append(point)
            batch_bytes += size
        upload()
        upload_seconds = time.monotonic() - upload_start
        build_start = time.monotonic()
        owned.api(endpoint, {"optimizers_config": {"indexing_threshold": 1}}, "PATCH")
        ready_deadline = time.monotonic() + owned.remaining(60)
        stable = 0
        samples = []
        while stable < 2:
            info = owned.api(endpoint)["result"]
            if info.get("status") == "red" or isinstance(info.get("optimizer_status"), dict):
                raise RuntimeError("owned index optimizer failed")
            count = owned.api(endpoint + "/points/count", {"exact": True}, "POST")["result"]["count"]
            detail = telemetry(owned.api("/telemetry?details_level=10"), collection)
            ready = (info.get("status") == "green" and info.get("optimizer_status") == "ok"
                     and count == len(artifact.repository_ids) and info.get("indexed_vectors_count", 0) > 0
                     and detail["indexed_vectors"] >= len(artifact.repository_ids))
            stable = stable + 1 if ready else 0
            samples.append({"elapsed_seconds": time.monotonic() - build_start, "status": info.get("status"),
                            "indexed_vectors": detail["indexed_vectors"], "exact_count": count})
            if time.monotonic() >= ready_deadline:
                raise TimeoutError("actual HNSW index readiness deadline exceeded")
            if stable < 2:
                time.sleep(.2)
        build_seconds = time.monotonic() - build_start
        before = telemetry(owned.api("/telemetry?details_level=10"), collection)
        timings, server_times, recalls = [], [], []
        effective_k = min(k, len(artifact.repository_ids))
        allowed = set(artifact.repository_ids)
        for _repeat in range(repeats):
            for query in artifact.queries:
                body = {"query": query.tolist(), "using": VECTOR_NAME, "limit": effective_k,
                        "params": {"exact": False, "hnsw_ef": 64}, "with_payload": False, "with_vector": False}
                began = time.monotonic()
                response = owned.api(endpoint + "/points/query", body, "POST")
                timings.append(time.monotonic() - began)
                server_times.append(response["time"])
                if type(server_times[-1]) not in (int, float) or not math.isfinite(server_times[-1]) or server_times[-1] < 0:
                    raise ValueError("invalid server query duration")
                approximate = _results(response, allowed, effective_k)
                body["params"]["exact"] = True
                exact = _results(owned.api(endpoint + "/points/query", body, "POST"), allowed, effective_k)
                for row in [*approximate, *exact]:
                    original = artifact.documents[artifact.repository_ids.index(row["id"])]
                    if not math.isclose(float(np.dot(query, original)), row["score"], abs_tol=1e-5, rel_tol=0):
                        raise ValueError("server score no longer resolves to original canonical vector row")
                # Ties at the exact cutoff are accepted by score, not arbitrary
                # ID ordering. This is ANN fidelity, never judged relevance.
                cutoff = min(row["score"] for row in exact)
                recalls.append(sum(row["score"] >= cutoff - 1e-6 for row in approximate) / effective_k)
        after = telemetry(owned.api("/telemetry?details_level=10"), collection)
        deltas = {key: after[key] - before[key] for key in ("unfiltered_hnsw", "unfiltered_plain", "unfiltered_exact")}
        if deltas["unfiltered_hnsw"] < len(timings) or deltas["unfiltered_plain"] != 0:
            raise ValueError("real HNSW traversal was not proven for measured queries")
        owned.stop()
        storage = storage_inventory(owned.storage, collection)
        if storage["indexed_vectors_from_hnsw_configs"] < len(allowed):
            raise ValueError("stored HNSW graphs do not cover the uploaded corpus")
        if source_identity() != source_hashes:
            raise ValueError("ANN tool source changed during measurement")
        return {"schema_version": 1, "configuration": artifact.configuration.model_dump(),
                "embedding_version": artifact.configuration.version, "report_path": artifact.report_path,
                "encoder_report_sha256": artifact.report_sha256, "artifact_path": artifact.artifact_path,
                "artifact_sha256": artifact.artifact_sha256,
                "artifact_digest_scope": "verified against encoder digest" if artifact.report["cost"].get("shadow_artifact_sha256") else "measured now; historical encoder did not record artifact digest",
                "registry_sha256": artifact.registry_sha256, "fixture_sha256": artifact.fixture_sha256,
                "encoder_cost": artifact.report["cost"], "label_provenance": artifact.report["label_provenance"],
                "encoder_created_at": artifact.report.get("created_at"),
                "encoder_hardware": artifact.report.get("hardware"),
                "encoder_libraries": artifact.report.get("libraries"),
                "encoder_sample_counts": artifact.report.get("sample_counts"),
                "created_at": datetime.now(timezone.utc).isoformat(), "server": owned.server, "image": IMAGE,
                "source_sha256": source_hashes,
                "libraries": {"python": sys.version.split()[0], "numpy": np.__version__, "transport": "stdlib urllib HTTP"},
                "resources": {"cpus": 2, "memory_bytes": 1024**3, "pids": 256},
                "collection": collection, "requested_ingest_configuration": config, "effective_configuration": info["config"],
                "points": len(allowed), "queries": len(artifact.queries), "repeats": repeats, "k": effective_k,
                "startup_seconds": startup_seconds, "upload_seconds": upload_seconds, "serialized_upload_bytes": upload_bytes,
                "upload_batches_seconds": batches, "index_build_ready_seconds": build_seconds,
                "readiness_samples": samples, "search_telemetry_deltas": deltas,
                "first_query_http_seconds": timings[0], "subsequent_query_p50_seconds": _percentile(timings[1:] or timings, .5),
                "subsequent_query_p95_seconds": _percentile(timings[1:] or timings, .95),
                "query_http_seconds": timings, "query_server_seconds": server_times,
                "ann_exact_tie_aware_recall": recalls, "storage": storage,
                "ann_index_bytes": storage["hnsw_files_logical_bytes"],
                "server_peak_memory_bytes": None, "server_peak_memory_reason": "not sampled; encoder RSS is separate",
                "scope": "frozen supplied corpus and actual HNSW; no production cost extrapolation or quality qualification",
                "promotion_decision": "none; no serving state changed"}
