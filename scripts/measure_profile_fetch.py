"""Bounded synthetic profile-read measurement; never installs or serves a cache.

The CLI creates/removes its own loopback-only Docker Qdrant. It accepts no target
URL and uses no configured storage credentials. All reuse counts are offline
opportunities with a free fixture version oracle, not measured cache hits.
"""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient, models
from config import EMBEDDING_MODEL_REVISION, REPOSITORY_EMBEDDING_MODEL
from retrieval.v2_retriever import QdrantV2Retriever, _USER_RETRIEVAL_PAYLOAD_FIELDS

VERSION_FIELDS = ("profile_version", "last_feedback_version", "feedback_state_revision")
IMAGE = "qdrant/qdrant:v1.18.2"


def profile_payload() -> dict[str, Any]:
    return {"skills": ["Python"], "tech_stack": ["FastAPI"], "interests": ["search"],
            "topics": ["vectors"], "embedding_dim": 384,
            "embedding_model": REPOSITORY_EMBEDDING_MODEL,
            "embedding_model_revision": EMBEDDING_MODEL_REVISION,
            "profile_version": 1, "last_feedback_version": 0,
            "feedback_state_revision": 0, "fixture_ledger": "x" * 16384}


@dataclass
class Opportunities:
    """Bounded post-read analyzer; contains hashes, never reusable profiles."""
    previous: dict[str, tuple[str, tuple[int, ...] | None]] = field(default_factory=dict)
    reads: int = 0
    valid: int = 0
    content_repeats: int = 0
    version_compatible_repeats: int = 0

    def observe(self, user_id: str, vector: list[float] | None,
                payload: dict[str, Any], oracle: dict[str, Any]) -> None:
        if self.reads >= 2048:
            raise ValueError("measurement is limited to 2048 reads")
        if user_id not in self.previous and len(self.previous) >= 256:
            raise ValueError("measurement is limited to 256 users")
        self.reads += 1
        if vector is None:
            self.previous.pop(user_id, None)
            return
        self.valid += 1
        digest = hashlib.sha256(json.dumps([vector, payload], sort_keys=True,
                                           allow_nan=False).encode()).hexdigest()
        values = tuple(oracle.get(key) for key in VERSION_FIELDS)
        version = values if all(type(v) is int and v >= 0 for v in values) and values[0] >= 1 else None
        old = self.previous.get(user_id)
        if old and old[0] == digest:
            self.content_repeats += 1
            if version is not None and old[1] == version:
                self.version_compatible_repeats += 1
        self.previous[user_id] = (digest, version)


@dataclass
class WireSpans:
    milliseconds: list[float] = field(default_factory=list)
    decoded_body_bytes: list[int] = field(default_factory=list)

    def request(self, request: Any) -> None:
        request.extensions["profile_measurement_start"] = time.perf_counter()

    def response(self, response: Any) -> None:
        if response.request.method == "POST" and response.request.url.path == "/collections/user_profiles/points":
            response.read()
            self.milliseconds.append(1000 * (time.perf_counter() - response.request.extensions["profile_measurement_start"]))
            self.decoded_body_bytes.append(len(response.content))


def summary(values: list[float]) -> dict[str, float]:
    return {key: float(np.percentile(values, percentile))
            for key, percentile in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))}


def docker(*args: str) -> str:
    return subprocess.check_output(["docker", *args], text=True, timeout=40).strip()


@contextmanager
def owned_qdrant():
    name = "weave-profile-measure-" + uuid.uuid4().hex[:12]
    try:
        docker("run", "--pull=never", "-d", "--name", name,
               "-p", "127.0.0.1::6333", "-e", "QDRANT__TELEMETRY_DISABLED=true", IMAGE)
        binding = json.loads(docker("inspect", name))[0]["NetworkSettings"]["Ports"]["6333/tcp"][0]
        url = "http://127.0.0.1:" + binding["HostPort"]
        with closing(QdrantClient(url=url, api_key=None, timeout=2, check_compatibility=False)) as probe:
            for attempt in range(80):
                try:
                    probe.get_collections()
                    break
                except Exception:
                    if attempt == 79:
                        raise
                    time.sleep(.1)
        yield url
    finally:
        subprocess.run(["docker", "rm", "-f", name], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=40)


def run_scenario(writer: QdrantClient, reader: QdrantV2Retriever,
                 spans: WireSpans, *, name: str, users: int, reads: int,
                 update_every: int = 0, version_only: bool = False) -> dict[str, Any]:
    ids = [str(uuid.UUID(int=index + 630000)) for index in range(users)]
    vector = np.random.default_rng(63).normal(size=384)
    vector /= np.linalg.norm(vector)
    oracle = profile_payload()
    writer.upsert("user_profiles", points=[models.PointStruct(id=uid, vector=vector.tolist(),
                  payload=oracle) for uid in ids], wait=True)
    analyzer = Opportunities()
    walls: list[float] = []
    start = len(spans.milliseconds)
    for index in range(reads):
        uid = ids[index % users]
        if update_every and index and index % update_every == 0:
            # Controlled fixture mutation, not the asynchronous feedback pipeline.
            oracle["last_feedback_version"] += 1
            oracle["feedback_state_revision"] += 1
            if version_only:
                writer.set_payload("user_profiles", payload={key: oracle[key] for key in VERSION_FIELDS}, points=[uid], wait=True)
            else:
                vector = np.roll(vector, 1)
                writer.upsert("user_profiles", points=[models.PointStruct(id=uid, vector=vector.tolist(), payload=oracle)], wait=True)
        before = time.perf_counter()
        actual, payload = reader._user_profile(uid)
        walls.append(1000 * (time.perf_counter() - before))
        assert actual is not None and "fixture_ledger" not in payload
        assert not any(key in payload for key in VERSION_FIELDS)
        analyzer.observe(uid, actual, payload, oracle)
    requests = len(spans.milliseconds) - start
    assert requests == reads, "every measured read must issue exactly one real HTTP retrieval"
    return {"name": name, "reads": reads, "users": users, "http_reads": requests,
            "valid_profiles": analyzer.valid, "content_repeats": analyzer.content_repeats,
            "free_version_oracle_repeats": analyzer.version_compatible_repeats,
            "wall_ms": summary(walls), "http_body_read_ms": summary(spans.milliseconds[start:]),
            "decoded_json_body_bytes": summary(spans.decoded_body_bytes[start:])}


def source_hashes() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
            ("retrieval/v2_retriever.py", "retrieval/runtime.py", "embedding/vector_contract.py",
             "config.py", "scripts/measure_profile_fetch.py")}


def measure(reads: int = 128) -> dict[str, Any]:
    if not 32 <= reads <= 256:
        raise ValueError("reads per scenario must be between 32 and 256")
    before = source_hashes()
    spans = WireSpans()
    with owned_qdrant() as url:
        with closing(QdrantClient(url=url, api_key=None, timeout=10)) as writer, closing(QdrantClient(
            url=url, api_key=None, timeout=10,
            event_hooks={"request": [spans.request], "response": [spans.response]},
        )) as wire:
            writer.create_collection("user_profiles", vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
            reader = QdrantV2Retriever(client=wire, heavy_ranker_enabled=False,
                                       minimum_eligible_repositories=0)
            scenarios = [run_scenario(writer, reader, spans, reads=reads, **spec) for spec in (
                {"name": "all_unique", "users": reads},
                {"name": "hot_unchanged", "users": 1},
                {"name": "round_robin_32", "users": 32},
                {"name": "hot_changed_every_8", "users": 1, "update_every": 8},
                {"name": "hot_version_only_every_8", "users": 1, "update_every": 8, "version_only": True},
            )]
    if before != source_hashes():
        raise RuntimeError("measured source changed during run")
    return {"schema": "profile-fetch-measurement-v1", "source_hashes": before,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "qdrant_image": IMAGE, "qdrant_image_id": json.loads(docker("image", "inspect", IMAGE))[0]["Id"],
            "embedding_model": REPOSITORY_EMBEDDING_MODEL, "embedding_revision": EMBEDDING_MODEL_REVISION,
            "projection": _USER_RETRIEVAL_PAYLOAD_FIELDS,
            "scope": "local sequential uncached reads; synthetic free-oracle opportunities; no cache or traffic estimate",
            "scenarios": scenarios}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reads", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(measure(args.reads), indent=2))
