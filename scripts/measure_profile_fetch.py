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
import os
from pathlib import Path
import subprocess
import stat
import sys
import time
from typing import Any
from unittest.mock import patch
import uuid
from urllib.parse import urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient, models
from retrieval.v2_retriever import QdrantV2Retriever, _USER_RETRIEVAL_PAYLOAD_FIELDS

VERSION_FIELDS = ("profile_version", "last_feedback_version", "feedback_state_revision")
IMAGE = "qdrant/qdrant@sha256:75eab8c4ba42096724fdcfde8b4de0b5713d529dde32f285a1f86fdcb2c9e50c"
OWNER_LABEL = "weave.profile-measure-owner"
FIXTURE_COLLECTION = "user_profiles"
FIXTURE_DIMENSIONS = 384
FIXTURE_MODEL = "all-MiniLM-L6-v2"
FIXTURE_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def fixture_reader(client):
    """Use the existing profile method with an explicit isolated MiniLM contract."""
    # This profile-only fixture does not run the required-heavy diversity path.
    # Isolate the constructor policy without changing the caller's environment.
    with patch.dict(os.environ, {"V2_DIVERSITY_MODE": "disabled"}):
        reader = QdrantV2Retriever(client=client, user_collection=FIXTURE_COLLECTION,
            heavy_ranker_enabled=False, heavy_ranker_required=False,
            heavy_ranker_traffic_percent=0, allow_unqualified_ranker=False,
            minimum_eligible_repositories=0, candidate_vector_mode="deferred")
    # Only the profile read is measured, not recommendation construction/health.
    # A developer's unrelated collection/dimension overrides must not change the
    # synthetic contract used by both writer and actual profile validation.
    reader.embedding_dim = FIXTURE_DIMENSIONS
    reader.embedding_model = FIXTURE_MODEL
    reader.embedding_model_revision = FIXTURE_REVISION
    reader.allow_missing_embedding_revision = False
    return reader


def profile_payload() -> dict[str, Any]:
    return {"skills": ["Python"], "tech_stack": ["FastAPI"], "interests": ["search"],
            "topics": ["vectors"], "embedding_dim": FIXTURE_DIMENSIONS,
            "embedding_model": FIXTURE_MODEL,
            "embedding_model_revision": FIXTURE_REVISION,
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
        if response.request.method == "POST" and response.request.url.path == f"/collections/{FIXTURE_COLLECTION}/points":
            response.read()
            self.milliseconds.append(1000 * (time.perf_counter() - response.request.extensions["profile_measurement_start"]))
            self.decoded_body_bytes.append(len(response.content))


def summary(values: list[float]) -> dict[str, float]:
    return {key: float(np.percentile(values, percentile))
            for key, percentile in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))}


class _LocalDocker:
    """Freeze a single explicit local daemon for startup, inspect and cleanup."""
    def __init__(self):
        target = os.environ.get("DOCKER_HOST") or "unix:///var/run/docker.sock"
        parsed = urlsplit(target)
        if (parsed.scheme != "unix" or parsed.netloc or not parsed.path.startswith("/")
                or parsed.query or parsed.fragment or target != "unix://" + parsed.path
                or any(ord(character) < 32 for character in target)):
            raise ValueError("profile measurement requires a local Docker Unix socket")
        if not stat.S_ISSOCK(Path(parsed.path).stat().st_mode):
            raise ValueError("profile measurement Docker Unix socket is unavailable")
        self.prefix = ("docker", "--host", target)
        self.environment = {key:value for key,value in os.environ.items() if not key.startswith("DOCKER_")}

    def run(self, *arguments: str, check=True):
        result = subprocess.run([*self.prefix,*arguments], env=self.environment,
            timeout=40, capture_output=True, text=True, check=False)
        if check and result.returncode:
            raise RuntimeError("owned profile Docker operation failed: " + arguments[0])
        return result

    def owned_info(self, name: str, owner: str):
        result = self.run("inspect", name, check=False)
        if result.returncode:
            if "no such object" in result.stderr.casefold() or "no such container" in result.stderr.casefold():
                return None
            raise RuntimeError("cannot verify owned profile container state")
        values = json.loads(result.stdout)
        if not isinstance(values,list) or len(values)!=1:
            raise RuntimeError("invalid owned profile container state")
        info = values[0]
        if info.get("Config",{}).get("Labels",{}).get(OWNER_LABEL)!=owner:
            raise RuntimeError("owned profile container label mismatch; refusing removal")
        return info


@dataclass(frozen=True)
class OwnedProfileTarget:
    url: str
    image_id: str


@contextmanager
def owned_qdrant():
    docker = _LocalDocker()
    name = "weave-profile-measure-" + uuid.uuid4().hex[:16]
    owner = uuid.uuid4().hex
    primary_error = None
    attempted = False
    try:
        # No pull; identity is captured from the SAME pinned local daemon.
        image_id = json.loads(docker.run("image","inspect",IMAGE).stdout)[0]["Id"]
        attempted = True  # A timed-out CLI may still have created the container.
        docker.run("run", "--pull=never", "-d", "--name", name, "--label", OWNER_LABEL+"="+owner,
            "--cpus","2","--memory","1g","--pids-limit","256",
            "-p", "127.0.0.1::6333", "-e", "QDRANT__TELEMETRY_DISABLED=true", IMAGE)
        info = docker.owned_info(name,owner)
        if info is None:
            raise RuntimeError("owned profile container was not created")
        bindings = info["NetworkSettings"]["Ports"]["6333/tcp"]
        if (len(bindings)!=1 or bindings[0].get("HostIp")!="127.0.0.1"
                or not str(bindings[0].get("HostPort","")).isdigit()
                or not 1<=int(bindings[0]["HostPort"])<=65535):
            raise ValueError("owned profile container must bind only to loopback")
        url = "http://127.0.0.1:" + bindings[0]["HostPort"]
        with closing(QdrantClient(url=url, api_key=None, timeout=2, check_compatibility=False)) as probe:
            for attempt in range(80):
                try:
                    probe.get_collections()
                    break
                except Exception:
                    if attempt == 79:
                        raise
                    time.sleep(.1)
        yield OwnedProfileTarget(url=url,image_id=image_id)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if attempted:
            try:
                # Inspect even after startup timeout/failure; never delete by a
                # guessed name or through a newly resolved environment/context.
                if docker.owned_info(name,owner) is not None:
                    docker.run("rm","-f","-v",name)
                    if docker.owned_info(name,owner) is not None:
                        raise RuntimeError("owned profile container remained after removal")
            except Exception:
                message = f"Owned Qdrant cleanup failed for {name}; verify/remove only this owned container"
                if primary_error is None:
                    raise RuntimeError(message) from None
                if hasattr(primary_error,"add_note"):
                    primary_error.add_note(message)
                try:
                    print(message,file=sys.stderr)
                except Exception:
                    pass


def run_scenario(writer: QdrantClient, reader: QdrantV2Retriever,
                 spans: WireSpans, *, name: str, users: int, reads: int,
                 update_every: int = 0, version_only: bool = False) -> dict[str, Any]:
    ids = [str(uuid.UUID(int=index + 630000)) for index in range(users)]
    vector = np.random.default_rng(63).normal(size=FIXTURE_DIMENSIONS)
    vector /= np.linalg.norm(vector)
    oracle = profile_payload()
    writer.upsert(FIXTURE_COLLECTION, points=[models.PointStruct(id=uid, vector=vector.tolist(),
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
                writer.set_payload(FIXTURE_COLLECTION, payload={key: oracle[key] for key in VERSION_FIELDS}, points=[uid], wait=True)
            else:
                vector = np.roll(vector, 1)
                writer.upsert(FIXTURE_COLLECTION, points=[models.PointStruct(id=uid, vector=vector.tolist(), payload=oracle)], wait=True)
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
    with owned_qdrant() as owned:
        url = owned.url
        image_id = owned.image_id
        with closing(QdrantClient(url=url, api_key=None, timeout=10)) as writer, closing(QdrantClient(
            url=url, api_key=None, timeout=10,
            event_hooks={"request": [spans.request], "response": [spans.response]},
        )) as wire:
            writer.create_collection(FIXTURE_COLLECTION, vectors_config=models.VectorParams(size=FIXTURE_DIMENSIONS, distance=models.Distance.COSINE))
            reader = fixture_reader(wire)
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
            "qdrant_image": IMAGE, "qdrant_image_id": image_id,
            "embedding_model": FIXTURE_MODEL, "embedding_revision": FIXTURE_REVISION,
            "projection": _USER_RETRIEVAL_PAYLOAD_FIELDS,
            "fixture_contract": {"collection": FIXTURE_COLLECTION, "dimensions": FIXTURE_DIMENSIONS,
                                 "model": FIXTURE_MODEL, "revision": FIXTURE_REVISION},
            "scope": "local sequential uncached reads; synthetic free-oracle opportunities; no cache or traffic estimate",
            "scenarios": scenarios}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reads", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(measure(args.reads), indent=2))
