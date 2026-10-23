"""Opt-in registered capture route with a uniquely owned real Qdrant server."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import stat
import subprocess
import time
from urllib.parse import urlsplit
import uuid

import pytest
from qdrant_client import QdrantClient

from api import v2
from api.contracts import CapturedRecommendationResponse
from inference.scoring_provenance import decode_owned_packet
from tests.test_scoring_provider import make_retriever, payload, post, USER, SECRET, CAPABILITY, COMMIT, trained_ranker

IMAGE = "qdrant/qdrant@sha256:75eab8c4ba42096724fdcfde8b4de0b5713d529dde32f285a1f86fdcb2c9e50c"


@contextmanager
def owned_server():
    # Local Unix sockets only, never Docker remote contexts or production URLs.
    target = os.getenv("DOCKER_HOST") or "unix:///var/run/docker.sock"
    parsed = urlsplit(target)
    if parsed.scheme != "unix" or parsed.netloc or parsed.query or parsed.fragment or not parsed.path.startswith("/"):
        raise RuntimeError("owned capture fixture requires a local Unix Docker socket")
    if not stat.S_ISSOCK(Path(parsed.path).stat().st_mode):
        raise RuntimeError("local Docker socket is unavailable")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("DOCKER_")}
    name = "weave-scoring-provider-" + uuid.uuid4().hex[:16]
    owner = uuid.uuid4().hex
    def docker(*arguments, check=True):
        result = subprocess.run(["docker", "--host", target, *arguments], env=environment,
                                check=False, capture_output=True, text=True, timeout=30)
        if check and result.returncode:
            raise RuntimeError("owned capture Docker fixture operation failed")
        return result
    primary_error = None
    try:
        docker("run", "--pull=never", "-d", "--name", name, "--label", "weave.scoring-capture-owner=" + owner,
               "--memory", "1g", "--cpus", "2", "--pids-limit", "256", "-p", "127.0.0.1::6333",
               "-e", "QDRANT__TELEMETRY_DISABLED=true", IMAGE)
        info = json.loads(docker("inspect", name).stdout)[0]
        binding = info["NetworkSettings"]["Ports"]["6333/tcp"]
        assert len(binding) == 1 and binding[0]["HostIp"] == "127.0.0.1"
        url = "http://127.0.0.1:" + binding[0]["HostPort"]
        client = QdrantClient(url=url, api_key=None, timeout=2, check_compatibility=False)
        try:
            deadline = time.monotonic() + 20
            while True:
                try:
                    client.get_collections()
                    break
                except Exception:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(.1)
            yield client
        finally:
            client.close()
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            result = docker("inspect", name, check=False)
            if result.returncode == 0:
                info = json.loads(result.stdout)[0]
                if info["Config"]["Labels"].get("weave.scoring-capture-owner") != owner:
                    raise RuntimeError("owned capture fixture label mismatch; refusing removal")
                docker("rm", "-f", "-v", name)
            elif "no such object" not in result.stderr.casefold() and "no such container" not in result.stderr.casefold():
                raise RuntimeError("could not verify capture fixture cleanup")
        except Exception as cleanup_error:
            if primary_error is None:
                raise
            if hasattr(primary_error, "add_note"):
                primary_error.add_note(str(cleanup_error))


@pytest.mark.integration
def test_registered_capture_with_actual_owned_qdrant_and_trained_ranker(monkeypatch, trained_ranker):
    if os.getenv("RUN_SCORING_CAPTURE_INTEGRATION") != "1":
        pytest.skip("set RUN_SCORING_CAPTURE_INTEGRATION=1 for a new owned local Qdrant server")
    monkeypatch.setenv("INTERNAL_API_SECRET", SECRET)
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("SCORING_CAPTURE_ENABLED", "true")
    monkeypatch.setenv("SCORING_CAPTURE_CAPABILITY", CAPABILITY)
    monkeypatch.setenv("SCORING_CAPTURE_CODE_COMMIT", COMMIT)
    with owned_server() as client:
        active, _, _ = make_retriever(client, trained_ranker)
        monkeypatch.setattr(v2, "retriever", lambda: active)
        response = post(payload())
        assert response.status_code == 200, response.text
        model = CapturedRecommendationResponse.model_validate_json(response.content)
        assert model.scoring_context.profile_version == "2"
        assert model.scoring_context.feedback_version == "3"
        assert len(response.content) <= 1_000_000
        for item in model.items:
            raw = decode_owned_packet(item.scored_input, model.scoring_context, owner_user_id=USER,
                generation_id=str(model.generation_id), repo_id=str(item.repo_id))
            assert len(raw) == 778 * 4
