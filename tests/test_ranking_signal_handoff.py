"""Opt-in synthetic enrichment -> real Node SQL/outbox -> local Qdrant probe."""
from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile
import time
from urllib.parse import urlsplit
import uuid

import pytest
import requests
from qdrant_client import QdrantClient, models

from acquisition.backend_client import BackendIngestionClient, repository_upsert_record
from acquisition.repository_enricher import EnrichmentResult, RepositoryEnricher
from api.contracts import RepositoryJob
from api.v2 import _repository_embedding_payload
from config import QDRANT_VECTOR_NAME
from embedding.repository_embedding import RepositoryEmbeddingConfig, build_vector_payload
from ingestion.features import activity_score, score_code_health, score_documentation, trend_velocity
from utils.readme_processor import process_markdown

ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = ("pushed_days_ago", "delta_3d", "delta_7d", "delta_30d", "mentionable_users_count",
                "readme_to_codebase_ratio", "recent_commits", "created_at", "updated_at")
pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    os.getenv("RUN_RANKING_SIGNAL_HANDOFF") != "1", reason="Explicit disposable cross-repo fixture opt-in required")]


@contextmanager
def backend_fixture(prefix):
    backend = Path(os.environ["WEAVE_BACKEND_ROOT"]).resolve()
    target = urlsplit(os.environ["SIGNAL_DATABASE_URL"])
    assert target.scheme in {"postgres", "postgresql"}
    assert target.hostname in {"127.0.0.1", "localhost", "::1"} and target.path == "/weave_scroll"
    assert not target.query and not target.fragment
    assert os.environ.get("NODE_ENV") != "production"
    assert (backend / "backend/contracts/ml-v2.openapi.json").read_bytes() == (ROOT / "contracts/ml-v2.openapi.json").read_bytes()
    env = dict(os.environ, WEAVE_BACKEND_ROOT=str(backend), SIGNAL_FIXTURE_PREFIX=prefix,
               DATABASE_URL=os.environ["SIGNAL_DATABASE_URL"], DB_SCHEMA_V2_WRITES="true",
               INTERNAL_API_SECRET="local-signal-fixture-secret", INTERNAL_API_SECRET_PREVIOUS="", NODE_ENV="test")
    with tempfile.TemporaryFile(mode="w+") as errors:
        process = subprocess.Popen([str(backend / "backend/node_modules/.bin/tsx"),
                                    str(ROOT / "tests/fixtures/ranking_signal_backend.mjs")],
                                   cwd=backend / "backend", env=env, stdout=subprocess.PIPE, stderr=errors, text=True)
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    errors.seek(0)
                    raise AssertionError("Backend fixture failed: " + errors.read()[-2000:])
                if select.select([process.stdout], [], [], .2)[0]:
                    line = process.stdout.readline().strip()
                    if line.startswith("SIGNAL_FIXTURE_PORT="):
                        yield f"http://127.0.0.1:{int(line.split('=')[1])}"
                        break
            else:
                raise AssertionError("Backend fixture did not bind within its deadline")
        finally:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
                raise AssertionError("Backend fixture cleanup exceeded its deadline")
            process.stdout.close()
            errors.seek(0)
            assert process.returncode == 0, errors.read()[-2000:]


def enriched_sources(prefix):
    now = datetime.now(timezone.utc)
    base = int(uuid.uuid4().hex[:12], 16) + 700_000_000_000_000
    readme = process_markdown("# Signal fixture\n\n## Installation\nInstall this synthetic fixture locally.\n\n"
                              "## Usage\nTrace ranking observations through the canonical pipeline.\n\n## License\nMIT\n")
    enricher = RepositoryEnricher(graphql_client=object())
    for index, variant in enumerate(("observed", "zero", "unknown")):
        raw = {"full_name": f"{prefix}/{variant}", "github_id": str(base + index + 1),
               "github_node_id": f"R_fixture_{base + index + 1}", "owner": {"github_id": str(base)},
               "html_url": f"https://github.com/{prefix}/{variant}", "size": 4,
               "stargazers_count": 17, "forks_count": 3, "open_issues_count": 2,
               "created_at": "2020-01-01T00:00:00Z", "updated_at": now.isoformat(),
               "pushed_at": (now - timedelta(days=2)).isoformat(),
               "description": "This fixture follows repository ranking metadata through canonical ingestion. It checks observations and deterministic defaults."}
        payload = enricher.to_osiris_payload(raw, readme=readme, topics=[], languages={"Python": 4096},
                    contributors=[{"login": "fixture-a"}, {"login": "fixture-b"}], events=[],
                    stargazers=[{"starred_at": (now - timedelta(days=days)).isoformat()} for days in (1, 6, 20)],
                    recent_commits=[(now - timedelta(days=days)).isoformat() for days in (1, 2, 3)])
        if variant == "zero":
            # Explicit zero/empty observations at the delivery boundary, not a
            # claim that the enrichment contributor proxy can observe zero.
            payload.update({field: 0 for field in OBSERVATIONS[:5]})
            payload.update(readme_to_codebase_ratio=0.0, recent_commits=[])
        elif variant == "unknown":
            # Sparse supported source: preserve absence, then expose serving defaults.
            for field in OBSERVATIONS:
                payload.pop(field, None)
            raw.pop("created_at", None)
            raw.pop("updated_at", None)
        yield variant, EnrichmentResult(raw["full_name"], payload, raw, readme, [], {"Python": 4096})


def test_ranking_signals_survive_real_canonical_handoff_and_local_qdrant():
    prefix = "signal-handoff-" + uuid.uuid4().hex[:12]
    sources = dict(enriched_sources(prefix))
    with backend_fixture(prefix) as origin, requests.Session() as session, closing(QdrantClient(":memory:")) as qdrant:
        session.trust_env = False
        client = BackendIngestionClient(base_url=origin, internal_secret="local-signal-fixture-secret",
                                        timeout=5, max_retries=1, session=session)
        delivery = client.upsert_repositories_detailed(sources.values())
        assert delivery.count == 3 and not delivery.failed, delivery.failed
        response = session.get(origin + "/fixture/snapshot", headers={"x-internal-secret": "local-signal-fixture-secret"}, timeout=5)
        response.raise_for_status()
        report = response.json()
        qdrant.create_collection("signal-fixture", vectors_config={QDRANT_VECTOR_NAME: models.VectorParams(size=384, distance=models.Distance.COSINE)})
        jobs = {row["payload"]["repository"]["full_name"].split("/")[-1]: row["payload"]
                for row in report["jobs"] if row["job_type"] == "repo_index"}
        assert set(jobs) == set(sources)
        for variant, source in sources.items():
            job = RepositoryJob.model_validate({"schema_version": 2, **jobs[variant]})
            wire = next(row for row in report["wire_requests"][0] if row["github_id"] == source.payload["github_id"])
            assert wire == repository_upsert_record(source)
            assert str(job.repo_id) == delivery.mappings[source.payload["github_id"]]["repo_id"]
            material = _repository_embedding_payload(job)
            if variant != "unknown":
                for field in OBSERVATIONS:
                    assert field in wire and field in job.repository.model_fields_set
                for field in OBSERVATIONS[:6]:
                    assert material[field] == source.payload[field]
                timestamps = lambda values: [datetime.fromisoformat(value.replace("Z", "+00:00")) for value in values]
                # Node intentionally canonicalizes ISO timestamps through Date,
                # retaining millisecond rather than Python microsecond precision.
                expected_commits = [value.replace(microsecond=value.microsecond // 1000 * 1000)
                                    for value in timestamps(source.payload["recent_commits"])]
                assert timestamps(material["recent_commits"]) == expected_commits
                for field in ("created_at", "updated_at"):
                    expected = timestamps([source.payload[field]])[0]
                    expected = expected.replace(microsecond=expected.microsecond // 1000 * 1000)
                    assert timestamps([material[field]])[0] == expected
            else:
                assert all(field not in wire and field not in job.repository.model_fields_set for field in OBSERVATIONS)
                assert material["created_at"] is None and material["pushed_days_ago"] == 999
                assert material["delta_3d"] == 0 and material["recent_commits"] == []
            vector = [1.0] + [0.0] * 383
            payload = build_vector_payload(material, repo_id=str(job.repo_id), final_embedding=vector,
                          readme_chunks=0, source_hash="a" * 64, config=RepositoryEmbeddingConfig())
            for field, score in (("activity_score", activity_score), ("trend_velocity", trend_velocity),
                                 ("code_health", score_code_health)):
                assert payload[field] == score(material)
            assert payload["doc_quality"] == score_documentation(material).score
            canonical = next(row for row in report["rows"] if row["repo_id"] == str(job.repo_id))
            assert jobs[variant]["repository"]["readme"] == canonical["readme"]
            assert job.repository.readme == canonical["readme"].strip()
            assert payload["readme_length"] == len(canonical["readme"].strip()), "ML validation trims boundary whitespace before reconstructing length"
            qdrant.upsert("signal-fixture", [models.PointStruct(id=str(job.repo_id), vector={QDRANT_VECTOR_NAME: vector}, payload=payload)])
            stored = qdrant.retrieve("signal-fixture", [str(job.repo_id)])[0].payload
            assert stored["github_id"] == wire["github_id"] and stored["repo_id"] == str(job.repo_id)
            assert stored["content_hash"] == jobs[variant]["repository"]["content_hash"]
            assert stored["created_at"] == payload["created_at"]
            for field in (*OBSERVATIONS[:5], "star_count", "fork_count", "open_issues_count"):
                assert stored[field] == material[field]
        assert all(datetime.fromisoformat(row["product_created_at"].replace("Z", "+00:00")).year > 2020 for row in report["rows"])
        print(json.dumps({"backend_revision": report["backend_revision"],
              "ml_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "ml_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
              "repositories": 3, "observed_fields": list(OBSERVATIONS),
              "scope": "synthetic real Node HTTP/auth/SQL/outbox and ML projection/local Qdrant; no encoder or production rollout"}))
