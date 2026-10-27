import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.contracts import RepositoryJob
from api.v2 import (
    FeedbackBatch,
    RecommendationRequest,
    _repository_embedding_payload,
    _repository_job_lock,
    _repository_job_status,
    router,
)
from embedding.qdrant_store import QdrantRepositoryStore
from retrieval.v2_retriever import RecommendationBatch, RankedRepository


def test_canonical_application_exposes_only_v2_api_paths():
    from app import app as canonical_app

    paths = set(canonical_app.openapi()["paths"])
    assert paths
    assert all(path.startswith("/api/v2/") for path in paths)


@pytest.mark.parametrize("exclusions", [
    [str(uuid.UUID(int=1))] * 2,
    [str(uuid.UUID(int=index + 1)) for index in range(501)],
    ["owner/repository"],
], ids=["duplicate", "over-500", "invalid-id"])
def test_recommendation_contract_rejects_invalid_exclusions(exclusions):
    with pytest.raises(ValidationError):
        RecommendationRequest(
            schema_version=2,
            generation_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            feed_version=1,
            limit=45,
            exclude_repo_ids=exclusions,
            context={"cold_start": False},
        )


def test_feedback_contract_enforces_dwell_and_unique_events():
    base = {
        "event_id": uuid.uuid4(), "user_id": uuid.uuid4(), "repo_id": uuid.uuid4(),
        "feedback_version": 1, "event_type": "dwell", "occurred_at": "2026-07-14T00:00:00Z",
    }
    with pytest.raises(ValidationError):
        FeedbackBatch(schema_version=2, events=[{**base, "dwell_ms": 2_999}])
    valid = {**base, "dwell_ms": 3_000}
    with pytest.raises(ValidationError):
        FeedbackBatch(schema_version=2, events=[valid, valid])


def test_repository_point_id_is_the_canonical_backend_uuid():
    repo_id = str(uuid.uuid4())
    assert QdrantRepositoryStore._point_id(repo_id) == repo_id
    with pytest.raises(ValueError):
        QdrantRepositoryStore._point_id("owner/repository")


def test_repository_job_accepts_canonical_node_v2_outbox_payload():
    repo_id = uuid.uuid4()
    job = RepositoryJob.model_validate(
        {
            "schema_version": 2,
            "job_id": str(uuid.uuid4()),
            "repo_id": str(repo_id),
            "content_version": 1,
            "repository": {
                "repo_id": str(repo_id),
                "github_id": "711550638",
                "github_node_id": "R_kgDOKmlmrg",
                "full_name": "datawhalechina/llm-universe",
                "owner": "datawhalechina",
                "name": "llm-universe",
                "url": "https://github.com/datawhalechina/llm-universe",
                "description": None,
                "readme": "Repository documentation",
                "readme_source_path": "docs/README.md",
                "readme_default_branch": "main",
                "readme_base_url": "https://raw.githubusercontent.com/datawhalechina/llm-universe/refs/heads/main/docs/",
                "primary_language": None,
                "languages": ["Jupyter Notebook", "Python"],
                "topics": ["langchain", "rag"],
                "star_count": 13_612,
                "fork_count": 1_383,
                "open_issues_count": 10,
                "pushed_at": "2026-02-24T14:33:21Z",
                "observed_at": "2026-07-22T14:18:33Z",
                "content_hash": "4fea9174cc2f3aca308a150360f01641",
            },
        }
    )

    assert job.repository.repo_id == repo_id
    assert job.repository.html_url == (
        "https://github.com/datawhalechina/llm-universe"
    )
    payload = _repository_embedding_payload(job)
    assert payload["repo_id"] == str(repo_id)
    assert payload["description"] == ""
    assert payload["primary_language"] == "Unknown"
    assert payload["readme_length"] == len("Repository documentation")
    assert payload["readme"] == "Repository documentation"
    assert payload["extracted_paragraphs"] == ["Repository documentation"]
    assert payload["readme_source_path"] == "docs/README.md"
    assert payload["readme_default_branch"] == "main"
    assert payload["readme_base_url"].endswith("/docs/")


def test_repository_job_accepts_readme_metadata_at_exact_backend_boundaries():
    repo_id = uuid.uuid4()
    source_path = "a" * (1_024 - len("/README.md")) + "/README.md"
    branch = "b" * 256
    raw_prefix = "https://raw.githubusercontent.com/"
    base_url = raw_prefix + "c" * (2_048 - len(raw_prefix) - 1) + "/"

    job = RepositoryJob.model_validate(
        {
            "schema_version": 2,
            "job_id": str(uuid.uuid4()),
            "repo_id": str(repo_id),
            "content_version": 1,
            "repository": {
                "repo_id": str(repo_id),
                "full_name": "owner/repository",
                "content_hash": "boundary-test",
                "readme_source_path": source_path,
                "readme_default_branch": branch,
                "readme_base_url": base_url,
            },
        }
    )

    assert len(job.repository.readme_source_path or "") == 1_024
    assert len(job.repository.readme_default_branch or "") == 256
    assert len(job.repository.readme_base_url or "") == 2_048


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("readme_source_path", "a" * (1_025 - len("/README.md")) + "/README.md"),
        ("readme_default_branch", "b" * 257),
        (
            "readme_base_url",
            "https://raw.githubusercontent.com/"
            + "c" * (2_049 - len("https://raw.githubusercontent.com/") - 1)
            + "/",
        ),
    ],
)
def test_repository_job_rejects_readme_metadata_over_backend_boundaries(
    field: str, value: str
):
    repo_id = uuid.uuid4()
    with pytest.raises(ValidationError):
        RepositoryJob.model_validate(
            {
                "schema_version": 2,
                "job_id": str(uuid.uuid4()),
                "repo_id": str(repo_id),
                "content_version": 1,
                "repository": {
                    "repo_id": str(repo_id),
                    "full_name": "owner/repository",
                    "content_hash": "boundary-test",
                    field: value,
                },
            }
        )


def test_repository_job_rejects_mismatched_nested_repo_id():
    with pytest.raises(ValidationError, match="repository.repo_id must match repo_id"):
        RepositoryJob.model_validate(
            {
                "schema_version": 2,
                "job_id": str(uuid.uuid4()),
                "repo_id": str(uuid.uuid4()),
                "content_version": 1,
                "repository": {
                    "repo_id": str(uuid.uuid4()),
                    "full_name": "owner/repository",
                    "content_hash": "mismatched-repo-test",
                },
            }
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"readme_source_path": "../README.md"},
        {"readme_source_path": "/README.md"},
        {"readme_base_url": "http://raw.githubusercontent.com/owner/repo/main/"},
        {"readme_base_url": "https://evil.example/owner/repo/main/"},
        {"readme_default_branch": "feature/../main"},
        {"readme_default_branch": "feature\\main"},
    ],
)
def test_repository_job_rejects_unsafe_readme_source_metadata(metadata):
    repo_id = uuid.uuid4()
    with pytest.raises(ValidationError):
        RepositoryJob.model_validate(
            {
                "schema_version": 2,
                "job_id": str(uuid.uuid4()),
                "repo_id": str(repo_id),
                "content_version": 1,
                "repository": {
                    "repo_id": str(repo_id),
                    "full_name": "owner/repository",
                    "content_hash": "unsafe-metadata-test",
                    **metadata,
                },
            }
        )


def test_repository_job_requires_content_hash_for_retry_fencing():
    repo_id = uuid.uuid4()
    with pytest.raises(ValidationError, match="content_hash is required"):
        RepositoryJob.model_validate(
            {
                "schema_version": 2,
                "job_id": str(uuid.uuid4()),
                "repo_id": str(repo_id),
                "content_version": 1,
                "repository": {
                    "repo_id": str(repo_id),
                    "full_name": "owner/repository",
                },
            }
        )


def test_v2_health_requires_internal_auth(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    response = client.get("/api/v2/health")
    assert response.status_code == 401

    monkeypatch.delenv("INTERNAL_API_SECRET")
    response = client.get("/api/v2/health")
    assert response.status_code == 503


def test_v2_auth_uses_configured_internal_header(monkeypatch, health_probe_cache):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    monkeypatch.setenv("INTERNAL_API_HEADER", "x-ml-service-secret")
    healthy = SimpleNamespace(health=MagicMock(return_value={"qdrant": "healthy"}))
    producer = SimpleNamespace(health=MagicMock(return_value={
        "redis": "healthy",
        "feedback_consumer_active": True,
    }))

    with patch("api.v2.retriever", return_value=healthy), patch(
        "api.v2.producer", return_value=producer
    ):
        assert client.get(
            "/api/v2/health",
            headers={"x-internal-secret": "test-internal-secret"},
        ).status_code == 401
        headers = {"x-ml-service-secret": "test-internal-secret"}
        # A router-only application has no lifespan/background cache population.
        assert client.get("/api/v2/health", headers=headers).status_code == 503
        assert client.get("/api/v2/health/deep", headers=headers).status_code == 200
        assert client.get("/api/v2/health", headers=headers).status_code == 200
        assert client.get("/api/v2/health/ready", headers=headers).status_code == 200
        # Age the observation deterministically without another dependency call.
        with health_probe_cache._lock:
            assert health_probe_cache._observed is not None
            health_probe_cache._observed -= health_probe_cache.settings.max_age_seconds + 1
        assert client.get("/api/v2/health/ready", headers=headers).status_code == 503

    healthy.health.assert_called_once()
    producer.health.assert_called_once()


def test_recommendation_response_reports_the_model_that_served_the_request(monkeypatch):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    user_id = uuid.uuid4()
    repo_id = uuid.uuid4()
    batch = RecommendationBatch(
        items=[RankedRepository(str(repo_id), 0.75, "semantic")],
        model_version="heavy-ranker-v2",
        embedding_version="repo-embedding-v2",
        ranker_applied=True,
    )
    fake_retriever = SimpleNamespace(
        recommend_batch=lambda *_args: batch,
        model_version="wrong-static-version",
        embedding_version="wrong-static-version",
    )

    with patch("api.v2.retriever", return_value=fake_retriever):
        response = client.post(
            "/api/v2/recommendations/generate",
            headers={"x-internal-secret": "test-internal-secret"},
            json={
                "schema_version": 2,
                "generation_id": str(uuid.uuid4()),
                "user_id": str(user_id),
                "feed_version": 1,
                "limit": 10,
                "exclude_repo_ids": [],
                "context": {"cold_start": False},
            },
        )

    assert response.status_code == 200
    assert response.json()["model_version"] == "heavy-ranker-v2"
    assert response.json()["embedding_version"] == "repo-embedding-v2"


def test_repository_jobs_are_idempotent_and_monotonic():
    job_id = str(uuid.uuid4())
    points = [
        SimpleNamespace(
            payload={"content_version": 7, "content_job_id": job_id}
        )
    ]
    assert _repository_job_status(
        points,
        version_field="content_version",
        job_field="content_job_id",
        requested_version=7,
        job_id=job_id,
    ) == ("duplicate", 7)
    assert _repository_job_status(
        points,
        version_field="content_version",
        job_field="content_job_id",
        requested_version=7,
        job_id=str(uuid.uuid4()),
    ) == ("current", 7)

    with pytest.raises(HTTPException) as exc_info:
        _repository_job_status(
            points,
            version_field="content_version",
            job_field="content_job_id",
            requested_version=6,
            job_id=str(uuid.uuid4()),
        )
    assert exc_info.value.status_code == 409


def test_repository_job_lock_uses_token_checked_release():
    redis = MagicMock()
    redis.set.return_value = True
    with patch("api.v2.producer", return_value=SimpleNamespace(redis=redis)):
        with _repository_job_lock(str(uuid.uuid4())):
            pass

    redis.set.assert_called_once()
    assert redis.set.call_args.kwargs == {"nx": True, "px": 600_000}
    redis.eval.assert_called_once()


def test_refresh_repository_job_backfills_missing_summary_fields_for_older_schema_v2_records():
    from api.v2 import _refresh_repository_job_locked
    from api.contracts import RepositoryRefreshJob, RepositoryFeaturePatch

    repo_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    lock = SimpleNamespace(assert_owned=lambda: None)

    # Create an older schema-v2 payload mock missing card_summary fields
    older_payload = {
        "github_id": "123",
        "html_url": "https://github.com/owner/repo",
        "repo_id": repo_id,
        "full_name": "owner/repo",
        "description": "test repo",
        "primary_language": "Python",
        "languages": ["Python"],
        "topics": [],
        "star_count": 10,
        "fork_count": 2,
        "open_issues_count": 0,
        "readme_length": 500,
        "readme_chunks": 1,
        "pushed_days_ago": 5,
        "delta_3d": 0,
        "delta_7d": 0,
        "delta_30d": 0,
        "mentionable_users_count": 1,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-02T00:00:00Z",
        "pushed_at": "2026-01-03T00:00:00Z",
        "discovery_category": None,
        "discovery_band": None,
        "category": "ML",
        "tags": [],
        "doc_quality": 0.5,
        "code_health": 0.8,
        "activity_score": 0.6,
        "trend_velocity": 0.1,
        "embedding_dim": 384,
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_model_revision": "main",
        "embedding_version": "v2",
        "feature_spec_version": "v2",
        "content_version": 1,
        "content_hash": "hash123",
        "model_version": "all-MiniLM-L6-v2",
        "indexed_at": "2026-01-04T00:00:00Z",
        "source_hash": "src123",
        "feature_version": 1,
        "feature_job_id": str(uuid.uuid4()),
        "serving_eligibility_version": "repository-vector-v2",
        # Notice: NO card_summary or card_summary_* fields!
    }

    mock_point = SimpleNamespace(id=repo_id, payload=older_payload)
    mock_store = MagicMock()
    mock_store.compare_and_set_features.return_value = SimpleNamespace(
        id=repo_id, payload={**older_payload, "feature_version": 2, "feature_job_id": job_id}
    )

    job = RepositoryRefreshJob(
        schema_version=2,
        repo_id=repo_id,
        job_id=job_id,
        feature_version=2,
        features=RepositoryFeaturePatch(star_count=15),
    )

    with patch("api.v2._repository_points", return_value=[mock_point]), \
         patch("api.v2.repository_store", return_value=mock_store):
        res = _refresh_repository_job_locked(job, lock)

    assert res["accepted"] is True
    assert res["status"] == "applied"
    assert res["feature_version"] == 2


def test_every_non_health_route_is_guarded(monkeypatch):
    from fastapi.testclient import TestClient
    import api.main as api_main

    monkeypatch.setenv("INTERNAL_API_SECRET", "test-secret")
    client = TestClient(api_main.app)
    for route in api_main.app.routes:
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", set())
        if not path:
            continue
        method = "POST" if "POST" in methods else "GET"
        if path == "/healthz" and method == "GET":
            continue
        response = client.request(method, path)
        assert response.status_code == 401, path
