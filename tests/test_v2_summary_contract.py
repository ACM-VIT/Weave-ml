"""Repository embed response, Qdrant replay, and summary-backfill contracts."""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import uuid

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.http import models

from api.contracts import RepositoryJob, RepositoryRefreshJob
from api import main as api_main
from api.v2 import _embed_repository_job_locked, _refresh_repository_job_locked, _repository_embedding_payload
from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
from embedding.qdrant_store import QdrantRepositoryStore
from ingestion.features import activity_score, score_code_health, trend_velocity
from summarization.contracts import (
    CARD_SUMMARY_FALLBACK_MODEL_VERSION,
    CARD_SUMMARY_PAYLOAD_FIELDS,
    CardSummaryArtifact,
    card_summary_from_payload,
    card_summary_payload,
)
from summarization.pipeline import CardSummaryPipeline


REPO_ID = "00000000-0000-4000-8000-000000000221"


class FakeEmbedder:
    def embed_texts(self, texts, *, normalize=True):
        return [self._vector() for _ in texts]

    def embed_text(self, text, *, normalize=True):
        return self._vector()

    @staticmethod
    def _vector():
        return [1.0] + [0.0] * 383


class OwnedLock:
    def assert_owned(self) -> None:
        return None


def _job(*, job_id: str | None = None, content_hash: str = "content-hash") -> RepositoryJob:
    return RepositoryJob.model_validate(
        {
            "schema_version": 2,
            "job_id": job_id or str(uuid.uuid4()),
            "repo_id": REPO_ID,
            "content_version": 7,
            "repository": {
                "repo_id": REPO_ID,
                "github_id": "123456",
                "full_name": "weave/summary-contract",
                "description": "A review workspace for teams maintaining shared developer tools.",
                "readme": "# Summary Contract\n\nA review workspace for shared developer tools.",
                "primary_language": "Python",
                "languages": ["Python"],
                "topics": ["developer-tools"],
                "content_hash": content_hash,
            },
        }
    )


def _artifact(summary: str = "A concise discovery summary explains the repository to developer-tool maintainers. Its focused review workflow helps distributed teams coordinate changes before they affect shared users and downstream integrations.") -> CardSummaryArtifact:
    return CardSummaryArtifact(
        summary=summary,
        highlights=["Review workflow"],
        prompt_version="repo-card-summary-v1",
        model_version="meta-llama/llama-3.3-70b-instruct",
        format_version="repo-card-summary-json-v1",
        source="generated",
    )


def _point(job: RepositoryJob, *, artifact: CardSummaryArtifact | None, stored_job: str) -> SimpleNamespace:
    payload = {
        "content_version": job.content_version,
        "content_job_id": stored_job,
        "content_hash": job.repository.content_hash,
        "embedding_version": "repo-embedding-v2",
    }
    if artifact is not None:
        payload.update(card_summary_payload(artifact))
    return SimpleNamespace(id=REPO_ID, payload=payload)


@pytest.mark.parametrize("same_job", [True, False], ids=["duplicate", "current"])
def test_duplicate_and_current_jobs_replay_the_durable_artifact(same_job: bool) -> None:
    request = _job()
    stored_job = str(request.job_id) if same_job else str(uuid.uuid4())
    pipeline = RepositoryEmbeddingPipeline(embedder=FakeEmbedder())
    artifact = _artifact()
    payload = _repository_embedding_payload(request)
    artifact._reuse_key = pipeline.card_summarizer._reuse_key(payload, payload).hex()
    point = _point(request, artifact=artifact, stored_job=stored_job)
    store = MagicMock()

    with patch("api.v2._repository_points", return_value=[point]), patch(
        "api.v2.repository_embedding_pipeline", return_value=pipeline
    ), patch("api.v2.repository_store", return_value=store):
        response = _embed_repository_job_locked(request, OwnedLock())

    assert response["schema_version"] == 2
    assert response["job_id"] == str(request.job_id)
    assert response["status"] == ("duplicate" if same_job else "current")
    assert response["card_summary"] == _artifact().model_dump(mode="json")
    store.compare_and_set_card_summary.assert_not_called()


def test_same_content_missing_summary_is_backfilled_without_replacing_content() -> None:
    request = _job()
    point = _point(request, artifact=None, stored_job=str(uuid.uuid4()))
    pipeline = RepositoryEmbeddingPipeline(embedder=FakeEmbedder())

    class Store:
        def compare_and_set_card_summary(self, *, expected_point, artifact):
            payload = dict(expected_point.payload)
            payload.update(card_summary_payload(artifact))
            return SimpleNamespace(id=expected_point.id, payload=payload)

    with patch("api.v2._repository_points", return_value=[point]), patch(
        "api.v2.repository_embedding_pipeline", return_value=pipeline
    ), patch("api.v2.repository_store", return_value=Store()):
        response = _embed_repository_job_locked(request, OwnedLock())

    assert response["status"] == "current"
    assert response["card_summary"]["source"] == "description_fallback"
    assert response["card_summary"]["summary"].startswith(
        "A review workspace for teams"
    )
    assert (
        response["card_summary"]["model_version"]
        == CARD_SUMMARY_FALLBACK_MODEL_VERSION
    )


def test_same_content_fallback_is_upgraded_after_provider_recovery(monkeypatch) -> None:
    from embedding.runtime import embedding_timeout_seconds
    from summarization.provider import OpenRouterSummaryProvider
    from summarization.settings import SummarySettings

    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "30")
    request = _job()
    fallback = CardSummaryPipeline().summarize(
        request.repository.model_dump(mode="json"),
        request.repository.model_dump(mode="json"),
    )
    point = _point(request, artifact=fallback, stored_job=str(uuid.uuid4()))
    generated_summary = (
        "A focused repository workspace helps maintainers review shared "
        "developer-tool changes with clear ownership, context, and durable "
        "decision history. Its coordinated workflow helps distributed teams "
        "catch risky updates before they disrupt downstream users or integrations."
    )

    settings = SummarySettings(api_key="isolated-summary-test-key", request_timeout_seconds=30)
    session = MagicMock()
    session.post.return_value.status_code = 200
    session.post.return_value.json.return_value = {
        "choices": [{"message": {"content": json.dumps({"summary": generated_summary})}}]
    }

    class Store:
        def compare_and_set_card_summary(self, *, expected_point, artifact):
            payload = dict(expected_point.payload)
            payload.update(card_summary_payload(artifact))
            return SimpleNamespace(id=expected_point.id, payload=payload)

    pipeline = RepositoryEmbeddingPipeline(
        embedder=FakeEmbedder(),
        card_summarizer=CardSummaryPipeline(
            settings=settings,
            provider=OpenRouterSummaryProvider(settings, session=session),
        ),
    )
    assert pipeline.card_summarizer.is_current(fallback) is True

    with patch("api.v2._repository_points", return_value=[point]), patch(
        "api.v2.repository_embedding_pipeline", return_value=pipeline
    ), patch("api.v2.repository_store", return_value=Store()):
        response = _embed_repository_job_locked(
            request, OwnedLock(), deadline=time.monotonic() + embedding_timeout_seconds()
        )

    assert response["status"] == "current"
    assert 0.1 < session.post.call_args.kwargs["timeout"] <= 20
    assert response["card_summary"]["source"] == "generated"
    assert response["card_summary"]["summary"] == generated_summary
    assert (
        response["card_summary"]["model_version"]
        == pipeline.card_summarizer.settings.model_id
    )


def test_durable_fallback_is_replayed_when_the_provider_cannot_fit_the_job_budget() -> None:
    request = _job()
    fallback = CardSummaryPipeline().summarize(
        request.repository.model_dump(mode="json"),
        request.repository.model_dump(mode="json"),
    )
    point = _point(request, artifact=fallback, stored_job=str(uuid.uuid4()))

    class UnexpectedProvider:
        def generate(self, *args, **kwargs):
            pytest.fail("a saved fallback must not wait through a full provider timeout")

    pipeline = RepositoryEmbeddingPipeline(
        embedder=FakeEmbedder(),
        card_summarizer=CardSummaryPipeline(
            provider=UnexpectedProvider(),
        ),
    )
    store = MagicMock()

    with patch("api.v2._repository_points", return_value=[point]), patch(
        "api.v2.repository_embedding_pipeline", return_value=pipeline
    ), patch("api.v2.repository_store", return_value=store), patch(
        "api.v2._repository_embedding_payload", side_effect=AssertionError("fallback replay must remain README-free")
    ):
        response = _embed_repository_job_locked(
            request,
            OwnedLock(),
            deadline=time.monotonic() + 5,
        )

    assert response["status"] == "current"
    assert response["card_summary"] == fallback.model_dump(mode="json")
    store.compare_and_set_card_summary.assert_not_called()


def test_same_version_different_content_cannot_attach_a_summary() -> None:
    request = _job(content_hash="requested")
    point = _point(request, artifact=None, stored_job=str(uuid.uuid4()))
    point.payload["content_hash"] = "stored"
    store = MagicMock()

    with patch("api.v2._repository_points", return_value=[point]), patch(
        "api.v2.repository_embedding_pipeline",
        return_value=RepositoryEmbeddingPipeline(embedder=FakeEmbedder()),
    ), patch("api.v2.repository_store", return_value=store):
        with pytest.raises(HTTPException) as exc_info:
            _embed_repository_job_locked(request, OwnedLock())

    assert exc_info.value.status_code == 409
    store.compare_and_set_card_summary.assert_not_called()


def test_same_version_missing_stored_content_hash_cannot_attach_a_summary() -> None:
    request = _job()
    point = _point(request, artifact=None, stored_job=str(uuid.uuid4()))
    point.payload.pop("content_hash")
    store = MagicMock()

    with patch("api.v2._repository_points", return_value=[point]), patch(
        "api.v2.repository_embedding_pipeline",
        return_value=RepositoryEmbeddingPipeline(embedder=FakeEmbedder()),
    ), patch("api.v2.repository_store", return_value=store):
        with pytest.raises(HTTPException) as exc_info:
            _embed_repository_job_locked(request, OwnedLock())

    assert exc_info.value.status_code == 409
    store.compare_and_set_card_summary.assert_not_called()


def test_artifact_hash_rejects_highlight_only_payload_tampering() -> None:
    payload = card_summary_payload(_artifact())
    payload["card_summary_highlights"] = ["Tampered highlight"]

    assert card_summary_from_payload(payload) is None


def test_applied_response_is_replayed_after_backend_crash() -> None:
    request = _job()
    pipeline = RepositoryEmbeddingPipeline(embedder=FakeEmbedder())
    state: dict[str, SimpleNamespace | None] = {"point": None}
    embed_calls = 0
    original_embed = pipeline.embed_repository

    def counted_embed(source, deadline=None):
        nonlocal embed_calls
        embed_calls += 1
        return original_embed(source, deadline=deadline)

    pipeline.embed_repository = counted_embed  # type: ignore[method-assign]

    class Store:
        def compare_and_set_content(self, result, *, expected_point):
            state["point"] = SimpleNamespace(id=REPO_ID, payload=dict(result.payload))
            return state["point"]

    with patch(
        "api.v2._repository_points",
        side_effect=lambda _repo_id: [] if state["point"] is None else [state["point"]],
    ), patch("api.v2.repository_embedding_pipeline", return_value=pipeline), patch(
        "api.v2.repository_store", return_value=Store()
    ):
        first = _embed_repository_job_locked(request, OwnedLock())
        retry = _embed_repository_job_locked(request, OwnedLock())

    assert first["status"] == "applied"
    assert retry["status"] == "duplicate"
    assert retry["card_summary"] == first["card_summary"]
    assert embed_calls == 1


def _qdrant_store() -> tuple[QdrantClient, QdrantRepositoryStore]:
    client = QdrantClient(":memory:")
    store = QdrantRepositoryStore(client=client)
    client.create_collection(
        store.collection_name,
        vectors_config={
            store.vector_name: models.VectorParams(
                size=store.vector_size,
                distance=models.Distance.COSINE,
            )
        },
    )
    return client, store


def test_qdrant_summary_cas_backfills_missing_artifact_and_fences_new_content() -> None:
    client, store = _qdrant_store()
    result = RepositoryEmbeddingPipeline(embedder=FakeEmbedder()).embed_repository(
        {
            "repo_id": REPO_ID,
            "full_name": "weave/qdrant-summary",
            "description": "A repository used to verify atomic summary updates.",
            "primary_language": "Python",
            "languages": ["Python"],
            "topics": ["qdrant"],
            "content_version": 1,
        }
    )
    result.payload["content_job_id"] = str(uuid.uuid4())
    store.upsert([result])
    client.delete_payload(
        store.collection_name,
        keys=list(CARD_SUMMARY_PAYLOAD_FIELDS),
        points=[REPO_ID],
        wait=True,
    )
    point = client.retrieve(store.collection_name, [REPO_ID], with_payload=True)[0]

    updated = store.compare_and_set_card_summary(
        expected_point=point,
        artifact=_artifact(),
    )
    assert card_summary_from_payload(updated.payload) == _artifact()

    stale = updated
    client.set_payload(
        store.collection_name,
        payload={"content_version": 2, "content_job_id": str(uuid.uuid4())},
        points=[REPO_ID],
        wait=True,
    )
    stale_write = store.compare_and_set_card_summary(
        expected_point=stale,
        artifact=_artifact("A different summary remains fenced from the newer content revision. Its text must not replace the artifact selected for the winning repository state."),
    )
    assert stale_write.payload["content_version"] == 2
    assert card_summary_from_payload(stale_write.payload) == _artifact()


def test_qdrant_summary_cas_fences_a_concurrent_highlight_update() -> None:
    client, store = _qdrant_store()
    result = RepositoryEmbeddingPipeline(embedder=FakeEmbedder()).embed_repository(
        {
            "repo_id": REPO_ID,
            "full_name": "weave/qdrant-summary-race",
            "description": "A repository used to verify atomic summary artifact updates.",
            "primary_language": "Python",
            "languages": ["Python"],
            "topics": ["qdrant"],
            "content_version": 1,
        }
    )
    result.payload["content_job_id"] = str(uuid.uuid4())
    store.upsert([result])
    stale = client.retrieve(store.collection_name, [REPO_ID], with_payload=True)[0]

    winning = store.compare_and_set_card_summary(
        expected_point=stale,
        artifact=_artifact(),
    )
    losing = store.compare_and_set_card_summary(
        expected_point=stale,
        artifact=CardSummaryArtifact(
            **{
                **_artifact().model_dump(),
                "highlights": ["Stale highlight"],
            }
        ),
    )

    assert card_summary_from_payload(winning.payload) == _artifact()
    assert card_summary_from_payload(losing.payload) == _artifact()


@pytest.mark.parametrize('description', [
    'A review workspace for teams maintaining shared developer tools.',
    'It supports common formats, e.g. JSON and YAML. Developers can extend it, e.g. with custom parsers.',
    'Tools detect GPL boundaries across projects. Teams can track policy changes.',
])
def test_provider_outage_allows_first_duplicate_and_current_fallback_deliveries(description) -> None:
    from summarization.provider import SummaryProviderError

    class UnavailableProvider:
        calls = 0

        def generate(self, source, *, repair_feedback=None, deadline=None):
            self.calls += 1
            raise SummaryProviderError("provider unavailable")

    provider = UnavailableProvider()
    pipeline = RepositoryEmbeddingPipeline(
        embedder=FakeEmbedder(),
        card_summarizer=CardSummaryPipeline(provider=provider),
    )
    client, store = _qdrant_store()
    request = _job()
    request.repository.description = description
    try:
        with patch("api.v2._repository_points", side_effect=lambda _id: client.retrieve(
            store.collection_name, [REPO_ID], with_payload=True
        )), patch("api.v2.repository_embedding_pipeline", return_value=pipeline), patch(
            "api.v2.repository_store", return_value=store
        ):
            first = _embed_repository_job_locked(request, OwnedLock())
            duplicate = _embed_repository_job_locked(request, OwnedLock())
            assert provider.calls == 1, "duplicate delivery must replay its stored result"
            next_request = _job()
            next_request.repository.description = description
            current = _embed_repository_job_locked(next_request, OwnedLock())
        assert [first["status"], duplicate["status"], current["status"]] == ["applied", "duplicate", "current"]
        assert first["card_summary"]["source"] == "description_fallback"
        assert first["card_summary"] == duplicate["card_summary"] == current["card_summary"]
        assert provider.calls == 2, "a separate delivery may attempt to upgrade a valid fallback"
    finally:
        client.close()


@pytest.mark.parametrize('same_job', [True, False], ids=['duplicate', 'current'])
def test_previously_saved_abbreviation_fallback_is_repaired(same_job):
    request = _job()
    request.repository.description = (
        'It supports common formats, e.g. JSON and YAML. '
        'Developers can extend it, e.g. with custom parsers.'
    )
    malformed = CardSummaryArtifact(**{
        **_artifact('It supports common formats, e.g. JSON and YAML.').model_dump(),
        'source': 'description_fallback',
        'model_version': CARD_SUMMARY_FALLBACK_MODEL_VERSION,
    })
    point = _point(request, artifact=malformed,
                   stored_job=str(request.job_id) if same_job else str(uuid.uuid4()))
    pipeline = RepositoryEmbeddingPipeline(embedder=FakeEmbedder())

    class Store:
        def compare_and_set_card_summary(self, *, expected_point, artifact):
            return SimpleNamespace(id=expected_point.id, payload={
                **expected_point.payload, **card_summary_payload(artifact),
            })

    with patch('api.v2._repository_points', return_value=[point]), patch(
        'api.v2.repository_embedding_pipeline', return_value=pipeline
    ), patch('api.v2.repository_store', return_value=Store()):
        response = _embed_repository_job_locked(request, OwnedLock())
    assert response['status'] == ('duplicate' if same_job else 'current')
    assert response['card_summary']['summary'] == request.repository.description


@pytest.mark.parametrize('same_job', [True, False], ids=['duplicate', 'current'])
@pytest.mark.parametrize('invalid_kind', ['GPL', 'quoted GPL', 'bracket GPL', 'highlight', 'unfinished', 'command', 'Markdown', 'copied'])
def test_invalid_fingerprinted_artifact_is_repaired_in_real_qdrant(same_job, invalid_kind):
    request = _job()
    pipeline = RepositoryEmbeddingPipeline(embedder=FakeEmbedder())
    repo = request.repository.model_dump(mode='json')
    result = pipeline.embed_repository(repo)
    old_artifact = pipeline.card_summarizer.summarize(repo, repo)
    values = old_artifact.model_dump()
    if invalid_kind == 'GPL':
        values['summary'] = 'GPL. Teams can redistribute the software.'
    elif invalid_kind in {'quoted GPL', 'bracket GPL'}:
        closer = '"' if invalid_kind == 'quoted GPL' else ']'
        values = _artifact(
            f'Maintainers describe a shared workspace for inspecting ownership decisions and coordinating repository changes.{closer} '
            'GPL. Teams can review package choices and keep their collaborative tooling aligned with project goals.'
        ).model_dump()
    elif invalid_kind == 'highlight':
        values['highlights'] = ['GPL v3 or later']
    elif invalid_kind == 'unfinished':
        values['summary'] = 'It supports shared workflows. Teams can review changes. Users share updates. Formats include, e.g.'
    elif invalid_kind == 'command':
        values['summary'] = 'Run npm install to use the tool. Teams can review changes.'
    elif invalid_kind == 'Markdown':
        values['summary'] = '# Review workspace. Teams can review changes.'
    else:
        values = _artifact().model_dump()
        request.repository.readme = values['summary']
        # Keep the complete canonical payload consistent with the copied source.
        result = pipeline.embed_repository(request.repository.model_dump(mode='json'))
    malformed = CardSummaryArtifact.model_validate(values)
    stored_job = str(request.job_id) if same_job else str(uuid.uuid4())
    result.payload['content_version'] = request.content_version
    result.payload['content_hash'] = request.repository.content_hash
    result.payload['content_job_id'] = stored_job
    result.payload.update(card_summary_payload(malformed))
    client, store = _qdrant_store()
    try:
        store.upsert([result])
        with patch('api.v2._repository_points', side_effect=lambda _id: client.retrieve(
            store.collection_name, [REPO_ID], with_payload=True
        )), patch('api.v2.repository_embedding_pipeline', return_value=pipeline), patch(
            'api.v2.repository_store', return_value=store
        ):
            repaired = _embed_repository_job_locked(request, OwnedLock())
            replayed = _embed_repository_job_locked(request, OwnedLock())
        assert repaired['status'] == ('duplicate' if same_job else 'current')
        assert repaired['card_summary'] != malformed.model_dump(mode='json')
        assert replayed['card_summary'] == repaired['card_summary']
        stored = client.retrieve(store.collection_name, [REPO_ID], with_payload=True)[0]
        current = card_summary_from_payload(stored.payload)
        assert current is not None
        assert pipeline.card_summarizer.is_current(current, repo=_repository_embedding_payload(request))
        assert stored.payload['content_version'] == request.content_version
        assert stored.payload['content_job_id'] == stored_job
    finally:
        client.close()


def test_unicode_boundary_artifact_survives_qdrant_persistence_and_replay():
    from pathlib import Path
    case = next(case for case in json.loads(
        Path('tests/fixtures/summary_compatibility.json').read_text()
    ) if case['name'] == '360 codepoints')

    class Provider:
        def generate(self, *args, **kwargs):
            return json.dumps({'summary': case['summary'], 'highlights': case['highlights']})

    pipeline = RepositoryEmbeddingPipeline(
        embedder=FakeEmbedder(), card_summarizer=CardSummaryPipeline(provider=Provider()),
    )
    client, store = _qdrant_store()
    request = _job()
    try:
        with patch('api.v2._repository_points', side_effect=lambda _id: client.retrieve(
            store.collection_name, [REPO_ID], with_payload=True
        )), patch('api.v2.repository_embedding_pipeline', return_value=pipeline), patch(
            'api.v2.repository_store', return_value=store
        ):
            responses = [
                _embed_repository_job_locked(request, OwnedLock()),
                _embed_repository_job_locked(request, OwnedLock()),
                _embed_repository_job_locked(_job(), OwnedLock()),
            ]
        assert [response['status'] for response in responses] == ['applied', 'duplicate', 'current']
        for response in responses:
            assert response['card_summary']['source'] == 'generated'
            assert response['card_summary']['summary'] == case['summary']
            assert response['card_summary']['highlights'] == case['highlights']
    finally:
        client.close()


@pytest.mark.parametrize("concurrent_summary", [False, True])
def test_pre_summary_point_refreshes_without_erasing_a_concurrent_summary(concurrent_summary) -> None:
    client, store = _qdrant_store()
    pipeline = RepositoryEmbeddingPipeline(embedder=FakeEmbedder())
    result = pipeline.embed_repository(_job().repository.model_dump(mode="json"))
    store.upsert([result])
    client.delete_payload(store.collection_name, keys=list(CARD_SUMMARY_PAYLOAD_FIELDS), points=[REPO_ID], wait=True)
    expected = client.retrieve(store.collection_name, [REPO_ID], with_payload=True)[0]
    request = RepositoryRefreshJob(
        schema_version=2, job_id=uuid.uuid4(), repo_id=REPO_ID,
        feature_version=2, features={"star_count": 42},
    )
    original_cas = store.compare_and_set_features

    def refresh_cas(*, expected_point, feature_payload):
        assert not set(CARD_SUMMARY_PAYLOAD_FIELDS).intersection(feature_payload)
        if concurrent_summary:
            store.compare_and_set_card_summary(expected_point=expected_point, artifact=_artifact())
        return original_cas(expected_point=expected_point, feature_payload=feature_payload)

    try:
        with patch("api.v2._repository_points", return_value=[expected]), patch(
            "api.v2.repository_store", return_value=store
        ), patch.object(store, "compare_and_set_features", side_effect=refresh_cas):
            response = _refresh_repository_job_locked(request, OwnedLock())
        stored = client.retrieve(store.collection_name, [REPO_ID], with_payload=True)[0]
        assert response["status"] == "applied"
        assert stored.payload["star_count"] == 42
        assert stored.payload["activity_score"] == 0.0
        assert stored.payload["feature_version"] == 2
        assert card_summary_from_payload(stored.payload) == (_artifact() if concurrent_summary else None)
        assert not set(CARD_SUMMARY_PAYLOAD_FIELDS).intersection(expected.payload)
    finally:
        client.close()


def _indexed_feature_refresh_fixture():
    body = _job().model_dump(mode="json")
    body["repository"].update(
        {
            "fork_count": 2,
            "open_issues_count": 4,
            "pushed_days_ago": 30,
            "mentionable_users_count": 4,
            "delta_3d": 3,
            "delta_7d": 7,
            "delta_30d": 30,
            "recent_commits": [datetime.now(timezone.utc).isoformat()],
            "updated_at": "2026-01-02T00:00:00Z",
            "pushed_at": "2026-01-01T00:00:00Z",
        }
    )
    pipeline = RepositoryEmbeddingPipeline(embedder=FakeEmbedder())
    result = pipeline.embed_repository(
        RepositoryJob.model_validate(body).repository.model_dump(mode="json")
    )
    client, store = _qdrant_store()
    store.upsert([result])
    point = client.retrieve(
        store.collection_name,
        [REPO_ID],
        with_payload=True,
        with_vectors=True,
    )[0]
    return client, store, point


def test_trend_velocity_clamps_decline_and_preserves_positive_scale() -> None:
    assert trend_velocity(
        {"delta_3d": -30, "delta_7d": 0, "delta_30d": 0}
    ) == 0.0
    assert trend_velocity(
        {"delta_3d": 30, "delta_7d": 7, "delta_30d": 30}
    ) == 0.4511


@pytest.mark.parametrize(
    ("feature_patch", "affected_scores"),
    [
        ({"fork_count": 8, "open_issues_count": 2}, {"code_health"}),
        ({"pushed_days_ago": 0}, {"code_health", "activity_score"}),
        ({"delta_3d": 30}, {"trend_velocity"}),
        (
            {"delta_3d": -30, "delta_7d": 0, "delta_30d": 0},
            {"trend_velocity"},
        ),
        ({"mentionable_users_count": 4}, set()),
        ({"mentionable_users_count": 99}, set()),
        ({"star_count": 42}, set()),
        (
            {
                "updated_at": "2026-06-02T00:00:00Z",
                "pushed_at": "2026-06-01T00:00:00Z",
            },
            set(),
        ),
    ],
    ids=[
        "counts",
        "recency",
        "positive-delta",
        "negative-delta",
        "unchanged-legacy-counter",
        "changed-legacy-counter",
        "unrelated-count",
        "timestamps",
    ],
)
def test_feature_refresh_derives_only_dependent_scores_without_reembedding(
    feature_patch, affected_scores,
) -> None:
    client, store, expected = _indexed_feature_refresh_fixture()
    request = RepositoryRefreshJob(
        schema_version=2,
        job_id=uuid.uuid4(),
        repo_id=REPO_ID,
        feature_version=1,
        features=feature_patch,
    )
    embedding_pipeline = MagicMock()

    try:
        with patch("api.v2._repository_points", return_value=[expected]), patch(
            "api.v2.repository_store", return_value=store
        ), patch(
            "api.v2.repository_embedding_pipeline", embedding_pipeline
        ):
            response = _refresh_repository_job_locked(request, OwnedLock())
        stored = client.retrieve(
            store.collection_name,
            [REPO_ID],
            with_payload=True,
            with_vectors=True,
        )[0]
        resulting_observation = {**expected.payload, **feature_patch}
        expected_scores = {
            field: expected.payload[field]
            for field in ("code_health", "activity_score", "trend_velocity")
        }
        if "code_health" in affected_scores:
            expected_scores["code_health"] = score_code_health(resulting_observation)
        if "activity_score" in affected_scores:
            expected_scores["activity_score"] = activity_score(
                {**resulting_observation, "recent_commits": []}
            )
        if "trend_velocity" in affected_scores:
            expected_scores["trend_velocity"] = trend_velocity(
                resulting_observation
            )

        assert response["status"] == "applied"
        assert expected.payload["activity_score"] != activity_score(
            {**expected.payload, "recent_commits": []}
        )
        for field, score in expected_scores.items():
            assert stored.payload[field] == score
        assert stored.payload["pushed_days_ago"] == feature_patch.get(
            "pushed_days_ago", expected.payload["pushed_days_ago"]
        )
        assert stored.vector == expected.vector
        embedding_pipeline.assert_not_called()
    finally:
        client.close()


def test_unrelated_refresh_does_not_evaluate_signed_trend_observations() -> None:
    client, store, _ = _indexed_feature_refresh_fixture()
    signed_deltas = {
        "delta_3d": -2_147_483_648,
        "delta_7d": -2_147_483_648,
        "delta_30d": -2_147_483_648,
    }
    client.set_payload(
        store.collection_name,
        payload=signed_deltas,
        points=[REPO_ID],
        wait=True,
    )
    expected = client.retrieve(
        store.collection_name,
        [REPO_ID],
        with_payload=True,
        with_vectors=True,
    )[0]
    request = RepositoryRefreshJob(
        schema_version=2,
        job_id=uuid.uuid4(),
        repo_id=REPO_ID,
        feature_version=1,
        features={"star_count": 42},
    )

    try:
        with patch("api.v2._repository_points", return_value=[expected]), patch(
            "api.v2.repository_store", return_value=store
        ), patch("api.v2.calculate_trend_velocity") as trend_calculator:
            response = _refresh_repository_job_locked(request, OwnedLock())
        stored = client.retrieve(
            store.collection_name,
            [REPO_ID],
            with_payload=True,
        )[0]

        assert response["status"] == "applied"
        assert stored.payload["trend_velocity"] == expected.payload["trend_velocity"]
        trend_calculator.assert_not_called()
    finally:
        client.close()


def test_feature_refresh_rejects_inconsistent_caller_derived_scores() -> None:
    client, store, expected = _indexed_feature_refresh_fixture()
    request = RepositoryRefreshJob(
        schema_version=2,
        job_id=uuid.uuid4(),
        repo_id=REPO_ID,
        feature_version=1,
        features={"pushed_days_ago": 0, "activity_score": 0.1234},
    )

    try:
        with patch("api.v2._repository_points", return_value=[expected]), patch(
            "api.v2.repository_store", return_value=store
        ), pytest.raises(HTTPException) as exc_info:
            _refresh_repository_job_locked(request, OwnedLock())

        assert exc_info.value.status_code == 422
        assert exc_info.value.detail == (
            "Repository derived features are inconsistent with the resulting "
            "observation; expected activity_score=1.0."
        )
        stored = client.retrieve(
            store.collection_name,
            [REPO_ID],
            with_payload=True,
        )[0]
        assert "feature_version" not in stored.payload
    finally:
        client.close()


@pytest.mark.parametrize("echo_kind", ["recomputed", "preserved"])
def test_feature_refresh_accepts_matching_derived_score_echo(echo_kind) -> None:
    client, store, expected = _indexed_feature_refresh_fixture()
    if echo_kind == "recomputed":
        feature_patch = {"pushed_days_ago": 0, "activity_score": 1.0}
        expected_activity = 1.0
    else:
        expected_activity = expected.payload["activity_score"]
        feature_patch = {
            "star_count": 42,
            "activity_score": expected_activity,
        }
    request = RepositoryRefreshJob(
        schema_version=2,
        job_id=uuid.uuid4(),
        repo_id=REPO_ID,
        feature_version=1,
        features=feature_patch,
    )

    try:
        with patch("api.v2._repository_points", return_value=[expected]), patch(
            "api.v2.repository_store", return_value=store
        ):
            response = _refresh_repository_job_locked(request, OwnedLock())
        stored = client.retrieve(
            store.collection_name,
            [REPO_ID],
            with_payload=True,
        )[0]

        assert response["status"] == "applied"
        assert stored.payload["activity_score"] == expected_activity
    finally:
        client.close()


def test_unrelated_refresh_validates_echo_against_preserved_score() -> None:
    client, store, expected = _indexed_feature_refresh_fixture()
    request = RepositoryRefreshJob(
        schema_version=2,
        job_id=uuid.uuid4(),
        repo_id=REPO_ID,
        feature_version=1,
        features={"star_count": 42, "activity_score": 0.0},
    )

    try:
        with patch("api.v2._repository_points", return_value=[expected]), patch(
            "api.v2.repository_store", return_value=store
        ), pytest.raises(HTTPException) as exc_info:
            _refresh_repository_job_locked(request, OwnedLock())

        assert exc_info.value.status_code == 422
        assert exc_info.value.detail.endswith(
            f"expected activity_score={expected.payload['activity_score']}."
        )
    finally:
        client.close()


def test_refresh_http_error_for_score_echo_matches_validation_contract(
    monkeypatch,
) -> None:
    client, store, expected = _indexed_feature_refresh_fixture()
    monkeypatch.setenv("INTERNAL_API_SECRET", "test-internal-secret")
    request_id = "refresh-score-contract"

    try:
        with patch("api.v2._repository_points", return_value=[expected]), patch(
            "api.v2.repository_store", return_value=store
        ), patch(
            "api.v2._repository_job_lock",
            return_value=nullcontext(OwnedLock()),
        ):
            response = TestClient(api_main.app).post(
                "/api/v2/repositories/refresh",
                headers={
                    "x-internal-secret": "test-internal-secret",
                    "x-request-id": request_id,
                },
                json={
                    "schema_version": 2,
                    "job_id": str(uuid.uuid4()),
                    "repo_id": REPO_ID,
                    "feature_version": 1,
                    "features": {
                        "pushed_days_ago": 0,
                        "activity_score": 0.1234,
                    },
                },
            )

        assert response.status_code == 422
        assert response.headers["x-request-id"] == request_id
        body = response.json()
        assert body["code"] == "REQUEST_VALIDATION_FAILED"
        assert body["retryable"] is False
        assert body["request_id"] == request_id
        assert body["detail"] == body["message"]
        assert body["details"] == [
            {
                "path": "body.features.activity_score",
                "message": "Value must match the effective server-owned score 1.0.",
                "type": "value_error.derived_feature_mismatch",
            }
        ]
    finally:
        client.close()


def test_feature_refresh_still_rejects_missing_core_payload_fields() -> None:
    pipeline = RepositoryEmbeddingPipeline(embedder=FakeEmbedder())
    result = pipeline.embed_repository(_job().repository.model_dump(mode="json"))
    result.payload.pop("star_count")
    from embedding.vector_contract import REPOSITORY_SERVING_ELIGIBILITY_FIELD, REPOSITORY_SERVING_ELIGIBILITY_VERSION
    result.payload[REPOSITORY_SERVING_ELIGIBILITY_FIELD] = REPOSITORY_SERVING_ELIGIBILITY_VERSION
    point = SimpleNamespace(id=REPO_ID, payload=result.payload)
    request = RepositoryRefreshJob(
        schema_version=2, job_id=uuid.uuid4(), repo_id=REPO_ID,
        feature_version=2, features={"pushed_days_ago": 5},
    )
    with patch("api.v2._repository_points", return_value=[point]), pytest.raises(ValueError, match="star_count"):
        _refresh_repository_job_locked(request, OwnedLock())


def test_paragraph_only_repository_job_supplies_readme_context_to_summary_provider() -> None:
    body = _job().model_dump(mode="json")
    body["repository"].pop("readme")
    paragraphs = ["A collaborative review workspace keeps design proposals and ownership decisions together for maintainers across many shared software projects."]
    body["repository"]["extracted_paragraphs"] = paragraphs
    request = RepositoryJob.model_validate(body)
    captured = []

    class Provider:
        def generate(self, source, *, repair_feedback=None, deadline=None):
            captured.append(source)
            return json.dumps({"summary": (
                "A focused repository workspace helps maintainers review shared developer-tool changes "
                "with clear ownership and durable decision history. Its coordinated workflow helps "
                "distributed teams catch risky updates before they disrupt downstream integrations."
            )})

    pipeline = RepositoryEmbeddingPipeline(embedder=FakeEmbedder(), card_summarizer=CardSummaryPipeline(provider=Provider()))
    client, store = _qdrant_store()
    try:
        with patch("api.v2._repository_points", return_value=[]), patch(
            "api.v2.repository_embedding_pipeline", return_value=pipeline
        ), patch("api.v2.repository_store", return_value=store):
            result = _embed_repository_job_locked(request, OwnedLock())
        assert result["status"] == "applied"
        assert result["card_summary"]["source"] == "generated"
        assert paragraphs[0] in captured[0]
    finally:
        client.close()


@pytest.mark.parametrize('metadata', [
    {'full_name': 'acme/license-checker', 'primary_language': 'Python'},
    {'full_name': 'acme/tool', 'primary_language': 'Unknown', 'topics': ['reference']},
])
def test_metadata_fallback_without_provider_persists_and_replays(metadata):
    pipeline = RepositoryEmbeddingPipeline(embedder=FakeEmbedder(), card_summarizer=CardSummaryPipeline())
    client, store = _qdrant_store()
    request = _job()
    request.repository.description = ''
    for name, value in metadata.items():
        setattr(request.repository, name, value)
    try:
        with patch('api.v2._repository_points', side_effect=lambda _id: client.retrieve(
            store.collection_name, [REPO_ID], with_payload=True
        )), patch('api.v2.repository_embedding_pipeline', return_value=pipeline), patch(
            'api.v2.repository_store', return_value=store
        ):
            first = _embed_repository_job_locked(request, OwnedLock())
            duplicate = _embed_repository_job_locked(request, OwnedLock())
            request.job_id = uuid.uuid4()
            current = _embed_repository_job_locked(request, OwnedLock())
        assert [first['status'], duplicate['status'], current['status']] == ['applied', 'duplicate', 'current']
        assert first['card_summary'] == duplicate['card_summary'] == current['card_summary']
        assert pipeline.card_summarizer.is_current(CardSummaryArtifact.model_validate(first['card_summary']))
    finally:
        client.close()
