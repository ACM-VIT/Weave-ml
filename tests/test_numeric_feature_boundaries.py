"""Numeric edge cases use real V2 validation and embedding, with no model I/O."""
import math
import uuid

import pytest

from api.contracts import RepositoryJob
from config import ZERO_RECENCY_EMBEDDING_VERSION
from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
from embedding.repository_embedding import RepositoryEmbeddingConfig, build_metadata_text
from ingestion.features import activity_score, trend_velocity
from summarization.pipeline import description_fallback_artifact


@pytest.mark.parametrize("value, expected", [(0, 0), (1, 1), (None, 999)])
def test_metadata_preserves_zero_separately_from_missing_recency(value, expected):
    assert f"Pushed days ago: {expected}\n" in build_metadata_text({"pushed_days_ago": value})
    assert "Pushed days ago: 999\n" in build_metadata_text({})


@pytest.mark.parametrize("deltas", [
    (0, 0, 0), (3, 7, 30), (2_147_483_647,) * 3,
    (-1, -1, -1), (-30, -70, -300), (-2_147_483_648,) * 3,
    (-3, 70, 30), (30, -700, 30),
])
def test_signed_trends_produce_a_finite_growth_score(deltas):
    repo = dict(zip(("delta_3d", "delta_7d", "delta_30d"), deltas))
    score = trend_velocity(repo)
    assert math.isfinite(score) and 0 <= score <= 1
    blend = 0.5 * deltas[0] / 3 + 0.3 * deltas[1] / 7 + 0.2 * deltas[2] / 30
    if blend <= 0:
        assert score == 0, "a growth-only feature must not treat decline as popularity"


def test_positive_growth_keeps_the_existing_log_scale_and_acceleration():
    assert trend_velocity({"delta_3d": 3, "delta_7d": 7, "delta_30d": 30}) == round(math.log1p(1) / math.log1p(500), 4)
    blend = 0.5 * 10 + 0.3 * 1 + 0.2 * 1
    assert trend_velocity({"delta_3d": 30, "delta_7d": 7, "delta_30d": 30}) == round(math.log1p(blend) / math.log1p(500) + 0.15, 4)


class Encoder:
    def __init__(self):
        self.texts = []

    def embed_texts(self, texts, **_options):
        self.texts.extend(texts)
        return [[1.0] + [0.0] * 383 for _ in texts]


class Summary:
    def summarize(self, _source, repo, **_options):
        return description_fallback_artifact(repo)


def job():
    return RepositoryJob.model_validate({
        "schema_version": 2, "job_id": str(uuid.uuid4()), "repo_id": str(uuid.uuid4()), "content_version": 1,
        "repository": {"full_name": "owner/declining", "content_hash": "numeric-boundary-fixture",
                       "pushed_days_ago": 0, "delta_3d": -30, "delta_7d": -70, "delta_30d": -300},
    })


@pytest.mark.parametrize("version, recency", [("repo-embedding-v2", 999), (ZERO_RECENCY_EMBEDDING_VERSION, 0)])
def test_valid_declining_v2_job_embeds_and_semantic_correction_is_explicitly_versioned(version, recency):
    request = job()
    source = request.repository.model_dump(mode="json")
    source.update(repo_id=str(request.repo_id), content_version=request.content_version)
    encoder = Encoder()
    result = RepositoryEmbeddingPipeline(config=RepositoryEmbeddingConfig(version=version),
        embedder=encoder, card_summarizer=Summary()).embed_repository(source)
    assert result.payload["pushed_days_ago"] == 0
    assert result.payload["delta_3d"] == -30 and result.payload["delta_7d"] == -70 and result.payload["delta_30d"] == -300
    assert result.payload["trend_velocity"] == 0
    assert result.embedding_version == version
    assert any(f"Pushed days ago: {recency}\n" in text for text in encoder.texts)
    assert all(math.isfinite(value) for value in result.final_embedding)


@pytest.mark.parametrize("owner", [None, {"login": "one-maintainer", "databaseId": 7},
                                  {"login": "an-organization", "databaseId": 8, "__typename": "Organization"}])
def test_enrichment_leaves_uncollected_contributor_evidence_unknown(owner):
    from acquisition.repository_enricher import RepositoryEnricher
    from acquisition.backend_client import repository_upsert_record
    data = {"nameWithOwner": "fixture/repository", "name": "repository", "databaseId": 123,
            "id": "R_fixture123", "url": "https://github.com/fixture/repository", "owner": owner,
            "pushedAt": "2026-10-24T00:00:00Z", "defaultBranchRef": {"name": "main", "target": {"history": {"nodes": []}}}}
    # A client without methods proves this fixture cannot collect contributor
    # data, request a new credential or add another GitHub API call.
    result = RepositoryEnricher(graphql_client=object())._process_graphql_data(data, None, None, readme_text="# Fixture")
    assert "mentionable_users_count" not in result.payload
    if owner:
        assert "mentionable_users_count" not in repository_upsert_record(result)
    else:
        with pytest.raises(ValueError, match="owner_github_id"):
            repository_upsert_record(result)  # Existing canonical identity requirement.
    assert result.payload["owner_github_id"] == (str(owner["databaseId"]) if owner else None), "identity remains separate from contributor evidence"


@pytest.mark.parametrize("days,expected", [(0, 1.0), (30, 0.5), (60, 0.25), (999, 0.0)])
@pytest.mark.parametrize("legacy_count", [None, 0, 1, 100])
def test_activity_fallback_uses_push_recency_without_owner_or_legacy_count(days, expected, legacy_count):
    assert activity_score({"pushed_days_ago": days, "recent_commits": [],
                           "mentionable_users_count": legacy_count}) == expected
    assert activity_score({"pushed_days_ago": days}) == expected


def test_observed_commits_remain_independent_of_the_legacy_counter():
    from datetime import datetime, timezone
    observed = {"recent_commits": [datetime.now(timezone.utc).isoformat()]}
    assert activity_score(observed) > 0
    assert activity_score({**observed, "mentionable_users_count": 0}) == activity_score({**observed, "mentionable_users_count": 100})
