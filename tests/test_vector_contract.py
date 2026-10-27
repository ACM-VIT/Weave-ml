"""Phase 1 tests for the vector-platform contract."""

import pytest

from config import (
    QDRANT_COLLECTION_NAME,
    QDRANT_DISTANCE,
    QDRANT_VECTOR_NAME,
    REPOSITORY_EMBEDDING_DIM,
    REPOSITORY_EMBEDDING_MODEL,
    USER_PROFILES_COLLECTION_NAME,
)
from embedding.qdrant_store import QdrantRepositoryStore
from embedding.repository_embedding import RepositoryEmbeddingConfig, build_vector_payload
from embedding.vector_contract import (
    REPOSITORY_COLLECTION_CONTRACT,
    REPOSITORY_PAYLOAD_FIELD_TYPES,
    REPOSITORY_PAYLOAD_REQUIRED_FIELDS,
    REPOSITORY_SERVING_ELIGIBILITY_FIELD,
    REPOSITORY_SERVING_ELIGIBILITY_VERSION,
    USER_PROFILE_COLLECTION_CONTRACT,
    canonical_backend_uuid,
    repository_payload_defaults,
    repository_point_id,
    user_point_id,
)

REPO_ID = "00000000-0000-4000-8000-000000000001"
OTHER_REPO_ID = "00000000-0000-4000-8000-000000000002"


def test_offline_lookahead_snapshots_mutable_topics():
    import hashlib
    from embedding.embedding_pipeline import RepositoryEmbeddingPipeline

    class Encoder:
        def embed_texts(self, texts):
            vectors = []
            for text in texts:
                vector = [0.0] * REPOSITORY_EMBEDDING_DIM
                vector[int.from_bytes(hashlib.sha256(text.encode()).digest()[:2]) % len(vector)] = 1.0
                vectors.append(vector)
            return vectors

    pipeline = RepositoryEmbeddingPipeline(embedder=Encoder())
    sources = [
        {"repo_id": REPO_ID, "full_name": "owner/first", "topics": ["search"]},
        {"repo_id": OTHER_REPO_ID, "full_name": "owner/second", "topics": ["storage"]},
    ]
    expected = [pipeline.embed_repository(source) for source in sources]
    shared_topics = []

    def reused_rows():
        for source in sources:
            shared_topics[:] = source["topics"]
            yield {**source, "topics": shared_topics}

    actual = list(pipeline.iter_embed(reused_rows()))
    assert [result.payload["topics"] for result in actual] == [["search"], ["storage"]]
    assert [result.source_hash for result in actual] == [result.source_hash for result in expected]
    assert [result.final_embedding for result in actual] == [result.final_embedding for result in expected]


@pytest.mark.parametrize(
    "value", [b"x" * 1_100_000, bytearray(b"large"), object(), 1 << 10000],
    ids=["large-bytes", "bytearray", "object", "large-integer"],
)
def test_offline_budget_rejects_unbounded_values_before_preparing_text(value, monkeypatch):
    from embedding.embedding_pipeline import RepositoryEmbeddingPipeline

    pipeline = RepositoryEmbeddingPipeline()
    monkeypatch.setattr(pipeline, "_prepare_repository", lambda *args, **kwargs: pytest.fail("prepared rejected source"))
    with pytest.raises((TypeError, ValueError), match="scalar"):
        next(pipeline.iter_embed([{
            "repo_id": REPO_ID, "full_name": "owner/repo", "extracted_paragraphs": [value],
        }]))


def test_cancelled_offline_job_does_not_advance_lazy_source():
    from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
    from utils.deadline import CooperativeDeadlineExceeded, RequestDeadline, deadline_scope

    fetched = []

    def sources():
        fetched.append(True)
        yield {"repo_id": REPO_ID, "full_name": "owner/repo"}

    deadline = RequestDeadline(30)
    deadline.cancel()
    with deadline_scope(deadline), pytest.raises(CooperativeDeadlineExceeded):
        next(RepositoryEmbeddingPipeline().iter_embed(sources()))
    assert fetched == []


@pytest.fixture
def offline_embedding_pipeline():
    import hashlib
    import numpy as np
    from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
    from embedding.embeddings import SentenceTransformerEmbedder
    from embedding.tower_cache import TowerEmbeddingCache
    from config import EMBEDDING_MODEL_REVISION
    from summarization.pipeline import CardSummaryPipeline
    from summarization.settings import SummarySettings

    class Model:
        max_seq_length = 256
        def __init__(self):
            self.calls = []
        def encode(self, texts, **kwargs):
            self.calls.append(tuple(texts))
            vectors = np.zeros((len(texts), REPOSITORY_EMBEDDING_DIM), dtype=np.float32)
            for index, text in enumerate(texts):
                vectors[index, int.from_bytes(hashlib.sha256(text.encode()).digest()[:2]) % REPOSITORY_EMBEDDING_DIM] = 1
            return vectors

    model = Model()
    pipeline = RepositoryEmbeddingPipeline(
        embedder=SentenceTransformerEmbedder(model=model),
        tower_cache=TowerEmbeddingCache(model=model, model_revision=EMBEDDING_MODEL_REVISION),
        card_summarizer=CardSummaryPipeline(settings=SummarySettings(api_key=None)),
    )
    return pipeline, model


def _offline_sources(count):
    import uuid
    for number in range(1, count + 1):
        yield dict(repo_id=str(uuid.UUID(int=number)), full_name=f"diagnostic/repo{number}",
            description=f"Repository {number} helps search local documents.",
            topics=["search"] if number % 2 else [],
            readme=("# Search\nLocal document retrieval. " * number) if number % 3 else "")


def test_streamed_towers_match_serial_empty_mixed_sources_and_cache(offline_embedding_pipeline):
    import math
    pipeline, model = offline_embedding_pipeline
    sources = list(_offline_sources(9))
    serial = [pipeline.embed_repository(row) for row in sources]
    # The exact model is warmed, but summaries/payloads must still be rebuilt.
    model.calls.clear()
    streamed = list(pipeline.iter_embed(iter(sources)))
    materialized = pipeline.embed_batch(iter(sources))
    assert model.calls == []
    assert serial[2].readme_embedding == [] and serial[0].readme_embedding
    assert serial[0].card_summary.source == "description_fallback"
    # Cold streaming exercises grouping rather than merely cached outputs.
    pipeline.tower_cache = None
    cold = list(pipeline.iter_embed(iter(sources)))
    expected_ids = [row["repo_id"] for row in sources]
    for results in (serial, streamed, materialized, cold):
        assert len(results) == len(sources)
        assert [row.repo_id for row in results] == expected_ids
        for expected, actual in zip(serial, results, strict=True):
            for field in ("readme_embedding", "metadata_embedding", "topic_embedding", "final_embedding"):
                assert getattr(actual, field) == pytest.approx(getattr(expected, field), abs=1e-7)
            assert actual.source_hash == expected.source_hash
            assert actual.card_summary == expected.card_summary
            # Execution time is the only deliberately different payload field.
            assert {key: value for key, value in actual.payload.items() if key != "indexed_at"} == {
                key: value for key, value in expected.payload.items() if key != "indexed_at"}
            assert math.fsum(value * value for value in actual.final_embedding) == pytest.approx(1)
    assert all(len(group) <= 8 and len(group) * model.max_seq_length <= 2048 for group in model.calls)
    assert any(len(group) == 8 for group in model.calls)


def test_stream_index_ambiguous_chunk_resumes_from_acknowledged_progress(offline_embedding_pipeline, monkeypatch):
    from contextlib import closing
    from qdrant_client import QdrantClient, models
    from embedding.embedding_pipeline import MAX_OFFLINE_REPOSITORIES
    pipeline, _ = offline_embedding_pipeline
    with closing(QdrantClient(":memory:")) as client:
        client.create_collection("offline", vectors_config={QDRANT_VECTOR_NAME: models.VectorParams(
            size=384, distance=models.Distance.COSINE)})
        store = QdrantRepositoryStore(client=client, collection_name="offline", corpus_authority=None)
        # Embedded Qdrant has no payload indexes; this case covers actual point writes,
        # validation and replay. Server bootstrap/index behavior has separate coverage.
        monkeypatch.setattr(store, "ensure_collection", store._validate_collection)
        pipeline.store = store
        original_upsert = client.upsert
        writes, pulled, acknowledged = [], [], []
        def write_then_lose_reply(**kwargs):
            writes.append([str(point.id) for point in kwargs["points"]])
            response = original_upsert(**kwargs)
            if len(writes) == 2:
                raise ConnectionError("controlled lost reply after stored chunk")
            return response
        monkeypatch.setattr(client, "upsert", write_then_lose_reply)
        def sources():
            for row in _offline_sources(17):
                pulled.append(row["repo_id"])
                yield row
        iterator = pipeline.iter_index(sources())
        acknowledged.append(next(iterator))
        assert len(pulled) == MAX_OFFLINE_REPOSITORIES
        with pytest.raises(ConnectionError, match="lost reply"):
            acknowledged.extend(iterator)
        assert len(acknowledged) == 8 and len(pulled) == 16
        assert client.count("offline", exact=True).count == 16
        assert all(len(chunk) <= 8 for chunk in writes)
        monkeypatch.setattr(client, "upsert", original_upsert)
        remaining = list(_offline_sources(17))[len(acknowledged):]
        resumed = list(pipeline.iter_index(iter(remaining)))
        assert [row.repo_id for row in acknowledged + resumed] == [row["repo_id"] for row in _offline_sources(17)]
        assert client.count("offline", exact=True).count == 17
        records = client.retrieve("offline", ids=[row.repo_id for row in resumed], with_vectors=True)
        by_id = {str(row.id): row for row in records}
        for result in resumed:
            assert by_id[result.repo_id].vector[QDRANT_VECTOR_NAME] == pytest.approx(result.final_embedding, abs=1e-7)
            assert by_id[result.repo_id].payload["source_hash"] == result.source_hash


def test_stream_store_bounds_actual_utf8_requests_with_skewed_payloads(offline_embedding_pipeline):
    from types import SimpleNamespace
    from qdrant_client.http.api.points_api import jsonable_encoder
    from qdrant_client import models
    pipeline, _ = offline_embedding_pipeline
    results = list(pipeline.iter_embed(_offline_sources(5)))
    results[1].payload["diagnostic_unicode"] = "界" * 1500
    requests = []
    def upsert(**kwargs):
        requests.append(kwargs["points"])
        return SimpleNamespace(status="completed")
    store = QdrantRepositoryStore(client=SimpleNamespace(upsert=upsert), corpus_authority=None)
    progress = list(store.iter_upsert(iter(results), max_points=64, max_request_bytes=16000))
    assert progress == results
    # Five points fit the count cap; only their actual encoded byte size splits them.
    assert len(requests) >= 2
    assert [str(point.id) for chunk in requests for point in chunk] == [row.repo_id for row in results]
    assert all(len(chunk) <= 64 and len(jsonable_encoder(models.PointsList(points=chunk)).encode("utf-8")) <= 16000
               for chunk in requests)


def test_repository_collection_contract_matches_central_config():
    assert REPOSITORY_COLLECTION_CONTRACT.collection_name == QDRANT_COLLECTION_NAME
    assert REPOSITORY_COLLECTION_CONTRACT.vector_name == QDRANT_VECTOR_NAME
    assert REPOSITORY_COLLECTION_CONTRACT.vector_size == REPOSITORY_EMBEDDING_DIM == 384
    assert REPOSITORY_COLLECTION_CONTRACT.distance == QDRANT_DISTANCE == "Cosine"
    assert REPOSITORY_COLLECTION_CONTRACT.model_name == REPOSITORY_EMBEDDING_MODEL


def test_user_collection_contract_preserves_existing_unnamed_vector():
    assert USER_PROFILE_COLLECTION_CONTRACT.collection_name == USER_PROFILES_COLLECTION_NAME
    assert USER_PROFILE_COLLECTION_CONTRACT.vector_name is None
    assert USER_PROFILE_COLLECTION_CONTRACT.vector_size == REPOSITORY_EMBEDDING_DIM
    assert USER_PROFILE_COLLECTION_CONTRACT.distance == QDRANT_DISTANCE
    assert USER_PROFILE_COLLECTION_CONTRACT.model_name == REPOSITORY_EMBEDDING_MODEL


def test_point_ids_are_canonical_backend_uuids():
    assert repository_point_id(REPO_ID) == repository_point_id(f" {REPO_ID} ")
    assert user_point_id(REPO_ID) == user_point_id(f" {REPO_ID} ")
    assert repository_point_id(REPO_ID) == user_point_id(REPO_ID) == REPO_ID
    assert repository_point_id(REPO_ID) != repository_point_id(OTHER_REPO_ID)


def test_backend_ids_are_canonical_valid_uuids():
    uppercase_uuid = "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"
    assert canonical_backend_uuid(uppercase_uuid, field_name="repo_id") == uppercase_uuid.lower()
    with pytest.raises(ValueError, match="backend-issued UUID"):
        canonical_backend_uuid("owner/repository", field_name="repo_id")


@pytest.mark.parametrize("value", ["", "   ", "repo-123", "owner/repository"])
def test_point_ids_reject_empty_identifiers(value):
    with pytest.raises(ValueError):
        repository_point_id(value)
    with pytest.raises(ValueError):
        user_point_id(value)


def test_point_ids_reject_non_string_identifiers():
    with pytest.raises(TypeError):
        repository_point_id(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        user_point_id(None)  # type: ignore[arg-type]


def test_store_uses_the_published_repository_point_id_helper():
    assert QdrantRepositoryStore._point_id(REPO_ID) == repository_point_id(REPO_ID)


def test_repository_payload_contract_has_identity_and_embedding_fields():
    assert tuple(REPOSITORY_PAYLOAD_FIELD_TYPES) == REPOSITORY_PAYLOAD_REQUIRED_FIELDS
    assert {"repo_id", "full_name"}.issubset(REPOSITORY_PAYLOAD_REQUIRED_FIELDS)
    assert {
        "github_id",
        "content_version",
        "content_hash",
        "embedding_dim",
        "embedding_model",
        "embedding_version",
        "model_version",
        "indexed_at",
        "source_hash",
        REPOSITORY_SERVING_ELIGIBILITY_FIELD,
    }.issubset(REPOSITORY_PAYLOAD_REQUIRED_FIELDS)


def test_repository_payload_defaults_are_fresh_and_match_current_model():
    first = repository_payload_defaults()
    second = repository_payload_defaults()

    first["languages"].append("Python")  # type: ignore[union-attr]

    assert second["languages"] == []
    assert second["embedding_dim"] == 384
    assert second["embedding_model"] == REPOSITORY_EMBEDDING_MODEL
    assert (
        second[REPOSITORY_SERVING_ELIGIBILITY_FIELD]
        == REPOSITORY_SERVING_ELIGIBILITY_VERSION
    )


def test_current_payload_builder_publishes_the_frozen_contract():
    payload = build_vector_payload(
        {
            "full_name": "owner/repository",
            "description": "Example repository",
            "primary_language": "Python",
            "languages": ["Python"],
            "topics": ["machine-learning"],
            "extracted_paragraphs": ["A documented project."],
        },
        repo_id=REPO_ID,
        final_embedding=[0.0] * 384,
        readme_chunks=1,
        source_hash="source-hash",
        config=RepositoryEmbeddingConfig(),
    )

    # Summary provenance and admission observations are optional internal metadata; existing vectors keep
    # the frozen required serving contract and need no reindex.
    assert set(payload) == (set(REPOSITORY_PAYLOAD_REQUIRED_FIELDS) - {
        REPOSITORY_SERVING_ELIGIBILITY_FIELD
    }) | {"card_summary_reuse_key", "admission_observation"}
    assert payload["card_summary_reuse_key"] is None
    assert payload["admission_observation"]["quality_approval"] == "not_evaluated"
    for field_name, expected_type in REPOSITORY_PAYLOAD_FIELD_TYPES.items():
        if field_name == REPOSITORY_SERVING_ELIGIBILITY_FIELD:
            continue
        assert isinstance(payload[field_name], expected_type), field_name


def test_language_numeric_metadata_does_not_change_semantic_towers_or_source_hash():
    import hashlib
    from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
    class Encoder:
        def embed_texts(self, texts):
            result = []
            for text in texts:
                vector = [0.] * 384
                vector[hashlib.sha256(text.encode()).digest()[0]] = 1.
                result.append(vector)
            return result
    pipeline = RepositoryEmbeddingPipeline(embedder=Encoder())
    source = dict(repo_id=REPO_ID, full_name="owner/repo", languages=["Python"], primary_language="Python",
        readme="A search project.", topics=["search"], language_total_size=1000,
        language_breakdown_complete=False,
        language_breakdown=[dict(name="Python", color="#3572A5", size=300, percentage=30.)])
    first = pipeline.embed_repository(source)
    changed = {**source, "language_breakdown": [dict(name="Python", color="#FFFFFF", size=500, percentage=50.)]}
    second = pipeline.embed_repository(changed)
    assert first.payload["language_breakdown"] == source["language_breakdown"]
    assert second.payload["language_breakdown"] == changed["language_breakdown"]
    assert first.source_hash == second.source_hash
    for tower in ("readme_embedding", "metadata_embedding", "topic_embedding", "final_embedding"):
        assert getattr(first, tower) == getattr(second, tower)
    # Old provider model dumps include optional nulls; those are unknown, not new evidence.
    from contracts.repository_languages import language_observation
    assert language_observation(dict(language_breakdown=None, language_total_size=None,
                                    language_breakdown_complete=None)) == {}


def test_language_percentage_rejects_contradictory_observed_bytes():
    from contracts.repository_languages import language_observation
    with pytest.raises(ValueError, match="percentage.*byte"):
        language_observation(dict(language_total_size=1000,
            language_breakdown=[dict(name="Python", size=300, percentage=99.)]))


def test_language_percentage_accepts_rounded_rows_without_inventing_unknowns():
    from contracts.repository_languages import language_observation
    rows = [dict(name=f"language{number}", size=1, percentage=16.67) for number in range(6)]
    observed = language_observation(dict(language_total_size=6, language_breakdown_complete=True,
                                        language_breakdown=rows))
    assert observed["language_breakdown"] == rows
    assert observed["language_breakdown_complete"] is True
    for total, row in [(3, dict(name="Python", size=1, percentage=33.33)),
                       (3, dict(name="Python", size=1, percentage=None)),
                       (None, dict(name="Python", size=1, percentage=None)),
                       (0, dict(name="Python", size=0, percentage=None))]:
        result = language_observation(dict(language_total_size=total, language_breakdown=[row]))
        assert result["language_breakdown"] == [row]


def test_unknown_language_percentages_do_not_expand_rounding_allowance():
    from contracts.repository_languages import language_observation
    unknown = [dict(name=f"unknown{number}", percentage=None) for number in range(98)]
    with pytest.raises(ValueError, match="percentages exceed"):
        language_observation(dict(language_breakdown=unknown + [
            dict(name="Python", percentage=50.4), dict(name="Rust", percentage=50.4)]))


def _issue25_source():
    return dict(
        repo_id=REPO_ID,
        full_name="owner/v2-index",
        description="Parses protocol v2 with 384-dimensional embeddings.",
        primary_language="Python",
        languages=["Python", "Rust"],
        topics=["search", "indexing"],
        category="vector search",
        readme="README documents v2 and 384-dimensional vectors.",
        star_count=127,
        fork_count=18,
        open_issues_count=6,
        pushed_days_ago=3,
        delta_3d=2,
        delta_7d=5,
        delta_30d=13,
        mentionable_users_count=42,
        discovery_category="recent",
        discovery_band="high",
    )


class _Issue25RecordingEncoder:
    def __init__(self):
        import hashlib
        self._hashlib = hashlib
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            vector = [0.0] * 384
            vector[int.from_bytes(self._hashlib.sha256(text.encode()).digest()[:2], "big") % 384] = 1.0
            vectors.append(vector)
        return vectors


def _issue25_pipeline(version=None):
    from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
    from embedding.repository_embedding import RepositoryEmbeddingConfig
    config = RepositoryEmbeddingConfig() if version is None else RepositoryEmbeddingConfig(version=version)
    return RepositoryEmbeddingPipeline(
        config=config,
        embedder=_Issue25RecordingEncoder(),
    )


def _issue25_vector_hash(vector):
    import hashlib
    import json
    return hashlib.sha256(json.dumps(vector, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def test_default_v2_metadata_and_embedding_outputs_remain_exact():
    from embedding.repository_embedding import RepositoryEmbeddingConfig, build_metadata_text
    pipeline = _issue25_pipeline()
    source = _issue25_source()
    result = pipeline.embed_repository(source)

    assert RepositoryEmbeddingConfig().version == "repo-embedding-v2"
    assert build_metadata_text(source) == (
        "Repository: owner/v2-index\n"
        "Description: Parses protocol v2 with 384-dimensional embeddings.\n"
        "Primary language: Python\n"
        "Stars: 127\n"
        "Forks: 18\n"
        "Open issues: 6\n"
        "Pushed days ago: 3\n"
        "Star deltas: 3d=2, 7d=5, 30d=13\n"
        "Contributor signal: mentionable_users_count=42\n"
        "Discovery category: recent\n"
        "Discovery band: high"
    )
    assert result.embedding_version == "repo-embedding-v2"
    assert result.source_hash == "25d12c250331ee764c1b5e7485643e682aec97d4cb2067006569c2d3049607a2"
    assert _issue25_vector_hash(result.metadata_embedding) == "dab2fb0e3dfb27b391df9a98e7838d09549dd4fe5f116038131fe8bb3bdbe57f"
    assert _issue25_vector_hash(result.final_embedding) == "03313ab315bcf001167eb5edfa66cc6a613d5e91f03281db97551c777c7bbd6b"


def test_stable_metadata_candidate_ignores_volatile_fields_but_preserves_content_and_payload():
    candidate_version = "repo-embedding-v3-stable-metadata-v1"
    source = _issue25_source()
    changed_metrics = {
        **source,
        "star_count": 9001,
        "fork_count": 701,
        "open_issues_count": 88,
        "pushed_days_ago": 27,
        "delta_3d": 14,
        "delta_7d": 72,
        "delta_30d": 310,
        "mentionable_users_count": 123,
        "discovery_category": "popular",
        "discovery_band": "low",
    }
    pipeline = _issue25_pipeline(candidate_version)
    first = pipeline.embed_repository(source)
    second = pipeline.embed_repository(changed_metrics)
    encoded_texts = pipeline.embedder.calls[0]
    metadata_text = next(text for text in encoded_texts if text.startswith("Repository:"))

    assert first.embedding_version == candidate_version
    assert first.payload["embedding_version"] == candidate_version
    assert first.source_hash == second.source_hash
    assert first.metadata_embedding == second.metadata_embedding
    assert first.final_embedding == second.final_embedding
    assert any("README documents v2 and 384-dimensional vectors." in text for text in encoded_texts)
    assert "Description: Parses protocol v2 with 384-dimensional embeddings." in metadata_text
    assert "Primary language: Python" in metadata_text
    assert "search" in metadata_text and "Rust" in metadata_text and "vector search" in metadata_text
    assert not any(label in metadata_text for label in (
        "Stars:", "Forks:", "Open issues:", "Pushed days ago:", "Star deltas:",
        "Contributor signal:", "Discovery category:", "Discovery band:",
    ))
    for field in (
        "star_count", "fork_count", "open_issues_count", "pushed_days_ago", "delta_3d",
        "delta_7d", "delta_30d", "mentionable_users_count", "discovery_category", "discovery_band",
    ):
        assert first.payload[field] == source[field]
        assert second.payload[field] == changed_metrics[field]

    changed_description = {**source, "description": "Parses protocol v2 with 768-dimensional embeddings."}
    third = pipeline.embed_repository(changed_description)
    assert third.source_hash != first.source_hash
    assert third.metadata_embedding != first.metadata_embedding
    assert third.final_embedding != first.final_embedding
    third_texts = pipeline.embedder.calls[-1]
    assert "768-dimensional" in next(text for text in third_texts if text.startswith("Repository:"))
    assert any("README documents v2 and 384-dimensional vectors." in text for text in third_texts)
    changed_readme = {**source, "readme": "README documents v3 and 384-dimensional vectors."}
    fourth = pipeline.embed_repository(changed_readme)
    assert fourth.source_hash != first.source_hash
    assert fourth.readme_embedding != first.readme_embedding
    assert fourth.final_embedding != first.final_embedding
    assert first.source_hash != _issue25_pipeline().embed_repository(source).source_hash


def test_stable_metadata_candidate_iter_index_rejects_before_qdrant_bootstrap(monkeypatch):
    from embedding import embedding_pipeline as pipeline_module
    pipeline = _issue25_pipeline("repo-embedding-v3-stable-metadata-v1")
    store_calls = []

    class LocalStore:
        def __init__(self, **kwargs):
            store_calls.append("construct")

        def ensure_collection(self):
            store_calls.append("ensure")

        def iter_upsert(self, results, **kwargs):
            store_calls.append("write")
            yield from results

    monkeypatch.setattr(pipeline_module, "QdrantRepositoryStore", LocalStore)
    with pytest.raises(ValueError):
        list(pipeline.iter_index([_issue25_source()]))
    assert store_calls == []
    assert pipeline.store is None

    with pytest.raises(ValueError):
        pipeline.search("a query for the candidate-only recipe")
    assert store_calls == []
    assert pipeline.store is None


def test_stable_metadata_candidate_is_rejected_by_qdrant_point_validation():
    from embedding.qdrant_store import QdrantRepositoryStore
    candidate = _issue25_pipeline("repo-embedding-v3-stable-metadata-v1").embed_repository(_issue25_source())
    store = QdrantRepositoryStore(client=object(), collection_name="isolated-test", corpus_authority=None)

    with pytest.raises(ValueError):
        store._validated_point(candidate, point_id=REPO_ID)


def _issue25_write_snapshot(path, repositories):
    import json
    path.write_text(json.dumps({"repositories": repositories}), encoding="utf-8")
    return path


def test_stable_metadata_export_preserves_frozen_payload_and_computed_readme_chunks(
    tmp_path,
):
    import json
    from pathlib import Path

    from config import EMBEDDING_MODEL_REVISION, STABLE_METADATA_EMBEDDING_VERSION
    from scripts.export_stable_metadata_candidate import export_snapshot

    fixture_path = Path(__file__).parent / "fixtures/recommendation_public_catalog/snapshot.json"
    source_row = json.loads(fixture_path.read_text(encoding="utf-8"))["repositories"][0]
    frozen_payload = dict(source_row["payload"])
    assert not frozen_payload.get("readme")
    input_path = _issue25_write_snapshot(tmp_path / "snapshot.json", [source_row])
    output_path = tmp_path / "candidate.jsonl"

    assert export_snapshot(
        input_path, output_path,
        pipeline=_issue25_pipeline(STABLE_METADATA_EMBEDDING_VERSION),
    ) == 1
    record = json.loads(output_path.read_text(encoding="utf-8"))
    exported_payload = record["payload"]
    expected_payload = dict(frozen_payload)
    expected_payload.pop(REPOSITORY_SERVING_ELIGIBILITY_FIELD)
    expected_payload["readme_chunks"] = 0
    expected_payload.update({
        "embedding_dim": REPOSITORY_EMBEDDING_DIM,
        "embedding_model": REPOSITORY_EMBEDDING_MODEL,
        "embedding_model_revision": EMBEDDING_MODEL_REVISION,
        "embedding_version": STABLE_METADATA_EMBEDDING_VERSION,
        "source_hash": record["source_hash"],
    })

    assert exported_payload == expected_payload
    assert exported_payload["doc_quality"] == frozen_payload["doc_quality"]
    assert exported_payload["category"] == frozen_payload["category"]
    assert exported_payload["readme_chunks"] == 0
    assert len(record["vector"]) == REPOSITORY_EMBEDDING_DIM
    assert record["recipe_id"] == STABLE_METADATA_EMBEDDING_VERSION


def test_stable_metadata_export_rejects_malformed_snapshot_without_output(tmp_path):
    from scripts.export_stable_metadata_candidate import export_snapshot

    input_path = tmp_path / "malformed.json"
    input_path.write_text('{"repositories": null}', encoding="utf-8")
    output_path = tmp_path / "candidate.jsonl"

    with pytest.raises(ValueError, match="frozen JSON snapshot"):
        export_snapshot(input_path, output_path, pipeline=_issue25_pipeline(
            "repo-embedding-v3-stable-metadata-v1",
        ))

    assert not output_path.exists()


def test_stable_metadata_export_does_not_overwrite_a_racing_destination(tmp_path, monkeypatch):
    from scripts import export_stable_metadata_candidate as exporter

    input_path = _issue25_write_snapshot(
        tmp_path / "snapshot.json",
        [{"repo_id": REPO_ID, "payload": _issue25_source()}],
    )
    output_path = tmp_path / "candidate.jsonl"
    write_exclusive = exporter._write_jsonl_exclusive

    def publish_racing_destination(path, records):
        path.write_text("created concurrently", encoding="utf-8")
        write_exclusive(path, records)

    monkeypatch.setattr(exporter, "_write_jsonl_exclusive", publish_racing_destination)
    with pytest.raises(FileExistsError):
        exporter.export_snapshot(
            input_path, output_path,
            pipeline=_issue25_pipeline("repo-embedding-v3-stable-metadata-v1"),
        )

    assert output_path.read_text(encoding="utf-8") == "created concurrently"


def test_stable_metadata_export_uses_configured_hf_cache_and_local_only(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from config import EMBEDDING_MODEL_REVISION
    from scripts import export_stable_metadata_candidate as exporter

    cache_root = tmp_path / "hf-cache"
    monkeypatch.setattr(exporter, "HF_HUB_CACHE", str(cache_root))
    model_snapshot = exporter._default_model_snapshot()
    model_snapshot.mkdir(parents=True)
    for name in ("modules.json", "config.json", "model.safetensors", "tokenizer.json"):
        (model_snapshot / name).touch()

    model = MagicMock()
    model.get_embedding_dimension.return_value = REPOSITORY_EMBEDDING_DIM
    constructor = MagicMock(return_value=model)
    pipeline_factory = MagicMock(return_value=object())
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", constructor)
    monkeypatch.setattr(exporter, "RepositoryEmbeddingPipeline", pipeline_factory)

    exporter._build_local_pipeline(model_snapshot)

    assert model_snapshot == (
        cache_root
        / "models--sentence-transformers--all-MiniLM-L6-v2"
        / "snapshots"
        / EMBEDDING_MODEL_REVISION
    )
    constructor.assert_called_once_with(
        str(model_snapshot), device="cpu", local_files_only=True,
    )
