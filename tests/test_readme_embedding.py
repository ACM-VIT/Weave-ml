"""README representation invariants without network/model downloads."""
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from config import TOKEN_SECTION_EMBEDDING_VERSION
from embedding.embeddings import SentenceTransformerEmbedder, aggregate_vectors
from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
from embedding.readme_embedding import (
    SECTION_CHUNK_LIMITS, prepare_readme_chunks, readme_source, token_chunks,
)
from embedding.repository_embedding import RepositoryEmbeddingConfig
from embedding.runtime import BoundedEmbeddingModel
from summarization.pipeline import description_fallback_artifact


def wordpiece_tokenizer():
    # Tiny local WordPiece vocabulary exercises subword expansion, source
    # offsets, special tokens and re-tokenization without Hugging Face access.
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789_-/.,"
    vocabulary = ["[UNK]", "[CLS]", "[SEP]", "[PAD]"] + list(alphabet) + ["##" + c for c in alphabet]
    backend = Tokenizer(models.WordPiece({word: index for index, word in enumerate(vocabulary)}, unk_token="[UNK]"))
    backend.normalizer = normalizers.BertNormalizer(lowercase=True)
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]")


@pytest.fixture
def tokenizer():
    return wordpiece_tokenizer()


def chunks(markdown, tokenizer, limit=64):
    return prepare_readme_chunks(markdown, tokenizer, sequence_limit=limit, overlap_tokens=8)


@pytest.mark.parametrize("text", [
    "distributed_transactional_vector_search " * 200,
    "Heading-free technical documentation explains durable transactions. " * 100,
    "中文检索与数据库 🚀 café naïve " * 100,
    "a" * 90 + " " + "z" * 90,
])
def test_every_chunk_fits_active_tokenizer_including_special_tokens(text, tokenizer):
    result = token_chunks(text, tokenizer, sequence_limit=64, overlap_tokens=8)
    assert len(result) > 1
    assert all(count <= 64 for _, count in result)
    assert all(len(tokenizer(chunk, add_special_tokens=True, truncation=False)["input_ids"]) == count for chunk, count in result)
    assert result[-1][0].endswith(text.rstrip()[-10:])


def test_no_tokens_are_lost_at_subword_boundaries(tokenizer):
    text = "abcdefghijklmnopqrstuvwxyz0123456789 " * 7
    result = token_chunks(text, tokenizer, sequence_limit=16, overlap_tokens=3)
    # Every unique source marker occurs in at least one chunk after overlap.
    for marker in ("abcdefgh", "ijklmnop", "qrstuvwx", "yz012345", "6789"):
        assert any(marker in chunk for chunk, _ in result)


def test_heading_free_and_paragraph_only_inputs_have_neutral_fallback(tokenizer):
    payload = {"extracted_paragraphs": ["Durable search indexes support efficient retrieval.", "Queries use a persistent vector index."]}
    result = chunks(readme_source(payload, payload), tokenizer)
    assert result and {chunk.section_class for chunk in result} == {"neutral"}
    assert sum(chunk.weight for chunk in result) == pytest.approx(1.0)


@pytest.mark.parametrize("markdown", ["", "  \n", "# License\n\nPermission is hereby granted to contributors.", "# Contributors\n## Usage\nThis is still an excluded subtree.", "```python\n# Features\nlicense = True\n```"])
def test_empty_and_only_boilerplate_has_no_readme_vector(markdown, tokenizer):
    assert chunks(markdown, tokenizer) == []


def test_nested_setext_and_fenced_headings_preserve_section_policy(tokenizer):
    markdown = """Overview
========
Durable vector storage.
### Details
Persistent search indexes.
## License
Excluded license text.
### Features
Still excluded.
## Architecture
Distributed transactions.
~~~python
## Sponsors
this is code
~~~
Reliable replication.
"""
    result = chunks(markdown, tokenizer)
    text = " ".join(chunk.text for chunk in result)
    assert "Durable" in text and "Persistent" in text and "Reliable" in text
    assert "Excluded" not in text and "Still excluded" not in text and "this is code" not in text
    assert all(chunk.section_class == "purpose" for chunk in result)


def test_boilerplate_volume_and_repeated_headings_do_not_increase_weight(tokenizer):
    markdown = "# Overview\nDurable transactions.\n" + "\n".join(
        f"## Installation {index}\nPackage setup commands require dependencies. " * 4 for index in range(150)
    ) + "\n# Sponsors\n" + "Sponsored by unrelated vendors. " * 200
    result = chunks(markdown, tokenizer)
    totals = defaultdict(float)
    for chunk in result:
        totals[chunk.section_class] += chunk.weight
    assert totals["purpose"] == pytest.approx(3.0)
    assert totals["setup"] == pytest.approx(0.1)
    assert len(result) <= sum(SECTION_CHUNK_LIMITS.values())
    assert all("Sponsored" not in chunk.text for chunk in result)
    # Directional test uses distinct content vectors, independent of cosine
    # judgments from a learned model.
    vector = aggregate_vectors([[1., 0.] if chunk.section_class == "purpose" else [0., 1.] for chunk in result], weights=[chunk.weight for chunk in result])
    assert vector[0] > 25 * vector[1]


@pytest.mark.parametrize("overlap", [-1, True, 62, 64])
def test_invalid_overlap_fails_before_any_encoding(tokenizer, overlap):
    with pytest.raises(ValueError, match="overlap"):
        token_chunks("example text", tokenizer, sequence_limit=64, overlap_tokens=overlap)


def test_original_markdown_wins_over_cleaned_document_text():
    source = SimpleNamespace(readme=SimpleNamespace(raw_markdown="# Features\nOriginal source", clean_text="Flattened source"))
    assert readme_source(source, {}) == "# Features\nOriginal source"
    assert readme_source({}, {"readme": "# Usage\nOriginal source", "clean_text": "Flattened"}).startswith("# Usage")


class Encoder:
    max_seq_length = 64

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            vector = [1., .2] if "Purpose" in text else [.2, 1.]
            vectors.append(vector + [0.] * 382)
        return np.asarray(vectors)


class Summarizer:
    def summarize(self, source, repo, **kwargs):
        return description_fallback_artifact(repo)


def repository(markdown):
    return {"repo_id": "00000000-0000-4000-8000-000000000001", "full_name": "test/readme", "readme": markdown, "description": "A documented tool.", "content_version": 1}


def test_heading_only_changes_invalidate_readme_cache_and_source_hash(tokenizer):
    encoder = Encoder(tokenizer)
    wrapper = BoundedEmbeddingModel(encoder, max_concurrency=1)
    assert wrapper.tokenizer.get_vocab() == tokenizer.get_vocab()
    cache = wrapper.tower_embedding_cache
    pipeline = RepositoryEmbeddingPipeline(config=RepositoryEmbeddingConfig(version=TOKEN_SECTION_EMBEDDING_VERSION, readme_chunk_overlap_tokens=8), embedder=SentenceTransformerEmbedder(model=wrapper), tower_cache=cache, card_summarizer=Summarizer())
    markdown = "# Overview\nPurpose search.\n# Details\nOther material."
    first = pipeline.embed_repository(repository(markdown))
    second = pipeline.embed_repository(repository(markdown.replace("# Overview", "# Installation")))
    assert first.source_hash != second.source_hash
    assert first.readme_embedding != second.readme_embedding
    assert len(encoder.calls[-1]) == second.readme_chunks
    # Exact repeat uses all towers; cached and uncached computations agree.
    repeat = pipeline.embed_repository(repository(markdown.replace("# Overview", "# Installation")))
    assert repeat.final_embedding == second.final_embedding
    pipeline.tower_cache = None
    assert pipeline.embed_repository(repository(markdown.replace("# Overview", "# Installation"))).final_embedding == second.final_embedding


def test_readme_workers_do_not_share_mutable_encoder_tokenizer_state(tokenizer):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    wrapper = BoundedEmbeddingModel(Encoder(tokenizer), max_concurrency=1)
    barrier = Barrier(2)
    text = "distributed search indexes " * 100

    def worker(limit):
        local = wrapper.tokenizer
        assert local is wrapper.tokenizer and local is not tokenizer
        barrier.wait(timeout=2)
        encoded = local(text, truncation=limit is not None, max_length=limit)
        return local, len(encoded["input_ids"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        short = pool.submit(worker, 16)
        complete = pool.submit(worker, None)
        short_tokenizer, short_count = short.result(timeout=5)
        complete_tokenizer, complete_count = complete.result(timeout=5)
    assert short_tokenizer is not complete_tokenizer
    assert short_count == 16 and complete_count > 16
    assert tokenizer.backend_tokenizer.truncation is None


def test_all_excluded_readme_renormalizes_metadata_and_topics(tokenizer):
    pipeline = RepositoryEmbeddingPipeline(config=RepositoryEmbeddingConfig(version=TOKEN_SECTION_EMBEDDING_VERSION, readme_chunk_overlap_tokens=8), embedder=SentenceTransformerEmbedder(model=Encoder(tokenizer)), card_summarizer=Summarizer())
    result = pipeline.embed_repository(repository("# License\nLicense content only."))
    assert result.readme_chunks == 0 and result.readme_embedding == []
    assert result.payload["embedding_version"] == TOKEN_SECTION_EMBEDDING_VERSION
    assert np.linalg.norm(result.final_embedding) == pytest.approx(1.0)


def test_reserved_v2_keeps_old_character_mean_path():
    # A tokenizer-free injected encoder continues to work for legacy jobs.
    encoder = SimpleNamespace(max_seq_length=256, encode=lambda texts, **kwargs: np.asarray([[1.] + [0.] * 383 for _ in texts]))
    result = RepositoryEmbeddingPipeline(config=RepositoryEmbeddingConfig(version="repo-embedding-v2"), embedder=SentenceTransformerEmbedder(model=encoder), card_summarizer=Summarizer()).embed_repository(repository("# License\nLicense content stays part of the legacy representation."))
    assert result.readme_chunks == 1
    assert result.embedding_version == "repo-embedding-v2"


def test_version_opt_in_does_not_reinterpret_existing_custom_legacy_labels():
    encoder = SimpleNamespace(max_seq_length=256, encode=lambda texts, **kwargs: np.asarray([[1.] + [0.] * 383 for _ in texts]))
    result = RepositoryEmbeddingPipeline(config=RepositoryEmbeddingConfig(version="repo-embedding-v2-custom"), embedder=SentenceTransformerEmbedder(model=encoder), card_summarizer=Summarizer()).embed_repository(repository("# License\nLegacy custom content."))
    assert result.readme_chunks == 1 and result.embedding_version == "repo-embedding-v2-custom"


def test_license_subject_matter_is_not_a_legal_boilerplate_section(tokenizer):
    result = chunks("# License compatibility checker\n## Overview\nDetect incompatible dependency policies.", tokenizer)
    assert result and any("incompatible" in chunk.text for chunk in result)


def test_new_revision_migrates_vector_while_same_revision_replay_retains_old_vector(monkeypatch, tokenizer):
    import uuid
    from fastapi import HTTPException
    from qdrant_client import QdrantClient
    from api import v2
    from api.contracts import RepositoryJob
    from embedding.qdrant_store import QdrantRepositoryStore
    from summarization.pipeline import CardSummaryPipeline

    client = QdrantClient(":memory:")
    store = QdrantRepositoryStore(client=client)
    store.ensure_collection()
    source = repository("# Overview\nPurpose search.\n# License\nLicense boilerplate.")
    source.pop("content_version")
    source["content_hash"] = "canonical-same-source-hash"
    request = RepositoryJob.model_validate({"schema_version": 2, "job_id": str(uuid.uuid4()), "repo_id": source["repo_id"], "content_version": 7, "repository": source})
    encoder = Encoder(tokenizer)
    pipeline = RepositoryEmbeddingPipeline(config=RepositoryEmbeddingConfig(version="repo-embedding-v2"), embedder=SentenceTransformerEmbedder(model=encoder), card_summarizer=CardSummaryPipeline())
    monkeypatch.setattr(v2, "repository_store", lambda: store)
    monkeypatch.setattr(v2, "_repository_points", lambda repo_id: client.retrieve(store.collection_name, ids=[repo_id], with_payload=True, with_vectors=True))
    monkeypatch.setattr(v2, "repository_embedding_pipeline", lambda: pipeline)
    lock = SimpleNamespace(assert_owned=lambda: None)
    assert v2._embed_repository_job_locked(request, lock)["status"] == "applied"
    calls = len(encoder.calls)
    pipeline.config.version = TOKEN_SECTION_EMBEDDING_VERSION
    pipeline.config.readme_chunk_overlap_tokens = 8
    replay = v2._embed_repository_job_locked(request, lock)
    assert replay["status"] == "duplicate" and replay["embedding_version"] == "repo-embedding-v2"
    assert len(encoder.calls) == calls
    newer = request.model_copy(update={"job_id": uuid.uuid4(), "content_version": 8})
    migrated = v2._embed_repository_job_locked(newer, lock)
    assert migrated["status"] == "applied" and migrated["embedding_version"] == TOKEN_SECTION_EMBEDDING_VERSION
    point = client.retrieve(store.collection_name, ids=[source["repo_id"]], with_payload=True, with_vectors=True)[0]
    assert point.payload["content_version"] == 8
    assert point.payload["content_hash"] == source["content_hash"]
    with pytest.raises(HTTPException):
        v2._embed_repository_job_locked(request, lock)
    assert client.retrieve(store.collection_name, ids=[source["repo_id"]])[0].payload["embedding_version"] == TOKEN_SECTION_EMBEDDING_VERSION
    client.close()


def test_shipped_ranker_does_not_claim_new_representation_compatibility():
    from pathlib import Path
    from inference.ranker_service import RankerService
    root = Path(__file__).resolve().parents[1] / "inference"
    with pytest.raises(ValueError, match="embedding contract is incompatible"):
        RankerService(model_path=str(root / "heavy_ranker.pt"), scaler_path=str(root / "feature_scaler.json"), manifest_path=str(root / "model_manifest.json"), expected_embedding_version=TOKEN_SECTION_EMBEDDING_VERSION, require_production_manifest=True)


def test_frozen_evaluation_report_binds_inputs_and_metrics():
    import hashlib
    import json
    from pathlib import Path
    from scripts.evaluate_readme_aggregation import FIXTURE, retrieval_metrics

    report = json.loads((Path(__file__).resolve().parents[1] / "docs/readme-aggregation-evaluation.json").read_text())
    fixture = json.loads(FIXTURE.read_bytes())
    assert report["fixture_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert len(fixture["repositories"]) == report["repository_count"] == 12
    repo_ids = {repo["repo_id"] for repo in fixture["repositories"]}
    assert all(set(query["relevant_repo_ids"]) <= repo_ids for query in fixture["queries"])
    assert report["strategies"]["token_sections"]["cost"]["readme_chunks_over_limit"] == 0
    metrics = retrieval_metrics([[.9, 1., .1]], [{"query": "sample", "relevant_repo_ids": ["relevant"]}], ["relevant", "other", "third"], k=1)
    assert metrics["mrr"] == .5 and metrics["recall_at_1"] == 0
    assert metrics["ndcg_at_1"] == 0
