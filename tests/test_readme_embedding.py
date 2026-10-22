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
    SECTION_CHUNK_LIMITS, _sections, prepare_readme_chunks, readme_source, token_chunks,
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


def test_no_source_ranges_are_lost_at_subword_boundaries(tokenizer):
    # Distinct markers bind chunks to their actual source positions, including
    # subword splits through a marker, rather than matching any repeated copy.
    text = " ".join(f"marker{index:04d}abcdefghijklmnopqrstuvwxyz" for index in range(40))
    result = token_chunks(text, tokenizer, sequence_limit=16, overlap_tokens=3)
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]

    def assert_complete(chunks):
        covered = set()
        previous_start = -1
        for chunk, _ in chunks:
            start = text.find(chunk, previous_start + 1)
            assert start >= 0
            covered.update(range(start, start + len(chunk)))
            previous_start = start
        expected = {position for start, end in offsets for position in range(start, end)}
        assert expected <= covered

    assert_complete(result)
    middle = len(result) // 2
    # Prove this assertion detects both middle and trailing chunk loss.
    with pytest.raises(AssertionError):
        assert_complete(result[:middle] + result[middle + 1:])
    with pytest.raises(AssertionError):
        assert_complete(result[:-1])


def test_heading_whitespace_does_not_stall_workers():
    import subprocess
    import sys
    from pathlib import Path

    # A subprocess bounds the regression itself if quadratic/cubic matching is
    # accidentally restored. The original 1,600-space case took seconds.
    script = """
from embedding.readme_embedding import _sections
from utils.readme_processor import MAX_PREPROCESS_CHARS
for size in (1600, MAX_PREPROCESS_CHARS - 32):
    for heading in ('# Overview' + ' ' * size + '!', '#' + ' ' * size,
                    '# Overview' + ' ' * size + '#'):
        sections = _sections(heading + '\\nUseful details.')
        assert any('Useful details.' in text for _, text in sections)
for text in ('[' + 'x' * 62000,
             '[ref]: /url\\n\"' + 'title\\n' * 9000,
             ('[ref]: /url\\n' * 4000),
             ('[bad]\\n===\\n' * 5000),
             ('[bad]: /url \"unclosed\\n===\\n' * 2500),
             '*' + ' \\t*' * 20000):
    _sections(text)
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=5,
                   cwd=Path(__file__).resolve().parents[1])


@pytest.mark.parametrize("inline", ["```pip install package```", "````inline `code` span````"])
def test_inline_backtick_spans_do_not_swallow_later_sections(inline):
    sections = _sections(inline + "\n\n## Features\nSemantic retrieval supports metadata filters.")
    assert any(category == "purpose" and "Semantic retrieval" in text for category, text in sections)


@pytest.mark.parametrize("fence", ["```python", "~~~~info `backticks` allowed"])
def test_real_fences_still_hide_code_and_preserve_following_sections(fence):
    close = fence[:3] if fence.startswith("`") else "~~~~~"
    sections = _sections(fence + "\n# Features\nsecret code\n" + close + "\n## Usage\nRun semantic queries.")
    assert any("Run semantic queries." in text for _, text in sections)
    assert all("secret code" not in text for _, text in sections)


@pytest.mark.parametrize("title", ["License ⚖️", "📜 License 📜", "Sponsors ❤️", "Generated API reference 📚"])
def test_decorated_legal_and_reference_subtrees_are_omitted(title):
    sections = _sections("# Features\nUseful details.\n## " + title + "\nUnrelated boilerplate.\n### Nested\nMore boilerplate.\n## Usage\nSemantic queries.")
    text = " ".join(text for _, text in sections)
    assert "Useful details." in text and "Semantic queries." in text
    assert "boilerplate" not in text


def test_descriptive_heading_only_readme_retains_features_and_literal_symbols(tokenizer):
    markdown = "# Features\n## Semantic vector search\n## Metadata filters\n## C# and .NET\n## License ⚖️\n### Excluded details"
    sections = _sections(markdown)
    assert all(category == "purpose" for category, _ in sections)
    text = " ".join(text for _, text in sections)
    assert "Semantic vector search" in text and "Metadata filters" in text
    assert "C# and .NET" in text and "Excluded" not in text
    result = chunks(markdown, tokenizer)
    assert result and all(chunk.section_class == "purpose" for chunk in result)


@pytest.mark.parametrize("title", [
    "A guide to checking\nLicense", "License\ncompatibility checker",
    "Features: license detection", "License compatibility checker ⚖️",
])
def test_full_setext_title_preserves_project_subject_matter(title):
    sections = _sections(title + "\n-------\nDetect incompatible dependency policies.")
    text = " ".join(text for _, text in sections)
    assert "Detect incompatible" in text
    assert " ".join(title.split()) in text


def test_multiline_legal_setext_title_is_omitted():
    assert _sections("License\nInformation\n-------\nPermission is hereby granted.") == []


@pytest.mark.parametrize("boundary", ["\n\n", "\n# New section\n", "\n- list item\n", "\n> quote\n", "\n```python\ncode\n```\n"])
def test_setext_title_stops_at_block_boundaries(boundary):
    sections = _sections("Useful introduction." + boundary + "License\n-------\nExcluded boilerplate.")
    text = " ".join(text for _, text in sections)
    assert "Useful introduction." in text
    assert "Excluded boilerplate." not in text


@pytest.mark.parametrize("definition", [
    "[home]: https://example.com",
    "   [home]: <https://example.com/a b> 'Project homepage'",
    r'[ho\]me]: /some_(nested_(url)) "An escaped \"title\""',
    r'[ho\[me]: /escaped\(destination\) (Title with escaped \(parentheses\))',
    "[home]: <>",
    "[home]:\n    /url\n    'Title'",
    "[multi\nline label]: /url 'A title\nwith more text'",
    '[home]: /url "Title\n===\nmore text"',
    '[home]: /url\n"Title\n===\nmore text"',
    "[multi\n===\nline label]: /url",
    "[home]:\n===",
    "[first]: /url\n[second]: /other-url",
])
def test_leading_reference_definitions_do_not_enter_setext_titles(definition):
    sections = _sections("# Overview\n" + definition + "\nLicense\n-------\nLegal boilerplate.")
    assert sections == [("purpose", "Overview")]


@pytest.mark.parametrize("marker", ["***", "___", "* * *", "  _\t_ _  ", "   - - -"])
def test_thematic_breaks_separate_setext_titles_from_prior_prose(marker):
    sections = _sections("# Overview\nUseful intro.\n" + marker + "\nLicense\n-------\nLegal boilerplate.")
    text = " ".join(text for _, text in sections)
    assert "Useful intro." in text and "Legal boilerplate." not in text


def test_dashed_setext_keeps_precedence_over_thematic_break():
    assert _sections("License\n---\nLegal boilerplate.") == []
    sections = _sections("A guide to checking\nLicense\n---\nDependency analysis.")
    assert sections == [("neutral", "A guide to checking License Dependency analysis.")]


def test_reference_definitions_cannot_interrupt_an_existing_paragraph():
    markdown = "# Overview\nUseful intro.\n[home]: /url\nLicense\n-------\nDependency analysis."
    text = " ".join(text for _, text in _sections(markdown))
    assert "[home]: /url License" in text and "Dependency analysis." in text


@pytest.mark.parametrize("prose", [
    "[home] missing colon: /url",
    r"[ho\]me: /url",
    "[nested[label]]: /url",
    "[]: /url", "[ ]: /url", "[" + "x" * 1000 + "]: /url",
    "[ ]: <>", "[ ]: <url with space>",
    "[home]:", "[home]: <unterminated", "[home]: /unbalanced(url",
    '[home]: /url "title" extra prose',
    "[home]: /url (nested(title))",
    "[home]: <url>(title without separator)",
])
def test_malformed_reference_like_prose_is_retained(prose):
    # A malformed definition is part of the multiline title; treating it as a
    # block boundary would incorrectly classify the final line as legal text.
    text = " ".join(text for _, text in _sections(prose + "\nLicense\n-------\nDependency analysis."))
    assert "Dependency analysis." in text


def test_next_line_invalid_reference_title_is_retained_as_prose():
    markdown = '[home]: /url\n"title" extra prose\nLicense\n-------\nDependency analysis.'
    text = " ".join(text for _, text in _sections(markdown))
    assert '"title" extra prose License' in text
    assert "[home]:" not in text and "Dependency analysis." in text


def test_indented_reference_like_code_is_not_removed():
    text = " ".join(text for _, text in _sections("    [home]: /url\n\nUseful prose."))
    assert "[home]: /url" in text and "Useful prose." in text


def test_definitions_without_real_heading_text_do_not_create_an_empty_setext_heading():
    sections = _sections("[ref]: /url\n===\nUseful prose.")
    assert sections == [("neutral", "=== Useful prose.")]


@pytest.mark.parametrize("marker", ["    ***", "\t___", "**", "__", "-_-", "+++"])
def test_non_thematic_markers_remain_in_multiline_titles(marker):
    text = " ".join(text for _, text in _sections("Useful intro.\n" + marker + "\nLicense\n---\nDependency analysis."))
    assert "Useful intro." in text and "Dependency analysis." in text


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
