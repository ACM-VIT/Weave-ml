"""Offline experimental planning checks; production cap policy remains pending."""
import hashlib

import numpy as np
import pytest

from config import TOKEN_SECTION_EMBEDDING_VERSION
from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
from embedding.embeddings import SentenceTransformerEmbedder
from embedding.repository_embedding import RepositoryEmbeddingConfig
from scripts.sweep_embedding_budget import Budget, encode_job, no_network, select_prefix, verify_cohorts
from summarization.pipeline import description_fallback_artifact
from tests.test_readme_embedding import wordpiece_tokenizer


class Encoder:
    max_seq_length = 64

    def __init__(self):
        self.tokenizer = wordpiece_tokenizer()
        self.texts = []
        self.calls = []

    def encode(self, texts, **kwargs):
        self.texts.extend(texts)
        self.calls.append(list(texts))
        result = []
        for text in texts:
            seed = hashlib.sha256(text.encode()).digest()
            vector = np.asarray([seed[i % len(seed)] + 1 for i in range(384)], dtype=np.float32)
            result.append(vector / np.linalg.norm(vector))
        return np.asarray(result)


class Summary:
    def summarize(self, source, repo, **kwargs):
        return description_fallback_artifact(repo)


def case():
    return {"name": "diagnostic", "readme": "# Overview\n" + "vector search storage " * 120 + "\n# Installation\n" + "install setup " * 80}


@pytest.mark.parametrize("version", ["repo-embedding-v2", TOKEN_SECTION_EMBEDDING_VERSION])
def test_uncapped_control_matches_existing_full_pipeline(version):
    fixture = case()
    measured = encode_job(fixture, version=version, model=Encoder(), budget=None)
    repo = {"repo_id": "00000000-0000-4000-8000-000000000058", "full_name": "diagnostic/repository", "description": "Offline budget diagnostic", "readme": fixture["readme"], "topics": ["search"], "languages": ["Python"], "content_version": 1}
    pipeline = RepositoryEmbeddingPipeline(config=RepositoryEmbeddingConfig(version=version), embedder=SentenceTransformerEmbedder(model=Encoder()), card_summarizer=Summary())
    actual = pipeline.embed_repository(repo)
    assert np.allclose(measured["final_vector"], actual.final_embedding, atol=1e-7)
    assert measured["retained_readme_chunks"] == actual.readme_chunks


def test_total_budget_reserves_mandatory_towers_and_stops_at_prefix_boundary():
    assert select_prefix([64, 64, 10], [20, 10], Budget(4, 158)) == 2
    # It cannot skip a large earlier chunk to choose a smaller later one.
    assert select_prefix([64, 100, 10], [20, 10], Budget(5, 110)) == 1
    with pytest.raises(ValueError, match="one README"):
        select_prefix([64], [20, 10], Budget(3, 93))


@pytest.mark.parametrize("version", ["repo-embedding-v2", TOKEN_SECTION_EMBEDDING_VERSION])
def test_budgeted_job_uses_actual_grouped_path_and_retains_deterministic_inputs(version):
    a, b = Encoder(), Encoder()
    budget = Budget(3, 192)
    first = encode_job(case(), version=version, model=a, budget=budget)
    second = encode_job(case(), version=version, model=b, budget=budget)
    assert a.texts == b.texts
    assert first["job_chunks"] <= budget.chunks
    assert first["job_encoded_tokens"] <= budget.tokens
    assert len(a.texts) == first["job_chunks"]
    assert np.allclose(first["final_vector"], second["final_vector"])
    assert 0 < first["retained_encoded_token_fraction"] < 1
    assert sum(value["retained_chunks"] for value in first["section_class_coverage"].values()) == first["retained_readme_chunks"]


def test_large_budget_preserves_control_exactly():
    control = encode_job(case(), version=TOKEN_SECTION_EMBEDDING_VERSION, model=Encoder(), budget=None)
    generous = encode_job(case(), version=TOKEN_SECTION_EMBEDDING_VERSION, model=Encoder(), budget=Budget(128, 65536))
    assert generous["final_vector"] == control["final_vector"]
    assert generous["budget_dropped_encoded_token_fraction"] == 0


def test_network_guard_fails_closed_and_restores_socket():
    import socket
    original = socket.create_connection
    with no_network() as attempts:
        with pytest.raises(RuntimeError, match="network forbidden"):
            socket.create_connection(("example.invalid", 443))
    assert len(attempts) == 1
    assert socket.create_connection is original


@pytest.mark.parametrize("field", ["source_hashes", "fixtures", "model_revision"])
def test_changing_inputs_cannot_produce_a_budget_comparison(field):
    from copy import deepcopy
    group = {"source_hashes": {"encoder": "original"}, "fixtures": [{"name": "readme", "sha256": "a"}], "model_revision": "pinned", "max_seq_length": 256}
    changed = deepcopy(group)
    if field == "fixtures":
        changed[field][0]["sha256"] = "different"
    else:
        changed[field] = "different"
    with pytest.raises(RuntimeError, match="identity changed"):
        verify_cohorts([group, changed])
    verify_cohorts([group, deepcopy(group)])


@pytest.mark.parametrize("version", ["repo-embedding-v2", TOKEN_SECTION_EMBEDDING_VERSION])
def test_actual_encoder_groups_cross_eight_input_boundary_and_count_padding(version):
    model = Encoder()
    long_case = {"name": "group-boundary", "readme": "# Overview\n" + "vector search storage " * 1600}
    result = encode_job(long_case, version=version, model=model, budget=Budget(12, 65536))
    assert len(model.texts) == 12
    assert [len(group) for group in model.calls] == [8, 4]
    assert result["encoder_groups"] == len(model.calls) == 2
    lengths = [[min(model.max_seq_length, len(model.tokenizer(text, add_special_tokens=True,
               truncation=False)["input_ids"])) for text in group] for group in model.calls]
    assert len(set(length for group in lengths for length in group)) > 1
    assert result["encoder_padded_tokens"] == sum(max(group) * len(group) for group in lengths)
    assert all(len(group) <= 8 and max(group) * len(group) <= 2048 for group in lengths)

def test_measured_wrapper_reuses_counts_without_a_second_tokenizer_pass():
    from scripts.sweep_embedding_budget import MeasuredModel
    class Untokenizable:
        max_seq_length = 64
        def tokenizer(self, *args, **kwargs): raise AssertionError("duplicate tokenizer pass")
        def encode(self, texts, **kwargs): return np.zeros((len(texts), 384))
    measured = MeasuredModel(Untokenizable(), [7, 11, 3])
    measured.encode(["first", "second"])
    measured.encode(["third"])
    assert measured.groups == 2 and measured.padded_tokens == 25
    with pytest.raises(ValueError, match="prepared token-count"):
        measured.encode(["unexpected"])

def test_job_reporting_does_not_retokenize_the_prepared_input_stream():
    model = Encoder()
    original = model.tokenizer
    calls = []
    def counted(text, **kwargs):
        calls.append(text)
        return original(text, **kwargs)
    model.tokenizer = counted
    result = encode_job(case(), version='repo-embedding-v2', model=model, budget=None)
    # The fake encoder performs no tokenization; the real legacy chunker and
    # planner need exactly one count per model input, including both towers.
    # The old reporting wrapper adds a second pass over this complete stream.
    assert len(calls) == result['job_chunks']
    assert len(calls) == len(model.texts)
