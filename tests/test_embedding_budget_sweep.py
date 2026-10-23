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


def test_source_identity_includes_cleaner_and_changes_with_its_bytes(monkeypatch, tmp_path):
    from scripts import sweep_embedding_budget as bench
    names = ["utils/readme_processor.py", "embedding/readme_embedding.py", "embedding/embeddings.py", "embedding/repository_embedding.py", "embedding/runtime.py", "config.py", "scripts/sweep_embedding_budget.py", "pyproject.toml", "uv.lock"]
    for name in names:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("original")
    monkeypatch.setattr(bench, "ROOT", tmp_path)
    monkeypatch.setattr(bench, "local_git", lambda *args: "\0".join(names) + "\0", raising=False)
    original = bench.source_hashes()
    (tmp_path / "utils/readme_processor.py").write_text("changed cleaner")
    changed = bench.source_hashes()
    assert original["utils/readme_processor.py"] != changed["utils/readme_processor.py"]
    assert len(original) == len(names)


def test_missing_promisor_blob_never_invokes_git_transport(tmp_path, monkeypatch):
    import os
    if os.name != 'posix':
        pytest.skip('POSIX executable Git remote-helper fixture')
    import subprocess
    from scripts.sweep_embedding_budget import local_git
    repo = tmp_path / "partial"
    repo.mkdir()
    def git(*args, data=None):
        return subprocess.check_output(["git", *args], cwd=repo, input=data, text=True).strip()
    git("init", "--quiet")
    blob = git("hash-object", "-w", "--stdin", data="README content")
    # A real missing promisor object would normally trigger a remote helper.
    git("config", "remote.origin.url", "proof::missing")
    git("config", "remote.origin.promisor", "true")
    git("config", "extensions.partialClone", "origin")
    (repo / ".git" / "objects" / blob[:2] / blob[2:]).unlink()
    helper = tmp_path / "git-remote-proof"
    marker = tmp_path / "transport-invoked"
    helper.write_text("#!/bin/sh\nprintf attempted > \"$PROOF_MARKER\"\nexit 1\n")
    helper.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("PROOF_MARKER", str(marker))
    with pytest.raises(RuntimeError, match="lazy fetching disabled"):
        local_git(repo, "show", blob)
    assert not marker.exists()
    # Positive control: the same missing object does invoke transport without
    # the guard, so absence above is not an inert fixture false-positive.
    env = dict(os.environ)
    env.pop("GIT_NO_LAZY_FETCH", None)
    subprocess.run(["git", "show", blob], cwd=repo, env=env,
                   capture_output=True, text=True, timeout=15, check=False)
    assert marker.read_text() == "attempted"


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


def test_unavailable_rss_is_explicitly_unknown(monkeypatch):
    from scripts import sweep_embedding_budget as bench
    monkeypatch.setattr(bench, 'resource', None)
    assert bench.peak_rss_mib() is None


def test_cli_help_remains_available_without_unix_resource_module():
    import subprocess
    import sys
    from pathlib import Path
    script = '''
import builtins, runpy, sys
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'resource':
        raise ImportError('resource unavailable on this platform')
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
sys.argv = ['sweep_embedding_budget', '--help']
runpy.run_path('scripts/sweep_embedding_budget.py', run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-c', script],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '--budgets' in result.stdout


@pytest.mark.parametrize('platform_name,expected', [('linux', 128/1024), ('darwin', 128/(1024**2))])
def test_peak_rss_preserves_platform_units(monkeypatch, platform_name, expected):
    from types import SimpleNamespace
    from scripts import sweep_embedding_budget as bench
    fake = SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda who: SimpleNamespace(ru_maxrss=128))
    monkeypatch.setattr(bench, 'resource', fake)
    monkeypatch.setattr(bench.sys, 'platform', platform_name)
    assert bench.peak_rss_mib() == expected
