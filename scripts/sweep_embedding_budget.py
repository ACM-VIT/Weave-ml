"""Offline diagnostic sweep; explicit experimental budgets, no production policy.

Each cohort runs in a fresh process, using only cached pinned MiniLM. Existing
chunking/section weights are reused. A budget is CHUNKS:TOKENS including metadata
and topics; experimental README selection is a whole-chunk prefix. No vectors
are persisted. Cosine drift is not retrieval quality.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import sys
import time
import tracemalloc

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import TOKEN_SECTION_EMBEDDING_VERSION
from embedding.embeddings import SentenceTransformerEmbedder, aggregate_vectors, chunk_text
from embedding.readme_embedding import prepare_readme_chunks, readme_source
from embedding.repository_embedding import (
    RepositoryEmbeddingConfig, build_metadata_text, build_readme_text,
    build_topic_text, combine_repo_tower,
)

PIN = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


@dataclass(frozen=True)
class Budget:
    chunks: int
    tokens: int

    @classmethod
    def parse(cls, value: str):
        chunks, tokens = map(int, value.split(":"))
        if not 3 <= chunks <= 128 or not 3 <= tokens <= 65536:
            raise ValueError("experimental budgets require 3..128 chunks and 3..65536 tokens")
        return cls(chunks, tokens)


def select_prefix(counts: list[int], mandatory: list[int], budget: Budget | None) -> int:
    """Use exact model-truncated token counts, including special/overlap tokens."""
    if budget is None:
        return len(counts)
    used, selected = sum(mandatory), 0
    for count in counts:
        if selected + len(mandatory) >= budget.chunks or used + count > budget.tokens:
            break
        selected += 1
        used += count
    if counts and not selected:
        raise ValueError("experimental budget cannot retain one README chunk plus mandatory towers")
    return selected


def fixtures(include_sibling=False):
    words = "Search indexes immutable document vectors with bounded workers and versioned storage. "
    cases = [
        ("synthetic_long_sections", "# Overview\n" + words * 260 + "\n# Architecture\n" + words * 180 + "\n# Installation\n" + "Install package and verify setup. " * 160),
        ("synthetic_heading_free", words * 700),
        ("synthetic_multilingual", "# Overview\n" + "向量检索 数据库 搜索 文档. البحث في المستندات. दस्तावेज खोज. café retrieval. " * 650),
    ]
    result = [{"name": name, "readme": text, "provenance": "generated stress diagnostic, not representative data"} for name, text in cases]
    repos = [("local_ml_readme", ROOT)]
    if include_sibling:
        repos.append(("local_weave_readme", ROOT.parent / "Weave"))
    for name, repo in repos:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        text = subprocess.check_output(["git", "show", f"{commit}:README.md"], cwd=repo, text=True)
        if len(text) > 200000:
            raise ValueError("local fixture exceeds diagnostic input guard")
        result.append({"name": name, "readme": text, "provenance": f"local git object {commit}:README.md"})
    for case in result:
        case["sha256"] = hashlib.sha256(case["readme"].encode()).hexdigest()
    return result


class MeasuredModel:
    def __init__(self, model):
        self.model = model
        self.tokenizer = model.tokenizer
        self.max_seq_length = model.max_seq_length
        self.groups = 0
        self.padded_tokens = 0

    def encode(self, texts, **kwargs):
        self.groups += 1
        counts = [min(self.max_seq_length, len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])) for text in texts]
        self.padded_tokens += max(counts) * len(counts)
        return self.model.encode(texts, **kwargs)


def encode_job(case, *, version, model, budget):
    config = RepositoryEmbeddingConfig(version=version)
    repo = {"full_name": "diagnostic/repository", "description": "Offline budget diagnostic", "readme": case["readme"], "topics": ["search"], "languages": ["Python"]}
    tracemalloc.start()
    started, cpu = time.perf_counter(), time.process_time()
    modern = version == TOKEN_SECTION_EMBEDDING_VERSION
    source = readme_source(repo, repo) if modern else build_readme_text(repo)
    chunks = (prepare_readme_chunks(source, model.tokenizer, sequence_limit=model.max_seq_length,
                                     overlap_tokens=config.readme_chunk_overlap_tokens) if modern else
              chunk_text(source, max_chars=config.readme_chunk_chars, overlap_chars=config.readme_chunk_overlap_chars))
    texts = [chunk.text for chunk in chunks]
    mandatory = [build_metadata_text(repo), build_topic_text(repo)]
    raw = [len(model.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) for text in texts]
    counts = [min(count, model.max_seq_length) for count in raw]
    required = [min(model.max_seq_length, len(model.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])) for text in mandatory]
    keep = select_prefix(counts, required, budget)
    measured = MeasuredModel(model)
    vectors = SentenceTransformerEmbedder(model=measured).embed_texts(texts[:keep] + mandatory)
    readme_vector = aggregate_vectors(vectors[:keep], weights=[chunk.weight for chunk in chunks[:keep]] if modern else None)
    final = combine_repo_tower(readme_embedding=readme_vector, metadata_embedding=vectors[-2], topic_embedding=vectors[-1], weights=config.tower_weights)
    wall, cpu_elapsed = time.perf_counter() - started, time.process_time() - cpu
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    classes = {}
    for index, chunk in enumerate(chunks):
        key = chunk.section_class if modern else "character_chunks"
        item = classes.setdefault(key, {"available_chunks": 0, "retained_chunks": 0, "available_encoded_tokens": 0, "retained_encoded_tokens": 0})
        item["available_chunks"] += 1
        item["available_encoded_tokens"] += counts[index]
        if index < keep:
            item["retained_chunks"] += 1
            item["retained_encoded_tokens"] += counts[index]
    headings = re.findall(r"^#{1,6}\s+(.+)$", case["readme"], re.M)[:32]
    retained_text = "\n".join(texts[:keep]).casefold()
    output = {"name": case["name"], "version": version, "source_chars": len(case["readme"]),
              "prepared_source_chars": len(source), "available_readme_chunks": len(chunks), "retained_readme_chunks": keep,
              "job_chunks": keep + 2, "job_encoded_tokens": sum(counts[:keep]) + sum(required),
              "baseline_readme_tokens_before_model_truncation": sum(raw),
              "baseline_readme_tokens_after_model_truncation": sum(counts),
              "retained_encoded_token_fraction": sum(counts[:keep]) / sum(counts),
              "model_truncated_input_fraction": 1 - sum(counts) / sum(raw),
              "budget_dropped_encoded_token_fraction": 1 - sum(counts[:keep]) / sum(counts),
              "section_class_coverage": classes,
              "heading_label_presence": {heading: heading.casefold() in retained_text for heading in headings},
              "wall_seconds": wall, "cpu_seconds": cpu_elapsed,
              "python_tracemalloc_peak_bytes": peak, "encoder_groups": measured.groups,
              "encoder_padded_tokens": measured.padded_tokens, "final_vector": final}
    if budget:
        assert output["job_chunks"] <= budget.chunks and output["job_encoded_tokens"] <= budget.tokens
    return output


@contextmanager
def no_network():
    originals = socket.socket.connect, socket.socket.connect_ex, socket.create_connection
    attempts = []
    def deny(*args, **kwargs):
        attempts.append(True)
        raise RuntimeError("network forbidden for offline budget sweep")
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    try:
        yield attempts
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = originals


def source_hashes():
    names = ("embedding/readme_embedding.py", "embedding/embeddings.py", "embedding/repository_embedding.py", "embedding/runtime.py", "config.py", "scripts/sweep_embedding_budget.py")
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def cohort(budget, include_sibling):
    before = source_hashes()
    with no_network() as attempts:
        import torch
        from sentence_transformers import SentenceTransformer
        from embedding.runtime import BoundedEmbeddingModel
        torch.set_num_threads(1)
        loaded = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", revision=PIN, local_files_only=True, device="cpu")
        model = BoundedEmbeddingModel(loaded, max_concurrency=1)
        model.encode(["Offline warmup"], normalize_embeddings=True, show_progress_bar=False)
        cases = fixtures(include_sibling)
        rows = [encode_job(case, version=version, model=model, budget=budget)
                for case in cases for version in ("repo-embedding-v2", TOKEN_SECTION_EMBEDDING_VERSION)]
    if attempts or before != source_hashes():
        raise RuntimeError("network attempt or measured source drift")
    return {"source_hashes": before, "model_revision": PIN, "max_seq_length": model.max_seq_length,
            "network_attempts": len(attempts), "budget": vars(budget) if budget else None,
            "cohort_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "fixtures": [{key: value for key, value in case.items() if key != "readme"} for case in cases], "rows": rows}


def verify_cohorts(cohorts):
    """Refuse comparisons across changing source or local README objects."""
    def identity(group):
        return (group["source_hashes"], group["model_revision"], group["max_seq_length"],
                [(case["name"], case["sha256"]) for case in group["fixtures"]])
    expected = identity(cohorts[0])
    if any(identity(group) != expected for group in cohorts[1:]):
        raise RuntimeError("source/model/fixture identity changed between budget cohorts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", nargs="+", required=True, help="explicit CHUNKS:TOKENS experimental values")
    parser.add_argument("--include-sibling", action="store_true", help="also read sibling Weave's committed README")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.child:
        result = cohort(None if args.budgets == ["control"] else Budget.parse(args.budgets[0]), args.include_sibling)
    else:
        budgets = [Budget.parse(value) for value in args.budgets]
        if len(budgets) > 4:
            parser.error("at most four experimental budgets")
        cohorts = []
        for value in ["control", *args.budgets]:
            command = [sys.executable, str(Path(__file__).resolve()), "--child", "--budgets", value]
            if args.include_sibling:
                command.append("--include-sibling")
            cohorts.append(json.loads(subprocess.check_output(command, cwd=ROOT, text=True, timeout=180)))
        verify_cohorts(cohorts)
        baseline = {(row["name"], row["version"]): row["final_vector"] for row in cohorts[0]["rows"]}
        for group in cohorts:
            for row in group["rows"]:
                vector = row.pop("final_vector")
                row["cosine_to_uncapped_control"] = float(np.clip(np.dot(vector, baseline[row["name"], row["version"]]), -1., 1.))
        result = {"schema": "embedding-budget-sweep-v1", "scope": "local diagnostic; no production budget/overflow/reindex decision or retrieval-quality evidence", "cohorts": cohorts}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
