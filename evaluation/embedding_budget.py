"""Bounded frozen inputs and retrieval evidence for offline budget experiments.

Shares the existing encoder-comparison fixture's canonical repository IDs,
query/relevance sets and explicit provenance. No production cap is selected.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import uuid

MAX_BYTES = 32 * 1024 * 1024
MAX_REPOSITORIES = 64
MAX_QUERIES = 64


def _invalid():
    raise ValueError("Invalid bounded embedding-budget fixture")


def _pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            _invalid()
        result[key] = value
    return result


def _text(value, maximum, *, empty=False):
    if not isinstance(value, str) or len(value) > maximum or (not empty and not value.strip()):
        _invalid()
    try:
        value.encode("utf8")
    except UnicodeError:
        _invalid()


def _identifier(value):
    if not isinstance(value, str):
        _invalid()
    try:
        if str(uuid.UUID(value)) != value:
            _invalid()
    except (ValueError, AttributeError):
        _invalid()


@dataclass(frozen=True)
class BudgetFixture:
    fixture_version: str
    sha256: str
    label_provenance: str
    cases: tuple[dict, ...]
    queries: tuple[dict, ...]


def load_budget_fixture(path: Path) -> BudgetFixture:
    """Read one immutable bounded regular JSON file; never follow a final link."""
    path = Path(path)
    path_stat = path.lstat()
    if not stat.S_ISREG(path_stat.st_mode):
        _invalid()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as source:
        before = os.fstat(source.fileno())
        if (not stat.S_ISREG(before.st_mode) or not 1 <= before.st_size <= MAX_BYTES
                or (path_stat.st_dev, path_stat.st_ino) != (before.st_dev, before.st_ino)):
            _invalid()
        raw = source.read(MAX_BYTES + 1)
        after = os.fstat(source.fileno())
    if (len(raw) != before.st_size or len(raw) > MAX_BYTES or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        _invalid()
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=lambda _: _invalid())
    except (ValueError, UnicodeError, RecursionError):
        _invalid()
    if not isinstance(value, dict) or set(value) != {"fixture_version", "provenance", "repositories", "queries"}:
        _invalid()
    _text(value["fixture_version"], 128)
    _text(value["provenance"], 4096)
    repositories, queries = value["repositories"], value["queries"]
    if (not isinstance(repositories, list) or not 1 <= len(repositories) <= MAX_REPOSITORIES
            or not isinstance(queries, list) or not 1 <= len(queries) <= MAX_QUERIES):
        _invalid()
    ids, cases = set(), []
    allowed = {"repo_id", "full_name", "description", "readme", "topics", "languages", "primary_language", "content_version"}
    for repository in repositories:
        if not isinstance(repository, dict) or not set(repository) <= allowed or not {"repo_id", "full_name", "readme"} <= set(repository):
            _invalid()
        _identifier(repository["repo_id"])
        if repository["repo_id"] in ids:
            _invalid()
        ids.add(repository["repo_id"])
        if "content_version" in repository and (type(repository["content_version"]) is not int
                or not 1 <= repository["content_version"] <= 2**63-1):
            _invalid()
        _text(repository["full_name"], 256)
        _text(repository.get("description", ""), 10000, empty=True)
        _text(repository["readme"], 200000, empty=True)
        language = repository.get("primary_language")
        if language is not None:
            _text(language, 128)
        for field in ("topics", "languages"):
            items = repository.get(field, [])
            if not isinstance(items, list) or len(items) > 100:
                _invalid()
            for item in items:
                _text(item, 128)
            if len(set(items)) != len(items):
                _invalid()
        cases.append({"name": repository["repo_id"], "readme": repository["readme"],
            "repository": dict(repository), "provenance": value["provenance"],
            "sha256": hashlib.sha256(repository["readme"].encode()).hexdigest()})
    seen = set()
    for query in queries:
        if not isinstance(query, dict) or set(query) != {"query", "relevant_repo_ids"}:
            _invalid()
        _text(query["query"], 4096)
        if query["query"] in seen:
            _invalid()
        seen.add(query["query"])
        relevant = query["relevant_repo_ids"]
        if (not isinstance(relevant, list) or not 1 <= len(relevant) <= len(ids)
                or any(not isinstance(item, str) for item in relevant)
                or len(set(relevant)) != len(relevant) or not set(relevant) <= ids):
            _invalid()
    return BudgetFixture(value["fixture_version"], hashlib.sha256(raw).hexdigest(), value["provenance"],
        tuple(cases), tuple(dict(query) for query in queries))


def budget_retrieval_report(document_vectors, query_vectors, fixture: BudgetFixture, *, k: int):
    """Actual supplied-qrel retrieval metrics; never a production quality gate."""
    import numpy as np
    from evaluation.retrieval_metrics import retrieval_metrics

    if type(k) is not int or not 1 <= k <= len(fixture.cases):
        raise ValueError("retrieval k must fit the frozen corpus")
    documents, queries = np.asarray(document_vectors), np.asarray(query_vectors)
    if (documents.shape != (len(fixture.cases), 384) or queries.shape != (len(fixture.queries), 384)
            or not np.all(np.isfinite(documents)) or not np.all(np.isfinite(queries))
            or not np.allclose(np.linalg.norm(documents, axis=1), 1, atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.norm(queries, axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError("retrieval requires finite normalized compatible MiniLM vectors")
    scores = np.clip(queries @ documents.T, -1, 1)
    # Report stable query references rather than copying supplied private text.
    labels = [{"query": hashlib.sha256(row["query"].encode()).hexdigest(),
               "relevant_repo_ids": row["relevant_repo_ids"]} for row in fixture.queries]
    report = retrieval_metrics(scores, labels, [case["name"] for case in fixture.cases], k=k)
    return {"fixture_sha256": fixture.sha256, "fixture_version": fixture.fixture_version,
        "label_provenance": fixture.label_provenance, "query_reference_kind": "sha256_of_exact_query_utf8",
        "quality_scope": "supplied binary qrels; representativeness/provenance not authenticated; no production qualification",
        "unjudged_scope": "only supplied relevant IDs count toward these binary-qrel metrics",
        "repository_ids": [case["name"] for case in fixture.cases], **report}
