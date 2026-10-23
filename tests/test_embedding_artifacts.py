"""Offline encoder evidence: real archive, identity and atomic publication checks."""
from __future__ import annotations

import io
import json
from pathlib import Path
import struct
import zipfile

import numpy as np
import pytest

from evaluation import embedding_artifacts as ann
from evaluation.embedding_registry import EmbeddingConfig


def bundle(root: Path, *, dimension=384, name="diagnostic", seed=19):
    root.mkdir(exist_ok=True)
    config = EmbeddingConfig.model_validate(dict(name=name, model_id="diagnostic/not-a-serving-model",
        revision="a" * 40, native_dimension=dimension, output_dimension=dimension,
        max_tokens=256, pooling="mean", normalize=True, instruction_aware=False,
        query_prompt="", document_prompt=""))
    registry = {"schema_version": 1, "models": [config.model_dump()]}
    ids = [f"00000000-0000-4000-8000-{index + 1:012d}" for index in range(12)]
    fixture = {"fixture_version": "ann-mechanics-v1", "provenance": "synthetic mechanical test; no quality labels",
               "repositories": [{"repo_id": repo_id, "full_name": f"test/repo-{index}", "readme": "fixture"}
                                for index, repo_id in enumerate(ids)],
               "queries": [{"query": f"diagnostic-{index}", "relevant_repo_ids": [repo_id]}
                           for index, repo_id in enumerate(ids)]}
    registry_path, fixture_path = root / "registry.json", root / "fixture.json"
    registry_path.write_bytes(ann.encode_json(registry))
    fixture_path.write_bytes(ann.encode_json(fixture))
    vectors = np.random.default_rng(seed).normal(size=(12, dimension)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    metadata = {"schema_version": 1, "embedding_version": config.version, "configuration": config.model_dump(),
                "fixture_sha256": ann.digest(fixture_path.read_bytes()), "repository_ids": ids,
                "queries": [row["query"] for row in fixture["queries"]], "roles": ["query", "document"]}
    artifact_path = root / "vectors.npz"
    np.savez_compressed(artifact_path, documents=vectors, queries=vectors, metadata_json=np.asarray(json.dumps(metadata)))
    report = {"schema_version": 1, "configuration": config.model_dump(), "embedding_version": config.version,
              "registry_sha256": ann.digest(registry_path.read_bytes()), "fixture_sha256": metadata["fixture_sha256"],
              "fixture_version": fixture["fixture_version"], "label_provenance": fixture["provenance"],
              "sample_counts": {"repositories": 12, "queries": 12},
              "cost": {"shadow_artifact": artifact_path.name, "shadow_artifact_bytes": artifact_path.stat().st_size,
                       "raw_repository_vector_bytes": vectors.nbytes}}
    report_path = root / "encoder.json"
    report_path.write_bytes(ann.encode_json(report))
    return report_path, registry_path, fixture_path


@pytest.fixture
def evidence(tmp_path):
    return bundle(tmp_path)


def mutate_report(paths, mutation):
    path = paths[0]
    report = json.loads(path.read_bytes())
    mutation(report)
    path.write_bytes(ann.encode_json(report))


def test_real_npz_identity_validation_preserves_original_immutable_rows(evidence):
    artifact = ann.load_artifact(*evidence)
    assert artifact.documents.shape == (12, 384)
    assert artifact.repository_ids[0] == "00000000-0000-4000-8000-000000000001"
    assert not artifact.documents.flags.writeable
    assert artifact.configuration.version == artifact.report["embedding_version"]
    assert artifact.artifact_sha256 == ann.digest(Path(artifact.artifact_path).read_bytes())


@pytest.mark.parametrize("field", ["registry_sha256", "fixture_sha256", "embedding_version", "fixture_version", "label_provenance"])
def test_tampered_report_binding_fails_before_server(evidence, field):
    mutate_report(evidence, lambda report: report.__setitem__(field, "wrong"))
    with pytest.raises(ValueError):
        ann.load_artifact(*evidence)


@pytest.mark.parametrize("name", ["../vectors.npz", "/tmp/vectors.npz", "vectors.json", "directory/vectors.npz"])
def test_no_artifact_path_traversal(evidence, name):
    mutate_report(evidence, lambda report: report["cost"].__setitem__("shadow_artifact", name))
    with pytest.raises(ValueError, match="basename"):
        ann.load_artifact(*evidence)


def test_wrong_registry_model_with_same_dimensions_rejected(evidence):
    mutate_report(evidence, lambda report: report["configuration"].__setitem__("model_id", "another/model"))
    with pytest.raises(ValueError, match="registry"):
        ann.load_artifact(*evidence)


@pytest.mark.parametrize("change", ["row_order", "role", "vector", "duplicate_member", "object", "declared_shape", "zip_budget"])
def test_hostile_archives_are_rejected_without_numpy_shape_allocation(evidence, monkeypatch, change):
    path = evidence[0].parent / "vectors.npz"
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    if change in {"row_order", "role"}:
        metadata = json.loads(np.load(io.BytesIO(members["metadata_json.npy"]), allow_pickle=False).item())
        if change == "row_order":
            metadata["repository_ids"].reverse()
        else:
            metadata["roles"].reverse()
        stream = io.BytesIO()
        np.save(stream, np.asarray(json.dumps(metadata)))
        members["metadata_json.npy"] = stream.getvalue()
    elif change == "vector":
        stream = io.BytesIO()
        np.save(stream, np.full((12, 384), np.nan, dtype=np.float32))
        members["documents.npy"] = stream.getvalue()
    elif change == "object":
        stream = io.BytesIO()
        np.save(stream, np.asarray([[object()]], dtype=object), allow_pickle=True)
        members["documents.npy"] = stream.getvalue()
    elif change == "declared_shape":
        header = repr({"descr": "<f4", "fortran_order": False, "shape": (2**60, 384)}).encode()
        members["documents.npy"] = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
        if change == "duplicate_member":
            archive.writestr("documents.npy", members["documents.npy"])
    mutate_report(evidence, lambda report: report["cost"].__setitem__("shadow_artifact_bytes", path.stat().st_size))
    if change == "zip_budget":
        monkeypatch.setattr(ann, "ARCHIVE_LIMIT", path.stat().st_size + 1)
    with pytest.raises(ValueError):
        ann.load_artifact(*evidence)


def test_optional_historical_digest_is_checked(evidence):
    mutate_report(evidence, lambda report: report["cost"].__setitem__("shadow_artifact_sha256", "0" * 64))
    with pytest.raises(ValueError, match="digest"):
        ann.load_artifact(*evidence)


def test_inputs_and_outputs_refuse_symlinks_and_existing_evidence(evidence, tmp_path):
    alias = tmp_path / "alias.json"
    alias.symlink_to(evidence[0])
    with pytest.raises(OSError):
        ann.read_json(alias)
    output = tmp_path / "report.json"
    ann.publish_new_json(output, {"complete": True})
    before = output.read_bytes()
    with pytest.raises(ValueError):
        ann.publish_new_json(output, {"complete": False})
    assert output.read_bytes() == before
    assert not list(tmp_path.glob(".embedding-ann-*"))


def test_atomic_report_publication_does_not_replace_racing_creator(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    link = ann.os.link

    def race(source, target, **kwargs):
        output.write_text("other complete result")
        return link(source, target, **kwargs)
    monkeypatch.setattr(ann.os, "link", race)
    with pytest.raises(FileExistsError):
        ann.publish_new_json(output, {"complete": True})
    assert output.read_text() == "other complete result"
    assert not list(tmp_path.glob(".embedding-ann-*"))


def test_duplicate_and_nonfinite_json_never_enter_validation():
    for raw in [b'{"id":1,"id":2}', b'{"value":NaN}', b'[' * 10000 + b']' * 10000]:
        with pytest.raises(ValueError):
            ann.decode_json(raw)
