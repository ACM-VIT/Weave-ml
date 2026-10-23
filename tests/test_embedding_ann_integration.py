"""Opt-in actual pinned-server HNSW, files, original UUIDs and cleanup."""
import os

import pytest

from evaluation import qdrant_ann as ann
from tests.test_embedding_artifacts import bundle


@pytest.mark.integration
@pytest.mark.parametrize("dimension", [384, 1024])
def test_owned_server_build_query_storage_and_cleanup(tmp_path, dimension, monkeypatch):
    if os.getenv("RUN_EMBEDDING_ANN_INTEGRATION") != "1":
        pytest.skip("set RUN_EMBEDDING_ANN_INTEGRATION=1 for a new owned Docker/Qdrant server")
    # Hostile environment settings cannot route this tool at a live service.
    monkeypatch.setenv("QDRANT_URL", "https://must-not-be-used.invalid")
    monkeypatch.setenv("QDRANT_API_KEY", "must-not-be-used")
    evidence = bundle(tmp_path, dimension=dimension)
    artifact = ann.load_artifact(*evidence)
    report = ann.measure_artifact(artifact, repeats=1)
    assert report["search_telemetry_deltas"]["unfiltered_hnsw"] >= 12
    assert report["search_telemetry_deltas"]["unfiltered_plain"] == 0
    assert report["ann_exact_tie_aware_recall"] == [1.] * 12
    assert report["effective_configuration"]["params"]["vectors"][ann.VECTOR_NAME]["size"] == dimension
    assert report["storage"]["indexed_vectors_from_hnsw_configs"] == 12
    assert report["ann_index_bytes"] > 0
    assert report["storage"]["categories"]["wal"]["logical_bytes"] > report["ann_index_bytes"]
    assert report["points"] == 12


@pytest.mark.integration
def test_approximate_flag_without_hnsw_index_does_not_qualify(tmp_path):
    if os.getenv("RUN_EMBEDDING_ANN_INTEGRATION") != "1":
        pytest.skip("explicit owned-server integration required")
    artifact = ann.load_artifact(*bundle(tmp_path))
    with ann.OwnedQdrant() as owned:
        collection = "not_indexed_" + owned.run_id[:16]
        path = "/collections/" + collection
        owned.api(path, {"vectors": {ann.VECTOR_NAME: {"size": 384, "distance": "Cosine"}},
                        "optimizers_config": {"indexing_threshold": 0}}, "PUT")
        owned.api(path + "/points?wait=true", {"points": [
            {"id": repo_id, "vector": {ann.VECTOR_NAME: row.tolist()}}
            for repo_id, row in zip(artifact.repository_ids, artifact.documents)]}, "PUT")
        owned.api(path + "/points/query", {"query": artifact.queries[0].tolist(), "using": ann.VECTOR_NAME,
                  "params": {"exact": False}, "limit": 3}, "POST")
        observed = ann.telemetry(owned.api("/telemetry?details_level=10"), collection)
        assert observed["indexed_vectors"] == 0
        assert observed["unfiltered_hnsw"] == 0
        owned.stop()
        with pytest.raises(ValueError, match="no actual HNSW"):
            ann.storage_inventory(owned.storage, collection)


@pytest.mark.integration
def test_actual_query_interruption_removes_owned_server_and_storage(tmp_path, monkeypatch):
    if os.getenv("RUN_EMBEDDING_ANN_INTEGRATION") != "1":
        pytest.skip("explicit owned-server integration required")
    artifact = ann.load_artifact(*bundle(tmp_path))
    original_api = ann.OwnedQdrant.api
    interrupted = []

    def api(self, path, *args, **kwargs):
        if path.endswith("/points/query"):
            interrupted.append(self)
            raise KeyboardInterrupt("deliberate cancellation of actual indexed experiment")
        return original_api(self, path, *args, **kwargs)
    monkeypatch.setattr(ann.OwnedQdrant, "api", api)
    with pytest.raises(KeyboardInterrupt):
        ann.measure_artifact(artifact, repeats=1)
    assert len(interrupted) == 1
    assert interrupted[0].root is None
    assert not interrupted[0].created
    assert interrupted[0]._owned_info() is None
