"""Owned ANN transport, storage, report and daemon isolation checks."""
import io
import json
import subprocess
from types import SimpleNamespace

import pytest

from evaluation import qdrant_ann as ann
from tests.test_embedding_artifacts import bundle, evidence


def telemetry_fixture(indexed=12, hnsw=None):
    index = {"index_name": ann.VECTOR_NAME}
    if hnsw is not None:
        index["unfiltered_hnsw"] = {"count": hnsw}
    return {"result": {"collections": {"collections": [{"id": "owned", "shards": [{"segments": [{
        "info": {"vector_data": {ann.VECTOR_NAME: {"num_indexed_vectors": indexed}}},
        "vector_index_searches": [index]}]}]}]}}}


def test_telemetry_requires_matching_detailed_collection_and_real_counters():
    before = ann.telemetry(telemetry_fixture(), "owned")
    after = ann.telemetry(telemetry_fixture(hnsw=12), "owned")
    assert after["unfiltered_hnsw"] - before["unfiltered_hnsw"] == 12
    assert before["indexed_vectors"] == 12
    with pytest.raises(ValueError):
        ann.telemetry(telemetry_fixture(), "someone-else")
    with pytest.raises(ValueError, match="details_level10"):
        ann.telemetry({"result": {"collections": {"collections": [{"id": "owned"}]}}}, "owned")


def test_storage_separates_graph_config_vector_and_sparse_wal(tmp_path):
    root = tmp_path / "collections" / "owned"
    index = root / "0/segments/s/vector_index-repo_embedding"
    index.mkdir(parents=True)
    (index / "graph.bin").write_bytes(b"graph")
    (index / "links_compressed.bin").write_bytes(b"links")
    (index / "hnsw_config.json").write_bytes(ann.encode_json({"m": 16, "indexed_vector_count": 12}))
    vector = root / "0/segments/s/vector_storage-repo_embedding"
    vector.mkdir()
    (vector / "matrix.dat").write_bytes(b"vector")
    wal = root / "0/wal"
    wal.mkdir()
    with (wal / "open-1").open("wb") as stream:
        stream.truncate(1024 * 1024)
    report = ann.storage_inventory(tmp_path, "owned")
    assert report["categories"]["hnsw_graph"]["logical_bytes"] == 10
    assert report["categories"]["vectors"]["logical_bytes"] == 6
    assert report["categories"]["wal"]["logical_bytes"] == 1024 * 1024
    assert report["hnsw_files_logical_bytes"] < report["total_logical_bytes"]
    assert report["indexed_vectors_from_hnsw_configs"] == 12
    (index / "unknown-format.bin").write_bytes(b"unknown")
    with pytest.raises(ValueError, match="unknown pinned"):
        ann.storage_inventory(tmp_path, "owned")


def test_storage_rejects_symlink_even_without_following(tmp_path):
    root = tmp_path / "collections/owned"
    root.mkdir(parents=True)
    (root / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        ann.storage_inventory(tmp_path, "owned")


def test_container_ownership_is_checked_before_cleanup(monkeypatch):
    owned = ann.OwnedQdrant()
    owned.created = True
    calls = []

    def command(arguments, **kwargs):
        calls.append(arguments)
        return SimpleNamespace(returncode=0, stdout=json.dumps([{"Config": {"Labels": {ann.LABEL: "other"}}}]))
    monkeypatch.setattr(owned, "_command", command)
    with pytest.raises(RuntimeError, match="ownership"):
        owned.cleanup()
    assert all(call[0] != "rm" for call in calls)


def test_missing_container_inspect_handles_actual_docker_lowercase_error(monkeypatch):
    owned = ann.OwnedQdrant()
    monkeypatch.setattr(owned, "_command", lambda *a, **kw: SimpleNamespace(
        returncode=1, stdout="[]", stderr="error: no such object: " + owned.name))
    assert owned._owned_info() is None


def test_interrupted_startup_cleans_private_directory(monkeypatch):
    owned = ann.OwnedQdrant()

    def command(arguments, **kwargs):
        if arguments[0] == "run":
            raise KeyboardInterrupt
        return SimpleNamespace(returncode=0, stdout="[]")
    monkeypatch.setattr(owned, "_command", command)
    monkeypatch.setattr(owned, "_owned_info", lambda: None)
    with pytest.raises(KeyboardInterrupt):
        owned.__enter__()
    assert owned.root is None and not owned.created


def test_cleanup_failure_is_reported_without_replacing_original_error(monkeypatch):
    owned = ann.OwnedQdrant()
    monkeypatch.setattr(owned, "cleanup", lambda: (_ for _ in ()).throw(RuntimeError("cleanup failed")))
    original = ValueError("original failure")
    assert owned.__exit__(ValueError, original, None) is False
    assert original.__notes__ == ["cleanup failed"]


def test_request_and_result_bounds_do_not_silently_truncate(monkeypatch):
    owned = ann.OwnedQdrant()
    owned.url = "http://127.0.0.1:1"
    monkeypatch.setattr(ann, "RPC_LIMIT", 10)
    with pytest.raises(ValueError, match="request byte"):
        owned.api("/", {"long": "payload"})
    with pytest.raises(ValueError):
        ann._results({"result": {"points": [{"id": "other", "score": 1.}]}}, {"original"}, 1)


@pytest.mark.parametrize("declared", [None, "1000000"])
def test_response_is_bounded_while_reading_not_after_unlimited_read(monkeypatch, declared):
    owned = ann.OwnedQdrant()
    owned.url = "http://127.0.0.1:1"
    monkeypatch.setattr(ann, "RPC_LIMIT", 10)
    stream = io.BytesIO(b"x" * 100)
    class Response:
        headers = {} if declared is None else {"Content-Length": declared}
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read1(self, count): return stream.read(count)
    monkeypatch.setattr(owned.opener, "open", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="response byte"):
        owned.api("/")
    assert stream.tell() <= 11


def test_owned_http_never_follows_redirects():
    with pytest.raises(ValueError, match="redirected"):
        ann._NoRedirect().redirect_request(None, None, 302, "", {}, "http://external.invalid")


def test_cli_invalid_evidence_and_existing_output_never_start_server(evidence, tmp_path, monkeypatch):
    from scripts import measure_embedding_ann as cli
    monkeypatch.setattr(cli, "measure_artifact", lambda *a, **kw: pytest.fail("must not start server"))
    output = tmp_path / "result.json"
    args = ["--reports", str(evidence[0]), "--registry", str(evidence[1]), "--fixture", str(evidence[2]), "--output", str(output)]
    assert cli.main([*args, "--repeats", "6"]) == 2
    output.write_text("preserved complete result")
    assert cli.main(args) == 2
    assert output.read_text() == "preserved complete result"


def test_cli_model_failure_publishes_no_partial_report(evidence, tmp_path, monkeypatch):
    from scripts import measure_embedding_ann as cli
    monkeypatch.setattr(cli, "measure_artifact", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("fixture failure")))
    output = tmp_path / "result.json"
    assert cli.main(["--reports", str(evidence[0]), "--registry", str(evidence[1]), "--fixture", str(evidence[2]), "--output", str(output)]) == 2
    assert not output.exists()


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("docker run", .01), FileNotFoundError("docker unavailable")])
def test_cli_docker_start_failure_cleans_storage_and_exits_without_traceback(evidence, tmp_path, monkeypatch, capsys, failure):
    from scripts import measure_embedding_ann as cli
    instances = []
    original_init = ann.OwnedQdrant.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        instances.append(self)

    def command(arguments, **kwargs):
        if arguments[0] == "run":
            raise failure
        return SimpleNamespace(returncode=0, stdout="[]")
    monkeypatch.setattr(ann.OwnedQdrant, "__init__", init)
    monkeypatch.setattr(ann.OwnedQdrant, "_command", staticmethod(command))
    monkeypatch.setattr(ann.OwnedQdrant, "_owned_info", lambda self: None)
    output = tmp_path / "result.json"
    assert cli.main(["--reports", str(evidence[0]), "--registry", str(evidence[1]), "--fixture", str(evidence[2]), "--output", str(output)]) == 2
    assert len(instances) == 1 and instances[0].root is None and not instances[0].created
    assert not output.exists()
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize("host", ["tcp://remote.invalid:2375", "ssh://remote.invalid", "http://127.0.0.1:2375",
                                  "unix://remote/socket", "unix:relative", "unix:///socket?query=1", ""])
def test_remote_or_ambiguous_docker_daemons_are_rejected_before_operations(monkeypatch, host):
    monkeypatch.setenv("DOCKER_HOST", host)
    monkeypatch.setattr(ann.subprocess, "run", lambda *a, **kw: pytest.fail("must not invoke Docker"))
    with pytest.raises(ValueError, match="local Docker Unix socket"):
        ann.OwnedQdrant()


def test_local_docker_socket_is_pinned_despite_context_and_later_environment_changes(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "unix:///run/user/1000/docker.sock")
    monkeypatch.setenv("DOCKER_CONTEXT", "remote-context")
    monkeypatch.setenv("DOCKER_TLS_VERIFY", "1")
    monkeypatch.setenv("DOCKER_CERT_PATH", "/unused/remote/certificates")
    owned = ann.OwnedQdrant()
    monkeypatch.setenv("DOCKER_HOST", "tcp://remote.invalid:2375")
    calls = []
    def command(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return SimpleNamespace(returncode=0, stdout="[]", stderr="")
    monkeypatch.setattr(ann.subprocess, "run", command)
    owned._command(["inspect", owned.name])
    arguments, options = calls[0]
    assert arguments[:3] == ["docker", "--host", "unix:///run/user/1000/docker.sock"]
    assert not {"DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"} & options["env"].keys()


def test_default_docker_daemon_is_local_even_with_remote_context(monkeypatch):
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setenv("DOCKER_CONTEXT", "remote-context")
    assert ann.OwnedQdrant()._docker_prefix == ["docker", "--host", "unix:///var/run/docker.sock"]
