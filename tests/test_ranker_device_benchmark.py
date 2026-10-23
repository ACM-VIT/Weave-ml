"""Real checked-in model checks for the offline device diagnostic."""
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from scripts.benchmark_ranker_devices import (
    ARTIFACTS, fixtures, run_case, summary, validate_artifacts, validate_options,
)


def test_actual_trained_cpu_serving_and_stages():
    result = run_case("cpu", 3, 2, 1)
    assert result["status"] == "measured"
    assert result["warm_serving"]["samples"] == 2
    assert result["cpu_reference_max_abs_error"] == 0
    assert result["cpu_reference_order_equal"] is True
    assert result["artifact"]["model_sha256"] == "ab7d804261bf204e51962192c26b445e058057993ddb67ff75cf32791cca9215"
    assert result["artifact"]["qualification_method"] == "manual_override"
    assert result["automatic_qualification_errors"]  # Remain honest about synthetic training.
    assert result["cuda_peak_allocated_bytes"] is None
    assert result["cuda_peak_allocated_bytes_scope"] == "post_warmup_headline_and_instrumented_calls"
    assert result["timed_compact_max_abs_error"] == 0
    try:
        import resource
    except ImportError:
        assert result["process_peak_rss_kib"] is None
        assert result["process_peak_rss_bytes"] is None
    else:
        assert result["process_peak_rss_kib"] > 0
        assert result["process_peak_rss_bytes"] == result["process_peak_rss_kib"] * 1024
    assert result["process_peak_rss_scope"] == "whole_process_imports_fixtures_headline_and_instrumented_calls"
    assert set(result["instrumented_stages"]) == {"host_input", "input_placement", "forward", "output_cpu", "remaining"}
    assert all(value["samples"] == 2 and value["p50_ms"] >= 0 for value in result["instrumented_stages"].values())


def test_cpu_child_command_uses_real_artifacts_without_network():
    script = ARTIFACTS.parent / "scripts/benchmark_ranker_devices.py"
    result = subprocess.run([sys.executable, str(script), "--device", "cpu", "--counts", "1", "--repeats", "1"],
                            timeout=30, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["status"] == "measured" and report["candidates"] == 1
    assert report["cpu_reference_order_equal"] is True


def test_unavailable_cuda_has_no_model_or_timing_claim(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("scripts.benchmark_ranker_devices.validate_artifacts", lambda *_: pytest.fail("Unavailable model loaded"))
    result = run_case("cuda", 150, 2, 1)
    assert result["status"] == "unavailable"
    assert "warm_serving" not in result and "artifact" not in result


@pytest.mark.parametrize("name", ["heavy_ranker.pt", "feature_scaler.json"])
def test_artifact_tampering_rejected_before_model_load(tmp_path, name):
    for file in ("heavy_ranker.pt", "feature_scaler.json", "model_manifest.json"):
        shutil.copyfile(ARTIFACTS / file, tmp_path / file)
    with (tmp_path / name).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="Unverified benchmark artifact"):
        validate_artifacts(tmp_path)


@pytest.mark.parametrize("values", [(0, 1, 1), (2001, 1, 1), (1, 0, 1), (1, 201, 1), (1, 1, 0), (1, 1, 9), (True, 1, 1)])
def test_execution_bounds(values):
    with pytest.raises(ValueError):
        validate_options(*values)


@pytest.mark.parametrize("samples", [[], [float("nan")], [float("inf")], [-1.0]])
def test_invalid_measurements_not_reported(samples):
    with pytest.raises(ValueError):
        summary(samples)


def test_fixture_dimensions_seed_and_normalization():
    user, skills, repos = fixtures(45)
    second_user, _, second_repos = fixtures(45)
    assert skills == ["python"] and len(repos) == 45
    np.testing.assert_array_equal(user, second_user)
    for first, second in zip(repos, second_repos):
        np.testing.assert_array_equal(first["embedding"], second["embedding"])
        assert first["embedding"].shape == (384,)
        assert np.isclose(np.linalg.norm(first["embedding"]), 1)


@pytest.mark.parametrize('only_call', [None, 5])
def test_timed_compact_score_regression_with_unchanged_order_is_rejected(monkeypatch, only_call):
    from inference.ranker_service import RankerService
    original = RankerService.score_batch_compact
    calls = 0
    def changed(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        rows = original(self, *args, **kwargs)
        if only_call is None or calls == only_call:
            return [{**row, 'final_score': row['final_score']+.01} for row in rows]
        return rows
    monkeypatch.setattr(RankerService, 'score_batch_compact', changed)
    # First call + three warmups precede timed call5; later timed output is clean.
    with pytest.raises(AssertionError):
        run_case('cpu', 3, 2, 1)


def test_source_change_during_case_is_rejected(monkeypatch):
    from scripts import benchmark_ranker_devices as benchmark
    state = {'inference/ranker_service.py': 'before'}
    monkeypatch.setattr(benchmark, '_IMPORTED_SOURCE_SHA256', dict(state))
    monkeypatch.setattr(benchmark, 'source_hashes', lambda: dict(state))
    def changed(*args):
        state['inference/ranker_service.py'] = 'after'
        return {'status': 'measured'}
    monkeypatch.setattr(benchmark, '_run_case', changed)
    with pytest.raises(RuntimeError, match='source changed'):
        benchmark.run_case('cpu', 1, 1, 1)


def test_timed_comparison_uses_identity_and_all_iterations():
    from scripts.benchmark_ranker_devices import compare_timed_compact
    reference = [{'repo_id':'a','final_score':.2},{'repo_id':'b','final_score':.1}]
    error, order = compare_timed_compact([(('b',.1),('a',.2))], reference)
    assert error == 0 and order is False
    with pytest.raises(AssertionError):
        compare_timed_compact([(('a',.21),('b',.11)), (('a',.2),('b',.1))], reference)


def test_import_help_and_unknown_memory_without_unix_resource():
    code = """
import builtins
real_import = builtins.__import__
def without_resource(name, *args, **kwargs):
    if name == 'resource':
        raise ModuleNotFoundError("No module named 'resource'")
    return real_import(name, *args, **kwargs)
builtins.__import__ = without_resource
from scripts import benchmark_ranker_devices as benchmark
assert benchmark._peak_process_rss_bytes() is None
benchmark.main(['--help'])
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '--counts' in result.stdout


@pytest.mark.parametrize('platform_name,expected', [('linux',128*1024),('darwin',128)])
def test_memory_reports_normalized_platform_bytes(monkeypatch,platform_name,expected):
    from types import SimpleNamespace
    from scripts import benchmark_ranker_devices as benchmark
    fake=SimpleNamespace(RUSAGE_SELF=0,getrusage=lambda who:SimpleNamespace(ru_maxrss=128))
    monkeypatch.setitem(sys.modules,'resource',fake)
    monkeypatch.setattr(benchmark.sys,'platform',platform_name)
    assert benchmark._peak_process_rss_bytes()==expected


def test_actual_cpu_measurement_keeps_unavailable_memory_unknown(monkeypatch):
    from scripts import benchmark_ranker_devices as benchmark
    monkeypatch.setattr(benchmark,'_peak_process_rss_bytes',lambda:None)
    result=benchmark.run_case('cpu',3,2,1)
    assert result['process_peak_rss_bytes'] is None
    assert result['process_peak_rss_kib'] is None
    assert result['cpu_reference_max_abs_error']==0
