"""Network-free trained-artifact device measurements; no production selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from unittest.mock import patch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    # Conservatively fingerprint all tracked project Python code, including
    # setup constructors and their transitive project helpers. The script is
    # included explicitly while it is still an untracked development tool.
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', '--', '*.py', 'pyproject.toml', 'uv.lock'], cwd=ROOT,
    ).decode().split('\0')
    paths = {ROOT / name for name in names if name}
    paths.add(Path(__file__).resolve())
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def require_sources(expected):
    if source_hashes() != expected:
        raise RuntimeError('project source changed during measurement; discard this comparison')
    return expected


_IMPORTED_SOURCE_SHA256 = source_hashes()

import numpy as np
import torch

from inference.feature_spec import EMBEDDING_DIM
from inference.ranker_service import RankerService

ARTIFACTS = ROOT / "inference"


def validate_artifacts(directory: Path) -> dict:
    manifest_bytes = (directory / "model_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    for name, field in (("heavy_ranker.pt", "model_sha256"), ("feature_scaler.json", "scaler_sha256")):
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if manifest.get(field) != actual:
            raise ValueError(f"Unverified benchmark artifact: {name}")
    return {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "model_sha256": manifest["model_sha256"],
            "scaler_sha256": manifest["scaler_sha256"],
            "model_version": manifest.get("model_version"),
            "qualification_method": manifest.get("qualification_method"),
            "training_data": manifest.get("training_data")}


def validate_options(count: int, repeats: int, threads: int) -> None:
    if any(type(value) is not int for value in (count, repeats, threads)):
        raise ValueError("Benchmark bounds must be integers")
    if not 1 <= count <= 2000 or not 1 <= repeats <= 200 or not 1 <= threads <= 8:
        raise ValueError("Bounds: candidates1..2000, repeats1..200, threads1..8")


def fixtures(count: int):
    rng = np.random.default_rng(67)
    vectors = rng.normal(size=(count + 1, EMBEDDING_DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    repos = [{"id": f"device-fixture-{i:04d}", "embedding": vector,
              "doc_quality": (i % 10) / 10, "code_health": .7,
              "star_count": i * 11, "readme_length": 500 + i,
              "topics": ["python"] if i % 2 else ["rust"],
              "activity_score": .6, "trend_velocity": .2}
             for i, vector in enumerate(vectors[1:])]
    return vectors[0], ["python"], repos


def summary(samples: list[float]) -> dict:
    values = np.asarray(samples, dtype=np.float64)
    if not len(values) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Invalid timing samples")
    return {"samples": len(samples), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95))}


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _timed(call, device: str):
    _sync(device)
    start = time.perf_counter()
    result = call()
    _sync(device)
    return result, (time.perf_counter() - start) * 1000


def profile_stages(service, user, skills, candidates, repeats, device):
    """Observe real serving methods; separate from uninstrumented headline time."""
    totals = {key: [] for key in ("host_input", "input_placement", "forward", "output_cpu", "remaining")}
    original_input = service._input_tensor
    original_cpu = torch.Tensor.cpu
    phases = {}

    def inputs(*args):
        saved = service.device
        service.device = torch.device("cpu")
        try:
            host, phases["host_input"] = _timed(lambda: original_input(*args), "cpu")
        finally:
            service.device = saved
        placed, phases["input_placement"] = _timed(lambda: host.to(saved), device)
        return placed

    def transfer(tensor, *args, **kwargs):
        result, duration = _timed(lambda: original_cpu(tensor, *args, **kwargs), device)
        phases["output_cpu"] = phases.get("output_cpu", 0) + duration
        return result

    def before(*_):
        _sync(device)
        phases["forward_start"] = time.perf_counter()

    def after(*_):
        _sync(device)
        phases["forward"] = (time.perf_counter() - phases["forward_start"]) * 1000

    hooks = [service.model.register_forward_pre_hook(before), service.model.register_forward_hook(after)]
    try:
        with patch.object(service, "_input_tensor", inputs), patch.object(torch.Tensor, "cpu", transfer):
            for _ in range(repeats):
                phases.clear()
                _, total = _timed(lambda: service.score_batch_compact(user, skills, candidates), device)
                measured = sum(phases[key] for key in totals if key != "remaining")
                phases["remaining"] = max(0.0, total - measured)
                for key in totals:
                    totals[key].append(phases[key])
    finally:
        for hook in hooks:
            hook.remove()
    return {key: summary(values) for key, values in totals.items()}


def compare_timed_compact(samples, reference):
    """Compare every timed compact output by ID, not a later diagnostic call."""
    reference_ids = tuple(row['repo_id'] for row in reference)
    if not samples or not reference_ids or len(set(reference_ids)) != len(reference_ids):
        raise ValueError('Invalid compact reference/sample identities')
    expected = {row['repo_id']: float(row['final_score']) for row in reference}
    right = np.asarray([expected[key] for key in reference_ids], dtype=np.float64)
    if not np.all(np.isfinite(right)):
        raise ValueError('Non-finite compact reference')
    maximum = 0.0
    order_equal = True
    for sample in samples:
        ids = tuple(key for key, _ in sample)
        if len(ids) != len(reference_ids) or len(set(ids)) != len(ids) or set(ids) != set(reference_ids):
            raise ValueError('Timed compact output changed candidate identities')
        actual = dict(sample)
        left = np.asarray([actual[key] for key in reference_ids], dtype=np.float64)
        if not np.all(np.isfinite(left)):
            raise ValueError('Non-finite timed compact output')
        np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)
        maximum = max(maximum, float(np.max(np.abs(left-right))))
        order_equal = order_equal and ids == reference_ids
    return maximum, order_equal


def run_case(device: str, count: int, repeats: int, threads: int, directory: Path = ARTIFACTS) -> dict:
    sources = require_sources(_IMPORTED_SOURCE_SHA256)
    previous_threads = torch.get_num_threads()
    try:
        result = _run_case(device, count, repeats, threads, directory)
        require_sources(sources)
        return {**result, 'source_sha256': dict(sources)}
    finally:
        torch.set_num_threads(previous_threads)


def _run_case(device: str, count: int, repeats: int, threads: int, directory: Path) -> dict:
    validate_options(count, repeats, threads)
    if device not in {"cpu", "cuda"}:
        raise ValueError("Supported benchmark devices: cpu/cuda")
    if device == "cuda" and not torch.cuda.is_available():
        return {"device": device, "candidates": count, "status": "unavailable", "reason": "CUDA unavailable in this PyTorch build/runtime"}
    provenance = validate_artifacts(directory)
    torch.set_num_threads(threads)
    start = time.perf_counter()
    # Explicit benchmark placement, without changing the serving constructor or
    # auto-selecting a GPU just to load a CPU reference. Runs in an isolated child.
    service = RankerService(model_path=str(directory / "heavy_ranker.pt"),
                            scaler_path=str(directory / "feature_scaler.json"),
                            manifest_path=str(directory / "model_manifest.json"),
                            require_production_manifest=True, device="cpu")
    if not service.ready:
        raise RuntimeError("Trained model did not load")
    load_ms = (time.perf_counter() - start) * 1000
    service.device = torch.device(device)
    _, placement_ms = _timed(lambda: service.model.to(service.device), device)
    user, skills, candidates = fixtures(count)
    invoke = lambda: service.score_batch_compact(user, skills, candidates)
    _, first_ms = _timed(invoke, device)
    for _ in range(3):
        invoke()
    _sync(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    samples = []
    timed_outputs = []
    for _ in range(repeats):
        actual, duration = _timed(invoke, device)
        samples.append(duration)
        timed_outputs.append(tuple((row['repo_id'], float(row['final_score'])) for row in actual))
    stages = profile_stages(service, user, skills, candidates, repeats, device)
    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    cuda_peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
    # Reference/parity work follows measurements and memory accounting.
    diagnostic = service.score_batch(user, skills, candidates)
    service.device = torch.device("cpu")
    service.model.to(service.device)
    reference = service.score_batch(user, skills, candidates)
    if len(actual) != count or len(diagnostic) != count or len(reference) != count:
        raise ValueError("Benchmark did not score every candidate")
    expected_by_id = {row["repo_id"]: row for row in reference}
    max_error = 0.0
    for row in diagnostic:
        expected = expected_by_id[row["repo_id"]]
        left = np.array(list(row["predictions"].values()) + [row["final_score"]])
        right = np.array(list(expected["predictions"].values()) + [expected["final_score"]])
        np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)
        max_error = max(max_error, float(np.max(np.abs(left - right))))
    compact_error, compact_order_equal = compare_timed_compact(timed_outputs, reference)
    max_error = max(max_error, compact_error)
    times = summary(samples)
    return {"device": device, "candidates": count, "status": "measured",
            "device_name": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
            "torch_cuda_version": torch.version.cuda,
            "artifact": provenance, "automatic_qualification_errors": list(service.qualification_errors),
            "artifact_load_cpu_ms": load_ms, "model_placement_ms": placement_ms,
            "first_serving_call_ms": first_ms, "warm_serving": times,
            "candidates_per_second": count * repeats * 1000 / sum(samples),
            "instrumented_stages": stages, "process_peak_rss_kib": rss_kib,
            "cuda_peak_allocated_bytes": cuda_peak,
            "cuda_peak_allocated_bytes_scope": "post_warmup_headline_and_instrumented_calls",
            "timed_compact_max_abs_error": compact_error,
            "cpu_reference_max_abs_error": max_error,
            "cpu_reference_order_equal": compact_order_equal}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 10, 45, 150])
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args(argv)
    if len(args.counts) > 5 or len(set(args.counts)) != len(args.counts):
        parser.error("Use at most five distinct candidate sizes")
    for count in args.counts:
        validate_options(count, args.repeats, args.threads)
    if args.device:
        if len(args.counts) != 1:
            parser.error("Child device run requires one candidate size")
        print(json.dumps(run_case(args.device, args.counts[0], args.repeats, args.threads, args.artifact_dir)))
        return 0
    sources = require_sources(_IMPORTED_SOURCE_SHA256)
    results = []
    for device in ("cpu", "cuda"):
        for count in args.counts:
            command = [sys.executable, str(Path(__file__).resolve()), "--device", device,
                       "--counts", str(count), "--repeats", str(args.repeats),
                       "--threads", str(args.threads), "--artifact-dir", str(args.artifact_dir)]
            start = time.perf_counter()
            child = subprocess.run(command, capture_output=True, text=True, timeout=180, check=True)
            row = json.loads(child.stdout)
            require_sources(sources)
            if row.get('source_sha256') != sources:
                raise RuntimeError('child measured different project sources; discard this matrix')
            row["process_wall_ms"] = (time.perf_counter() - start) * 1000
            results.append(row)
    require_sources(sources)
    print(json.dumps({"schema": "ranker-device-diagnostic-v1", "python": platform.python_version(),
                      "recorded_at": datetime.now(timezone.utc).isoformat(), "source_sha256": sources,
                      "torch": torch.__version__, "platform": platform.platform(),
                      "processor": platform.processor(), "threads": args.threads,
                      "input_kind": "fixed synthetic embeddings and features; no relevance judgments",
                      "unmeasured_modes": ["compile", "ONNX", "quantization"],
                      "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
