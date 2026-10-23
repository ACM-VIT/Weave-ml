# Ranker device evidence (#67)

This diagnostic measures the checked-in trained artifact through real compact
serving calls. It does not choose or change the production device, qualify a new
model, provision hardware, or establish recommendation relevance/calibration.

From the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/benchmark_ranker_devices.py --repeats 30
.venv/bin/python -m pytest tests/test_ranker_device_benchmark.py
```

The default matrix uses candidate counts 1, 10, 45 and 150, each in a fresh child
process on CPU and on CUDA if actually available. Optional bounds are at most
five distinct counts, 2,000 candidates, 200 repeats and eight CPU threads per case.
Each child has a 180-second outer timeout. No network, product records, external
services, artifact conversion or model-file writes are involved.

## Explicit device configuration

`V2_HEAVY_RANKER_DEVICE=auto` preserves current serving selection: CUDA when
available, CPU otherwise. `cpu` explicitly places the ranker on CPU even on a
CUDA-capable host; `cuda` requires actual CUDA availability. An unavailable
explicit CUDA request raises during artifact loading. Existing required-ranker
readiness stays closed; optional hybrid behavior follows the existing policy.
There is no silent CPU substitution for an explicit CUDA request.
The default project lock installs CPU-only PyTorch. Using CUDA also requires a
separately qualified CUDA-enabled build and actual supported hardware.

The network-free production preflight validates the setting, including rejecting
blank/unknown values, without pretending to validate remote hardware. Applications
can also pass `device="cpu"`/`"cuda"`/`"auto"` directly to `RankerService`.
Defaults, manifests, scaling, model weights and qualification remain unchanged.
Example env files document the default; actual env files are not modified.

## Measurement contract

- Check model/scaler checksums against the manifest before loading. Preserve the
  manifest digest, training-data identity, qualification method and automatic
  qualification errors in the report. Source SHA-256 identities and UTC report
  time make the tested implementation explicit. The current artifact's manual override
  does not turn its synthetic training into real-user quality evidence.
- Load the actual artifact on CPU, then explicitly place this benchmark instance
  on the selected device. Report loading and placement separately. Runtime
  `RankerService` automatic selection remains the default.
- Use deterministic, normalized synthetic 384-dimensional user/repository vectors
  and real serving feature preparation/scaling, five model heads, value scoring
  and stable sorting. These inputs exercise compute; they are not a corpus or
  relevance judgments.
- Record the first serving call separately, then three warmup calls followed by
  repeated uninstrumented whole calls. Warm p50/p95 and candidates/second include
  feature preparation, input assembly, transfers, forward pass, output conversion,
  value scoring and sorting. Loading excludes Python/Torch import time; outer
  process wall time includes imports, all repetitions, diagnostics and reference
  work, so it is not a cold-request latency.
  Checksum verification precedes the loading timer and may warm filesystem pages;
  fresh processes still share the OS file cache. No cold-disk result is claimed.
- In a separate instrumented pass, observe existing serving methods to separate
  host input assembly, device placement, forward, output-to-CPU and remaining
  work. Remaining includes feature scaling/scoring/sorting and instrumentation
  overhead. CPU placement and `.cpu()` can be no-ops, not PCIe transfers. Stage
  medians do not sum to an uninstrumented whole-call median.
- Synchronize actual CUDA work around timed regions. GPU measurements are emitted
  only on a CUDA-capable build/runtime. Report tensor peak allocated bytes for
  CUDA separately from Linux whole-process peak RSS. RSS includes imports,
  fixtures, artifact loading and instrumented calls; it is not model-only memory.
  Reference evaluation follows memory accounting.
- Compare all five clipped head predictions and final scores to the same artifact
  on CPU after measurement, with numerical tolerances (`rtol=1e-5`, `atol=1e-6`).
  Record exact ordering separately so tolerance does not hide changed rankings.
  Prediction parity is distinct from accuracy or calibration against real labels.

Timing follows PyTorch's guidance on [asynchronous CUDA execution and
synchronization](https://docs.pytorch.org/docs/stable/notes/cuda) and
[warmup/thread controls for benchmarks](https://docs.pytorch.org/docs/stable/benchmark_utils.html).
Allocator memory and process RSS have the different scopes described above.

## Local observation, 23 October 2026

Intel Core Ultra 7 255H, Linux x86_64, Python 3.12.14, Torch 2.13.0+cpu;
one Torch/OMP/OpenBLAS thread, 30 warm samples per fresh process. Other issue
workers were active; this is a development-machine observation rather than a
quiet-host capacity or production p95 guarantee.

| Candidates | First serving call ms | Warm p50 ms | Warm p95 ms | Whole-process peak RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.049 | 0.592 | 0.645 | 242.1 |
| 10 | 3.596 | 1.097 | 1.120 | 242.3 |
| 45 | 4.112 | 2.645 | 2.683 | 242.4 |
| 150 | 10.020 | 7.793 | 8.387 | 242.9 |

CPU artifact loading took 23.9–25.1 ms, separately from the first call. Every
count matched its CPU reference scores and order exactly. The artifact SHA-256
was `ab7d804261bf204e51962192c26b445e058057993ddb67ff75cf32791cca9215`;
the manifest retains `qualification_method=manual_override` and its synthetic
training provenance and qualification warnings.

CUDA was unavailable in this installed CPU build/runtime. No GPU transfer,
latency, memory or throughput result was measured or inferred. Compile, ONNX and
quantization were not evaluated; no converted artifacts were created. CPU-only
evidence does not establish that CPU beats a real GPU deployment.

## Decision still required

#67 remains open pending the requested human hardware/cost/quality decision.
Run the same matrix on approved actual target hardware with pinned software and
representative candidate/feature distributions before making a comparative
selection. Supply genuine judged relevance/calibration data and agree tolerances
before converting or quantizing an artifact; this fixed-input numerical check
does not substitute for them.

Record selected hardware/mode, supported failure/fallback behavior, thread and
batch budgets, measured cost, model/scaler/manifest identities, quality gates and
rollback conditions. Enabling a selected production device or converting a model
still requires that decision and relevant validation. This PR only prepares the
explicit option and diagnostic; production settings and artifacts are preserved.
