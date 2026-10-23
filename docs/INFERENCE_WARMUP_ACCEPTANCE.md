# Inference warmup acceptance

The existing startup implementation already executes two encoder texts and one
778-value ranker input, validates dimensions/finite output, and retains one warmup
state per loaded model. This slice adds direct acceptance of that behavior.
Mandatory artifact/provenance and network-free startup remain intact; no runtime
algorithm or production configuration is changed.

Tests execute actual encode/forward methods, including a real PyTorch forward in
inference mode, concurrent/repeated warmup calls, invalid dimensions/outputs and
sticky failure. The actual integrated API lifespan and registered readiness route
are tested with bounded dependency/model stand-ins: required failure cannot open
readiness; repeated probes do not repeat inference; timed-out startup retains its
resources until the actual encoder or ranker exits and cannot become ready late. Production
cannot disable mandatory warmup. These unit stand-ins do not qualify a real model.

The separate profiler uses the actual cached pinned MiniLM encoder and checked-in
ranker with manifest/model/scaler hashes, shared with the [device diagnostic](RANKER_DEVICE_EVALUATION.md) from
#67. Two fresh sequential CPU children compare a loaded-model-only counterfactual
with actual warmup. It measures model/pipeline startup work, first encode/ranking
and steady inference, verifies once-only warmup and warm/control numerical parity,
and reports process memory. It blocks socket connections and requires local-only
loading; missing artifacts fail rather than download. No user/corpus data or
external API is read or written. The ranker manifest records synthetic training
and an existing manual override; these checks confer no new model qualification. CPU placement is explicit in this diagnostic;
it makes no hardware-selection or GPU-performance claim.

```bash
MIN_ELIGIBLE_REPOSITORIES=0 uv run pytest tests/test_inference_warmup_acceptance.py
uv run python -m scripts.profile_inference_warmup --candidates 150 --repeats 10 --threads 1
```

The load-only control is not a permitted deployed production mode. Timings exclude
network services and full HTTP request processing. Candidate vectors/features and
query text are fixed synthetic inputs, not relevance judgments. First/steady
component inference and startup overhead must be reported separately; warmup may
move work into startup without establishing a universal speedup or first-request
SLO. The first measured pair below used stable source `bd6c6c2`, before the later
explicit device option; the loader's scoped legacy CPU placement was verified.
The warmup implementation is retained, rather than rewritten from these results.

## Recorded CPU pair, 23 October 2026

Both fresh sequential children used one CPU thread, the cached pinned encoder and
150 synthetic ranking candidates. Each steady series has ten samples. Encoding
precedes ranking, then steady calls alternate. These observations overlapped an
unrelated immutable-container build; filesystem cache, process order and host
contention are not controlled production conditions.

| Measurement | Loaded-model-only control | Existing warmup |
| --- | ---: | ---: |
| Model/pipeline setup | 6,070.35 ms | 3,427.73 ms |
| Explicit warmup work | 0 ms | 76.16 ms |
| Startup work including imports/setup | 7,638.54 ms | 5,082.87 ms |
| First encode | 75.18 ms | 28.20 ms |
| Steady encode p50 / p95 | 13.74 / 17.33 ms | 14.37 / 23.35 ms |
| First 150-candidate ranking | 13.70 ms | 19.55 ms |
| Steady ranking p50 / p95 | 9.54 / 11.22 ms | 10.26 / 14.26 ms |
| Warmup encode / forward calls | 0 / 0 | 1 / 1 |
| Peak process RSS | 510.09 MiB | 514.12 MiB |

Warm/control output differences were zero for both encoder vectors and ranker
scores. Repeated warmup/status inspections performed no additional inference.
The shorter setup/startup observation in the warm child does not isolate warmup:
process order, OS caches and concurrent host work all vary. It is not evidence
that extra inference reduces model load cost. The slower first
ranking and warmed steady tails remain visible. No universal first-request or
steady-state improvement follows from this single pair.

Recorded component source SHA256:

| Source | SHA256 |
| --- | --- |
| `embedding/runtime.py` | `e81433a84281c14363a06bb2898744e3a34104dcc96e9e6593a081842d66d16e` |
| `inference/ranker_service.py` | `ac5e1167f0854053964bb6549805e471a6750f610e33bada8b389965ee7778c6` |
| `api/main.py` | `1dd8a7043dddfc454d7288e308f83a56ed8a554898faa15f2c64f2738e8104db` |

Seventeen direct tests pass for the startup/failure/readiness/once-only boundaries.
They preserve mandatory warmup in production and actual worker-exit cleanup.
Remote CI, production startup/traffic, GPU behavior and recommendation quality
remain separate from these local checks and diagnostic samples.


## Final device-option checkpoint

The same bounded pair was repeated on `bff3c5f` with the explicit CPU constructor
and a guard that rejects even swallowed socket attempts. Both measured children ran in one source-exclusive window; component hashes
remain in the JSON. The published driver additionally rejects a source change
between children, verified with controlled driver fault injection.
Both encoder and ranker output differences were again zero. These are another
single ordered pair, with the same ten-sample steady series and 150 candidates.

| Measurement | Loaded-model-only control | Existing warmup |
| --- | ---: | ---: |
| Model/pipeline setup | 5,087.35 ms | 3,521.27 ms |
| Explicit warmup work | 0.00 ms | 31.72 ms |
| Startup work including imports/setup | 6,667.50 ms | 5,180.44 ms |
| First encode | 49.90 ms | 14.98 ms |
| First 150-candidate ranking | 17.72 ms | 9.46 ms |
| Steady encode p50 / p95 | 15.60 / 24.83 ms | 13.11 / 14.20 ms |
| Steady ranking p50 / p95 | 10.24 / 16.25 ms | 8.79 / 9.26 ms |
| Peak process RSS | 510.62 MiB | 511.70 MiB |
| Warmup encode / forward calls | 0 / 0 | 1 / 1 |
| Attempted socket connections | 0 | 0 |

The final ranker source SHA256 is
`5efc0d29ecba6826f8865f78df544fff6a18a1a5c6658dc5f248a81dfa99a191`;
encoder runtime and API lifespan hashes match the table above. The earlier slower
warm ranking observation remains part of the evidence. Changed setup/startup
numbers across ordered pairs cannot establish a causal production improvement.
No runtime warmup algorithm or model/artifact was changed to obtain these results.

This slice depends on [PR #143](https://github.com/ACM-VIT/Weave-ml/pull/143)'s
shared artifact/fixture helpers and device option.
Seventeen direct cases pass at the final checkpoint, including timeout during
both encoder and ranker native work. Source inspection and these local cases do
not replace deployed traffic, real relevance judgments or remote CI.
