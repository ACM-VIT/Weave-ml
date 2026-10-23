# Shared encoder fairness

The shared model already uses FIFO admission and repository encoding releases its
slot between groups of at most eight texts / 2,048 padded tokens. This work adds
acceptance coverage and a reproducible report, without changing that scheduler,
model concurrency, chunking policy, vectors, deadlines or production configuration.

`tests/test_encoder_fairness.py` controls actual encoder-entry barriers rather
than assuming thread timing. Three continuing streams (25 README chunks, three
unique searches, three profile texts) take FIFO turns through the real grouping
and shared model wrapper. It verifies complete indexing, no skipped chunks and
one actual encode at a time. Fake-clock expiry and queued cancellation remove
work without calling the encoder. Cancelling a running caller's await retains the
model slot until its actual encode returns; the next caller remains queued.
Existing service-runtime tests separately cover HTTP executor/admission retention.

## Reproduce locally

Use Python 3.12 with the locked development dependencies. The pinned MiniLM model
must already exist in the Hugging Face cache; the script sets offline flags and
passes `local_files_only=True`. It does not download a model, start services or
read/write product data. It uses one CPU thread and synthetic software README,
query and profile text. Redirect JSON outside the source tree if retaining it.

```bash
uv run pytest tests/test_encoder_fairness.py -q
uv run python -m scripts.benchmark_encoder_fairness > /tmp/encoder-fairness.json
uv run python -m scripts.benchmark_encoder_fairness \
  --transport http --only-grouped --repository-jobs 1 > /tmp/encoder-http.json
```

The default comparison has two 366,608-character README jobs, 169 current character
chunks each, two client streams of 64 unique queries and 32 profile texts. The
ungrouped control deliberately calls the same real encoder with the full chunk
list under one permit. It is a counterfactual control, not a historical release
or deployed before/after comparison. Both modes retain the same FIFO model gate.

The `http` mode exercises the actual `api.main` application in process, including
authentication, request/response validation, the bounded search executor, query
cache and encoder. Qdrant returns an explicitly empty fixture response. Startup
lifespan, live dependency access and repository writes stay off. It retains the
normal search settings (one worker, four outstanding jobs, four-second timeout)
in the isolated process. Successful and 503 response durations are reported
separately; failures cannot be counted as fast successful searches. It waits for
actual search workers to drain before retiring the fixture.

## Measured encoder comparison, 23 October 2026

One local run used the cached pinned MiniLM revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, CPU, one PyTorch thread, Python 3.12.14.
These are encoder/job durations, excluding HTTP, Qdrant, providers and storage.
The JSON records the exact source hashes, configuration and sample counts.

| Measurement | Ungrouped control | Current grouped |
| --- | ---: | ---: |
| Entire fixture | 47.821 s | 48.305 s |
| Completed README jobs | 2 | 2 |
| README encoder calls / largest group | 2 / 169 texts | 44 / 8 texts |
| Maximum actual encoder concurrency | 1 | 1 |
| Queries starting during README work | 0 | 42 |
| Profile encodes starting during README work | 0 | 21 |
| Query embedding duration p95 / p99 | 127.89 / 31,262.34 ms | 2,229.90 / 2,433.89 ms |
| Maximum query embedding duration | 42,773.78 ms | 2,499.54 ms |
| Query encoder wait p95 / p99 | 93.30 / 31,207.61 ms | 2,198.16 / 2,403.60 ms |
| Query encoder service p95 / p99 | 59.37 / 61.68 ms | 58.59 / 62.81 ms |
| Profile encoder wait p95 / p99 | 101.18 / 29,513.23 ms | 2,220.34 / 2,399.77 ms |
| Profile encoder service p95 / p99 | 38.43 / 38.83 ms | 49.93 / 63.87 ms |
| README group service p95 / p99 | 21,468.75 / 21,481.30 ms | 1,171.13 / 1,203.26 ms |

Grouping permits interactive work to progress during indexing and reduces the
largest stalls in this fixture. The p95 is worse: grouping sustains contention,
whereas the control blocks a few queries for a long time before the remaining
queries run quickly. Total duration is slightly higher. This tradeoff must remain
visible; no blanket latency improvement follows from the p99 result.

Hardware, stream counts, document lengths and arrival order affect these values.
They are not a production latency bound, live Qdrant test or retrieval-quality
qualification. Default v2 character chunking and the independent total-job-budget
policy remain unchanged.

## Integrated search fixture, 23 October 2026

The HTTP command above was also executed with one README job, 128 search
requests from two client streams and 32 profile encodes. It used the same real
cached model, 169 README chunks and default bounded search settings. All 128
responses returned HTTP 200; there were no capacity or deadline failures. This
ASGI fixture replaces Qdrant with an empty result and does not run startup.

| Measurement | Current grouped, integrated ASGI fixture |
| --- | ---: |
| Entire fixture / completed README jobs | 27.927 s / 1 |
| Successful HTTP search p95 / p99 / maximum | 2,127.48 / 2,198.09 / 2,224.75 ms |
| Search encoder wait p95 / p99 | 1,010.81 / 1,077.25 ms |
| Search encoder service p95 / p99 | 68.53 / 69.60 ms |
| Profile encoder wait p95 / p99 | 1,098.46 / 1,119.66 ms |
| Profile encoder service p95 / p99 | 69.93 / 73.05 ms |
| README group wait p95 / p99 | 106.38 / 107.00 ms |
| README group service p95 / p99 | 1,049.82 / 1,080.27 ms |
| Query / profile calls starting during README work | 21 / 21 |
| README groups / maximum group size | 22 / 8 texts |
| Maximum actual encoder concurrency | 1 |

The API fixture and the two-README encoder comparison are different workloads;
their percentiles must not be presented as a before/after pair. No timeout,
concurrency or admission limit was relaxed to obtain the HTTP result. Both are
local acceptance evidence, not a prediction for another arrival pattern or
hardware, a production capacity result or a real Qdrant latency test.
