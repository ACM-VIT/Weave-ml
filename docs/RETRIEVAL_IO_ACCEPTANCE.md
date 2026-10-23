# Shared retrieval I/O and cooperative budget acceptance

The active retriever already overlaps profile/discovery work through a
process-wide bounded FIFO executor, checks cooperative local deadlines and drains
actual child work before returning outer admission. These tests and measurements
complete direct acceptance coverage for issues #54 and #55. They do not add a
second executor, alter ranking or deadlines, or release slots on caller expiry.

## Reproduce

```bash
uv run pytest tests/test_retrieval_io_acceptance.py -q
uv run python -m scripts.benchmark_retrieval_io > /tmp/retrieval-io.json
```

The benchmark uses the actual `QdrantV2Retriever`, canonical UUID/eligible payload
validation, candidate fusion/feed shaping and shared I/O executor. It supplies
four deterministic repository fixtures and a compatible user vector, with an
injected 10 ms dependency delay. No Qdrant/Redis/PostgreSQL service, credentials,
encoder, heavy-ranker artifact or product records are accessed. Optional hybrid
ranking is explicitly selected only for this fixture; production required-heavy
behavior is unchanged and is covered by existing retrieval tests.

The default workload has 24 requests from four concurrent callers, six dependency
calls per request and a five-second local budget. The one-worker control and
four-worker fan-out use the same current code and must return identical ordered
repository IDs, scores and sources per seed. This is a worker-budget control,
not a historical release or deployed before/after comparison. A third workload
injects a timeout in every trend channel while retaining other real retriever
results. JSON contains per-stage calls, queue/service time, errors, peak/running/
outstanding work and caller durations; timeout outcomes remain explicit.

## Controlled local result, 23 October 2026

| Measurement | One-worker control | Shared four-worker fan-out | Trend channel times out |
| --- | ---: | ---: | ---: |
| Request p50 / p95 / p99 | 244.08 / 271.82 / 272.19 ms | 63.38 / 68.47 / 71.32 ms | 63.92 / 69.30 / 71.96 ms |
| Entire workload | 1.515 s | 0.385 s | 0.389 s |
| Dependency calls | 144 | 144 | 144 |
| Peak actual dependency calls | 1 | 4 | 4 |
| Injected dependency timeouts | 0 | 0 | 24 |
| Failed requests | 0 | 0 | 0 |
| Dependency calls after budget stopped | 0 | 0 | 0 |

Mean queue/service time per actual channel call:

| Channel | Control queue / service | Fan-out queue / service |
| --- | ---: | ---: |
| Profile | 84.22 / 10.26 ms | 14.12 / 10.29 ms |
| Semantic | 99.99 / 10.42 ms | 16.43 / 10.41 ms |
| Trend | 92.41 / 10.52 ms | 15.81 / 10.56 ms |
| Activity | 97.45 / 10.48 ms | 16.20 / 10.50 ms |
| Popularity | 102.41 / 10.47 ms | 16.77 / 10.48 ms |
| Freshness | 101.37 / 10.46 ms | 17.16 / 10.48 ms |

This shows overlap and its load cost in the injected workload. It does not prove
a real Qdrant latency/throughput improvement, hardware capacity or recommendation
quality. The unit test separately holds actual RPC entry barriers across four
recommendations: two running calls and one queued call share a capacity of three;
additional submitters wait, and all 24 calls settle without exceeding two running.

## Cancellation and actual worker exit

The script also invokes the actual `run_read_only_request` disconnect watcher and
recommendation executor against a fake request and blocked dependency. It is a
request-budget lifecycle fixture, not a browser disconnect or HTTP wire test.
After the caller returns, the fixture deliberately holds the already running
calls for another 50 ms. It measures that tail rather than pretending thread
cancellation interrupts dependency work.

| Result | Disconnect | 1.1-second budget expiry |
| --- | ---: | ---: |
| Caller duration | 53.39 ms | 1,101.82 ms |
| Caller return to observed worker settlement | 52.20 ms | 52.08 ms |
| Already running calls when caller returned | 2 | 2 |
| New RPCs after caller return / budget stop | 0 / 0 | 0 / 0 |
| Outer admission retained until actual exit | Yes | Yes |
| A new request rejected while worker remained live | Yes | Yes |
| Outstanding/running children after settlement | 0 / 0 | 0 / 0 |

The existing native-future completion metrics also recorded approximately
51.71 ms from cancellation to worker exit and 52.87 ms from timeout to worker
exit. Status polling adds a small observation delay. These values follow the
intentional injected tail and are not production cancellation bounds.

Fake-clock regressions additionally expire work while waiting for aggregate
admission and after admission in the executor queue: neither invokes its RPC.
Expiry between retrieval channels prevents new calls and floors Qdrant's integer
timeout to the remaining budget. A search whose encoder finishes after expiry
never starts Qdrant. Relative header durations can reduce a local bound but never
extend it; absolute monotonic timestamps are not transmitted. Existing service,
search/query-cache and encoder tests retain their independent running-work and
summary completion-reserve guarantees.
