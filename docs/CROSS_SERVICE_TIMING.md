# Cross-service stage timing (Weave #134)

The existing authenticated `/api/v2/metrics` exports cumulative
`ml_stage_duration_seconds` histograms. Existing metrics are preserved. Each
process uses a fixed operation/stage/outcome label space: at most 500 families,
or 7,500 bucket/count/sum series. No user, repository, query, request, job or
trace identifiers appear in labels.

Finite bucket bounds in seconds are .005, .025, .1, .5, 1, 4, 5, 8, 12, 30, 45,
120. The +Inf bucket equals count. The 4/5/8/12/45 boundaries match default
search/health/feedback/recommendation/refresh budgets. Outcomes are `ok`,
`rejected`, `error`, `timeout`, `cancelled`; compare them independently.

| Stage | Interval |
| --- | --- |
| http | Post-auth middleware entry through response object return, excluding body delivery |
| span | Existing internal worker scope |
| queue | In-process submission through physical worker start, or rejected/cancelled admission |
| service | Physical worker start through exit, including failures |
| late_work | Actual worker execution after caller abandonment or shared health expiry |
| model_lock | FIFO shared encoder admission wait |
| encoding | Model encode call while holding encoder admission |
| qdrant | Recommendation retrieval or semantic-search dependency call |
| ranking | Recommendation ranker batch call |
| response_build | Recommendation dictionary/dataclass construction |

Nested stages overlap service time. `response_build` excludes FastAPI response
validation, JSON serialization and socket delivery. Queue wait is distinct from
backend durable outbox age. Cancellation and timeout never release admission or
finish service observations while native work still runs.

Only the authenticated internal HTTP boundary adopts an exact v00 traceparent
with nonzero trace/span IDs. Server spans and copied executor children get fresh
span IDs; request IDs remain independent. New roots sample 1/16 and children
inherit sampling. Histograms include unsampled work. The existing protected deep
health diagnostic exposes a lossy ring of at most 256 sampled records, 512 encoded
bytes each and 128 KiB total. Records contain only validated opaque IDs, fixed
labels, duration, wall-clock end and optional remaining local deadline at entry.
Telemetry reads the remaining budget without expiring/cancelling it. No request
contents or exception text is retained. ML durable feedback stream lineage is
not propagated by this change.

For fleet collection, retain per-process identity and restart metadata outside
labels. Collect twice, subtract cumulative buckets/count/sum within the same
process, discard reset intervals, then sum matching deltas across processes.
Estimate percentiles from the combined buckets. Never average local percentiles
or add repeated diagnostic snapshots. Missing spans and ring eviction mean a
trace may be incomplete. For complete local traces, residual attribution uses
the union of direct-child intervals, not the sum of overlapping nested durations.
End timestamps use wall time and can skew; durations use local monotonic time.

Reproduce with locked dependencies from the repository root:

```sh
uv sync --locked --group dev
.venv/bin/python -m pytest tests/ -m 'not integration and not benchmark' -q
.venv/bin/python -m scripts.benchmark_tracing --iterations 20000 --scrapes 100
```

The microbenchmark uses isolated child processes and limits native thread pools
to one. It reports arithmetic-only baseline, unsampled/sampled child scopes,
full-domain metrics render cost, CPU time, output bytes and Linux peak RSS.
It measures local instrumentation and rendering, excluding HTTP/network scrape
cost, model serving throughput and production latency. Run it on an otherwise
idle host. The paired Weave `docs/observability/cross-service-timing.md` describes
the Node-to-FastAPI loopback contract fixture and app visibility correlation.
Live rollout/load collection remains Weave #135 work.
