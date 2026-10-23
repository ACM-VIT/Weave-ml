# Process-local GitHub quota acceptance

GitHub acquisition already shares quota state by an opaque credential/endpoint
scope and returns retryable README work to the bounded scheduler. This package
adds direct concurrent/reset/cancellation acceptance and a reproducible injected
workload. It does not replace the coordinator, change retries, increase worker
counts or access live GitHub.

```bash
uv run pytest tests/test_github_quota_acceptance.py -q
uv run python -m scripts.benchmark_github_quota > /tmp/github-quota.json
```

The real `GitHubGraphQLClient.execute`, shared quota coordinator, logical request
budget and `ReadmeScheduler` run against a fake HTTP session and fake monotonic /
wall clocks. A small fixture caller performs metadata and blob stages through
`execute`. The first blob request returns 429 with five seconds of Retry-After
and an exhausted-primary reset. Successful metadata must replay inside the same
logical operation when the blob retry is dispatched; it must not be fetched again.
The script never creates a real requests transport or uses an actual token.

The driver advances virtual time and wakes the real scheduler dispatcher. Only
the driver yields wall-clock time. A worker calling the client sleep path fails
the fixture immediately, so a claimed zero sleep count cannot hide long sleepers.
The normal scheduler owns job admission, pending retries and worker sessions.
The script reports virtual throughput separately from wall execution time.

## Deterministic checks

- Eight simultaneous callers are deferred throughout a reset window. Once that
  window opens, exactly one caller reserves each staggered admission turn.
- The actual client/scheduler fixture completes all jobs, preserves one metadata
  fetch per job, frees transport workers while retries remain queued and keeps
  admitted jobs within the existing `2 * workers` ceiling.
- Cancelling a deferred job cannot start another request. A retry that cannot fit
  the run deadline fails before waiting; expiry while queued removes the work.
  Running IO ownership remains covered by the existing scheduler tests.
- Permanent HTTP404 is not retried and retains HTTP error classification. Repeated
  429s exhaust exactly the original two-retry/three-attempt blob budget, even
  across scheduler deferrals; the successful metadata stage still executes once.
- A transport success arriving after expiry is rejected before parsing, caching
  or returning it. Changing authorization cannot replay an earlier successful
  response, and scope keys/error messages do not contain the supplied token.

## Injected workload result, 23 October 2026

One run used eight logical jobs and two transport workers. These are synthetic
clock/session results, not real API latency or quota measurements.

| Measurement | Observed |
| --- | ---: |
| Completed logical jobs | 8 / 8 |
| HTTP fixture requests / injected429s | 17 / 1 |
| Maximum metadata attempts per logical job | 1 |
| Additional blob retry attempts | 1 |
| Shared coordinator reserved requests | 17 |
| Known remaining quota after local reservations | 84 |
| Admission deferrals | 27 |
| Maximum admitted / queued jobs | 2 / 2 |
| Maximum active transport workers | 2 |
| Queued retries observed with zero active workers | Yes |
| Transport-worker sleep calls | 0 |
| Minimum post-reset request spacing | 100 ms |
| Virtual elapsed / virtual useful throughput | 6.55 s / 1.22 jobs per virtual second |
| Wall duration of the accelerated fixture | 0.158 s |
| Settled admitted / queued / active work | 0 / 0 / 0 |
| All worker sessions closed | Yes |

The first successful fixture response establishes a remaining value of100; 16
subsequent local reservations leave84. The absolute fixture request count is17.
These are separate measurements, not a claim that every real GraphQL query has
one unit of cost. The production coordinator also accepts observed GraphQL cost.
Queue/deferral counts and virtual completion time can vary slightly with thread
interleaving; the invariant checks do not require one exact scheduler ordering.

There is no live throughput or speedup claim. The fixture does not validate
GitHub's query schema, remote rate-cost behavior or a production ingestion run.
It verifies client retry/admission and scheduler ownership around injected
responses. Existing canonical README, partial GraphQL and summary completion
reserve behavior remain unchanged and retain their independent tests.

The coordinator is deliberately process-local. Different processes, other GitHub
clients and tokens sharing an account can consume the same real quota. This
report does not establish distributed quota coordination or authorize deployment,
new credentials or a live benchmark.
