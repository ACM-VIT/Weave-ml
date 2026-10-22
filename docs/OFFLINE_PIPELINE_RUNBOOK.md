# Offline Pipeline Operations Runbook

This runbook covers the two Person 1 worker processes:

1. repository corpus acquisition, enrichment, and backend v2 delivery;
2. enriched GitHub Trending snapshot delivery to backend v2.

Both are offline jobs. Do not invoke either command from an API request handler.
For the Redis/Qdrant online services, deployment, feedback replay, alerting, and
rollout procedures, use the [V2 production operations runbook](PRODUCTION_RUNBOOK.md).

## Safety contract

- The backend is the production source of truth and creates canonical repository UUIDs.
- Workers write neither PostgreSQL nor Qdrant directly.
- Backend repository mappings are validated before a source is counted as delivered.
- Trending snapshots are complete and atomically activated by the backend.
- Repository and user embeddings must use the model and dimension published by the embedding owner.
- Checkpoints contain repository identities and bounded error messages, not tokens or README content.

## Configuration

Copy `.env.example` to `.env` and replace every placeholder used by the worker.

Required for production corpus acquisition:

- `GITHUB_TOKEN`
- `BACKEND_URL`
- `INTERNAL_API_SECRET`
- `CORPUS_TARGET_COUNT` (default `50000`)
- `ACQUISITION_MAX_CYCLES` (default `1`)
- `ACQUISITION_CHECKPOINT_PATH`

`OPENROUTER_API_KEY` is optional. Without it, README Markdown restructuring is skipped while normal README processing continues.

Trending uses the same backend credentials and enriches scraped repositories before publishing a snapshot.

## Network-free validation

These commands validate parsing and required configuration without contacting GitHub, Postgres, Qdrant, or OpenRouter:

```bash
uv run python main.py --validate-config
uv run python trending_service.py --validate-config
```

Validation success does not prove that credentials are accepted or that a remote service is healthy. It proves the local worker configuration is internally valid.

## Corpus acquisition

Run one bounded production cycle (`ingestion_engine.py` remains a compatibility
alias for this same command):

```bash
uv run python main.py \
  --limit 150 \
  --batch-size 15 \
  --workers 4 \
  --min-readme-chars 200 \
  --max-cycles 1
```

The target defaults to `CORPUS_TARGET_COUNT`. `--limit` bounds one discovery cycle; it does not request the full corpus target at once.

Embedding and Qdrant indexing are scheduled by the backend outbox after successful repository delivery.

### Stop and resume

- Stop the foreground process with `Ctrl+C` or the platform's normal `SIGTERM` shutdown.
- Do not delete the checkpoint after an interrupted or degraded run.
- Restart with the same configuration and checkpoint path.
- Pending backend delivery is re-enriched from its repository identity before retry.

The default checkpoint is `.cache/acquisition_checkpoint.json`. Writes use atomic replacement. Do not edit the file while a worker is running.

### Corpus exit codes

- `0`: the bounded invocation completed, including a no-work target-reached run.
- `1`: configuration, startup, backend verification, or pipeline execution failed.
- `2`: failures were recorded and the run made no backend-delivery progress.

Inspect the final `Corpus run report` log and the checkpoint's `last_run`,
`failures`, and `pending_persistence` fields before retrying an exit code `2`.
`pending_index` is retained only for checkpoint backward compatibility; the
production worker does not write Qdrant directly.

## Trending refresh

Run one forced refresh and publish an atomic snapshot to the backend:

```bash
uv run python trending_service.py --once
```

Run the long-lived scheduler:

```bash
uv run python trending_service.py --scheduled
```

The scheduler handles `SIGINT` and `SIGTERM`. A normal stop clears its in-process schedule and exits the loop.

The production path is `trending worker → backend snapshot API → backend outbox → ML refresh → Qdrant`. The trending worker never connects to Qdrant or patches repository payloads directly. An incomplete enrichment batch or failed backend snapshot request fails the refresh atomically; inspect the backend outbox and ML refresh workers for downstream delivery failures.

## Diagnosis

| Symptom | Meaning | Safe action |
|---|---|---|
| `GITHUB_TOKEN` configuration error | Token is absent or still a placeholder | Set a real token and rerun network-free validation |
| Backend verification failure | Authenticated repository delivery cannot be trusted | Correct `BACKEND_URL`/`INTERNAL_API_SECRET`; do not bypass delivery |
| No acquisition progress | Discovery returned nothing new, filtering rejected everything, or the backend canonical count did not advance | Inspect rejection and failure records before increasing limits |
| `pending_persistence` is non-empty | Repositories still require enrichment and backend delivery | Rerun the same bounded corpus command |
| Legacy `pending_index` is non-empty | A pre-cutover checkpoint still records direct-index work | Do not write Qdrant directly; reconcile those repository IDs through backend ingestion/outbox and then retire the legacy checkpoint state |
| Trending repository is missing in Qdrant | The backend outbox or ML refresh has not completed, or the repository has no approved corpus point | Inspect backend outbox/ML refresh state; let normal corpus acquisition approve missing repositories rather than writing Qdrant from the trending worker |
| Unsupported embedding model | The embedding owner has not published a compatible model/dimension contract | Keep the current model until the shared embedding interface is updated |

## Verification before handoff

Run the deterministic suite:

```bash
uv run pytest -q
```

Tests mock external services. A default test run must not contact GitHub, Postgres, Qdrant, or OpenRouter.

Before handing a new corpus to online serving, confirm the backend outbox has
completed repository embed/refresh jobs and authenticated V2 health reports an
eligible count above `MIN_ELIGIBLE_REPOSITORIES`. Points missing the pinned
embedding revision or required feature-spec version must be reindexed through
the backend outbox; do not weaken production eligibility to accept them. The
same validated reindex must stamp
`serving_eligibility_version=repository-vector-v2` atomically with the vector;
never backfill that marker with a payload-only patch. See the production
runbook for the coordinated cutover and the limits of write-time vector
certification.

## GitHub quota and retry coordination

Acquisition clients in one Python process coordinate by endpoint and an opaque,
process-keyed credential fingerprint. Tokens and authorization headers are not
stored in the registry, counters or logs. Separate processes, different tokens
sharing an account, and other consumers are not coordinated; use an external
quota service if a deployment needs account-wide admission.

Known remaining quota is reserved conservatively. HTTP 429, rate-limited 403,
and GraphQL RATE_LIMITED responses honor both Retry-After (seconds or HTTP date)
and reset signals. Resumption is jittered and staggered. Auth/permission and
other permanent HTTP errors do not retry. A request allows at most max_retries
additional attempts (default four), including scheduler resumptions. Exhausted
transport/server/rate-limit failures do not start new singleton fallback loops.

README delay queues retain the original logical future, repository identity and
attempt budget, with at most twice GITHUB_README_WORKERS admitted jobs. They do
not occupy transport workers while backing off. Live IO retains its worker and
admission until completion, including during cancellation. Metadata/discovery
callers share quota but wait synchronously; their waits check cancellation and
deadlines every 50ms. A running requests call is bounded by its transport timeout
and cannot be forcibly interrupted by Python. The optional acquisition deadline
is an absolute time.monotonic() value, propagated to admission, retry waits and
request timeouts. It is not a hard wall-clock bound on socket trickle reads.

GitHubGraphQLClient.rate_limits.snapshot() exposes process-shared requests,
admission_deferred, throttled, retries, waiting and remaining counters without
credential labels. These are admission/attempt counters, not measured throughput
or proof of bytes saved. Fake-clock concurrency/cancellation and injected-throttle
throughput measurements remain required before claiming runtime acceptance.
