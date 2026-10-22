# V2 Production Operations Runbook

This runbook is the operating contract for the Redis + Qdrant online ML path:

```text
backend outbox/API -> ML API -> Redis stream -> ordered feedback worker -> Qdrant
                    ML API -> Qdrant retrieval/ranking -> backend feed serve
                    ML API -> scoped summary provider (bounded derived text only)
```

PostgreSQL, GitHub, legacy OpenRouter, acquisition, training, and trending
credentials do not belong in either online container. A dedicated scoped
`SUMMARY_API_KEY` is the sole online LLM credential. The backend owns product
state and canonical UUIDs; ML owns derived vectors and recommendation
computation.

## Release gates

Do not expose the service broadly until all of these are true:

1. The backend uses canonical UUID `repo_id` and monotonic content, feature,
   profile, and feedback versions.
2. Repository points have the configured embedding model, pinned revision,
   embedding version, dimension, content version, feature-spec version, and
   `serving_eligibility_version=repository-vector-v2` certification.
3. The eligible repository count is at least `MIN_ELIGIBLE_REPOSITORIES`.
4. The dedicated smoke user is onboarded and can receive at least one result.
5. Redis and both Qdrant collection contracts pass authenticated V2 health.
6. The feedback consumer heartbeat is live and pending/lag/DLQ values are below
   their hard thresholds.
7. CI has passed the real V2 API -> Redis -> consumer -> Qdrant integration test.
8. The GitHub `production` environment has required reviewers and restricted
   deployment branches configured by a repository administrator.

## Production configuration

Start from [`deploy/production.env.example`](../deploy/production.env.example),
not the combined local `.env.example`. Install it as the root-owned
`/etc/gh-social/ml.env` with mode `600`. Do not put it in Git or an image layer.

The deployment and both systemd units run the network-free validator before a
restart. Run both modes manually as well: the first parses the raw file to catch
malformed or duplicate entries, while the second validates the effective
container environment and confirms that its model and revision match the baked
artifact.

```bash
sudo install -o root -g root -m 600 /secure/path/completed-ml.env /etc/gh-social/ml.env

sudo docker run --rm --network none --no-healthcheck --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=16m \
  -v /etc/gh-social/ml.env:/run/ml.env:ro \
  -v /usr/local/lib/gh-social-ml/validate_production_config.py:/run/validate_production_config.py:ro \
  gh-social-ml:current \
  python /run/validate_production_config.py --env-file /run/ml.env

sudo docker run --rm --network none --no-healthcheck --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=16m \
  --env-file /etc/gh-social/ml.env \
  -v /usr/local/lib/gh-social-ml/validate_production_config.py:/run/validate_production_config.py:ro \
  gh-social-ml:current \
  python /run/validate_production_config.py
```

Success validates parsing and cross-field invariants only. It does not contact
Redis, Qdrant, or the API. Never paste the env file into an issue or deployment
log. Validation errors contain variable names, not values.

### Required schema

The deployment template is authoritative. The groups below explain their
purpose and production rule.

| Group | Variables | Production rule |
| --- | --- | --- |
| Boundary/auth | `APP_ENV`, `V2_FEEDBACK_CONSUMER_REQUIRED`, `INTERNAL_API_HEADER`, `INTERNAL_API_SECRET` | Exactly `production`, `true`, `x-internal-secret`, and a unique secret of exactly 64 lowercase hexadecimal characters. Generate it independently per environment with `openssl rand -hex 32`. |
| Redis stream | `REDIS_AUTH_MODE=acl_url`, credential-bearing `REDIS_URL`, `FEEDBACK_STREAM_NAME`, `FEEDBACK_STREAM_MAXLEN`, `FEEDBACK_CONSUMER_GROUP`, `FEEDBACK_CONSUMER_PREFIX`, `FEEDBACK_IDEMPOTENCY_TTL_SECONDS` | Redis 6.2+ with a least-privilege ACL principal scoped to the documented stream, lock, dedupe, heartbeat, attempt, and replay keys/commands. Use `redis://` only when transport security is provided by the private network boundary; use `rediss://` when the Redis endpoint provides TLS. The exact maximum counts outstanding work; new events receive retryable backpressure at capacity. ACK/source deletion and DLQ transfer are atomic. |
| Consumer | `FEEDBACK_READ_BATCH_SIZE`, `FEEDBACK_READ_BLOCK_MS`, `FEEDBACK_RECLAIM_IDLE_MS`, `FEEDBACK_MAX_DELIVERY_ATTEMPTS`, `FEEDBACK_CONSUMER_HEARTBEAT_KEY`, `FEEDBACK_CONSUMER_HEARTBEAT_TTL_SECONDS` | Bounded reads/reclaims; required heartbeat. |
| User lock | `FEEDBACK_USER_LOCK_PREFIX`, `FEEDBACK_USER_LOCK_TTL_SECONDS`, `FEEDBACK_USER_LOCK_WAIT_SECONDS`, `FEEDBACK_USER_LOCK_RENEW_INTERVAL_SECONDS` | One namespace for onboarding and feedback; renewal is less than half the TTL. |
| DLQ and state bounds | `FEEDBACK_DEAD_LETTER_STREAM`, `FEEDBACK_DEAD_LETTER_MAXLEN`, `FEEDBACK_REJECTION_HISTORY_SIZE`, `FEEDBACK_MAX_TRACKED_REPOSITORIES`, `FEEDBACK_MAX_USER_STATE_BYTES` | The DLQ and per-user feedback ledger are bounded by both count and serialized bytes. A full DLQ leaves source work pending instead of trimming an unresolved incident. Recommendation reads use a narrow payload projection and never transfer this ledger. |
| Feedback/Qdrant | `FEEDBACK_QDRANT_TIMEOUT_SECONDS`, `FEEDBACK_DWELL_MIN_SECONDS`, `FEEDBACK_DWELL_FULL_CREDIT_SECONDS`, `FEEDBACK_DWELL_MAX_ALPHA` | Finite, bounded timeout and dwell policy. |
| Feedback health | `FEEDBACK_HEALTH_WARN_*`, `FEEDBACK_HEALTH_MAX_*` | Warnings do not exceed hard failures; hard values fail readiness. |
| Qdrant | `QDRANT_URL`, `QDRANT_AUTH_MODE=api_key`, required `QDRANT_API_KEY`, `QDRANT_TIMEOUT_SECONDS`, `QDRANT_DISTANCE`, `QDRANT_COLLECTION_NAME`, `QDRANT_VECTOR_NAME`, `USER_PROFILES_COLLECTION`, optional `USER_PROFILE_VECTOR_NAME`, `VECTOR_DIMENSION`, `V2_USER_COLLECTION_REQUIRED` | Qdrant 1.18.0+ and authenticated, restricted writes are required for fencing and serving-marker integrity. Repository and user contracts must be compatible. The current user collection uses its unnamed vector. |
| Embedding | `EMBEDDING_MODEL`, `EMBEDDING_MODEL_REVISION`, `REPOSITORY_EMBEDDING_VERSION`, `V2_COMPATIBLE_EMBEDDING_VERSIONS`, `REPOSITORY_FEATURE_SPEC_VERSION`, `V2_REQUIRED_FEATURE_SPEC_VERSION`, `V2_REQUIRED_CONTENT_VERSION`, `V2_ALLOW_MISSING_EMBEDDING_REVISION`, `README_CHUNK_CHARS`, `README_CHUNK_OVERLAP_CHARS` | Exact baked model/revision; explicit compatible versions; missing revision is forbidden in production; chunking is bounded. |
| Card summary | `SUMMARY_PROVIDER`, `SUMMARY_API_URL`, `SUMMARY_API_KEY`, `SUMMARY_MODEL_ID`, `SUMMARY_PROMPT_VERSION`, `SUMMARY_FORMAT_VERSION`, `SUMMARY_INPUT_MAX_CHARS`, `SUMMARY_TEMPERATURE`, `SUMMARY_MAX_OUTPUT_TOKENS`, `SUMMARY_REQUEST_TIMEOUT_SECONDS`, `SUMMARY_RPM_LIMIT`, `SUMMARY_MAX_RETRIES`, `SUMMARY_RETRY_BASE_SECONDS` | Dedicated least-privilege provider key; HTTPS transport; frozen prompt/schema identity; low temperature; bounded input, output, timeout, retries, and request rate. |
| Readiness | `MIN_ELIGIBLE_REPOSITORIES` | Must be a positive capacity chosen for the launch market, not merely `1` for convenience. |
| Model runtime | `EMBEDDING_WARMUP_ON_STARTUP`, `EMBEDDING_MAX_CONCURRENCY`, `EMBEDDING_EXECUTOR_WORKERS`, `EMBEDDING_CPU_THREADS`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | Warmup and offline flags are true; concurrency is bounded. |
| Ranking | `ML_MODEL_VERSION`, `V2_HEAVY_RANKER_ENABLED`, `V2_HEAVY_RANKER_REQUIRED`, `V2_HEAVY_RANKER_TRAFFIC_PERCENT`, `V2_HEAVY_RANKER_CANARY_SALT`, `V2_ALLOW_UNQUALIFIED_HEAVY_RANKER`, `V2_EXPLORATION_FRACTION`, `V2_MAX_SAME_LANGUAGE` | Unqualified artifacts are forbidden. Required-heavy mode mandates 100% traffic and fails closed rather than serving the hybrid ranker. |
| Service isolation | `V2_RECOMMENDATION_TIMEOUT_SECONDS`, `V2_RECOMMENDATION_EXECUTOR_WORKERS`, `V2_RECOMMENDATION_MAX_OUTSTANDING`; the `V2_FEEDBACK_*`, `V2_REFRESH_*`, `V2_SEARCH_*`, and `V2_HEALTH_*` executor-worker, max-outstanding, and timeout variables | Recommendations, feedback mutation, repository refresh, semantic search, and health work use separate bounded executors with fail-fast admission. Search shares the frozen encoder with indexing but does not queue behind summary-provider waits. Capacity includes running and queued work. Timed-out work retains capacity until it really exits; do not raise limits without host-specific load and dependency-failure tests. |
| Repository jobs | `REPOSITORY_JOB_LOCK_TTL_MS`, `REPOSITORY_JOB_LOCK_WAIT_SECONDS` | Renewable lock TTL exceeds wait time. |
| Deployment smoke | `ML_SMOKE_USER_ID`, `ML_SMOKE_RECOMMENDATION_LIMIT`, `ML_SMOKE_EXPECT_MIN_ITEMS`, `ML_SMOKE_TIMEOUT_SECONDS` | Dedicated, onboarded, non-human user; bounded read-only recommendation smoke. |

`DATABASE_URL`, `TEST_DATABASE_URL`, `LOCAL_DATABASE_URL`,
`SUPABASE_DATABASE_URL`, `GITHUB_TOKEN`, legacy `OPENROUTER_API_KEY`, and
`GROQ_API_KEY` are rejected in the online env file.

## Model cache and collection migration

Before enabling these binaries against an existing persistent Qdrant cluster,
upgrade the server to at least 1.18.0 using Qdrant's supported sequential-minor
upgrade procedure and take a verified snapshot. Readiness reports the connected
server version and fails closed below that minimum; do not bypass this gate,
because onboarding, repository jobs, and feedback rely on conditional writes.
The deployment then runs `python -m scripts.bootstrap_qdrant` before retagging
the image. This idempotently creates or validates both collections and waits
for every required repository payload index; incompatible index types fail the
deployment before traffic moves.

The image bakes
`sentence-transformers/all-MiniLM-L6-v2` at immutable revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`. Production sets Hugging Face and
Transformers offline modes. API startup loads and validates one process-scoped
model, then repository and user jobs share it through a bounded executor.
Recommendation work does not wait in that executor.

Points created before the revision and feature-spec fields were introduced are
not production-eligible. Re-embed/reindex them through the normal backend
outbox. Do not set `V2_ALLOW_MISSING_EMBEDDING_REVISION=true` to hide an
incomplete migration. During a deliberate embedding migration, list only
versions known to share the same model/revision/dimension contract in
`V2_COMPATIBLE_EMBEDDING_VERSIONS`; health and responses must report the
version actually used.

The serving eligibility marker is a write-time certification, not a live
per-request proof of vector presence. `QdrantRepositoryStore.upsert` rejects a
caller-supplied marker, validates the vector and pre-serving payload, then
atomically writes that vector with
`serving_eligibility_version=repository-vector-v2`. Hybrid recommendation
queries filter and count that indexed marker without transferring repository
vectors. This keeps the default serving path bounded, but it relies on Qdrant
write credentials being restricted to the validated ML ingestion path. Do not
grant general services permission to patch or forge this field, and do not
backfill it with `set_payload`.

This contract requires a coordinated migration because old points have no
marker and will fail closed:

1. Keep the old serving image active.
2. Run the candidate image's collection bootstrap to create the keyword index.
3. Re-embed/reindex the required corpus through the backend outbox and candidate
   ingestion path; this validates and atomically certifies each vector.
4. Verify the certified count is at least `MIN_ELIGIBLE_REPOSITORIES` and audit
   a bounded sample with vectors included.
5. Deploy the new serving image. Its health count and smoke check require the
   same marker/version before traffic can move.

Deleting or replacing vectors through another Qdrant writer can invalidate the
certification without changing its payload. Writer isolation is therefore part
of the serving contract; periodic offline vector-present sampling remains
necessary for corruption detection.

## Backend ordering and idempotency

The safe delivery order is:

1. create the canonical backend repository and deliver its content embedding
   job;
2. deliver monotonic repository feature refreshes;
3. onboard the canonical user and wait for the user vector;
4. deliver ordered feedback with contiguous per-user `feedback_version`;
5. request recommendations and persist only the slice actually served.

Repository and onboarding `job_id` values are idempotency keys. Retrying the
same job/version is safe. Reusing a job ID for different content is a conflict.
Older versions return HTTP 409. Lock or dependency pressure returns HTTP 503
with `Retry-After`; retry with bounded exponential backoff and jitter. Contract
errors return HTTP 422 and require producer repair, not blind retry.

Feedback event IDs are deduplicated atomically with stream append. Retry the
entire batch after an uncertain/partial publish; accepted versus duplicate
counts remain accurate. Do not send feedback until both repository and user
vectors exist. If delivery order is temporarily violated, the event remains
retryable rather than being acknowledged and lost.

## Recommendation determinism and call budget

The default personalized hybrid path has a fixed upper budget of six Qdrant
round trips: one batched canonical/legacy user-profile retrieval, one semantic
query, and four bounded discovery-channel scrolls. It performs no Redis call,
online embedding, heavy-model inference, or repository-vector transfer.
Missing-profile discovery uses five Qdrant calls. CI asserts this budget so a
new per-item lookup cannot silently create an N+1 latency regression.

Freshness is evaluated at the UTC hour first observed for a generation. A
process-local FIFO retains 65,536 generation anchors for up to six hours. This
makes retrying one `generation_id` stable across a UTC-hour boundary without
adding a shared-cache call, while a new generation immediately receives the
current freshness hour. The current production image runs one API worker, so
normal retries share that bounded cache. Eviction, process restart, or routing
one generation across multiple replicas can recalculate the anchor; a future
multi-replica deployment that requires global retry identity must add backend
generation time to the request contract or use affinity/shared idempotency
state after measuring its latency.

## Feedback retry, rejection, and replay

The ordered consumer handles failures as follows:

- Missing user/repository vectors, dependency errors, version gaps, and a busy
  renewable lock remain pending and are reclaimed without a tight loop.
- After `FEEDBACK_MAX_DELIVERY_ATTEMPTS`, retryable events are written to the
  bounded DLQ before the source message is acknowledged. Their cursor is not
  advanced.
- A terminal-invalid event is also preserved in the DLQ. If it is exactly the
  next version and user state is valid, the consumer records a bounded
  rejection on the user payload and advances the cursor without changing the
  vector. This prevents the next valid version from being blocked.
- A terminal event with a gap, missing user, or corrupt stored state is marked
  unreconciled; an operator must repair the prerequisite before replay.
- Reusing an event ID with different content is a terminal
  `EVENT_ID_PAYLOAD_CONFLICT`. Reusing the current feedback version with a
  different event ID is a terminal `VERSION_EVENT_CONFLICT`.
- Adding persistent state for a new repository at
  `FEEDBACK_MAX_TRACKED_REPOSITORIES` is terminal; existing state and valid
  reversals remain usable so a reversal can release capacity.
- At main-stream capacity, the API returns retryable backpressure and does not
  create a dedupe key. At DLQ capacity, the source entry stays pending and
  readiness remains failed until an operator resolves or replays incidents.

Inspect only the specific DLQ entry needed for an incident. The replay command
is non-mutating unless `--execute` is supplied:

```bash
sudo docker exec gh-social-ml-feedback \
  python -m feedback.v2_replay --source-id 1234567890-0 --dry-run

sudo docker exec gh-social-ml-feedback \
  python -m feedback.v2_replay --source-id 1234567890-0 --execute
```

Normal replay is for retry-exhausted dependency/gap events after repairing the
missing vector or version. A terminal-invalid entry is refused by default;
`--allow-terminal --execute` is only for an unreconciled entry whose cursor was
not advanced. A cursor-advanced rejection cannot be rewritten at the same
version: issue a corrected compensating backend event at the next feedback
version instead. Replay is idempotent by DLQ source ID. It checks main-stream
capacity atomically; overload leaves the DLQ entry untouched, while a successful
transfer deletes the resolved source entry. Record the incident, source ID,
repair, and result.

## Health and smoke checks

All health tiers retain internal authentication:

- `GET /api/v2/health/live` is process liveness only. It never queries Redis or
  Qdrant, loads models, or submits executor work. Use it for frequent liveness.
- `GET /api/v2/health/ready` and legacy `GET /api/v2/health` read the same bounded
  snapshot without dependency work. Use readiness for traffic/container probes.
  Initial, failed, timed-out, or stale required diagnostics return HTTP 503.
- `GET /api/v2/health/deep` requests detailed diagnostics, sharing a single actual
  coordinator and the same cooldown as the background refresh. Each actual cycle
  submits exactly two parallel dependency checks to its own two-worker executor,
  never back to the reserved coordinator pool. Both actual children are joined
  before coordinator completion, even on partial submission or failure, so timed-out
  waiters cannot release their ownership early or race shared-client cleanup.
  It may return a recent snapshot rather than force a new check and never queues
  duplicate refreshes.

The process refreshes diagnostics after startup and every
`V2_HEALTH_REFRESH_SECONDS` (default 15) after the previous bounded wait.
`V2_HEALTH_MAX_AGE_SECONDS` (default 45) is measured from the diagnostic start
using a monotonic clock, not completion or response time. It must cover refresh
plus two `V2_HEALTH_TIMEOUT_SECONDS` deadlines (default 5 each). Responses expose
UTC checked/completed timestamps, age, freshness, in-flight state and safe
per-dependency status. A timed-out/stuck worker retains its admission slot until
it really exits, and cannot later turn its timed-out result into healthy data.
A failed refresh immediately invalidates earlier success. Shutdown stops new
refreshes immediately and retires the actual diagnostic future before closing
shared clients or clearing their caches. If a native RPC is still running,
cleanup is deferred to its completion callback rather than blocking the API
shutdown indefinitely or closing dependencies underneath it. Native RPC/thread
termination remains subject to the existing client timeouts and process manager;
this is not a promise to forcibly kill a running worker. Each captured client,
executor and cache cleanup action is attempted independently. Failures report
only fixed step names in sanitized logs and `runtime_cleanup`; retirement tracks
scheduled/running/completed/failed state rather than treating scheduling as
success. A later shutdown call retries only failed captured actions without
recreating clients. Incomplete cleanup blocks a new runtime startup and readiness.
No fleet-wide or production performance gain is implied by these defaults.

The Docker image uses `--readiness-only`, which does no refresh. Deployment
smoke checks use the explicit deep tier and retain the complete contract gate.
A production-ready response
must report:

- healthy repository and required user collection contracts;
- configured model, immutable revision, embedding version and allowlist;
- serving eligibility version and validated-vector-at-upsert evidence;
- eligible repository count at or above the configured minimum;
- Redis pending, lag, source length, DLQ length, and active consumer heartbeat;
- heavy-ranker readiness/qualification when required;
- `database_required: false`.

Run the cached container probe and the full deployment gate from inside the API container:

```bash
sudo docker exec gh-social-ml \
  python -m scripts.production_smoke --readiness-only --timeout 8

sudo docker exec gh-social-ml \
  python -m scripts.production_smoke --health-only --timeout 8

sudo docker exec gh-social-ml \
  python -m scripts.production_smoke --timeout 10
```

The final command sends one deterministic request for `ML_SMOKE_USER_ID` and
validates UUIDs, uniqueness, finite scores, item count, model version, and the
embedding allowlist. It does not mutate a user vector. Never use a real user as
the smoke identity.

For health load comparisons, use the separate fixed route latency series for
`/health/live`, `/health/ready` and `/health/deep`, plus
`ml_service_executor_outstanding{operation="health"}` and its admission/timeout
counters. `ml_health_dependency_checks_total{dependency,outcome}` counts logical
Qdrant/Redis diagnostic invocations, not individual client RPCs;
`ml_health_dependency_duration_seconds_sum{dependency}` records their elapsed
worker time. Compare client-side RPC instrumentation separately when measuring
actual Qdrant/Redis call counts. Routine probe traffic should leave the logical
check rate unchanged at the configured controlled cadence. Benchmarks and live
outage/load verification must still be run before rollout.

## Metrics scrape and alert wiring

`GET /api/v2/metrics` exposes bounded, fixed-cardinality Prometheus text and is
protected by the same internal header as every other V2 endpoint. The checked-in
unit binds the API to `127.0.0.1:8000`, so run the scraper on the host or put an
authenticated private proxy in front of it; never expose the metrics route
directly to the internet. Provision only the secret value into a root-managed
file readable by the Prometheus process. Do not parse or mount the complete ML
env file into Prometheus.

Prometheus versions that support custom `http_headers` can use this scrape job:

```yaml
scrape_configs:
  - job_name: gh-social-ml
    scrape_interval: 15s
    scrape_timeout: 5s
    metrics_path: /api/v2/metrics
    scheme: http
    http_headers:
      X-Internal-Secret:
        files:
          - /etc/prometheus/secrets/gh-social-ml-internal-secret
    static_configs:
      - targets: ["127.0.0.1:8000"]
```

The secret file contains only the exact 64-character value and must be supplied
through the deployment secret manager, not committed. Validate and reload the
scraper with `promtool check config`, then confirm `up{job="gh-social-ml"} == 1`
and that the target never logs or displays request headers. If the installed
Prometheus does not support custom-header files, use a loopback proxy that adds
the header from its secret store; do not put the secret in the metrics URL.

Wire alerts for at least:

- `up == 0` or scrape errors;
- sustained 5xx ratio from `ml_api_requests_total`;
- recommendation p95/p99 from `ml_api_request_duration_seconds_bucket` against
  the backend deadline and product SLO;
- any increase in `ml_service_executor_rejected_total` or
  `ml_service_executor_timed_out_total` and sustained
  `ml_service_executor_outstanding >= ml_service_executor_capacity`;
- an abnormal ratio of `ml_empty_recommendations_total` to
  `ml_recommendations_total`;
- heavy-ranker fallback codes above the canary baseline.

The following is a starting rule file; replace the example latency/error
thresholds with the approved launch SLOs and keep them below backend deadlines:

```yaml
groups:
  - name: gh-social-ml
    rules:
      - alert: GhSocialMlMetricsDown
        expr: up{job="gh-social-ml"} == 0
        for: 2m
      - alert: GhSocialMlHigh5xxRatio
        expr: |
          sum(rate(ml_api_requests_total{job="gh-social-ml",status_class="5xx"}[5m]))
          /
          clamp_min(sum(rate(ml_api_requests_total{job="gh-social-ml"}[5m])), 0.001)
          > 0.01
        for: 5m
      - alert: GhSocialMlRecommendationP95Slow
        expr: |
          histogram_quantile(0.95,
            sum by (le) (
              rate(ml_api_request_duration_seconds_bucket{
                job="gh-social-ml",
                route="/api/v2/recommendations/generate"
              }[5m])
            )
          ) > 2.5
        for: 5m
      - alert: GhSocialMlExecutorSaturated
        expr: |
          ml_service_executor_outstanding{job="gh-social-ml"}
          >= on(job, instance, operation)
          ml_service_executor_capacity{job="gh-social-ml"}
        for: 1m
      - alert: GhSocialMlExecutorRejectedOrTimedOut
        expr: |
          increase(ml_service_executor_rejected_total{job="gh-social-ml"}[5m]) > 0
          or
          increase(ml_service_executor_timed_out_total{job="gh-social-ml"}[5m]) > 0
```

Run `promtool check rules` in CI for the monitoring repository. Route metrics
down, executor rejection/timeout, sustained 5xx, and the product's paging
latency threshold to the on-call service; route early latency, fallback, and
empty-feed deviations as rollout warnings with links to this runbook.

The executor series cover the reserved recommendation, feedback,
repository-refresh, and health pools. Keep scrape labels to `job`, `instance`,
and the emitted fixed labels; never add user, repository, request, event, or
generation IDs. These metrics are process-local and reset on restart. The
current unit runs one API process; a multi-worker or multi-replica topology must
scrape every process and aggregate by stable labels instead of scraping through
a load balancer.

## Deployment and rollback

A successful `Test Trending Service` push run on `main` starts deployment for
that exact tested SHA. A stale completion is skipped if `main` has advanced.
One non-cancelling deployment concurrency group prevents two workflow releases
from racing image aliases on the host. The workflow targets the GitHub
`production` environment. Repository administrators must configure its required
reviewers and branch restrictions; declaring the environment in YAML does not
create those protection rules.

Configure `AWS_HOST`, `AWS_USER`, `AWS_PRIVATE_KEY`, and
`AWS_HOST_FINGERPRINT` as production-environment secrets. The fingerprint must
be the pinned SSH host-key fingerprint (for example, `SHA256:...`) verified
through an out-of-band trusted channel. Both upload and command actions reject a
different key. Rotate this secret only as part of an intentional host-key
rotation; never accept a fingerprint learned from the failing deployment path.
The deploy principal must have non-interactive (`sudo -n`) authorization for
the documented Docker, systemd, file-install, disk-inspection, and journal
operations. Scope sudoers to this host and deployment path, and test it before
cutover; the workflow never waits for or supplies a sudo password.

Deployment order is:

1. run the complete test suite, build the immutable SHA-tagged image, and verify
   that the tested SHA is still `main`;
2. upload over fingerprint-pinned SSH, verify the SHA-256 manifest for the
   archive/size metadata/helpers/units, and require free Docker storage of at
   least the larger of the inspected image size or three times the compressed
   archive, plus 2 GiB, before loading;
3. install SHA-versioned validator and smoke helpers, run both network-free
   configuration modes, and idempotently bootstrap/validate Qdrant;
4. preserve the prior `current` image under a transaction-only tag and back up
   the active helpers, both unit files, and their enabled/active states;
5. atomically replace each helper and unit file, retag the candidate as
   `current`, then restart feedback before API;
6. wait for authenticated V2 readiness and run the deterministic recommendation
   smoke using the activated helper copied from the candidate's versioned set;
7. only after successful verification, move the former image to `previous` and
   retain the five newest immutable 40-hex SHA tags. Retention never prunes
   `current`, `previous`, volumes, arbitrary images, or helper directories.

Any ordinary command error, interrupt, or termination after step 4 invokes the
same compensating transaction: restore the prior image alias, helpers, units,
enablement and activity; restart in dependency order; then run authenticated
health and recommendation verification with the restored prior smoke helper.
An existing `current` release without both helpers and both unit files is
rejected before mutation because it has no complete rollback set. The workflow
still fails even when rollback succeeds. A first deployment has no prior image
to verify, and a rollback that fails any restoration or smoke check is a
critical incident.
Successful transaction backups and SHA-versioned helpers remain under
`/usr/local/lib/gh-social-ml`; remove old directories only through a separate,
reviewed retention procedure after the rollback window.

Qdrant bootstrap occurs before service mutation and may create collections or
indexes. It is designed to be additive and idempotent, but the host transaction
does not reverse Qdrant schema or data. A migration that is not backward
compatible therefore requires the staged collection procedure above, not an
assumption that image rollback will undo it. A host crash, kernel kill, or lost
SSH process can also bypass shell traps; after any interrupted run, inspect the
transaction backup and verify aliases, units, helpers, and both smokes before
resuming traffic.

Useful diagnosis commands:

```bash
sudo systemctl status gh-social-ml-feedback gh-social-ml
sudo journalctl -u gh-social-ml-feedback -u gh-social-ml -n 200 --no-pager
sudo docker inspect --format '{{.State.Health.Status}}' gh-social-ml
sudo docker image ls gh-social-ml
sudo df -h "$(sudo docker info --format '{{.DockerRootDir}}')"
```

The image healthcheck reads the secret from the running API container and does
not embed it in an image layer. The feedback unit disables that API-specific
healthcheck. Deployment/systemd authenticated checks remain the release
authority.

## Current topology and coordination limits

The checked-in production topology is intentionally a single-host starting
point, not a highly available design:

- one API container and one feedback container run on one Docker/systemd host;
  a host failure is a full outage, and each in-place restart creates a brief
  connection gap because there is no load-balancer drain or blue/green peer;
- the API unit binds only `127.0.0.1:8000`. A backend on another host cannot
  reach it until an authenticated private reverse proxy or load balancer is
  provisioned; that network/TLS component is not checked in here;
- `current`, `previous`, helper versions, and transaction backups are local to
  that host. They do not recover a lost instance or Docker volume; replicate
  artifacts externally and rebuild hosts from automation;
- GitHub concurrency serializes this repository's workflow runs only. Manual
  host changes or a different pipeline can still race deployment unless every
  operator uses one host-level lock and the same transaction procedure;
- Redis and Qdrant availability, backup, restore, TLS, ACL/key rotation, and
  topology are external prerequisites. Image rollback cannot repair their
  outage, restore their data, or reverse an incompatible schema migration;
- the API and feedback quotas bound container consumption but do not reserve
  physical CPU. Size the host for their combined 3-CPU/3-GiB ceilings plus
  Docker, kernel, monitoring, and dependency-client overhead, then load-test
  simultaneous feedback, refresh, health, scrape, and recommendation traffic;
- metrics are process-local and reset on restart. They cover the four API
  executors, but embedding-job capacity, host/container pressure, Redis,
  Qdrant, and systemd health need additional instrumentation/exporters before
  broad rollout;
- the metrics scraper receives the general internal API secret, which also
  authorizes mutation routes. Keep it on a trusted loopback boundary and plan a
  dedicated read-only metrics credential or mTLS identity before moving the
  scraper across a trust boundary.

Do not call the service highly available until there are at least two
independently scheduled API instances behind health-aware routing, a singleton
or partition-safe feedback-consumer strategy, external artifact/backup
recovery, and a tested host-loss exercise. Retry determinism across API replicas
also requires backend-supplied generation time or measured shared/affinity
state as described above.

## Alerts

| Signal | Warning | Page / release block | First response |
| --- | --- | --- | --- |
| Metrics scrape | One missed interval | `up == 0` for 2 minutes | Check the API unit, loopback route, secret rotation, and Prometheus target error. |
| Reserved executor | Outstanding near capacity | Any rejection/timeout or sustained full capacity | Stop rollout; inspect the named operation, dependency latency, and host pressure before changing bounds. |
| Feedback pending | `FEEDBACK_HEALTH_WARN_PENDING` | `FEEDBACK_HEALTH_MAX_PENDING` | Check worker heartbeat, lock contention, and Qdrant latency. |
| Feedback lag | `FEEDBACK_HEALTH_WARN_LAG` | `FEEDBACK_HEALTH_MAX_LAG` | Stop rollout; inspect dependency errors and reclaim rate. |
| Stream length | `FEEDBACK_HEALTH_WARN_STREAM_LENGTH` | `FEEDBACK_HEALTH_MAX_STREAM_LENGTH` | Confirm trim settings and producer/consumer rates. |
| DLQ | Any growth | `FEEDBACK_HEALTH_MAX_DEAD_LETTER` | Classify codes; repair prerequisites before replay. |
| Consumer heartbeat | Missing | Immediate | Restart/diagnose the dedicated worker; API health must fail closed. |
| Eligible corpus | Near minimum | Below minimum | Pause deployment; reindex incompatible points. |
| Empty/short feeds | Sustained above baseline | Smoke user below minimum or user-facing spike | Check eligibility filters, exclusions, and Qdrant errors. |
| Heavy fallback | Above canary baseline | Sustained model/dependency fallback | Set traffic to `0`; inspect bounded reason counters. |
| Recommendation latency | p95/p99 budget breach | Sustained SLO breach | Roll back the release; check Qdrant, executor saturation, CPU and memory. |
| Embedding jobs | Queue/latency growth | Recommendation starvation or host pressure | Keep concurrency bounded; scale workers only after measurement. |

Logs must include request/event/job identifiers and stable status/error codes,
but never secrets, credential-bearing URLs, README contents, complete vectors,
or raw dependency exceptions.

## Heavy-ranker promotion

Production is configured as strict V2 heavy-only:

```dotenv
V2_HEAVY_RANKER_ENABLED=true
V2_HEAVY_RANKER_REQUIRED=true
V2_HEAVY_RANKER_TRAFFIC_PERCENT=100
V2_ALLOW_UNQUALIFIED_HEAVY_RANKER=false
```

The checked-in model was trained on synthetic interactions and is deliberately
not production-qualified, so this configuration blocks deployment until it is
replaced. A replacement artifact requires real telemetry identity, immutable model and
scaler hashes, pinned embedding model/revision and compatible versions,
feature/value-function versions, training timestamp, code version, offline
ranking metrics, calibration metrics, and slice evaluation. Promote it through
offline evaluation and shadow comparison before replacing the production
artifact. Required-heavy serving fails closed when
the onboarding vector or heavy result is unavailable; it never silently serves
the hybrid ranker. Roll back the complete release instead of enabling hybrid
traffic; do not retrain or relabel synthetic data as production data.

## Rollout sequence

1. **Staging:** exercise authenticated backend -> API -> Redis -> consumer ->
   Qdrant and verify DLQ/replay with disposable identities.
2. **Offline/shadow evaluation:** collect real evaluation telemetry and compare
   the candidate heavy ranker without exposing an unqualified artifact.
3. **Qualified staging:** install the production-qualified artifact with strict
   heavy-only flags and watch quality, calibration, latency, empty feeds, lag,
   and DLQ.
4. **Production release:** deploy the same immutable artifact and roll back the
   complete release if any gate fails.

No component can guarantee zero latency. The goal is bounded work and no
meaningful hot-path regression: shared warmed models, isolated embedding work,
bounded Redis/Qdrant operations, deterministic feed shaping, and immediate
ranking rollback controls.
