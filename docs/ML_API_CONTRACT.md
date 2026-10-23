# Versioned backend/ML contract

`contracts/ml-v2.openapi.json` is generated from the registered FastAPI product
routes and their Pydantic request/response models. It describes schema version 2,
canonical backend UUIDs, recommendation provenance, versioned card summaries,
onboarding, refresh, feedback and semantic search. Authentication uses the default
`x-internal-secret` header; an operator-configured header override must be coordinated
with callers. Operational health/metrics are outside this product artifact.

Regenerate without starting the application lifespan, models or dependency clients:

```bash
uv run python -m scripts.export_ml_openapi
uv run python -m scripts.export_ml_openapi --check
```

CI checks deterministic export and compares an established base contract with
`oasdiff`, failing on incompatible changes. Initial publication has no previous
artifact to compare. The comparison stays in CI: remote review upload and comment
publication are disabled. Response schemas validate successful route output;
runtime validators still enforce contextual constraints that OpenAPI cannot fully
express (matching nested IDs, timestamp rules, meaningful profile context, summary
quality and version/CAS semantics).

Weave vendors a release copy and generates its TypeScript client types with a
pinned generator. Export changes first, update the consumer's release copy,
regenerate, compile and run the paired HTTP contract test before rollout. Additive
provider changes land before consumers requiring them. Breaking V2 changes need a
new versioned route/contract and compatible migration, not a weakened drift check.
Rollback keeps the previous compatible provider/consumer pair.

## Ownership

`ACM-VIT/Weave/database/migrations` exclusively owns product PostgreSQL migrations.
Online ML does not connect to product PostgreSQL, perform product DDL, or mutate
product counters. It validates its Qdrant/vector/model contracts through existing
readiness checks. Historical SQL adapters are outside the production online graph;
`tests/test_online_architecture.py` guards that boundary. Product schema changes
must use backend migrations and its audit, not ML startup compatibility SQL.

Impressions belong to immutable backend telemetry, with serve/session/position
attribution. They are intentionally rejected by online feedback ingestion and
cannot update online preference vectors. Feedback versions accept exact decimal
strings from JavaScript, then validate/normalize to Python integers.

## Verification scope

The paired Node/Python test uses the real backend client, registered FastAPI routes,
Pydantic models, HTTP auth and admission. It replaces external encoder/storage
boundaries and proves wire compatibility, not live PostgreSQL/Qdrant durability or
ranking quality. Existing service/database integration suites cover those separate
properties. No schema export or contract test requires production credentials.

## Durable feedback event acknowledgements

A successful `POST /api/v2/feedback/batch` returns HTTP 202 with:

```json
{
  "accepted": 1,
  "duplicates": 1,
  "durable": true,
  "acknowledged_event_ids": [
    "00000000-0000-4000-8000-000000000001",
    "00000000-0000-4000-8000-000000000002"
  ]
}
```

The required ID list contains every request event, in request order, including
already accepted duplicates. IDs are canonical UUID strings, unique, and bounded
to 1–100 entries; `accepted + duplicates` equals its length. Input UUID case does
not change event identity. These IDs are returned only after the durable producer
successfully finishes. They acknowledge Redis acceptance, not completed preference
learning or Qdrant application. Backend product counters remain backend-owned.

On an event-ID/payload conflict (409), partial script/pipeline/storage failure
(503), capacity failure, or response timeout, there is **no acknowledgement list**.
Some events may already be durable; aggregate progress in error details is
operational context, never permission to acknowledge particular backend jobs.
Retry the appropriate unchanged events with their original IDs. Deduplication
makes a successful replay acknowledge the complete request without adding those
events again. No partial-success ACK feature or producer algorithm is introduced.

Deploy this additive provider response before a backend consumer that depends on
it. Old consumers can continue reading aggregate fields. Consumers supporting
rollback must treat missing IDs conservatively: a singleton legacy durable 202
can identify its one request, whereas a multi-event aggregate cannot identify a
subset. Consumer retries must retain immutable event identities. Keep the previous
compatible provider/consumer pair available; never remove durable pending work to
make a rollback appear healthy.

This delta is based on provider contract PR128 and paired with Weave PR193 for
Weave issue128. Merge/rollout must include that generated-contract prerequisite;
refresh the consumer's pinned provider commit and schema digest and rerun its
paired HTTP test. This document does not authorize deployment or issue closure.

Focused HTTP/producer tests are in `tests/test_feedback_ack_contract.py`.
`tests/test_feedback_ack_redis_integration.py` additionally runs a real loopback
HTTP server and the actual producer/Lua against an explicitly selected disposable
Redis database. It has no runtime-URL fallback and deletes only random test keys:

```bash
RUN_FEEDBACK_ACK_REDIS_INTEGRATION=1 \
FEEDBACK_ACK_TEST_REDIS_URL=redis://127.0.0.1:DISPOSABLE_PORT/15 \
uv run pytest tests/test_feedback_ack_redis_integration.py -q
```

The caller must provide a disposable plain loopback Redis instance; the URL guard
alone cannot establish ownership of a running database. The normal integration
suite remains separate from this opt-in acceptance check.
