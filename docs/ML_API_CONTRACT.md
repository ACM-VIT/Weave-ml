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
