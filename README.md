# gh-social-ml

Machine-learning and data-pipeline services for [gh-social](https://github.com/SUBHRO71/gh-social). This repository discovers and enriches GitHub repositories, builds repository and user embeddings, retrieves and ranks recommendations, assembles feed slices, and processes recommendation feedback.

## What Is Here

- `acquisition/` and `ingestion/`: GitHub discovery, enrichment, classification, and quality filtering.
- `embedding/`: SentenceTransformer repository embeddings and Qdrant indexing.
- `retrieval/v2_retriever.py`: canonical candidate retrieval, heavy ranking, and feed shaping.
- `inference/`: learned ranking assets and final feed assembly.
- `feedback/`: feedback queue producer, consumer, and embedding updates.
- `trending/`: scheduled GitHub Trending ingestion.
- `api/main.py`: the integrated internal FastAPI service.
- `app.py`: an alias for the canonical V2 ASGI application.
- `Ranking_Training_module/`: training and synthetic-data utilities for the heavy ranker.
- `tests/`: unit, integration, and benchmark coverage.

## Requirements

- Python 3.10 or newer
- Qdrant for repository and user vectors
- Redis for durable v2 feedback streaming
- PostgreSQL only for offline acquisition and evaluation tooling
- A GitHub personal access token for acquisition and trending ingestion

Local development downloads the configured SentenceTransformer on first use.
The production image instead bakes an immutable model revision and starts with
Hugging Face/Transformers offline mode enabled; runtime downloads are refused.

## Quick Start

```bash
git clone https://github.com/SUBHRO71/gh-social-ml.git
cd gh-social-ml

# Install uv first: https://docs.astral.sh/uv/getting-started/installation/
uv sync --locked
cp .env.example .env
uv run python main.py --validate-config
```

On PowerShell, use `Copy-Item .env.example .env` instead of `cp` if `cp` is not available.

Configure at least the services needed for the command you are running:

```dotenv
GITHUB_TOKEN=github_pat_...
DATABASE_URL=postgresql://user:password@localhost:5432/gh_social
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=osiris_research_corpus_v2_20260722_r1
REDIS_URL=redis://localhost:6379/0
# Production requires exactly 64 lowercase hexadecimal characters:
# openssl rand -hex 32
INTERNAL_API_SECRET=replace-with-64-lowercase-hex-characters
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_MODEL_REVISION=c9745ed1d9f207416be6d2e6f8de32d1f16199bf
```

Do not commit `.env` or production credentials.

## Run The APIs

Start the integrated API:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Run the ordered V2 feedback consumer as a separate process:

```bash
python -m feedback.v2
```

Its OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

| Method | Path | Purpose | Authentication |
| --- | --- | --- | --- |
| `GET` | `/api/v2/health` | ML, Qdrant, and feedback-stream readiness | `X-Internal-Secret` |
| `GET` | `/api/v2/metrics` | Fixed-cardinality Prometheus metrics | `X-Internal-Secret` |
| `POST` | `/api/v2/recommendations/generate` | Generate canonical `repo_id` recommendations | `X-Internal-Secret` |
| `POST` | `/api/v2/feedback/batch` | Durably accept ordered interaction feedback | `X-Internal-Secret` |
| `POST` | `/api/v2/users/onboard` | Create or update a versioned user profile vector | `X-Internal-Secret` |
| `POST` | `/api/v2/repositories/embed` | Embed versioned canonical repository content | `X-Internal-Secret` |
| `POST` | `/api/v2/repositories/refresh` | Refresh versioned repository features | `X-Internal-Secret` |

The protected endpoints fail closed when `INTERNAL_API_SECRET` is missing.
Production requires exactly 64 lowercase hexadecimal characters generated with
`openssl rand -hex 32`. Backend requests must send that exact value in the
`X-Internal-Secret` header. The application exposes only `/api/v2` product
endpoints; feed assembly is an internal stage of recommendation generation.

## Run The Pipelines

Discover, enrich, filter, and index repositories:

```bash
python main.py --limit 150 --batch-size 15 --workers 4
```

Refresh GitHub Trending once or run the scheduler:

```bash
python trending_service.py --once
python trending_service.py --scheduled
```

Additional evaluation, seeding, onboarding, and integration utilities are in `scripts/`. Run a script with `--help` before using it against shared infrastructure.

Repository acquisition and trending synchronization are offline worker jobs, not request-path services. See the [offline pipeline operations runbook](docs/OFFLINE_PIPELINE_RUNBOOK.md) for configuration validation, bounded runs, safe shutdown, checkpoint resume, and diagnosis.

Production configuration, feedback replay, alerts, deployment, rollback, and
rollout gates are in the [V2 production operations runbook](docs/PRODUCTION_RUNBOOK.md).

## Schema Cutover Contract

The V2 boundary keeps backend-owned product state in `app` and immutable delivery telemetry in `telemetry`. Online ML accepts versioned backend-owned jobs, uses Redis and Qdrant for derived state, and sends only bounded derived repository material to the scoped card-summary provider.

New work must follow these boundaries:

- Identify every recommendation and feedback signal by canonical `repo_id`. Treat GitHub `owner/name` only as transitional ingestion input.
- Let the gh-social backend own PostgreSQL product mutations and telemetry transactions. The ML service must not receive direct production database credentials after cutover.
- Store typed recommendation entries in Redis with repository ID, score, source, model version, and summary ID.
- Record only the feed slice actually served, with `serve_id`, `session_id`, ordered positions, source, and model version.
- Preserve the original interaction event type. Do not reduce events to a calculated feedback score before they are stored.
- Consume backend-created ML outbox records with idempotency keys and retry semantics.
- Ignore quick passive swipes. The coordinated client/backend contract records an impression after one second of visibility and dwell after three seconds.
- Do not recreate a standalone `user_feedback` table. Current state belongs in reactions and saves; immutable events and derived per-user/repository rollups belong in telemetry.

Deploy the ML service together with the matching gh-social backend and Expo V2 contracts. Run the production smoke test before shifting traffic.

## Production Processes

CI builds one immutable `gh-social-ml` image and runs it under two independent
systemd units:

```text
gh-social-ml.service           uvicorn api.main:app
gh-social-ml-feedback.service  python -m feedback.v2
```

The API durably accepts V2 feedback into a bounded Redis stream; only the
dedicated feedback process updates Qdrant in strict per-user
`feedback_version` order. Retryable gaps/dependency failures remain pending,
terminal events are preserved in a bounded DLQ, and replay is dry-run by
default. V2 readiness fails closed when the consumer heartbeat, stream health,
collection contracts, pinned embedding identity, model warmup, or eligible
corpus minimum is unhealthy. Repository eligibility also requires the indexed
`repository-vector-v2` serving marker, stamped only when the ML store validates
and atomically upserts the repository vector and payload.

Production starts from [`deploy/production.env.example`](deploy/production.env.example)
and keeps the completed file root-owned at `/etc/gh-social/ml.env` with mode
`600`. It must contain no database, GitHub, or legacy README-rewrite provider
credentials. The only online LLM credential is the scoped `SUMMARY_API_KEY`
required by the bounded card-summary path. Run the network-free preflight
before a manual restart:

Production Redis configuration is transport-aware: use an ACL-authenticated
`redis://` URL for a private, network-isolated endpoint, or the equivalent
`rediss://` URL when the endpoint provides TLS. The URL scheme selects the
transport; both modes require the dedicated ACL username and password.

```bash
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

The image uses a process-scoped warmed encoder, bounded embedding executor, and
resource-conscious BLAS/PyTorch defaults. One locked environment is currently
installed into the shared API/feedback image because runtime and offline code
still share project modules and one `uv.lock`; the online import-graph tests
enforce that database code is not imported, and the deployment validator
prevents database credentials from entering the containers.

Production configuration uses `V2_HEAVY_RANKER_ENABLED=true`,
`V2_HEAVY_RANKER_REQUIRED=true`, and traffic percent `100`. In that mode every
request with an onboarding profile traverses V2 semantic/discovery retrieval,
the heavy ranker, social adjustment, and feed assembly. A missing user vector,
unavailable model, or invalid heavy-ranker output fails the request with a
retryable dependency error; it is never served by the hybrid ranking fallback.
The checked-in heavy ranker was trained on synthetic interactions and carries
an explicit manual production qualification for the initial V2 launch. The
manifest preserves that provenance and exposes the automatic qualification
warnings; replace it with a telemetry-trained artifact once real interactions
are available.

Main-branch deployments use a non-cancelling concurrency group and the GitHub
`production` environment. Repository administrators must configure required
reviewers and branch restrictions for that environment. After restart, the
workflow verifies authenticated health, feedback heartbeat, collection/model
contracts, corpus readiness, and one deterministic recommendation for a
dedicated smoke user. A failed forward deploy restores the snapshotted prior
image, helpers, and units, then verifies the rollback with the same checks.

## Tests

Run the default suite:

```bash
pytest
```

Useful focused commands:

```bash
pytest -m unit
pytest tests/test_feedback.py
pytest tests/test_production_config.py tests/test_production_smoke.py
pytest --cov=. --cov-report=term-missing
```

Integration tests require their declared external services and may use
`TEST_DATABASE_URL`. CI provisions PostgreSQL, Redis, and Qdrant and exercises
the actual V2 feedback flow from the authenticated API batch endpoint through
the Redis stream and ordered consumer to the Qdrant user-vector update. Marker
definitions and discovery settings live in `pytest.ini`.

The hardened V2 write path requires Qdrant 1.18.0 or newer for conditional
insert/update fencing; readiness rejects older servers.

## Contributing

Create a focused branch, add or update tests for behavior changes, and run the relevant test groups before opening a pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

This project is licensed under the [MIT License](LICENSE).
