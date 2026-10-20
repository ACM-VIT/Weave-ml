# Phase 4: repository semantic search

The Expo app calls authenticated Node only. Node calls this authenticated internal ML endpoint. Qdrant owns vectors; ML never hydrates PostgreSQL or returns GitHub URLs, repository text, profiles or cards.

`POST /api/v2/search/repositories` requires the configured internal-secret header (default `X-Internal-Secret`). Request:

```json
{
  "schema_version": 2,
  "search_id": "8ce37bc1-9da5-4cff-a014-68008c626a65",
  "query": "image understanding",
  "limit": 100,
  "exclude_repo_ids": [],
  "filters": { "languages": ["Python"], "topics": ["computer-vision"] }
}
```

Query: trimmed 2–512 characters. Limit: strict integer 1–100. Exclusions: at most 500 unique UUIDs. Filters: at most 10 language names (1–64 characters) and 10 lowercase topic slugs (1–100 characters). Languages match the primary indexed language exactly; each filter is OR within its array, AND across filter kinds. Unknown fields and schema versions are rejected.

Response:

```json
{
  "schema_version": 2,
  "search_id": "8ce37bc1-9da5-4cff-a014-68008c626a65",
  "embedding_version": "repo-embedding-v2",
  "items": [{ "repo_id": "c4618175-d13c-42fb-83ac-d3267b294bf3", "score": 0.81 }]
}
```

Scores are finite cosine similarities in [-1,1]. Floating point overshoot within 1e-6 of either endpoint is clamped; larger excursions and nonfinite scores are discarded. IDs are unique canonical backend UUIDs, sorted by descending score with UUID tie-breaking. The search uses the same process-shared frozen encoder/revision as repository ingestion, without constructing a summary provider. It uses a reserved bounded search executor, a four-second default API deadline, and the existing V2 serving eligibility/model/revision/dimension/content/feature filters before the Qdrant limit. Only the current embedding version participates in one response. Payload projections exclude document text and vectors. A topic keyword payload index is included in the existing Qdrant schema/bootstrap contract; existing collections need the normal schema/index reconciliation before serving topic-filtered search with strict index enforcement.

Search admission is configured with `V2_SEARCH_EXECUTOR_WORKERS` (default 1), `V2_SEARCH_MAX_OUTSTANDING` (default 4), and `V2_SEARCH_TIMEOUT_SECONDS` (default 4). This queue is separate from repository embedding and synchronous summary-provider waits; encoder access still shares `EMBEDDING_MAX_CONCURRENCY`. Cancelled/timed-out work keeps its admission slot until the underlying worker exits. Search requests have their own fixed metrics route label and executor saturation metrics.

Missing/incorrect authentication returns 401 (503 when the service secret is unconfigured); validation returns 422; deadline/capacity returns 503. Node deliberately uses a shorter 1.8-second timeout and a lexical fallback, independently validates IDs/scores/envelopes, and hydrates only active canonical PostgreSQL records. Search IDs correlate requests and responses, not authorization.

Run the fast contract/local-Qdrant tests with `uv run --locked --python 3.12 pytest tests/test_repository_search.py`. For the real encoder conceptual-query check, cache the configured immutable model revision, then set `RUN_LOCAL_SEMANTIC_MODEL_TEST=1 HF_HUB_OFFLINE=1`. Tests use in-memory Qdrant and no PostgreSQL/Redis/production service.
