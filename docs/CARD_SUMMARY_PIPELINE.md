# V2 repository card-summary pipeline

This pipeline keeps three artifacts separate:

1. `readme` is the canonical, unchanged GitHub Markdown. Acquisition also records
   `readme_source_path`, `readme_default_branch`, and `readme_base_url` so the app
   can resolve safe relative media in the full README view.
2. `card_summary` is a two- or three-sentence discovery artifact. Generated
   output is limited to 360 characters and never falls back to README text.
3. Derived text is a cleaned copy used for embedding and bounded provider input.
   It is not canonical content and is not a serving fallback.

## Versioned contract

- Prompt: `repo-card-summary-v1`
- Generated model: configured by `SUMMARY_MODEL_ID`
- Description-only fallback model identity: `repository-description-fallback-v1`
- Format: `repo-card-summary-json-v1`
- Provider response keys: `summary` and optional `highlights` only
- Embed response artifact:

```json
{
  "schema_version": 2,
  "accepted": true,
  "status": "applied",
  "job_id": "00000000-0000-4000-8000-000000000000",
  "repo_id": "00000000-0000-4000-8000-000000000000",
  "content_version": 1,
  "embedding_version": "repo-embedding-v2",
  "card_summary": {
    "summary": "Two short factual sentences.",
    "highlights": [],
    "prompt_version": "repo-card-summary-v1",
    "model_version": "meta-llama/llama-3.3-70b-instruct",
    "format_version": "repo-card-summary-json-v1",
    "source": "generated"
  }
}
```

The artifact is written into the repository's Qdrant payload in the same CAS as
new content. Duplicate/current deliveries replay that durable artifact. If a
same-content legacy point has no current artifact, a summary-only CAS backfills
it without replacing its vector or independently refreshed feature state.
The fallback has a distinct deterministic model identity and is considered
current only while no provider is configured. When a provider is enabled, a
later delivery or summary-backfill retries generation and can activate the
provider-backed tuple without pretending the fallback came from that model.

## Historical comparison

The historical `feature/gemma-readme-markdown` work was inspected at local
`6ed61fb`, cached remote `be503fb`, and commits `2ff71cd`, `0a5faec`, `a5e0547`,
and `c845278`. Its Gemma/Groq/OpenRouter prompts intentionally rewrote the full
README, preserved installation commands and code, and allowed long output. Its
legacy `readme_summary` was up to 5,000 characters of cleaned README. Those
semantics are unsuitable for cards.

Phase 3 retains only the useful operational patterns: low temperature, explicit
no-hallucination instructions, bounded rate limiting, 429 retries, outer-fence
cleanup, one repair attempt, and graceful fallback. Golden historical/current/new
examples live in `tests/fixtures/repository_summary_golden.json`.

## Safe dry-run reindex

The canonical backend owns repository content and the durable outbox, so the
backfill command remains backend-owned rather than giving the online ML service
database access. Against fixtures or an isolated local database only:

```bash
cd /Users/medhanshadhlakha/weave-product-fixes/backend
npm run reindex:ml -- --summary-backfill --campaign=phase3-card-summary-v1
```

Dry-run is the default. It reports `would_queue` and uses the idempotency key
`repo_index:<repoId>:<contentVersion>:campaign:phase3-card-summary-v1`. After
reviewing the isolated database and report, execution must be requested explicitly:

```bash
cd /Users/medhanshadhlakha/weave-product-fixes/backend
npm run reindex:ml -- --summary-backfill --campaign=phase3-card-summary-v1 --execute
```

Do not run this command against production as part of Phase 3.

Processing unchanged content is idempotent: the ML endpoint replays an existing
generated artifact with matching prompt/model/format versions, while the backend
persists and activates it before completing the outbox delivery. A
`description_fallback` remains eligible for a later provider-backed backfill.

## Timeout budget

With the production bounds, one provider generation can use three attempts at
up to 30 seconds each plus one- and two-second backoffs. One invalid-output
repair may repeat that budget, for a conservative worst case of about 186
seconds plus local request waits capped at two seconds. A longer local rate-limit
wait or provider `Retry-After` fails to the safe description artifact instead of
overrunning the lease. The backend repository-index transport therefore uses a
240-second request timeout and a five-minute outbox lease/reconciliation window.
Keep those values above the configured summary budget; ordinary recommendation
reads retain their shorter timeout. Qdrant replay makes a transport retry safe,
but undersizing the timeout would cause needless aborts and lock contention.
