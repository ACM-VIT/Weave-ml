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
new content. Duplicate/current deliveries replay that durable artifact only when
its internal source/version provenance matches. If a
same-content legacy point has no current artifact, a summary-only CAS backfills
it without replacing its vector or independently refreshed feature state.
The fallback has a distinct deterministic model identity and stays safe to
replay during provider outages. A later current-content delivery or explicit
summary-backfill can retry generation when a provider and deadline budget are
available. Exact duplicate jobs retain their durable response.

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
generated artifact with matching source/model/prompt/format/validation versions, while the backend
persists and activates it before completing the outbox delivery. A
`description_fallback` remains eligible for a later provider-backed backfill.

## Bounded source analysis

Without a configured provider, generation returns the same description-only
fallback before reading or parsing any README source. Canonical Markdown and
the fallback's content/version rules are unchanged.

Provider input has an independent preprocessing budget, not just the
`SUMMARY_INPUT_MAX_CHARS` transport limit:

- Each complete evidence variant is at most 64,000 Unicode codepoints and
  256,000 UTF-8 bytes. Length is checked before encoding or stripping input.
- At most 80 extracted paragraphs are joined, with a combined 64,000-codepoint
  bound, including separators. Individual paragraphs are not prefix-truncated.
- Source collection and the Markdown scan share a 512,000-character work budget.
  The scan visits at most 4,096 lines, using one-line lookahead rather than a
  full-file split. Metadata cleanup retains its existing independent bounds.
- The structural scan retains only high-signal/neutral line offsets, bounded by
  the line ceiling. One final selector cleans lines once in metadata, high-signal,
  then neutral order and charges only clipped text actually appended. Retained
  cleaned text is capped by the prompt budget, so a long unspaced line cannot
  consume a separate excerpt allowance and hide later useful lines. Heading,
  setext, nested excluded-section, and fenced-code handling are preserved.

Prompt clipping never clips the copy-check evidence. Complete, in-budget
variants, including excluded sections and text beyond the prompt, remain in
copy detection. `SummarySource.analysis_truncated` explicitly reports excerpt
clipping or a limit-exceeded analysis; `analysis_limit` distinguishes the latter.
An exceeded evidence/analysis budget skips the provider and uses the metadata
fallback, rather than generating or validating against only a source prefix.
Direct validation also rejects oversized comparison strings. Generated replay
with repository context is not current when its source analysis exceeds a
budget; it must take the normal regeneration/fallback path. Existing generated
replay still performs safety/copy revalidation even if no provider is enabled.
Durable fallback replay does not parse the README.

## Reusable validation and artifacts

Each pipeline owns an LRU capped at 32 entries and 16 MiB of accounted retained
Python objects, including tokens, ngram tuples/sets, prompt evidence and artifacts.
Entries larger than the byte limit are not retained. No reuse crosses processes;
normal runtime cleanup releases the pipeline and its cache. The separate flight
map admits at most 32 distinct generation jobs. Saturation uses the same safe
metadata fallback. Duplicate callers share active work without new provider calls;
a wait ends before the caller's completion reserve or after one configured
request timeout when no deadline was supplied. Wait expiry does not cancel the
owner. Exceptions always release the flight slot.

A SHA-256 key covers exact bounded source fields (including full evidence beyond
the prompt and object README variants), repository identity/content hash, model,
provider endpoint, prompt, format, input/output settings, source-analysis version
and validation version. Oversized/unsupported key inputs bypass reuse. Bump
`SOURCE_ANALYSIS_VERSION` or `SUMMARY_VALIDATION_VERSION` in
`summarization/reuse.py` whenever the corresponding policy changes.

Complete comparison variants are tokenized once and retain an eight-word index.
Highlights, the summary and a repair all share that immutable analysis. Every
original SequenceMatcher window, including the final off-grid window, remains
eligible. A multiset token-intersection bound skips only windows that cannot
reach the unchanged 0.84 ratio. No source suffix or excluded section is omitted.

Only successful generated artifacts are retained; callers receive independent
copies. Transient fallbacks share an active flight but are not saved as generated
cache results, so the next call can regenerate. Qdrant additionally persists
`card_summary_reuse_key`, covered by the existing artifact fingerprint/CAS. This
internal provenance is absent from the public backend artifact JSON. A restarted
pipeline rechecks both the key and safety against complete evidence. A legacy
generated artifact without provenance takes summary-only regeneration/backfill;
fallback replay remains independent of README parsing. API replay uses the same
derived paragraph projection as generation and computes it once per job.

Run the network-free local profile with:

```bash
python3 -m scripts.profile_summary_reuse --baseline-ref upsteam/main --repeats 20
```

On local Python 3.14, a 62,061-character synthetic README took 0.234 CPU seconds
for 20 original validations versus 0.063 seconds using prepared analysis (3.7x).
Preparing analysis 20 times separately took 0.050 seconds. Twenty unchanged
inputs made one stub-provider call and 19 artifact hits (95%); twenty changed
inputs made 20 calls and no artifact hits. With the profile's four-entry limit,
unchanged inputs retained about 0.6 MiB of accounted objects and changed inputs
retained about 2.4 MiB. Tracemalloc peak allocations were about 1.4 and 3.8 MiB,
respectively. These are local synthetic measurements, not production throughput
or provider latency claims; the script reports CPU, calls, hits and memory anew.

## Timeout budget

The existing default provider timeout remains five seconds and the shared job
deadline reserves ten seconds for completion. At the configurable upper bound,
one provider generation can use three attempts at up to 30 seconds each plus one- and two-second backoffs. One invalid-output
repair may repeat that budget, for a conservative worst case of about 186
seconds plus local request waits capped at two seconds. A longer local rate-limit
wait or provider `Retry-After` fails to the safe description artifact instead of
overrunning the lease. The backend repository-index transport therefore uses a
240-second request timeout and a five-minute outbox lease/reconciliation window.
Keep those values above the configured summary budget; ordinary recommendation
reads retain their shorter timeout. Qdrant replay makes a transport retry safe,
but undersizing the timeout would cause needless aborts and lock contention.
