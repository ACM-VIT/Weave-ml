# Candidate vector transport

The default `V2_CANDIDATE_VECTOR_MODE=deferred` retrieves payload-only candidates,
then fetches each retained vector once. `inline` is an explicit alternative for
required-heavy requests: validate the user profile first, request fresh atomic
vector/payload pairs on candidate channels, and avoid retained hydration only
when every retained candidate has a valid matching pair. Optional hybrid requests
remain repository-vector-free in either setting. Invalid mode values fail setup.

Inline validates every observation, including repeated IDs. Canonical identity,
eligibility, embedding identity/dimension/finite values and content/feature fields
retain the existing checks. Different versions, ranking payloads or vectors across
channels invalidate that ID for the request. The bounded request-local digest
observer never authorizes a vector from a shared cache or another request.
Metadata-only social candidates retain deferred fresh hydration; the social
provider's quotas/provenance remain intact. Snapshot validity is at the atomic
read, not a guarantee that a repository cannot change after that read.

This change depends on the social provider in PR #135 (`c3237ba`). Defaults,
production configuration and the authoritative discovery-cache gap in #53 are
unchanged. No product PostgreSQL access is added.

## Reproduction and scope

```bash
MIN_ELIGIBLE_REPOSITORIES=0 uv run pytest tests/test_candidate_vector_transport.py tests/test_inline_candidate_transport.py tests/test_inline_candidate_snapshots.py
uv run python -m scripts.benchmark_candidate_vectors
```

The actual retriever executes both modes, including ranking/assembly and all RPC
scheduling. Only datastore responses and heavy scores are deterministic fixtures.
Responses are serialized/decoded as JSON; bytes and thread CPU are measured.
The fixture has 20 points per channel: either 100 distinct IDs (low overlap) or
20 shared IDs (high overlap). No live Qdrant, trained model qualification,
production capacity or recommendation quality is measured. The JSON records the
exact retriever source SHA256. Three samples alternate mode order; ranges remain
in the JSON. Measurements below were taken 23 October 2026, source SHA256
`50cb915bd4e0961284f08b69db0067b139cad57624aa9a9c17ff082ea720ec6f`.

| Fixture measurement | Validated inline | Deferred |
| --- | ---: | ---: |
| Low-overlap bytes | 868,170 | 947,937 |
| High-overlap bytes | 868,154 | 254,985 |
| Low / 0 ms RTT median request | 101.14 ms | 94.22 ms |
| High / 0 ms RTT median request | 84.54 ms | 27.55 ms |
| Low / 50 ms RTT median request | 230.27 ms | 250.95 ms |
| High / 50 ms RTT median request | 212.25 ms | 174.00 ms |
| Low / 0 ms JSON decode CPU | 20.00 ms | 18.29 ms |
| High / 0 ms JSON decode CPU | 18.10 ms | 4.92 ms |
| Transport calls | 6 | 7 |

The deferred high-overlap fixture saves about 71% of response bytes. Low overlap
repeats retained metadata and transfers about 9% more bytes. Inline avoids one
hydration call but loses profile/discovery overlap and validates every pair:
high RTT alone does not establish a win. Its first low-overlap zero-RTT sample
was noisy (90.62–293.23 ms range). Select a mode only from equivalent workload
measurements; no adaptive RTT guess or universal speedup is claimed.

Tests cover retained/truncated vector bounds, static ranking parity between the
actual modes, deferred deletion/version/eligibility/shape rejection, inline
content/feature/topic/vector conflicts, shape/nonfinite rejection, missing-profile
failure before vector reads, hybrid vector-free reads, social hydration and
absence of cross-request pair reuse. The candidate floor remains 50. Existing
strict-heavy failures and default deferred freshness checks remain intact.
