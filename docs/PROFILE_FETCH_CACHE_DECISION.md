# User-profile fetch: cache decision preparation

Issue [#63](https://github.com/ACM-VIT/Weave-ml/issues/63) requires a human
cost/freshness decision before cache implementation. **No decision is recorded,
no profile cache is installed, and current serving remains uncached.** This
artifact prepares that decision; it does not complete the issue.

## Measured boundary

`scripts/measure_profile_fetch.py` measures the actual
`QdrantV2Retriever._user_profile` path: bounded retrieval admission/worker,
HTTP Qdrant retrieval, SDK decoding, and vector/payload compatibility validation.
Every request still reads Qdrant. It loads no encoder/ranker and does not measure
full recommendations, network deployment topology, or user-visible latency.
The fixture creates and finally removes its own Qdrant v1.18.2 container using
an ephemeral loopback port. There is no target-URL option, configured credential
reuse, production dataset, or shared-service mutation.

The existing payload projection includes skills, technology stack, interests,
topics, embedding dimension/model/revision. It excludes `profile_version`,
`last_feedback_version`, and `feedback_state_revision`. A fixture 16 KiB ledger
is also excluded. **These measurements do not include future exact-version field
wire/read overhead.** Existing source is unchanged. Model compatibility remains
checked; missing/incompatible profiles remain unavailable.

The HTTP span ends after decoded response-body reading; total wall time also
includes worker admission, SDK parsing and validation. Byte counts are decoded
JSON body bytes, not TCP/TLS or total network traffic. Startup/import/collection
creation/fixture writes are outside the measured reads. Runs are sequential;
there is no contention, multi-process invalidation or realistic arrival trace.

## Controlled workload and results

Measured locally on 23 October 2026 against source commit
`47e71e2f99e0282ced9e1bc83a8e1046507191a2` (PR #145 prerequisite).
The measured retriever SHA256 was
`42a8ce81920503b7a606fd72c1d6c2db91b075c0bc1d671df19f3e38863d914b`;
the measured driver SHA256 was
`3daa46b17a8a2c3d9ec501802756f618eddbf0d11f2e95e7dc1b4fdfc45a879d`.
Qdrant image ID:
`sha256:75eab8c4ba42096724fdcfde8b4de0b5713d529dde32f285a1f86fdcb2c9e50c`.
Model compatibility was MiniLM revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`; no model was loaded.

| Synthetic scenario | Wall p50 / p95 / p99 ms | HTTP body p50 / p95 ms | Content repeats | Free-version-oracle repeats |
| --- | --- | --- | --- | --- |
| 128 unique users | 4.48 / 5.89 / 6.54 | 3.09 / 3.99 | 0/128 (0%) | 0/128 (0%) |
| One unchanged user | 4.48 / 6.09 / 6.89 | 3.09 / 4.24 | 127/128 (99.2%) | 127/128 (99.2%) |
| 32 users, round robin | 4.24 / 5.65 / 6.03 | 2.83 / 3.92 | 96/128 (75%) | 96/128 (75%) |
| One user, vector/state change every 8 reads | 4.54 / 6.43 / 6.81 | 3.08 / 4.26 | 112/128 (87.5%) | 112/128 (87.5%) |
| One user, version-only change every 8 reads | 4.40 / 5.78 / 6.31 | 3.01 / 3.87 | 127/128 (99.2%) | 112/128 (87.5%) |

All 640 reads returned valid profiles and issued exactly one HTTP retrieval each.
Decoded JSON body p50/p95/p99/max were all 5,004 bytes in each scenario;
total wall maxima across scenarios were 6.44–8.56 ms. The owned container was removed after the run.
This is one sequential cohort in the table's order, without randomized order,
repeated-run uncertainty estimates, injected WAN latency or production load.
No eliminated-read/speedup measurement was performed.

For each scenario, users have deterministic synthetic 384-dimensional compatible
vectors. Update scenarios perform synchronous fixture writes every eight reads;
they do not exercise the durable asynchronous feedback/onboarding pipeline.
The version-only scenario changes applied-state revision fields without changing
the returned projected content. It demonstrates why equal returned content is
insufficient to authorize a versioned cache entry.

The analyzer retains only bounded per-user hashes and version tuples after each
read. Its two counters mean:

- Content repeat: the same user's returned validated vector and projected payload
  equal its preceding observation. Equal profiles belonging to different users
  never share an opportunity.
- Free-version-oracle repeat: content repeats and the fixture writer's known
  positive profile/nonnegative feedback/state revisions also match. This oracle
  is free only in the experiment; the current reader does not return it. Missing,
  malformed or changed revisions cannot authorize a repeat.

Neither counter is an observed cache hit rate, a production expected hit rate,
or a deployable estimate of eliminated reads. A cache still needs an authoritative
way to learn version changes across processes; querying Qdrant merely to obtain
the key may preserve the read being optimized. Synthetic request ordering gives
bounds/examples rather than a representative traffic distribution. No maximum
staleness or failure policy can be inferred from local latency.

## Decision for the human owner

Choose **defer caching and retain uncached reads**, or **proceed with a separately
reviewed bounded prototype** after judging the measured cost against complexity.
If proceeding, provide the maximum permitted delay from *applied Qdrant*
feedback/onboarding change to its use in recommendations, and the required
behavior when the version/invalidation authority is unavailable. Backend feedback
acceptance is not proof that ordered learning has been applied. There is no
implicit approval of TTL-only reuse, stale serving, or a fail-open policy.

For a defensible expected hit rate, supply an authorized bounded aggregate trace
with per-user pseudonymous request timing, applied profile/feedback revision
changes, and process distribution, or accept that expected production hit rate
remains unknown. No raw user vectors, profile text or credentials are needed.

## Acceptance retained after a go decision

A future implementation must demonstrate user isolation, compatible vector and
version keys, onboarding/feedback invalidation, multi-process writers/readers,
concurrent updates, expiry, stale/missing versions, authority outages, and measured
feedback-effect delay under the chosen policy. Preserve writer CAS/fencing,
ordered durable feedback, and strict profile/ranker behavior. The 13 focused checks pass and
cover measurement identity/projection/compatibility/oracle correctness only;
they do not claim that those cache acceptance cases pass.

Reproduce from the ML checkout with the project's locked environment:

```sh
uv run --locked pytest -q tests/test_profile_fetch_measurement.py
uv run --locked python scripts/measure_profile_fetch.py --reads 128
```

The CLI requires Docker and an already cached `qdrant/qdrant:v1.18.2` image;
`--pull=never` prevents implicit downloads. JSON evidence belongs in a private
runtime directory, not in serving state or tracked generated artifacts.
