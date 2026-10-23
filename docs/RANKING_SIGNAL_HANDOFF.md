# Ranking signal handoff (ML #150)

This audit covers the canonical V2 acquisition path, not every possible data
producer or a deployed release. The backend fixture selects commit
`1346b79e93915f879e34761ab41648d07dbe9ef7`; the ML change starts at
`7787af5a7933d8c047143b017c44ae90c72e2a92`. The reproducible test prints both
checked-out revisions and whether ML has uncommitted changes. Use its final clean
PR-head run for exact-pair evidence; a development run with a dirty worktree is
base-plus-patch evidence. Matching OpenAPI artifacts are checked before startup.

## Verified gap and correction

`RepositoryEnricher.to_osiris_payload` produces richer observations than
`repository_upsert_record` previously forwarded. With this exact backend pair,
the actual SQL/outbox path does **not** reconstruct the omitted commit history,
star deltas, contributor proxy, README/codebase ratio, GitHub timestamps or
precomputed push age. They reach ML as contract defaults instead. This is a
verified loss on the tested path, not proof that every producer loses them.

ML #155 removes the historical owner-presence contributor proxy described above.
The fixture supplies an explicit legacy caller count solely to verify compatible
transport; canonical enrichment does not collect or infer contributor population.

The delivery projection now forwards these already-supported optional fields,
plus discovery category/band. It preserves explicit zero and empty observations;
missing/None fields remain absent. No API schema or database migration changes.
The backend projection and strict ML DTO already accept these fields at the
selected commits. Compatibility with older deployed backends is not certified.

## Source, ownership, transformation and defaults

| Signal | Authoritative input / transformation | Canonical handoff / ML result | Missing-value meaning |
| --- | --- | --- | --- |
| GitHub IDs / backend repo UUID | GitHub numeric IDs; backend assigns product UUID | Actual SQL mapping and outbox UUID become Qdrant point identity | Never use owner/name as the point ID |
| stars, forks, open issues | GitHub current counts; enrichment normalizes counts | Backend stat snapshots; index and refresh DTOs; raw Qdrant counts | Existing count defaults collapse unavailable values to 0; no new observation guarantee |
| push timestamp / push age | GitHub pushed_at; enrichment computes elapsed whole days | UTC timestamp and explicit pushed_days_ago; used by code health/activity | Absent age remains 999, even with a known push timestamp; backend/ML do not infer it on this path |
| 3/7/30-day star deltas | Enrichment's sampled-star/event estimate | Optional signed fields in index/refresh sources; ML derives trend_velocity | Absent fields default to 0; observed zero is distinguishable by wire presence before DTO defaulting |
| mentionable_users_count | Deprecated optional caller field; canonical enrichment no longer infers it from owner presence | Preserved for wire/payload compatibility; excluded from activity scoring | Missing community evidence is **unknown**, with legacy zero placeholder; see [ML #155 policy](CONTRIBUTOR_SIGNAL_POLICY.md) |
| recent_commits | GraphQL commit timestamps collected by enrichment | Index-only source; Node canonicalizes timestamps to milliseconds; ML derives activity_score | Absent history defaults to []; raw timestamps are not separate final Qdrant payload fields |
| README/codebase ratio | Enrichment README length / max(repository KiB * 1024, 1) | Index-only source; participates in ML documentation score | Absent ratio defaults to 0.0; raw ratio is not a final Qdrant payload field |
| GitHub created_at | GitHub repository creation timestamp | Index-only source and final Qdrant created_at | Missing stays null; backend product-row created_at is a different clock and is never substituted |
| GitHub updated_at | GitHub repository metadata timestamp | Optional index/refresh source and Qdrant timestamp | Missing stays null; do not substitute product-row update time |
| README length / content | Canonical Markdown stored by backend | Node recomputes length; ML's existing StrictModel strips boundary whitespace and recomputes length again | Length differences are attributable normalization, not evidence of a lost README |
| doc_quality | ML documentation scorer over parsed README, ratio and metadata | Recomputed during vector payload construction | Not an upstream authoritative score to forward |
| code_health | ML issue/fork density, push age and language count | Recomputed during vector payload construction | Uses the explicit/defaulted inputs above |
| activity_score | ML commit history, or push-age/contributor fallback | Recomputed during vector payload construction | Empty history selects fallback; zero is not proof of observed inactivity |
| trend_velocity | ML blend of star-delta windows | Recomputed during vector payload construction | Signed-trend correction is separate ML #148 / PR #157 |
| skill_match_score | User/repository keyword intersection at ranker serving | Request-local ranker input | Deliberately not stored in repository vectors |
| observed_at / indexed_at | Acquisition observation clock / ML indexing clock | Observation retained in backend stat snapshots; indexed_at generated by ML | Neither is repository creation time |

Presence is inspected at the wire and `RepositorySource.model_fields_set` before
Pydantic defaults. The fixture distinguishes known positive, explicit zero/empty,
and absent observations there. Final numerical serving fields still use their
declared defaults and do not encode that distinction; this PR does not invent
new observed-quality flags. In particular, a synthetic explicit-zero probe at the
delivery boundary does not claim GitHub enrichment can observe zero contributors.

## Reproduce safely

Use the locked ML environment and an explicitly selected backend checkout with
installed locked Node dependencies. The PostgreSQL target must be a disposable
loopback database named `weave_scroll`, with the backend schema already prepared.
The fixture refuses other hosts/databases and production mode, binds only to
loopback, creates three unique synthetic repositories and cleans its own rows.
It never runs migrations or truncates shared tables.

```sh
RUN_RANKING_SIGNAL_HANDOFF=1 \
WEAVE_BACKEND_ROOT=/absolute/path/to/backend-checkout \
SIGNAL_DATABASE_URL=postgresql://postgres@127.0.0.1:55420/weave_scroll \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m pytest tests/test_ranking_signal_handoff.py -q -s
```

GitHub responses are synthetic inputs to the real enrichment transformation.
The actual Python ingestion client sends HTTP through Node's real internal
authentication, body parser, controller and canonical SQL/outbox code. The
persisted index job is validated by the real ML contract and route projection,
then its vector payload is upserted/retrieved in in-memory Qdrant. A deterministic
vector replaces encoder inference; this is not model-quality evidence. The ML
HTTP server, job lock/consumer lifecycle and production Qdrant are outside this
probe; their existing contract/runtime tests remain separate evidence.

## Repair and release boundary

Only observed source fields are added to an established allowlist. Existing
points are unchanged by installing this code. On subsequent ingestion, the
backend's existing content hash includes creation time, commit history and ratio;
changes enqueue an index job. Dynamic counts/deltas/push age use the existing
refresh path. Refresh-derived consistency remains ML #151 / PR #159 work.
Any bulk repair/reindex campaign, training, qualification or production rollout
requires its own authorization and acceptance evidence; none was performed here.
