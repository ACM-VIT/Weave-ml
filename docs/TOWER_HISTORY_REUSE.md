# README tower reuse on local Git history

The existing bounded tower cache already reuses unchanged semantic inputs while
rebuilding current payloads and independently calling the summary pipeline. This
acceptance slice replays genuine immutable README Git objects with the current
reserved v2 README projection and compares every reused tower with a cold encode.
It changes no embedding/runtime/storage behavior.

## Exact sample and results, 23 October 2026

| Local repository / cohort | Frozen commit | Snapshots | Reused README towers | Chunk encodes avoided / baseline |
| --- | --- | ---: | ---: | ---: |
| Weave, consecutive first-parent commits | `25a44523b96caa2e57d7f45b34cd430f5888a6f6` | 40 | 39 (97.5%) | 39 / 40 |
| Weave-ml, consecutive first-parent commits | `7a7091cc45fda7884cfd21e582c1e7bd9f69d509` | 40 | 33 (82.5%) | 172 / 209 |
| Weave-ml, README-changing first-parent commits | same ML commit | 15 | 1 (6.67%) | 1 / 57 |

The first two cohorts use a deterministic text-sensitive control encoder for
cache/counting equivalence. The third also ran with the actual cached pinned
MiniLM revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, CPU, one thread,
offline loading. Its maximum absolute cold-versus-reused tower difference was
zero; 14 entries occupied 43,008 vector bytes. The combined cold-reference plus
warm replay took 14.11 seconds, which is verification work, not a pipeline speedup.

Every result records commit, blob, committed timestamp, path, byte size, current
input fingerprint, chunk count and reuse. The report excludes raw Markdown.
A Git commit timestamp is provenance, not an acquisition/ingestion timestamp.
No new checkout or network access is used. The CLI bounds history to 200 commits
and each README to 1 MB; absent, non-UTF8 or oversized inputs are reported skipped.

README changes are genuine versioned source updates. They are not complete
historical repository upsert payloads. Historical GitHub topics, stars,
descriptions, backend content versions, actual ingestion-job selection and deployed
model/cache lifetime remain unknown. Code-only commits may never cause an index
job. Consequently the consecutive-commit ratios are not production skip ratios;
the README-changing cohort is a deliberately separate and much lower estimate.
Metadata/topic tower hits and summary reuse cannot be inferred from this sample.
The measured update-only reuse does not justify adding persistent storage or a
more complex cache; this change retains the existing bounded in-memory cache.

## Reproduction

```bash
uv run python -m scripts.benchmark_tower_history --repo ../Weave --ref 25a44523b96caa2e57d7f45b34cd430f5888a6f6 --limit 40
uv run python -m scripts.benchmark_tower_history --repo . --ref 7a7091cc45fda7884cfd21e582c1e7bd9f69d509 --limit 40
uv run python -m scripts.benchmark_tower_history --repo . --ref 7a7091cc45fda7884cfd21e582c1e7bd9f69d509 --limit 15 --changes-only --cached-real-model
MIN_ELIGIBLE_REPOSITORIES=0 uv run pytest tests/test_tower_history.py tests/test_embedding_validation.py tests/test_v2_qdrant_fencing.py
```

The model flag requires the already-cached pinned artifact and never downloads it.
Without that flag the control encoder measures cache equivalence/counting only.
JSON contains exact provenance and explicitly unknown historical fields. Inputs
are evaluated under today's v2 projection, not a claim about historical model use.

Nine new tests verify immutable Git-object provenance, missing/oversized/non-UTF8 README reporting,
change-only cohort selection, path bounds, bounded-LRU numerical equivalence and
all-tower-hit storage fencing. The integrated test generates three current content
versions with six tower hits and three independent summary invocations, then uses
actual local Qdrant conditional writes: version 3 wins and a stale version 2 holder
cannot replace its content/job/hash/eligibility. Existing numeric-only and
one-tower-change tests remain part of the 72 passing related checks.

This is scoped actual README-history evidence. A representative full-payload
production reuse rate still requires authorized point-in-time upsert history;
unknown fields must not be filled with current GitHub values. No deployment,
production load measurement or recommendation-quality claim is made.
