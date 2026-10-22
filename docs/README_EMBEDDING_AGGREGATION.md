# Token-aware README aggregation

Issue #27 adds the explicitly selected embedding version
`repo-embedding-v3-token-sections-v1`. Set `REPOSITORY_EMBEDDING_VERSION` to that
exact value to select it. `repo-embedding-v2` and existing custom legacy labels
retain character chunking and equal mean pooling. Keeping the default and
production example on v2 avoids silently changing existing vectors or claiming
that the shipped v2 heavy ranker is qualified for a different representation.
There is no separate strategy switch that can stamp the new computation as v2.

## Representation

The embedding path prefers original Markdown from acquisition or the backend
`readme` field. Cleaned text and extracted paragraphs remain supported when
Markdown is unavailable. Canonical source Markdown is never modified.

The parser recognizes ATX and Setext headings, nested section inheritance, and
backtick/tilde fences. It omits fenced code and entire License, Sponsors,
Changelog, Contributors, Contributing, Acknowledgements, table-of-contents and
API/generated-reference subtrees. This is a deterministic English-heading
heuristic; unknown/localized headings remain neutral. License-related project
subject matter is preserved when its heading is not a legal-section label.

Each retained section is cleaned and split using the loaded encoder's own fast
tokenizer and `max_seq_length`. The content budget reserves the tokenizer's
special tokens. Chunks use source character offsets rather than decoded token
strings, with 32 token overlap by default. Every exact chunk string is tokenized
again without truncation to verify that special tokens plus content fit the
active limit. `RepositoryEmbeddingConfig.readme_chunk_overlap_tokens` supports
in-process overrides and rejects overlap that cannot fit the active budget.

Class weights are total budgets, divided equally among that class's selected
chunks:

| Class | Headings | Total weight | Maximum chunks |
| --- | --- | ---: | ---: |
| Purpose | Overview, Introduction, About, Features, Highlights, Use Cases, Architecture, How It Works, Usage, What Is, Why, Motivation | 3.0 | 32 |
| Neutral | Preamble, heading-free prose, unknown sections | 1.0 | 24 |
| Setup | Installation, Setup, Getting Started, Quick Start, Build | 0.1 | 8 |

Therefore multiplying low-information sections or their length cannot increase
their class weight. Input analysis retains at most the first 64,000 characters,
matching the existing Markdown processing bound. At most 64 README chunks reach
the bounded encoder, with deterministic first-in-class selection. Content beyond
these limits is deliberately omitted; the guarantee is that emitted chunks fit,
not that arbitrarily large documents are fully represented. Prefix/class limits
can omit late useful sections and should be evaluated on a representative corpus.

Absent classes contribute no weight. Empty, code-only or entirely excluded
READMEs produce no README tower; metadata and topics are renormalized using the
existing tower weights. No provider call is introduced by this representation;
card summaries remain an independent artifact.

The tower cache hashes the emitted text and its section-class/weight plan, token
overlap, active sequence length, exact loaded model/revision and embedding
version. The source fingerprint also includes the weighted chunk plan. Changing
only an Overview heading to Installation therefore cannot reuse the old README
vector even when the cleaned body text is identical. Future policy changes need
a new embedding version and preprocessing identity.

## Reproducible offline comparison

The fixed fixture is `tests/fixtures/readme_retrieval.json`: 12 complete synthetic
repositories and 12 manually assigned queries. It includes long technical,
heading-free and boilerplate-heavy READMEs, with generic metadata and distracting
contribution/install material. It is a controlled diagnostic, not a production
relevance benchmark or heavy-ranker qualification.

With the model cached at revision
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, run:

```bash
python3 -m scripts.evaluate_readme_aggregation --output /tmp/readme-evaluation.json
```

The script forces local-only model loading, CPU execution and one Torch thread.
It uses no Qdrant, database or summary provider. Both strategies share the same
model/revision, queries, repository metadata and tower weights. The baseline
runs the actual v2 2,500-character/250-overlap mean-pooling path, including its
encoder truncation. Reports include README-only and final-tower MRR, Recall@3,
nDCG@3, per-query rankings, fixture SHA-256, encoder input/token counts and wall
time. Both paths retain the same metadata/topic truncation behavior; this issue
changes README chunks only. Timing is machine- and warmup-dependent.

A measured pinned-model comparison is checked in at
`docs/readme-aggregation-evaluation.json`. The fixture SHA-256 in that report
binds the complete repository/query inputs. Its diagnostic improvements should
be reproduced and then checked on a larger real repository/query relevance set
before serving qualification.

## Coordinated activation and migration

This change ships an opt-in representation. It does not deploy, rewrite the
production corpus, or change the ranker manifest. Existing user profiles and
feedback-adjusted user vectors are retained; evaluate them against the new
repository representation rather than resetting their state.

1. Keep current v2 serving active. Run the offline comparison and a representative
   real-corpus relevance/ranker evaluation in an isolated environment with the
   pinned model. Use the exact v3 version for the candidate writer. Production
   configuration and the checked-in ranker still declare only v2; the shipped
   ranker correctly refuses v3. Do not add v3 to its manifest just to pass startup.
2. Qualify/retrain the ranker for the new repository representation, including
   existing user vectors. Produce its versioned model/scaler/manifest and retain
   the previous artifacts for rollback. This is a separate rollout gate, not
   established by the small synthetic diagnostic.
3. Coordinate a backend-owned reindex campaign. Dry-run a bounded list containing
   canonical repo IDs, current content hashes, stored content versions and the
   proposed **strictly newer** backend-issued content revisions. Compare source
   counts, expected v3 count and minimum eligible corpus before approving the
   campaign. The ML checkout does not own the backend command; use the backend's
   reviewed reindex/outbox workflow, not a local vector overwrite script.
4. Deliver a small canary through the normal authenticated V2 repository-embed
   endpoint and backend outbox, with fresh job IDs and newer canonical revisions.
   A replay with the same content revision is duplicate/current and keeps the
   stored v2 vector even when the writer is configured for v3. A same-version new
   job is not a migration. Do not lower versions, forge hashes, delete points,
   bypass Redis locks/CAS, or use summary backfill to migrate vectors. Backend
   support to issue new revisions for unchanged source is required if its current
   reindex workflow only replays the existing revision.
5. Inspect canary payload/version/dimension/model revision, finite normalized
   vectors and serving eligibility. Verify duplicate replay is idempotent and
   delayed older content cannot overwrite the migrated vector. Existing
   independently refreshed feature revisions must remain intact. Local Qdrant
   regression tests exercise this newer-revision/unchanged-replay path.
6. Plan an explicit maintenance/coordinated cutover: v2-only serving excludes
   migrated v3 points, so replacing points in the shared collection reduces the
   old eligible corpus during backfill. Pause or otherwise coordinate traffic
   and the backlog, and confirm the completed v3 eligible count before switching.
   Configure `V2_COMPATIBLE_EMBEDDING_VERSIONS` to versions actually qualified by
   the candidate ranker and include the configured embedding version. Do not
   assume shared model/dimension makes both aggregations ranker-compatible.
7. Run production config validation, collection contract/bootstrap checks, model
   warmup, health/eligible-corpus checks and the production smoke test with the
   matching backend contract before moving traffic. Watch retrieval quality,
   latency and eligibility. For rollback, restore the qualified v2 writer/ranker
   and restore/rebuild a sufficient v2 corpus through another backend-owned newer
   revision campaign before restoring v2-only traffic. Merely changing an env
   value does not restore replaced vectors.
