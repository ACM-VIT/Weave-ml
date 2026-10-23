# Backend-sourced social candidate channel

This is the provider portion of issue #34. The ML channel is implemented and
locally exercised; its paired backend SQL/feed/generated-client producer is a
separate pending change. A supported source label does not mean its product-data
projection exists. Keep the issue open until every remaining criterion below is
satisfied. No production flag or environment file is changed by this work.

## Ownership and wire contract

Only the authenticated backend supplies `social_candidates` on recommendation
requests. The optional list contains at most 100 unique repository/source records:

```json
{
  "repo_id": "00000000-0000-4000-8000-000000000001",
  "owner_github_id": "123456",
  "source": "followed_owner",
  "support": 1,
  "source_score": 0.5
}
```

This example illustrates the schema, not a live repository or social fact. Repo
IDs are backend-issued UUIDs; owner GitHub IDs are positive decimal strings in the
backend bigint range. Names/handles are never substituted for immutable owners.
Support and source score must be derived from actual permitted source rows. Raw
actor identities or private board/activity material must not be sent to ML.
Duplicate repository/source rows or conflicting immutable owner attribution fail.
The backend must validate visibility, actor state, permission/consent and canonical
identity before handoff; ML cannot infer those product rules from Qdrant.

The provider recognizes followed-owner, followed-like/save, co-like/save,
board-topic and adjacent-interest labels. Collaborative labels require at least
two distinct supporters; direct followed-owner/own adjacent-interest evidence
requires one. These are conservative experimental support floors, not a privacy
policy or proof of consent. Initial paired backend work supplies verified active
followed-owner public repositories. Other channels require their own real
viewer-scoped backend projection and acceptance before use.

## Retrieval, scoring and bounds

The source projection is request-local and never enters shared discovery cache.
The channel reserves at most 100 canonical lookup IDs from the existing overall
candidate-read budget, then makes one payload-only Qdrant retrieve through the
existing shared FIFO executor/deadline. It does not allocate another pool, fetch
product SQL or embed repositories on request. Excluded IDs never enter lookup.

Every returned point must match a requested canonical identity and all existing
serving-eligibility/model/revision/dimension/content/feature contracts. Ambiguous
duplicate points are discarded. Evidence overlapping an existing candidate must
refer to the same content/feature snapshot. Eligible source candidates are fused
with deterministic reciprocal-rank fusion (`k=60`), limited to 20 pool candidates,
two repositories per immutable owner and eight per contributing source by default.
Overall retained pool size remains unchanged; quota reservation lets genuinely
new social IDs reach heavy ranking rather than being cut off behind the ANN
prefix. These are channel quotas, not global final-feed owner/topic/fork caps.

Social candidates remain metadata-only until the ordinary retained vector pass.
That pass revalidates atomic compatible payload/vector material and rejects stale
snapshots. Strict-heavy mode still requires the user vector, qualified ranker and
complete valid model output. A social lookup failure can leave other valid
channels available; it cannot turn shadow data into a fallback or bypass a
required dependency failure. Model scoring and the existing bounded social boost
remain intact. No serving eligibility marker is written by retrieval.

Recommendation items retain finite bounded `retrieval_scores` maps alongside the
final score/source. Raw source contributions survive fusion, heavy ranking, the
legacy bounded boost and assembly. They are not probabilities or proof of causal
quality uplift. Durable item/impression feature snapshots and training exports
remain issue #26, not an implied side effect of this response field.

## Rollout and observability

- `V2_SOCIAL_CANDIDATE_MODE=disabled`: no social lookup, injection or ranking work.
- `shadow`: perform bounded lookup/selection and aggregate diagnostics while
  preserving the normal ranking pool and returned feed.
- `enabled`: admit eligible quota-bound social candidates before ordinary ranking.
- `V2_SOCIAL_CANDIDATE_TRAFFIC_PERCENT`: 0..100, deterministic canonical-user cohort.
  Excluded cohort members behave as disabled. The shared policy is never mutated
  by a previous user's decision. No generation seed, user or source material is
  inserted into a global cache.

Examples keep mode disabled and traffic zero. A backend producer must be enabled
and compatible before any useful source records exist; setting ML mode alone
cannot invent them. Health reports configured mode/percentage and aggregate
shadow/injected/lookup-failure counts. Candidate transport metrics add one fixed
`social` channel, retaining sizes/counts only and no user/repository/actor IDs.
Selected-pool counters do not count impressions or prove recommendation quality.

## Verification and dependencies

Focused tests exercise deterministic fusion, support floors, immutable-owner and
source caps, bad metadata, missing/ineligible/excluded points, real local Qdrant
filter/projection behavior, genuinely new IDs reaching required-heavy input,
strict dependency failures, disabled/shadow feed parity, stable rollout and the
registered authenticated HTTP request/response shape. Only the heavy model is a
deterministic test double in local-Qdrant integration; this is structural
acceptance, not representative model quality or full production Qdrant/SQL QA.

This provider stacks on ML PR131 and its generated-contract prerequisite PR128.
The consumer should stack on Weave193/192, vendor this exact OpenAPI digest,
regenerate client types and exercise the real paired HTTP plus backend SQL
visibility/provenance paths. Deploy compatible provider before consumer; disabling
the flags and returning to the previous compatible pair rolls back the channel.
No deployment, shared migration or model promotion is authorized here.

Original remaining #34 acceptance is preserved: implement and verify permitted
collaborative, followed-like/save, board-neighborhood and adjacent-interest
projections where required; privacy/consent and source-support evidence for each;
fanout/popularity/actor domination limits; deterministic duplicate fusion; flags
for shadow and gradual rollout; actual offline relevance, coverage, novelty and
source-contribution reports with trustworthy judgments. The current synthetic
structural fixtures cannot replace genuine quality evidence or close the issue.
