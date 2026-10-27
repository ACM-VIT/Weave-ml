# Repository admission: proposal v1, observation only

Related: ACM-VIT/Weave-ml#152. **Owner approval is pending.** This change prepares
one shared policy observation; it does not complete the issue's approved-policy
acceptance criterion. There is no enforcement flag or deployment step in this PR.

## Separate decisions

| Stage | Meaning and current authority |
| --- | --- |
| Acquisition acceptance | A source can enter the ingestion workflow. `main.filter_enriched` retains its existing configurable README/metadata gate. This is not quality approval. |
| Schema validity | Backend ingestion and ML V2 independently validate identity, field bounds and transport contracts. A content observation does not validate a schema. |
| Indexing eligibility | A schema-valid source must produce a valid vector and validated payload. V2 does not apply a product-quality threshold today. |
| Quality approval | A separate product decision. `quality_approval=not_evaluated` always; heuristic scores and the legacy compatibility `APPROVED` string are not canonical approval. No learned quality model is implemented. |
| Serving eligibility | Existing ML technical validation plus backend hydration/status policy. `serving_eligibility_version=repository-vector-v2` proves vector/payload validation only. It does not certify quality or override backend status. |

The pure evaluator in `contracts/admission.py` is the canonical **proposal**,
version `repository-admission-proposal-v1`. It emits `admission_observation` in
active acquisition source payloads, compatibility `IngestionResult` (including
early rejection), and Qdrant vector payloads built by V2/direct embedding. It is
recomputed from source rather than trusting supplied approval claims. A supplied
raw README overrides its claimed length. Missing length requests review. These
observations do not participate in content hashes, version fencing, public API
responses, backend ingestion requests, or retrieval filters.

Agreement requires equivalent evidence: V2 derives length from raw README when
present and defaults an omitted README/length to zero, yielding `review` with
`readme_missing`. A direct input with no length yields `readme_length_unknown`.
Neither case invents evidence from summaries, and neither can pass by default.
Malformed truthy language/topic values do not supply candidate evidence; the
observation still does not replace schema validation.

## Concrete owner decision

The proposed content candidate rule is README length >= 200 characters and at
least one nonempty description/languages/topics value. `pass` means candidate
evidence passed this proposal, never quality approved. `review` carries stable
reason codes: `readme_length_unknown`, `readme_missing`, `readme_thin`, and/or
`repository_metadata_empty`. Thin and shell fixtures request review; stale,
zero-star and well-documented fixtures can pass. This preserves low-star discovery
as a proposed choice. There is **no proposed star floor**, no freshness rejection
threshold, and no documentation/code-health blend threshold.

Before enabling rejection, the owner must approve:

1. The README threshold and whether thin/empty repositories are rejected or sent
   to a review queue, including the tradeoff for new or code-only projects.
2. Low-star discovery without a popularity floor and any separately evaluated
   quality model/threshold. The historic 50-star floor is not a recommendation.
3. Whether inactivity should affect admission or only ranking, and who owns
   evidence freshness (coordinate lifecycle issue #153 and refresh issue #85).
4. Which boundary enforces the approved version, how the backend stores and
   forwards that decision, and whether old content is grandfathered.

Record the approved decision/version and representative outcome matrix in a
follow-up before changing a gate. Observational agreement does not establish
agreement of existing enforcement: the compatibility scorer still has its legacy
50-star / 200-character prefilters and 0.60 blend, while acquisition has only its
current README/metadata filter and V2 relies on schema/vector validation. Tests
explicitly retain the low-star legacy rejection to make that difference visible.

## Cross-repository compatibility

Inspected Weave `dev` at `a41c54ca619e5d06a46a471c163f86744606c972`:
`backend/contracts/ingestion.v2.ts`, `services/mlRepositoryPayloadService.ts`,
`services/mlService.ts`, and `services/feedV2Service.ts`. Backend source projection
is explicit, index responses are strict, and feed hydration checks active status.
No admission state is added to its outbox or public contracts. Acquisition's
existing explicit backend delivery projection omits the local observation;
embedding recomputes it from source. Therefore current Weave and this ML change
remain wire compatible without a companion PR. Cross-repo code inspection is
not a running backend integration test.

## Rollout and rollback

Merge/deploy of this preparation does not enable a new rejection decision. Existing
points without an observation remain readable under current retrieval rules;
missing observation means unknown, never approved. Existing points keep their
old policy provenance until a normal content write recomputes an observation.
Same-version replay does not rewrite observation-only metadata. No deletion,
automatic reindex, migration, or corpus cleanup is performed.

A later approved policy needs additive backend/ML contracts, tested old/new
combinations, an explicit decision for existing points, and observation comparison
before rejection is enabled. New policy semantics must increment the policy
version. Roll back this preparation by reverting its code: readers ignore the
additional nested Qdrant field, so old and new points coexist with no cleanup.
Rollback of future enforcement must disable that gate and preserve corpus data;
it must not erase rejected or grandfathered content automatically.
