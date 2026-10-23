# Proposed observable ranking targets (#154)

`contracts/ranking_targets_v1.json` is **a proposal, not an approved training or
serving contract**. Product/data owners choose semantics and business weights.
This PR adds offline measurement preparation only. It does not complete #154,
change any weight/model/output, qualify the synthetic artifact, or duplicate #28.

## Evidence and limits

Inspected ML `72d3b97` and Weave **dev**
`a41c54ca619e5d06a46a471c163f86744606c972` (main `ad4e4f9` is older):

- ML `api/contracts.py` has README-open/GitHub-open/save/dwell, rejects online
  impressions, and has no click/follow event. `inference/model_manifest.json`
  explicitly records synthetic training and a manual production override.
- Weave `src/components/home/RepositoryFeedItem.tsx` records impressions after a
  visibility timer, dwell on visibility completion, and distinct open actions.
  `src/api/activity.ts` attaches session/serve/position and clamps dwell to
  3–300 seconds. These are device reports, not proof of user intent or page load.
- Weave `backend/services/activityV2Service.ts` validates attributed serve items,
  persists event IDs idempotently, and handles save state transactionally.
  `backend/contracts/interactions.v2.ts` has no follow action. Account follow is
  available in the user controller, but does not establish repository-owner
  follow, exposure linkage, consented event history or reversal labels.
- `backend/services/mlFeedbackPayloadService.ts` strips session/serve/position
  and excludes impressions from ML delivery. Online feedback alone cannot build
  the denominator. #26 must export backend telemetry and serve/prediction
  snapshots together; user/repo-only or nearest-timestamp joins are unsafe.
- Save changes from list membership can be emitted without a feed serve. The
  exporter must reconcile all repository membership changes for baseline/end
  state and only attest complete coverage when attribution is unambiguous.
  Exclude rows otherwise. Aggregate counters and present-day tables are not a
  point-in-time label history.

There is **no legitimate real-data report bundled** and no claim that targets
are calibrated. Tests use artificial examples only to verify arithmetic and
exclusions. Declaring `real_observed` in an input is an exporter assertion, not
proof: owners must audit consent, provenance, completeness, prediction timing
and cohort selection before using any generated report. No companion Weave PR
is needed for this offline proposal; actual event/contract additions require
an owner-approved follow meaning and coordinated version transition.

## Offline interchange (not an HTTP schema)

Run `python -m scripts.evaluate_ranking_targets /secure/export.json`.
The command reads a local JSON object and prints aggregate JSON to stdout.
It has no network/database/model execution and writes no dataset. Store exports
and reports under approved access/retention controls; reports can identify small
cohorts, so do not publish them or identifiers in PRs. Existing API/OpenAPI and
serving schemas are unchanged.

Required top-level keys: `metadata`, `exposures` and `events` (arrays).

`metadata` requires:

| Field | Meaning |
| --- | --- |
| `target_version` | `ranking-targets-v1-proposal` |
| `data_kind` | `real_observed`; synthetic/mixed data rejected |
| `dataset_id`, `lineage` | Frozen snapshot identifier and auditable extraction/code/query reference |
| `consent_basis`, `cohort` | Owner-reviewed lawful consent/retention basis and cohort definition |
| `model_version` | One frozen artifact; split datasets by model version |
| `observation_cutoff` | Timezone-aware event-time completeness watermark after client lag reconciliation |

Each exposure contains `event_id` (persisted impression ID), `user_id`,
`session_id`, nonnull `serve_id`, `repo_id`, nonnegative integer `position`,
`occurred_at` and `predicted_at` (timezone-aware, prediction no later than
impression), `eligible` (boolean), `initial_saved` (boolean or null),
`coverage` (explicit boolean keys `ctr`, `save`, `gh_open`, `dwell`) and
`predictions` (all five output names, finite numbers in [0,1]). `eligible`
attests consented, foreground viewable impressions with verified serve item and
point-in-time model/features. `coverage` attests complete delivery/instrumentation
for that head over the full window, including baseline/save-state reconciliation.
Never infer completeness merely because some events exist. Export verified
predictions captured at serving, not re-scoring with future user features.
Unknown/missing serve attribution must be quarantined upstream, not filled in.

Each action contains `event_id`, the exact same five attribution-key fields,
`event_type`, `occurred_at`, and optional `dwell_ms` (integer 3000–300000 for dwell,
null/absent otherwise). Use persisted authenticated `user_id`, not a client
claim. Only v2 action types are accepted; no follow events. The exporter maps
`client_occurred_at` to `occurred_at` after clock-quality checks and retains raw
receipt timestamps in its audit source. Timezone offsets normalize through
aware datetime comparison. An action at window end is excluded. Exact duplicate
IDs collapse; conflicting reuse or exposure/action ID collision fails closed.

Repeated impression IDs with different content fail. Distinct impressions on
the same user/session/serve/repo/position key make labels ambiguous. No action
is assigned to two exposures. Actions before their impression also make the
key ambiguous: a one-second visibility delay cannot be repaired by guessing.
Unmatched actions are counted but never enter a denominator. Simultaneous
save/unsave is ambiguous. Missing dwell is never interpreted as zero because
short episodes are suppressed. All labels in immature windows are censored,
including early positives, to avoid asymmetric maturity selection. Follow is
always `unsupported`, never a negative example.

## Reading the report

Per head, `states` counts observed/missing/censored/ambiguous/ineligible or
unsupported rows. Binary heads report Brier, log loss, ten-bin calibration,
ECE and positives on **observed rows only**. Dwell reports MSE/MAE and bins as
regression diagnostics; its thresholded, clipped episode target differs from
full engagement duration and from synthetic simulator behavior. Calibration is
conditional on cohort and logging, not evidence of causal ranking improvement.
Empty samples and zero score denominators produce null metrics rather than
zero-quality or zero-share claims.

Value diagnostics use actual finite predictions for eligible rows and the
current imported `VALUE_WEIGHTS`: mean `weight * output`, share of summed
scores and mean per-row share (excluding zero-score rows). The coefficient
ratio `20 / 28.1 = 71.17%` is **not** an actual score contribution; observed
prediction values determine that share. Missing labels do not prevent measuring
prediction contributions. `follow_zero_shadow` only measures the arithmetic
score delta from removing follow, explicitly never promotion evidence: it does
not evaluate ranking quality or justify a new production weight. Feedback-vector
alpha is unrelated and is not used anywhere in this report.

## Owner decision and release packet

Before #154 acceptance, named product/data owners must approve the versioned
dictionary (README-open vs alternative click; denominator; 24h window and lag;
save reversal/baseline; dwell meaning/units; follow meaning). For follow, choose
one: a consented source with explicit target identity, exposure, baseline,
follow/unfollow ordering and missing-data policy; or evaluate removal of its
influence in a **new** value-function version. The current draft approves
neither. Record decision/owner/time and immutable dictionary hash in #154.

Then #26 supplies a frozen, audited real export and #33 provides temporal
holdout quality, source/position/cohort coverage, calibration/contributions,
confidence/uncertainty and per-request ranking comparisons against the current
version. No confidence or promotion threshold is invented here. #28 continues
to own synthetic-model qualification. Have owners compare the new target's
support/censoring with the current model's training target before retraining.

A future production change requires new model/value-function/output provenance,
old/new caller compatibility (Weave dev contract and generated ML output),
shadow quality and latency evidence, approved rollout cohort/monitor/stop
thresholds and a tested restoration of the previous immutable model, scaler,
weights and version identifiers. Preserve historical old-version reports and
never reinterpret their labels in place. This additive offline PR can be rolled
back by removing these files without touching online services; it needs no
migration or deploy. Future follow telemetry must use expand/observe/evaluate
before enabling a changed objective and needs its own paired release plan.
