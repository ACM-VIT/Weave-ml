# Repository lifecycle: candidate verified-status-v1

Refs ACM-VIT/Weave-ml#153. This is prepared code and a reviewable policy, **not
production activation or completed operational acceptance**. Both gates default
to `disabled`. The candidate follows the [issue policy handoff](https://github.com/ACM-VIT/Weave-ml/issues/153#issuecomment-6373510644);
that comment records an external person's account of approval. The owner must
approve this exact policy and rollout directly before either gate is enabled.

## Candidate decision

| Observation | New recommendation selection after refresh completion |
| --- | --- |
| Complete verified GitHub archived=true or disabled=true | Excluded |
| Complete verified false/false, newer than last verified observation | Eligible; revives a prior exclusion |
| Failed request, permissions error, timeout, ambiguous 404, partial response | Preserve last verified status; never infer deletion or restoration |
| No previous verified observation | Eligible (legacy compatibility) |
| Old push/commit activity, even many years | Eligible; recency remains a ranking feature |
| Older verified evidence | Ignore; cannot undo newer evidence |
| Equal-time conflicting verified flags | Reject with 422; investigate producer data |

Freshness fallback is explicit: last verified status has **no automatic expiry**.
Unavailable attempts update attempt provenance without clearing verified evidence.
A newer verified restoration may apply even if a failed attempt happened later.
Future and timezone-naive observations are rejected. There is no inactivity-age
threshold, purge, inferred death, or status/model-compatibility coupling.

`lifecycle` is an optional strict object in RepositorySource and
RepositoryFeaturePatch: `status` (verified/unavailable), `observed_at` (aware
ISO8601), `source` (github-graphql-repository/github-rest-repository), and both
strict boolean `archived`/`disabled` when verified. Unavailable forbids flags.
The internal authenticated producer owns the numeric GitHub identity and actual
response provenance; a source string is not cryptographic proof. Only service
observations from those GitHub repository responses may assert flags. A search
query containing archived:false, an embedding, a name, or a push date is not proof.

## Ownership and completion boundary

Offline ML acquisition asks GitHub for both flags and carries only complete
observations into the canonical backend ingestion record. Missing or failed
metadata leaves existing evidence unchanged. Weave `dev` owns ingestion
validation, its PostgreSQL transaction, feature version and durable `repo_refresh`
outbox delivery. Its source/feature projection forwards lifecycle only when
`ML_REPOSITORY_LIFECYCLE_OBSERVATIONS=verified-status-v1`; lifecycle is excluded
from the content hash, so a status-only change never requires embedding again.
The durable outbox JSON contains the evidence; no new database table/migration is
needed. Qdrant holds the last verified state and last observation attempt.

The defined boundary is the completed, verified ML Qdrant CAS for the repository,
not acquisition search, SQL enqueue, HTTP dispatch, or an existing generated feed.
Requests selecting candidates after this boundary exclude archived/disabled
repositories before semantic and every discovery channel's bounded Qdrant limits.
Search and social hydration share the same filter/defensive payload check.
Already generated feeds/in-flight selections are outside this boundary.

Verified time orders lifecycle independently of content and feature versions.
Delayed/current feature jobs can apply a lifecycle-only CAS without rewinding
counts or versions; stale status cannot revive. Content and feature CAS selectors
include lifecycle snapshot fields, fencing lease-lost writers. New content merges
rather than erases newer status. Active policy bypasses shared discovery pools,
whose payload snapshots could otherwise outlive a status transition.

## Compatible rollout, owner decisions and rollback

1. Owner explicitly approves the table above, especially legacy unknown eligibility
   and indefinite verified-status retention, and assigns refresh coverage/cadence
   under related #85. No scheduler or corpus-wide backfill is authorized here.
2. Merge/deploy ML additive contract and CAS support first with
   `V2_REPOSITORY_LIFECYCLE_POLICY=disabled`. New ML accepts old producers.
   Existing index bootstrap prepares boolean indexes for lifecycle_archived and
   lifecycle_disabled; disabled serving does not require these indexes in health.
3. Merge/deploy the linked Weave companion against `dev`, pinned to this ML schema.
   Its generated snapshot also incorporates previously existing score descriptions;
   those inherited descriptions do not change wire behavior. Old ML rejects new
   lifecycle fields, so keep the backend forwarding gate disabled until ML is ready.
4. In an approved staging environment enable backend forwarding, verify outbox
   delivery/CAS, archive/disable/restoration fixtures and old/new content jobs.
   Then explicitly approve enabling ML `verified-status-v1`. Its readiness requires
   the boolean indexes. Verify every channel, cache bypass, failed observation
   retention and stable/unknown projects before considering production activation.
5. Production activation/backfill needs separate owner authorization. Record
   refresh coverage and completed CAS boundaries; unobserved legacy repositories
   intentionally remain eligible and newly archived repositories remain eligible
   until authoritative observation and delivery. Do not claim continuous coverage
   merely because ingestion is implemented. Run the existing bounded refresh
   operator after approval; do not scan/delete the corpus automatically.

Rollback: set ML policy to `disabled` first (restores legacy selection), then stop
backend forwarding if needed. Retain Qdrant evidence and queued jobs. Never roll
ML back to a strict pre-lifecycle schema while lifecycle jobs can still be sent;
drain/disable new forwarding and reconcile the existing queue with the owner
first. Re-enabling uses retained verified evidence; audit freshness/coverage
before doing so. No data deletion is required.

## Verification scope

The focused lifecycle suite exercises actual local Qdrant filters, semantic and
all discovery channels with bounded limits, no-activity/unknown compatibility,
status-only CAS, out-of-order/revival/unavailable observations, content preservation
and stale-writer fencing. The linked Node/Python wire fixture uses real HTTP
routes, auth, CAS and local Qdrant, with deterministic encoding/local lease only.
The companion's disposable PostgreSQL test verifies archive-only ingestion queues
a durable refresh without changing content tuples/version or index job count.
These fixtures establish code/contract behavior, not GitHub refresh coverage,
production readiness, or an approved production policy.
