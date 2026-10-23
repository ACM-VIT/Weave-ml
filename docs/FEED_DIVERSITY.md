# Observed identity and embedding diversity

This is a partial implementation of issue24. It stacks on PR136's corrected
fresh vector/payload transport and PR135's independent social candidates. No
model/revision/dimension/ranker-feature changes or production promotion occur.

## Disabled by default

`V2_DIVERSITY_MODE=disabled` and `V2_DIVERSITY_TRAFFIC_PERCENT=0` preserve the
existing feed. Shadow mode runs the proposed selection only when fresh strict
heavy vectors already exist; it preserves baseline output and adds no repository
vector reads to hybrid. Enabled mode requires strict-heavy ranking and a maximum
150-candidate pool. A stable canonical-user hash selects traffic; request state
never changes another user's policy. Real qualification must precede rollout.

MMR normalizes relevance and penalizes cosine similarity using a bounded150x150
matrix. Configurable weights, duplicate threshold and hard owner/topic/category/
fork-family caps apply. Configured exploration reserves independent source
buckets outside personalized candidates; bounded backtracking avoids a greedy
choice consuming all topic capacity. The original retrieval source and scores
survive selection. Recent exclusions are supplied by the backend's bounded
telemetry/session history and checked again locally; serving is not an impression.

Unknown immutable identity, missing vectors or required exploration shortfall
fails enabled requests as a retryable dependency. Hard caps never relax to fill
a page. Quota-search exhaustion and unknown/shortfall counters are explicit.
Shadow reports unavailability and retains the baseline. Cooperative deadline
checks bound synchronous work between candidate/search steps; the single matrix
operation is bounded by pool/dimension. No global vector cache is introduced.

## Source metadata and paired backend prerequisite

`ACQUISITION_LINEAGE_ENABLED=false` keeps the current acquisition query/delivery
unchanged. When enabled, existing shared-budget GraphQL single/batch metadata
queries select isFork plus four observed parent edges. The parser identifies a
family only after an explicit nonfork terminal node. Truncated, inaccessible or
missing ancestry remains unknown. Handles/names never establish identity.

REST parser support distinguishes immediate parent from the ultimate network
source as documented by [GitHub](https://docs.github.com/en/rest/repos/repos#get-a-repository).
It introduces no additional REST network client. All numeric identities are
positive decimal strings within signed-bigint range; source and UTC observation
time accompany known ancestry. Unknown fork state never defaults to false.

Strict optional RepositorySource fields and atomic vector payloads preserve
owner identity, fork state/parent/family and source/time. Legacy payloads remain
compatible. Candidate projections, retained hydration and inline pair digests
include the scalar identity snapshot. Discovery caches preserve those scalar
fields; heavy-only topics/category still arrive through fresh vector hydration.
Core serving markers are never stamped by a payload-only update.

The canonical backend still needs persistence/provenance/versioned outbox wiring
for these new lineage fields before the acquisition flag can be enabled against
it. It must schedule metadata changes even when README hash is unchanged,
preserve newer authoritative observations and handle repository transfers by
numeric identity. The owner identity already exists in canonical PostgreSQL but
must be forwarded to ML. Deployment requires a coordinated consumer pin update
and explicitly authorized source refresh/reindex for the existing corpus. A new
schema or flag does not supply missing lineage or establish readiness.

## Evidence and remaining acceptance

Tests cover actual local Qdrant filters/projections/retained vectors and real
retriever/assembler wiring with a deterministic test ranker, including independent
social exploration, shadow parity, recent exclusion, missing identity failure and
vector-free hybrid. Pure controls cover owner/topic/category/fork/near-duplicate
caps, seed determinism, hard sparse behavior and quota-search bounds. No test
fixture populates the normal corpus.

A synthetic150x384→45 selector workload exposed repeated scalar cosine cost. The
bounded matrix replacement measured about88ms median and124ms maximum over ten
local requests before later input/deadline guards; this is a physical diagnostic,
not production latency or recommendation-quality evidence. Repeat representative
measurements before rollout.

Full issue24 stays open: paired canonical backend metadata and genuine corpus
refresh, representative relevance/catalogue coverage/intra-list similarity/novelty
reports, and any required product acceptance of changed diversity tradeoffs.
No real-user judgments or production approval are fabricated. Issue26's immutable
point-in-time impression/export provenance remains separate work; this change
preserves existing source fields without pretending a serve is an exposure.
