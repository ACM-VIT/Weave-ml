# Proposed model journal archive contract — decision pending (#194)

Status: **not approved; developer protocol only**. ModelRuntime does not instantiate
VerifiedJournalTier. This change enables no live compaction, creates no collection,
and supplies no migration or cleanup command. Issue #194 remains incomplete.

## Decision for the human owner

Approve or amend the following concrete proposal before integration:

| Topic | Proposed decision |
| --- | --- |
| Ownership | ML service operators own one dedicated, preprovisioned vectorless Qdrant archive collection and its dedicated Redis reservation ledger. Assign a named owner and recovery on-call before deployment. |
| Retained data | Retain every original journal record byte, operation alias, withdrawal attestation, vector pin, prepared/rejected record, complete snapshot and checkpoint moved to archive. No proof is declared obsolete merely because a newer checkpoint exists. |
| Window and deletion | No expiry or deletion from archive in this proposal. Only verified duplicate hot copies may be physically deleted. Any finite retention window, user erasure or removal of archived proofs requires a new approved semantics/privacy decision and protocol. |
| Old operations | Preserve existing exact-byte duplicate/conflict detection and existing bounded history/status traversal limits. An archive is read-through storage, never evidence that a prepare committed. Current user head remains sole commit authority. |
| Admission identities | Keep Redis operation IDs, version identities, withdrawal lookup and stable initialization guards intact. Their capacity limits remain effective. No producer TTL, pending stream trim, ACK rule or initialization behavior changes. |
| Access/privacy | Archive contains sensitive original profile/feedback bytes and complete user snapshots. Apply the same or stricter service-only access, encryption in transit/at rest, audit logging and backup access controls as the hot journal. Operators must assign/access-review concrete principals; this code does not provision these controls. |
| Authentication | Service-owned secret of at least 32 bytes signs a receipt binding format, hot namespace, archive namespace, approved decision digest, full key and SHA-256/byte length. Store the key outside Redis/Qdrant. Receipt authenticates provenance; it is not human approval. Rotation requires verifying/re-signing all receipts with a separately approved recovery procedure. |
| Capacity | Choose explicit finite archive record/byte ceilings and infrastructure disk/Redis memory limits from measured overhead. Refuse compaction at archive capacity; admission can then exhaust hot capacity. Infinite admission with finite total storage and permanent exact proofs is not promised. |
| Recovery | Redis routes/reservations/readers, both Qdrant collections, signing secret and decision digest form one recovery boundary. Stop readers/writers before consistent restore, rollback or namespace changes; never restore these independently and resume automatically. |

The pending approval must record approver/date, decision digest, named archive
owner, access principals, concrete quotas and incident/backup owner. Merely passing
a digest to the constructor does not establish approval. Live migration,
credentials and cleanup require separate authorization as specified by #194.

## Implemented journal-only protocol

`VerifiedJournalTier` implements the same immutable-record interface already used
by Journal and Coordinator. The dedicated archive is itself BudgetedRecords;
its permanent record/byte reservations and temporary recovery copies share a
finite archive ceiling. Hot and archive Qdrant collections must differ. All
journal users must use this tier before any compaction; running an old
BudgetedRecords writer concurrently bypasses the fence and is unsupported.

1. Read/recover the exact hot bytes under a durable per-key reader fence.
2. Reserve archive capacity, freeze/copy the exact bytes and verify readback.
3. Atomically publish an HMAC receipt only if the hot identity is identical,
   no recovery candidate remains and no hot reader is active. Archive routes
   fence new hot reservations and redirect subsequent reads to verified archive.
4. Read through the published receipt before deletion. Delete the identical hot
   copy with a strongly ordered Qdrant operation and confirm physical absence.
5. Atomically remove the identical hot reservation and decrement count/bytes.
   A retry sees the receipt and absent reservation and refunds zero a second time.

A crash before publication leaves hot storage charged/readable and any archive
copy permanently charged. A crash after publication redirects reads even if hot
deletion/refund has not finished. A lost deletion response is retried and checked
for absence before refund. A lost refund response is safe to retry. Failures
propagate as capacity/unavailability errors; no exception silently authorizes ACK.

The reader fence has **no timer**: a worker killed during hot IO leaves a durable
reader entry that prevents compaction for that key. Failed hot IO, including a
timeout whose request may still execute remotely, also retains its reader fence;
a successful retry does not clear that earlier uncertainty. Reader-map entries and
per-key concurrency are bounded by max hot records. Recovery requires stopping
all journal IO, proving old workers and outstanding remote requests cannot resume, validating hot/archive records
and reservations, then explicitly repairing abandoned reader entries. No automated
reaper is provided. Availability is sacrificed rather than allowing stale IO to
reinsert an uncharged point after deletion.

Lost/corrupt archive receipts, missing archive bytes and independent archive
reservation disagreement fail closed. A missing hot record with an archive copy
but no routing proof cannot be treated as a new identity. Rebuilding receipt data
is a controlled recovery operation; no reader guesses proof contents.

## Measurements and validation

`metrics()` reports hot/archive final and temporary-copy payload reservations,
record counts, admission headroom, routing entries and reader entries using one
Redis transaction. Compaction returns elapsed seconds, refunded records and bytes.
Routing/reader metadata bounds count key/value UTF-8 bytes; they exclude Redis
hash/node/allocator overhead. These are logical storage metrics, **not physical
Qdrant/base64/index/segment/disk or Redis memory measurements**. Infrastructure
capacity planning must measure those overheads and account for both copies during
migration and deferred Qdrant segment reclamation. Raw payload quotas include
in-flight archive recovery copies; no disk refund is inferred from the metrics.

The local real-Redis fixture admits ten distinct exact-byte records through a hot
ceiling of two records, compacts each, verifies old reads/conflicts, and then
refuses the eleventh archive copy at the explicit ten-record archive ceiling.
It records per-call cost and before/after headroom; this is a synthetic protocol
fixture, not a production throughput claim. Fault injection covers archive write
response loss, published receipt, physical deletion and accounting response loss,
retry/double-refund behavior, blocked in-flight recovery, orphaned reader fences
and missing/corrupt proofs. Those unit fixtures use an in-memory immutable record
adapter and actual Redis Lua, not Qdrant durability or Redis restart testing.

A synthetic local run on 24 October 2026 (Python 3.14.2, Redis 8.10.2,
in-memory immutable record adapter, Unix socket; ten 28-byte records) reported:

| Logical metric | Before: two hot records, no copies moved | After: ten records admitted and moved |
| --- | ---: | ---: |
| Hot records / payload bytes | 2 / 56 | 0 / 0 |
| Archive records / payload bytes | 0 / 0 | 10 / 280 |
| Hot record headroom | 0 | 2 |
| Archive record headroom | 10 | 0 |
| Routing records / conservative key-value byte bound | 0 / 0 | 10 / 3,030 |

All ten compactions refunded 280 hot payload bytes in total. A repeated run of
the same ten-record fixture measured **0.321 ms median / 0.449 ms nearest-rank
p95** per compaction (ten samples). These costs exclude Qdrant IO and durable
Redis persistence and are not service latency estimates. Total retained original
bytes grew to 280; hot reclamation does not remove archive obligations.

The local Qdrant in-memory engine plus owned Redis also exercises the actual
runtime/Journal/Coordinator scenario below. This validates reducer/replay and
adapter integration, not Qdrant server or persistence durability.

An opt-in owned Docker integration test uses actual QdrantRecords, Journal,
ModelProcessor and Coordinator to archive a history containing a reversal and
withdrawal, then check exact replay, old duplicates/conflicts and guards. Run:

```sh
python -m pytest tests/test_model_retention.py -q
RUN_MODEL_OPERATION_INTEGRATION=1 python -m pytest tests/test_model_operations_integration.py -k explicit_verified_tier -q
```

## Remaining before #194 can be completed

- Human contract approval with concrete ownership, quotas and access/recovery rules.
- Archive-aware Redis operation-ID/version/withdrawal admission proofs and safe,
  bounded reclaim of those existing ledgers. This draft reclaims only hot journal
  reservations; the configured runtime still eventually refuses new identities.
- Approved runtime integration/migration and prevention of mixed old/new writers.
- Real Redis persistence/restart and Qdrant durability/process-death acceptance;
  response-loss injection alone is not evidence for infrastructure durability.
- Comparable physical retained-byte and sustained operation-admission measurements
  covering journal plus identity proofs, Qdrant segment reclamation and Redis
  metadata. No live workload or production performance was measured here.
