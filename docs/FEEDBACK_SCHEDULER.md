# Bounded cross-user feedback processing

The feedback CLI polls with a persistent scheduler across live and reclaim reads.
`FEEDBACK_CONSUMER_WORKERS` defaults to 4 and accepts 1..16. Capacity counts both
queued and executing messages, capped at `max(workers, read_batch_size)` (at most
1000). Redis reads are capped to available capacity. Each canonical user has one
executing event and at most one waiting event; round-robin turns permit other users
to progress. There is no unbounded executor queue or per-user thread pool.

A hot user's excess deliveries stay in the Redis pending list. They are not
acknowledged, silently dropped or held in an unbounded local buffer. They become
eligible through the existing reclaim cadence, which can delay that user's
throughput in exchange for cross-user fairness and bounded memory. A failed lane
also releases queued successors back to pending rather than repeatedly attempting
them ahead of its failed predecessor. Active message IDs deduplicate local reclaim
while the actual operation remains in flight.

Local lanes alone do not establish distributed ordering. Every event still takes
the renewable Redis user lock; the actual applier checks its persisted version,
identity and rejection fence, and conditionally writes Qdrant through the existing
CAS guard. An out-of-order live/reclaimed event remains a retryable gap. This also
applies to a poison event: `rejected_gap` must not be acknowledged or removed before
its predecessor. Exact-next poison events retain durable DLQ/rejection fencing;
duplicates do not learn twice. Existing retry caps, lock loss and DLQ capacity
policy remain intact. Backend product state and outbox ordering are unchanged.

Repository snapshots remain limited to each Redis response. The coordinator
captures the validated immutable repository snapshot; the worker installs it in
its own thread-local context only for that event and restores the prior context.
User ledgers are always read under their distributed lock, never prefetched.

SIGTERM/SIGINT stop polling and admission. Unstarted work remains durable pending.
Close waits for synchronous workers to exit with lock renewal, consumer heartbeat
and clients still available; it cannot cancel an already running Qdrant request.
Only then are owned clients closed and the owning heartbeat removed. A hard crash
relies on lock expiry, cursor/CAS checks and Redis reclaim. `run_once()` defaults
to draining admitted work for compatibility; production uses `run_once(wait=False)`
so slow users cannot block subsequent polls. Worker count 1 restores serial apply.

Tests cover parallel users across polls, one active apply per user, lane bounds,
active-ID reclaim deduplication, failed-lane deferral, actual-worker shutdown,
poison gaps and the existing feedback invariants. Optional two-process coverage
requires disposable loopback Redis 6397 and Qdrant 6347 via
`FEEDBACK_SCHEDULER_TEST_REDIS_URL` and `FEEDBACK_SCHEDULER_TEST_QDRANT_URL`.
That fixture creates unique streams/collections and never flushes a database.
It verifies crash-owner reclaim, ambiguous duplicate delivery, poison rejection,
ordered final cursors, empty pending backlog and four durable DLQ entries.

Fixture throughput and completion age use a matched simulated 5 ms dependency delay;
they are not production capacity or latency guarantees. No deployment or shared
service changes are part of this implementation. Separately budget and measure
real dependency limits before changing deployed worker counts.

Ordering gaps do not consume dependency-failure retry attempts: a later valid
event waiting on a locally deferred or reclaimed predecessor stays pending even
when the dependency-attempt cap is 1. Missing predecessors therefore require
explicit reconciliation rather than silently dead-lettering healthy successors.
Admission stop is published before acquiring the poll lock and before CLI signal
drain, and checked at submission and worker entry. Production preflight validates
the optional worker count with the same 1-16 range as runtime.
