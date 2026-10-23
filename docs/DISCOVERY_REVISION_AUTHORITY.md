# Managed discovery revision authority

The configured API factory now supplies an authoritative revision provider and
shares its managed authority with repository writers. Defaults remain uncached.
Local configured-path activation is verified; production activation and external
topology/admin prerequisites are not claimed.

Redis atomic begin/read/finish scripts validate INFO server run_id and replication
master_replid/role together with the authority record. Begin marks a random token
DIRTY before any Qdrant mutation. Only a proven COMPLETED operation can finish its
own token; multiple writers keep reads uncached until every token completes.
Tokens never expire. Unknown work, lost responses, crashes, missing records,
protocol/epoch mismatch and overflow cannot become READY through elapsed time.
The record permits at most 64 writers and bounded token/proof/revision values.
The store covers create/index/upsert/content-CAS/feature-CAS/summary-CAS writes.
Its bounded canonical deletion helper accepts at most 100 IDs. Same-name empty
replacement requires explicit maintenance proof. Both are Python maintenance
primitives, with no public HTTP/product-state deletion route.
Normal reader failure returns no revision; enabled writer begin failure prevents
Qdrant dispatch and lets the existing durable delivery path retry.

## Consistency preconditions

These are external operational requirements, not properties proven by INFO:

- Exactly one authoritative Redis endpoint is reachable by all participating
  writers/readers, and former primaries are fenced before a new one is used.
  All participants must use the same derived authority scope for the same physical
  corpus. Qdrant endpoint aliases/configuration changes require coordinated
  invalidation and retirement; a second scope cannot authorize concurrent writers.
- All repository writers and managed create/delete/replacement paths participate.
  Mixed old/new writers or unrestricted raw Qdrant admin mutations invalidate reuse.
- Redis administrative restore/rollback and authority-key access are managed.
  Same-process raw RESTORE can revive old READY without changing INFO identity.
- Recovery has explicit evidence that all old writers are retired and every old
  Qdrant operation completed. A client timeout, empty record or quiet period is
  insufficient: server-side work may still settle later.

Real isolated Redis7.4.5 experiments reproduced same-process promotion changing
master_replid while keeping run_id, old READY rejection after restart/restore,
and the limits above. Two simultaneously reachable primaries and same-process
administrative rollback cannot be detected by a local role/epoch check. Redis
replication/WAIT does not provide strong consistency or a cross-store transaction.
See [Redis replication](https://redis.io/docs/latest/operate/oss_and_stack/management/replication/)
and [Qdrant consistency](https://qdrant.tech/documentation/scaling/consistency-guarantees/).

## Configuration and maintenance

`CORPUS_AUTHORITY_ENABLED` defaults false. Enabling additionally requires
`CORPUS_AUTHORITY_MANAGED_BOUNDARY=single-endpoint-fenced-admin-v1` as an explicit
operator attestation of the external preconditions. This value does not enforce
a topology or prove an ACL. A scoped `CORPUS_AUTHORITY_REDIS_URL` may be supplied;
otherwise the existing REDIS_URL is used. No URL/credential is printed. No
production ACL/configuration is changed by this implementation.

The minimal script commands are EVAL, INFO, HGET and HSET on the one derived
`weave:corpus-authority:v1:*` key scope. INFO capability and synthetic scoped ACL
are tested on Redis7.4.5; unsupported/denied commands fail closed, not a reason to
automatically raise privileges or silently replace the protocol with a counter.
The authority key includes the Qdrant endpoint/collection/vector scope digest.
No user ID, query, exclusion, repository payload or feedback work is stored there.

```bash
uv run python -m scripts.corpus_revision status
uv run python -m scripts.corpus_revision recover  # dry-run, never clears DIRTY
uv run python -m scripts.corpus_revision invalidate  # dry-run, changes nothing
```

Recovery execution additionally requires a bounded maintenance proof ID and all
four explicit retirement/completion/fencing/admin assertions. These are claims
made by the maintenance operator from external evidence; the CLI cannot discover
that evidence or stop old Qdrant server work. It has no timeout-based override.
Do not run recovery against shared/production targets without operational authority.

Rollout first deploys the code with reuse disabled. Under controlled maintenance,
pause/drain repository writes and prove all old server operations settled; retire
all pre-protocol instances. Configure every participating instance with the enabled
protocol and externally enforced topology/admin boundary. With no trusted record,
readers are uncached and enabled writers refuse dispatch. Bootstrap only after all
maintenance proofs are established, then verify status and the selected target.

Before rollback or raw administrative mutation, execute an authorized explicit
invalidation and confirm UNTRUSTED at the single authoritative endpoint. It keeps
all unknown writer tokens and refuses new protocol writes; no old worker can
finish back to READY. Only then retire/drain old instances/operations and disable
reuse everywhere before untracked writers return. Recovery requires fresh proof,
never a timeout. This availability tradeoff is deliberate. Feedback streams are
unrelated durable work and must never be deleted or used as authority storage.

## Verification scope

```bash
RUN_CORPUS_AUTHORITY_INTEGRATION=1 uv run pytest tests/test_corpus_revision.py
```

The opt-in fixture creates and removes its own Redis7.4.5 --network-none container;
it accepts no shared Redis URL. It tests the actual Lua protocol. The local-Qdrant
callback test performs a real write then raises a simulated lost ack. Separate actual API-factory acceptance uses fresh Redis7.4.5/Qdrant1.18.2
containers on an owned bridge with loopback ephemeral ports. A submitted write
settles after the client timeout; DIRTY persists even after completion until proved
recovery. Live uncached retrieval sees content version 2, not the old cached pool.
All owned containers/networks/threads are cleaned up. Neither fixture uses shared
URLs, downloads a model or changes production credentials/configuration.

```bash
RUN_CORPUS_AUTHORITY_INTEGRATION=1 MIN_ELIGIBLE_REPOSITORIES=0 uv run pytest tests/test_corpus_revision_operations.py -k actual_api -s
```

The latter command emits measurements. On 23 October 2026, its 20 synthetic
canonical points/10 retained candidates used four cold Qdrant discovery reads and
zero additional warm reads, with two Redis EVALs in each case. One observed pool
miss/hit took 26.80/2.98 ms; cache age was 0.02872 seconds. The two-request hit
fraction was 50%, not a representative traffic estimate. Four languages remained
in the candidate slice. Dirty/recovered lookups each made four fresh Qdrant reads
and observed the completed content version. These are direct discovery calls
through actual API factories, with schema initialization excluded and hybrid mode
explicitly selected; they are not full HTTP/heavy-model or production speedups.
Retriever source SHA256 was
`42a8ce81920503b7a606fd72c1d6c2db91b075c0bc1d671df19f3e38863d914b`.
Existing independent single-flight/TTL/failure/memory/refill/personal-isolation
checks remain part of aggregate coverage. No production activation, failover
topology, recommendation quality or representative performance is claimed.
