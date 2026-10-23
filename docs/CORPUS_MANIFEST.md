# Backend corpus inventory for acquisition

Offline acquisition reads `GET /api/internal/v2/ingestion/repositories/manifest`
with `x-internal-secret`. Online ML gains no PostgreSQL dependency. Deploy the
paired backend endpoint before ML to enable deduplication; an older/unavailable
backend leaves the existing idempotent enrichment/upsert path operational and
reports `manifest_status=unavailable`. A verified empty inventory reports `empty`.
No partial scan is adopted. `get_repo_count()` still counts accepted work in this
invocation, not the global catalog size; the manifest does not change target-count
semantics or promise a consistent catalog total.

Version 1 pages accept `limit=1..250` (default 250) and an opaque `cursor`. Responses
contain `version`, `as_of`, `items`, and nullable `next_cursor`. Entries contain
backend `repo_id`, decimal-string `github_id`, normalized `full_name`, content hash,
positive content version, `content_updated_at`, and nullable `source_updated_at`.
They contain no README, vectors, summaries, credentials or product user data.

Pagination uses the canonical UUID primary key and a signed, domain-separated
cursor binding limit, cutoff, last UUID, contract version and a 15-minute expiry.
Invalid, changed-limit, unsupported and expired cursors return 409; invalid query
shapes return 400. Dependency failures return 503 and a true empty page returns 200.
Each query has a two-second database statement budget and reads at most 251 rows.
No schema migration, long-lived SQL snapshot or Redis inventory is required.

This is a conservative cutoff inventory, **not snapshot isolation across HTTP
pages**. Records created or changed after `as_of`, inactive repositories, and rows
without content are omitted. A concurrent change can omit an entry, which leaves
it eligible for enrichment rather than falsely suppressing unknown work. The
client validates stable cutoff/version, monotonic canonical IDs, numeric GitHub
identity, case-insensitive name uniqueness and content/freshness types. It rejects
repeated cursors, mixed/partial pages and bodies above 512 KiB. A scan is bounded to
100000 records, 400 requests and 120 seconds. Exceeding any limit reports unavailable
and keeps normal acquisition. This is not an incremental change-log protocol.

Acquisition caches one successful manifest per client invocation. Fresh entries
are excluded before enrichment; stale candidates discovered by the existing
strategy remain refreshable. `ACQUISITION_REFRESH_AFTER_SECONDS` defaults to 86400
and accepts 1..2592000 via the offline client's settings. It uses canonical content
update time, conservatively: an unchanged refresh that the backend does not write
may remain stale and eligible. This does not implement the complete per-source
refresh scheduler tracked by #85, or assert that GitHub has not changed since the
last observation. Successful upserts join the invocation's exclusion set without
requiring another scan. The cache never replaces canonical backend persistence.

Compatibility and rollback: publish backend first, then ML. Rolling ML back only
loses the optimization; rolling backend back makes the new client report unavailable
and perform ordinary enrichment. No old write contract is changed. Never deploy or
run acquisition against a shared target merely to validate this endpoint.

The owned inventory HTTP path uses cancellable async I/O with the remaining scan
deadline covering connection, headers and body, including slow Content-Length
streams. It closes the actual stream before synchronous consumption returns.
The explicit continuation field is required even on the final page. Accepted
names are held separately from the downloaded inventory cap, including when the
manifest already contains 100000 records or inventory is unavailable.
