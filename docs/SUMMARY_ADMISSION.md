# Summary provider admission

The online summary runtime uses its configured `REDIS_URL` for atomic admission,
without quota polling or sleeping. A missing/unavailable coordinator denies the
provider enhancement and retains the validated description fallback. Summary
work precedes encoder acquisition; it never holds the shared model semaphore
while waiting for quota. Existing bounded HTTP retry backoff and the embedding
completion reserve remain enforced. This adds no product PostgreSQL access.

`SUMMARY_RPM_LIMIT` is a shared fixed-minute request ceiling (fractional rates use
one request per correspondingly longer window). `SUMMARY_TPM_LIMIT` defaults to
120000: each attempt reserves the serialized input's UTF-8 byte count plus its
maximum output tokens. This deliberately conservative bound includes the prompt
and schema, and charges retries/repairs without refunds. It is not a billing
measurement. Fixed windows permit bursts at boundaries; they are not a rolling
60-second traffic guarantee. Redis server time owns windows and leases.

All summary processes using the same API endpoint/key share the request/token
budget, including across models. Namespace keys contain hashes, not credentials.
Differing quota configurations in a live window fail closed; coordinate settings
changes across processes and allow old quota TTLs to expire before expecting the
new budget. The Redis runtime ACL must allow the `weave:summary:*` namespace and TIME, EVAL,
HGET/HSET/HINCRBY, PEXPIRE, SET and EXISTS. Until separately configured in an
authorized rollout, denied commands fail closed to fallback.

Different consumers of the same provider key outside this runtime
are not covered. Directly instantiated/injected offline providers use an explicitly
instance-local policy unless a shared admission object is supplied.

Three availability failures (`SUMMARY_FAILURE_THRESHOLD`) open a model-scoped
circuit for 30 seconds (`SUMMARY_COOLDOWN_SECONDS`). One time-limited probe
is admitted after cooldown; its permit lasts the configured HTTP timeout plus
one second. Recovery and opening advance epochs so late older outcomes cannot
close/reopen newer circuit state. Abandoned pre-HTTP attempts release probe
ownership without labeling a caller deadline as provider unavailability.
HTTP/envelope failures affect availability; invalid generated content retains
its existing validation, one repair, and safe fallback behavior.

Provider failures defer same-content fallback upgrades for the cooldown in Redis.
The key includes canonical repository/content hash and model/prompt/format identity;
changed content is immediately eligible. Markers expire and do not turn a fallback
into a permanent generated result. During coordinator failure, retain an already
accepted fallback instead of attempting another provider call. Local test/offline
markers are bounded to 1024 entries; shared markers have bounded TTLs and key sizes.

The authenticated metrics endpoint exposes nine fixed outcome counters under
`ml_summary_provider_events_total`, including probes, circuit/quota rejection,
coordination failure and fallback deferral. Scraping performs no Redis/provider
work and does not initialize the summary runtime.

Run deterministic provider tests with `uv run pytest tests/test_summary_admission.py`.
For real Lua and separate-process quota coverage, start a disposable Redis on
loopback port6397, then set `SUMMARY_TEST_REDIS_URL=redis://127.0.0.1:6397/0`.
The fixture rejects other hosts/ports, uses unique namespaces and never clears a
shared database. No external provider calls or production credentials are needed.
