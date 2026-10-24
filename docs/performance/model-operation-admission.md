# Ordered model-operation batch admission

`OperationStream.admit_batch` validates at most 50 envelopes before its first
admission and reuses parsed envelopes without reserializing original UTF-8.
Eligible EVAL scripts use nontransactional pipelines bounded to 50 commands and
one MiB of RESP command bytes, including repeated Lua source and keys. Lua
retains its atomic duplicate, conflict, capacity, withdrawal and guard checks.

Profiles are synchronous barriers: previously queued admissions finish before
profile preflight, and later work waits for its result. With ready-predecessor
recovery enabled, a repeated user starts another window so capacity recovery is
resolved before sending later work for that user. Concurrent producers retain
Redis's existing script ordering; no cross-producer total ordering is promised.
Same-user-heavy requests therefore retain synchronous round trips. Withdrawals
continue to use the existing individually atomic, synchronous path.

A pipeline can partially commit. Capacity, per-command errors, incomplete replies
and connection failures produce no successful batch response. Retry the exact
complete original batch; committed scripts prove duplicates, and uncommitted
scripts remain eligible for admission. The client cannot infer rollback from a
503. The recovery slot remains exactly one global entry above normal capacity;
it does not bypass identity or initialization-guard budgets.

## Reproducible local diagnostic

Run `python3 -m tests.benchmark_model_operation_admission`. The benchmark starts
an owned, persistence-disabled local Redis process using a Unix socket and
terminates it afterwards. Each case uses 10 warmups plus 100 timed fresh batches
of valid Begin envelopes with the same controlled policy and payload shape.
Setup and clearing owned fixture state are outside timing. Both lanes validate
all inputs first; serial then reparses each input as the previous route did.
Batch validates once. No artificial network delay is added. Recovery is enabled
with a callback that reports no ready predecessor; these uncrowded workloads do
not invoke it. Production proof lookups, contention and Qdrant are not measured.

Measured on macOS, Python 3.14.2, Redis 8.10.2, 24 October 2026. Times in ms:

| Workload | Size | Serial EVAL / trips | Batch EVAL / trips | Serial p50 / p95 | Batch p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Distinct users | 1 | 1 / 1 | 1 / 1 | .083 / .102 | .067 / .084 |
| Distinct users | 10 | 10 / 10 | 10 / 1 | .801 / .878 | .428 / .482 |
| Distinct users | 50 | 50 / 50 | 50 / 1 | 4.128 / 4.375 | 2.015 / 2.087 |
| Same user | 1 | 1 / 1 | 1 / 1 | .086 / .105 | .073 / .089 |
| Same user | 10 | 10 / 10 | 10 / 10 | .792 / .870 | .624 / .708 |
| Same user | 50 | 50 / 50 | 50 / 50 | 4.106 / 4.238 | 3.040 / 3.189 |

EVAL counts exclude Redis-internal Lua calls. Trips count synchronous EVAL and
pipeline execute calls, excluding setup and fixture clearing. These measurements
show local call reduction and parser reuse; they are not production latency or
throughput claims. Deployment/load testing remains separate.

The deterministic suite in `tests/test_model_operation_batch.py` uses its own
local Redis process and covers exact bytes, conflicts and duplicates, complete
validation, count and byte limits, concurrent producers, partial capacity,
ready recovery, profile/withdrawal ordering, guard capacity, real command errors,
injected response loss and malformed replies. Existing opt-in full-runtime
Redis/Qdrant/Docker tests remain separate.
