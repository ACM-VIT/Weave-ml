"""Reproducible, network-local admission diagnostic, not a production load test.

Run: python3 -m tests.benchmark_model_operation_admission
Requires redis-server on PATH. Each trial clears only its owned Redis process.
Both lanes validate the complete batch before admission; the legacy lane then
reparses each envelope, matching the previous route. Timings include validation,
Lua execution and replies, but exclude fixture setup and cleanup. Commands count
EVAL calls, and round trips count synchronous EVAL or pipeline execute calls;
Redis-internal Lua commands and setup are excluded. No artificial RTT is added.
"""
import json
from statistics import median
from time import perf_counter

from feedback.model_wire import parse_operation
from tests.test_model_operation_batch import local_redis, CountedRedis, stream, originals


def main():
    # Use the same explicitly owned Redis fixture as the acceptance tests.
    fixture = local_redis.__wrapped__()
    client = next(fixture)
    report = []
    try:
        for same_user in [False, True]:
            for size in [1, 10, 50]:
                raws = originals(size, same_user=same_user)
                for lane in ['serial', 'batch']:
                    times = []
                    for trial in range(110):
                        client.flushdb()
                        counted = CountedRedis(client)
                        s = stream(counted)
                        started = perf_counter()
                        if lane == 'serial':
                            for raw in raws:
                                parse_operation(raw, maximum_bytes=s.limits.input_bytes, allowed_policies=s.policies)
                            [s.admit(raw) for raw in raws]
                        else:
                            s.admit_batch(raws)
                        elapsed = (perf_counter() - started) * 1000
                        if trial >= 10:
                            times.append(elapsed)
                    report.append(dict(workload='same-user' if same_user else 'distinct-users',
                                       size=size, lane=lane, commands=counted.commands,
                                       roundtrips=counted.roundtrips,
                                       p50_ms=round(median(times), 3),
                                       p95_ms=round(sorted(times)[94], 3)))
        print(json.dumps(report, indent=2))
    finally:
        try:
            next(fixture)
        except StopIteration:
            pass


if __name__ == '__main__':
    main()
