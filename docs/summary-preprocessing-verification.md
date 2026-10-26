# README preprocessing verification (#86)

The current main branch already implements bounded source analysis and a
provider-disabled metadata fallback. This follow-up verifies those behaviors;
it does not change the serving policy or claim a new production speedup.

Run the network-free profile from the repository root:

```sh
python -m scripts.profile_summary_preprocessing --repeats 5
```

The report separates CPU timing from traced peak Python allocations. Input
construction and imports happen before measurement. CPU sampling reads the clock
between batches of calls, doubling batches up to 4,096 calls to amortize clock
overhead. Sampling targets 100 ms with a hard cap of 100,000 calls;
`cpu_target_reached` reports whether that target was reached. All rows below
reached it. This avoids per-call clock overhead and misleading zero timings
on coarse process clocks.
Measurements include warm interpreter effects and should be compared on the
same host. They exclude provider transport, canonical content hashing outside
the summary pipeline, ingestion, and embedding. No external services or model
downloads are involved.

Local results refreshed on 2026-10-26 (Windows, Python 3.12.13), replacing the
initial measurements that included a clock read on nearly every iteration:

| Input | Characters | Analysis CPU per call (ms) | Peak traced bytes | Outcome |
| --- | ---: | ---: | ---: | --- |
| Prose | 16,000 | 3.13 | 132,324 | Bounded prompt |
| Prose | 64,000 | 13.54 | 528,899 | Bounded prompt |
| Prose | 1,000,000 | 0.0015 | 2,152 | Character-limit fallback |
| Prose | 4,000,000 | 0.0015 | 2,152 | Character-limit fallback |
| Unclosed markup | 16,000 | 3.13 | 130,909 | Bounded prompt |
| Unclosed markup | 64,000 | 18.75 | 521,709 | Bounded prompt |
| Unclosed markup | 4,000,000 | 0.0015 | 2,152 | Character-limit fallback |
| Many short lines | 16,000 | 4.91 | 487,975 | Line-limit fallback |
| Many short lines | 64,000 | 5.36 | 487,975 | Line-limit fallback |
| Many short lines | 4,000,000 | 0.0015 | 2,152 | Character-limit fallback |

Provider-disabled runs on four-million-character inputs used 3,559–3,606 traced
bytes and 0.086–0.098 ms per call. All returned the identical metadata fallback.
These are observations from one run, not CI timing thresholds.

`tests/test_summary_preprocessing.py` checks that disabled providers never read
the source object, hash source identity, or invoke the parser, for either overflow
policy. It covers character and line rejection before cleanup, exact LF/CRLF/
Unicode line boundaries, unchanged canonical Markdown, and tail/derived evidence
retention when only the prompt is clipped. Existing golden, prefix-policy and
copy-detection tests remain the source of truth for generated output safety.

Validation: the initial 156 tests passed across preprocessing, prefix, card-summary
and reuse suites. After the timing correction, all 10 preprocessing/profile tests
passed, including deterministic checks for batched clock reads and the hard call
cap. The profile is diagnostic; it deliberately has no wall-clock pass/fail thresholds.
