# README fetch strategy validation (#76)

This is preparation for the issue's required human-approved GitHub experiment.
It does not select a new strategy or establish production/API cost results.
Current acquisition already uses metadata-first immutable reads with a bounded
legacy compatibility path. No runtime configuration changes are included here.

Run the no-network proposal from the repository root:

```bash
.venv/bin/python scripts/compare_readme_strategies.py
.venv/bin/python -m pytest tests/test_readme_strategy_comparison.py
```

The proposed sample lives in `tests/fixtures/readme_strategy_sample.json`:
the two Weave projects plus eight public projects with varied documentation.
These names are a proposal, not verified evidence of their present README sizes,
branches, visibility or coverage of every edge case. A reviewer should approve
or replace them before a live run. Synthetic fixtures cover nested/non-default
branch paths, precedence, empty and partial responses, and immutable blob failure.

## Matched comparison

`compare_repository` accepts an injected transport and exercises existing source
parsers directly. Neither strategy uses acquisition, blob or documentation caches.
One shared reference lookup resolves a PUBLIC repository and immutable commit.
Both strategies read that same commit; a moving HEAD cannot fake a mismatch.

- Aliases: one query returns text for every supported candidate, using the existing
  legacy selector and path precedence.
- Metadata: one query returns candidate metadata; at most one further query reads
  and verifies the selected canonical blob's size and Git identity.

Report shared lookup costs separately, then calls, elapsed milliseconds, normalized
decoded JSON bytes, actual GraphQL `rateLimit.cost` (unknown stays null), partial
response flags, content digest/length, source path and branch for each strategy.
No raw README, provider error text or credentials belong in the report. Missing
and failed/incomplete are different outcomes. A usable partial response does not
prove complete resolution of higher-priority aliases; inspect its error flag.

The legacy media base uses a mutable branch while metadata-first uses the pinned
commit. `media_base_equal` reports this difference separately from content/path
parity. Do not misrepresent the URLs as identical. Readme source metadata must
continue supporting relative images and links after a strategy decision.

## Required live validation and safeguards

The CLI deliberately exposes only a plan. No live transport is installed. Before
adding/executing it, record human approval of the environment and concrete sample
required by #76. Use the stored github.com developer login and verify its actual
token subject is `TarunRam-git`; never use the service `GITHUB_TOKEN` or trust an
environment override as developer identity. Never print that token.

The prepared budget is at most 10 repositories and 40 GraphQL attempts, plus one
REST identity check. Use a fixed HTTPS GitHub endpoint, no redirects/proxies,
bounded socket and total-session deadlines (180 seconds), streaming response caps
(32 MiB decoded per call), no retries, and stop before fewer than 100 remaining
rate-limit points. `Measurement` counts failed attempts and validates injected
response envelopes, but cannot enforce streaming/socket limits on an arbitrary
transport; a live adapter must enforce those itself before parsing any body.
Never call it public-safe merely because the sample names look public: check
repository visibility before requesting content. Do not persist private results.

Use equivalent, order-balanced repeated samples for p50/p95. A single ten-project
pass is a compatibility pilot and cannot establish representative tail latency.
Record exact source commits, sample approval, GitHub date, partial/throttle/errors,
request and actual cost totals, byte scope, and coverage gaps. No estimated cost
should be presented as a returned API measurement. Missing rate-limit telemetry
or incomplete canonical content makes that part of the experiment inconclusive.

## Decision and rollback

Review canonical content/path parity, media behavior, bytes, calls, cost and error
rate together. Metadata can save body bytes while adding round trips; do not adopt
it solely from a local fixture's latency. Keep the existing compatibility path
unchanged until the human review chooses a strategy. Any later runtime change
needs its own reviewed diff, regression checks and coordinated rollback to the
previous qualified source version. This tooling neither toggles production nor
authorizes a live experiment, and #76 remains open until its live evidence and
human decision requirements are satisfied.
