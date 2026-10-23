# Immutable README cache acceptance

The canonical acquisition path revalidates repository access and the current
default-branch commit on every operation. It resolves README candidates at that
immutable commit and verifies the fetched blob's byte count and Git object hash.
Manifest, canonical blob and processed-document caches have separate identities
and bounds. Their implementation already exists; this package verifies that path
rather than replacing it with a repository-name cache.

Run from the repository root after installing locked development dependencies:

```bash
uv run pytest tests/test_immutable_readme_cache.py \
  tests/test_github_readme_acquisition.py tests/test_readme_scheduler.py -q
uv run python -m scripts.benchmark_readme_cache --iterations 10 --delay-ms 5
```

The new tests and benchmark use an actual loopback GraphQL server and the real
client, immutable cache and Markdown processor. The server models synthetic Git
objects; no GitHub endpoint, real credential, model, database or Qdrant is used.
It records HTTP request counts and JSON request/response body bytes. These are
measured transfers, distinct from the cache's `bytes_reused` accounting.

Acceptance covers unchanged content, branch/commit/path/blob changes, credential
and repository isolation, access revocation, opaque-auth bypass, missing versus
transient errors, partial candidate selection, hash/byte corruption, parser/media
context invalidation and mutable-document isolation. Six real concurrent clients
make six access checks but share one manifest and one blob request. Separate
producer/follower cancellation tests preserve healthy work and retry ownership.
Small bound tests exercise LRU eviction, oversized whole-artifact bypass and
in-flight producer admission. Failed content never becomes a valid empty blob.

## Measured controlled workload

Python 3.12, ten operations, one synthetic 58,060-byte README, 5 ms injected delay
per request. The uncached control disables retention but uses the same immutable
ref/manifest/blob protocol, real processor and identical output. It is not the
older inline-alias acquisition implementation or a live GitHub benchmark.

| Measurement | Retention disabled | Retention enabled |
| --- | ---: | ---: |
| Ref / manifest / blob requests | 10 / 10 / 10 | 10 / 1 / 1 |
| JSON response body bytes | 597,510 | 61,353 |
| First operation | 95.7 ms | 101.2 ms |
| Median of remaining nine operations | 79.3 ms | 11.1 ms |
| Accounted retained cache payload | 0 bytes | 232,295 bytes |

Retention avoided 18 requests and 536,157 response-body bytes in this run.
Manifest/blob/document stored-value hit ratios were each 9/10. Every warm
operation still revalidated access/ref. Cold acquisition did not improve. Timing
includes local HTTP/client/processing costs and the injected delay; it does not
establish real GitHub quota cost, production latency or cross-process reuse.

Cache snapshots report hits, misses, coalescing, evictions, failures, entries,
accounted payload bytes and in-flight producers. The reported stored-value hit
ratio is hits divided by hits + producer misses + coalesced followers. Payload
accounting excludes Python object/key overhead and is not an exact heap/RSS
measurement. Production bounds remain 512 manifests/2 MiB, 256 blobs/64 MiB,
256 documents/32 MiB and 32 producers per layer; oversized content is returned
whole without retention. No cache limits, source semantics or runtime settings
are changed by these acceptance files.

The script is a standalone diagnostic: it temporarily substitutes isolated
process-local cache objects and restores them on exit. Do not run it inside a
live service process. Existing full-suite CI discovers the unit-marked tests;
the benchmark remains an explicit bounded command without a flaky latency gate.
