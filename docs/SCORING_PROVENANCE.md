# Exact scored-input provenance: private preparation

`inference.scoring_provenance` provides an internal request-scoped capture and a
lossless packet codec for the actual778-value float32 heavy-ranker input. No
registered HTTP route, environment flag, retriever call, database writer or export
command activates it. Existing `score_batch` and `score_batch_compact` signatures,
scores and response fields remain unchanged. This is one prerequisite for ML26,
not a completed impression dataset or authorization to retain user vectors.

## Fidelity and lifetime

A trusted in-process caller may enter `capture_scored_inputs(context)` around a
ranker call. The caller must derive the context from the profile actually loaded,
including canonical owner/generation UUIDs, profile/feedback versions, effective
user-vector hash, embedding identity and code revision. The ranker verifies the
context against its actual loaded model/scaler/manifest hashes. These hashes are
computed from the same bytes parsed/loaded at construction, not a later reread of
possibly changed artifact paths.

Capture copies the request-owned contiguous float32 matrix after dense feature
scaling/clipping and before device transfer. Each row includes the exact effective
user vector, exact repository vector and ten effective dense values. Canonical
repository ID, content/feature versions, source/content hashes and embedding
version come from that same scored candidate payload. Missing metadata, duplicate
IDs, mismatched owner inputs/artifacts and invalid/nonfinite dimensions fail the
opt-in operation; historical inputs are never reconstructed from current Qdrant.
The scope permits at most150 unique rows and clears its retained references on
exit, including exceptions. Clearing references is not secure memory erasure.
Nested scopes are rejected; concurrent contexts are isolated.

The packet uses little-endian float32/base64 and binds canonical context, snapshot
and input bytes with SHA256. This is integrity checking, not authentication,
consent or an export permission. Legacy profiles have no stored user recipe
identifier: explicit null remains valid and `scoring_metadata_complete=false`.
Exact numerical replay is distinct from complete recipe metadata. Version fields
are canonical signed64 decimal strings to avoid JavaScript rounding.

`seal_scoring_response` requires a normally validated heavy-response envelope and
actual scored frames, preserves scores/source fields and returns exact UTF8 bytes.
Emit those exact bytes after any future contract negotiation. It rejects oversize
without trimming candidates or silently dropping provenance. No current API emits
this packet or bypasses its response validator.

## Verification and measured cost

Tests exercise the checked-in trained artifact on CPU: the packet bytes match the
actual model-forward tensor, replay produces identical scores, default calls are
unchanged, two concurrent owners remain isolated, and errors clear scope state.
The artifact was trained on synthetic data and manually qualified previously;
this test proves numerical fidelity, not real recommendation quality.

The exact whole-response budget is tested for every output count0–150 with maximal
bounded metadata,8 embedding recipes,16 source scores, signed64 versions, extreme
finite negative floats,40-character UTC time and Unicode/control escaping. The
largest tested legal envelope is994,973 bytes against a1,000,000-byte cap. An
arbitrary larger envelope fails explicitly; it is not truncated.

A local matched diagnostic used actual checked-in weights, synthetic candidate
vectors/payloads, CPU with one process-local torch thread, warmup and10 interleaved
samples per variant. Median process CPU milliseconds:

| Candidates | Default score | Capture | Capture and seal | Sealed bytes |
| --- | ---: | ---: | ---: | ---: |
|1 |0.63 |0.77 |1.04 |6,254 |
|10 |1.14 |1.87 |3.13 |48,569 |
|45 |2.61 |5.67 |10.80 |213,122 |
|150 |7.40 |18.07 |34.37 |706,795 |

These are local input/serialization costs, not HTTP, storage or production latency.
No runtime device/thread setting changed. Maximum rows require approximately467KB
of raw copied inputs in addition to serialization buffers and ordinary inference.

## Required before any capture activation

- Wire trusted loaded-profile/candidate context and a generated paired response
  contract without changing the default score/public feed API. The current profile
  fetch projection omits profile/feedback revisions; obtain those from the SAME
  retrieved point before activation, never a later read or the backend accepted
  feedback version. Current profile-fetch measurements do not include that future
  additional field transport.
- Backend-owned private immutable generation+repository storage, idempotent equal
  retry/conflicting-hash handling and persistence before queue publication. Link
  PER serve item: refills can mix generations, while the current serve header
  records only the first item's generation. Never add product SQL to online ML.
- Enforced storage/read/export roles, authenticated owner binding, selected bounded
  retention, expiry/account-deletion cleanup and scoped export with private0600
  files. These are not implemented here; a disabled flag alone is insufficient.
- A point-in-time export with explicit received-at watermark/label horizon,
  late-arrival/censoring and original event/source/position linkage. Every durable
  serve is a served exposure with unknown rendering until real client evidence;
  it is not a fabricated impression or a negative label. Qualified no-action
  impressions and fast action-only exposures must both survive export.

Full ML26 remains open. Existing historical vectors predating capture are unknown;
no current-vector join can repair them. There is no production capture, retention
policy selection, corpus refresh, training promotion or historical rewrite here.
