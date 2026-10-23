# Offline embedding model comparison

Issue #32's offline path is separate from serving. Changing this registry does
not change the production MiniLM encoder, repository/user collections, feedback
vectors or 778-input ranker. This evaluator cannot promote a model.

`tests/fixtures/embedding_models.json` pins three configurations: current serving
MiniLM at 384 dimensions, Qwen3-Embedding-0.6B at 384, and Qwen at 1024. The MiniLM
revision is the existing production pin, not the latest model-branch head. Qwen
revision was resolved from the public model repository on 23 October 2026 and frozen
as `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. No remote custom code is enabled.

The model card specifies native 1024 dimensions with Matryoshka outputs 32..1024,
last-token pooling and query-side instructions. Documents receive no instruction.
See the [Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) and
[Sentence Transformers API](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).
These upstream claims are not Weave-specific quality or operational evidence.

## Run

```bash
uv run python -m scripts.compare_embedding_models \
  --registry tests/fixtures/embedding_models.json \
  --fixture tests/fixtures/readme_retrieval.json \
  --models minilm-384 qwen3-384 qwen3-1024 \
  --threads 2 --batch-size 1 --repeats 2 --k 3 \
  --output /tmp/weave-embedding-comparison.json
```

Models must already be cached at their exact revisions. `--allow-download`
explicitly permits downloading public pinned model assets into the local cache;
credentials are not supplied. Each selected configuration runs serially in a
fresh CPU process with a ten-minute worker timeout and bounded threads, batches,
repetitions and fixture sizes. Failure exits nonzero and does not replace the
aggregate report with a partial success. Successful child evidence remains
available for diagnosis. No network service, product database or Qdrant client
is used by the comparison path.

One document warmup is excluded from timing. Document encoding is measured over
the entire frozen corpus, at the requested batch size. Query latency samples
measure each query separately. The report records actual model load time,
corpus encoding time/throughput, query p50/p95 and sample arrays/counts, process
peak RSS, library versions, dtype, thread count, configuration, source hashes,
fixture and registry hashes, and creation time. PeakRSS covers imports/model
load/warmup/inference in that fresh worker; it is not model-only memory.

Input text is the same frozen full name, description, topics and README prefix
recipe for each model. Character and token bounds, pooling, normalization,
revision, selected dimensions and both role prompts are part of the embedding
version. Token truncation uses each model's own tokenizer. The initial 256-token
budget is deliberately bounded and does not benchmark Qwen's full context window.
This is a raw text-retrieval diagnostic, not the production weighted README /
metadata / topic tower or heavy ranker.

## Identity and shadow artifacts

Registry entries include model ID, immutable revision, native/output dimension,
maximum tokens, pooling, normalization, instruction capability, query/document
prompts, float dtype and input recipe. Add/select an entry through JSON and
`--models`; no production source edits are needed. The loader verifies native
dimension, available context and pooling against the actual loaded model.

`encode_role` always supplies the role prompt explicitly, truncates to the chosen
dimension and normalizes afterward. It rejects incomplete, nonfinite,
non-floating, incorrectly sized or non-normalized outputs. `compatible_scores`
requires matching full embedding identity and query/document roles before cosine.
An equal dimension alone never makes two encoders or prompt versions compatible.

Each measured worker writes a versioned `.vectors.npz` shadow alongside its JSON
report. It contains query/document matrices and identity metadata with the frozen
repository/query order. This supports side-by-side in-memory artifact evaluation
without touching production collections. Raw repository vector bytes and actual
compressed shadow artifact bytes are reported separately. **Neither is an ANN
index-size measurement.** `ann_index_bytes` remains unknown; no index was built.

The measured reindex cost is encoding the supplied frozen corpus only. A full
catalogue reindex and its CPU, time, ANN storage and migration costs remain
unmeasured; do not turn this small fixture into a production estimate.

## Evidence and acceptance boundary

The default frozen fixture contains 12 synthetic repositories and 12 queries with
synthetic binary relevance labels. Recall@K, NDCG@K and MRR are diagnostic only;
identical perfect scores would not establish representative recommendation
quality. Labels are neither human judgments nor real-event holdouts. Runtime
measurement is real model inference even though label provenance is synthetic;
those are separate claims.

Issue 32 remains open until a representative judged benchmark and actual index /
full reindex assessment justify a promotion decision. The current decision is
**do not promote**. A future production change requires separately versioned
repository AND user vectors, profile rebuild and compatible feedback/CAS state,
a full validated canonical reindex and eligibility audit, and a ranker/scaler /
feature manifest compatible with the selected encoder and dimensions. Never write new-model
vectors into existing 384-dimensional collections or stamp serving eligibility
with a payload-only patch. Keep old collections and compatible model artifacts
available for rollback until coordinated acceptance is complete. Switching the
offline registry does not authorize that migration or deployment.

## Measured diagnostic, 23 October 2026

Actual pinned-model CPU inference completed for all three configurations. Host:
Intel Core Ultra 7 255H, Linux x86_64, 15 GiB RAM; each fresh worker used two PyTorch
threads, float32, batch size 1 and a 256-token budget. Two complete corpus encoding
samples and 24 individual query samples per model followed one document warmup.
Workers ran serially. This was a shared development machine, not an isolated
production load test; sample counts do not establish production latency percentiles.

| Configuration | MRR / Recall@3 / NDCG@3 | Model load s | Corpus encode mean s | Documents/s | Query p50 / p95 ms | Peak process MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MiniLM 384 | 1 / 1 / 1 | 0.203 | 0.739 | 16.234 | 9.06 / 9.85 | 522.87 |
| Qwen 384 | 1 / 1 / 1 | 2.791 | 21.590 | 0.556 | 386.01 / 403.72 | 3840.10 |
| Qwen 1024 | 1 / 1 / 1 | 1.593 | 21.439 | 0.560 | 387.29 / 404.92 | 3840.20 |

| Configuration | Raw repository vector bytes | Compressed query+document shadow bytes | ANN index bytes |
| --- | ---: | ---: | --- |
| MiniLM 384 | 18,432 | 36,154 | Unknown; no ANN index built |
| Qwen 384 | 18,432 | 36,288 | Unknown; no ANN index built |
| Qwen 1024 | 49,152 | 93,381 | Unknown; no ANN index built |

Libraries: Sentence Transformers 5.6.0, Transformers 5.14.1, PyTorch 2.13.0+cpu.
The reported load times exclude the one-time Qwen model download. All models
used the same frozen input bytes and source implementation (commit `c90f877`):

- Fixture SHA256: `daafdef84724656040342ea37cf5d4a3f01eba6f72307e2b5979a99cf8c9af2d`.
- Registry SHA256: `6416feee389ae4224ad0124ebaa9347695af842954bd6bc1286492611cfaf8ea`.
- Registry module SHA256: `7fd61e3d7f27dd11288800177c1d0f6f6dc92b95059b0b2f9e5677551d09a1c9`.
- Runner SHA256: `2029f5c25434294a60e99eb55f7d687ce70b80e6b1d0ac17e3ebd2ae710e8a72`.

The fixture is too easy to distinguish quality: every model retrieves its known
synthetic relevant item first. Qwen's substantially higher measured CPU and memory
cost has no established quality benefit here. The evaluated outcome is **no
promotion**. This result neither proves MiniLM is best on real users nor qualifies
any model for a new workload. Representative judgments, real ANN index size,
full-corpus reindex and cross-system migration acceptance remain outstanding.
