# Offline recommendation benchmark

This is the infrastructure portion of issue #33. It scores **supplied frozen
outputs**, separately for retrieval, ranking and final feed assembly. It does not
run the live recommendation service, produce human judgments, qualify the current
ranker, or change serving configuration. No online module imports this package.

The checked-in bundle, orders, vectors, labels and timing samples are entirely
synthetic correctness diagnostics. They are not a representative quality report,
measured model latency, a promoted baseline or evidence for model promotion.
The full issue remains open until those genuine inputs and decisions exist.

## Reproduce the diagnostic

From the repository root, after `uv sync --locked --dev`:

```bash
uv run python -m scripts.evaluate_recommendations \
  --bundle tests/fixtures/recommendation_benchmark/bundle.json \
  --run tests/fixtures/recommendation_benchmark/run.json \
  --baseline tests/fixtures/recommendation_benchmark/baseline.json \
  --policy tests/fixtures/recommendation_benchmark/policy.json \
  --output /tmp/weave-recommendation-diagnostic.json

uv run pytest tests/test_recommendation_benchmark.py \
  tests/test_benchmark_gates.py tests/test_recommendation_benchmark_cli.py
```

Exit status is 0 for passing requested checks, 1 for failed gates, and 2 for
invalid input. The output contains a report, assessment and optional qualification
manifest. Gate failures retain their report for inspection. Invalid input does
not replace an existing output; always inspect the exit code. Output replacement
is atomic and cannot target an input file, including through a symlink. Each JSON
input is limited to 64 MiB; duplicate keys and nonfinite numbers are rejected.

Adding `--qualify` to the diagnostic **must fail**. Synthetic evidence, synthetic
timings and diagnostic baselines cannot qualify a model even when their smoke
gates pass. No command creates or activates a promoted baseline automatically.

## Versioned inputs

The strict Pydantic models in `evaluation/recommendation_benchmark.py` and
`evaluation/benchmark_gates.py` define version 1; unknown fields and invalid types
fail. Each model exposes `model_json_schema()` for consumers. The small committed
JSON files demonstrate the entire wire format. Do not put private user exports
or real credentials in these fixture paths or commit them.

- **Bundle:** snapshot identity, timezone-aware creation time, provenance,
  embedding identity, frozen corpus, intents/profiles, exclusions and graded
  judgments. Repository/profile IDs are canonical UUIDs. Synthetic UUIDs belong
  only to the isolated test fixture; real snapshots use backend-issued IDs.
- **Embedding identity:** model, immutable 40-character revision, dimensions,
  pooling, normalization, query/document prompts and input recipe. Repository
  vectors must be finite, normalized and dimensionally compatible. Run and bundle
  identities must match exactly before cosine calculation. The diagnostic uses
  three dimensions and cannot stand in for the production 384-dimensional model.
- **Provenance:** `synthetic`, `human_judgment` or `event_holdout`, plus an audit
  reference. Real holdouts require cutoff < label-horizon end <= ingestion
  watermark. The declaration cannot prove authenticity or point-in-time feature
  correctness: those need an externally audited export and collection process.
  An automated proxy label must not be declared a human judgment. Do not export
  raw identity, private activity or unconsented profile material.
- **Run:** model/scaler digests, feature schema, code commit, creation time,
  embedding identity and timing kind. Every frozen profile supplies each stage's
  ordered candidates, finite score, primary retrieval source, source score and
  latency sample. The producing runner must record its actual stage boundaries,
  model/feature/code identity and measured inference timings. This evaluator
  cannot authenticate those observations from a JSON declaration.
- **Policy:** K, minimum profile count, explicit metric limits and allowed absolute
  regression deltas. Metric directions are fixed in code. The committed limits
  exercise synthetic correctness; they are not production quality thresholds.
- **Baseline:** an explicitly selected diagnostic or previously promoted report,
  provenance reference, policy digest and report digest. The producing approval
  process owns designation of the last promoted baseline. Do not relabel the
  diagnostic to manufacture one.

Canonical SHA-256 content identities include all bundle and run fields, independent
of JSON formatting. Baseline comparisons require the same bundle, snapshot,
embedding identity, K, feature schema and policy. A new data snapshot requires an
explicit baseline evaluation on that snapshot; changing data cannot silently
improve a comparison. The candidate model/code may differ from the baseline.
No full model artifacts are loaded while scoring. For qualification review the
CLI additionally requires actual `--model-artifact` and `--scaler-artifact` files
and verifies their bytes against the supplied run digests.

## Metrics and counts

Metrics use the supplied order, without re-sorting or removing invalid candidates.
K applies independently to each stage. Unknown repository IDs fail validation;
known duplicate, excluded, archived and duplicate-fork outputs remain measurable.

| Metric | Definition |
| --- | --- |
| Judged Recall@K | Unique returned positive judgments / all positive judgments for that profile |
| Judged NDCG@K | Gain `2^grade - 1`, log2 discount, ideal from all judged corpus items |
| Judged MRR@K | Reciprocal rank of the first positive judgment within K |
| Judgment coverage | Judged returned positions / all returned positions |
| Catalogue coverage | Unique nonarchived returned IDs / nonarchived frozen corpus size |
| Intra-list similarity | Mean pairwise compatible-vector cosine, with pair count |
| Novelty | Mean `-log2((popularity+1)/sum_corpus(popularity+1))` from the frozen snapshot |
| Owner/topic concentration | Mean per-profile maximum attributed share; multiple topics split weight equally |
| Metadata coverage | Known owner/topic/fork-family positions / returned positions |
| Repeat / exclusion / archive rates | Violating returned positions / returned positions |
| Fork repeat rate | Extra positions per known fork family / returned positions |
| Source contribution | Returned positions by supplied primary source / all positions |
| Latency p50/p95 | Nearest-rank quantiles of supplied per-profile stage samples |

Each stage reports profile, judged-positive-profile, returned-position,
unjudged-position, similarity-pair and latency-sample counts. Empty outputs do
not invent relevance or similarity measurements. Profiles without judged
positives are excluded from relevance means and counted; gates reject incomplete
quality-profile coverage. Repeated IDs earn relevance gain only once.

Missing judgments are **unknown**, never stored as grade zero. The `judged_*`
metrics use only known positives/grades and are diagnostics under incomplete
judgments; their explicit coverage must accompany them. Qualification requires
complete top-K judgment coverage at every stage. It does not establish the
completeness or representativeness of the overall judged corpus by itself.

Owner/topic/fork metadata is not guessed. Null owners/families and empty topics
lower metadata coverage instead of looking like proven diversity. Final-assembly
fork caps are distinct from candidate-stage concentration. Source scores are
retained by the run digest; the summary reports primary source attribution, not
unobserved multi-channel support or causal quality uplift.

## Regression and qualification boundaries

`assess` rejects missing metrics/stages, nonfinite inputs, incompatible baselines,
insufficient samples, absolute-limit violations and regressions. Tests inject
real output-list regressions plus quality, diversity, safety and latency metric
regressions. Diagnostic comparisons deliberately cannot promote anything.

Qualification review additionally requires a promoted nonsynthetic baseline,
nonsynthetic candidate evidence, measured timings, complete judgments, metadata,
zero repeated/excluded/archived outputs, and rules covering quality, diversity,
latency and compatibility evidence. Passing produces a content-bound manifest
whose status is `evaluation_passed_requires_authorized_promotion`. It records the
data snapshot, feature schema, model/scaler hashes, code commit, metrics/report
digest and creation time. It is **not** a production approval, signature,
configuration update or deployment action. External provenance, thresholds,
representative coverage, real stage instrumentation and authorized promotion
still require review. The existing production manifest and manual qualification
are untouched. Rollback remains the previous compatible serving artifact pair.

Remaining #33 acceptance: collect genuine representative judgments with a rubric
and audited provenance; freeze actual stage outputs and measured model timings;
establish a reviewed promoted baseline and evidence-based thresholds; run and
inspect candidate comparisons; bind accepted reports into the artifact-producing
promotion process. Real-event time holdouts additionally depend on trustworthy
exposure and point-in-time feature exports from #26. Synthetic diagnostics must
never substitute for those missing results.
