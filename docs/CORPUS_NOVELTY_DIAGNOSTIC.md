# Corpus novelty diagnostic status

The compatibility ingestion path does not compare repository embeddings. Its
zero vector is a placeholder, so `NoveltyMatrix.measurement_status` is
`unmeasured`, and its final score and component penalties are `null`. Console
summaries must preserve that unknown state instead of ranking repositories or
printing a fabricated score. The path's existing prefilters and documentation
and code-health checks remain independent legacy behavior; they are not novelty
decisions.

The compatibility path may still collect local history for exploratory output.
`dynamic_cluster_discovery` reports a centroid distance only when it has at
least four finite, non-zero vectors with one shared dimension and at least two
observed categories. Missing, zero, malformed, mixed-dimension, or insufficient
vectors produce an explicit `unmeasured` result. This is descriptive topology,
not a novelty score, semantic label, threshold, or admission rule.

No bounded reference-set evaluation is included yet. The repository's existing
Qdrant nearest-neighbor tooling reads a configured collection; it does not by
itself provide a versioned, reviewed reference set or ground-truth groups for
forks/duplicates, adjacent ideas, and genuinely different projects. Such
groups, the intended role of novelty, and any thresholds require owner review.
Until then, admission penalties and novelty-based rejection remain disabled.
