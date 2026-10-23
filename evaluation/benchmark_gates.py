"""Offline regression and qualification checks; no model activation authority."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from evaluation.recommendation_benchmark import (
    Commit, Digest, Number, Provenance, STAGES, StrictModel, Text, aware_timestamp, digest,
)

# Directions are semantic, not caller-controlled knobs that can invert a gate.
DIRECTIONS = {
    'judged_recall_at_k': 'higher', 'judged_ndcg_at_k': 'higher', 'judged_mrr_at_k': 'higher',
    'judgment_coverage': 'higher', 'catalogue_coverage': 'higher', 'novelty_bits': 'higher',
    'owner_metadata_coverage': 'higher', 'topic_metadata_coverage': 'higher',
    'fork_metadata_coverage': 'higher',
    'intra_list_similarity': 'lower', 'owner_concentration': 'lower', 'topic_concentration': 'lower',
    'repeat_rate': 'lower', 'excluded_rate': 'lower', 'archived_rate': 'lower', 'fork_repeat_rate': 'lower',
    'latency_p50_ms': 'lower', 'latency_p95_ms': 'lower',
}
SAFETY = {'repeat_rate', 'excluded_rate', 'archived_rate'}
COUNTS = {'profiles', 'quality_profiles', 'returned', 'unjudged_returned', 'similarity_pairs', 'latency_samples'}


class StageReport(StrictModel):
    metrics: dict[str, Number | None]
    counts: dict[str, Annotated[int, Field(ge=0)]]
    source_share: dict[str, Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]]

    @model_validator(mode='after')
    def complete(self):
        if set(self.metrics) != set(DIRECTIONS) or set(self.counts) != COUNTS:
            raise ValueError('report must contain every metric and sample count')
        if self.counts['profiles'] < 1 or self.counts['latency_samples'] != self.counts['profiles']:
            raise ValueError('report has inconsistent profile/latency counts')
        if self.counts['quality_profiles'] > self.counts['profiles'] or self.counts['unjudged_returned'] > self.counts['returned']:
            raise ValueError('report has inconsistent quality/judgment counts')
        for name, value in self.metrics.items():
            if value is None:
                continue
            if name == 'intra_list_similarity':
                if not -1 <= value <= 1:
                    raise ValueError('invalid cosine metric')
            elif name in {'novelty_bits', 'latency_p50_ms', 'latency_p95_ms'}:
                if value < 0:
                    raise ValueError('invalid negative metric')
            elif not 0 <= value <= 1:
                raise ValueError('ratio metric must be between zero and one')
        return self


class Artifact(StrictModel):
    schema_version: Literal[1]
    model_id: Text
    model_sha256: Digest
    scaler_sha256: Digest
    feature_schema: Text
    code_commit: Commit
    created_at: str
    timing_kind: Literal['synthetic', 'measured']
    _timestamp = field_validator('created_at')(aware_timestamp)


class Report(StrictModel):
    schema_version: Literal[1]
    bundle_sha256: Digest
    run_sha256: Digest
    snapshot_id: Text
    # Reuse the same audited provenance parser as the input bundle.
    provenance: Provenance
    embedding_sha256: Digest
    k: Annotated[int, Field(ge=1, le=1000)]
    artifact: Artifact
    stages: dict[Literal['retrieval', 'ranking', 'assembly'], StageReport]

    @model_validator(mode='after')
    def complete_stages(self):
        if set(self.stages) != set(STAGES):
            raise ValueError('report requires all stages')
        if len({stage.counts['profiles'] for stage in self.stages.values()}) != 1:
            raise ValueError('stage profile counts differ')
        return self


class Baseline(StrictModel):
    schema_version: Literal[1]
    status: Literal['diagnostic', 'promoted']
    reference: Text
    policy_sha256: Digest
    report_sha256: Digest
    report: Report

    @model_validator(mode='after')
    def intact(self):
        if digest(self.report) != self.report_sha256:
            raise ValueError('baseline report digest mismatch')
        return self


class Rule(StrictModel):
    stage: Literal['retrieval', 'ranking', 'assembly']
    metric: str
    # Absolute limits, plus an absolute delta in the metric's own units.
    minimum: Number | None = None
    maximum: Number | None = None
    max_regression: Annotated[float, Field(ge=0, allow_inf_nan=False)]

    @model_validator(mode='after')
    def meaningful(self):
        if self.metric not in DIRECTIONS:
            raise ValueError('unknown metric')
        if self.minimum is None and self.maximum is None:
            raise ValueError('gate requires an absolute limit')
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError('inverted gate limits')
        return self


class Policy(StrictModel):
    schema_version: Literal[1]
    policy_id: Text
    reference: Text
    k: Annotated[int, Field(ge=1, le=1000)]
    min_profiles: Annotated[int, Field(ge=1)]
    rules: Annotated[list[Rule], Field(min_length=1, max_length=100)]

    @model_validator(mode='after')
    def distinct(self):
        keys = {(r.stage, r.metric) for r in self.rules}
        if len(keys) != len(self.rules):
            raise ValueError('duplicate policy rule')
        return self


def assess(report: dict, baseline: Baseline, policy: Policy, *, qualify: bool = False) -> dict:
    """Check evidence against an explicitly selected baseline; never promote it."""
    candidate = Report.model_validate(report)
    previous = baseline.report
    failures = []
    if baseline.policy_sha256 != digest(policy):
        failures.append('baseline policy digest mismatch')
    for field in ('bundle_sha256', 'snapshot_id', 'embedding_sha256', 'k'):
        if getattr(candidate, field) != getattr(previous, field):
            failures.append(f'baseline {field} mismatch')
    if candidate.k != policy.k:
        failures.append('policy k mismatch')
    if candidate.artifact.feature_schema != previous.artifact.feature_schema:
        failures.append('feature schema incompatible with baseline')
    for stage in STAGES:
        counts = candidate.stages[stage].counts
        if counts['profiles'] < policy.min_profiles or counts['quality_profiles'] != counts['profiles']:
            failures.append(f'{stage}: insufficient judged profile samples')
    checks = []
    for rule in policy.rules:
        value = candidate.stages[rule.stage].metrics[rule.metric]
        old = previous.stages[rule.stage].metrics[rule.metric]
        reasons = []
        if value is None or old is None:
            reasons.append('missing metric')
        else:
            if rule.minimum is not None and value < rule.minimum:
                reasons.append('below minimum')
            if rule.maximum is not None and value > rule.maximum:
                reasons.append('above maximum')
            regression = old - value if DIRECTIONS[rule.metric] == 'higher' else value - old
            if regression > rule.max_regression + 1e-12:
                reasons.append('baseline regression')
        name = f'{rule.stage}.{rule.metric}'
        failures.extend(f'{name}: {reason}' for reason in reasons)
        checks.append({'gate': name, 'value': value, 'baseline': old, 'passed': not reasons})
    if qualify:
        if candidate.provenance.kind == 'synthetic' or previous.provenance.kind == 'synthetic':
            failures.append('synthetic evidence cannot qualify a production model')
        if baseline.status != 'promoted':
            failures.append('qualification requires the last promoted baseline')
        if candidate.artifact.timing_kind != 'measured' or previous.artifact.timing_kind != 'measured':
            failures.append('qualification requires measured inference timings')
        required = {(stage, metric) for stage in STAGES for metric in
                    ('judged_recall_at_k', 'judged_ndcg_at_k', 'judged_mrr_at_k', 'judgment_coverage', 'latency_p95_ms')}
        required.update(('assembly', metric) for metric in DIRECTIONS if metric not in {'latency_p50_ms'})
        indexed = {(r.stage, r.metric): r for r in policy.rules}
        if not required <= set(indexed):
            failures.append('qualification policy omits required quality/diversity/latency/compatibility gates')
        for stage, metric in sorted(required & set(indexed)):
            rule = indexed[(stage, metric)]
            required_limit = 'minimum' if DIRECTIONS[metric] == 'higher' else 'maximum'
            if getattr(rule, required_limit) is None:
                failures.append(f'{stage}.{metric}: qualification requires {required_limit}')
        for stage in STAGES:
            metrics = candidate.stages[stage].metrics
            for name in sorted(SAFETY):
                if metrics[name] != 0:
                    failures.append(f'{stage}: {name} must be zero')
            if metrics['judgment_coverage'] != 1:
                failures.append(f'{stage}: qualification requires complete top-k judgments')
        for name in ('owner_metadata_coverage', 'topic_metadata_coverage', 'fork_metadata_coverage'):
            if candidate.stages['assembly'].metrics[name] != 1:
                failures.append(f'assembly: qualification requires {name}')
    return {'schema_version': 1, 'purpose': 'qualification_review' if qualify else 'diagnostic',
            'passed': not failures, 'failures': failures, 'checks': checks,
            'report_sha256': digest(candidate), 'baseline_sha256': digest(baseline), 'policy_sha256': digest(policy)}


def qualification_manifest(report: dict, baseline: Baseline, policy: Policy) -> dict:
    """Bind passing evidence for human review, without changing serving artifacts."""
    parsed = Report.model_validate(report)
    assessment = assess(report, baseline, policy, qualify=True)
    if assessment.get('purpose') != 'qualification_review' or assessment.get('passed') is not True:
        raise ValueError('passing qualification assessment required')
    if assessment.get('report_sha256') != digest(parsed):
        raise ValueError('assessment/report digest mismatch')
    return {'schema_version': 1, 'status': 'evaluation_passed_requires_authorized_promotion',
            'report_sha256': digest(parsed), 'assessment_sha256': digest(assessment),
            'data_snapshot_sha256': parsed.bundle_sha256, 'snapshot_id': parsed.snapshot_id,
            'model_artifact': parsed.artifact.model_dump(), 'embedding_sha256': parsed.embedding_sha256}
