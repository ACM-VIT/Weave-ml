"""Versioned, service-free recommendation evidence and metrics.

Inputs and timings are supplied evidence, not observations made by this module.
Unknown relevance is reported explicitly; no fixture qualifies a serving model.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
from typing import Annotated, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

Text = Annotated[str, StringConstraints(min_length=1, max_length=1024)]
Digest = Annotated[str, StringConstraints(pattern=r'^[a-f0-9]{64}$')]
Commit = Annotated[str, StringConstraints(pattern=r'^[a-f0-9]{40}$')]
Number = Annotated[float, Field(allow_inf_nan=False)]
STAGES = ('retrieval', 'ranking', 'assembly')
# Bound additional memory independently of corpus size/profile count. Each run
# owns its cache, so identical IDs in another snapshot cannot reuse old vectors.
_SIMILARITY_CACHE_PAIRS = 65536


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


def canonical_id(value: str) -> str:
    if str(uuid.UUID(value)) != value:
        raise ValueError('identities must be canonical UUID strings')
    return value


def aware_timestamp(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('timestamp must contain a timezone')
    return value


class EmbeddingIdentity(StrictModel):
    model: Text
    revision: Commit
    dimension: Annotated[int, Field(ge=1, le=4096)]
    pooling: Text
    normalized: Literal[True]
    query_prompt: Annotated[str, StringConstraints(max_length=4096)]
    document_prompt: Annotated[str, StringConstraints(max_length=4096)]
    recipe: Text


class Provenance(StrictModel):
    kind: Literal['synthetic', 'human_judgment', 'event_holdout']
    reference: Text
    # For real evidence this points to an externally audited collection/rubric.
    # A string declaration alone does not authenticate human or event provenance.
    observation_cutoff: str | None = None
    label_horizon_end: str | None = None
    ingestion_watermark: str | None = None

    @model_validator(mode='after')
    def temporal_order(self):
        values = [self.observation_cutoff, self.label_horizon_end, self.ingestion_watermark]
        if self.kind == 'event_holdout':
            if any(value is None for value in values):
                raise ValueError('event holdout requires cutoff, horizon and watermark')
            for value in values:
                aware_timestamp(value)
            dates = [datetime.fromisoformat(value.replace('Z', '+00:00')) for value in values]
            if not dates[0] < dates[1] <= dates[2]:
                raise ValueError('holdout must satisfy cutoff < horizon <= watermark')
        return self


class Repository(StrictModel):
    repo_id: str
    owner: Text | None
    topics: Annotated[list[Text], Field(max_length=100)]
    fork_family: Text | None
    vector: Annotated[list[Number], Field(min_length=1, max_length=4096)]
    popularity: Annotated[int, Field(ge=0, le=10**12)]
    archived: bool
    _id = field_validator('repo_id')(canonical_id)


class Profile(StrictModel):
    profile_id: str
    intent: Text
    seen: Annotated[list[str], Field(max_length=10000)]
    disliked: Annotated[list[str], Field(max_length=10000)]
    _id = field_validator('profile_id')(canonical_id)

    @field_validator('seen', 'disliked')
    @classmethod
    def canonical_exclusions(cls, value):
        for identifier in value:
            canonical_id(identifier)
        if len(set(value)) != len(value):
            raise ValueError('duplicate exclusion identity')
        return value


class Judgment(StrictModel):
    profile_id: str
    repo_id: str
    grade: Annotated[int, Field(ge=0, le=3)]
    _ids = field_validator('profile_id', 'repo_id')(canonical_id)


class Bundle(StrictModel):
    schema_version: Literal[1]
    snapshot_id: Text
    created_at: str
    provenance: Provenance
    embedding: EmbeddingIdentity
    repositories: Annotated[list[Repository], Field(min_length=1, max_length=10000)]
    profiles: Annotated[list[Profile], Field(min_length=1, max_length=1000)]
    judgments: Annotated[list[Judgment], Field(min_length=1, max_length=1000000)]
    _timestamp = field_validator('created_at')(aware_timestamp)

    @model_validator(mode='after')
    def references(self):
        repos = {r.repo_id for r in self.repositories}
        profiles = {p.profile_id for p in self.profiles}
        if len(repos) != len(self.repositories) or len(profiles) != len(self.profiles):
            raise ValueError('duplicate corpus/profile identity')
        pairs = set()
        for row in self.judgments:
            if row.repo_id not in repos or row.profile_id not in profiles:
                raise ValueError('judgment references unknown corpus/profile identity')
            pair = (row.profile_id, row.repo_id)
            if pair in pairs:
                raise ValueError('duplicate or conflicting judgment')
            pairs.add(pair)
        for repo in self.repositories:
            if len(repo.vector) != self.embedding.dimension:
                raise ValueError('repository vector dimension does not match embedding identity')
            norm = math.sqrt(sum(x * x for x in repo.vector))
            if not math.isfinite(norm) or abs(norm - 1) > 1e-5:
                raise ValueError('repository vectors must be finite normalized nonzero vectors')
            if len(set(repo.topics)) != len(repo.topics):
                raise ValueError('duplicate repository topic')
        if not any(not repo.archived for repo in self.repositories):
            raise ValueError('corpus has no eligible repositories')
        return self


class Candidate(StrictModel):
    repo_id: str
    score: Number
    source: Text
    source_score: Number
    _id = field_validator('repo_id')(canonical_id)


class Output(StrictModel):
    profile_id: str
    stages: dict[Literal['retrieval', 'ranking', 'assembly'], Annotated[list[Candidate], Field(max_length=1000)]]
    latency_ms: dict[Literal['retrieval', 'ranking', 'assembly'], Annotated[float, Field(ge=0, allow_inf_nan=False)]]
    _id = field_validator('profile_id')(canonical_id)

    @model_validator(mode='after')
    def all_stages(self):
        if set(self.stages) != set(STAGES) or set(self.latency_ms) != set(STAGES):
            raise ValueError('all three stages and latency samples are required')
        return self


class Run(StrictModel):
    schema_version: Literal[1]
    embedding: EmbeddingIdentity
    model_id: Text
    model_sha256: Digest
    scaler_sha256: Digest
    feature_schema: Text
    code_commit: Commit
    created_at: str
    timing_kind: Literal['synthetic', 'measured']
    outputs: Annotated[list[Output], Field(min_length=1, max_length=1000)]
    _timestamp = field_validator('created_at')(aware_timestamp)


def digest(value) -> str:
    """Canonical content identity, independent of JSON indentation/key order."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode='json')
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _mean(values):
    return sum(values) / len(values) if values else None


def _quantile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def _clamped_cosine(left: list[float], right: list[float]) -> float:
    # Keep the original product/reduction order and per-pair clamp. Normalized
    # inputs permit a small tolerance, so an averaged-vector shortcut can differ.
    return max(-1., min(1., sum(x * y for x, y in zip(left, right))))


def evaluate(bundle: Bundle, run: Run, *, k: int) -> dict:
    """Measure supplied stage outputs, retaining violations and unknown labels."""
    if type(k) is not int or not 1 <= k <= 1000:
        raise ValueError('k must be an integer from 1 to 1000')
    if bundle.embedding != run.embedding:
        raise ValueError('incompatible embedding identity (model/revision/dimension/prompts/recipe)')
    profiles = {p.profile_id: p for p in bundle.profiles}
    if len(run.outputs) != len(profiles) or {o.profile_id for o in run.outputs} != set(profiles):
        raise ValueError('run profiles must match every frozen profile exactly once')
    repos = {r.repo_id: r for r in bundle.repositories}

    @lru_cache(maxsize=_SIMILARITY_CACHE_PAIRS)
    def pair_similarity(first: str, second: str) -> float:
        return _clamped_cosine(repos[first].vector, repos[second].vector)

    labels_by_profile = defaultdict(dict)
    for row in bundle.judgments:
        labels_by_profile[row.profile_id][row.repo_id] = row.grade
    # Reuse positive identities and ideal DCG across all three stages.
    quality_by_profile = {}
    for profile_id in profiles:
        labels = labels_by_profile[profile_id]
        relevant = {repo_id for repo_id, grade in labels.items() if grade > 0}
        ideal = sorted((2**grade - 1 for grade in labels.values()), reverse=True)[:k]
        quality_by_profile[profile_id] = (
            relevant, sum(g / math.log2(i + 2) for i, g in enumerate(ideal)),
        )
    eligible_count = sum(not r.archived for r in bundle.repositories)
    popularity_total = sum(r.popularity + 1 for r in bundle.repositories)
    stages = {}
    for stage in STAGES:
        recalls, ndcgs, mrrs, novelties, owners, topics = [], [], [], [], [], []
        similarity_sum = 0.
        similarity_pairs = 0
        returned = unjudged = repeats = excluded = archived = forks = owner_known = topic_known = family_known = 0
        sources = Counter()
        catalogue = set()
        latencies = []
        for output in run.outputs:
            profile = profiles[output.profile_id]
            labels = labels_by_profile[profile.profile_id]
            rows = output.stages[stage][:k]
            if any(row.repo_id not in repos for row in output.stages[stage]):
                raise ValueError('run contains an unknown repository identity')
            ids = [row.repo_id for row in rows]
            unique_ids = set(ids)
            returned += len(ids)
            repeats += len(ids) - len(unique_ids)
            exclusions = set(profile.seen) | set(profile.disliked)
            excluded += sum(repo_id in exclusions for repo_id in ids)
            archived += sum(repos[repo_id].archived for repo_id in ids)
            catalogue.update(repo_id for repo_id in ids if not repos[repo_id].archived)
            unjudged += sum(repo_id not in labels for repo_id in ids)
            # Repeated IDs can never earn gain/recall more than once.
            encountered = set()
            gains = []
            for repo_id in ids:
                grade = labels.get(repo_id, 0)
                gains.append(2**grade - 1 if repo_id not in encountered else 0)
                encountered.add(repo_id)
            relevant, denominator = quality_by_profile[profile.profile_id]
            if relevant:
                recalls.append(len(unique_ids & relevant) / len(relevant))
                ndcgs.append(sum(g / math.log2(i + 2) for i, g in enumerate(gains)) / denominator)
                mrrs.append(next((1 / (i + 1) for i, g in enumerate(gains) if g > 0), 0.))
            selected = [repos[repo_id] for repo_id in ids]
            family_counts = Counter(r.fork_family for r in selected if r.fork_family is not None)
            forks += sum(count - 1 for count in family_counts.values())
            family_known += sum(family_counts.values())
            owner_counts = Counter(r.owner for r in selected if r.owner is not None)
            owner_known += sum(owner_counts.values())
            topic_counts = Counter()
            for repo in selected:
                if repo.topics:
                    topic_known += 1
                    for topic in repo.topics:
                        topic_counts[topic] += 1 / len(repo.topics)
                novelties.append(-math.log2((repo.popularity + 1) / popularity_total))
            if selected:
                owners.append(max(owner_counts.values(), default=0) / len(selected))
                topics.append(max(topic_counts.values(), default=0) / len(selected))
            for i, a in enumerate(selected):
                for b in selected[i + 1:]:
                    # Reuse only the pair score, not the visit: duplicates still
                    # contribute at every original position in every stage.
                    first, second = sorted((a.repo_id, b.repo_id))
                    similarity_sum += pair_similarity(first, second)
                    similarity_pairs += 1
            sources.update(row.source for row in rows)
            latencies.append(output.latency_ms[stage])
        ratio = lambda count: count / returned if returned else 0.
        stages[stage] = {
            'metrics': {
                'judged_recall_at_k': _mean(recalls), 'judged_ndcg_at_k': _mean(ndcgs),
                'judged_mrr_at_k': _mean(mrrs), 'judgment_coverage': 1 - ratio(unjudged) if returned else 0.,
                'catalogue_coverage': len(catalogue) / eligible_count,
                'intra_list_similarity': similarity_sum / similarity_pairs if similarity_pairs else None, 'novelty_bits': _mean(novelties),
                'owner_concentration': _mean(owners), 'topic_concentration': _mean(topics),
                'fork_metadata_coverage': ratio(family_known), 'owner_metadata_coverage': ratio(owner_known), 'topic_metadata_coverage': ratio(topic_known),
                'repeat_rate': ratio(repeats), 'excluded_rate': ratio(excluded),
                'archived_rate': ratio(archived), 'fork_repeat_rate': ratio(forks),
                'latency_p50_ms': _quantile(latencies, .5), 'latency_p95_ms': _quantile(latencies, .95),
            },
            'counts': {'profiles': len(profiles), 'quality_profiles': len(recalls), 'returned': returned,
                       'unjudged_returned': unjudged, 'similarity_pairs': similarity_pairs, 'latency_samples': len(latencies)},
            'source_share': {source: ratio(count) for source, count in sorted(sources.items())},
        }
    return {'schema_version': 1, 'bundle_sha256': digest(bundle), 'run_sha256': digest(run),
            'snapshot_id': bundle.snapshot_id, 'provenance': bundle.provenance.model_dump(),
            'embedding_sha256': digest(bundle.embedding), 'k': k,
            'artifact': run.model_dump(exclude={'outputs', 'embedding'}), 'stages': stages}
