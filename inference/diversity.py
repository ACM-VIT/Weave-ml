"""Bounded deterministic embedding MMR with hard identity/topic/fork caps.

The caller owns canonical eligibility, fresh vector/payload pairing, source
buckets and recent-window exclusions. This module never hydrates or embeds.
Sparse pools return fewer rows rather than silently violating hard controls.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import uuid
from typing import Callable, Literal, Sequence

import numpy as np

Bucket = Literal['personalized', 'adjacent', 'trending', 'exploration']


@dataclass(frozen=True)
class DiversityCandidate:
    repo_id: str
    score: float
    vector: tuple[float, ...]
    owner_github_id: str | None
    fork_family_github_id: str | None
    topics: tuple[str, ...] = ()
    category: str | None = None
    bucket: Bucket = 'personalized'


@dataclass(frozen=True)
class DiversityPolicy:
    dimensions: int = 384
    relevance_weight: float = 0.7
    max_per_owner: int = 2
    max_per_topic: int = 5
    max_per_category: int = 8
    max_per_fork_family: int = 1
    duplicate_cosine: float = 0.98
    minimum_exploration: int = 2
    max_pool: int = 150
    quota_search_limit: int = 5000

    def __post_init__(self):
        for field in ('dimensions', 'max_per_owner', 'max_per_topic', 'max_per_category',
                      'max_per_fork_family', 'max_pool'):
            value = getattr(self, field)
            if type(value) is not int or not 1 <= value <= (4096 if field == 'dimensions' else 150):
                raise ValueError(f'Invalid {field}')
        if type(self.quota_search_limit) is not int or not 1 <= self.quota_search_limit <= 50000:
            raise ValueError('Invalid quota_search_limit')
        if type(self.minimum_exploration) is not int or not 0 <= self.minimum_exploration <= 150:
            raise ValueError('Invalid minimum_exploration')
        if type(self.relevance_weight) not in {int, float} or not math.isfinite(self.relevance_weight) or not 0 <= self.relevance_weight <= 1:
            raise ValueError('Invalid relevance_weight')
        if type(self.duplicate_cosine) not in {int, float} or not math.isfinite(self.duplicate_cosine) or not 0 < self.duplicate_cosine <= 1:
            raise ValueError('Invalid duplicate_cosine')


@dataclass(frozen=True)
class DiversityResult:
    candidates: tuple[DiversityCandidate, ...]
    exploration_count: int
    requested_exploration: int
    shortfall: int
    exploration_shortfall: int
    excluded_recent: int
    unknown_identity: int
    intra_list_similarity: float | None
    quota_search_exhausted: bool


def _decimal(value: str | None) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 19 and value.isascii() and value.isdecimal() and value[0] != '0' and int(value) <= 2**63 - 1


def rerank_diverse(
    candidates: Sequence[DiversityCandidate], *, target_size: int, seed: str,
    policy: DiversityPolicy = DiversityPolicy(), recent_repo_ids: frozenset[str] = frozenset(),
    personalized_top_ids: frozenset[str] = frozenset(),
    check_work: Callable[[], None] = lambda: None,
) -> DiversityResult:
    """Reserve feasible exploration first, then maximize relevance/diversity.

    Relevance is min/max normalized over the eligible request pool. Similarity
    uses normalized repository embeddings. Seeded ties use SHA256, independent
    of input order or Python hash randomization. Unknown immutable identities
    are omitted and counted, never inferred from names or zero-filled.
    """
    if type(target_size) is not int or not 0 <= target_size <= 150:
        raise ValueError('target_size must be between 0 and 150')
    if not isinstance(seed, str) or not 1 <= len(seed) <= 256:
        raise ValueError('seed must be bounded nonempty text')
    if len(candidates) > policy.max_pool:
        raise ValueError('Candidate pool exceeds configured bound')
    recent = {str(uuid.UUID(value)) for value in recent_repo_ids}
    personalized = {str(uuid.UUID(value)) for value in personalized_top_ids}
    identities: set[str] = set()
    pool: list[tuple[DiversityCandidate, tuple[float, ...]]] = []
    excluded = unknown = 0
    for candidate in candidates:
        check_work()
        repo_id = str(uuid.UUID(candidate.repo_id))
        if candidate.repo_id != repo_id or repo_id in identities:
            raise ValueError('Candidates require unique canonical repository UUIDs')
        identities.add(repo_id)
        if candidate.bucket not in {'personalized', 'adjacent', 'trending', 'exploration'}:
            raise ValueError('Invalid candidate bucket')
        if type(candidate.score) not in {int, float} or not math.isfinite(candidate.score):
            raise ValueError('Candidate score must be finite')
        if type(candidate.vector) is not tuple or len(candidate.vector) != policy.dimensions or any(
            type(value) not in {int, float} or not math.isfinite(value) for value in candidate.vector
        ):
            raise ValueError('Candidate requires a finite compatible repository vector')
        norm = math.sqrt(math.fsum(value * value for value in candidate.vector))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError('Candidate repository vector must have finite nonzero norm')
        if (type(candidate.topics) is not tuple or len(candidate.topics) > 200
                or any(not isinstance(topic, str) or not 1 <= len(topic) <= 128
                       or topic != topic.strip().casefold() for topic in candidate.topics)
                or len(set(candidate.topics)) != len(candidate.topics)
                or (candidate.category is not None and (not isinstance(candidate.category, str)
                    or not 1 <= len(candidate.category) <= 128
                    or candidate.category != candidate.category.strip().casefold()))):
            raise ValueError('Candidate topics/category must be bounded canonical labels')
        if repo_id in recent:
            excluded += 1
            continue
        if not _decimal(candidate.owner_github_id) or not _decimal(candidate.fork_family_github_id):
            unknown += 1
            continue
        pool.append((candidate, tuple(value / norm for value in candidate.vector)))
    scores = [candidate.score for candidate, _ in pool]
    low = min(scores, default=0)
    spread = max(scores, default=0) - low
    relevance = {candidate.repo_id: (candidate.score - low) / spread if spread else 1.0 for candidate, _ in pool}
    indices = {candidate.repo_id: index for index, (candidate, _) in enumerate(pool)}
    normalized = np.asarray([vector for _, vector in pool], dtype=np.float64).reshape((-1, policy.dimensions))
    # At most150x150 scalars: compute each dot product once, not once per
    # candidate per prior selection per greedy iteration.
    check_work()
    similarities = np.clip(normalized @ normalized.T, -1.0, 1.0)
    check_work()
    selected: list[tuple[DiversityCandidate, tuple[float, ...]]] = []
    owners: dict[str, int] = {}
    families: dict[str, int] = {}
    topics: dict[str, int] = {}
    categories: dict[str, int] = {}

    def cosine(a: str, b: str) -> float:
        return float(similarities[indices[a], indices[b]])

    def exploration(candidate: DiversityCandidate) -> bool:
        return candidate.bucket != 'personalized' and candidate.repo_id not in personalized

    def feasible(row: tuple[DiversityCandidate, tuple[float, ...]]) -> bool:
        candidate, vector = row
        return (owners.get(candidate.owner_github_id, 0) < policy.max_per_owner
                and families.get(candidate.fork_family_github_id, 0) < policy.max_per_fork_family
                and all(topics.get(topic, 0) < policy.max_per_topic for topic in candidate.topics)
                and (candidate.category is None or categories.get(candidate.category, 0) < policy.max_per_category)
                and all(cosine(candidate.repo_id, existing.repo_id) < policy.duplicate_cosine for existing, _ in selected))

    def key(row: tuple[DiversityCandidate, tuple[float, ...]]):
        candidate, vector = row
        similarity = max((max(0.0, cosine(candidate.repo_id, existing.repo_id)) for existing, _ in selected), default=0.0)
        mmr = policy.relevance_weight * relevance[candidate.repo_id] - (1 - policy.relevance_weight) * similarity
        tie = hashlib.sha256((seed + '\0' + candidate.repo_id).encode()).digest()
        return (-mmr, tie, candidate.repo_id)

    def record(row, delta):
        candidate = row[0]
        for mapping, value in ((owners, candidate.owner_github_id), (families, candidate.fork_family_github_id)):
            mapping[value] = mapping.get(value, 0) + delta
        for topic in candidate.topics:
            topics[topic] = topics.get(topic, 0) + delta
        if candidate.category is not None:
            categories[candidate.category] = categories.get(candidate.category, 0) + delta

    required = min(target_size, policy.minimum_exploration)
    explore_pool = sorted((row for row in pool if exploration(row[0])), key=key)
    attempts = 0
    exhausted = False

    def reserve(start):
        nonlocal attempts, exhausted
        if len(selected) >= required:
            return True
        if len(explore_pool) - start < required - len(selected):
            return False
        for index in range(start, len(explore_pool)):
            check_work()
            if attempts >= policy.quota_search_limit:
                exhausted = True
                return False
            attempts += 1
            row = explore_pool[index]
            if not feasible(row):
                continue
            selected.append(row)
            record(row, 1)
            if reserve(index + 1):
                return True
            record(row, -1)
            selected.pop()
            if exhausted:
                return False
        return False

    # Bounded backtracking avoids a high-scoring exploratory row consuming all
    # topic/family capacity when a compatible quota exists among alternatives.
    # If the budget cannot establish the quota, report it explicitly; serving
    # callers requiring a guarantee must reject an exploration shortfall.
    reserve(0)
    for chosen in selected:
        pool.remove(chosen)
    while len(selected) < target_size:
        check_work()
        eligible = [row for row in pool if feasible(row)]
        if not eligible:
            break
        chosen = min(eligible, key=key)
        pool.remove(chosen)
        selected.append(chosen)
        record(chosen, 1)
    # Selection reserves quota first; presentation retains MMR selection order.
    # All returned source buckets remain explicit rather than being relabeled.
    count = sum(exploration(candidate) for candidate, _ in selected)
    pair_scores = [cosine(a[0].repo_id, b[0].repo_id) for i, a in enumerate(selected) for b in selected[i + 1:]]
    return DiversityResult(tuple(candidate for candidate, _ in selected), count, required,
                           target_size - len(selected), max(0, required - count), excluded, unknown,
                           math.fsum(pair_scores) / len(pair_scores) if pair_scores else None, exhausted)
