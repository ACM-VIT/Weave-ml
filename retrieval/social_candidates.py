"""Bounded social candidates supplied by the backend's visibility projection.

This module never queries product SQL or infers immutable owners from handles.
Candidate scores/support are derived upstream; no social records are invented.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Annotated, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

SocialSource = Literal['followed_owner', 'followed_like', 'followed_save', 'co_like', 'co_save', 'board_topic', 'adjacent_interest']
COLLABORATIVE_SOURCES = {'co_like', 'co_save', 'followed_like', 'followed_save', 'board_topic'}


class SocialCandidate(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    repo_id: str
    owner_github_id: Annotated[str, StringConstraints(pattern=r'^[1-9][0-9]{0,18}$')]
    source: SocialSource
    support: Annotated[int, Field(ge=1, le=100000)]
    source_score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]

    @field_validator('repo_id')
    @classmethod
    def canonical_repo(cls, value):
        return str(uuid.UUID(value))

    @field_validator('owner_github_id')
    @classmethod
    def canonical_owner(cls, value):
        if int(value) > 2**63 - 1:
            raise ValueError('owner GitHub ID exceeds canonical backend bigint range')
        return value


def validate_social_candidates(values: list[SocialCandidate]) -> list[SocialCandidate]:
    if len(values) > 100:
        raise ValueError('social candidate projection exceeds 100 records')
    pairs = set()
    owners = {}
    for value in values:
        pair = (value.repo_id, value.source)
        if pair in pairs:
            raise ValueError('duplicate repository/source in social projection')
        pairs.add(pair)
        if value.repo_id in owners and owners[value.repo_id] != value.owner_github_id:
            raise ValueError('conflicting immutable repository owner attribution')
        owners[value.repo_id] = value.owner_github_id
    return values


@dataclass(frozen=True)
class SocialPolicy:
    mode: Literal['disabled', 'shadow', 'enabled'] = 'disabled'
    max_lookup: int = 100
    max_candidates: int = 20
    max_per_owner: int = 2
    max_per_source: int = 8
    minimum_collaborative_support: int = 2

    def __post_init__(self):
        if self.mode not in {'disabled', 'shadow', 'enabled'}:
            raise ValueError('invalid social candidate rollout mode')
        for name in ('max_lookup', 'max_candidates', 'max_per_owner', 'max_per_source', 'minimum_collaborative_support'):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 100:
                raise ValueError(f'{name} must be an integer from 1 to 100')
        if self.max_candidates > self.max_lookup:
            raise ValueError('selection quota cannot exceed lookup budget')
        if self.minimum_collaborative_support < 2:
            raise ValueError('collaborative candidates need at least two supporters')

    @property
    def injects(self):
        return self.mode == 'enabled'


@dataclass(frozen=True)
class FusedSocialCandidate:
    repo_id: str
    owner_github_id: str
    fusion_score: float
    # Sorted immutable tuples retain raw source-specific score and support.
    evidence: tuple[tuple[str, float, int], ...]

    @property
    def primary_source(self):
        return min(self.evidence, key=lambda row: (-row[1], row[0]))[0]

    @property
    def source_scores(self):
        return {source: score for source, score, _ in self.evidence}


def _supported(candidate: SocialCandidate, policy: SocialPolicy):
    minimum = policy.minimum_collaborative_support if candidate.source in COLLABORATIVE_SOURCES else 1
    return candidate.support >= minimum and candidate.source_score > 0


def lookup_social_ids(candidates: list[SocialCandidate], policy: SocialPolicy, excluded: set[str]) -> list[str]:
    validate_social_candidates(candidates)
    if policy.mode == 'disabled':
        return []
    ordered = sorted((row for row in candidates if row.repo_id not in excluded and _supported(row, policy)),
                     key=lambda row: (-row.source_score, -row.support, row.repo_id, row.source))
    return list(dict.fromkeys(row.repo_id for row in ordered))[:policy.max_lookup]


def select_social_candidates(candidates: list[SocialCandidate], policy: SocialPolicy, *, eligible_repo_ids: set[str], excluded: set[str]) -> list[FusedSocialCandidate]:
    """Fuse only explicitly requested, compatible eligible IDs with fixed RRF k=60."""
    lookup = set(lookup_social_ids(candidates, policy, excluded))
    if not lookup:
        return []
    channels = defaultdict(list)
    for row in candidates:
        if row.repo_id in lookup and row.repo_id in eligible_repo_ids and row.repo_id not in excluded and _supported(row, policy):
            channels[row.source].append(row)
    contributions = defaultdict(list)
    owners = {}
    for source in sorted(channels):
        ordered = sorted(channels[source], key=lambda row: (-row.source_score, -row.support, row.repo_id))
        for rank, row in enumerate(ordered, 1):
            contributions[row.repo_id].append((source, row.source_score, row.support, 1 / (60 + rank)))
            owners[row.repo_id] = row.owner_github_id
    fused = [FusedSocialCandidate(repo_id, owners[repo_id], math.fsum(row[3] for row in evidence),
                                  tuple((source, score, support) for source, score, support, _ in evidence))
             for repo_id, evidence in contributions.items()]
    fused.sort(key=lambda row: (-row.fusion_score, row.repo_id))
    selected = []
    owner_counts, source_counts = Counter(), Counter()
    for row in fused:
        if owner_counts[row.owner_github_id] >= policy.max_per_owner:
            continue
        if any(source_counts[source] >= policy.max_per_source for source, _, _ in row.evidence):
            continue
        selected.append(row)
        owner_counts[row.owner_github_id] += 1
        source_counts.update(source for source, _, _ in row.evidence)
        if len(selected) >= policy.max_candidates:
            break
    return selected
