"""Explicit offline treatments and declared backend-shaped social evidence.

These declarations do not authenticate source provenance or recheck SQL privacy.
They cannot authorize private activity or production activation.
"""
from __future__ import annotations
from dataclasses import fields
from datetime import datetime
from typing import Annotated, Literal
from pydantic import Field, JsonValue, field_validator, model_validator
from evaluation.recommendation_benchmark import StrictModel, Text, aware_timestamp, canonical_id
from inference.diversity import DiversityPolicy
from retrieval.social_candidates import SocialCandidate, SocialPolicy, validate_social_candidates


class FrozenTreatment(StrictModel):
    name: Literal['baseline','diversity','social','combined']
    diversity: dict[str,JsonValue] | None
    social: dict[str,JsonValue] | None

    @model_validator(mode='after')
    def exact_explicit_policy(self):
        expected={'baseline':(False,False),'diversity':(True,False),
                  'social':(False,True),'combined':(True,True)}[self.name]
        if (self.diversity is not None,self.social is not None)!=expected:
            raise ValueError('treatment name must match enabled policies')
        if self.diversity is not None:
            if set(self.diversity)!={f.name for f in fields(DiversityPolicy)}:
                raise ValueError('offline diversity requires every explicit existing policy field')
            policy=DiversityPolicy(**self.diversity)
            if policy.dimensions!=384 or policy.max_pool!=150 or policy.quota_search_limit>5000:
                raise ValueError('offline diversity must retain current dimensions/pool/work bounds')
        if self.social is not None:
            if set(self.social)!={f.name for f in fields(SocialPolicy)}:
                raise ValueError('offline social requires every explicit existing policy field')
            policy=SocialPolicy(**self.social)
            if policy.mode!='enabled' or policy.traffic_percent!=100 or policy.max_candidates>20:
                raise ValueError('offline enabled cohort must retain bounded social allocation')
        return self


class FrozenSocialEvidence(StrictModel):
    viewer_id: str
    observed_at: str
    declared_origin: Literal['synthetic_fixture','backend_export']
    declared_policy: Literal['public-signals-own-seeds-v1']
    reference: Text
    candidates: Annotated[list[SocialCandidate],Field(max_length=100)]
    _id=field_validator('viewer_id')(canonical_id)
    _time=field_validator('observed_at')(aware_timestamp)

    @model_validator(mode='after')
    def safe_projection_shape(self):
        validate_social_candidates(self.candidates)
        allowed={'followed_owner','followed_like','co_like','board_topic','adjacent_interest'}
        if any(row.source not in allowed for row in self.candidates):
            raise ValueError('private cross-user save channels are outside this replay policy')
        return self


class FrozenVariantContext(StrictModel):
    treatment: FrozenTreatment
    social_evidence: Annotated[list[FrozenSocialEvidence],Field(max_length=32)]

    def validate_for(self,snapshot):
        profiles={p.profile_id for p in snapshot.bundle.profiles}
        repos={r.repo_id:r for r in snapshot.bundle.repositories}
        viewers=[row.viewer_id for row in self.social_evidence]
        if len(viewers)!=len(set(viewers)) or not set(viewers)<=profiles:
            raise ValueError('social evidence requires unique frozen viewers')
        as_of=datetime.fromisoformat(snapshot.as_of.replace('Z','+00:00'))
        for projection in self.social_evidence:
            if datetime.fromisoformat(projection.observed_at.replace('Z','+00:00'))>as_of:
                raise ValueError('social evidence is later than frozen as_of')
            for row in projection.candidates:
                if row.repo_id not in repos or repos[row.repo_id].owner!=row.owner_github_id:
                    raise ValueError('social evidence owner/repository differs from frozen corpus')
        if self.treatment.diversity is not None and self.treatment.diversity['minimum_exploration']>snapshot.limit:
            raise ValueError('offline exploration request exceeds output size')
        return self
