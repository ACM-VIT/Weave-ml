"""Model-independent frozen evaluation sources; no model vector fields."""
from __future__ import annotations
import hashlib
import json
from typing import Annotated, Literal
import uuid
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)


def canonical_uuid(value):
    if str(uuid.UUID(value))!=value: raise ValueError('source requires canonical UUID')
    return value


class RepositorySource(Strict):
    repo_id:str
    full_name:Annotated[str,Field(min_length=1,max_length=512)]
    description:Annotated[str,Field(max_length=65536)]
    readme:Annotated[str,Field(max_length=1048576)]
    owner:str|None
    topics:Annotated[list[str],Field(max_length=100)]
    fork_family:str|None
    popularity:Annotated[int,Field(ge=0,le=10**12)]
    archived:bool
    # Complete non-model raw/derived source used for ranking features. Model
    # identifiers/vectors belong to each run, never the common dataset identity.
    ranking_material:dict[str,JsonValue]
    _id=field_validator('repo_id')(canonical_uuid)
    @field_validator('ranking_material')
    @classmethod
    def no_model_material(cls,value):
        forbidden={'vector','vectors','embedding','embedding_model','embedding_dim','embedding_version',
                   'embedding_model_revision','model_version','repo_embedding','user_vector'}
        def visit(node,depth=0):
            if depth>16:raise ValueError('source nesting budget')
            if isinstance(node,dict):
                if forbidden.intersection(node):raise ValueError('model material cannot identify raw dataset')
                for child in node.values():visit(child,depth+1)
            elif isinstance(node,list):
                for child in node:visit(child,depth+1)
        visit(value)
        return value


class ProfileSource(Strict):
    profile_id:str
    intent:Annotated[str,Field(min_length=1,max_length=1024)]
    # Exact original context, not the vector encoded from it. Empty context is
    # explicit for synthetic vector-only fixtures, never invented live history.
    context_text:Annotated[str,Field(max_length=1048576)]
    skills:Annotated[list[str],Field(max_length=100)]
    tech_stack:Annotated[list[str],Field(max_length=100)]
    interests:Annotated[list[str],Field(max_length=100)]
    topics:Annotated[list[str],Field(max_length=100)]
    seen:Annotated[list[str],Field(max_length=10000)]
    disliked:Annotated[list[str],Field(max_length=10000)]
    _id=field_validator('profile_id')(canonical_uuid)


class Label(Strict):
    profile_id:str
    repo_id:str
    grade:Annotated[int,Field(ge=0,le=3)]
    _ids=field_validator('profile_id','repo_id')(canonical_uuid)


class SourceDataset(Strict):
    schema_version:Literal[1]
    snapshot_id:Annotated[str,Field(min_length=1,max_length=1024)]
    reference:Annotated[str,Field(min_length=1,max_length=1024)]
    # Bind provenance/censoring declarations and temporal split input alongside
    # raw sources. This content identity does not authenticate those declarations.
    provenance:dict[str,JsonValue]
    split:dict[str,str]
    repositories:Annotated[list[RepositorySource],Field(min_length=1,max_length=10000)]
    profiles:Annotated[list[ProfileSource],Field(min_length=1,max_length=1000)]
    judgments:Annotated[list[Label],Field(min_length=1,max_length=1000000)]

    @model_validator(mode='after')
    def coherent(self):
        repos=[x.repo_id for x in self.repositories];profiles=[x.profile_id for x in self.profiles]
        if len(repos)!=len(set(repos)) or len(profiles)!=len(set(profiles)) or set(self.split)!=set(profiles):
            raise ValueError('source identities/split must cover every profile once')
        if any(not value or len(value)>128 for value in self.split.values()):raise ValueError('explicit bounded split required')
        pairs=set()
        for row in self.judgments:
            pair=(row.profile_id,row.repo_id)
            if pair in pairs or pair[0] not in profiles or pair[1] not in repos:raise ValueError('invalid source judgment identity')
            pairs.add(pair)
        raw=json.dumps(self.model_dump(mode='json'),allow_nan=False,sort_keys=True,separators=(',',':')).encode()
        if len(raw)>32*1024*1024:raise ValueError('raw dataset exceeds32MiB')
        return self

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.model_dump(mode='json'),allow_nan=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def validate_bundle(self,bundle):
        if bundle.snapshot_id!=self.snapshot_id or bundle.provenance.model_dump(mode='json',exclude_none=True)!={k:v for k,v in self.provenance.items() if v is not None}:
            raise ValueError('source snapshot/provenance differs from bundle')
        sources={x.repo_id:x for x in self.repositories}
        if set(sources)!={x.repo_id for x in bundle.repositories}:raise ValueError('source corpus differs')
        for row in bundle.repositories:
            source=sources[row.repo_id]
            for name in ('owner','topics','fork_family','popularity','archived'):
                if getattr(source,name)!=getattr(row,name):raise ValueError('source repository metadata differs')
        profiles={x.profile_id:x for x in self.profiles}
        if set(profiles)!={x.profile_id for x in bundle.profiles}:raise ValueError('source profiles differ')
        for row in bundle.profiles:
            source=profiles[row.profile_id]
            for name in ('intent','seen','disliked'):
                if getattr(source,name)!=getattr(row,name):raise ValueError('source profile/exclusions differ')
        if [x.model_dump() for x in self.judgments]!=[x.model_dump() for x in bundle.judgments]:
            raise ValueError('source judgments differ')
