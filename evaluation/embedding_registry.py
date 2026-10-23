"""Offline model identities and vector compatibility, separate from serving config."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(min_length=1, max_length=1024)]


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)
    name: Text
    model_id: Text
    revision: Annotated[str, StringConstraints(pattern=r'^[0-9a-f]{40}$')]
    native_dimension: Annotated[int, Field(ge=1, le=8192)]
    output_dimension: Annotated[int, Field(ge=1, le=8192)]
    max_tokens: Annotated[int, Field(ge=8, le=32768)]
    pooling: Literal['mean', 'last_token', 'cls', 'max']
    normalize: Literal[True]
    instruction_aware: bool
    query_prompt: Annotated[str, StringConstraints(max_length=4096)]
    document_prompt: Annotated[str, StringConstraints(max_length=4096)]
    dtype: Literal['float32'] = 'float32'
    input_recipe: Literal['metadata-readme-v1'] = 'metadata-readme-v1'
    document_char_limit: Annotated[int, Field(ge=256, le=65536)] = 64000

    @model_validator(mode='after')
    def supported(self):
        if self.output_dimension > self.native_dimension:
            raise ValueError('output dimension exceeds native model dimension')
        if not self.instruction_aware and (self.query_prompt or self.document_prompt):
            raise ValueError('non-instruction model must not receive role prompts')
        if self.instruction_aware and not self.query_prompt.strip():
            raise ValueError('instruction-aware evaluation requires an explicit query prompt')
        return self

    @property
    def version(self) -> str:
        payload = self.model_dump(exclude={'name'})
        checksum = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return f'offline-embedding-v1:{checksum}'


class Registry(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    schema_version: Literal[1]
    models: Annotated[list[EmbeddingConfig], Field(min_length=1, max_length=20)]

    @model_validator(mode='after')
    def unique(self):
        if len({model.name for model in self.models}) != len(self.models):
            raise ValueError('registry names must be unique')
        return self

    def get(self, name: str) -> EmbeddingConfig:
        for model in self.models:
            if model.name == name:
                return model
        raise ValueError(f'unknown offline model configuration: {name}')


@dataclass(frozen=True)
class VectorBatch:
    version: str
    role: Literal['query', 'document']
    values: np.ndarray

    @classmethod
    def create(cls, values, config: EmbeddingConfig, *, role: Literal['query', 'document']):
        if role not in {'query', 'document'}:
            raise ValueError('unknown embedding role')
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != config.output_dimension:
            raise ValueError('model returned incompatible embedding dimensions')
        if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
            raise ValueError('model returned nonfinite or non-floating vectors')
        norms = np.linalg.norm(values, axis=1)
        if not np.allclose(norms, 1., atol=1e-5, rtol=0):
            raise ValueError('model output must be normalized after dimensional truncation')
        stored = values.astype(np.float32, copy=True)
        stored.flags.writeable = False
        return cls(config.version, role, stored)


def encode_role(model, texts: list[str], config: EmbeddingConfig, *, role: Literal['query', 'document'], batch_size: int) -> VectorBatch:
    if role not in {'query', 'document'}:
        raise ValueError('unknown embedding role')
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError('encoding requires nonempty text strings')
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError('batch size must be from 1 to 128')
    prompt = config.query_prompt if role == 'query' else config.document_prompt
    vectors = model.encode(texts, prompt=prompt, truncate_dim=config.output_dimension,
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False, batch_size=batch_size)
    if len(vectors) != len(texts):
        raise ValueError('encoder returned an incomplete batch')
    return VectorBatch.create(vectors, config, role=role)


def compatible_scores(queries: VectorBatch, documents: VectorBatch) -> np.ndarray:
    if queries.version != documents.version:
        raise ValueError('incompatible model/revision/dimension/prompt/input identity')
    if queries.role != 'query' or documents.role != 'document':
        raise ValueError('cosine requires query and document roles')
    return queries.values @ documents.values.T
