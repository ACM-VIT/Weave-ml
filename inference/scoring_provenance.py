"""Lossless, owner-bound scored-input packets; no capture/storage is activated.

Hashes provide integrity/binding, not authentication or export authority. Callers
must capture the actual float32 inference row inside its request-owned lifetime,
not reconstruct it later from current vectors. Persistence/retention/access and
served-exposure linkage belong to the backend and remain separate prerequisites.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
import math
import struct
from threading import current_thread
from typing import Annotated, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator
from inference.feature_spec import FEATURE_ORDER

DIMENSIONS = 778
USER_BYTES = 384 * 4
INPUT_BYTES = DIMENSIONS * 4
MAX_ITEMS = 150
DEFAULT_WIRE_LIMIT = 1_000_000
Digest = Annotated[str, StringConstraints(pattern=r'^[a-f0-9]{64}$')]
Revision = Annotated[str, StringConstraints(pattern=r'^[a-f0-9]{40}$')]
Version = Annotated[str, StringConstraints(min_length=1, max_length=128)]
DecimalVersion = Annotated[str, StringConstraints(pattern=r'^(0|[1-9][0-9]{0,18})$')]


def _json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()


def _canonical(value: str) -> str:
    if not isinstance(value, str) or str(uuid.UUID(value)) != value:
        raise ValueError('Provenance identity must be a canonical UUID')
    return value


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True, hide_input_in_errors=True)


class ScoringContext(Strict):
    encoding: Literal['float32-le-v1'] = 'float32-le-v1'
    dimensions: Literal[778] = 778
    user_id: str
    generation_id: str
    scored_at: Annotated[str, StringConstraints(min_length=1, max_length=40)]
    feed_version: DecimalVersion
    profile_version: DecimalVersion
    feedback_version: DecimalVersion
    feedback_state_revision: DecimalVersion
    embedding_model: Version
    embedding_revision: Revision
    # Legacy profiles do not store a recipe identifier; explicit None is honest.
    user_embedding_version: Version | None
    repository_embedding_versions: Annotated[tuple[Version, ...], Field(min_length=1, max_length=8)]
    user_vector_sha256: Digest
    model_version: Annotated[str, StringConstraints(min_length=1, max_length=512)]
    model_sha256: Digest
    scaler_sha256: Digest
    manifest_sha256: Digest
    feature_spec_version: Literal['v2']
    feature_order_sha256: Digest
    code_commit: Revision
    _ids = field_validator('user_id', 'generation_id')(_canonical)

    @field_validator('feed_version', 'profile_version', 'feedback_version', 'feedback_state_revision')
    @classmethod
    def exact_versions(cls, value, info):
        if int(value) > 2**63-1 or (info.field_name in {'feed_version', 'profile_version'} and value == '0'):
            raise ValueError('Invalid exact scored version')
        return value

    @field_validator('scored_at')
    @classmethod
    def explicit_scoring_time(cls, value):
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
            raise ValueError('Scoring time must be explicit UTC')
        return value

    @field_validator('feature_order_sha256')
    @classmethod
    def exact_feature_order(cls, value):
        if value != hashlib.sha256(_json(FEATURE_ORDER)).hexdigest():
            raise ValueError('Scored feature order does not match v2 input specification')
        return value

    @model_validator(mode='after')
    def distinct_embedding_versions(self):
        if len(set(self.repository_embedding_versions)) != len(self.repository_embedding_versions):
            raise ValueError('Duplicate repository embedding version')
        return self


class RepositorySnapshot(Strict):
    repo_id: str
    content_version: DecimalVersion
    feature_version: DecimalVersion
    content_hash: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=128)]
    source_hash: Digest
    embedding_version_index: Annotated[int, Field(ge=0, le=7)]
    _id = field_validator('repo_id')(_canonical)

    @field_validator('content_version', 'feature_version')
    @classmethod
    def exact_versions(cls, value, info):
        if int(value) > 2**63-1 or (info.field_name == 'content_version' and value == '0'):
            raise ValueError('Invalid exact repository version')
        return value


class ScoredPacket(Strict):
    snapshot: RepositorySnapshot
    input_base64: Annotated[str, StringConstraints(min_length=4152, max_length=4152)] = Field(repr=False)
    binding_sha256: Digest


@dataclass(frozen=True)
class ScoredFrame:
    user_id: str
    generation_id: str
    snapshot: RepositorySnapshot
    input_bytes: bytes = field(repr=False)
    context: ScoringContext = field(repr=False)


def _validate_input(raw: bytes, context: ScoringContext):
    if type(raw) is not bytes or len(raw) != INPUT_BYTES:
        raise ValueError('Scored input must be exactly 778 immutable float32 values')
    if any(not math.isfinite(value[0]) for value in struct.iter_unpack('<f', raw)):
        raise ValueError('Scored input must be finite float32')
    if hashlib.sha256(raw[:USER_BYTES]).hexdigest() != context.user_vector_sha256:
        raise ValueError('Scored user vector does not match captured context')


def packet_for_frame(frame: ScoredFrame, context: ScoringContext) -> ScoredPacket:
    if frame.user_id != context.user_id or frame.generation_id != context.generation_id:
        raise ValueError('Scored frame ownership does not match captured context')
    if frame.context != context:
        raise ValueError('Scored frame context does not match capture-time context')
    if frame.snapshot.embedding_version_index >= len(context.repository_embedding_versions):
        raise ValueError('Repository embedding version is not in captured context')
    _validate_input(frame.input_bytes, context)
    binding = _json({'context':context.model_dump(mode='json'), 'snapshot':frame.snapshot.model_dump(mode='json')})
    return ScoredPacket(snapshot=frame.snapshot, input_base64=base64.b64encode(frame.input_bytes).decode('ascii'),
                        binding_sha256=hashlib.sha256(binding + frame.input_bytes).hexdigest())


def decode_owned_packet(packet: ScoredPacket, context: ScoringContext, *,
                        owner_user_id: str, generation_id: str, repo_id: str) -> bytes:
    """Check trusted caller ownership selectors before returning sensitive bytes.

    This is not an authorization service. The caller still needs authenticated
    owner/export permissions; an integrity digest is not a signature.
    """
    if (_canonical(owner_user_id) != context.user_id or _canonical(generation_id) != context.generation_id
            or _canonical(repo_id) != packet.snapshot.repo_id):
        raise ValueError('Scored packet ownership mismatch')
    if context.encoding != 'float32-le-v1' or context.dimensions != DIMENSIONS:
        raise ValueError('Scored packet input specification mismatch')
    try:
        raw = base64.b64decode(packet.input_base64, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid scored packet encoding') from exc
    expected = packet_for_frame(ScoredFrame(context.user_id, context.generation_id, packet.snapshot, raw, context), context)
    if expected.binding_sha256 != packet.binding_sha256:
        raise ValueError('Scored packet integrity mismatch')
    return raw


def seal_scoring_response(response: dict, context: ScoringContext, frames: tuple[ScoredFrame, ...], *,
                          byte_limit: int = DEFAULT_WIRE_LIMIT) -> bytes:
    """Return the exact bounded wire bytes to emit, without mutating base scores.

    Validates selected IDs against actual scored frames, binds one shared context,
    and fails on oversize rather than trimming candidates or dropping provenance.
    Call AFTER normal response validation; emit these exact bytes unchanged.
    """
    if type(byte_limit) is not int or not 1 <= byte_limit <= DEFAULT_WIRE_LIMIT:
        raise ValueError('Invalid scored response byte limit')
    items = response.get('items')
    if not isinstance(items, list) or not 0 <= len(items) <= MAX_ITEMS or len(frames) > MAX_ITEMS:
        raise ValueError('Scored response exceeds allowed output bounds')
    if (response.get('user_id') != context.user_id or response.get('generation_id') != context.generation_id
            or type(response.get('feed_version')) is not int or response.get('feed_version') != int(context.feed_version)):
        raise ValueError('Response ownership/version does not match scored context')
    if any(frame.user_id != context.user_id or frame.generation_id != context.generation_id for frame in frames):
        raise ValueError('Scored frame pool ownership mismatch')
    if any(frame.context != context for frame in frames):
        raise ValueError('Scored frame pool context mismatch')
    if (response.get('served_ranker') != 'heavy' or response.get('ranker_applied') is not True
            or response.get('heavy_ranker_selected') is not True
            or response.get('fallback_code') is not None or response.get('fallback_reason') is not None
            or response.get('retrieval_mode') not in {'personalized', 'cold_start_profile'}
            or response.get('model_version') != context.model_version):
        raise ValueError('Response does not match actual heavy scoring identity')
    by_id = {frame.snapshot.repo_id:frame for frame in frames}
    if len(by_id) != len(frames):
        raise ValueError('Duplicate scored frame identity')
    selected = []
    seen = set()
    for item in items:
        repo_id = item.get('repo_id') if isinstance(item, dict) else None
        if repo_id not in by_id or repo_id in seen:
            raise ValueError('Response contains duplicate or unscored repository')
        seen.add(repo_id)
        selected.append({**item, 'scored_input':packet_for_frame(by_id[repo_id], context).model_dump(mode='json')})
    versions = sorted({context.repository_embedding_versions[by_id[row['repo_id']].snapshot.embedding_version_index]
                       for row in selected})
    expected_version = versions[0] if len(versions) == 1 else 'compatible-mixed:' + ','.join(versions)
    if (selected and (response.get('embedding_versions') != versions or response.get('embedding_version') != expected_version)):
        raise ValueError('Response embedding version binding mismatch')
    output = {**response, 'items':selected, 'scoring_context':context.model_dump(mode='json'),
              'scoring_metadata_complete':context.user_embedding_version is not None}
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))
    pieces = []
    size = 0
    for piece in encoder.iterencode(output):
        encoded = piece.encode()
        size += len(encoded)
        if size > byte_limit:
            raise ValueError('Scored response exceeds exact wire byte budget')
        pieces.append(encoded)
    return b''.join(pieces)


class ScoredInputCapture:
    """One private inference scope, bounded to150 exact candidate rows.

    The caller must derive context from the profile actually loaded and perform
    any authorized sealing/archive work before scope exit. No API uses this yet.
    """
    def __init__(self, context: ScoringContext):
        self.context = context
        self._execution_thread = current_thread()
        self._closed = False
        self._frames: list[ScoredFrame] = []
        self._ids: set[str] = set()

    @property
    def frames(self) -> tuple[ScoredFrame, ...]:
        return tuple(self._frames)

    def close(self):
        self._closed = True
        self._frames.clear()
        self._ids.clear()

    def record(self, ranker, candidates, matrix):
        if self._closed:
            raise ValueError('Scored capture scope is closed')
        if current_thread() is not self._execution_thread:
            raise ValueError('Scored capture mutation must stay in its owning execution thread')
        if (ranker.model_version != self.context.model_version
                or ranker.embedding_model != self.context.embedding_model
                or ranker.embedding_model_revision != self.context.embedding_revision
                or any(ranker.loaded_artifact_sha256.get(key) != getattr(self.context,key)
                       for key in ('model_sha256','scaler_sha256','manifest_sha256'))):
            raise ValueError('Captured context does not identify actual loaded ranker artifacts')
        if (matrix.dtype.kind != 'f' or matrix.dtype.itemsize != 4 or not matrix.flags.c_contiguous):
            raise ValueError('Actual scored capture requires contiguous float32 input')
        if len(self._frames)+len(candidates)>MAX_ITEMS or matrix.shape != (len(candidates),DIMENSIONS):
            raise ValueError('Scored capture exceeds bounded request input')
        pending=[]
        for candidate,row in zip(candidates,matrix):
            version = candidate.get('embedding_version')
            if version not in self.context.repository_embedding_versions:
                raise ValueError('Actual scored repository embedding version is unknown')
            repo_id = _canonical(candidate.get('id'))
            if repo_id in self._ids or any(frame.snapshot.repo_id == repo_id for frame in pending):
                raise ValueError('Duplicate actual scored repository input')
            def exact_version(key):
                # Fresh index payloads have no refresh counter. Match the
                # repository store's zero version without weakening content IDs.
                value=candidate.get(key,0) if key=='feature_version' else candidate[key]
                if key=='feature_version' and value is None: value=0
                if type(value) is int: return str(value)
                if isinstance(value,str): return value
                raise ValueError('Actual repository version must be exact decimal or integer')
            snapshot=RepositorySnapshot(repo_id=repo_id,content_version=exact_version('content_version'),
                feature_version=exact_version('feature_version'),content_hash=candidate['content_hash'],
                source_hash=candidate['source_hash'],embedding_version_index=self.context.repository_embedding_versions.index(version))
            frame=ScoredFrame(self.context.user_id,self.context.generation_id,snapshot,row.astype('<f4',copy=False).tobytes(),self.context)
            _validate_input(frame.input_bytes,self.context)
            pending.append(frame)
        self._frames.extend(pending)
        self._ids.update(frame.snapshot.repo_id for frame in pending)


_capture: ContextVar[ScoredInputCapture | None] = ContextVar('private_scored_input_capture',default=None)

@contextmanager
def capture_scored_inputs(context: ScoringContext):
    """Private opt-in lifetime; never activated by untrusted request/config fields."""
    if _capture.get() is not None:
        raise ValueError('Nested scored-input capture is ambiguous')
    capture=ScoredInputCapture(context)
    token=_capture.set(capture)
    try:
        yield capture
    finally:
        capture.close()
        _capture.reset(token)


def capture_ranker_input(ranker, candidates, matrix):
    capture=_capture.get()
    if capture is not None:
        capture.record(ranker,candidates,matrix)
