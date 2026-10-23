"""Request-owned exact scoring metadata; callers authorize capture separately."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Callable

from inference.feature_spec import FEATURE_ORDER
from inference.scoring_provenance import ScoredFrame, ScoringContext, capture_scored_inputs


def exact_version(value, *, positive=False, absent_zero=False) -> str:
    if value is None and absent_zero:
        value = 0
    if type(value) is int:
        value = str(value)
    if not isinstance(value, str) or not re.fullmatch(r'0|[1-9][0-9]{0,18}', value):
        raise ValueError('Scoring revision must be an exact canonical decimal')
    if int(value) > 2**63 - 1 or (positive and value == '0'):
        raise ValueError('Scoring revision is outside its bounded range')
    return value


@dataclass(frozen=True)
class CaptureOptions:
    feed_version: str
    code_commit: str

    def __post_init__(self):
        # These values come from the authenticated server request/release, never
        # a profile payload or a second asynchronous state read.
        if type(self.feed_version) is not str:
            raise ValueError('Captured feed revision must remain a decimal string')
        exact_version(self.feed_version, positive=True)
        if not isinstance(self.code_commit, str) or not re.fullmatch('[a-f0-9]{40}', self.code_commit):
            raise ValueError('Captured release identity must be a full commit')


@dataclass(frozen=True)
class CapturedScoring:
    context: ScoringContext = field(repr=False)
    frames: tuple[ScoredFrame, ...] = field(repr=False)
    raw_heavy_scores: tuple[tuple[str, float], ...] = field(repr=False)


def scoring_context_for(*, ranker, user_id, generation_id, user_vector, profile,
                        candidates, metadata, options: CaptureOptions) -> ScoringContext:
    """Use the exact vector and payload from one existing profile retrieval."""
    if not candidates:
        raise ValueError('No actual heavy-scored rows are available for capture')
    versions = []
    for item in candidates:
        payload = metadata.get(item.repo_id)
        if not isinstance(payload, dict) or not isinstance(payload.get('embedding_version'), str):
            raise ValueError('Actual candidate embedding identity is unavailable')
        versions.append(payload['embedding_version'])
    # Use the same conversion as _score_batch/_input_tensor. The collector then
    # checks this hash against the actual float32 matrix prefix, independently.
    user_bytes = ranker._embedding(user_vector, field_name='captured user vector').astype('<f4', copy=False).tobytes()
    feature_bytes = json.dumps(FEATURE_ORDER, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    hashes = ranker.loaded_artifact_sha256
    return ScoringContext(
        user_id=user_id, generation_id=generation_id,
        scored_at=datetime.now(timezone.utc).isoformat(), feed_version=options.feed_version,
        profile_version=exact_version(profile.get('profile_version'), positive=True),
        feedback_version=exact_version(profile.get('last_feedback_version'), absent_zero=True),
        feedback_state_revision=exact_version(profile.get('feedback_state_revision'), absent_zero=True),
        embedding_model=ranker.embedding_model, embedding_revision=ranker.embedding_model_revision,
        user_embedding_version=profile.get('embedding_version'),
        repository_embedding_versions=tuple(sorted(set(versions))),
        user_vector_sha256=hashlib.sha256(user_bytes).hexdigest(),
        model_version=ranker.model_version, model_sha256=hashes.get('model_sha256'),
        scaler_sha256=hashes.get('scaler_sha256'), manifest_sha256=hashes.get('manifest_sha256'),
        feature_spec_version=ranker.feature_spec_version,
        feature_order_sha256=hashlib.sha256(feature_bytes).hexdigest(), code_commit=options.code_commit,
    )


def score_with_capture(context: ScoringContext, score: Callable):
    """Create and close the collector inside the actual scoring worker thread.

    Only immutable finalized material crosses the worker boundary. Ordinary
    requests do not call this function or create a collector.
    """
    with capture_scored_inputs(context) as capture:
        result = score()
        ranked = result[0]
        if ranked is None or not ranked:
            raise ValueError('Required heavy scoring did not produce captured rows')
        frames = capture.frames
        scores = tuple((item.repo_id, float(item.score)) for item in ranked)
        if (len(scores) != len(frames) or len({repo_id for repo_id, _ in scores}) != len(scores)
                or {repo_id for repo_id, _ in scores} != {frame.snapshot.repo_id for frame in frames}
                or any(not math.isfinite(value) for _, value in scores)):
            raise ValueError('Actual heavy scores and captured input rows disagree')
        finalized = CapturedScoring(context, frames, scores)
    return result, finalized
