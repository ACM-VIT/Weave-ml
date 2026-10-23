"""Independent opt-in scoring capability; no collection/export policy is enabled."""
from __future__ import annotations

import os
import re
import json
import math
from dataclasses import asdict, dataclass, field

from fastapi import HTTPException, Request

from config import constant_time_secret_matches
from inference.scoring_runtime import CaptureOptions
from inference.scoring_provenance import DEFAULT_WIRE_LIMIT, packet_for_frame, seal_scoring_response
from inference.runtime import RecommendationCapacityError, RecommendationDeadlineExceeded
from api.contracts import CapturedRecommendationResponse, UnscoredEmptyRecommendationResponse
from api.metrics import record_recommendation
from retrieval.social_candidates import SocialCandidate
from utils.deadline import CooperativeDeadlineExceeded, check_deadline

CAPABILITY_HEADER = "x-scoring-capture-capability"


class CaptureHTTPException(HTTPException):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(status_code=status_code, detail=message)
        self.error_code = code
        self.retryable = False


@dataclass(frozen=True)
class CapturedRequest:
    user_id: str
    generation_id: str
    feed_version: str
    limit: int
    excluded: tuple[str, ...]
    social_ids: tuple[str, ...]
    social_candidates: tuple[SocialCandidate, ...]
    cold_start: bool
    locale: str | None
    options: CaptureOptions


@dataclass(frozen=True)
class CapturedJobOutcome:
    wire: bytes | None = field(default=None, repr=False)
    failure_code: str | None = None


def execute_captured_request(active_retriever, request: CapturedRequest) -> CapturedJobOutcome:
    """Run and seal inside the actual worker; no private exception crosses it."""
    try:
        check_deadline()
        batch = active_retriever.recommend_batch(request.user_id, request.limit, list(request.excluded),
            request.generation_id, request.cold_start, list(request.social_ids), list(request.social_candidates),
            scoring_capture=request.options)
        items = batch.items
        if len({item.repo_id for item in items}) != len(items) or any(not math.isfinite(item.score) for item in items):
            raise ValueError("Invalid actual recommendation output")
        body = {"schema_version": 2, "generation_id": request.generation_id, "user_id": request.user_id,
                "feed_version": request.feed_version, "model_version": batch.model_version,
                "embedding_version": batch.embedding_version, "embedding_versions": list(batch.embedding_versions),
                "served_ranker": batch.served_ranker, "heavy_ranker_selected": batch.heavy_ranker_selected,
                "ranker_applied": batch.ranker_applied, "fallback_code": batch.fallback_code,
                "fallback_reason": batch.fallback_reason, "retrieval_mode": batch.retrieval_mode,
                "context_status": {"cold_start_applied": request.cold_start,
                                   "locale": "reserved_unused" if request.locale else "not_provided"},
                "items": [asdict(item) for item in items]}
        if not items:
            # Only an existing genuinely successful empty result reaches here.
            # Required-heavy no-eligible behavior still raises in the retriever.
            body["scoring_capture"] = {"schema_version": 1, "status": "not_scored_empty"}
            validated = UnscoredEmptyRecommendationResponse.model_validate_json(json.dumps(body, allow_nan=False), strict=True)
            wire = json.dumps(validated.model_dump(mode="json"), ensure_ascii=False, allow_nan=False,
                              sort_keys=True, separators=(",", ":")).encode()
            if len(wire) > DEFAULT_WIRE_LIMIT:
                raise ValueError("Captured response exceeds exact wire budget")
        else:
            captured = batch.captured_scoring
            if captured is None:
                raise ValueError("Requested capture did not score actual inputs")
            by_id = {frame.snapshot.repo_id: frame for frame in captured.frames}
            scores = dict(captured.raw_heavy_scores)
            body["scoring_capture"] = {"schema_version": 1, "status": "captured"}
            body["scoring_context"] = captured.context.model_dump(mode="json")
            body["scoring_metadata_complete"] = captured.context.user_embedding_version is not None
            body["items"] = [{**item, "raw_heavy_score": scores[item["repo_id"]],
                "scored_input": packet_for_frame(by_id[item["repo_id"]], captured.context).model_dump(mode="json")}
                for item in body["items"]]
            validated = CapturedRecommendationResponse.model_validate_json(json.dumps(body, allow_nan=False), strict=True)
            normalized = validated.model_dump(mode="json")
            for item in normalized["items"]:
                item.pop("scored_input")
            wire = seal_scoring_response(normalized, captured.context, captured.frames, byte_limit=DEFAULT_WIRE_LIMIT)
        check_deadline()
        record_recommendation(served_ranker=batch.served_ranker, fallback_code=batch.fallback_code, item_count=len(items))
        return CapturedJobOutcome(wire=wire)
    except (CooperativeDeadlineExceeded, RecommendationDeadlineExceeded, RecommendationCapacityError):
        raise
    except Exception:
        # No validation object, .errors()[input], response/packet repr, traceback
        # or chained exception leaves this private scope for logging/Sentry.
        return CapturedJobOutcome(failure_code="SCORING_CAPTURE_UNAVAILABLE")


def authorized_capture_options(request: Request, feed_version: str) -> CaptureOptions:
    supplied = request.headers.getlist(CAPABILITY_HEADER)
    # Existing observability records selected trace/request IDs only. Remove
    # this additional secret from the request before downstream handlers/hooks.
    request.scope["headers"] = [(key, value) for key, value in request.scope["headers"]
                                if key.lower() != CAPABILITY_HEADER.encode()]
    if hasattr(request, "_headers"):
        del request._headers
    enabled = os.getenv("SCORING_CAPTURE_ENABLED", "false")
    if enabled not in {"true", "false"}:
        raise CaptureHTTPException(503, "SCORING_CAPTURE_CONFIG_INVALID", "Scoring capture configuration is invalid.")
    if enabled == "false":
        raise CaptureHTTPException(503, "SCORING_CAPTURE_DISABLED", "Scoring capture is disabled.")
    capability = os.getenv("SCORING_CAPTURE_CAPABILITY", "")
    commit = os.getenv("SCORING_CAPTURE_CODE_COMMIT", "")
    baked_commit = os.getenv("BAKED_ML_RELEASE_ID", "")
    if (not re.fullmatch("[a-f0-9]{64}", capability) or not re.fullmatch("[a-f0-9]{40}", commit)
            or not re.fullmatch("[a-f0-9]{40}", baked_commit) or commit != baked_commit):
        raise CaptureHTTPException(503, "SCORING_CAPTURE_CONFIG_INVALID", "Scoring capture configuration is invalid.")
    if len(supplied) != 1 or not constant_time_secret_matches(supplied[0], capability):
        raise CaptureHTTPException(403, "SCORING_CAPTURE_FORBIDDEN", "Scoring capture capability is invalid or missing.")
    return CaptureOptions(feed_version=feed_version, code_commit=baked_commit)
