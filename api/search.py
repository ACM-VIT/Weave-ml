"""Backend-only semantic search; canonical repository data belongs to Node."""
from __future__ import annotations

import math
from typing import Any

from qdrant_client import models
from api.contracts import RepositorySearchRequest
from config import QDRANT_VECTOR_NAME
from embedding.vector_contract import validate_embedding_vector
from retrieval.v2_retriever import _REPOSITORY_RETRIEVAL_PAYLOAD_FIELDS


_COSINE_ROUNDOFF_TOLERANCE = 1e-6


def search_repositories(request: RepositorySearchRequest, retriever: Any, embedder: Any) -> dict:
    vector = validate_embedding_vector(
        embedder.embed_text(request.query),
        expected_size=retriever.embedding_dim,
        field_name="search query embedding",
    )
    eligibility = retriever._eligible_filter()
    must = list(eligibility.must or [])
    # One current embedding version per search response. No mixed vector spaces.
    must.append(models.FieldCondition(key="embedding_version", match=models.MatchValue(value=retriever.embedding_version)))
    for field, values in (("primary_language", request.filters.languages), ("topics", request.filters.topics)):
        if values:
            must.append(models.FieldCondition(key=field, match=models.MatchAny(any=values)))
    exclusions = {str(value) for value in request.exclude_repo_ids}
    query_filter = models.Filter(
        must=must,
        must_not=[models.HasIdCondition(has_id=sorted(exclusions))] if exclusions else None,
    )
    response = retriever.client.query_points(
        collection_name=retriever.repository_collection,
        query=vector, using=QDRANT_VECTOR_NAME, query_filter=query_filter,
        limit=request.limit, with_payload=_REPOSITORY_RETRIEVAL_PAYLOAD_FIELDS, with_vectors=False,
    )
    unique: dict[str, float] = {}
    for point in response.points:
        candidate = retriever._eligible_repository_point(point, require_vector=False)
        score = point.score
        if (candidate is None or candidate.repo_id in exclusions
            or isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not -1 - _COSINE_ROUNDOFF_TOLERANCE <= score <= 1 + _COSINE_ROUNDOFF_TOLERANCE):
            continue
        # Float32 cosine calculations can overshoot the mathematical endpoints.
        score = max(-1.0, min(1.0, float(score)))
        unique[candidate.repo_id] = max(unique.get(candidate.repo_id, -1.0), score)
    return {
        "schema_version": 2, "search_id": str(request.search_id),
        "embedding_version": retriever.embedding_version,
        "items": [{"repo_id": key, "score": score} for key, score in
                  sorted(unique.items(), key=lambda item: (-item[1], item[0]))[:request.limit]],
    }
