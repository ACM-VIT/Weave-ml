"""End-to-end repository embedding pipeline."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import islice
from typing import Any

from utils.deadline import check_deadline

from config import TOKEN_SECTION_EMBEDDING_VERSION, ZERO_RECENCY_EMBEDDING_VERSION
from summarization.contracts import CardSummaryArtifact
from summarization.pipeline import CardSummaryPipeline

from .batch_limits import check_field_budget
from .embeddings import SentenceTransformerEmbedder, aggregate_vectors, chunk_text
from .qdrant_store import QdrantRepositoryStore
from .readme_embedding import README_STRATEGY_VERSION, prepare_readme_chunks, readme_source
from .repository_embedding import (
    RepositoryEmbeddingConfig,
    RepositoryEmbeddingResult,
    build_metadata_text,
    build_readme_text,
    build_topic_text,
    build_vector_payload,
    coerce_payload,
    combine_repo_tower,
    source_fingerprint,
)
from .tower_cache import TowerEmbeddingCache
from .vector_contract import resolve_repository_identity, validate_embedding_vector

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _EmbeddingPlan:
    repo: dict[str, Any]
    repo_id: str
    legacy_readme: bool
    readme_text: str
    metadata_text: str
    topic_text: str
    readme_chunks: list
    readme_weights: list[float] | None
    card_summary: CardSummaryArtifact
    cache: TowerEmbeddingCache | None
    fingerprints: dict
    vectors: dict
    missing: list[str]
    inputs: dict[str, list[str]]

    @property
    def texts(self) -> list[str]:
        return [text for tower in self.missing for text in self.inputs[tower]]


# Bounds apply to the offline iterator, without changing online admission.
MAX_OFFLINE_REPOSITORIES = 8
MAX_OFFLINE_SOURCE_BYTES = 1024 * 1024
MAX_OFFLINE_REPOSITORY_TEXTS = 2048


def _validate_offline_source(source: Any) -> None:
    """Reject oversized inputs without truncating their embedding identity.

    Walk fields without building a JSON copy. EnrichmentResult README fields
    live outside its payload and count against the same repository budget.
    """
    payload = source if isinstance(source, Mapping) else getattr(source, "payload", None)
    if not isinstance(payload, Mapping):
        raise TypeError("repository source must be a payload dict or an object with a payload dict")
    values = [payload]
    readme = getattr(source, "readme", None)
    if readme is not None:
        values.extend(getattr(readme, field, None) for field in (
            "raw_markdown", "readme_md", "clean_text",
        ))

    check_field_budget(values, max_bytes=MAX_OFFLINE_SOURCE_BYTES, label="offline repository")


class RepositoryEmbeddingPipeline:
    """Build README, metadata, topic, and final repository embeddings."""

    def __init__(
        self,
        *,
        config: RepositoryEmbeddingConfig | None = None,
        embedder: SentenceTransformerEmbedder | None = None,
        store: QdrantRepositoryStore | None = None,
        card_summarizer: CardSummaryPipeline | None = None,
        tower_cache: TowerEmbeddingCache | None = None,
    ) -> None:
        self.config = config or RepositoryEmbeddingConfig()
        self.embedder = embedder or SentenceTransformerEmbedder(self.config.model_name)
        self.tower_cache = tower_cache
        self.store = store
        self.card_summarizer = card_summarizer or CardSummaryPipeline()

    def summarize_repository(
        self,
        source: Any,
        repo: dict[str, Any] | None = None,
        *,
        deadline: float | None = None,
    ) -> CardSummaryArtifact:
        """Generate a versioned card artifact from bounded derived source text."""

        payload = dict(repo) if repo is not None else coerce_payload(source)
        return self.card_summarizer.summarize(source, payload, deadline=deadline)

    def embed_repository(self, source: Any, deadline: float | None = None) -> RepositoryEmbeddingResult:
        """Embed one approved repository payload or EnrichmentResult."""
        plan = self._prepare_repository(source, deadline=deadline)
        texts = plan.texts
        encoded = self.embedder.embed_texts(texts) if texts else []
        return self._finish_repository(plan, encoded)

    def _prepare_repository(
        self, source: Any, *, deadline: float | None = None, max_texts: int | None = None,
    ) -> _EmbeddingPlan:
        repo = coerce_payload(source)
        repo_id, full_name = resolve_repository_identity(repo)
        repo["repo_id"] = repo_id
        repo["full_name"] = full_name
        # The below text builders are for splitting source material into the
        # three approved towers before final weighted composition.
        # Preserve legacy projection exactly; v3 parses the bounded Markdown
        # source once instead of first flattening it through the old cleaner.
        legacy_readme = self.config.version != TOKEN_SECTION_EMBEDDING_VERSION
        readme_text = (
            build_readme_text(source) if legacy_readme else readme_source(source, repo)
        )
        metadata_text = build_metadata_text(
            repo, preserve_legacy_zero_recency=self.config.version != ZERO_RECENCY_EMBEDDING_VERSION,
        )
        topic_text = build_topic_text(repo)
        card_summary = self.summarize_repository(source, repo, deadline=deadline)

        # The reserved v2 identity must continue producing v2 vectors while a
        # separately versioned corpus and ranker are qualified for cutover.
        readme_weights = None
        readme_plan = None
        if legacy_readme:
            readme_chunks = chunk_text(
                readme_text,
                max_chars=self.config.readme_chunk_chars,
                overlap_chars=self.config.readme_chunk_overlap_chars,
                max_chunks=max_texts - 2 if max_texts is not None else None,
            )
        else:
            model = self.embedder.model
            readme_chunks = prepare_readme_chunks(
                readme_text, model.tokenizer,
                sequence_limit=model.max_seq_length,
                overlap_tokens=self.config.readme_chunk_overlap_tokens,
            )
            readme_weights = [chunk.weight for chunk in readme_chunks]
            readme_plan = [(chunk.section_class, chunk.weight) for chunk in readme_chunks]
        # A cache is supplied only by the frozen, revision-pinned runtime.
        # Unknown/injected encoders continue through the uncached path.
        cache = self.tower_cache
        if cache is not None and (
            type(self.embedder) is not SentenceTransformerEmbedder
            or not cache.owns(self.embedder.model)
        ):
            cache = None
        inputs = {
            "readme": [chunk.text for chunk in readme_chunks],
            "metadata": [metadata_text],
            "topics": [topic_text],
        }
        fingerprints = {}
        vectors = {}
        missing = []
        for tower, texts in inputs.items():
            # Preserve the empty README's absent-vector weighting semantics.
            if not texts:
                vectors[tower] = []
                continue
            key = cache.fingerprint(
                tower=tower, texts=texts, config=self.config,
                readme_plan=readme_plan if tower == "readme" else None,
            ) if cache else None
            fingerprints[tower] = key
            vector = cache.get(key, expected_size=self.config.embedding_dim) if cache else None
            if vector is None:
                missing.append(tower)
            else:
                vectors[tower] = vector

        return _EmbeddingPlan(
            repo, repo_id, legacy_readme, readme_text, metadata_text, topic_text,
            readme_chunks, readme_weights, card_summary, cache, fingerprints,
            vectors, missing, inputs,
        )

    def _finish_repository(self, plan: _EmbeddingPlan, encoded: list) -> RepositoryEmbeddingResult:
        repo, repo_id = plan.repo, plan.repo_id
        legacy_readme, readme_text = plan.legacy_readme, plan.readme_text
        metadata_text, topic_text = plan.metadata_text, plan.topic_text
        readme_chunks, readme_weights = plan.readme_chunks, plan.readme_weights
        card_summary, cache = plan.card_summary, plan.cache
        fingerprints, vectors, missing = plan.fingerprints, plan.vectors, plan.missing
        inputs = plan.inputs
        texts_to_encode = plan.texts
        if len(encoded) != len(texts_to_encode):
            raise ValueError("encoder must return exactly one vector per tower input")
        offset = 0
        for tower in missing:
            count = len(inputs[tower])
            tower_vectors = [
                validate_embedding_vector(
                    vector, expected_size=self.config.embedding_dim,
                    field_name=f"{tower} embedding for {repo_id}",
                )
                for vector in encoded[offset:offset + count]
            ]
            offset += count
            vectors[tower] = (
                aggregate_vectors(tower_vectors, weights=readme_weights)
                if tower == "readme" else tower_vectors[0]
            )
        readme_embedding = vectors["readme"]
        metadata_embedding = vectors["metadata"]
        topic_embedding = vectors["topics"]
        final_embedding = combine_repo_tower(
            readme_embedding=readme_embedding,
            metadata_embedding=metadata_embedding,
            topic_embedding=topic_embedding,
            weights=self.config.tower_weights,
        )
        final_embedding = validate_embedding_vector(
            final_embedding,
            expected_size=self.config.embedding_dim,
            field_name=f"embedding for {repo_id}",
        )

        source_hash = source_fingerprint(
            self.config.model_name,
            self.config.version,
            readme_text,
            metadata_text,
            topic_text,
        )
        if not legacy_readme:
            source_hash = source_fingerprint(source_hash, README_STRATEGY_VERSION, json.dumps({
                "chunks": [(chunk.text, chunk.section_class, chunk.weight) for chunk in readme_chunks],
                "overlap_tokens": self.config.readme_chunk_overlap_tokens,
                "max_seq_length": self.embedder.model.max_seq_length,
                "tower_weights": self.config.tower_weights,
            }, sort_keys=True, separators=(",", ":")))
        payload = build_vector_payload(
            repo,
            repo_id=repo_id,
            final_embedding=final_embedding,
            readme_chunks=len(readme_chunks),
            source_hash=source_hash,
            config=self.config,
            card_summary=card_summary,
        )
        # Publish only successful computations. These are semantic vectors,
        # never a content-version/CAS result or a cached summary/payload.
        if cache is not None:
            for tower in missing:
                cache.put(
                    fingerprints[tower], vectors[tower], expected_size=self.config.embedding_dim,
                )
        logger.info("Embedded repository %s with %d README chunks", repo_id, len(readme_chunks))
        return RepositoryEmbeddingResult(
            repo_id=repo_id,
            final_embedding=final_embedding,
            readme_embedding=readme_embedding,
            metadata_embedding=metadata_embedding,
            topic_embedding=topic_embedding,
            payload=payload,
            readme_chunks=len(readme_chunks),
            source_hash=source_hash,
            embedding_model=self.config.model_name,
            embedding_version=self.config.version,
            card_summary=card_summary,
        )

    def iter_embed(self, sources: Iterable[Any]) -> Iterator[RepositoryEmbeddingResult]:
        """Embed an ordered, bounded window with cross-repository encoder groups.

        No background producer is started. At most eight source plans and their
        bounded texts/vectors are retained; advancing the consumer admits the
        next window. A failure aborts this window and leaves earlier results
        available to a caller that checkpoints completed repository IDs.
        """
        source_iterator = iter(sources)
        while True:
            plans = []
            for source in islice(source_iterator, MAX_OFFLINE_REPOSITORIES):
                check_deadline()
                _validate_offline_source(source)
                plan = self._prepare_repository(source, max_texts=MAX_OFFLINE_REPOSITORY_TEXTS)
                if sum(len(texts) for texts in plan.inputs.values()) > MAX_OFFLINE_REPOSITORY_TEXTS:
                    raise ValueError("offline repository exceeds the 2048 encoder input budget")
                plans.append(plan)
            if not plans:
                return
            texts = [text for plan in plans for text in plan.texts]
            encoded = self.embedder.embed_texts(texts) if texts else []
            if len(encoded) != len(texts):
                raise ValueError("encoder must return exactly one vector per tower input")
            offset = 0
            for plan in plans:
                check_deadline()
                count = len(plan.texts)
                yield self._finish_repository(plan, encoded[offset:offset + count])
                offset += count
            # Release the completed window before pulling the next one.
            del plans, texts, encoded, plan, source

    def embed_batch(self, sources: Iterable[Any]) -> list[RepositoryEmbeddingResult]:
        """Materialized compatibility API; use iter_embed for bounded memory."""
        return list(self.iter_embed(sources))

    def iter_index(self, sources: Iterable[Any]) -> Iterator[RepositoryEmbeddingResult]:
        """Yield only stored results, with backpressure and bounded writes.

        A failed/ambiguous write propagates. Completed chunks remain stored;
        deterministic point IDs make an offline rerun safe. This helper must
        not be used for online canonical writes, which require content CAS.
        """
        if self.store is None:
            self.store = QdrantRepositoryStore(vector_size=self.config.embedding_dim)
        self.store.ensure_collection()
        if isinstance(self.store, QdrantRepositoryStore):
            yield from self.store.iter_upsert(
                self.iter_embed(sources), max_points=MAX_OFFLINE_REPOSITORIES,
            )
            return
        # Preserve injected stores implementing the existing upsert protocol.
        results = iter(self.iter_embed(sources))
        while chunk := list(islice(results, MAX_OFFLINE_REPOSITORIES)):
            check_deadline()
            self.store.upsert(chunk)
            yield from chunk

    def index_batch(self, sources: Iterable[Any]) -> list[RepositoryEmbeddingResult]:
        """Materialized compatibility API; use iter_index for large corpora."""
        return list(self.iter_index(sources))

    def search(self, query: str, *, limit: int = 5, exact: bool = True) -> list[dict]:
        """Embed a text query and search the configured Qdrant collection."""
        # The below query embedding is for searching with the same model/config
        # used during repository indexing.
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if self.store is None:
            self.store = QdrantRepositoryStore(vector_size=self.config.embedding_dim)
        self.store.validate_collection()
        query_vector = validate_embedding_vector(
            self.embedder.embed_text(query),
            expected_size=self.config.embedding_dim,
            field_name="query embedding",
        )
        return self.store.search(query_vector, limit=limit, exact=exact)


def embed_repositories(
    sources: Iterable[Any],
    *,
    pipeline: RepositoryEmbeddingPipeline | None = None,
) -> list[RepositoryEmbeddingResult]:
    """Convenience function for batch repository embedding."""
    active_pipeline = pipeline or RepositoryEmbeddingPipeline()
    return active_pipeline.embed_batch(sources)


def index_repositories(
    sources: Iterable[Any],
    *,
    pipeline: RepositoryEmbeddingPipeline | None = None,
) -> list[RepositoryEmbeddingResult]:
    """Convenience function for embedding and storing repositories in Qdrant."""
    active_pipeline = pipeline or RepositoryEmbeddingPipeline()
    return active_pipeline.index_batch(sources)


def iter_embed_repositories(
    sources: Iterable[Any], *, pipeline: RepositoryEmbeddingPipeline | None = None,
) -> Iterator[RepositoryEmbeddingResult]:
    """Stream repository embeddings with bounded lookahead."""
    yield from (pipeline or RepositoryEmbeddingPipeline()).iter_embed(sources)


def iter_index_repositories(
    sources: Iterable[Any], *, pipeline: RepositoryEmbeddingPipeline | None = None,
) -> Iterator[RepositoryEmbeddingResult]:
    """Stream acknowledged offline indexing progress."""
    yield from (pipeline or RepositoryEmbeddingPipeline()).iter_index(sources)
