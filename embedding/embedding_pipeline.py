"""End-to-end repository embedding pipeline."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any

from config import TOKEN_SECTION_EMBEDDING_VERSION
from summarization.contracts import CardSummaryArtifact
from summarization.pipeline import CardSummaryPipeline

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
        metadata_text = build_metadata_text(repo)
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

        # Fill the same bounded encoder groups with only the missing towers.
        texts_to_encode = [text for tower in missing for text in inputs[tower]]
        encoded = self.embedder.embed_texts(texts_to_encode) if texts_to_encode else []
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

    def embed_batch(self, sources: Iterable[Any]) -> list[RepositoryEmbeddingResult]:
        """Embed multiple approved repositories."""
        return [self.embed_repository(source) for source in sources]

    def index_batch(self, sources: Iterable[Any]) -> list[RepositoryEmbeddingResult]:
        """Embed repositories and upsert them to Qdrant."""
        # The below lazy store initialization is for allowing embedding-only
        # callers to run without a Qdrant instance.
        if self.store is None:
            self.store = QdrantRepositoryStore(vector_size=self.config.embedding_dim)
        self.store.ensure_collection()
        results = self.embed_batch(sources)
        self.store.upsert(results)
        return results

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
