"""Text embedding helpers for repository documents."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from config import README_CHUNK_CHARS, README_CHUNK_OVERLAP_CHARS, REPOSITORY_EMBEDDING_MODEL
from .vector_contract import validate_embedding_vector

logger = logging.getLogger(__name__)


Vector = list[float]
MAX_ENCODER_BATCH_TEXTS = 8
MAX_ENCODER_BATCH_TOKENS = 2048


@dataclass(slots=True)
class TextChunk:
    """A chunk of source text prepared for embedding."""

    text: str
    index: int
    start_char: int
    end_char: int


class SentenceTransformerEmbedder:
    """Lazy sentence-transformers wrapper used by repository embedding stages."""

    def __init__(
        self,
        model_name: str = REPOSITORY_EMBEDDING_MODEL,
        *,
        model: object | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = model

    @property
    def model(self):
        # The below lazy load is for keeping normal imports lightweight; the
        # transformer model is initialized only when embeddings are requested.
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for repository embeddings. "
                    "Run 'uv sync' to install project dependencies."
                ) from exc
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: Sequence[str], *, normalize: bool = True) -> list[Vector]:
        """Encode ordered, token-bounded groups, releasing the model between them.

        The encoder retains its existing per-input truncation. Bound padded
        tensor tokens by the loaded sequence limit, not a word-count estimate.
        Lightweight encoders without tokenizer metadata use MiniLM's 256 cap.
        """
        if not texts:
            return []
        model = self.model
        sequence_limit = getattr(model, "max_seq_length", 256)
        if (
            isinstance(sequence_limit, bool)
            or not isinstance(sequence_limit, Integral)
            or not 1 <= sequence_limit <= MAX_ENCODER_BATCH_TOKENS
        ):
            raise ValueError("encoder max_seq_length exceeds the bounded token budget")
        group_size = min(
            MAX_ENCODER_BATCH_TEXTS, MAX_ENCODER_BATCH_TOKENS // sequence_limit
        )
        result: list[Vector] = []
        for start in range(0, len(texts), group_size):
            group = list(texts[start:start + group_size])
            vectors = model.encode(
                group,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
            if len(vectors) != len(group):
                raise ValueError("encoder must return exactly one vector per input")
            result.extend(np.asarray(vector, dtype=np.float32).tolist() for vector in vectors)
        return result

    def embed_text(self, text: str, *, normalize: bool = True) -> Vector:
        """Embed one text value."""
        vectors = self.embed_texts([text], normalize=normalize)
        return vectors[0] if vectors else []


def chunk_text(
    text: str,
    *,
    max_chars: int = README_CHUNK_CHARS,
    overlap_chars: int = README_CHUNK_OVERLAP_CHARS,
) -> list[TextChunk]:
    """Split text into overlapping chunks suitable for README embeddings."""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be greater than or equal to 0")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    clean = (text or "").strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [TextChunk(text=clean, index=0, start_char=0, end_char=len(clean))]

    chunks: list[TextChunk] = []
    start = 0
    index = 0
    while start < len(clean):
        hard_end = min(start + max_chars, len(clean))
        # The below boundary selection is for keeping README chunks readable by
        # avoiding paragraph or sentence splits when a nearby boundary exists.
        end = _best_chunk_boundary(clean, start, hard_end)
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(TextChunk(text=chunk, index=index, start_char=start, end_char=end))
            index += 1
        if end >= len(clean):
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def aggregate_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    weights: Sequence[float] | None = None,
) -> Vector:
    """Average and L2-normalize vectors."""
    if not vectors:
        return []

    expected_size = len(vectors[0])
    if expected_size == 0:
        raise ValueError("vectors must not contain empty vectors")
    validated_vectors = [
        validate_embedding_vector(
            vector,
            expected_size=expected_size,
            field_name=f"vectors[{index}]",
        )
        for index, vector in enumerate(vectors)
    ]
    matrix = np.asarray(validated_vectors, dtype=np.float32)
    if weights is not None:
        if len(weights) != len(matrix):
            raise ValueError("weights length must match vectors length")
        validated_weights: list[float] = []
        for index, raw_weight in enumerate(weights):
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, Real):
                raise TypeError(f"weights[{index}] must be a real number")
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"weights[{index}] must be finite and non-negative")
            validated_weights.append(weight)
        weight_array = np.asarray(validated_weights, dtype=np.float32)
        total = float(weight_array.sum())
        if total <= 0:
            raise ValueError("weights must have a positive total")
        vector = np.average(matrix, axis=0, weights=weight_array / total)
    else:
        vector = np.mean(matrix, axis=0)

    # The below normalization is for cosine-distance storage in Qdrant.
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("cannot normalize a zero-length or non-finite aggregate vector")
    vector = vector / norm
    return validate_embedding_vector(
        vector.astype(np.float32).tolist(),
        expected_size=expected_size,
        field_name="aggregate vector",
    )


def _best_chunk_boundary(text: str, start: int, hard_end: int) -> int:
    if hard_end >= len(text):
        return len(text)
    window = text[start:hard_end]
    for marker in ("\n\n", "\n", ". "):
        idx = window.rfind(marker)
        if idx > int(len(window) * 0.55):
            return start + idx + len(marker)
    return hard_end
