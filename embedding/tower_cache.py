"""Process-local reuse of validated tower vectors for one frozen encoder.

No source text, repository payloads, summaries or durable state are retained.
There is no single-flight queue: misses run in the existing admitted worker,
so this cache adds no in-flight work or waiting slots of its own.
"""
from __future__ import annotations

import hashlib
import json
import threading
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from config import REPOSITORY_FEATURE_SPEC_VERSION
from .embeddings import MAX_ENCODER_BATCH_TEXTS, MAX_ENCODER_BATCH_TOKENS
from .vector_contract import validate_embedding_vector


# Bump when text projection, chunking, normalization or aggregation changes.
TOWER_PREPROCESSING_VERSION = "repository-tower-text-chunk-mean-l2-v1"


@dataclass(frozen=True, slots=True)
class _Entry:
    fingerprint: bytes
    data: bytes


class TowerEmbeddingCache:
    """Entry- and byte-bounded LRU, scoped to an exact immutable loaded model.

    Replacing the model requires a new cache; undeclared in-place mutation of
    frozen model weights/tokenizer is unsupported. No reuse survives a restart
    or crosses processes. Packed doubles preserve the existing Python vectors.
    """

    def __init__(
        self, *, model: Any, model_revision: str,
        max_entries: int = 768, max_bytes: int = 2_359_296,
    ) -> None:
        for name, value in (("max_entries", max_entries), ("max_bytes", max_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(model_revision, str) or not model_revision.strip():
            raise ValueError("model_revision must identify the loaded model")
        self.model = model
        self._encoder = getattr(model, "_model", model)
        self.model_revision = model_revision
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._entries: OrderedDict[bytes, _Entry] = OrderedDict()
        self._bytes = 0
        self._hits = self._misses = self._invalid = 0

    def owns(self, model: Any) -> bool:
        return model is self.model and getattr(model, "_model", model) is self._encoder

    def fingerprint(self, *, tower: str, texts: list[str], config: Any) -> bytes:
        """Hash exact semantic inputs and the complete tower computation spec."""
        spec = {
            "tower": tower, "texts": texts,
            "model": config.model_name, "model_revision": self.model_revision,
            "embedding_version": config.version, "dimension": config.embedding_dim,
            "preprocessing": TOWER_PREPROCESSING_VERSION,
            "feature_spec": REPOSITORY_FEATURE_SPEC_VERSION,
            "encoder_batch_texts": MAX_ENCODER_BATCH_TEXTS,
            "encoder_batch_tokens": MAX_ENCODER_BATCH_TOKENS,
            "chunk_chars": config.readme_chunk_chars,
            "chunk_overlap_chars": config.readme_chunk_overlap_chars,
            "weights": config.tower_weights,
            "max_seq_length": getattr(self.model, "max_seq_length", 256),
            "normalize": True, "encoder_output": "float32",
        }
        return hashlib.sha256(json.dumps(
            spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")).digest()

    def get(self, key: bytes, *, expected_size: int) -> list[float] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                try:
                    if entry.fingerprint != key or len(entry.data) != expected_size * 8:
                        raise ValueError("cached tower identity or dimension mismatch")
                    values = array("d")
                    values.frombytes(entry.data)
                    vector = validate_embedding_vector(
                        values.tolist(), expected_size=expected_size,
                        field_name="cached tower embedding",
                    )
                except (TypeError, ValueError, OverflowError):
                    self._bytes -= len(self._entries.pop(key).data)
                    self._invalid += 1
                else:
                    self._entries.move_to_end(key)
                    self._hits += 1
                    return vector
            self._misses += 1
            return None

    def put(self, key: bytes, vector: list[float], *, expected_size: int) -> None:
        validated = validate_embedding_vector(
            vector, expected_size=expected_size, field_name="tower embedding",
        )
        data = array("d", validated).tobytes()
        if len(data) > self.max_bytes:
            return
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= len(previous.data)
            while self._entries and (
                len(self._entries) >= self.max_entries
                or self._bytes + len(data) > self.max_bytes
            ):
                _, removed = self._entries.popitem(last=False)
                self._bytes -= len(removed.data)
            self._entries[key] = _Entry(key, data)
            self._bytes += len(data)

    def snapshot(self) -> dict[str, int]:
        """Fixed-cardinality counters for measuring reuse before durable storage."""
        with self._lock:
            return {
                "hits": self._hits, "misses": self._misses,
                "invalid": self._invalid, "entries": len(self._entries),
                "vector_bytes": self._bytes,
            }
