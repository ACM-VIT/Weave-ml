"""Export a bounded, local-only stable-metadata embedding candidate as JSONL.

This tool reads a frozen JSON snapshot, uses the pinned local MiniLM cache, and
never constructs a Qdrant store or marks candidate vectors as serving eligible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from huggingface_hub.constants import HF_HUB_CACHE

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    EMBEDDING_MODEL_REVISION,
    REPOSITORY_EMBEDDING_DIM,
    REPOSITORY_EMBEDDING_MODEL,
    STABLE_METADATA_EMBEDDING_VERSION,
)
from embedding.embedding_pipeline import RepositoryEmbeddingPipeline  # noqa: E402
from embedding.embeddings import SentenceTransformerEmbedder  # noqa: E402
from embedding.repository_embedding import (  # noqa: E402
    RepositoryEmbeddingConfig,
    build_stable_metadata_text,
)
from embedding.vector_contract import REPOSITORY_SERVING_ELIGIBILITY_FIELD  # noqa: E402
from summarization.pipeline import CardSummaryPipeline  # noqa: E402
from summarization.settings import SummarySettings  # noqa: E402


MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_EXPORT_REPOSITORIES = 32
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
OUTPUT_SCHEMA = "repository-embedding-candidate-v1"
EXPECTED_MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
SUPPORTED_MINILM_NAMES = {"all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"}


def _default_model_snapshot() -> Path:
    return (
        Path(HF_HUB_CACHE)
        / "models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
        / EXPECTED_MODEL_REVISION
    )


def _read_snapshot(
    path: Path, *, limit: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_EXPORT_REPOSITORIES:
        raise ValueError(f"limit must be between 1 and {MAX_EXPORT_REPOSITORIES}")
    with path.open("rb") as source_file:
        raw = source_file.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError(f"input snapshot exceeds {MAX_INPUT_BYTES} bytes")
    document = json.loads(raw)
    if not isinstance(document, dict) or not isinstance(document.get("repositories"), list):
        raise ValueError("input must be a frozen JSON snapshot with a repositories list")
    rows = document["repositories"]
    if not 1 <= len(rows) <= MAX_EXPORT_REPOSITORIES:
        raise ValueError(f"snapshot must contain 1 to {MAX_EXPORT_REPOSITORIES} repositories")

    sources: list[dict[str, Any]] = []
    frozen_payloads: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for row in rows[:limit]:
        if not isinstance(row, dict):
            raise ValueError("each snapshot repository must be an object")
        repo_id = row.get("repo_id")
        if not isinstance(repo_id, str):
            raise ValueError("each snapshot repository requires a canonical repo_id")
        try:
            if str(uuid.UUID(repo_id)) != repo_id:
                raise ValueError
        except (ValueError, AttributeError) as exc:
            raise ValueError("each snapshot repository requires a canonical repo_id") from exc
        if repo_id in seen_ids:
            raise ValueError("snapshot repository IDs must be unique")
        seen_ids.add(repo_id)
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("each snapshot repository requires a payload object")
        payload_id = payload.get("repo_id")
        if payload_id is not None and payload_id != repo_id:
            raise ValueError("snapshot row and payload repository IDs do not match")
        source = dict(payload)
        source["repo_id"] = repo_id
        sources.append(source)
        frozen_payloads[repo_id] = dict(payload)
    return sources, frozen_payloads


def _build_local_pipeline(model_snapshot: Path) -> RepositoryEmbeddingPipeline:
    if EMBEDDING_MODEL_REVISION != EXPECTED_MODEL_REVISION:
        raise ValueError("candidate export requires the pinned MiniLM model revision")
    if REPOSITORY_EMBEDDING_MODEL not in SUPPORTED_MINILM_NAMES:
        raise ValueError("candidate export requires all-MiniLM-L6-v2")
    if REPOSITORY_EMBEDDING_DIM != 384:
        raise ValueError("candidate export requires 384-dimensional repository vectors")
    if model_snapshot.name != EXPECTED_MODEL_REVISION or not model_snapshot.is_dir():
        raise ValueError("the exact pinned MiniLM snapshot is not present in the local cache")
    required_files = ("modules.json", "config.json", "model.safetensors", "tokenizer.json")
    if any(not (model_snapshot / name).is_file() for name in required_files):
        raise ValueError("the local MiniLM snapshot is incomplete")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        str(model_snapshot), device="cpu", local_files_only=True,
    )
    get_dimension = getattr(model, "get_embedding_dimension", None)
    dimension = get_dimension() if callable(get_dimension) else model.get_sentence_embedding_dimension()
    if dimension != REPOSITORY_EMBEDDING_DIM:
        raise ValueError("the cached MiniLM model does not produce 384-dimensional vectors")
    config = RepositoryEmbeddingConfig(version=STABLE_METADATA_EMBEDDING_VERSION)
    return RepositoryEmbeddingPipeline(
        config=config,
        embedder=SentenceTransformerEmbedder(model=model),
        store=None,
        tower_cache=None,
        card_summarizer=CardSummaryPipeline(settings=SummarySettings(api_key=None)),
    )


def _write_jsonl_exclusive(path: Path, records: list[dict[str, Any]]) -> None:
    encoded = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        for record in records
    ).encode("utf-8")
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise ValueError(f"candidate export exceeds {MAX_OUTPUT_BYTES} bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        # Hard-link publication is atomic and refuses to overwrite an existing
        # user artifact, even if one appears after argument validation.
        os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def export_snapshot(
    input_path: Path,
    output_path: Path,
    *,
    limit: int = MAX_EXPORT_REPOSITORIES,
    model_snapshot: Path | None = None,
    pipeline: RepositoryEmbeddingPipeline | None = None,
) -> int:
    """Compute and write candidate vectors without initializing serving I/O."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("candidate output cannot overwrite its input snapshot")
    if output_path.exists():
        raise FileExistsError(f"candidate output already exists: {output_path}")
    sources, frozen_payloads = _read_snapshot(input_path, limit=limit)
    active_pipeline = pipeline or _build_local_pipeline(model_snapshot or _default_model_snapshot())
    if active_pipeline.config.version != STABLE_METADATA_EMBEDDING_VERSION:
        raise ValueError("export requires the stable-metadata candidate recipe")
    if active_pipeline.store is not None or active_pipeline.tower_cache is not None:
        raise ValueError("export requires a compute-only pipeline with no store or tower cache")

    records: list[dict[str, Any]] = []
    expected_ids = {source["repo_id"] for source in sources}
    source_by_id = {source["repo_id"]: source for source in sources}
    emitted_ids: set[str] = set()
    for result in active_pipeline.iter_embed(sources):
        if (
            result.repo_id not in expected_ids
            or result.repo_id in emitted_ids
            or result.embedding_version != STABLE_METADATA_EMBEDDING_VERSION
        ):
            raise ValueError("embedding pipeline returned an incompatible candidate result")
        emitted_ids.add(result.repo_id)
        computed_payload = dict(result.payload)
        if computed_payload.get("embedding_version") != STABLE_METADATA_EMBEDDING_VERSION:
            raise ValueError("candidate payload recipe does not match its vector")
        computed_vector_fields = {
            "embedding_dim": len(result.final_embedding),
            "embedding_model": result.embedding_model,
            "embedding_model_revision": EMBEDDING_MODEL_REVISION,
            "embedding_version": result.embedding_version,
            # This count describes the candidate vector, while ranking fields stay frozen.
            "readme_chunks": result.readme_chunks,
            "source_hash": result.source_hash,
        }
        if (
            result.repo_id != computed_payload.get("repo_id")
            or result.source_hash != computed_payload.get("source_hash")
            or result.embedding_version != STABLE_METADATA_EMBEDDING_VERSION
            or any(
                computed_payload.get(key) != value
                for key, value in computed_vector_fields.items()
            )
        ):
            raise ValueError("candidate payload identity does not match its computed vector")
        payload = dict(frozen_payloads[result.repo_id])
        payload.pop(REPOSITORY_SERVING_ELIGIBILITY_FIELD, None)
        payload.update(computed_vector_fields)
        source = {
            **source_by_id[result.repo_id],
            "repo_id": computed_payload["repo_id"],
            "full_name": computed_payload["full_name"],
        }
        records.append({
            "schema": OUTPUT_SCHEMA,
            "recipe_id": STABLE_METADATA_EMBEDDING_VERSION,
            "repo_id": result.repo_id,
            "source_hash": result.source_hash,
            "vector": result.final_embedding,
            "payload": payload,
            "metadata_text": build_stable_metadata_text(source),
        })
    if emitted_ids != expected_ids:
        raise ValueError("embedding pipeline did not return one result per snapshot repository")
    _write_jsonl_exclusive(output_path, records)
    return len(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Frozen JSON snapshot with repository payloads")
    parser.add_argument("--output", type=Path, required=True, help="New local JSONL output path; existing files are preserved")
    parser.add_argument("--limit", type=int, default=MAX_EXPORT_REPOSITORIES, help=f"At most {MAX_EXPORT_REPOSITORIES} rows")
    parser.add_argument("--model-snapshot", type=Path, default=_default_model_snapshot(), help="Exact local MiniLM snapshot directory")
    args = parser.parse_args(argv)
    try:
        exported = export_snapshot(
            args.input,
            args.output,
            limit=args.limit,
            model_snapshot=args.model_snapshot,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        print(json.dumps({"exported": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({
        "exported": exported,
        "recipe_id": STABLE_METADATA_EMBEDDING_VERSION,
        "output": str(args.output.resolve()),
        "serving_eligible": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
