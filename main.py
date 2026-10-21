"""Command-line entry point for the production repository corpus pipeline.

The command performs a bounded, resumable pass through discovery, enrichment,
quality filtering, and authenticated delivery to backend v2. The backend owns
PostgreSQL identity and schedules ML/Qdrant work through its durable outbox.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("pipeline.acquisition")


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def run_acquisition(
    token: str,
    *,
    limit: int = 150,
    batch_size: int = 15,
    workers: int = 4,
    existing_repos: set[str] | None = None,
) -> list[Any]:
    """Backward-compatible list-returning acquisition wrapper."""
    from acquisition.pipeline import run_acquisition as _run_acquisition

    return _run_acquisition(
        token,
        limit=limit,
        batch_size=batch_size,
        workers=workers,
        existing_repos=existing_repos,
    )


def filter_enriched(
    enriched: list[Any],
    *,
    min_readme_chars: int = 200,
) -> tuple[list[Any], list[tuple[Any, list[str]]]]:
    """Split enriched repositories into approved and rejected audit groups."""
    if min_readme_chars < 1:
        raise ValueError("min_readme_chars must be a positive integer")

    kept: list[Any] = []
    dropped: list[tuple[Any, list[str]]] = []
    for repository in enriched:
        payload = repository.payload
        reasons: list[str] = []
        readme_length = payload.get("readme_length", 0)
        if readme_length == 0:
            reasons.append("no README")
        elif readme_length < min_readme_chars:
            reasons.append(
                f"README too thin ({readme_length} chars < {min_readme_chars})"
            )

        if not any(
            (
                bool((payload.get("description") or "").strip()),
                bool(payload.get("languages")),
                bool(payload.get("topics")),
            )
        ):
            reasons.append("shell repo: no description, languages, or topics")

        if reasons:
            dropped.append((repository, reasons))
        else:
            kept.append(repository)
    return kept, dropped


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError(
            f"must be a positive integer (>= 1), got {number}"
        )
    return number


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from acquisition.config import positive_int

    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Corpus pipeline: Discovery → Enrichment → Quality Filter → Backend v2"
        ),
    )
    parser.add_argument("--limit", type=_positive_int, default=150)
    parser.add_argument("--batch-size", type=_positive_int, default=15)
    parser.add_argument("--workers", type=_positive_int, default=4)
    parser.add_argument("--min-readme-chars", type=_positive_int, default=200)
    parser.add_argument("--corpus-target", type=_positive_int, default=None)
    parser.add_argument("--max-cycles", type=_positive_int, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate offline-worker configuration without opening network connections",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.corpus_target is None:
        args.corpus_target = positive_int(
            os.getenv("CORPUS_TARGET_COUNT", "50000"),
            name="CORPUS_TARGET_COUNT",
        )
    if args.max_cycles is None:
        args.max_cycles = positive_int(
            os.getenv("ACQUISITION_MAX_CYCLES", "1"),
            name="ACQUISITION_MAX_CYCLES",
        )
    if args.checkpoint_path is None:
        args.checkpoint_path = Path(
            os.getenv(
                "ACQUISITION_CHECKPOINT_PATH",
                ".cache/acquisition_checkpoint.json",
            )
        )
    return args


def _configuration_errors(args: argparse.Namespace) -> list[str]:
    """Return configuration errors without contacting external services."""
    errors: list[str] = []
    from acquisition.config import metadata_batch_size

    try:
        metadata_batch_size()
    except ValueError as exc:
        errors.append(str(exc))
    token = os.getenv("GITHUB_TOKEN")
    if not token or token == "your_github_token_here":
        errors.append("GITHUB_TOKEN is missing or still uses the placeholder value")
    if not os.getenv("BACKEND_URL"):
        errors.append("BACKEND_URL is required for production corpus ingestion")
    if not os.getenv("INTERNAL_API_SECRET"):
        errors.append("INTERNAL_API_SECRET is required for production corpus ingestion")
    return errors


def main(argv: list[str] | None = None) -> int:
    """Run one bounded corpus-ingestion invocation and return an exit code."""
    try:
        args = _parse_args(argv)
    except ValueError as exc:
        _setup_logging()
        logger.error("Invalid corpus configuration: %s", exc)
        return 1
    _setup_logging(args.log_level)

    if args.validate_config:
        errors = _configuration_errors(args)
        if errors:
            for error in errors:
                logger.error("Configuration error: %s", error)
            return 1
        logger.info("Corpus worker configuration is valid; no connections were opened.")
        return 0

    token = os.getenv("GITHUB_TOKEN")
    if not token or token == "your_github_token_here":
        logger.error("Set GITHUB_TOKEN in your environment or .env file first.")
        return 1

    from acquisition.checkpoint import CorpusCheckpoint
    from acquisition.config import CorpusPipelineSettings
    from acquisition.corpus_pipeline import CorpusPipeline
    from acquisition.pipeline import enrich_repository_ids, run_acquisition_detailed
    from acquisition.backend_client import BackendIngestionClient

    settings = CorpusPipelineSettings(
        target_count=args.corpus_target,
        max_cycles=args.max_cycles,
        checkpoint_path=args.checkpoint_path,
    ).validated()

    def acquire(**kwargs: Any):
        return run_acquisition_detailed(token, metadata_projection="delivery", **kwargs)

    def retry(names: list[str]):
        return enrich_repository_ids(
            token,
            names,
            batch_size=args.batch_size,
            workers=args.workers,
            metadata_projection="delivery",
        )

    backend = BackendIngestionClient(
        base_url=os.environ["BACKEND_URL"],
        internal_secret=os.environ["INTERNAL_API_SECRET"],
    )

    pipeline = CorpusPipeline(
        database=backend,
        acquire=acquire,
        acquire_retries=retry,
        quality_filter=filter_enriched,
        indexer=lambda _sources: [],
        settings=settings,
        checkpoint=CorpusCheckpoint(settings.checkpoint_path),
        allow_qdrant_without_postgres=False,
        indexing_enabled=False,
    )
    try:
        report = pipeline.run(
            limit=args.limit,
            batch_size=args.batch_size,
            workers=args.workers,
            min_readme_chars=args.min_readme_chars,
        )
    except Exception as exc:
        logger.error("Corpus pipeline failed: %s", exc, exc_info=True)
        return 1

    logger.info("Corpus run report: %s", report.as_dict())
    return 2 if report.failures and not (report.persisted or report.indexed) else 0


if __name__ == "__main__":
    sys.exit(main())
