#!/usr/bin/env python3
"""Main entry point for the GitHub Trending ingestion service.

This script provides a command-line interface for running the trending repository
ingestion engine. It supports both scheduled (daemon) mode and one-time execution.

Usage:
    # Run as a scheduled service (refreshes every 24 hours)
    uv run python trending_service.py --scheduled

    # Run a single refresh cycle
    uv run python trending_service.py --once

    # Run with custom configuration
    uv run python trending_service.py --once --limit 50 --refresh-hours 12
"""

import argparse
import logging
import os
import sys

# Load environment variables from .env file before importing config
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional, continue without it

from trending.scheduler import run_scheduler, run_once
from trending.config import validate_config, TRENDING_REPO_LIMIT_STR, TRENDING_REFRESH_HOURS_STR
from trending.logger import setup_logger


def parse_args(argv: list[str] | None = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GitHub Trending Repository Ingestion Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --scheduled      Run as a scheduled service (24-hour refresh cycle)
  %(prog)s --once           Run a single refresh cycle
  %(prog)s --once --limit 50  Fetch 50 repositories instead of default 30
        """,
    )

    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--scheduled",
        action="store_true",
        help="Run as a scheduled service (refreshes every TRENDING_REFRESH_HOURS)",
    )
    mode_group.add_argument(
        "--once",
        action="store_true",
        help="Run a single refresh cycle and exit",
    )
    parser.add_argument("--force", action="store_true", help="Override freshness for an explicitly requested --once refresh")

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=f"Number of repositories to fetch (default: {TRENDING_REPO_LIMIT_STR})",
    )

    parser.add_argument(
        "--refresh-hours",
        type=int,
        default=None,
        help=f"Refresh interval in hours (default: {TRENDING_REFRESH_HOURS_STR})",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (default: INFO)",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (default: stdout)",
    )

    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit without running",
    )

    args = parser.parse_args(argv)
    if args.force and not args.once:
        parser.error("--force requires --once")
    return args


def main(argv: list[str] | None = None):
    """Main entry point for the trending service."""
    args = parse_args(argv)

    # Setup logging
    logger = setup_logger(
        name="trending_service",
        level=args.log_level,
        log_file=args.log_file,
    )

    logger.info("GitHub Trending Ingestion Service")
    logger.info("=" * 60)

    # Override configuration if command-line arguments provided (before validation)
    if args.limit is not None:
        import trending.config as config
        config.TRENDING_REPO_LIMIT_STR = str(args.limit)
        logger.info(f"Override: TRENDING_REPO_LIMIT = {args.limit}")

    if args.refresh_hours is not None:
        import trending.config as config
        config.TRENDING_REFRESH_HOURS_STR = str(args.refresh_hours)
        logger.info(f"Override: TRENDING_REFRESH_HOURS = {args.refresh_hours}")

    # Validate configuration (after CLI overrides)
    config_errors = validate_config()
    from acquisition.config import readme_workers
    try:
        readme_workers()
    except ValueError as exc:
        config_errors.append(str(exc))
    state_path = os.getenv("TRENDING_STATE_PATH", "").strip()
    if os.getenv("TRENDING_COORDINATION_MODE") != "singleton_file" or not state_path:
        config_errors.append(
            "Choose TRENDING_COORDINATION_MODE=singleton_file and a shared persistent "
            "TRENDING_STATE_PATH; independent scheduler hosts require backend lease support"
        )
    if config_errors:
        logger.error("Configuration validation failed:")
        for error in config_errors:
            logger.error(f"  - {error}")
        sys.exit(1)

    if args.validate_config:
        missing = [
            name for name in ("GITHUB_TOKEN", "BACKEND_URL", "INTERNAL_API_SECRET")
            if not os.getenv(name)
        ]
        if missing:
            for name in missing:
                logger.error("  - %s is required", name)
            sys.exit(1)
        logger.info("Configuration validation passed.")
        sys.exit(0)

    # Check if mode is specified
    if not args.scheduled and not args.once:
        logger.error("Either --scheduled or --once must be specified (unless using --validate-config)")
        sys.exit(1)

    # Run in requested mode
    try:
        from acquisition.backend_client import BackendIngestionClient
        from trending.backend_storage import BackendTrendingStorage
        from trending.coordination import SingletonTrendingCoordinator

        backend = BackendIngestionClient(
            base_url=os.environ["BACKEND_URL"],
            internal_secret=os.environ["INTERNAL_API_SECRET"],
        )
        storage = BackendTrendingStorage(
            backend=backend,
            github_token=os.environ["GITHUB_TOKEN"],
            coordinator=SingletonTrendingCoordinator(state_path),
        )
        if args.scheduled:
            logger.info("Starting scheduled mode...")
            run_scheduler(storage=storage)
        elif args.once:
            logger.info("Starting single refresh cycle...")
            success = run_once(force=args.force, storage=storage)
            if success:
                logger.info("Single refresh cycle completed successfully.")
                sys.exit(0)
            else:
                logger.error("Single refresh cycle failed.")
                sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.error(f"Fatal error: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
