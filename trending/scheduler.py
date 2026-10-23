"""Schedule GitHub Trending collection and backend snapshot delivery."""

from __future__ import annotations

from contextlib import nullcontext
import logging
import signal
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False

import trending.config as config
from .errors import IncompleteTrendingSnapshot
from .fetcher import TrendingFetcher
from .storage import TrendingStorage
from .logger import get_logger

logger = get_logger(__name__)


class TrendingScheduler:
    """Scheduler for trending repository ingestion engine.

    This class manages the periodic refresh of trending repositories using
    the schedule library. It supports both scheduled and one-time execution.
    """

    def __init__(
        self,
        fetcher: TrendingFetcher | None = None,
        storage: Any | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize the trending scheduler.

        Args:
            fetcher: Optional TrendingFetcher instance. If not provided,
                a new fetcher will be created.
            storage: Delivery adapter. Production injects
                ``BackendTrendingStorage``; the default PostgreSQL adapter is
                retained only for local tests.
        """
        if not HAS_SCHEDULE:
            raise ImportError(
                "schedule library is not installed. "
                "Run 'uv sync' to install the scheduling dependency."
            )

        self.fetcher = fetcher or TrendingFetcher()
        self.storage = storage or TrendingStorage()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.running = False
        self.scheduler = schedule.Scheduler()

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        if threading.current_thread() is not threading.main_thread():
            logger.warning("Signal handlers can only be set from the main thread. Skipping signal handler setup.")
            return
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals gracefully.

        Args:
            signum: Signal number.
            frame: Current stack frame.
        """
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        self.stop()

    def refresh_trending_repositories(self, force: bool = False) -> bool:
        """Claim singleton ownership before freshness checks or discovery."""
        try:
            errors = config.validate_config()
            if errors:
                logger.error("Trending configuration errors: %s", "; ".join(errors))
                return False
            session = getattr(self.storage, "refresh_session", None)
            with (session(force=force, refresh_hours=config.TRENDING_REFRESH_HOURS)
                  if callable(session) else nullcontext(True)) as eligible:
                if not eligible:
                    return True
                return self._refresh_trending_repositories(force=force)
        except Exception:
            logger.exception("Trending refresh coordination failed")
            return False

    def _refresh_trending_repositories(self, force: bool = False) -> bool:
        """Fetch trending repositories and publish one complete snapshot.

        This is the main refresh operation that:
        1. Fetches trending repositories from GitHub
        2. Delivers them through the configured storage boundary
        3. Records the successful refresh timestamp

        Returns:
            True if refresh succeeded, False otherwise.
        """
        logger.info("=" * 60)
        logger.info("Starting trending repository refresh cycle")
        logger.info("=" * 60)

        try:
            # Validate configuration
            config_errors = config.validate_config()
            
            if config_errors:
                logger.error("Configuration errors:")
                for error in config_errors:
                    logger.error(f"  - {error}")
                return False

            # Check if enough time has passed since last refresh (24-hour guardrail)
            if not force:
                last_refresh = self.storage.get_last_refresh_time()
                if last_refresh:
                    time_since_refresh = self._clock() - last_refresh
                    hours_since_refresh = time_since_refresh.total_seconds() / 3600
                    if hours_since_refresh < config.TRENDING_REFRESH_HOURS:
                        logger.info(
                            f"Skipping refresh: only {hours_since_refresh:.1f} hours "
                            f"since last refresh (required: {config.TRENDING_REFRESH_HOURS} hours)"
                        )
                        return True  # Return True to indicate no error, just skipped

            # Initialize storage schema if needed
            if self.storage.enabled:
                self.storage.init_schema()
            else:
                logger.warning("Storage not enabled. Skipping snapshot delivery.")
                return False

            # Fetch trending repositories
            logger.info("Fetching trending repositories from GitHub...")
            repositories = self.fetcher.fetch_trending_repositories()

            if not repositories:
                logger.warning("No repositories fetched from GitHub.")
                return False

            logger.info(f"Fetched {len(repositories)} repositories from GitHub.")

            # The production adapter publishes one atomic backend-v2 snapshot.
            refresh_timestamp = self._clock()
            # Bound retry starts, not the duration of already-active HTTP calls.
            retry_deadline = time.monotonic() + 120
            for attempt in range(3):
                try:
                    upserted_count = self.storage.upsert_repositories(
                        repositories, refresh_timestamp
                    )
                    break
                except IncompleteTrendingSnapshot:
                    # Reuse the adapter's successful records before its cache
                    # expires. Never retry configuration, invalid records or
                    # backend publication errors through this enrichment loop.
                    if attempt == 2 or time.monotonic() + 1 >= retry_deadline:
                        raise
                    logger.warning("Incomplete trending enrichment; retrying same snapshot (%d/3)", attempt + 2)
                    time.sleep(1)
                    if time.monotonic() >= retry_deadline:
                        raise

            # Check if all repositories were successfully upserted
            if upserted_count != len(repositories):
                logger.warning(
                    f"Partial snapshot delivery: only {upserted_count}/{len(repositories)} repositories accepted."
                )
                return False

            logger.info(
                "Successfully published %d repositories through the configured "
                "snapshot boundary.",
                upserted_count,
            )

            # Log summary
            last_refresh = self.storage.get_last_refresh_time()
            if last_refresh:
                logger.info(f"Last refresh timestamp: {last_refresh.isoformat()}")

            logger.info("=" * 60)
            logger.info("Trending repository refresh cycle completed successfully")
            logger.info("=" * 60)

            return True

        except Exception as exc:
            logger.error(f"Trending repository refresh failed: {exc}", exc_info=True)
            logger.info("=" * 60)
            logger.info("Trending repository refresh cycle failed")
            logger.info("=" * 60)
            return False

    def start_scheduled(self) -> None:
        """Start the scheduled refresh cycle.

        This method blocks and runs the refresh cycle every TRENDING_REFRESH_HOURS.
        Use stop() to gracefully shutdown the scheduler.
        """
        if self.running:
            logger.warning("Scheduler is already running.")
            return

        self._setup_signal_handlers()
        logger.info(f"Starting trending scheduler (refresh every {config.TRENDING_REFRESH_HOURS} hours)")
        self.running = True

        # Schedule the refresh job
        # Read config dynamically to ensure CLI overrides are respected
        refresh_hours = config.TRENDING_REFRESH_HOURS
        self.scheduler.every(refresh_hours).hours.do(self.refresh_trending_repositories)

        # Run once immediately on startup
        logger.info("Running initial refresh on startup...")
        self.refresh_trending_repositories(force=False)

        # Main scheduling loop
        try:
            while self.running:
                self.scheduler.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down...")
        finally:
            self.running = False
            self.scheduler.clear()
            logger.info("Scheduler stopped.")

    def start_once(self, force: bool = False) -> bool:
        """Run a single refresh cycle and return.

        Args:
            force: If True, bypass the 24-hour guardrail.

        Returns:
            True if refresh succeeded, False otherwise.
        """
        logger.info("Running single refresh cycle...")
        return self.refresh_trending_repositories(force=force)

    def stop(self) -> None:
        """Stop the scheduled refresh cycle."""
        if not self.running:
            logger.warning("Scheduler is not running.")
            return

        logger.info("Stopping trending scheduler...")
        self.running = False
        self.scheduler.clear()


def run_scheduler(*, storage: Any | None = None) -> None:
    """Entry point for running the trending scheduler.

    This function starts the scheduled refresh cycle and blocks until
    interrupted. Use this as the main entry point for the trending service.
    """
    logger.info("Initializing trending scheduler...")

    try:
        scheduler = TrendingScheduler(storage=storage)
        scheduler.start_scheduled()
    except Exception as exc:
        logger.error(f"Failed to start trending scheduler: {exc}", exc_info=True)
        raise


def run_once(force: bool = False, *, storage: Any | None = None) -> bool:
    """Entry point for running a single refresh cycle.

    Args:
        force: If True, bypass the 24-hour guardrail.

    Returns:
        True if refresh succeeded, False otherwise.
    """
    logger.info("Initializing single refresh cycle...")

    try:
        scheduler = TrendingScheduler(storage=storage)
        return scheduler.start_once(force=force)
    except Exception as exc:
        logger.error(f"Failed to run single refresh cycle: {exc}", exc_info=True)
        return False
