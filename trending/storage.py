"""PostgreSQL storage for trending repositories.

This module handles database operations for trending repositories, including
schema initialization, upsert operations, and metadata tracking.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from database.connector import PostgreSQLConnector

import trending.config as config
from .logger import get_logger

logger = get_logger(__name__)


class TrendingStorage:
    """Storage manager for trending repositories in PostgreSQL.

    This class handles all database operations for trending repositories,
    including schema initialization, upsert operations, and metadata tracking.
    """

    def __init__(self, database_url: str | None = None) -> None:
        """Initialize the trending storage manager.

        Args:
            database_url: PostgreSQL database URL. If not provided,
                uses DATABASE_URL from config.
        """
        self.db_url = database_url or config.DATABASE_URL
        self.connector = PostgreSQLConnector(database_url=self.db_url)
        self.enabled = self.connector.enabled

        if not self.enabled:
            logger.warning(
                "Trending storage disabled: DATABASE_URL not set or pg8000 not installed."
            )

    def init_schema(self) -> None:
        """Initialize database schema for trending repositories.

        Creates two tables:
        1. trending_repositories - Stores trending repository data
        2. trending_metadata - Stores metadata like last refresh timestamp
        """
        if not self.enabled:
            logger.warning("Skipping schema initialization: database not enabled.")
            return

        logger.info("Initializing trending repository schema...")
        conn = None
        try:
            conn = self.connector.connect()
            cursor = conn.cursor()

            # Create trending_repositories table
            create_repos_table = f"""
            CREATE TABLE IF NOT EXISTS {config.TRENDING_TABLE_NAME} (
                repo_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                full_name VARCHAR(255) NOT NULL UNIQUE,
                name VARCHAR(200) NOT NULL,
                owner VARCHAR(100) NOT NULL,
                url VARCHAR(500) NOT NULL,
                description TEXT,
                star_count INT DEFAULT 0,
                daily_stars INT DEFAULT 0,
                fork_count INT DEFAULT 0,
                created_at TIMESTAMP,
                pushed_at TIMESTAMP,
                primary_language VARCHAR(50),
                topics JSONB DEFAULT '[]'::jsonb,
                readme TEXT,
                default_branch VARCHAR(100),
                first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                trending_rank INT
            );
            """
            cursor.execute(create_repos_table)
            conn.commit()
            logger.info(f"Table '{config.TRENDING_TABLE_NAME}' created or verified.")
            logger.info("Note: star_count represents Total Lifetime Stars, daily_stars represents stars gained today.")

            # Create trending_metadata table
            create_metadata_table = f"""
            CREATE TABLE IF NOT EXISTS {config.TRENDING_METADATA_TABLE_NAME} (
                key VARCHAR(100) PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            cursor.execute(create_metadata_table)
            conn.commit()
            logger.info(f"Table '{config.TRENDING_METADATA_TABLE_NAME}' created or verified.")

            # Create indexes for better query performance
            indexes = [
                f"CREATE INDEX IF NOT EXISTS idx_{config.TRENDING_TABLE_NAME}_star_count "
                f"ON {config.TRENDING_TABLE_NAME}(star_count DESC);",
                f"CREATE INDEX IF NOT EXISTS idx_{config.TRENDING_TABLE_NAME}_last_seen_at "
                f"ON {config.TRENDING_TABLE_NAME}(last_seen_at DESC);",
                f"CREATE INDEX IF NOT EXISTS idx_{config.TRENDING_TABLE_NAME}_primary_language "
                f"ON {config.TRENDING_TABLE_NAME}(primary_language);",
            ]

            for index_query in indexes:
                try:
                    cursor.execute(index_query)
                    conn.commit()
                except Exception as exc:
                    logger.warning(f"Failed to create index: {exc}")
                    conn.rollback()

            logger.info("Trending repository schema initialized successfully.")

        except Exception as exc:
            logger.error(f"Failed to initialize trending schema: {exc}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def upsert_repositories(
        self, repositories: list[dict[str, Any]], refresh_timestamp: datetime | None = None
    ) -> int:
        """Upsert trending repositories into the database.

        Args:
            repositories: List of normalized repository dictionaries.
            refresh_timestamp: Timestamp for this refresh cycle.
                If not provided, uses current time.

        Returns:
            Number of successfully upserted repositories.
        """
        if not self.enabled:
            logger.warning("Skipping repository upsert: database not enabled.")
            return 0

        if not repositories:
            logger.info("No repositories to upsert.")
            return 0

        refresh_ts = refresh_timestamp or datetime.now(timezone.utc)
        logger.info(f"Upserting {len(repositories)} trending repositories...")

        conn = None
        upserted_count = 0
        successful_full_names = []
        try:
            conn = self.connector.connect()
            cursor = conn.cursor()

            upsert_query = f"""
            INSERT INTO {config.TRENDING_TABLE_NAME} (
                repo_id, full_name, name, owner, url, description,
                star_count, daily_stars, fork_count, created_at, pushed_at,
                primary_language, topics, readme, default_branch,
                first_seen_at, last_seen_at, trending_rank
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                CAST(%s AS jsonb), %s, %s, %s, %s, %s
            )
            ON CONFLICT (full_name) DO UPDATE SET
                description = EXCLUDED.description,
                star_count = EXCLUDED.star_count,
                daily_stars = EXCLUDED.daily_stars,
                fork_count = EXCLUDED.fork_count,
                pushed_at = EXCLUDED.pushed_at,
                primary_language = EXCLUDED.primary_language,
                topics = EXCLUDED.topics,
                readme = EXCLUDED.readme,
                default_branch = EXCLUDED.default_branch,
                last_seen_at = EXCLUDED.last_seen_at,
                trending_rank = EXCLUDED.trending_rank;
            """

            for rank, repo in enumerate(repositories, start=1):
                savepoint_name = f"repo_upsert_{rank}"
                try:
                    cursor.execute(f"SAVEPOINT {savepoint_name};")
                    
                    # Generate UUID for new repositories
                    repo_uuid = str(uuid.uuid4())

                    # Parse timestamps
                    created_at = self._parse_timestamp(repo.get("created_at"))
                    pushed_at = self._parse_timestamp(repo.get("pushed_at"))

                    # First seen at: use current time for new repos, existing for updates
                    # We'll check if the repo exists first
                    cursor.execute(
                        f"SELECT first_seen_at FROM {config.TRENDING_TABLE_NAME} WHERE full_name = %s;",
                        (repo["full_name"],),
                    )
                    existing = cursor.fetchone()

                    if existing:
                        first_seen_at = existing[0]
                    else:
                        first_seen_at = refresh_ts

                    params = (
                        repo_uuid,
                        repo["full_name"],
                        repo["name"],
                        repo["owner"],
                        repo["url"],
                        repo["description"][:2000],  # Cap description length
                        repo["star_count"],
                        repo.get("daily_stars", 0),
                        repo["fork_count"],
                        created_at,
                        pushed_at,
                        repo["primary_language"],
                        json.dumps(repo.get("topics", [])),
                        repo.get("readme", ""),
                        repo.get("default_branch", "main"),
                        first_seen_at,
                        refresh_ts,
                        rank,
                    )

                    cursor.execute(upsert_query, params)
                    upserted_count += 1
                    successful_full_names.append(repo["full_name"])

                except Exception as exc:
                    logger.warning(f"Failed to upsert repo {repo.get('full_name')}: {exc}")
                    try:
                        cursor.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name};")
                    except Exception as rollback_exc:
                        logger.critical(f"Failed to rollback savepoint {savepoint_name}: {rollback_exc}")
                        break
                    continue

            logger.info(f"Successfully upserted {upserted_count}/{len(repositories)} repositories.")

            # Remove repositories that are no longer trending (not in current successful batch)
            # Only perform cleanup if we have a 100% successful batch write
            if successful_full_names and len(successful_full_names) == len(repositories):
                delete_query = f"""
                DELETE FROM {config.TRENDING_TABLE_NAME}
                WHERE full_name NOT IN ({', '.join(['%s'] * len(successful_full_names))});
                """
                cursor.execute(delete_query, tuple(successful_full_names))
                deleted_count = cursor.rowcount
                logger.info(f"Removed {deleted_count} repositories no longer trending.")
            elif successful_full_names and len(successful_full_names) != len(repositories):
                logger.warning(f"Skipping cleanup: batch write was not 100% successful ({len(successful_full_names)}/{len(repositories)} upserted).")

            # Update metadata with last refresh timestamp
            # Only advance timestamp if batch was 100% successful
            if len(successful_full_names) == len(repositories):
                self._update_metadata(cursor, "last_refresh", refresh_ts.isoformat())
                self._update_metadata(cursor, "repo_count", str(len(successful_full_names)))

            # Single commit at the end for atomic transaction
            conn.commit()

        except Exception as exc:
            logger.error(f"Failed to upsert repositories: {exc}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

        return upserted_count

    def get_last_refresh_time(self) -> datetime | None:
        """Get the last refresh timestamp from metadata.

        Returns:
            Last refresh timestamp, or None if not found.
        """
        if not self.enabled:
            return None

        conn = None
        try:
            conn = self.connector.connect()
            cursor = conn.cursor()

            cursor.execute(
                f"SELECT value FROM {config.TRENDING_METADATA_TABLE_NAME} WHERE key = %s;",
                ("last_refresh",),
            )
            row = cursor.fetchone()

            if row:
                last_refresh = datetime.fromisoformat(row[0])
                if last_refresh.tzinfo is None:
                    last_refresh = last_refresh.replace(tzinfo=timezone.utc)
                return last_refresh
            return None

        except Exception as exc:
            logger.error(f"Failed to get last refresh time: {exc}")
            return None
        finally:
            if conn:
                conn.close()

    def get_trending_repositories(
        self, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Fetch trending repositories from the database.

        Args:
            limit: Maximum number of repositories to fetch.
            offset: Number of repositories to skip.

        Returns:
            List of repository dictionaries.
        """
        if not self.enabled:
            return []

        conn = None
        try:
            conn = self.connector.connect()
            cursor = conn.cursor()

            query = f"""
            SELECT
                full_name, name, owner, url, description,
                star_count, daily_stars, fork_count, created_at, pushed_at,
                primary_language, topics, readme, default_branch,
                first_seen_at, last_seen_at, trending_rank
            FROM {config.TRENDING_TABLE_NAME}
            ORDER BY trending_rank ASC
            LIMIT %s OFFSET %s;
            """

            cursor.execute(query, (limit, offset))
            rows = cursor.fetchall()

            columns = [
                "full_name", "name", "owner", "url", "description",
                "star_count", "daily_stars", "fork_count", "created_at", "pushed_at",
                "primary_language", "topics", "readme", "default_branch",
                "first_seen_at", "last_seen_at", "trending_rank",
            ]

            repositories = []
            for row in rows:
                repo_dict = dict(zip(columns, row))
                # Parse JSONB fields (only if they're still strings)
                if isinstance(repo_dict["topics"], str):
                    repo_dict["topics"] = json.loads(repo_dict["topics"]) if repo_dict["topics"] else []
                repositories.append(repo_dict)

            return repositories

        except Exception as exc:
            logger.error(f"Failed to fetch trending repositories: {exc}")
            return []
        finally:
            if conn:
                conn.close()

    def cleanup_old_repositories(self, days_to_keep: int = 30) -> int:
        """Remove repositories that haven't been seen in recent refresh cycles.

        Args:
            days_to_keep: Number of days to keep repositories since last_seen_at.

        Returns:
            Number of repositories removed.
        """
        if not self.enabled:
            return 0

        conn = None
        try:
            conn = self.connector.connect()
            cursor = conn.cursor()

            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

            delete_query = f"""
            DELETE FROM {config.TRENDING_TABLE_NAME}
            WHERE last_seen_at < %s;
            """

            cursor.execute(delete_query, (cutoff_date,))
            deleted_count = cursor.rowcount
            conn.commit()

            logger.info(f"Cleaned up {deleted_count} old trending repositories.")
            return deleted_count

        except Exception as exc:
            logger.error(f"Failed to cleanup old repositories: {exc}")
            if conn:
                conn.rollback()
            return 0
        finally:
            if conn:
                conn.close()

    def _update_metadata(self, cursor, key: str, value: str) -> None:
        """Update a metadata key-value pair.

        Args:
            cursor: Database cursor.
            key: Metadata key.
            value: Metadata value.
        """
        upsert_query = f"""
        INSERT INTO {config.TRENDING_METADATA_TABLE_NAME} (key, value, updated_at)
        VALUES (%s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (key) DO UPDATE SET
            value = EXCLUDED.value,
            updated_at = CURRENT_TIMESTAMP;
        """
        cursor.execute(upsert_query, (key, value))

    @staticmethod
    def _parse_timestamp(timestamp_str: str | None) -> datetime | None:
        """Parse a timestamp string into a datetime object.

        Args:
            timestamp_str: ISO format timestamp string.

        Returns:
            Datetime object, or None if parsing fails.
        """
        if not timestamp_str:
            return None
        try:
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None
