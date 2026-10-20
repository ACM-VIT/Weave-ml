"""PostgreSQL database connector for repository ingestion and updates.

Supports both local PostgreSQL and cloud-hosted Supabase databases.
Connection is configured via DATABASE_URL environment variable:

  Local:    postgresql://user@localhost:5432/gh_social
  Supabase: postgresql://postgres.xxx:password@host:port/postgres
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
import ssl
import re
import uuid
from urllib.parse import urlparse, parse_qs
from typing import Any

from acquisition.identity import normalize_repository_name, repository_identity_key

try:
    import pg8000.dbapi
    HAS_PG8000 = True
except ImportError:
    HAS_PG8000 = False

logger = logging.getLogger("pipeline.database")

# Supabase host patterns that require SSL
_SUPABASE_HOST_RE = re.compile(
    r"\.supabase\.(co|com|in|io)$|supabase\.co$|pooler\.supabase\.com$", re.I
)


@dataclass(slots=True)
class RepositoryUpsertResult:
    """Exact per-repository outcome for a corpus persistence batch."""

    succeeded: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.succeeded)


class PostgreSQLConnector:
    """Connector for standard PostgreSQL and Supabase databases.

    Automatically detects whether the target is a local PostgreSQL instance
    or a Supabase-hosted database and configures SSL accordingly.
    """
    _migration_columns = [
        ("owner_id", "VARCHAR(100)"),
        ("repo_name", "VARCHAR(200)"),
        ("full_name", "VARCHAR(255)"),
        ("description", "TEXT"),
        ("primary_language", "VARCHAR(50)"),
        ("special_label", "VARCHAR(50)"),
        ("language_used", "JSONB DEFAULT '[]'::jsonb"),
        ("topics", "JSONB DEFAULT '[]'::jsonb"),
        ("readme_summary", "TEXT"),
        ("readme_md", "TEXT"),
        ("star_count", "INT DEFAULT 0"),
        ("forks_count", "INT DEFAULT 0"),
        ("pr_count", "INT DEFAULT 0"),
        ("likes_count", "INT DEFAULT 0"),
        ("comments_count", "INT DEFAULT 0"),
        ("saves_count", "INT DEFAULT 0"),
        ("views_count", "INT DEFAULT 0"),
    ]


    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.enabled = bool(self.database_url)
        import threading
        self._local = threading.local()

        if not self.enabled:
            logger.warning(
                "DATABASE_URL is not set. Database integration will be disabled. "
                "Set it in your .env file — e.g. postgresql://user@localhost:5432/gh_social"
            )
            return

        if not HAS_PG8000:
            logger.warning(
                "DATABASE_URL is set but pg8000 is not installed. Database integration will be disabled. "
                "Run 'uv sync' to install the database storage dependency."
            )
            self.enabled = False
            return

        try:
            self.conn_params = self._parse_url(self.database_url)
            self._is_supabase = self._detect_supabase(self.database_url)
        except Exception as exc:
            logger.error(f"Failed to parse DATABASE_URL: {exc}. Database integration disabled.")
            self.enabled = False

    # ── URL Parsing ───────────────────────────────────────────────────────────

    @staticmethod
    def _get_sslmode(url: str) -> str | None:
        """Get the sslmode query parameter from the URL."""
        qs = parse_qs(urlparse(url).query)
        return qs.get("sslmode", [None])[0]

    def _parse_url(self, url: str) -> dict[str, Any]:
        """Parse PostgreSQL connection URL into pg8000 parameters.

        Handles:
          - postgresql:// and postgres:// schemes
          - Password-less local connections (Homebrew/peer auth)
          - Supabase URLs with password and SSL
          - Query-string parameters (?sslmode=require)
        """
        result = urlparse(url)
        username = result.username or os.getenv("USER", "postgres")
        password = result.password  # None for local, present for Supabase
        database = result.path.lstrip("/") if result.path else "postgres"
        hostname = result.hostname or "localhost"
        port = result.port or 5432

        params: dict[str, Any] = {
            "user": username,
            "host": hostname,
            "port": int(port),
            "database": database,
        }

        # Only include password if it's actually set (local PG often has no password)
        if password:
            params["password"] = password

        sslmode = self._get_sslmode(url)
        is_supabase = self._detect_supabase(url)

        # Configure SSL context if remote/Supabase or explicit sslmode is requested
        if is_supabase or sslmode in ("require", "verify-ca", "verify-full"):
            ssl_context = ssl.create_default_context()
            
            if sslmode in ("verify-ca", "verify-full"):
                ssl_context.check_hostname = (sslmode == "verify-full")
                ssl_context.verify_mode = ssl.CERT_REQUIRED
            else:
                # 'require' or default Supabase behavior:
                # Relax checks for hostname and certificate to allow connecting to poolers/proxies
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
            params["ssl_context"] = ssl_context

        return params

    @staticmethod
    def _detect_supabase(url: str) -> bool:
        """Return True if the URL points to a Supabase-hosted database."""
        hostname = urlparse(url).hostname or ""
        return bool(_SUPABASE_HOST_RE.search(hostname))

    # ── Connection Management ─────────────────────────────────────────────────

    def connect(self) -> pg8000.dbapi.Connection:
        """Establish a new connection to the PostgreSQL database."""
        if not self.enabled:
            raise RuntimeError("Database connector is not enabled (missing or invalid DATABASE_URL).")
        return pg8000.dbapi.connect(**self.conn_params)

    def _get_connection(self) -> pg8000.dbapi.Connection:
        """Get a reusable connection, reconnecting if the previous one is stale."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                # Lightweight health-check
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                return conn
            except Exception:
                # Connection is dead — close and reconnect
                try:
                    conn.close()
                except Exception:
                    pass
                self._local.conn = None

        self._local.conn = self.connect()
        return self._local.conn

    def verify_connection(self) -> bool:
        """Test the database connection and return True if successful.

        Logs detailed diagnostics on failure. Useful at startup to
        distinguish "no DATABASE_URL" from "URL is set but connection fails".
        """
        if not self.enabled:
            return False

        try:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            conn.close()
            logger.info(f"Database connection verified: {version}")
            return True
        except Exception as exc:
            host = self.conn_params.get("host", "?")
            port = self.conn_params.get("port", "?")
            db = self.conn_params.get("database", "?")
            user = self.conn_params.get("user", "?")
            has_pw = "yes" if self.conn_params.get("password") else "no"
            has_ssl = "yes" if "ssl_context" in self.conn_params else "no"
            logger.error(
                f"Database connection FAILED: {exc}\n"
                f"  host={host}  port={port}  database={db}  user={user}  "
                f"password_set={has_pw}  ssl={has_ssl}"
            )
            return False

    # ── Schema Initialization ─────────────────────────────────────────────────

    def init_db(self) -> None:
        """Initialize pgcrypto extension and the Repo table if they do not exist."""
        if not self.enabled:
            return

        logger.info("Initializing PostgreSQL database schemas...")
        conn = None
        try:
            conn = self.connect()
            cursor = conn.cursor()

            # Enable pgcrypto for UUID gen_random_uuid()
            try:
                cursor.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto";')
                conn.commit()
            except Exception as exc:
                # pgcrypto is enabled by default on Supabase and most managed PG hosts.
                # On local PG it might need superuser — not fatal.
                logger.warning(f"Could not enable pgcrypto extension: {exc}. Continuing...")
                conn.rollback()  # clear the failed transaction state

            # Create table if missing — matches the backend team's schema
            create_table_query = """
            CREATE TABLE IF NOT EXISTS Repo (
                repo_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                github_repo_url VARCHAR(500) NOT NULL UNIQUE,
                owner_id VARCHAR(100) NOT NULL,
                repo_name VARCHAR(200) NOT NULL,
                full_name VARCHAR(255) NOT NULL,
                description TEXT,
                primary_language VARCHAR(50),
                special_label VARCHAR(50),
                language_used JSONB DEFAULT '[]'::jsonb,
                topics JSONB DEFAULT '[]'::jsonb,
                readme_summary TEXT,
                readme_md TEXT,
                star_count INT DEFAULT 0,
                likes_count INT DEFAULT 0,
                comments_count INT DEFAULT 0,
                saves_count INT DEFAULT 0,
                views_count INT DEFAULT 0,
                forks_count INT DEFAULT 0,
                pr_count INT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            try:
                cursor.execute(create_table_query)
                conn.commit()
            except Exception as exc:
                # If creating table fails (possibly due to gen_random_uuid() being unavailable),
                # rollback the failed transaction state and try creating without the DEFAULT constraint.
                logger.warning(
                    f"Failed to create table with gen_random_uuid() default: {exc}. "
                    "Attempting fallback table creation without DEFAULT constraint..."
                )
                conn.rollback()
                cursor = conn.cursor()
                fallback_table_query = """
                CREATE TABLE IF NOT EXISTS Repo (
                    repo_id UUID PRIMARY KEY,
                    github_repo_url VARCHAR(500) NOT NULL UNIQUE,
                    owner_id VARCHAR(100) NOT NULL,
                    repo_name VARCHAR(200) NOT NULL,
                    full_name VARCHAR(255) NOT NULL,
                    description TEXT,
                    primary_language VARCHAR(50),
                    special_label VARCHAR(50),
                    language_used JSONB DEFAULT '[]'::jsonb,
                    topics JSONB DEFAULT '[]'::jsonb,
                    readme_summary TEXT,
                    readme_md TEXT,
                    star_count INT DEFAULT 0,
                    likes_count INT DEFAULT 0,
                    comments_count INT DEFAULT 0,
                    saves_count INT DEFAULT 0,
                    views_count INT DEFAULT 0,
                    forks_count INT DEFAULT 0,
                    pr_count INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
                cursor.execute(fallback_table_query)
                conn.commit()

            for col_name, col_def in self._migration_columns:
                try:
                    cursor.execute(
                        f"ALTER TABLE Repo ADD COLUMN IF NOT EXISTS {col_name} {col_def};"
                    )
                    conn.commit()
                except Exception:
                    conn.rollback()

            # Ensure github_repo_url limit is increased if existing table has VARCHAR(200)
            try:
                cursor.execute(
                    "ALTER TABLE Repo ALTER COLUMN github_repo_url TYPE VARCHAR(500);"
                )
                conn.commit()
            except Exception:
                conn.rollback()

            logger.info("Database schemas verified successfully.")
        except Exception as exc:
            logger.error(f"Database initialization failed: {exc}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    # ── Repository Upsert ─────────────────────────────────────────────────────

    def upsert_repositories(self, results: list[Any]) -> int:
        """Upsert a list of EnrichmentResult objects into the Repo table.

        Returns the number of successfully upserted repositories.
        """
        return self.upsert_repositories_detailed(results).count

    def upsert_repositories_detailed(self, results: list[Any]) -> RepositoryUpsertResult:
        """Upsert repositories and return the exact successful and failed identities."""
        outcome = RepositoryUpsertResult()
        if not self.enabled:
            logger.warning("Database integration disabled; skipping upsert.")
            outcome.failed.update(
                {
                    str(getattr(result, "repo_id", "unknown/repository")): "database integration disabled"
                    for result in results
                }
            )
            return outcome

        if not results:
            logger.info("No repositories to save.")
            return outcome

        logger.info("Upserting %d repositories into PostgreSQL...", len(results))
        conn = None
        try:
            conn = self.connect()
            cursor = conn.cursor()

            upsert_query = """
            INSERT INTO Repo (
                repo_id, github_repo_url, owner_id, repo_name, full_name, description,
                primary_language, special_label, language_used, topics, readme_summary,
                readme_md, star_count, forks_count, pr_count
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                CAST(%s AS jsonb), CAST(%s AS jsonb),
                %s, %s, %s, %s, %s
            )
            ON CONFLICT (github_repo_url) DO UPDATE SET
                owner_id = EXCLUDED.owner_id,
                repo_name = EXCLUDED.repo_name,
                full_name = EXCLUDED.full_name,
                description = EXCLUDED.description,
                primary_language = EXCLUDED.primary_language,
                special_label = EXCLUDED.special_label,
                language_used = EXCLUDED.language_used,
                topics = EXCLUDED.topics,
                readme_summary = EXCLUDED.readme_summary,
                readme_md = EXCLUDED.readme_md,
                star_count = EXCLUDED.star_count,
                forks_count = EXCLUDED.forks_count,
                pr_count = EXCLUDED.pr_count,
                updated_at = CURRENT_TIMESTAMP;
            """

            for r in results:
                identity = normalize_repository_name(getattr(r, "repo_id", ""))
                identity = identity or str(getattr(r, "repo_id", "unknown/repository"))
                savepoint_created = False
                try:
                    cursor.execute("SAVEPOINT row_upsert;")
                    savepoint_created = True
                    full_name, params = self._build_upsert_params(cursor, r)
                    cursor.execute(upsert_query, params)
                    cursor.execute("RELEASE SAVEPOINT row_upsert;")
                    outcome.succeeded.append(full_name)
                except Exception as row_exc:
                    logger.error("Failed to upsert repo %s: %s", identity, row_exc)
                    outcome.failed[identity] = str(row_exc)[:500]
                    if savepoint_created:
                        try:
                            cursor.execute("ROLLBACK TO SAVEPOINT row_upsert;")
                            cursor.execute("RELEASE SAVEPOINT row_upsert;")
                        except Exception as rb_exc:
                            logger.error("Failed to rollback to savepoint: %s", rb_exc)

            conn.commit()
            logger.info(
                "Database upsert complete. %d/%d rows successfully upserted.",
                outcome.count,
                len(results),
            )
        except Exception as exc:
            logger.error(f"Database transaction failed: {exc}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

        return outcome

    @staticmethod
    def _build_upsert_params(cursor: Any, result: Any) -> tuple[str, tuple[Any, ...]]:
        """Validate and map one enrichment result inside its row savepoint."""
        warnings = list(getattr(result, "warnings", []) or [])
        if warnings:
            raise ValueError(
                "repository enrichment is incomplete: "
                + "; ".join(map(str, warnings))[:500]
            )
        payload = result.payload
        raw = result.raw_repository
        full_name = normalize_repository_name(result.repo_id)
        if not full_name:
            raise ValueError("invalid repository identity")

        github_repo_url = payload.get("html_url") or f"https://github.com/{full_name}"
        owner_id = (raw.get("owner") or {}).get("login") or full_name.partition("/")[0]
        repo_name = raw.get("name") or full_name.partition("/")[2]
        description = str(payload.get("description") or "")[:2000]
        primary_language = (
            payload.get("primary_language") or raw.get("language") or "Unknown"
        )
        languages_json = json.dumps(result.languages or {})
        topics_json = json.dumps(result.topics or [])
        canonical_readme = getattr(result.readme, "raw_markdown", "") or ""

        repo_uuid = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"github:{full_name.casefold()}")
        )
        cursor.execute(
            "SELECT repo_id, github_repo_url FROM Repo "
            "WHERE LOWER(full_name) = %s LIMIT 1;",
            (repository_identity_key(full_name),),
        )
        existing_identity = cursor.fetchone()
        if isinstance(existing_identity, (tuple, list)) and len(existing_identity) >= 2:
            repo_uuid = str(existing_identity[0])
            github_repo_url = str(existing_identity[1])

        return full_name, (
            repo_uuid,
            github_repo_url,
            owner_id,
            repo_name,
            full_name,
            description,
            primary_language,
            payload.get("special_label"),
            languages_json,
            topics_json,
            None,
            canonical_readme,
            int(payload.get("star_count") or 0),
            int(payload.get("fork_count") or 0),
            int(raw.get("pull_requests_count") or 0),
        )

    # ── Query Helpers ─────────────────────────────────────────────────────────

    def get_repo_count(self) -> int:
        """Return the total number of repos in the database."""
        if not self.enabled:
            return 0
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM Repo;")
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def get_existing_repository_names(self) -> set[str]:
        """Return all persisted repository names for pre-enrichment deduplication."""
        if not self.enabled:
            return set()
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT full_name FROM Repo WHERE full_name IS NOT NULL;")
            return {
                normalized
                for row in cursor.fetchall()
                if row and (normalized := normalize_repository_name(row[0]))
            }
        finally:
            conn.close()

    def get_repositories_by_full_names(self, full_names: list[str]) -> list[dict[str, Any]]:
        """Rebuild embedding-compatible payloads for persisted indexing retries."""
        if not self.enabled or not full_names:
            return []
        normalized = [name for value in full_names if (name := normalize_repository_name(value))]
        if not normalized:
            return []
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT full_name, github_repo_url, description, primary_language,
                       language_used, topics, readme_summary, readme_md,
                       star_count, forks_count, pr_count, updated_at
                FROM Repo
                WHERE LOWER(full_name) IN (%s);
                """ % ", ".join(["%s"] * len(normalized)),
                tuple(repository_identity_key(name) for name in normalized),
            )
            rows = cursor.fetchall()
            payloads: list[dict[str, Any]] = []
            for row in rows:
                (
                    full_name,
                    github_repo_url,
                    description,
                    primary_language,
                    language_used,
                    topics,
                    _legacy_readme_summary,
                    readme_md,
                    star_count,
                    forks_count,
                    pr_count,
                    updated_at,
                ) = row
                languages = json.loads(language_used) if isinstance(language_used, str) else (language_used or {})
                parsed_topics = json.loads(topics) if isinstance(topics, str) else (topics or [])
                canonical_readme = readme_md or None
                payloads.append(
                    {
                        "id": full_name,
                        "full_name": full_name,
                        "html_url": github_repo_url,
                        "description": description or "",
                        "primary_language": primary_language or "Unknown",
                        "languages": list(languages) if isinstance(languages, dict) else list(languages),
                        "topics": list(parsed_topics),
                        "readme": canonical_readme,
                        "readme_length": len(canonical_readme or ""),
                        "star_count": int(star_count or 0),
                        "fork_count": int(forks_count or 0),
                        "pr_count": int(pr_count or 0),
                        "updated_at": updated_at.isoformat() if hasattr(updated_at, "isoformat") else updated_at,
                    }
                )
            return payloads
        finally:
            conn.close()

    def get_repos(self, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Fetch repos from the database, ordered by star_count descending."""
        if not self.enabled:
            return []
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT repo_id, github_repo_url, full_name, description,
                       primary_language, star_count, forks_count, pr_count,
                       language_used, topics, updated_at
                FROM Repo
                ORDER BY star_count DESC NULLS LAST
                LIMIT %s OFFSET %s;
                """,
                (limit, offset),
            )
            columns = [
                "repo_id", "github_repo_url", "full_name", "description",
                "primary_language", "star_count", "forks_count", "pr_count",
                "language_used", "topics", "updated_at",
            ]
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        finally:
            conn.close()
