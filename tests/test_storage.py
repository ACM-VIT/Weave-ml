"""Unit tests for trending/storage.py module."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import pytest

from trending.config import DATABASE_URL
from trending.storage import TrendingStorage


@pytest.mark.unit
class TestTrendingStorageInit:
    """Test TrendingStorage initialization."""

    def test_init_with_default_database_url(self, monkeypatch):
        """Test initialization with default database URL."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")
        # Reload config to pick up the environment variable
        import importlib
        import trending.config
        importlib.reload(trending.config)
        storage = TrendingStorage()
        # db_url will be DATABASE_URL from config
        assert storage.db_url == trending.config.DATABASE_URL
        assert storage.connector is not None

    def test_init_with_custom_database_url(self):
        """Test initialization with custom database URL."""
        custom_url = "postgresql://test:test@localhost:5432/test_db"
        storage = TrendingStorage(database_url=custom_url)
        assert storage.db_url == custom_url

    def test_init_without_database_url(self, monkeypatch):
        """Test initialization when DATABASE_URL is not set."""
        monkeypatch.delenv("DATABASE_URL", raising=False)
        # Reload config to pick up the change
        import importlib
        import trending.config
        importlib.reload(trending.config)
        storage = TrendingStorage()
        assert storage.enabled is False


@pytest.mark.unit
class TestParseTimestamp:
    """Test _parse_timestamp static method."""

    def test_parse_valid_iso_timestamp(self):
        """Test parsing valid ISO timestamp."""
        timestamp = "2024-06-20T12:00:00Z"
        result = TrendingStorage._parse_timestamp(timestamp)
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 20

    def test_parse_timestamp_with_timezone(self):
        """Test parsing timestamp with timezone offset."""
        timestamp = "2024-06-20T12:00:00+05:30"
        result = TrendingStorage._parse_timestamp(timestamp)
        assert isinstance(result, datetime)

    def test_parse_none_timestamp(self):
        """Test parsing None timestamp."""
        result = TrendingStorage._parse_timestamp(None)
        assert result is None

    def test_parse_empty_string_timestamp(self):
        """Test parsing empty string timestamp."""
        result = TrendingStorage._parse_timestamp("")
        assert result is None

    def test_parse_invalid_timestamp(self):
        """Test parsing invalid timestamp."""
        result = TrendingStorage._parse_timestamp("invalid-timestamp")
        assert result is None


@pytest.mark.unit
class TestInitSchema:
    """Test init_schema method."""

    def test_init_schema_when_disabled(self):
        """Test schema initialization when storage is disabled."""
        storage = TrendingStorage()
        storage.enabled = False
        # Should not raise error, just log warning
        storage.init_schema()

    def test_init_schema_creates_tables(self, mock_database_connector):
        """Test that schema initialization creates tables."""
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_database_connector.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        storage.init_schema()
        
        # Verify table creation queries were executed
        assert mock_cursor.execute.call_count >= 2
        assert mock_conn.commit.call_count >= 2

    def test_init_schema_handles_index_creation_failure(self, mock_database_connector):
        """Test that index creation failures don't break schema init."""
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_database_connector.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Make index creation fail
        execute_call_count = [0]
        def execute_side_effect(query, params=None):
            execute_call_count[0] += 1
            if execute_call_count[0] > 2:  # After table creation
                raise Exception("Index creation failed")
        mock_cursor.execute.side_effect = execute_side_effect
        
        # Should not raise error
        storage.init_schema()


@pytest.mark.unit
class TestUpsertRepositories:
    """Test upsert_repositories method."""

    def test_upsert_when_disabled(self):
        """Test upsert when storage is disabled."""
        storage = TrendingStorage()
        storage.enabled = False
        
        repos = [{"full_name": "test/repo", "name": "repo"}]
        result = storage.upsert_repositories(repos)
        
        assert result == 0

    def test_upsert_empty_repository_list(self, mock_database_connector):
        """Test upsert with empty repository list."""
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        result = storage.upsert_repositories([])
        
        assert result == 0

    def test_upsert_success(self, mock_database_connector, sample_normalized_repo):
        """Test successful repository upsert."""
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_database_connector.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None  # New repository
        
        repos = [sample_normalized_repo]
        result = storage.upsert_repositories(repos)
        
        assert result == 1
        assert mock_cursor.execute.called
        assert mock_conn.commit.called

    def test_upsert_preserves_full_readme_without_legacy_length_cap(
        self, mock_database_connector, sample_normalized_repo
    ):
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_database_connector.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        canonical_readme = "# README\n\n" + "x" * 20_000
        repository = {**sample_normalized_repo, "readme": canonical_readme}

        assert storage.upsert_repositories([repository]) == 1

        insert_call = next(
            call
            for call in mock_cursor.execute.call_args_list
            if "INSERT INTO" in str(call.args[0])
        )
        assert insert_call.args[1][13] == canonical_readme

    def test_upsert_updates_existing_repository(self, mock_database_connector, sample_normalized_repo):
        """Test upsert updates existing repository."""
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_database_connector.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        # Simulate existing repository
        mock_cursor.fetchone.return_value = (datetime.now(timezone.utc),)
        
        repos = [sample_normalized_repo]
        result = storage.upsert_repositories(repos)
        
        assert result == 1

    def test_upsert_handles_individual_failures(self, mock_database_connector, sample_normalized_repo):
        """Test upsert stops after first repository-level DB failure."""
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_database_connector.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        
        # Make first repo succeed, second repo fail on the upsert query
        repo_count = [0]
        def execute_side_effect(query, params=None):
            # Only fail on the INSERT/UPDATE query for the second repo
            if "INSERT INTO" in query or "ON CONFLICT" in query:
                repo_count[0] += 1
                if repo_count[0] == 2:  # Second repository's upsert fails
                    raise Exception("Upsert failed")
        mock_cursor.execute.side_effect = execute_side_effect
        
        repos = [sample_normalized_repo, sample_normalized_repo]
        result = storage.upsert_repositories(repos)
        
        # Should succeed for first repo, stop after second fails
        assert result == 1

    def test_upsert_updates_metadata(self, mock_database_connector, sample_normalized_repo):
        """Test that upsert updates metadata."""
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_database_connector.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        
        repos = [sample_normalized_repo]
        storage.upsert_repositories(repos)
        
        # Verify metadata update was called
        assert mock_cursor.execute.call_count >= 2  # At least upsert + metadata updates


@pytest.mark.unit
class TestGetLastRefreshTime:
    """Test get_last_refresh_time method."""

    def test_get_last_refresh_when_disabled(self):
        """Test get last refresh when storage is disabled."""
        storage = TrendingStorage()
        storage.enabled = False
        
        result = storage.get_last_refresh_time()
        
        assert result is None

    def test_get_last_refresh_success(self, mock_database_connector):
        """Test successful last refresh time retrieval."""
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_database_connector.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = ("2024-06-20T12:00:00",)
        
        result = storage.get_last_refresh_time()
        
        assert isinstance(result, datetime)
        assert result.year == 2024

    def test_get_last_refresh_not_found(self, mock_database_connector):
        """Test when last refresh time is not found."""
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_database_connector.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        
        result = storage.get_last_refresh_time()
        
        assert result is None

    def test_get_last_refresh_error_handling(self, mock_database_connector):
        """Test error handling in get last refresh time."""
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        mock_database_connector.connect.side_effect = Exception("Database error")
        
        result = storage.get_last_refresh_time()
        
        assert result is None


@pytest.mark.unit
class TestGetTrendingRepositories:
    """Test get_trending_repositories method."""

    def test_get_repositories_when_disabled(self):
        """Test get repositories when storage is disabled."""
        storage = TrendingStorage()
        storage.enabled = False
        
        result = storage.get_trending_repositories()
        
        assert result == []

    def test_get_repositories_success(self, mock_database_connector):
        """Test successful repository retrieval."""
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_database_connector.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            ("test/repo", "repo", "test", "https://github.com/test/repo", "Test",
             100, 100, 10, "2024-01-01", "2024-06-01", "Python", '["ml", "ai"]',
             "# README", "main", "2024-01-01", "2024-06-01", 1)
        ]
        
        result = storage.get_trending_repositories()
        
        assert len(result) == 1
        assert result[0]["full_name"] == "test/repo"
        assert result[0]["topics"] == ["ml", "ai"]

    def test_get_repositories_with_limit_and_offset(self, mock_database_connector):
        """Test repository retrieval with limit and offset."""
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_database_connector.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        
        storage.get_trending_repositories(limit=10, offset=5)
        
        # Verify limit and offset were passed
        call_args = mock_cursor.execute.call_args
        assert call_args[0][1] == (10, 5)


@pytest.mark.unit
class TestCleanupOldRepositories:
    """Test cleanup_old_repositories method."""

    def test_cleanup_when_disabled(self):
        """Test cleanup when storage is disabled."""
        storage = TrendingStorage()
        storage.enabled = False
        
        result = storage.cleanup_old_repositories()
        
        assert result == 0

    def test_cleanup_success(self, mock_database_connector):
        """Test successful cleanup of old repositories."""
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_database_connector.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 5
        
        result = storage.cleanup_old_repositories(days_to_keep=30)
        
        assert result == 5
        assert mock_cursor.execute.called
        assert mock_conn.commit.called

    def test_cleanup_error_handling(self, mock_database_connector):
        """Test error handling in cleanup."""
        storage = TrendingStorage()
        storage.connector = mock_database_connector
        storage.enabled = True
        
        mock_database_connector.connect.side_effect = Exception("Database error")
        
        result = storage.cleanup_old_repositories()
        
        assert result == 0
