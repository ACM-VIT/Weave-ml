"""Pytest configuration and fixtures for trending service tests."""

import os
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def anyio_backend():
    """The feedback consumer is asyncio-native; do not duplicate it under Trio."""
    return "asyncio"


# ── Test Configuration ─────────────────────────────────────────────────────────────

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests that require external services"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (network calls, heavy computation)"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"
    )


# ── Environment Setup ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables before each test."""
    # Set required environment variables for testing
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")
    monkeypatch.setenv("TRENDING_REPO_LIMIT", "30")
    monkeypatch.setenv("TRENDING_REFRESH_HOURS", "24")


@pytest.fixture
def health_probe_cache(monkeypatch):
    """Isolate router-only health observations; requests explicitly refresh them."""
    import threading
    from api import v2
    from utils import runtime_lifecycle

    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("EMBEDDING_WARMUP_ON_STARTUP", "false")
    monkeypatch.setenv("V2_HEALTH_REFRESH_SECONDS", "15")
    monkeypatch.setenv("V2_HEALTH_MAX_AGE_SECONDS", "45")
    monkeypatch.setenv("V2_HEALTH_TIMEOUT_SECONDS", "5")
    monkeypatch.setattr(v2, "_runtime_retirement", None)
    monkeypatch.setattr(runtime_lifecycle, "_retiring", threading.Event())
    monkeypatch.setattr(v2, "embedding_runtime_status", lambda: {"embedding_runtime_ready": True})
    monkeypatch.setattr(v2, "service_runtime_status", lambda: {})
    monkeypatch.setattr(v2, "recommendation_runtime_status", lambda: {})
    if v2.health_diagnostics.cache_info().currsize:
        v2.health_diagnostics().close()
    v2.health_diagnostics.cache_clear()
    diagnostics = v2.health_diagnostics()
    try:
        yield diagnostics
    finally:
        diagnostics.close()
        v2.health_diagnostics.cache_clear()


# ── Mock Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_database_connector():
    """Create a mock database connector."""
    connector = MagicMock()
    connector.enabled = True
    connector.connect.return_value = MagicMock()
    return connector


# ── Test Data Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_normalized_repo():
    """Sample normalized repository dictionary."""
    return {
        "full_name": "test-owner/test-repo",
        "name": "test-repo",
        "owner": "test-owner",
        "url": "https://github.com/test-owner/test-repo",
        "description": "A test repository",
        "star_count": 1000,
        "fork_count": 50,
        "created_at": "2024-01-01T00:00:00Z",
        "pushed_at": "2024-06-01T00:00:00Z",
        "primary_language": "Python",
        "topics": ["machine-learning", "data-science"],
        "readme": "# Test Repository\n\nThis is a test README file.",
        "default_branch": "main"
    }


# ── Database Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_database_url():
    """Provide test database URL."""
    return os.getenv("TEST_DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")


@pytest.fixture
def clean_database(test_database_url):
    """Fixture to provide a clean database for each test."""
    # This would typically set up and tear down a test database
    # For now, we'll use a mock connector
    yield test_database_url
    # Cleanup would happen here


# ── Performance Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def performance_thresholds():
    """Define performance thresholds for benchmarks."""
    return {
        "fetch_latency_seconds": 30,
        "db_upsert_seconds": 120,
        "total_cycle_seconds": 300,
        "memory_mb": 500
    }
