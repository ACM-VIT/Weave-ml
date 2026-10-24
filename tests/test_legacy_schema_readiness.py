"""Legacy initialization must inspect schema without repairing shared tables."""
from contextlib import contextmanager
import os
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import urlsplit

import pg8000.dbapi
import pytest

from database.connector import PostgreSQLConnector
from database.schema_readiness import SchemaCompatibilityError, _compatible
from trending.storage import TrendingStorage
import trending.config as trending_config


def test_repository_missing_schema_fails_without_startup_ddl():
    connector = PostgreSQLConnector(database_url="postgresql://fixture@localhost/fixture")
    connection = MagicMock()
    cursor = connection.cursor.return_value
    cursor.fetchall.return_value = []
    connector.connect = lambda: connection
    with pytest.raises(SchemaCompatibilityError, match="missing columns.*owner-managed migrations"):
        connector.init_db()
    assert cursor.execute.call_count == 1
    assert cursor.execute.call_args.args[0].startswith("SELECT ")
    assert cursor.execute.call_args.args[1] == ("repo",)
    connection.rollback.assert_called_once()
    connection.commit.assert_not_called()
    connection.close.assert_called_once()


class ReadOnlyConnection:
    def __init__(self, connection):
        self.connection = connection
        self.statements = []
        self.closed = 0

    def cursor(self):
        cursor = self.connection.cursor()
        owner = self

        class Cursor:
            def execute(self, sql, parameters=None):
                owner.statements.append(sql)
                assert sql.lstrip().upper().startswith("SELECT "), "Schema readiness attempted a write or DDL"
                return cursor.execute(sql, parameters)

            def fetchall(self):
                return cursor.fetchall()

        return Cursor()

    def commit(self):
        self.connection.commit()

    def rollback(self):
        self.connection.rollback()

    def close(self):
        # The adapter must close its lease; the fixture retains the physical
        # connection only to inspect its temporary schema and clean it afterward.
        self.closed += 1


@contextmanager
def disposable_database():
    target = urlsplit(os.environ["LEGACY_SCHEMA_DATABASE_URL"])
    assert target.scheme in {"postgres", "postgresql"}
    assert target.hostname in {"localhost", "127.0.0.1", "::1"} and target.path == "/weave_scroll"
    assert not target.query and not target.fragment and os.environ.get("APP_ENV") != "production"
    connector = PostgreSQLConnector(database_url=os.environ["LEGACY_SCHEMA_DATABASE_URL"])
    connection = connector.connect()
    try:
        yield connection
    finally:
        connection.close()  # Session-local fixture tables disappear here.


integration = pytest.mark.skipif(os.environ.get("RUN_LEGACY_SCHEMA_READINESS") != "1",
                                 reason="Requires explicit disposable PostgreSQL opt-in")


@pytest.mark.integration
@integration
@pytest.mark.parametrize("fault", [None, "compatible-owner-types", "missing-counter", "narrow-url", "wrong-counter-type"])
def test_real_repository_readiness_rejects_drift_in_read_only_transactions(fault):
    with disposable_database() as connection:
        cursor = connection.cursor()
        # Independent historical table shape; no product table is created or
        # altered. This TEMP table shadows an identically named public object.
        cursor.execute("""CREATE TEMP TABLE repo (
            repo_id uuid,github_repo_url varchar(500),owner_id varchar(100),repo_name varchar(200),
            full_name varchar(255),description text,primary_language varchar(50),special_label varchar(50),
            language_used jsonb,topics jsonb,readme_summary text,readme_md text,star_count integer,
            likes_count integer,comments_count integer,saves_count integer,views_count integer,
            forks_count integer,pr_count integer,created_at timestamp,updated_at timestamp)""")
        if fault == "compatible-owner-types":
            cursor.execute("ALTER TABLE pg_temp.repo ALTER COLUMN github_repo_url TYPE varchar(1000)")
            cursor.execute("ALTER TABLE pg_temp.repo ALTER COLUMN owner_id TYPE varchar")
            cursor.execute("ALTER TABLE pg_temp.repo ALTER COLUMN repo_name TYPE text")
            cursor.execute("ALTER TABLE pg_temp.repo ALTER COLUMN description TYPE varchar")
            cursor.execute("ALTER TABLE pg_temp.repo ALTER COLUMN created_at TYPE timestamp(6)")
            cursor.execute("ALTER TABLE pg_temp.repo ALTER COLUMN updated_at TYPE timestamp(3)")
            cursor.execute("ALTER TABLE pg_temp.repo ALTER COLUMN likes_count TYPE bigint")
            cursor.execute("ALTER TABLE pg_temp.repo ADD COLUMN owner_metadata jsonb")
        elif fault == "missing-counter":
            cursor.execute("ALTER TABLE pg_temp.repo DROP COLUMN likes_count")
        elif fault == "narrow-url":
            cursor.execute("ALTER TABLE pg_temp.repo ALTER COLUMN github_repo_url TYPE varchar(200)")
        elif fault == "wrong-counter-type":
            cursor.execute("ALTER TABLE pg_temp.repo ALTER COLUMN saves_count TYPE text")
        cursor.execute("SET default_transaction_read_only=on")
        connection.commit()
        lease = ReadOnlyConnection(connection)
        connector = PostgreSQLConnector(database_url=os.environ["LEGACY_SCHEMA_DATABASE_URL"])
        connector.connect = lambda: lease
        if fault not in {None, "compatible-owner-types"}:
            with pytest.raises(SchemaCompatibilityError, match="owner-managed migrations"):
                connector.init_db()
        else:
            connector.init_db()
        assert lease.closed == 1 and len(lease.statements) == 1
        cursor.execute("SHOW transaction_read_only")
        assert cursor.fetchone()[0] == "on"


@pytest.mark.integration
@integration
@pytest.mark.parametrize("shape", ["historical", "compatible-owner-types", "missing-metadata"])
def test_real_trending_readiness_reads_both_tables_without_ddl(monkeypatch, shape):
    with disposable_database() as connection:
        cursor = connection.cursor()
        cursor.execute("""CREATE TEMP TABLE ready_trending (
            repo_id uuid,full_name varchar(255),name varchar(200),owner varchar(100),url varchar(500),
            description text,star_count integer,daily_stars integer,fork_count integer,created_at timestamp,
            pushed_at timestamp,primary_language varchar(50),topics jsonb,readme text,default_branch varchar(100),
            first_seen_at timestamp,last_seen_at timestamp,trending_rank integer)""")
        if shape != "missing-metadata":
            cursor.execute("CREATE TEMP TABLE ready_metadata (key varchar(100),value text,updated_at timestamp)")
        if shape == "compatible-owner-types":
            cursor.execute("ALTER TABLE pg_temp.ready_trending ALTER COLUMN url TYPE text")
            cursor.execute("ALTER TABLE pg_temp.ready_trending ALTER COLUMN owner TYPE varchar")
            cursor.execute("ALTER TABLE pg_temp.ready_trending ALTER COLUMN name TYPE varchar(400)")
            cursor.execute("ALTER TABLE pg_temp.ready_trending ALTER COLUMN description TYPE varchar")
            cursor.execute("ALTER TABLE pg_temp.ready_trending ALTER COLUMN star_count TYPE bigint")
            cursor.execute("ALTER TABLE pg_temp.ready_trending ALTER COLUMN created_at TYPE timestamp(6)")
            cursor.execute("ALTER TABLE pg_temp.ready_trending ADD COLUMN owner_metadata jsonb")
            cursor.execute("ALTER TABLE pg_temp.ready_metadata ALTER COLUMN updated_at TYPE timestamp(6)")
        cursor.execute("SET default_transaction_read_only=on")
        connection.commit()
        monkeypatch.setattr(trending_config, "TRENDING_TABLE_NAME", "pg_temp.ready_trending")
        monkeypatch.setattr(trending_config, "TRENDING_METADATA_TABLE_NAME", "pg_temp.ready_metadata")
        lease = ReadOnlyConnection(connection)
        storage = TrendingStorage(database_url=os.environ["LEGACY_SCHEMA_DATABASE_URL"])
        storage.enabled = True
        storage.connector = SimpleNamespace(connect=lambda: lease)
        if shape == "missing-metadata":
            with pytest.raises(SchemaCompatibilityError, match="missing columns"):
                storage.init_schema()
        else:
            storage.init_schema()
        assert lease.closed == 1 and len(lease.statements) == 2


@pytest.mark.unit
@pytest.mark.parametrize("actual,required,allowed", [
    ("character varying(1000)", "varchar(500)", True),
    ("character varying", "varchar(500)", True),
    ("text", "varchar(500)", True),
    ("character varying(200)", "varchar(500)", False),
    ("character varying", "text", True),
    ("text", "text", True),
    ("character varying(500)", "text", False),
    ("bigint", "integer", True),
    ("smallint", "integer", False),
    ("text", "integer", False),
    ("timestamp(6) without time zone", "timestamp without time zone", True),
    ("timestamp(3) without time zone", "timestamp without time zone", True),
    ("timestamp without time zone", "timestamp without time zone", True),
    ("timestamp(6) with time zone", "timestamp without time zone", False),
])
def test_independent_postgresql_type_declarations(actual, required, allowed):
    assert _compatible(actual, required) is allowed
