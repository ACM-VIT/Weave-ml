"""Read-only compatibility checks for deprecated direct-SQL adapters.

Canonical V2 product schema belongs to the backend migration repository. These
checks neither provision old tables nor import SQL into the online ML graph.
"""
from __future__ import annotations

import re
from typing import Mapping


class SchemaCompatibilityError(RuntimeError):
    """An owner-managed schema must be prepared before using a legacy adapter."""


LEGACY_REPOSITORY_COLUMNS = {
    "repo_id": "uuid", "github_repo_url": "varchar(500)", "owner_id": "varchar(100)",
    "repo_name": "varchar(200)", "full_name": "varchar(255)", "description": "text",
    "primary_language": "varchar(50)", "special_label": "varchar(50)",
    "language_used": "jsonb", "topics": "jsonb", "readme_summary": "text", "readme_md": "text",
    "star_count": "integer", "likes_count": "integer", "comments_count": "integer",
    "saves_count": "integer", "views_count": "integer", "forks_count": "integer", "pr_count": "integer",
    "created_at": "timestamp without time zone", "updated_at": "timestamp without time zone",
}
LEGACY_TRENDING_COLUMNS = {
    "repo_id": "uuid", "full_name": "varchar(255)", "name": "varchar(200)",
    "owner": "varchar(100)", "url": "varchar(500)", "description": "text",
    "star_count": "integer", "daily_stars": "integer", "fork_count": "integer",
    "created_at": "timestamp without time zone", "pushed_at": "timestamp without time zone",
    "primary_language": "varchar(50)", "topics": "jsonb", "readme": "text",
    "default_branch": "varchar(100)", "first_seen_at": "timestamp without time zone",
    "last_seen_at": "timestamp without time zone", "trending_rank": "integer",
}
LEGACY_TRENDING_METADATA_COLUMNS = {
    "key": "varchar(100)", "value": "text", "updated_at": "timestamp without time zone",
}


def _compatible(actual: str, required: str) -> bool:
    width = re.fullmatch(r"varchar\((\d+)\)", required)
    if width:
        if actual in {"text", "character varying"}:
            return True
        actual_width = re.fullmatch(r"character varying\((\d+)\)", actual)
        return actual_width is not None and int(actual_width[1]) >= int(width[1])
    if required == "text":
        return actual in {"text", "character varying"}
    if required == "timestamp without time zone":
        return re.fullmatch(r"timestamp(?:\([0-6]\))? without time zone", actual) is not None
    if required == "integer":
        return actual in {"integer", "bigint"}
    return actual == required


def validate_table_schema(cursor, table: str, required: Mapping[str, str]) -> None:
    # Table identity is a bound regclass value, never interpolated SQL. pg_attribute
    # inspects metadata only, without reading product rows or changing objects.
    cursor.execute(
        "SELECT attname,pg_catalog.format_type(atttypid,atttypmod) "
        "FROM pg_catalog.pg_attribute "
        "WHERE attrelid=pg_catalog.to_regclass(%s) AND attnum>0 AND NOT attisdropped",
        (table,),
    )
    actual = dict(cursor.fetchall())
    missing = sorted(set(required) - actual.keys())
    mismatches = sorted(name for name in required.keys() & actual.keys()
                        if not _compatible(actual[name], required[name]))
    if missing or mismatches:
        details = []
        if missing:
            details.append("missing columns: " + ", ".join(missing))
        if mismatches:
            details.append("incompatible column types: " + ", ".join(mismatches))
        raise SchemaCompatibilityError(
            f"Legacy table {table!r} is not ready ({'; '.join(details)}). "
            "Apply owner-managed migrations or use canonical backend V2 ingestion; ML will not change the schema."
        )
