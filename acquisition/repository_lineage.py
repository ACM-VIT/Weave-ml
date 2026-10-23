"""Parse observed immutable GitHub identity without inferring fork ancestry.

REST ``source`` is the ultimate fork-network root, whereas ``parent`` is only
the immediate parent. GraphQL ancestry establishes a root only when a supplied
chain reaches an explicit isFork=false node. Partial/private ancestry is unknown.
No network I/O, handle matching, catalogue writes or canonical UUID generation.
"""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class RepositoryLineage:
    github_id: str
    owner_github_id: str
    is_fork: bool | None
    fork_parent_github_id: str | None
    fork_family_github_id: str | None
    observed_at: str
    source: str


def _id(value: Any) -> str:
    if type(value) is int:
        value = str(value)
    if (not isinstance(value, str) or not 1 <= len(value) <= 19 or not value.isascii()
            or not value.isdecimal() or value[0] == '0' or int(value) > 2**63 - 1):
        raise ValueError('GitHub identity must be a positive signed-bigint decimal')
    return value


def _observed(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('Lineage requires an explicit timezone-aware observation time')
    return value.astimezone(timezone.utc).isoformat()


def _fork(value: Any) -> bool | None:
    if value is not None and type(value) is not bool:
        raise ValueError('Fork state must be explicit boolean or unknown')
    return value


def lineage_from_rest(repository: Mapping[str, Any], *, observed_at: datetime) -> RepositoryLineage:
    github_id = _id(repository.get('id'))
    owner = repository.get('owner')
    if not isinstance(owner, Mapping):
        raise ValueError('Repository owner identity is required')
    owner_id = _id(owner.get('id'))
    fork = _fork(repository.get('fork'))
    parent = repository.get('parent')
    source = repository.get('source')
    parent_id = _id(parent.get('id')) if isinstance(parent, Mapping) else None
    family_id = _id(source.get('id')) if isinstance(source, Mapping) else None
    if parent is not None and not isinstance(parent, Mapping) or source is not None and not isinstance(source, Mapping):
        raise ValueError('Malformed repository lineage objects')
    if fork is False:
        if parent_id is not None or family_id is not None:
            raise ValueError('Nonfork contradicts supplied fork ancestry')
        family_id = github_id
    elif fork is True:
        if github_id in {parent_id, family_id}:
            raise ValueError('Fork ancestry cannot refer to the fork itself')
    elif parent_id is not None or family_id is not None:
        raise ValueError('Ancestry without explicit fork state is ambiguous')
    return RepositoryLineage(github_id, owner_id, fork, parent_id, family_id,
                             _observed(observed_at), 'github-rest-repository')


def lineage_from_graphql(repository: Mapping[str, Any], *, observed_at: datetime) -> RepositoryLineage:
    github_id = _id(repository.get('databaseId'))
    owner = repository.get('owner')
    if not isinstance(owner, Mapping):
        raise ValueError('Repository owner identity is required')
    owner_id = _id(owner.get('databaseId'))
    fork = _fork(repository.get('isFork'))
    family_id = None
    parent_id = None
    current = repository
    visited: set[str] = set()
    for depth in range(17):
        current_id = _id(current.get('databaseId'))
        if current_id in visited:
            raise ValueError('Cyclic GitHub fork ancestry')
        visited.add(current_id)
        state = _fork(current.get('isFork'))
        parent = current.get('parent')
        if state is False:
            if parent is not None:
                raise ValueError('Nonfork contradicts supplied fork parent')
            family_id = current_id
            break
        if state is None:
            if parent is not None:
                raise ValueError('Parent without explicit fork state is ambiguous')
            break
        if parent is None:
            break
        if not isinstance(parent, Mapping):
            raise ValueError('Malformed GraphQL fork parent')
        if depth == 0:
            parent_id = _id(parent.get('databaseId'))
        if depth == 16:
            raise ValueError('GitHub lineage exceeds the bounded ancestry depth')
        current = parent
    return RepositoryLineage(github_id, owner_id, fork, parent_id, family_id,
                             _observed(observed_at), 'github-graphql-repository')
