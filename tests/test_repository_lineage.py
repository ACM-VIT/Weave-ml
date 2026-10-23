from datetime import datetime, timezone
import pytest
from acquisition.repository_lineage import lineage_from_graphql, lineage_from_rest

NOW = datetime(2026, 10, 23, tzinfo=timezone.utc)


def test_rest_uses_ultimate_source_not_immediate_parent_or_mutable_handles():
    raw = {'id': 30, 'owner': {'id': 7, 'login': 'old'}, 'fork': True,
           'parent': {'id': 20}, 'source': {'id': 10}, 'full_name': 'old/name'}
    parsed = lineage_from_rest(raw, observed_at=NOW)
    assert (parsed.github_id, parsed.owner_github_id, parsed.fork_parent_github_id,
            parsed.fork_family_github_id) == ('30', '7', '20', '10')
    assert lineage_from_rest({**raw, 'owner': {'id': 7, 'login': 'renamed'}, 'full_name': 'new/name'}, observed_at=NOW) == parsed
    assert parsed.observed_at == '2026-10-23T00:00:00+00:00'


def test_rest_partial_and_unknown_lineage_stays_unknown():
    raw = {'id': 30, 'owner': {'id': 7}, 'fork': True, 'parent': {'id': 20}}
    assert lineage_from_rest(raw, observed_at=NOW).fork_family_github_id is None
    assert lineage_from_rest({'id': 30, 'owner': {'id': 7}}, observed_at=NOW).is_fork is None
    assert lineage_from_rest({'id': 30, 'owner': {'id': 7}, 'fork': False}, observed_at=NOW).fork_family_github_id == '30'


def test_graphql_chain_must_reach_an_explicit_nonfork():
    root = {'databaseId': 10, 'isFork': False}
    parent = {'databaseId': 20, 'isFork': True, 'parent': root}
    fork = {'databaseId': 30, 'owner': {'databaseId': 7}, 'isFork': True, 'parent': parent}
    parsed = lineage_from_graphql(fork, observed_at=NOW)
    assert parsed.fork_parent_github_id == '20'
    assert parsed.fork_family_github_id == '10'
    assert lineage_from_graphql({**fork, 'parent': {'databaseId': 20, 'isFork': True}}, observed_at=NOW).fork_family_github_id is None


@pytest.mark.parametrize('bad', [True, 1.5, 'handle', '001', '0', '-1', '9'*20, '١', None])
def test_numeric_identity_never_guessed_or_coerced_from_invalid_values(bad):
    with pytest.raises(ValueError):
        lineage_from_rest({'id': bad, 'owner': {'id': 7}}, observed_at=NOW)


def test_contradictions_cycles_and_missing_timestamps_fail():
    with pytest.raises(ValueError, match='contradicts'):
        lineage_from_rest({'id': 30, 'owner': {'id': 7}, 'fork': False, 'source': {'id': 10}}, observed_at=NOW)
    with pytest.raises(ValueError, match='itself'):
        lineage_from_rest({'id': 30, 'owner': {'id': 7}, 'fork': True, 'source': {'id': 30}}, observed_at=NOW)
    cyclic = {'databaseId': 30, 'owner': {'databaseId': 7}, 'isFork': True}
    cyclic['parent'] = cyclic
    with pytest.raises(ValueError, match='Cyclic'):
        lineage_from_graphql(cyclic, observed_at=NOW)
    with pytest.raises(ValueError, match='timezone'):
        lineage_from_rest({'id': 30, 'owner': {'id': 7}}, observed_at=NOW.replace(tzinfo=None))
