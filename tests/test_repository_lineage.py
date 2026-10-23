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


def test_observed_metadata_survives_strict_api_and_atomic_vector_payload():
    from api.contracts import RepositorySource
    from embedding.repository_embedding import RepositoryEmbeddingConfig, build_vector_payload
    from embedding.vector_contract import validate_diversity_metadata
    source = RepositorySource(full_name='owner/repository',github_id='30',owner_github_id='7',
        is_fork=True,fork_parent_github_id='20',fork_family_github_id='10',
        lineage_source='github-graphql-repository',lineage_observed_at=NOW)
    payload = build_vector_payload(source.model_dump(),repo_id='00000000-0000-4000-8000-000000000001',
        final_embedding=[1.0]+[0.0]*383,readme_chunks=0,source_hash='source',config=RepositoryEmbeddingConfig())
    assert payload['owner_github_id']=='7' and payload['fork_family_github_id']=='10'
    assert payload['lineage_observed_at']==NOW.isoformat()
    assert 'serving_eligibility_version' not in payload
    assert all(value is None for value in validate_diversity_metadata({}).values())
    for patch in [{'fork_family_github_id':'30'},{'lineage_source':None},{'is_fork':None},
                  {'owner_github_id':str(2**63)},{'lineage_observed_at':'2026-10-23T00:00:00'}]:
        with pytest.raises(ValueError): RepositorySource.model_validate({**source.model_dump(),**patch})


def test_lineage_acquisition_is_opt_in_and_uses_existing_graphql_queries(monkeypatch):
    from acquisition.graphql_queries import build_batch_metadata_query, repository_metadata_query
    from acquisition.github_graphql_client import GitHubGraphQLClient
    from acquisition.repository_enricher import RepositoryEnricher
    from acquisition.backend_client import repository_upsert_record
    base = build_batch_metadata_query([('owner','repository')],projection='delivery')
    assert 'isFork' not in base and 'isFork' not in repository_metadata_query()
    rich = build_batch_metadata_query([('owner','repository')],projection='delivery',include_lineage=True)
    assert rich.count('parent {') == 4 and 'isFork' in repository_metadata_query(include_lineage=True)
    monkeypatch.delenv('ACQUISITION_LINEAGE_ENABLED',raising=False)
    assert not GitHubGraphQLClient._lineage_enabled()
    monkeypatch.setenv('ACQUISITION_LINEAGE_ENABLED','true')
    assert GitHubGraphQLClient._lineage_enabled()
    data = {'id':'node30','databaseId':30,'nameWithOwner':'owner/repository','name':'repository',
        'description':'Actual fixture description','url':'https://github.com/owner/repository',
        'owner':{'databaseId':7,'login':'owner'},'isFork':True,'parent':{'databaseId':10,'isFork':False},
        'defaultBranchRef':{'name':'main','target':{'history':{'nodes':[]}}}}
    result = RepositoryEnricher(graphql_client=object())._process_graphql_data(data,None,None,readme_text='# Fixture')
    record = repository_upsert_record(result)
    assert record['owner_github_id']=='7' and record['is_fork'] is True and record['fork_family_github_id']=='10'
    assert record['lineage_source']=='github-graphql-repository'
