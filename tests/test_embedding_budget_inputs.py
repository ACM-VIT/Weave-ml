"""Frozen supplied-corpus acceptance; no model/policy/production access."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.embedding_budget import load_budget_fixture, budget_retrieval_report


def fixture():
    return {"fixture_version":"budget-fixture-v1","provenance":"synthetic test judgments, not real quality evidence",
        "repositories":[{"repo_id":f"00000000-0000-4000-8000-{index:012d}","full_name":f"fixture/{index}",
            "description":"search","readme":"# Original heading\nLong document text.","topics":["search"],"languages":["Python"]}
            for index in (1,2)],
        "queries":[{"query":"private query text must not be copied to the retrieval report",
                    "relevant_repo_ids":["00000000-0000-4000-8000-000000000001"]}]}


def write(tmp_path,value):
    path=tmp_path/'corpus.json';path.write_text(json.dumps(value));return path


def test_frozen_corpus_preserves_canonical_content_and_all_tower_metadata(tmp_path):
    value=fixture();value['repositories'][0]['readme']='  # 标题\n\nCanonical text.\n'
    loaded=load_budget_fixture(write(tmp_path,value))
    assert loaded.cases[0]['readme']==value['repositories'][0]['readme']
    assert loaded.cases[0]['repository']==value['repositories'][0]
    assert loaded.cases[0]['name']==value['repositories'][0]['repo_id']
    assert len(loaded.sha256)==64 and loaded.label_provenance==value['provenance']


@pytest.mark.parametrize('mutation',['duplicate_repo','duplicate_query','unknown_label','no_label','extra_field',
    'bad_uuid','oversized_readme','bad_topics','duplicate_topics','unicode_surrogate','too_many_repos'])
def test_invalid_or_ambiguous_frozen_inputs_are_rejected(tmp_path,mutation):
    value=fixture()
    if mutation=='duplicate_repo':value['repositories'].append(copy.deepcopy(value['repositories'][0]))
    elif mutation=='duplicate_query':value['queries'].append(copy.deepcopy(value['queries'][0]))
    elif mutation=='unknown_label':value['queries'][0]['relevant_repo_ids']=['00000000-0000-4000-8000-000000000003']
    elif mutation=='no_label':value['queries'][0]['relevant_repo_ids']=[]
    elif mutation=='extra_field':value['repositories'][0]['secret']='not a permitted fixture field'
    elif mutation=='bad_uuid':value['repositories'][0]['repo_id']='owner/name'
    elif mutation=='oversized_readme':value['repositories'][0]['readme']='x'*200001
    elif mutation=='bad_topics':value['repositories'][0]['topics']=['search',False]
    elif mutation=='duplicate_topics':value['repositories'][0]['topics']=['search','search']
    elif mutation=='unicode_surrogate':value['repositories'][0]['readme']='\ud800'
    else:value['repositories']*=33
    with pytest.raises(ValueError):load_budget_fixture(write(tmp_path,value))


@pytest.mark.parametrize('raw',['{"fixture_version":"a","fixture_version":"b"}','{"bad":NaN}','[]'])
def test_duplicate_keys_nonfinite_json_and_wrong_root_rejected(tmp_path,raw):
    path=tmp_path/'invalid.json';path.write_text(raw)
    with pytest.raises(ValueError):load_budget_fixture(path)


def test_nonregular_symlink_and_preallocation_bounds(tmp_path):
    path=write(tmp_path,fixture());link=tmp_path/'link';link.symlink_to(path)
    with pytest.raises((ValueError,OSError)):load_budget_fixture(link)
    with pytest.raises(ValueError):load_budget_fixture(tmp_path)
    huge=tmp_path/'huge.json'
    with huge.open('wb') as stream:stream.truncate(32*1024*1024+1)
    with pytest.raises(ValueError):load_budget_fixture(huge)


def test_actual_retrieval_metric_changes_when_budget_vector_loses_relevant_direction(tmp_path):
    loaded=load_budget_fixture(write(tmp_path,fixture()))
    x=np.zeros(384,dtype=np.float32);x[0]=1
    y=np.zeros(384,dtype=np.float32);y[1]=1
    baseline=budget_retrieval_report([x,y],[x],loaded,k=1)
    capped=budget_retrieval_report([y,x],[x],loaded,k=1)
    assert baseline['recall_at_1']==baseline['ndcg_at_1']==baseline['mrr']==1
    assert capped['recall_at_1']==capped['ndcg_at_1']==0 and capped['mrr']==0.5
    assert baseline['repository_ids']==capped['repository_ids']
    assert loaded.queries[0]['query'] not in json.dumps(baseline)
    assert 'no production qualification' in baseline['quality_scope']


@pytest.mark.parametrize('fault',['dimension','nonfinite','unnormalized','k'])
def test_incompatible_quality_inputs_never_become_metrics(tmp_path,fault):
    loaded=load_budget_fixture(write(tmp_path,fixture()));vectors=np.eye(2,384);query=np.eye(1,384);k=1
    if fault=='dimension':vectors=vectors[:,:383]
    elif fault=='nonfinite':vectors[0,0]=float('nan')
    elif fault=='unnormalized':vectors[0]*=2
    else:k=3
    with pytest.raises(ValueError):budget_retrieval_report(vectors,query,loaded,k=k)


def test_existing_encoder_comparison_fixture_is_a_supported_frozen_input():
    path=Path(__file__).resolve().parent/'fixtures/readme_retrieval.json'
    loaded=load_budget_fixture(path)
    assert loaded.cases and loaded.queries and 'synthetic' in loaded.label_provenance.lower()
