from dataclasses import replace
import uuid

import pytest

from retrieval.social_candidates import SocialCandidate, SocialPolicy, lookup_social_ids, select_social_candidates, validate_social_candidates


def candidate(number, *, owner='1', source='followed_owner', score=.5, support=1):
    return SocialCandidate(repo_id=str(uuid.UUID(int=number)), owner_github_id=owner,
                           source=source, source_score=score, support=support)


def test_independent_social_ids_have_bounded_canonical_lookup():
    rows = [candidate(i) for i in range(1, 101)]
    policy = SocialPolicy(mode='enabled', max_lookup=10, max_candidates=5)
    ids = lookup_social_ids(rows, policy, {rows[0].repo_id})
    assert len(ids) == 10 and rows[0].repo_id not in ids
    assert ids == sorted(ids)
    assert lookup_social_ids(rows, SocialPolicy(), set()) == []


def test_immutable_owner_and_source_caps_prevent_one_actor_or_channel_dominating():
    rows = [candidate(i, owner=str(1 if i < 8 else i)) for i in range(1, 20)]
    policy = SocialPolicy(mode='enabled', max_per_owner=2, max_per_source=5)
    result = select_social_candidates(rows, policy, eligible_repo_ids={r.repo_id for r in rows}, excluded=set())
    assert len(result) == 5
    assert sum(row.owner_github_id == '1' for row in result) == 2
    assert len({row.repo_id for row in result}) == len(result)


def test_deterministic_cross_source_rrf_fusion_preserves_actual_source_scores():
    rows = [candidate(1, score=.8), candidate(2, owner='2', score=.7),
            candidate(2, owner='2', source='co_save', score=.9, support=3)]
    policy = SocialPolicy(mode='enabled')
    result = select_social_candidates(rows, policy, eligible_repo_ids={r.repo_id for r in rows}, excluded=set())
    reversed_result = select_social_candidates(list(reversed(rows)), policy, eligible_repo_ids={r.repo_id for r in rows}, excluded=set())
    assert result == reversed_result
    assert result[0].repo_id == rows[1].repo_id
    assert result[0].fusion_score == pytest.approx(1/62 + 1/61)
    assert result[0].source_scores == {'co_save': .9, 'followed_owner': .7}
    assert result[0].primary_source == 'co_save'


def test_sparse_collaborative_support_and_ineligible_or_excluded_ids_cannot_enter():
    rows = [candidate(1, source='co_like'), candidate(2), candidate(3), candidate(4, score=0)]
    policy = SocialPolicy(mode='enabled')
    result = select_social_candidates(rows, policy, eligible_repo_ids={rows[0].repo_id, rows[1].repo_id, rows[3].repo_id}, excluded={rows[1].repo_id})
    assert result == []


def test_shadow_can_measure_candidates_without_permission_to_inject():
    rows = [candidate(1)]
    shadow = SocialPolicy(mode='shadow')
    assert not shadow.injects
    assert SocialPolicy(mode='enabled').injects
    assert select_social_candidates(rows, shadow, eligible_repo_ids={rows[0].repo_id}, excluded=set())


@pytest.mark.parametrize('change', [{'owner_github_id':'handle'}, {'owner_github_id':'01'}, {'owner_github_id':str(2**63)},
    {'source_score':float('nan')}, {'source_score':1.1}, {'support':True}, {'source':'private_board_dump'}, {'repo_id':'owner/name'}])
def test_untrusted_candidate_metadata_is_rejected(change):
    row = candidate(1).model_dump()
    with pytest.raises(ValueError): SocialCandidate.model_validate({**row, **change})


def test_conflicting_owner_or_duplicate_source_projection_fails_closed():
    with pytest.raises(ValueError, match='conflicting'):
        validate_social_candidates([candidate(1), candidate(1, owner='2', source='co_save', support=2)])
    with pytest.raises(ValueError, match='duplicate'):
        validate_social_candidates([candidate(1), candidate(1)])
    with pytest.raises(ValueError, match='100'):
        validate_social_candidates([candidate(i) for i in range(101)])


@pytest.mark.parametrize('change', [{'mode':'maybe'}, {'max_lookup':0}, {'max_candidates':101}, {'max_per_owner':True}, {'minimum_collaborative_support':1}])
def test_invalid_policy_cannot_weaken_bounds(change):
    with pytest.raises(ValueError): SocialPolicy(**change)

# Registered retrieval integration uses the real local Qdrant engine for filters,
# projections and vector reads; only the heavy model is a deterministic test model.
from tests.test_v2_retrieval import local_recommendation_corpus, _corpus_retriever


def social_for(corpus):
    return SocialCandidate(repo_id=corpus.repo_ids[-1], owner_github_id='123456',
                           source='followed_owner', support=1, source_score=.5)


def test_independent_social_id_reaches_required_heavy_ranking_and_wire_provenance(local_recommendation_corpus):
    corpus = local_recommendation_corpus
    retriever = _corpus_retriever(corpus, heavy=True)
    retriever.social_policy = SocialPolicy(mode='enabled')
    social = social_for(corpus)
    baseline = retriever.recommend_batch(corpus.user_id, 10, [], 'same-generation')
    baseline_ranked = {row['id'] for row in retriever.ranker.calls[-1][2]}
    assert social.repo_id not in baseline_ranked

    def rank_all(user_vector, skills, candidates):
        retriever.ranker.calls.append((user_vector, skills, candidates))
        return [{'repo_id': row['id'], 'final_score': 1. if row['id'] == social.repo_id else .1} for row in candidates]
    retriever.ranker.score_batch = rank_all
    batch = retriever.recommend_batch(corpus.user_id, 10, [], 'same-generation', False, [], [social])
    assert batch.ranker_applied and batch.served_ranker == 'heavy'
    assert social.repo_id in {row['id'] for row in retriever.ranker.calls[-1][2]}
    assert len(retriever.ranker.calls[-1][2]) <= retriever.max_candidates
    item = next(item for item in batch.items if item.repo_id == social.repo_id)
    assert item.source == 'followed_owner'
    assert item.retrieval_scores['followed_owner'] == .5
    social_reads = [call.kwargs for call in corpus.client.retrieve.call_args_list
                    if call.kwargs.get('ids') == [social.repo_id]]
    assert any(row['with_vectors'] is False for row in social_reads)
    assert all(len(row.kwargs['ids']) <= 100 for row in corpus.client.retrieve.call_args_list
               if row.kwargs.get('with_vectors') is False)


def test_shadow_and_disabled_do_not_change_feed_or_ranking_pool(local_recommendation_corpus):
    corpus = local_recommendation_corpus
    retriever = _corpus_retriever(corpus, heavy=True)
    social = social_for(corpus)
    retriever.social_policy = SocialPolicy(mode='disabled')
    disabled = retriever.recommend_batch(corpus.user_id, 10, [], 'same-generation', False, [], [social])
    before = [row['id'] for row in retriever.ranker.calls[-1][2]]
    retriever.social_policy = SocialPolicy(mode='shadow')
    shadow = retriever.recommend_batch(corpus.user_id, 10, [], 'same-generation', False, [], [social])
    assert disabled.items == shadow.items
    assert before == [row['id'] for row in retriever.ranker.calls[-1][2]]
    assert social.repo_id not in before
    assert retriever._ranking_counter_snapshot()['social_shadow_candidates'] == 1
    assert retriever._ranking_counter_snapshot()['social_injected_candidates'] == 0


def test_excluded_or_ineligible_social_points_never_reach_ranker(local_recommendation_corpus):
    from config import QDRANT_VECTOR_NAME
    from embedding.vector_contract import REPOSITORY_SERVING_ELIGIBILITY_FIELD
    corpus = local_recommendation_corpus
    retriever = _corpus_retriever(corpus, heavy=True)
    retriever.social_policy = SocialPolicy(mode='enabled')
    social = social_for(corpus)
    retriever.recommend_batch(corpus.user_id, 10, [social.repo_id], 'excluded', False, [], [social])
    assert social.repo_id not in {row['id'] for row in retriever.ranker.calls[-1][2]}
    corpus.client.set_payload(corpus.repository_collection,
        payload={REPOSITORY_SERVING_ELIGIBILITY_FIELD: 'invalid'}, points=[social.repo_id])
    retriever.recommend_batch(corpus.user_id, 10, [], 'invalid', False, [], [social])
    assert social.repo_id not in {row['id'] for row in retriever.ranker.calls[-1][2]}


def test_social_lookup_failure_does_not_hide_or_weaken_required_heavy_dependencies(local_recommendation_corpus, monkeypatch):
    from retrieval.v2_retriever import RetrievalDependencyError
    corpus = local_recommendation_corpus
    retriever = _corpus_retriever(corpus, heavy=True)
    retriever.social_policy = SocialPolicy(mode='enabled')
    social = social_for(corpus)
    monkeypatch.setattr(retriever, '_fetch_social_candidates', lambda ids: (_ for _ in ()).throw(TimeoutError('test outage')))
    batch = retriever.recommend_batch(corpus.user_id, 10, [], 'outage', False, [], [social])
    assert batch.ranker_applied
    assert social.repo_id not in {item.repo_id for item in batch.items}
    assert retriever._ranking_counter_snapshot()['social_lookup_failures'] == 1
    retriever.ranker = None
    with pytest.raises(RetrievalDependencyError, match='required V2 heavy'):
        retriever.recommend_batch(corpus.user_id, 10, [], 'missing-model', False, [], [social])


def test_registered_http_contract_accepts_backend_evidence_and_returns_source_scores(monkeypatch):
    from fastapi.testclient import TestClient
    from api.main import app
    from retrieval.v2_retriever import RankedRepository, RecommendationBatch
    from types import SimpleNamespace
    seen = []
    row = candidate(1)
    def recommend(*args):
        seen.append(args)
        return RecommendationBatch(items=[RankedRepository(row.repo_id, .9, 'followed_owner', {'followed_owner': .5})],
            model_version='test-heavy', embedding_version='repo-embedding-v2', ranker_applied=True, served_ranker='heavy')
    monkeypatch.setenv('INTERNAL_API_SECRET', 'isolated-social-contract')
    monkeypatch.setattr('api.v2.retriever', lambda: SimpleNamespace(recommend_batch=recommend))
    body = {'schema_version':2,'generation_id':str(uuid.uuid4()),'user_id':str(uuid.uuid4()),
            'feed_version':1,'limit':10,'context':{},'social_candidates':[row.model_dump()]}
    client = TestClient(app)
    response = client.post('/api/v2/recommendations/generate', headers={'x-internal-secret':'isolated-social-contract'}, json=body)
    assert response.status_code == 200, response.text
    assert seen[0][6] == [row]
    assert response.json()['items'][0]['retrieval_scores'] == {'followed_owner': .5}
    assert 'owner_github_id' not in response.json()['items'][0]
    body['social_candidates'].append(row.model_dump())
    assert client.post('/api/v2/recommendations/generate', headers={'x-internal-secret':'isolated-social-contract'}, json=body).status_code == 422
    assert len(seen) == 1
