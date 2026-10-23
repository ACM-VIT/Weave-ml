from dataclasses import replace
import math
import pytest
from inference.diversity import DiversityCandidate, DiversityPolicy, rerank_diverse


def candidate(n, vector, **patch):
    return DiversityCandidate(f'00000000-0000-4000-8000-{n:012}', 1-n/100,
                              tuple(vector), str(n), str(100+n), **patch)


def policy(**patch):
    return DiversityPolicy(dimensions=3, minimum_exploration=0, **patch)


def test_mmr_penalizes_actual_embedding_similarity_with_configurable_weight():
    rows = [candidate(1,(1,0,0)), candidate(2,(.95,.31,0)), candidate(3,(0,1,0))]
    relevance = rerank_diverse(rows,target_size=2,seed='generation',policy=policy(relevance_weight=1))
    diversity = rerank_diverse(rows,target_size=2,seed='generation',policy=policy(relevance_weight=.1))
    assert relevance.candidates == tuple(rows[:2])
    assert diversity.candidates == (rows[0], rows[2])
    assert diversity.intra_list_similarity == pytest.approx(0)
    assert relevance.intra_list_similarity > .9


def test_recent_exclusions_and_near_duplicates_never_refill_the_page():
    rows = [candidate(1,(1,0,0)),candidate(2,(100,0,0)),candidate(3,(0,1,0))]
    result = rerank_diverse(rows,target_size=3,seed='generation',policy=policy(),
                            recent_repo_ids=frozenset({rows[2].repo_id}))
    assert result.candidates == (rows[0],)
    assert result.excluded_recent == 1 and result.shortfall == 2


def test_hard_owner_topic_category_and_fork_family_controls_survive_sparse_pools():
    rows = [candidate(1,(1,0,0),topics=('python',),category='tools'),
            candidate(2,(0,1,0),topics=('python',),category='tools'),
            candidate(3,(0,0,1),topics=('rust',),category='science'),
            candidate(4,(-1,0,0),topics=('go',),category='other')]
    for field, value in [('owner_github_id', rows[0].owner_github_id),
                         ('fork_family_github_id', rows[0].fork_family_github_id)]:
        changed = [rows[0],replace(rows[1],**{field:value})]
        result = rerank_diverse(changed,target_size=2,seed='g',policy=policy(max_per_owner=1))
        assert len(result.candidates) == 1
    for bound in [{'max_per_topic':1},{'max_per_category':1}]:
        result = rerank_diverse(rows,target_size=4,seed='g',policy=policy(**bound))
        assert len(result.candidates) == 3 and rows[1] not in result.candidates


def test_exploration_reserves_genuinely_new_sources_outside_personalized_top_slice():
    rows = [candidate(1,(1,0,0)),candidate(2,(0,1,0)),
            candidate(3,(0,0,1),bucket='trending'),candidate(4,(-1,0,0),bucket='adjacent')]
    result = rerank_diverse(rows,target_size=3,seed='g',policy=replace(policy(),minimum_exploration=2),
                           personalized_top_ids=frozenset({rows[0].repo_id,rows[1].repo_id}))
    assert result.exploration_count == 2 and result.exploration_shortfall == 0
    assert {rows[2].repo_id,rows[3].repo_id} <= {row.repo_id for row in result.candidates}
    assert {row.bucket for row in result.candidates} >= {'adjacent','trending'}
    # Relabeling an already personalized item cannot satisfy exploration.
    result = rerank_diverse([replace(rows[0],bucket='exploration')],target_size=1,seed='g',
        policy=replace(policy(),minimum_exploration=1),personalized_top_ids=frozenset({rows[0].repo_id}))
    assert result.exploration_count == 0 and result.exploration_shortfall == 1


def test_seeded_ties_are_input_order_independent_and_missing_evidence_remains_explicit():
    rows = [replace(candidate(1,(1,0,0)),score=1), replace(candidate(2,(0,1,0)),score=1),
            replace(candidate(3,(0,0,1)),score=1)]
    first = rerank_diverse(rows,target_size=2,seed='same',policy=policy())
    assert rerank_diverse(list(reversed(rows)),target_size=2,seed='same',policy=policy()) == first
    assert len({rerank_diverse(rows,target_size=1,seed=str(i),policy=policy()).candidates[0].repo_id for i in range(10)}) > 1
    unknown = rerank_diverse([replace(rows[0],owner_github_id=None),replace(rows[1],fork_family_github_id=None)],
                            target_size=2,seed='g',policy=policy())
    assert unknown.unknown_identity == 2 and unknown.shortfall == 2
    assert unknown.intra_list_similarity is None


@pytest.mark.parametrize('patch', [{'vector':(0,0,0)}, {'vector':(1,2)}, {'vector':(math.nan,1,0)},
    {'score':math.inf}, {'repo_id':'owner/name'}, {'topics':('Python',)}, {'owner_github_id':'handle'}])
def test_invalid_vectors_and_labels_fail_and_unknown_identity_is_never_inferred(patch):
    row = replace(candidate(1,(1,0,0)),**patch)
    if 'owner_github_id' in patch:
        assert rerank_diverse([row],target_size=1,seed='g',policy=policy()).unknown_identity == 1
    else:
        with pytest.raises(ValueError): rerank_diverse([row],target_size=1,seed='g',policy=policy())


def test_pool_target_and_policy_bounds_are_enforced():
    row = candidate(1,(1,0,0))
    with pytest.raises(ValueError,match='unique'): rerank_diverse([row,row],target_size=1,seed='g',policy=policy())
    with pytest.raises(ValueError,match='pool'): rerank_diverse([row]*151,target_size=1,seed='g',policy=policy())
    with pytest.raises(ValueError,match='target_size'): rerank_diverse([],target_size=151,seed='g',policy=policy())
    with pytest.raises(ValueError): policy(relevance_weight=math.nan)
    assert rerank_diverse([],target_size=0,seed='g',policy=policy()).shortfall == 0


def test_exploration_reservation_backtracks_instead_of_greedily_consuming_all_topic_capacity():
    rows = [candidate(1,(1,0,0),topics=('python','rust'),bucket='exploration'),
            candidate(2,(0,1,0),topics=('python',),bucket='exploration'),
            candidate(3,(0,0,1),topics=('rust',),bucket='exploration')]
    result = rerank_diverse(rows,target_size=2,seed='g',
        policy=replace(policy(max_per_topic=1),minimum_exploration=2))
    assert set(row.repo_id for row in result.candidates) == {rows[1].repo_id,rows[2].repo_id}
    assert result.exploration_shortfall == 0 and not result.quota_search_exhausted


def test_quota_search_has_an_explicit_work_bound_and_never_relaxes_hard_caps():
    rows = [candidate(1,(1,0,0),topics=('python','rust'),bucket='exploration'),
            candidate(2,(0,1,0),topics=('python',),bucket='exploration'),
            candidate(3,(0,0,1),topics=('rust',),bucket='exploration')]
    result = rerank_diverse(rows,target_size=2,seed='g',
        policy=replace(policy(max_per_topic=1),minimum_exploration=2,quota_search_limit=1))
    assert result.quota_search_exhausted and result.exploration_shortfall == 1
    assert len(result.candidates) == 1
