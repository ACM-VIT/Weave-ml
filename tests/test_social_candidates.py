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
