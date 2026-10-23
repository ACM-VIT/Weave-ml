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


def test_actual_assembler_requires_qualified_vectors_metadata_and_preserves_source_scores():
    from inference.feed_assembly import FeedAssemblySystem, DiversityUnavailableError
    rows = [candidate(1,(1,0,0)),candidate(2,(0,1,0),bucket='trending'),candidate(3,(0,0,1),bucket='adjacent')]
    assembly = [{'repo_id':r.repo_id,'score':r.score,'source':r.bucket,'candidate_bucket':r.bucket,
        'owner_github_id':r.owner_github_id,'fork_family_github_id':r.fork_family_github_id,
        'retrieval_scores':{r.bucket:r.score}} for r in rows]
    vectors = {r.repo_id:list(r.vector) for r in rows}
    assembler = FeedAssemblySystem(freshness_boost=0)
    shaped = assembler.shape_batch(assembly,target_size=2,diversity_policy=replace(policy(),minimum_exploration=1),
        diversity_vectors=vectors,diversity_seed='g')
    assert len(shaped)==2 and any(row['candidate_bucket']!='personalized' for row in shaped)
    assert all(row['retrieval_scores']==next(r['retrieval_scores'] for r in assembly if r['repo_id']==row['repo_id']) for row in shaped)
    with pytest.raises(DiversityUnavailableError,match='vectors'):
        assembler.shape_batch(assembly,target_size=2,diversity_policy=policy(),diversity_vectors={})
    with pytest.raises(DiversityUnavailableError,match='metadata'):
        assembler.shape_batch([{**assembly[0],'owner_github_id':None}],target_size=1,
            diversity_policy=policy(),diversity_vectors=vectors)


def test_diversity_cooperatively_stops_inside_candidate_work():
    calls = 0
    def check():
        nonlocal calls
        calls += 1
        if calls==2: raise TimeoutError('request deadline')
    with pytest.raises(TimeoutError,match='deadline'):
        rerank_diverse([candidate(1,(1,0,0)),candidate(2,(0,1,0))],target_size=2,seed='g',policy=policy(),check_work=check)

# Real local Qdrant filters/projections and fresh retained vector reads; only the
# model scores are deterministic fixture outputs, not real-user quality labels.
from tests.test_v2_retrieval import local_recommendation_corpus, _corpus_retriever


@pytest.fixture
def diversity_corpus(local_recommendation_corpus):
    from config import QDRANT_VECTOR_NAME
    from qdrant_client import models
    corpus = local_recommendation_corpus
    points = corpus.client.retrieve(collection_name=corpus.repository_collection,ids=corpus.repo_ids,
                                    with_vectors=False,with_payload=True)
    rebuilt = []
    for index,point in enumerate(points):
        payload = {**point.payload,'github_id':str(100000+index),'owner_github_id':str(index//2+1),
            'is_fork':False,'fork_family_github_id':str(100000+index),
            'lineage_source':'github-graphql-repository','lineage_observed_at':'2026-10-23T00:00:00+00:00',
            'topics':[f'topic_{index%20}'],'category':f'category_{index%10}'}
        vector = [0.0]*384
        vector[0],vector[1],vector[2+index%382] = 1.0,index/600,.8
        rebuilt.append(models.PointStruct(id=str(point.id),vector={QDRANT_VECTOR_NAME:vector},payload=payload))
    corpus.client.upsert(collection_name=corpus.repository_collection,points=rebuilt)
    corpus.client.retrieve.reset_mock()
    return corpus


def test_actual_retriever_diversity_shadow_and_enabled_preserve_fresh_pairs_and_source_evidence(diversity_corpus):
    from retrieval.social_candidates import SocialCandidate,SocialPolicy
    corpus = diversity_corpus
    retriever = _corpus_retriever(corpus,heavy=True)
    retriever.social_policy = SocialPolicy(mode='enabled')
    social = SocialCandidate(repo_id=corpus.repo_ids[-1],owner_github_id='300',source='followed_owner',support=1,source_score=.5)
    baseline = retriever.recommend_batch(corpus.user_id,10,[],'diversity-generation',False,[],[social])
    retriever.diversity_traffic_percent = 100
    retriever.diversity_policy = DiversityPolicy(minimum_exploration=1)
    retriever.diversity_mode = 'shadow'
    shadow = retriever.recommend_batch(corpus.user_id,10,[],'diversity-generation',False,[],[social])
    assert shadow.items == baseline.items
    retriever.diversity_mode = 'enabled'
    enabled = retriever.recommend_batch(corpus.user_id,10,[],'diversity-generation',False,[],[social])
    assert enabled.ranker_applied and enabled.served_ranker=='heavy'
    assert social.repo_id in {row.repo_id for row in enabled.items}
    assert next(row for row in enabled.items if row.repo_id==social.repo_id).retrieval_scores['followed_owner']==.5
    assert len(enabled.items)==10
    assert retriever._ranking_counter_snapshot()['diversity_served_requests']==1
    excluded = next(row.repo_id for row in enabled.items if row.repo_id!=social.repo_id)
    replay = retriever.recommend_batch(corpus.user_id,10,[excluded],'diversity-excluded',False,[],[social])
    assert excluded not in {row.repo_id for row in replay.items}
    corpus.client.set_payload(collection_name=corpus.repository_collection,
                              payload={'owner_github_id':None},points=[corpus.repo_ids[0]])
    from retrieval.v2_retriever import RetrievalDependencyError
    with pytest.raises(RetrievalDependencyError,match='diversity'):
        retriever.recommend_batch(corpus.user_id,10,[],'diversity-unknown',False,[],[social])


def test_diversity_does_not_add_repository_vector_reads_to_hybrid_and_rejects_enabled_hybrid(local_recommendation_corpus,monkeypatch):
    corpus = local_recommendation_corpus
    retriever = _corpus_retriever(corpus,heavy=False)
    retriever.diversity_mode='shadow'; retriever.diversity_traffic_percent=100
    retriever.recommend_batch(corpus.user_id,10,[],'hybrid-shadow')
    assert len(corpus.client.retrieve.call_args_list)==1
    assert all(not call.kwargs.get('with_vectors') for call in corpus.client.query_points.call_args_list)
    assert retriever._ranking_counter_snapshot()['diversity_unavailable_requests']==1
    monkeypatch.setenv('V2_DIVERSITY_MODE','enabled')
    with pytest.raises(ValueError,match='strict-heavy'):
        _corpus_retriever(corpus,heavy=False)


@pytest.mark.parametrize('mode', ['disabled', 'shadow', 'enabled'])
def test_full_semantic_pool_retains_independent_exploration_only_when_enabled(diversity_corpus, mode):
    corpus = diversity_corpus
    independent = set(corpus.repo_ids[300:450])
    corpus.client.set_payload(corpus.repository_collection,
        payload={'star_count': 10000, 'trend_velocity': 10000., 'activity_score': 1.},
        points=sorted(independent))
    retriever = _corpus_retriever(corpus, heavy=True)
    retriever.diversity_mode = mode
    retriever.diversity_traffic_percent = 100
    retriever.diversity_policy = DiversityPolicy(minimum_exploration=2)
    batch = retriever.recommend_batch(corpus.user_id, 30, [], 'full-pool')
    ranked_ids = {row['id'] for row in retriever.ranker.calls[-1][2]}
    assert len(ranked_ids) == 150
    if mode == 'enabled':
        assert len(ranked_ids & independent) >= 2
        assert len({row.repo_id for row in batch.items} & independent) >= 2
        assert batch.ranker_applied
    else:
        assert not ranked_ids.intersection(independent)


def test_shadow_skips_optional_selector_near_parent_deadline(diversity_corpus, monkeypatch):
    from utils.deadline import RequestDeadline, check_deadline, deadline_scope
    corpus = diversity_corpus
    retriever = _corpus_retriever(corpus, heavy=True)
    baseline = retriever.recommend_batch(corpus.user_id, 10, [], 'near-deadline')
    retriever.diversity_mode = 'shadow'
    retriever.diversity_traffic_percent = 100
    retriever.diversity_policy = DiversityPolicy(minimum_exploration=0)
    clock = [0.]
    deadline = RequestDeadline(30, clock=lambda: clock[0])
    original_score = retriever.ranker.score_batch
    def finish_near_deadline(*args):
        result = original_score(*args)
        clock[0] = deadline.expires_at - .001
        return result
    monkeypatch.setattr(retriever.ranker, 'score_batch', finish_near_deadline)
    original_shape = retriever.assembler.shape_batch
    shadow_calls = []
    def slow_optional(*args, **kwargs):
        if kwargs.get('diversity_policy') is not None:
            shadow_calls.append(True)
            clock[0] += .002
            check_deadline()
        return original_shape(*args, **kwargs)
    monkeypatch.setattr(retriever.assembler, 'shape_batch', slow_optional)
    with deadline_scope(deadline):
        result = retriever.recommend_batch(corpus.user_id, 10, [], 'near-deadline')
    assert result.items == baseline.items
    assert not shadow_calls
    assert not deadline.cancelled.is_set()


@pytest.mark.parametrize('parent_cancelled', [False, True])
def test_shadow_child_expiry_is_isolated_but_parent_cancellation_is_authoritative(
    diversity_corpus, monkeypatch, parent_cancelled,
):
    from utils.deadline import (
        CooperativeDeadlineExceeded, RequestDeadline, check_deadline,
        current_deadline, deadline_scope,
    )
    corpus = diversity_corpus
    retriever = _corpus_retriever(corpus, heavy=True)
    baseline = retriever.recommend_batch(corpus.user_id, 10, [], 'child-budget')
    retriever.diversity_mode = 'shadow'
    retriever.diversity_traffic_percent = 100
    retriever.diversity_policy = DiversityPolicy(minimum_exploration=0)
    clock = [0.]
    parent = RequestDeadline(30, clock=lambda: clock[0])
    original = retriever.assembler.shape_batch
    children = []
    def expire_optional(*args, **kwargs):
        if kwargs.get('diversity_policy') is not None:
            child = current_deadline()
            assert child is not parent
            assert child.expires_at < parent.expires_at - .05
            children.append(child)
            if parent_cancelled:
                parent.cancel('cancelled')
            else:
                clock[0] = child.expires_at + .001
            check_deadline()
        return original(*args, **kwargs)
    monkeypatch.setattr(retriever.assembler, 'shape_batch', expire_optional)
    with deadline_scope(parent):
        if parent_cancelled:
            with pytest.raises(CooperativeDeadlineExceeded):
                retriever.recommend_batch(corpus.user_id, 10, [], 'child-budget')
        else:
            result = retriever.recommend_batch(corpus.user_id, 10, [], 'child-budget')
            assert result.items == baseline.items
            assert parent.remaining() > 29
    assert len(children) == 1
    assert parent.cancelled.is_set() is parent_cancelled


@pytest.mark.parametrize('defect', ['same_family', 'duplicate_vector'])
@pytest.mark.parametrize(('social_count', 'pool_cap'), [(0, 150), (20, 150), (100, 150), (50, 50)])
def test_enabled_pool_preserves_compatible_discovery_breadth_and_personalization(
    diversity_corpus, defect, social_count, pool_cap,
):
    from config import QDRANT_VECTOR_NAME
    from qdrant_client import models
    from retrieval.config import DISCOVERY_LIMIT
    from retrieval.social_candidates import SocialCandidate, SocialPolicy
    corpus = diversity_corpus
    independent = set(corpus.repo_ids[300:450])
    corpus.client.set_payload(corpus.repository_collection,
        payload={'star_count': 10000, 'trend_velocity': 10000., 'activity_score': 1.},
        points=sorted(independent))
    first, second, third = corpus.repo_ids[300:303]
    if defect == 'same_family':
        corpus.client.set_payload(corpus.repository_collection,
            payload={'is_fork': True, 'fork_parent_github_id': '900000',
                     'fork_family_github_id': '900000'}, points=[first, second])
    else:
        point = corpus.client.retrieve(corpus.repository_collection, ids=[first],
                                      with_payload=True, with_vectors=True)[0]
        other = corpus.client.retrieve(corpus.repository_collection, ids=[second],
                                      with_payload=True, with_vectors=True)[0]
        corpus.client.upsert(corpus.repository_collection, points=[models.PointStruct(
            id=second, vector=point.vector, payload=other.payload)])
    retriever = _corpus_retriever(corpus, heavy=True)
    retriever.max_candidates = pool_cap
    retriever.diversity_mode = 'enabled'
    retriever.diversity_traffic_percent = 100
    retriever.diversity_policy = DiversityPolicy(minimum_exploration=2)
    retriever.social_policy = SocialPolicy(mode='enabled', max_candidates=100, max_per_source=100)
    social_ids = corpus.repo_ids[450:450 + social_count]
    social = [SocialCandidate(repo_id=repo_id, owner_github_id=str(index // 2 + 1),
              source='followed_owner', support=1, source_score=1.)
              for index, repo_id in enumerate(corpus.repo_ids) if repo_id in social_ids]
    batch = retriever.recommend_batch(corpus.user_id, 30, [], 'compatible-breadth', False, [], social)
    pool = {row['id'] for row in retriever.ranker.calls[-1][2]}
    assert len(pool) == pool_cap
    assert len(pool & independent) == DISCOVERY_LIMIT
    assert {first, second, third} <= pool
    admitted_social = min(social_count, pool_cap - DISCOVERY_LIMIT - 1)
    assert len(set(social_ids) & pool) == admitted_social
    # Tiny full-social custom pools explicitly prioritize enabled discovery and
    # the semantic floor; no claim that every social reservation can always fit.
    # Abundant independent rows cannot displace every personalized ANN row.
    assert len(pool & set(corpus.repo_ids[:150])) == pool_cap - DISCOVERY_LIMIT - admitted_social
    assert len(batch.items) == 30 and batch.ranker_applied
    assert len({row.repo_id for row in batch.items} & independent) >= 2


def test_enabled_quota_cannot_silently_eliminate_present_personalized_candidates(diversity_corpus):
    from retrieval.v2_retriever import RetrievalDependencyError
    retriever = _corpus_retriever(diversity_corpus, heavy=True)
    retriever.max_candidates = 50
    retriever.diversity_mode = 'enabled'
    retriever.diversity_traffic_percent = 100
    retriever.diversity_policy = DiversityPolicy(minimum_exploration=50)
    with pytest.raises(RetrievalDependencyError, match='personalized candidate floor'):
        retriever.recommend_batch(diversity_corpus.user_id, 50, [], 'incompatible-quota')
