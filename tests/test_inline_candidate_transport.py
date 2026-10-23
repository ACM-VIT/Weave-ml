"""Fresh atomic vector pairs on the actual opt-in required-heavy path."""
import pytest
from scripts.benchmark_candidate_vectors import FixtureRanker, WireQdrant, isolated_io
from retrieval.v2_retriever import QdrantV2Retriever, RetrievalDependencyError


def engine(client, **kwargs):
    return QdrantV2Retriever(client=client, ranker=FixtureRanker(),
        candidate_vector_mode='inline', minimum_eligible_repositories=0, **kwargs)


def test_actual_inline_avoids_retained_vector_round_trip():
    client = WireQdrant()
    with isolated_io():
        batch = engine(client).recommend_batch(client.user_id, 10, [])
    assert batch.ranker_applied
    assert len(client.records) == 6
    assert client.records[0]['stage'] == 'profile'
    assert not any(row['stage'] == 'vectors' for row in client.records)


def test_inline_option_keeps_hybrid_repository_reads_vector_free():
    client = WireQdrant()
    with isolated_io():
        batch = engine(client, heavy_ranker_enabled=False).recommend_batch(client.user_id, 10, [])
    assert not batch.ranker_applied
    assert all(not row['vector_ids'] for row in client.records)


def test_missing_required_profile_fails_before_repository_vector_reads():
    class Missing(WireQdrant):
        def retrieve(self, **kwargs):
            if kwargs['collection_name'] == 'user_profiles':
                return self.exchange('profile', [])
            return super().retrieve(**kwargs)
    client = Missing()
    with isolated_io(), pytest.raises(RetrievalDependencyError):
        engine(client).recommend_batch(client.user_id, 10, [])
    assert all(not row['vector_ids'] for row in client.records)


@pytest.mark.parametrize('change', ['content', 'feature', 'topics', 'vector', 'shape', 'finite', 'ineligible'])
def test_conflicting_fresh_channels_cannot_contribute_to_heavy_ranker(change):
    class Conflicting(WireQdrant):
        def query_points(self, **kwargs):
            result = super().query_points(**kwargs)
            point = result.points[0]
            if change == 'content':
                point.payload['content_version'] += 1
            elif change == 'feature':
                point.payload['feature_version'] += 1
            elif change == 'topics':
                point.payload['topics'] = ['conflicting-topic']
            elif change == 'shape':
                point.vector['repo_embedding'] = [1.]
            elif change == 'finite':
                point.vector['repo_embedding'][0] = float('nan')
            elif change == 'vector':
                point.vector['repo_embedding'] = list(reversed(self.vector))
            else:
                point.payload['serving_eligibility_version'] = 'invalid'
            self.bad_id = point.id
            return result
    client = Conflicting()
    ranker = FixtureRanker()
    with isolated_io():
        retriever = engine(client)
        retriever.ranker = ranker
        retriever.recommend_batch(client.user_id, 10, [])
    assert client.bad_id not in {row['id'] for row in ranker.candidates}


def test_metadata_only_social_candidate_retains_fresh_vector_hydration():
    from uuid import UUID
    from retrieval.social_candidates import SocialCandidate, SocialPolicy
    client = WireQdrant()
    social_id = str(UUID(int=500))
    with isolated_io():
        retriever = engine(client, social_policy=SocialPolicy(mode='enabled'))
        retriever.recommend_batch(client.user_id, 10, [], social_candidates=[SocialCandidate(
            repo_id=social_id, owner_github_id='123', source='followed_owner', source_score=.5, support=1,
        )])
    reads = [row for row in client.records if row['stage'] == 'vectors']
    assert len(reads) == 2  # metadata-only social lookup, then retained hydration
    assert reads[0]['vector_ids'] == []
    assert social_id in reads[1]['vector_ids']
    assert social_id in {row['id'] for row in retriever.ranker.candidates}


def test_inline_pairs_are_not_reused_across_requests():
    client = WireQdrant()
    with isolated_io():
        retriever = engine(client)
        retriever.recommend_batch(client.user_id, 10, [])
        first = len(client.records)
        retriever.recommend_batch(client.user_id, 10, [])
    assert first == 6 and len(client.records) == 12
    assert all(row['stage'] != 'vectors' for row in client.records)


def test_invalid_transport_mode_is_rejected():
    with pytest.raises(ValueError, match='deferred or inline'):
        QdrantV2Retriever(client=WireQdrant(), candidate_vector_mode='auto-guessed', heavy_ranker_enabled=False)


from tests.test_v2_retrieval import local_recommendation_corpus, _corpus_retriever


def test_local_qdrant_inline_projection_and_social_hydration(local_recommendation_corpus):
    from retrieval.social_candidates import SocialCandidate, SocialPolicy
    corpus = local_recommendation_corpus
    retriever = _corpus_retriever(corpus, heavy=True)
    retriever.candidate_vector_mode = 'inline'
    retriever.social_policy = SocialPolicy(mode='enabled')
    social_id = corpus.repo_ids[-1]
    batch = retriever.recommend_batch(corpus.user_id, 10, [], social_candidates=[SocialCandidate(
        repo_id=social_id, owner_github_id='123', source='followed_owner', source_score=.5, support=1,
    )])
    assert batch.ranker_applied
    assert social_id in {row['id'] for row in retriever.ranker.calls[-1][2]}
    repository_reads = [call.kwargs for call in corpus.client.retrieve.call_args_list
                        if call.kwargs.get('collection_name') == retriever.repository_collection]
    assert any(row.get('with_vectors') is False for row in repository_reads)
    assert any(row.get('with_vectors') and social_id in row['ids'] for row in repository_reads)
