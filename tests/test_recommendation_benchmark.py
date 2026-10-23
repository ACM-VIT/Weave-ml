"""Hand-calculated diagnostics, not human recommendation judgments."""
from __future__ import annotations

import copy
import math

import pytest

from evaluation.recommendation_benchmark import Bundle, Run, evaluate


def diagnostic():
    rid = lambda n: f'00000000-0000-4000-8000-{n:012d}'
    embedding = dict(model='diagnostic-basis', revision='a' * 40, dimension=3,
                     pooling='fixture', normalized=True, query_prompt='', document_prompt='', recipe='v1')
    repos = [dict(repo_id=rid(i), owner=f'owner-{i}', topics=['backend' if i < 4 else 'frontend'],
                  fork_family=f'family-{i}', vector=v, popularity=i, archived=i == 6)
             for i, v in enumerate([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                                     [0., 0., 1.], [0., .6, .8], [-1., 0., 0.]], 1)]
    repos[1]['owner'] = repos[0]['owner']
    repos[1]['fork_family'] = repos[0]['fork_family']
    profiles = [dict(profile_id=rid(100), intent='synthetic backend testing', seen=[rid(4)], disliked=[rid(6)]),
                dict(profile_id=rid(101), intent='synthetic frontend testing', seen=[rid(1)], disliked=[rid(2)])]
    grades = [[3, 2, 1, 0, 0, 0], [0, 0, 1, 3, 2, 0]]
    bundle = dict(schema_version=1, snapshot_id='synthetic-smoke-v1', created_at='2026-10-23T00:00:00Z',
                  provenance=dict(kind='synthetic', reference='invented unit-test labels; no human judgment'),
                  embedding=embedding, repositories=repos, profiles=profiles,
                  judgments=[dict(profile_id=p['profile_id'], repo_id=r['repo_id'], grade=g)
                             for p, row in zip(profiles, grades) for r, g in zip(repos, row)])
    lists = [[[2, 3, 1, 5], [1, 2, 3, 5], [1, 3, 5]], [[3, 4, 5], [4, 3, 5], [4, 5, 3]]]
    run = dict(schema_version=1, embedding=copy.deepcopy(embedding), model_id='synthetic-order-v1', model_sha256='b' * 64,
               scaler_sha256='c' * 64, feature_schema='diagnostic-v1', code_commit='d' * 40,
               created_at='2026-10-23T00:00:00Z', timing_kind='synthetic',
               outputs=[dict(profile_id=p['profile_id'], stages={stage: [dict(repo_id=rid(n), score=float(10-j),
                        source='exploration' if j == 2 else 'personalized', source_score=.5)
                        for j, n in enumerate(ids)] for stage, ids in zip(('retrieval', 'ranking', 'assembly'), rows)},
                        latency_ms=dict(retrieval=1., ranking=2., assembly=3.)) for p, rows in zip(profiles, lists)])
    return bundle, run


def report(bundle=None, run=None):
    b, r = diagnostic()
    return evaluate(Bundle.model_validate(bundle or b), Run.model_validate(run or r), k=3)


def test_hand_calculated_graded_metrics_and_stage_separation():
    result = report()
    ideal = 7 + 3 / math.log2(3) + .5
    actual = 7 + 1 / math.log2(3)
    final = result['stages']['assembly']
    assert final['metrics']['judged_ndcg_at_k'] == pytest.approx((actual / ideal + 1) / 2)
    assert final['metrics']['judged_recall_at_k'] == pytest.approx(5 / 6)
    assert final['metrics']['judged_mrr_at_k'] == 1
    assert final['metrics']['catalogue_coverage'] == .8
    assert final['metrics']['repeat_rate'] == 0
    assert final['metrics']['excluded_rate'] == 0
    assert final['metrics']['fork_repeat_rate'] == 0
    assert result['stages']['ranking']['metrics']['fork_repeat_rate'] > 0
    assert result['stages']['retrieval']['metrics']['judged_ndcg_at_k'] != final['metrics']['judged_ndcg_at_k']
    assert final['counts']['profiles'] == 2
    assert final['counts']['returned'] == 6
    assert final['counts']['latency_samples'] == 2
    assert final['metrics']['latency_p95_ms'] == 3
    assert result == report()


def test_unknown_judgments_are_counted_and_not_silently_declared_negative():
    b, r = diagnostic()
    b['judgments'] = [j for j in b['judgments'] if j['repo_id'] != b['repositories'][4]['repo_id']]
    result = report(b, r)['stages']['assembly']
    assert result['metrics']['judgment_coverage'] == pytest.approx(4 / 6)
    assert result['counts']['unjudged_returned'] == 2


def test_duplicate_and_excluded_outputs_are_measured_without_sanitizing():
    b, r = diagnostic()
    rows = r['outputs'][0]['stages']['assembly']
    rows[1] = copy.deepcopy(rows[0])
    rows[2]['repo_id'] = b['profiles'][0]['seen'][0]
    result = report(b, r)['stages']['assembly']
    assert result['metrics']['repeat_rate'] == pytest.approx(1 / 6)
    assert result['metrics']['excluded_rate'] == pytest.approx(1 / 6)
    assert result['metrics']['judged_recall_at_k'] <= 1
    assert result['metrics']['judged_ndcg_at_k'] <= 1


@pytest.mark.parametrize('mutation', ['nan', 'zero', 'dimension', 'duplicate_repo', 'conflicting_judgment', 'boolean_grade', 'naive_time'])
def test_invalid_frozen_input_fails(mutation):
    b, _ = diagnostic()
    if mutation == 'nan': b['repositories'][0]['vector'][0] = float('nan')
    if mutation == 'zero': b['repositories'][0]['vector'] = [0., 0., 0.]
    if mutation == 'dimension': b['repositories'][0]['vector'] = [1.]
    if mutation == 'duplicate_repo': b['repositories'].append(copy.deepcopy(b['repositories'][0]))
    if mutation == 'conflicting_judgment': b['judgments'].append(copy.deepcopy(b['judgments'][0]))
    if mutation == 'boolean_grade': b['judgments'][0]['grade'] = True
    if mutation == 'naive_time': b['created_at'] = '2026-10-23T00:00:00'
    with pytest.raises(ValueError): Bundle.model_validate(b)


def test_run_requires_complete_profiles_stages_and_matching_embedding_identity():
    b, r = diagnostic()
    del r['outputs'][0]['stages']['ranking']
    with pytest.raises(ValueError): report(b, r)
    b, r = diagnostic()
    r['outputs'].pop()
    with pytest.raises(ValueError, match='profiles'): report(b, r)
    b, r = diagnostic()
    r['embedding']['query_prompt'] = 'changed meaning'
    with pytest.raises(ValueError, match='embedding'): report(b, r)


def test_similarity_reuses_pairs_across_profiles_stages_and_reverse_order(monkeypatch):
    from evaluation import recommendation_benchmark as benchmark
    b, r = diagnostic()
    # Preserve per-position repeated-ID contributions and self-pairs. Near-unit
    # vectors at the accepted tolerance make each pair's clamp observable.
    b['repositories'][0]['vector'] = [math.sqrt(1.000001), 0., 0.]
    b['repositories'][1]['vector'] = [-math.sqrt(1.000001), 0., 0.]
    ids = [repo['repo_id'] for repo in b['repositories']]
    orders = [[ids[0], ids[0], ids[1], ids[2]], [ids[2], ids[1], ids[0], ids[0]]]
    for index, output in enumerate(r['outputs']):
        for stage in output['stages']:
            output['stages'][stage] = [dict(repo_id=repo_id, score=float(4-i),
                source='diagnostic', source_score=.5) for i, repo_id in enumerate(orders[index])]
    bundle, run = Bundle.model_validate(b), Run.model_validate(r)
    pairs = set()
    for output in run.outputs:
        for rows in output.stages.values():
            for i, a in enumerate(rows):
                for c in rows[i+1:]:
                    pairs.add(tuple(sorted((a.repo_id, c.repo_id))))
    calls = []
    original = benchmark._clamped_cosine
    def counted(left, right):
        calls.append((left, right))
        return original(left, right)
    monkeypatch.setattr(benchmark, '_clamped_cosine', counted)
    result = benchmark.evaluate(bundle, run, k=4)
    assert len(calls) == len(pairs)
    repos = {repo.repo_id: repo for repo in bundle.repositories}
    for stage in ('retrieval', 'ranking', 'assembly'):
        expected, visits = 0., 0
        for output in run.outputs:
            rows = output.stages[stage]
            for i, a in enumerate(rows):
                for c in rows[i+1:]:
                    expected += max(-1., min(1., sum(x*y for x,y in zip(
                        repos[a.repo_id].vector, repos[c.repo_id].vector))))
                    visits += 1
        assert result['stages'][stage]['counts']['similarity_pairs'] == visits == 12
        assert result['stages'][stage]['metrics']['intra_list_similarity'] == expected / visits
    assert result['stages']['assembly']['metrics']['repeat_rate'] == .25
    assert result['stages']['assembly']['metrics']['intra_list_similarity'] == -1 / 6


@pytest.mark.parametrize('capacity', [0, 1, 2])
def test_similarity_cache_eviction_preserves_every_report_field(monkeypatch, capacity):
    from evaluation import recommendation_benchmark as benchmark
    b, r = diagnostic()
    bundle, run = Bundle.model_validate(b), Run.model_validate(r)
    expected = benchmark.evaluate(bundle, run, k=4)
    monkeypatch.setattr(benchmark, '_SIMILARITY_CACHE_PAIRS', capacity)
    assert benchmark.evaluate(bundle, run, k=4) == expected


def test_similarity_cache_cannot_reuse_another_snapshots_vectors():
    from evaluation import recommendation_benchmark as benchmark
    b, r = diagnostic()
    run = Run.model_validate(r)
    first = benchmark.evaluate(Bundle.model_validate(b), run, k=3)
    b['repositories'][0]['vector'] = [-1., 0., 0.]
    second = benchmark.evaluate(Bundle.model_validate(b), run, k=3)
    assert first['stages']['ranking']['metrics']['intra_list_similarity'] != second['stages']['ranking']['metrics']['intra_list_similarity']
