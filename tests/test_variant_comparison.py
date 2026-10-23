"""Synthetic actual-component treatments, never real relevance/privacy evidence."""
import copy,math
from pathlib import Path
import pytest
from tests.test_stage_runner import snapshot_data,ROOT
from tests.test_tower_history import require_local_only_git
from evaluation.variant_comparison import VariantExperiment,compare_experiment
from evaluation.recommendation_benchmark import digest


def diagnostic_experiment():
    snapshot=snapshot_data();old=copy.deepcopy(snapshot['repositories'][0]['payload'])
    rid=lambda n:f'00000000-0000-4000-8000-{n:012d}'
    repos=[];points=[]
    for i in range(220):
        group=i//5;a=.99-group*.003 if i<200 else -.4
        vector=[0.]*384;vector[0]=a;vector[group+1]=math.sqrt(1-a*a);vector[i+70]=.01
        norm=math.sqrt(sum(x*x for x in vector));vector=[x/norm for x in vector]
        github=10000+i;family=10000+group*5;owner=str(1+(i//3)%24)
        stars=10000-i if i<120 else 300+i if i<200 else 1
        topics=[f'synthetic-topic-{group%8}']
        repos.append(dict(repo_id=rid(i+1),owner=owner,topics=topics,fork_family=str(family),
                          vector=vector,popularity=stars,archived=i==199))
        p=copy.deepcopy(old)
        p.update(repo_id=rid(i+1),github_id=str(github),full_name=f'synthetic/repository{i}',owner_github_id=owner,
                 is_fork=i%5!=0,fork_parent_github_id=str(family) if i%5 else None,fork_family_github_id=str(family),
                 topics=topics,category=f'synthetic-category-{group%4}',star_count=stars,
                 trend_velocity=10.+group if 25<=group<36 else .1 if i<200 else 0.,
                 activity_score=.95 if 25<=group<36 else .2 if i<200 else 0.,
                 pushed_days_ago=1 if i<200 else 999,pushed_at='2026-10-23T00:00:00Z' if i<200 else '2020-01-01T00:00:00Z')
        points.append(dict(repo_id=rid(i+1),payload=p))
    snapshot['repositories']=points;snapshot['bundle']['repositories']=repos;snapshot['limit']=15
    snapshot['bundle']['snapshot_id']='synthetic-variant-compute-v1'
    snapshot['bundle']['provenance']['reference']='generated labels, vectors, lineage and social signals; no human or backend evidence'
    snapshot['bundle']['judgments']=[dict(profile_id=p['profile_id'],repo_id=r['repo_id'],
        grade=3 if (i//5)%8==j else 1 if (i//5)%4==j else 0)
        for j,p in enumerate(snapshot['bundle']['profiles']) for i,r in enumerate(repos)]
    second=[0.]*384;second[0]=.98;second[1]=.2;norm=math.sqrt(sum(x*x for x in second))
    snapshot['profiles'][1]['vector']=[x/norm for x in second]
    diversity=dict(dimensions=384,relevance_weight=.7,max_per_owner=2,max_per_topic=5,max_per_category=8,
        max_per_fork_family=1,duplicate_cosine=.98,minimum_exploration=2,max_pool=150,quota_search_limit=5000)
    social=dict(mode='enabled',max_lookup=100,max_candidates=20,max_per_owner=2,max_per_source=8,
                minimum_collaborative_support=2,traffic_percent=100.)
    evidence=[]
    for profile in snapshot['profiles']:
        rows=[]
        for i in range(200,210):
            for source,support in [('followed_owner',1),('co_like',3)]:
                rows.append(dict(repo_id=rid(i+1),owner_github_id=repos[i]['owner'],source=source,support=support,source_score=.9))
        evidence.append(dict(viewer_id=profile['profile_id'],observed_at=snapshot['as_of'],declared_origin='synthetic_fixture',
            declared_policy='public-signals-own-seeds-v1',reference='generated public-source-shaped diagnostic; not backend privacy proof',candidates=rows))
    treatments=[dict(name='baseline',diversity=None,social=None),dict(name='diversity',diversity=diversity,social=None),
                dict(name='social',diversity=None,social=social),dict(name='combined',diversity=diversity,social=social)]
    return dict(schema_version=1,snapshot=snapshot,treatments=treatments,social_evidence=evidence,k=15)


@pytest.fixture(scope='module')
def actual_comparison():
    require_local_only_git()
    value=VariantExperiment.model_validate(diagnostic_experiment())
    return value,compare_experiment(value,ROOT/'inference')


def test_actual_same_input_conditions_export_required_metrics_and_contribution(actual_comparison):
    experiment,comparison=actual_comparison
    assert comparison['label_provenance']['kind']=='synthetic'
    assert comparison['variant_order']==['baseline','diversity','social','combined']
    identities=[]
    for name,value in comparison['results'].items():
        exported=value['execution'];identities.append((exported['snapshot_sha256'],exported['artifacts'],exported['source_sha256']))
        for stage in ('retrieval','ranking','assembly'):
            metrics=value['report']['stages'][stage]['metrics']
            assert {'judged_ndcg_at_k','judged_recall_at_k','catalogue_coverage','intra_list_similarity','novelty_bits'}<=set(metrics)
            assert metrics['excluded_rate']==0 and metrics['archived_rate']==0
        assert value['report']['stages']['assembly']['source_share']
        if name in {'social','combined'}:
            counts=value['overlapping_source_presence']['retrieval']['all_source_counts']
            assert counts.get('co_like',0)>0 and counts.get('followed_owner',0)>0
        for output in exported['run']['outputs']:
            calls=exported['configuration']['runtime']['shape_calls_by_profile'][output['profile_id']]
            assert calls==(2 if name in {'diversity','combined'} else 1)
    assert all(value==identities[0] for value in identities)


def test_actual_diversity_hard_controls_and_exploration(actual_comparison):
    experiment,comparison=actual_comparison
    repos={r.repo_id:r for r in experiment.snapshot.bundle.repositories}
    for name in ('diversity','combined'):
        for output in comparison['results'][name]['execution']['run']['outputs']:
            rows=output['stages']['assembly'];ids=[r['repo_id'] for r in rows]
            assert len(ids)==len(set(ids))
            from collections import Counter
            assert max(Counter(repos[i].owner for i in ids).values())<=2
            assert max(Counter(repos[i].fork_family for i in ids).values())<=1
            assert max(Counter(t for i in ids for t in repos[i].topics).values())<=5
            for pos,a in enumerate(ids):
                for b in ids[pos+1:]:
                    assert sum(x*y for x,y in zip(repos[a].vector,repos[b].vector))<.98
            assert sum(row['source']!='semantic' for row in rows)>=2


@pytest.mark.parametrize('kind',['private_source','wrong_owner','duplicate_viewer','missing_policy','too_many_rows','late_evidence'])
def test_invalid_source_privacy_bounds_and_policy_rejected(kind):
    raw=diagnostic_experiment()
    if kind=='private_source':raw['social_evidence'][0]['candidates'][0]['source']='co_save'
    if kind=='wrong_owner':raw['social_evidence'][0]['candidates'][0]['owner_github_id']='99999'
    if kind=='duplicate_viewer':raw['social_evidence'][1]=copy.deepcopy(raw['social_evidence'][0])
    if kind=='missing_policy':del raw['treatments'][1]['diversity']['max_per_owner']
    if kind=='too_many_rows':raw['social_evidence'][0]['candidates']*=6
    if kind=='late_evidence':raw['social_evidence'][0]['observed_at']='2026-10-25T00:00:00Z'
    with pytest.raises((ValueError,TypeError)):VariantExperiment.model_validate(raw)


@pytest.mark.parametrize('field', ['labels', 'scope', 'deltas', 'source_presence', 'source_scope', 'exposure_scope', 'time', 'seed'])
def test_parent_rejects_corrupted_derived_evidence(actual_comparison, field):
    from evaluation.variant_comparison import validate_comparison_result
    experiment, original = actual_comparison
    value = copy.deepcopy(original)
    if field == 'labels': value['label_provenance']['kind'] = 'human'
    if field == 'scope': value['scope'] = 'production qualification proved'
    if field == 'deltas': value['delta_from_baseline']['baseline']['assembly']['catalogue_coverage'] = 0.75
    if field == 'source_presence': value['results']['baseline']['overlapping_source_presence']['assembly']['positions'] = 12345
    if field == 'source_scope': value['source_presence_scope'] = 'causal attribution'
    if field == 'exposure_scope': value['exposure_scope'] = 'all recent-window privacy checks passed'
    if field == 'time': value['results']['baseline']['execution']['as_of'] = '2030-01-01T00:00:00Z'
    if field == 'seed': value['results']['baseline']['execution']['seed'] = 'different-seed'
    with pytest.raises(ValueError):
        validate_comparison_result(value, experiment)


def test_parent_rejects_source_dirty_state_mismatch(actual_comparison):
    from evaluation.variant_comparison import validate_comparison_result
    experiment, original=actual_comparison
    value=copy.deepcopy(original)
    execution=value['results']['social']['execution']
    execution['source_tree_dirty']=not execution['source_tree_dirty']
    with pytest.raises(ValueError,match='source identity'):
        validate_comparison_result(value,experiment)


