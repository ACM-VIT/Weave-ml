"""Actual heavy-path transport bounds and validated-inline ranking parity."""
import pytest

from scripts.benchmark_candidate_vectors import run_case


def test_vectors_are_fetched_once_only_for_deduplicated_retained_candidates():
    result = run_case(overlap='low', max_candidates=50)
    retained = result['ranker_candidate_ids']
    assert len(retained) == len(set(retained)) == 50
    assert result['repository_vector_ids'] == retained
    vectors = [row for row in result['records'] if row['stage'] == 'vectors']
    assert len(vectors) == 1 and set(vectors[0]['ids']) == set(retained)
    initial = [row for row in result['records'] if row['stage'] not in {'profile', 'vectors'}]
    assert all(row['vector_ids'] == [] for row in initial)
    dropped = {repo_id for row in initial for repo_id in row['ids']} - set(retained)
    assert dropped and not dropped.intersection(result['repository_vector_ids'])
    assert result['ranker_applied'] is True


def test_hybrid_only_candidate_path_never_fetches_repository_vectors():
    result = run_case(overlap='low', heavy=False)
    assert result['repository_vector_ids'] == []
    assert not any(row['stage'] == 'vectors' for row in result['records'])
    assert result['ranker_applied'] is False


@pytest.mark.parametrize('overlap', ['low', 'high'])
def test_required_heavy_ranking_matches_validated_inline_transport_mode(overlap):
    current = run_case(overlap=overlap, capture_inputs=True)
    control = run_case(overlap=overlap, inline=True, capture_inputs=True)
    assert current['output'] == control['output']
    assert current['ranker_inputs'] == control['ranker_inputs']
    inputs = current['ranker_inputs']
    assert inputs['user_terms'] and inputs['user_vector']
    rows = list(inputs['candidates'].values())
    assert len({tuple(row['embedding']) for row in rows}) == len(rows)
    assert all(row['topics'] and row['tags'] and row['languages'] for row in rows)
    assert current['ranker_applied'] is control['ranker_applied'] is True


@pytest.mark.parametrize('change', ['deleted', 'ineligible', 'version', 'shape'])
def test_retained_vector_fetch_rejects_changed_snapshot_before_heavy_ranking(change):
    result = run_case(overlap='high', change=change)
    assert result['changed_id'] is not None
    assert result['changed_id'] not in result['ranker_candidate_ids']
    assert result['changed_id'] not in {row[0] for row in result['output']}
    assert result['ranker_applied'] is True


def test_measured_transport_tradeoff_keeps_extra_round_trip_visible():
    low = {mode: run_case(overlap='low', inline=mode) for mode in (False, True)}
    high = {mode: run_case(overlap='high', inline=mode) for mode in (False, True)}
    assert low[False]['transport_calls'] == high[False]['transport_calls'] == 7
    assert low[True]['transport_calls'] == high[True]['transport_calls'] == 6
    assert low[False]['wire_bytes'] > low[True]['wire_bytes']
    assert high[False]['wire_bytes'] < high[True]['wire_bytes']
    assert all(result['json_decode_cpu_ms'] > 0 for cases in (low, high) for result in cases.values())
