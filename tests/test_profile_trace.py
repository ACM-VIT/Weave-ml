"""Supplied-trace math, missingness, privacy and file/CLI boundaries."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from evaluation.profile_trace import analyze_trace, load_trace, validate_trace

USER='1'*64;PEER='2'*64;PROCESS='3'*64;OTHER_PROCESS='4'*64;CLOCK='5'*64;CONTRACT='6'*64


def record(**changes):
    return {'subject_key':USER,'process_key':PROCESS,'clock_key':CLOCK,'at_ms':0,'profile_version':'1',
        'feedback_version':'0','feedback_state_revision':'0','embedding_contract_key':CONTRACT,
        'compatible':True,'fetch_duration_ms':5,**changes}


def trace(*rows):
    return {'schema_version':1,'provenance':{'kind':'synthetic','reference':'isolated test fixture, not traffic evidence'},'records':list(rows)}


def test_identical_versions_never_reuse_another_user_and_process_scope_is_explicit():
    value=trace(record(),record(subject_key=PEER,at_ms=1),record(process_key=OTHER_PROCESS,at_ms=2),record(at_ms=3))
    report=analyze_trace(value,max_age_ms=[4])
    upper=report['unbounded_free_version_oracle']
    assert upper['global_duplicate_version_keys']==2
    assert upper['process_local_duplicate_version_keys']==1
    assert upper['subjects_observed_on_multiple_processes']==1
    assert report['max_age_experiments'][0]['compatible_repeat_opportunities']==1
    assert 'not actual cache hits' in report['scope']
    assert USER not in json.dumps(report) and PROCESS not in json.dumps(report)
    assert value['provenance']['reference'] not in json.dumps(report)
    assert len(report['provenance']['reference_sha256'])==64


def test_explicit_age_boundary_and_nonrenewing_fill_semantics():
    report=analyze_trace(trace(record(at_ms=0),record(at_ms=9),record(at_ms=18)),max_age_ms=[9,10])
    assert [row['compatible_repeat_opportunities'] for row in report['max_age_experiments']]==[0,1]
    assert report['observed_cost']=={'samples':3,'p50_ms':5,'p95_ms':5}


def test_version_and_embedding_identity_changes_do_not_become_hits():
    value=trace(record(),record(at_ms=1,feedback_version='1'),record(at_ms=2,feedback_version='1',feedback_state_revision='1'),
        record(at_ms=3,embedding_contract_key='7'*64))
    report=analyze_trace(value,max_age_ms=[100])
    assert report['unbounded_free_version_oracle']['global_duplicate_version_keys']==0
    assert report['max_age_experiments'][0]['compatible_repeat_opportunities']==0


def test_missing_revisions_compatibility_clock_and_process_are_unknown_not_zero():
    value=trace(record(profile_version=None),record(compatible=None),record(process_key=None),record(clock_key=None),record(at_ms=None),record(compatible=False))
    report=analyze_trace(value,max_age_ms=[100])
    coverage=report['coverage']
    assert coverage['missing_revision']==coverage['unknown_compatibility']==coverage['incompatible_reads']==1
    assert coverage['missing_process']==coverage['missing_clock']==coverage['missing_timestamp']==1
    assert report['clock_and_order']['temporally_eligible_reads']==0
    assert report['max_age_experiments'][0]['fraction_of_temporally_eligible_reads'] is None
    all_unknown=analyze_trace(trace({}))
    assert all_unknown['unbounded_free_version_oracle']['global_fraction_of_eligible_reads'] is None
    assert all_unknown['observed_cost']['p95_ms'] is None


def test_clock_domains_and_order_do_not_imply_cross_process_freshness():
    report=analyze_trace(trace(record(at_ms=2),record(at_ms=1),record(at_ms=3,clock_key='8'*64)),max_age_ms=[100])
    assert report['clock_and_order']['input_rows_out_of_timestamp_order']==1
    assert report['clock_and_order']['comparable_same_subject_request_gaps']==1
    assert report['max_age_experiments'][0]['compatible_repeat_opportunities']==1


def test_chronological_revision_regression_excluded_from_time_opportunities():
    report=analyze_trace(trace(record(at_ms=0,feedback_version='1'),record(at_ms=1,feedback_version='2'),record(at_ms=2,feedback_version='1')),max_age_ms=[100])
    assert report['clock_and_order']['chronological_revision_regressions_excluded_from_time_experiments']==1
    assert report['clock_and_order']['temporally_eligible_reads']==2
    assert report['max_age_experiments'][0]['compatible_repeat_opportunities']==0


@pytest.mark.parametrize('field,value',[('user_id','raw-identity-not-permitted'),('vector',[1.0]),('subject_key','raw-user'),
    ('profile_version','0'),('feedback_version',1),('feedback_version',True),('feedback_version','01'),
    ('feedback_state_revision',str(2**63)),('at_ms',True),('at_ms',2**53),('compatible',1),
    ('fetch_duration_ms',float('nan')),('fetch_duration_ms',10**1000)])
def test_raw_identity_payloads_and_lossy_or_malformed_revisions_are_rejected(field,value):
    with pytest.raises(ValueError,match='PROFILE_TRACE_INVALID'):validate_trace(trace(record(**{field:value})))


def test_invalid_unicode_reference_is_rejected_with_bounded_error():
    value=trace(record());value['provenance']['reference']='\ud800'
    with pytest.raises(ValueError,match='PROFILE_TRACE_INVALID'):validate_trace(value)


@pytest.mark.parametrize('ages',[[0],[True],[1,1],list(range(1,7)),None])
def test_explicit_experiment_bounds_do_not_choose_or_coerce_a_ttl(ages):
    with pytest.raises(ValueError):analyze_trace(trace(record()),max_age_ms=ages)


def test_frozen_regular_file_and_exact_large_versions(tmp_path):
    path=tmp_path/'trace.json';value=trace(record(profile_version=str(2**63-1)));path.write_text(json.dumps(value))
    actual,digest=load_trace(path)
    assert actual==value and len(digest)==64
    link=tmp_path/'link';link.symlink_to(path)
    with pytest.raises((ValueError,OSError)):load_trace(link)
    huge=tmp_path/'huge'
    with huge.open('wb') as stream:stream.truncate(32*1024*1024+1)
    with pytest.raises(ValueError):load_trace(huge)


@pytest.mark.parametrize('fault',['duplicate_root','duplicate_nested','nan','wrong_root'])
def test_duplicate_nonfinite_or_incorrect_json_root_rejected(tmp_path,fault):
    raw=json.dumps(trace(record()))
    if fault=='duplicate_root':raw=raw.replace('"schema_version": 1','"schema_version": 1, "schema_version": 1')
    elif fault=='duplicate_nested':raw=raw.replace('"at_ms": 0','"at_ms": 0, "at_ms": 0')
    elif fault=='nan':raw=raw.replace('"fetch_duration_ms": 5','"fetch_duration_ms": NaN')
    else:raw='[]'
    path=tmp_path/'invalid';path.write_text(raw)
    with pytest.raises(ValueError):load_trace(path)


@pytest.mark.parametrize('compatible',[None,False])
def test_known_newer_ineligible_revision_still_blocks_later_older_reuse(compatible):
    report=analyze_trace(trace(record(at_ms=0,feedback_version='1'),
        record(at_ms=1,feedback_version='2',compatible=compatible),
        record(at_ms=2,feedback_version='1')),max_age_ms=[100])
    assert report['clock_and_order']['chronological_revision_regressions_excluded_from_time_experiments']==1
    assert report['clock_and_order']['temporally_eligible_reads']==1
    assert report['max_age_experiments'][0]['compatible_repeat_opportunities']==0


def test_actual_file_cli_reports_only_aggregates_without_services(tmp_path):
    path=tmp_path/'trace.json';path.write_text(json.dumps(trace(record(),record(at_ms=1))))
    script=Path(__file__).resolve().parents[1]/'scripts/analyze_profile_trace.py'
    result=subprocess.run([sys.executable,str(script),'--trace',str(path),'--max-age-ms','2'],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    assert report['max_age_experiments'][0]['compatible_repeat_opportunities']==1
    assert USER not in result.stdout and 'no cache installed' in report['decision']
