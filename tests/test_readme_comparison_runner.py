"""Real parser/runner and default-off CLI orchestration without live GitHub."""
import argparse
import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
import subprocess
import sys
import threading

import pytest

from scripts import compare_readme_strategies as comparison
from tests.test_readme_strategy_comparison import Fixture


def sample(tmp_path):
    path=tmp_path/'sample.json';path.write_text(json.dumps({'repositories':['Example/project']}));return path


def test_both_real_parser_orders_and_empty_content_metadata_distinction():
    for order in [('aliases','metadata'),('metadata','aliases')]:
        measured=comparison.Measurement(Fixture({'README.md':'canonical'}))
        result=comparison.compare_repository('Example/project',measured,strategies=order)
        assert list(result['strategies'])==list(order)
        assert result['canonical_content_equal'] and result['content_path_branch_equal']
        assert result['shared_reference_elapsed_ms']>=0
        assert all(row['elapsed_ms']>=0 for row in result['strategies'].values())
    result=comparison.compare_repository('Example/project',comparison.Measurement(Fixture()))
    assert result['canonical_content_equal'] is True
    assert result['content_path_branch_equal'] is False  # Explicit missing branch metadata difference.


def test_default_cli_does_not_inspect_a_credential_or_instantiate_live_transport(tmp_path,monkeypatch,capsys):
    path=sample(tmp_path)
    monkeypatch.setattr(comparison,'LiveGitHubTransport',lambda *a,**k:pytest.fail('default used a token'))
    monkeypatch.setenv('GITHUB_README_COMPARISON_TOKEN','must-not-be-read')
    assert comparison.main(['--sample',str(path),'--repeats','2'])==0
    value=json.loads(capsys.readouterr().out)
    assert value['mode']=='plan_only' and value['max_graphql_requests']==8
    assert 'must-not-be-read' not in json.dumps(value)


def test_reference_transport_failure_is_a_reported_trial_without_content_calls():
    from evaluation.readme_experiment import summarize_trials
    calls=[]
    def unavailable(query,variables):
        calls.append(query)
        raise OSError('private-provider-response-or-token')
    measurement=comparison.Measurement(unavailable)
    result=comparison.compare_repository('Example/project',measurement)
    report=summarize_trials([result])
    assert len(calls)==1 and result['status']=='reference_failed'
    assert report['reference_failed_trials']==report['failed_trials']==1
    assert report['failure_rate_of_all_trials']==1
    assert report['strategies']['aliases']['not_run_trials']==1
    assert 'private-provider' not in json.dumps([result,report])


def test_cli_rejects_bad_sample_without_private_json_or_transport(tmp_path,monkeypatch,capsys):
    path=tmp_path/'invalid.json';path.write_text('{"private-input":"do-not-print-this",bad}')
    monkeypatch.setattr(comparison,'LiveGitHubTransport',lambda *a,**k:pytest.fail('bad sample connected'))
    with pytest.raises(SystemExit) as caught:
        comparison.main(['--sample',str(path),'--execute-approved'])
    assert caught.value.code==2
    output=capsys.readouterr()
    assert 'invalid bounded sample' in output.err and 'do-not-print-this' not in output.err


def test_actual_private_worker_runs_both_parsers_and_cancels_hard_watchdog(tmp_path,monkeypatch,capsys):
    path=sample(tmp_path);repos,digest=comparison._load_sample_snapshot(path)
    timers=[]
    class Timer:
        def __init__(self,seconds,callback):self.seconds=seconds;self.cancelled=False;timers.append(self)
        def start(self):pass
        def cancel(self):self.cancelled=True
    class ApprovedFixtureTransport:
        def __init__(self,token):
            assert token=='owned-fixture-developer-token';self.observations=[];self.fixture=Fixture({'README.md':'private source content do not export'})
        def verify_identity(self):self.observations.append({'kind':'identity','http_status':200,'wire_bytes':1,'decoded_body_bytes':1,'transport_elapsed_ms':1,'http_request_started':True})
        @property
        def last_measurement(self):return self.observations[-1]
        def __call__(self,query,variables):
            value=self.fixture(query,variables);length=len(json.dumps(value).encode())
            self.observations.append({'kind':'graphql','http_status':200,'wire_bytes':length,'decoded_body_bytes':length,'transport_elapsed_ms':1,'http_request_started':True})
            return value
    monkeypatch.setattr(comparison,'start_worker_watchdog',lambda: _timer())
    def _timer():
        timer=Timer(180,None);timer.start();return timer
    monkeypatch.setattr(comparison,'LiveGitHubTransport',ApprovedFixtureTransport)
    monkeypatch.setattr(comparison,'_source_hashes',lambda:{'frozen_source':'a'*64})
    monkeypatch.setenv('READONLY_FIXTURE_TOKEN','owned-fixture-developer-token')
    args=SimpleNamespace(sample=path,repeats=2,token_env='READONLY_FIXTURE_TOKEN')
    assert comparison._live_worker(args,repos,digest)==0
    output=capsys.readouterr().out;report=json.loads(output)
    assert report['physical_graphql_attempts']==8 and report['identity_http_checks']==1
    assert report['sample_sha256']==digest and len(report['trials'])==2
    assert report['trials'][0]['strategy_order']!=report['trials'][1]['strategy_order']
    assert report['summary']['canonical_content_equal_trials']==2
    assert report['summary']['strategies']['aliases']['wire_bytes_including_shared_reference']>0
    assert 'private source content' not in output and 'owned-fixture-developer-token' not in output
    assert timers[0].seconds==180 and timers[0].cancelled


@pytest.mark.parametrize('fault',['timeout','child_error','invalid_json'])
def test_failed_worker_never_overwrites_report_or_prints_private_stderr(tmp_path,monkeypatch,capsys,fault):
    path=sample(tmp_path);output=tmp_path/'new-report.json';calls=[]
    def child(command,**options):
        calls.append(command)
        assert options['timeout']==180 and '--sample-sha256' in command
        assert 'private-token' not in command
        if fault=='timeout':raise subprocess.TimeoutExpired(command,180)
        return SimpleNamespace(returncode=2 if fault=='child_error' else 0,stdout='not-json',stderr='private-token and provider content')
    monkeypatch.setattr(comparison,'_source_hashes',lambda:{'source':'stable'})
    monkeypatch.setattr(comparison.subprocess,'run',child)
    assert comparison.main(['--sample',str(path),'--execute-approved','--output',str(output)])==2
    assert calls and not output.exists()
    captured=capsys.readouterr();assert 'private-token' not in captured.out+captured.err
    assert 'README_EXPERIMENT_UNAVAILABLE' in captured.err
    output.write_bytes(b'prior immutable evidence')
    calls.clear()
    assert comparison.main(['--sample',str(path),'--execute-approved','--output',str(output)])==2
    assert not calls and output.read_bytes()==b'prior immutable evidence'


def test_worker_cannot_execute_a_sample_different_from_parent_snapshot(tmp_path,monkeypatch):
    path=sample(tmp_path)
    monkeypatch.setattr(comparison,'_live_worker',lambda *a:pytest.fail('changed sample executed'))
    with pytest.raises(SystemExit):
        comparison.main(['--sample',str(path),'--worker','--execute-approved','--sample-sha256','0'*64])


def test_actual_cli_parent_and_worker_publish_owned_http_measurement(tmp_path):
    """Real CLI processes/parsers/files; loopback HTTP replaces GitHub TLS only."""
    from tests.test_tower_history import require_local_only_git
    require_local_only_git()
    fixture=Fixture({'README.md':'private canonical fixture content must never be exported'})
    calls=[]
    token='owned-loopback-fixture-credential'
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path=='/user' and self.headers.get('Authorization')=='Bearer '+token
            calls.append('identity');self.respond({'login':'TarunRam-git'})
        def do_POST(self):
            assert self.path=='/graphql' and self.headers.get('Authorization')=='Bearer '+token
            value=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append('graphql');self.respond(fixture(value['query'],value['variables']))
        def respond(self,value):
            raw=json.dumps(value).encode();self.send_response(200)
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    bootstrap=tmp_path/'bootstrap';bootstrap.mkdir()
    (bootstrap/'sitecustomize.py').write_text(
        'import http.client\noriginal=http.client.HTTPConnection\n'
        'def owned_connection(host, **options):\n'
        '    assert host == "api.github.com"\n'
        f'    return original("127.0.0.1",{server.server_port},timeout=options["timeout"])\n'
        'http.client.HTTPSConnection=owned_connection\n')
    source=Path(comparison.__file__).resolve()
    path=sample(tmp_path);output=tmp_path/'report.json'
    environment={**os.environ,'PYTHONPATH':str(bootstrap),'FIXTURE_README_TOKEN':token,
        'PYTHONDONTWRITEBYTECODE':'1'}
    try:
        completed=subprocess.run([sys.executable,'-B',str(source),'--sample',str(path),
            '--repeats','2','--execute-approved','--token-env','FIXTURE_README_TOKEN',
            '--output',str(output)],cwd=source.parents[1],env=environment,
            capture_output=True,text=True,timeout=25)
        assert completed.returncode==0,completed.stderr
        raw=output.read_text();report=json.loads(raw)
        assert report['physical_graphql_attempts']==8 and report['identity_http_checks']==1
        assert calls.count('identity')==1 and calls.count('graphql')==8
        assert report['summary']['canonical_content_equal_trials']==2
        assert report['summary']['failure_rate_of_all_trials']==0
        assert report['summary']['strategies']['metadata']['wire_bytes_including_shared_reference']>0
        assert token not in raw+completed.stdout+completed.stderr
        assert 'private canonical fixture content' not in raw+completed.stdout+completed.stderr
        assert not list(tmp_path.glob('.readme-experiment-*'))
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)
        assert not thread.is_alive()


def test_reference_graphql_errors_count_as_failures_and_preserve_reference_costs():
    from evaluation.readme_experiment import summarize_trials
    def failed(query, variables):
        return {'data':{'repository':None,'rateLimit':{'cost':None,'remaining':1000}},
                'errors':[{'message':'private provider details'}]}
    trial=comparison.compare_repository('Example/project',comparison.Measurement(failed))
    report=summarize_trials([trial])
    assert report['failed_trials']==report['reference_failed_trials']==1
    assert report['failure_rate_of_all_trials']==1
    costs=report['shared_reference_totals']
    assert costs['graphql_calls']==1
    assert costs['rate_cost'] is None
    assert costs['normalized_json_bytes']>0
    assert costs['wire_bytes'] is None
    assert 'private provider details' not in json.dumps([trial,report])


def test_worker_refuses_source_changed_since_import_before_any_transport(monkeypatch,capsys):
    monkeypatch.setattr(comparison,'_IMPORTED_SOURCE_SHA256',{'source':'old'},raising=False)
    monkeypatch.setattr(comparison,'_source_hashes',lambda:{'source':'new'})
    monkeypatch.setattr(comparison,'LiveGitHubTransport',lambda *a:pytest.fail('source drift reached network'))
    args=SimpleNamespace(repeats=1,token_env='NO_TOKEN')
    assert comparison._live_worker(args,['Example/project'],'f'*64)==2
    assert not capsys.readouterr().out


def test_parent_passes_source_identity_before_child_and_refuses_other_source(tmp_path,monkeypatch):
    source={'source':'parent-frozen'}
    monkeypatch.setattr(comparison,'_source_hashes',lambda:dict(source))
    def child(command,**options):
        assert command[command.index('--expected-source-sha256')+1]==comparison._source_digest(source)
        return SimpleNamespace(returncode=0,stdout=json.dumps({'source_sha256':{'source':'other'}}),stderr='')
    monkeypatch.setattr(comparison.subprocess,'run',child)
    output=tmp_path/'report.json'
    assert comparison.main(['--sample',str(sample(tmp_path)),'--execute-approved','--output',str(output)])==2
    assert not output.exists()
