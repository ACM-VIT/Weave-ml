"""Owned local HTTP transport, budget, privacy and report arithmetic coverage."""
import gzip
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import traceback

import pytest

from evaluation import readme_experiment as experiment

TOKEN="owned-fixture-token-not-a-real-credential"


@pytest.fixture
def owned_http(monkeypatch):
    responses=[];requests=[];hosts=[]
    class Handler(BaseHTTPRequestHandler):
        def handle_request(self):
            length=int(self.headers.get('Content-Length','0'))
            requests.append((self.command,self.path,self.headers.get('Authorization'),self.rfile.read(length)))
            status,headers,body=responses.pop(0)
            self.send_response(status)
            for key,value in headers.items():self.send_header(key,value)
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        do_GET=handle_request;do_POST=handle_request
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    original=http.client.HTTPConnection
    def local_connection(host,**kwargs):
        hosts.append(host);assert host=='api.github.com'
        return original('127.0.0.1',server.server_port,timeout=kwargs['timeout'])
    monkeypatch.setattr(experiment.http.client,'HTTPSConnection',local_connection)
    try:yield responses,requests,hosts
    finally:server.shutdown();server.server_close();thread.join(timeout=2);assert not thread.is_alive()


def identity(responses,login='TarunRam-git'):
    responses.append((200,{},json.dumps({'login':login}).encode()))


def test_fixed_host_same_token_identity_and_actual_payload_bytes(owned_http,monkeypatch):
    responses,requests,hosts=owned_http;identity(responses)
    raw=b'{ "data": { "result": "content" } }\n'
    responses.append((200,{},raw))
    monkeypatch.setenv('HTTPS_PROXY','http://unreachable.invalid:3128')
    transport=experiment.LiveGitHubTransport(TOKEN);transport.verify_identity()
    assert transport('query { value }',{})=={'data':{'result':'content'}}
    assert hosts==['api.github.com','api.github.com']
    assert [row[:2] for row in requests]==[('GET','/user'),('POST','/graphql')]
    assert all(row[2]=='Bearer '+TOKEN for row in requests)
    assert transport.last_measurement['wire_bytes']==len(raw)
    assert transport.last_measurement['decoded_body_bytes']==len(raw)
    assert transport.last_measurement['transport_elapsed_ms']>=0
    assert TOKEN not in json.dumps(transport.observations)+repr(transport)


def test_wrong_subject_never_permits_graphql_and_identity_is_not_retried(owned_http):
    responses,requests,_=owned_http;identity(responses,'someone-else')
    transport=experiment.LiveGitHubTransport(TOKEN)
    with pytest.raises(experiment.ReadmeExperimentError,match='IDENTITY'):transport.verify_identity()
    with pytest.raises(experiment.ReadmeExperimentError,match='IDENTITY'):transport('query {}',{})
    with pytest.raises(experiment.ReadmeExperimentError,match='IDENTITY'):transport.verify_identity()
    assert len(requests)==1


def test_redirect_is_rejected_without_following_location(owned_http):
    responses,requests,_=owned_http;responses.append((307,{'Location':'https://untrusted.invalid/token'},b'private provider error'))
    with pytest.raises(experiment.ReadmeExperimentError,match='HTTP'):
        experiment.LiveGitHubTransport(TOKEN).verify_identity()
    assert len(requests)==1


def test_gzip_wire_and_decoded_bytes_are_distinct(owned_http):
    responses,_,_=owned_http;identity(responses)
    raw=json.dumps({'data':{'content':'word '*100}}).encode();packed=gzip.compress(raw)
    responses.append((200,{'Content-Encoding':'gzip'},packed))
    transport=experiment.LiveGitHubTransport(TOKEN);transport.verify_identity();transport('query {}',{})
    assert transport.last_measurement['wire_bytes']==len(packed)
    assert transport.last_measurement['decoded_body_bytes']==len(raw)


@pytest.mark.parametrize('compression',[False,True])
def test_streaming_wire_or_decompression_limit_prevents_unbounded_body(owned_http,monkeypatch,compression):
    responses,_,_=owned_http;identity(responses)
    transport=experiment.LiveGitHubTransport(TOKEN);transport.verify_identity()
    monkeypatch.setattr(experiment,'MAX_RESPONSE_BYTES',128)
    raw=b'{"private":"'+b'x'*2000+b'"}'
    responses.append((200,{'Content-Encoding':'gzip'} if compression else {},gzip.compress(raw) if compression else raw))
    with pytest.raises(experiment.ReadmeExperimentError,match='BODY_LIMIT'):transport('query {}',{})
    assert transport.last_measurement['wire_bytes']<=129


@pytest.mark.parametrize('raw',[b'{"login":"TarunRam-git","login":"other"}',b'{"login":NaN}',b'[]',b'not json'])
def test_ambiguous_or_invalid_provider_json_never_becomes_identity(owned_http,raw):
    responses,_,_=owned_http;responses.append((200,{},raw))
    with pytest.raises(experiment.ReadmeExperimentError,match='JSON'):
        experiment.LiveGitHubTransport(TOKEN).verify_identity()


def test_private_transport_error_has_no_original_context_or_private_traceback(monkeypatch):
    private='private-provider-body-and-'+TOKEN
    class Broken:
        sock=None
        def request(self,*args,**kwargs):raise ValueError(private)
        def close(self):pass
    monkeypatch.setattr(experiment.http.client,'HTTPSConnection',lambda *args,**kwargs:Broken())
    with pytest.raises(experiment.ReadmeExperimentError) as caught:
        experiment.LiveGitHubTransport(TOKEN).verify_identity()
    error=caught.value
    assert error.__context__ is None and error.__cause__ is None
    assert '_request_private' not in [frame.name for frame in traceback.extract_tb(error.__traceback__)]
    assert private not in ''.join(traceback.format_exception(error))


def test_expired_budget_and_call_ceiling_prevent_connections(monkeypatch):
    monkeypatch.setattr(experiment.http.client,'HTTPSConnection',lambda *args,**kwargs:pytest.fail('prevented request connected'))
    transport=experiment.LiveGitHubTransport(TOKEN);transport.deadline=0
    with pytest.raises(experiment.ReadmeExperimentError,match='DEADLINE'):transport.verify_identity()
    transport=experiment.LiveGitHubTransport(TOKEN);transport.observations.extend([{}]*41);transport._verified=True
    with pytest.raises(experiment.ReadmeExperimentError,match='BOUND'):transport('query {}',{})


@pytest.mark.parametrize('token',['',TOKEN+'\n','x'*4097])
def test_invalid_header_credentials_fail_before_any_network(token):
    with pytest.raises(experiment.ReadmeExperimentError,match='CONFIG'):experiment.LiveGitHubTransport(token)


def observation(cost=1):
    return {'cost':cost,'wire_bytes':10,'decoded_body_bytes':20,'decoded_json_bytes':25,
            'transport_elapsed_ms':1,'http_request_started':True,'graphql_errors':False}


def test_repeated_order_balancing_summary_counts_and_unknown_values():
    orders=[]
    def compare(repository,measurement,*,strategies):
        orders.append((repository,strategies))
        return {'repository':repository,'status':'comparable','shared_reference_requests':[observation()],
            'shared_reference_elapsed_ms':1,'canonical_content_equal':True,'content_path_branch_equal':False,'media_base_equal':False,
            'strategies':{name:{'status':'content','requests':[observation()],'elapsed_ms':2} for name in strategies}}
    trials=experiment.comparison_trials(['a/one','b/two'],2,None,compare)
    assert orders[0][1]==orders[3][1]==('aliases','metadata')
    assert orders[1][1]==orders[2][1]==('metadata','aliases')
    report=experiment.summarize_trials(trials);left=report['strategies']['aliases']
    assert left['graphql_calls_including_shared_reference']==8
    assert left['rate_cost_including_shared_reference']==8
    assert left['wire_bytes_including_shared_reference']==80
    assert left['successful_end_to_end_p50_ms']==left['successful_end_to_end_p95_ms']==3
    assert left['transport_p50_ms']==2
    assert report['canonical_content_equal_trials']==4 and report['content_path_branch_equal_trials']==0
    trials[0]['strategies']['aliases']['status']='failed'
    trials[0]['strategies']['aliases']['requests'][0]['cost']=None
    trials[1].pop('strategies')
    left=experiment.summarize_trials(trials)['strategies']['aliases']
    assert left['rate_cost_including_shared_reference'] is None
    assert left['failure_rate_of_attempted']==1/3 and left['not_run_trials']==1
    assert left['successful_latency_samples']==2


@pytest.mark.parametrize('repositories,repeats',[(['a/b']*6,2),(['a/b'],0),(['a/b'],True)])
def test_trial_bounds_prevent_comparison_calls(repositories,repeats):
    with pytest.raises(experiment.ReadmeExperimentError,match='BOUND'):
        experiment.comparison_trials(repositories,repeats,None,lambda *a,**k:pytest.fail('unbounded trial'))


def test_atomic_new_report_preserves_existing_evidence_and_temp_cleanup(tmp_path):
    path=tmp_path/'report.json';experiment.publish_new_report(path,{'result':'first'})
    before=path.read_bytes()
    with pytest.raises(FileExistsError):experiment.publish_new_report(path,{'result':'second'})
    assert path.read_bytes()==before and not list(tmp_path.glob('.readme-experiment-*'))
    assert path.stat().st_mode&0o077==0


def test_sample_regular_file_and_preallocation_bounds(tmp_path):
    path=tmp_path/'sample.json';path.write_text('{"repositories":["a/b"]}')
    assert experiment.read_sample_bytes(path)==path.read_bytes()
    link=tmp_path/'link';link.symlink_to(path)
    with pytest.raises((experiment.ReadmeExperimentError,OSError)):experiment.read_sample_bytes(link)
    large=tmp_path/'large'
    with large.open('wb') as stream:stream.truncate(65537)
    with pytest.raises(experiment.ReadmeExperimentError,match='BOUND'):experiment.read_sample_bytes(large)


def test_actual_owned_worker_process_exits_on_watchdog_without_leaving_child():
    import subprocess
    import sys
    from pathlib import Path
    module=Path(experiment.__file__).resolve()
    program=("import sys,time;sys.path.insert(0,sys.argv[1]);"
             "from readme_experiment import start_worker_watchdog;"
             "start_worker_watchdog(.05);time.sleep(10)")
    result=subprocess.run([sys.executable,'-B','-c',program,str(module.parent)],
        capture_output=True,text=True,timeout=3)
    assert result.returncode==124
    assert result.stdout==result.stderr==''
