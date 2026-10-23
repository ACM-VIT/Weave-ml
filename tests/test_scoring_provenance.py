from dataclasses import replace
import hashlib
import json
import math
import struct
import uuid

import pytest
from inference.feature_spec import FEATURE_ORDER
from inference.scoring_provenance import (
    ScoringContext, RepositorySnapshot, ScoredFrame, INPUT_BYTES,
    packet_for_frame, decode_owned_packet, seal_scoring_response,
)


def identifier(n): return str(uuid.UUID(int=n))


def raw_input(n):
    values = [float(i)/100 for i in range(778)]
    values[400] = n/100
    return struct.pack('<778f',*values)


def frame(n, ctx=None):
    ctx = context() if ctx is None else ctx
    return ScoredFrame(ctx.user_id,ctx.generation_id,RepositorySnapshot(repo_id=identifier(100+n),
        content_version='1',feature_version='0',content_hash='a'*64,source_hash='b'*64,embedding_version_index=0),
        raw_input(n),ctx)


def context():
    return ScoringContext(user_id=identifier(1),generation_id=identifier(2),scored_at='2026-10-23T00:00:00Z',feed_version='1',
        profile_version='2',feedback_version='3',feedback_state_revision='4',
        embedding_model='all-MiniLM-L6-v2',embedding_revision='c'*40,user_embedding_version=None,
        repository_embedding_versions=('repo-embedding-v2',),
        user_vector_sha256=hashlib.sha256(raw_input(1)[:1536]).hexdigest(),
        model_version='heavy-ranker-v2',model_sha256='d'*64,scaler_sha256='e'*64,manifest_sha256='f'*64,
        feature_spec_version='v2',feature_order_sha256=hashlib.sha256(json.dumps(FEATURE_ORDER,separators=(',',':')).encode()).hexdigest(),
        code_commit='a'*40)


def response(count):
    return {'schema_version':2,'generation_id':identifier(2),'user_id':identifier(1),'feed_version':1,
        'model_version':'heavy-ranker-v2','embedding_version':'repo-embedding-v2','embedding_versions':['repo-embedding-v2'],
        'context_status':{'cold_start_applied':False,'locale':'not_provided'},'served_ranker':'heavy',
        'ranker_applied':True,'heavy_ranker_selected':True,'fallback_code':None,'fallback_reason':None,
        'retrieval_mode':'personalized','items':[{'repo_id':identifier(100+n),'score':.5,'source':'semantic',
        'retrieval_scores':{'semantic':.7}} for n in range(count)]}


def test_actual_float32_bytes_roundtrip_and_private_values_do_not_enter_repr():
    item,ctx=frame(1),context()
    packet=packet_for_frame(item,ctx)
    raw=decode_owned_packet(packet,ctx,owner_user_id=ctx.user_id,generation_id=ctx.generation_id,repo_id=item.snapshot.repo_id)
    assert raw==item.input_bytes and len(raw)==INPUT_BYTES
    assert struct.unpack('<778f',raw)[400]==pytest.approx(.01)
    assert packet.input_base64 not in repr(packet)
    assert repr(item.input_bytes) not in repr(item)


def test_owner_generation_repository_and_version_tampering_fail_without_exposing_bytes():
    item,ctx=frame(1),context();packet=packet_for_frame(item,ctx)
    for key in ['owner_user_id','generation_id','repo_id']:
        selectors={'owner_user_id':ctx.user_id,'generation_id':ctx.generation_id,'repo_id':item.snapshot.repo_id}
        selectors[key]=identifier(999)
        with pytest.raises(ValueError,match='ownership'):decode_owned_packet(packet,ctx,**selectors)
    for changed in [ctx.model_copy(update={'profile_version':'3'}),ctx.model_copy(update={'feedback_version':'4'}),
                    ctx.model_copy(update={'model_sha256':'0'*64})]:
        with pytest.raises(ValueError,match='integrity'):
            decode_owned_packet(packet,changed,owner_user_id=ctx.user_id,generation_id=ctx.generation_id,repo_id=item.snapshot.repo_id)
    with pytest.raises(ValueError,match='ownership'):
        packet_for_frame(replace(item,user_id=identifier(999)),ctx)
    changed=packet.model_copy(update={'snapshot':packet.snapshot.model_copy(update={'content_version':'2'})})
    with pytest.raises(ValueError,match='integrity'):
        decode_owned_packet(changed,ctx,owner_user_id=ctx.user_id,generation_id=ctx.generation_id,repo_id=item.snapshot.repo_id)


def test_nonfinite_wrong_dimension_or_wrong_user_vector_is_not_reconstructed_or_coerced():
    item,ctx=frame(1),context()
    for raw in [b'',item.input_bytes[:-4],struct.pack('<f',math.nan)+item.input_bytes[4:],
                struct.pack('<f',0)+item.input_bytes[4:]]:
        # Last payload deliberately changes user bytes; original first value is0,
        # so use an actually different finite value for the ownership proof.
        if raw==item.input_bytes:raw=struct.pack('<f',42)+raw[4:]
        with pytest.raises(ValueError):packet_for_frame(replace(item,input_bytes=raw),ctx)
    with pytest.raises(ValueError,match='UTC'):
        ScoringContext.model_validate({**ctx.model_dump(),'scored_at':'2026-10-23T00:00:00'})
    with pytest.raises(ValueError,match='feature order'):
        ScoringContext.model_validate({**ctx.model_dump(),'feature_order_sha256':'0'*64})


def test_selected_inputs_are_bound_to_actual_scored_ids_and_base_score_api_is_unmodified():
    original=response(2);before=json.loads(json.dumps(original));ctx=context()
    wire=seal_scoring_response(original,ctx,(frame(0),frame(1),frame(2)))
    body=json.loads(wire)
    assert original==before
    assert [r['score'] for r in body['items']]==[r['score'] for r in before['items']]
    assert body['scoring_metadata_complete'] is False
    assert body['scoring_context']['user_embedding_version'] is None
    assert len(body['items'])==2 and all('scored_input' in row for row in body['items'])
    with pytest.raises(ValueError,match='unscored'):seal_scoring_response(original,ctx,(frame(0),))
    with pytest.raises(ValueError,match='Duplicate'):seal_scoring_response(original,ctx,(frame(0),frame(0)))
    with pytest.raises(ValueError,match='ownership'):
        seal_scoring_response(original,ctx,(frame(0),replace(frame(1),user_id=identifier(999))))


@pytest.mark.parametrize('character', ['🚀', '\x00'])
@pytest.mark.parametrize('opaque_hash', [False, True])
def test_exact_wire_budget_holds_for_every_allowed_output_size_including_max_versions_and_escaping(character, opaque_hash):
    ctx=context().model_copy(update={'model_version':character*512,'embedding_model':character*128,
        'feed_version':str(2**63-1),'profile_version':str(2**63-1),'feedback_version':str(2**63-1),'feedback_state_revision':str(2**63-1),
        'user_embedding_version':character*128,'scored_at':'2026-10-23T00:00:00.'+'1'*19+'Z',
        'repository_embedding_versions':tuple(character*127+str(i) for i in range(8))})
    frames=tuple(replace(frame(i,ctx),snapshot=frame(i,ctx).snapshot.model_copy(update={
        'content_hash':character*128 if opaque_hash else 'a'*64,
        'content_version':str(2**63-1),'feature_version':str(2**63-1),'embedding_version_index':i%8})) for i in range(150))
    sizes=[]
    for count in range(151):
        base=response(count)
        versions=sorted(set(ctx.repository_embedding_versions[i%8] for i in range(count)))
        base['context_status']['locale']='reserved_unused'
        base.update(feed_version=int(ctx.feed_version),model_version=ctx.model_version,embedding_versions=versions,
            embedding_version=versions[0] if len(versions)==1 else 'compatible-mixed:'+','.join(versions))
        for row in base['items']:
            row.update(source=character*128,score=-1.7976931348623157e308,
                retrieval_scores={chr(97+i)*32:-1.7976931348623157e308 for i in range(16)})
        expected={**base,'items':[{**row,'scored_input':packet_for_frame(frames[i],ctx).model_dump(mode='json')}
            for i,row in enumerate(base['items'])],'scoring_context':ctx.model_dump(mode='json'),
            'scoring_metadata_complete':ctx.user_embedding_version is not None}
        encoded=json.dumps(expected,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':')).encode()
        if len(encoded)>1_000_000:
            with pytest.raises(ValueError,match='byte budget'):seal_scoring_response(base,ctx,frames)
        else:
            assert seal_scoring_response(base,ctx,frames)==encoded
        sizes.append(len(encoded))
    assert sizes==sorted(sizes) and sizes[-1]>sizes[0]+900_000
    with pytest.raises(ValueError,match='bounds'):seal_scoring_response(response(151),context(),frames)


def test_oversize_envelope_fails_explicitly_without_trimming_scores_or_silently_dropping_packets():
    base=response(1);before=json.loads(json.dumps(base))
    wire=seal_scoring_response(base,context(),(frame(0),))
    assert seal_scoring_response(base,context(),(frame(0),),byte_limit=len(wire))==wire
    with pytest.raises(ValueError,match='byte budget'):
        seal_scoring_response(base,context(),(frame(0),),byte_limit=len(wire)-1)
    base['unbounded_extra']='x'*1_000_000
    with pytest.raises(ValueError,match='byte budget'):seal_scoring_response(base,context(),(frame(0),))
    assert response(1)==before


def test_exact_decimal_version_fields_reject_lossy_or_noncanonical_values():
    original=context().model_dump()
    for field in ('feed_version','profile_version','feedback_version','feedback_state_revision'):
        for value in (1,'01','-1',str(2**63),'1.0'):
            with pytest.raises(ValueError):
                ScoringContext.model_validate({**original,field:value})
    for field in ('feed_version','profile_version'):
        with pytest.raises(ValueError):ScoringContext.model_validate({**original,field:'0'})
    snapshot=frame(0).snapshot.model_dump()
    for field in ('content_version','feature_version'):
        for value in (1,'01','-1',str(2**63)):
            with pytest.raises(ValueError):RepositorySnapshot.model_validate({**snapshot,field:value})

@pytest.fixture(scope='module')
def actual_capture_ranker():
    from pathlib import Path
    import torch
    from inference.ranker_service import RankerService
    root=Path(__file__).resolve().parents[1]/'inference'
    ranker=RankerService(model_path=str(root/'heavy_ranker.pt'),scaler_path=str(root/'feature_scaler.json'),
        manifest_path=str(root/'model_manifest.json'),require_production_manifest=True)
    ranker.device=torch.device('cpu');ranker.model.to(ranker.device)
    return ranker


def actual_context(ranker,user_id=identifier(1),generation_id=identifier(2),user=None):
    import numpy as np
    if user is None:user=np.frombuffer(frame(0).input_bytes[:1536],dtype='<f4').copy()
    return context().model_copy(update={**ranker.loaded_artifact_sha256,'model_version':ranker.model_version,
        'embedding_model':ranker.embedding_model,'embedding_revision':ranker.embedding_model_revision,
        'repository_embedding_versions':(ranker.embedding_version,),
        'user_id':user_id,'generation_id':generation_id,
        'user_vector_sha256':hashlib.sha256(user.astype('<f4',copy=False).tobytes()).hexdigest()})


def actual_candidates(ranker,count=2):
    import numpy as np
    return [{'id':identifier(100+i),'embedding':np.full(384,.01+i/100,dtype=np.float32),
        'content_version':1,'feature_version':0,'content_hash':'a'*64,'source_hash':'b'*64,
        'embedding_version':ranker.embedding_version,'topics':['python'],'star_count':10**30,
        'readme_length':250,'doc_quality':.7} for i in range(count)]


def test_actual_trained_artifact_capture_matches_forward_tensor_and_exact_score_replay(actual_capture_ranker):
    import numpy as np
    import torch
    from inference.scoring_provenance import capture_scored_inputs
    ranker=actual_capture_ranker;user=np.frombuffer(frame(0).input_bytes[:1536],dtype='<f4').copy()
    candidates=actual_candidates(ranker);ctx=actual_context(ranker,user=user)
    baseline=ranker.score_batch_compact(user,['python'],candidates)
    forwarded=[]
    hook=ranker.model.register_forward_pre_hook(lambda _module,args:forwarded.append(args[0].detach().cpu().numpy().astype('<f4').tobytes()))
    try:
        with capture_scored_inputs(ctx) as capture:
            scored=ranker.score_batch_compact(user,['python'],candidates)
            assert scored==baseline
            frames=capture.frames
            assert b''.join(row.input_bytes for row in frames)==forwarded[0]
            assert len(frames)==2
            replay=np.frombuffer(forwarded[0],dtype='<f4').reshape(2,778).copy()
            assert ranker._rank_results(torch.from_numpy(replay),candidates)==scored
            # The extreme dense star count is captured after scaler+clip, not raw.
            from inference.ranker_service import STANDARDIZED_FEATURE_CLIP
            assert abs(replay[0,771])<=STANDARDIZED_FEATURE_CLIP
            candidates[0]['embedding'][:]=99;user[:]=99
            packet=packet_for_frame(frames[0],ctx)
            assert decode_owned_packet(packet,ctx,owner_user_id=ctx.user_id,generation_id=ctx.generation_id,
                repo_id=frames[0].snapshot.repo_id)==frames[0].input_bytes
        assert capture.frames==()
        assert len(forwarded)==2
    finally:hook.remove()


@pytest.mark.parametrize('version_overlay',[{}, {'feature_version':None}, {'feature_version':0}],
    ids=['first-index','legacy-null','explicit-zero'])
def test_capture_accepts_first_index_payload_without_feature_version(actual_capture_ranker,version_overlay):
    import numpy as np
    from embedding.qdrant_store import QdrantRepositoryStore
    from embedding.repository_embedding import build_vector_payload, RepositoryEmbeddingConfig
    from inference.scoring_provenance import capture_scored_inputs
    ranker=actual_capture_ranker
    vector=np.zeros(384,dtype=np.float32);vector[0]=1
    payload=build_vector_payload({'full_name':'owner/first-index','content_version':1,
        'content_hash':'first-index-v1','topics':['python']},repo_id=identifier(100),
        final_embedding=vector.tolist(),readme_chunks=1,source_hash='b'*64,
        config=RepositoryEmbeddingConfig(version=ranker.embedding_version))
    assert 'feature_version' not in payload
    candidate={**payload,'id':payload['repo_id'],'embedding':vector,**version_overlay}
    # Indexing does not invent a refresh version; the canonical store treats
    # a first-index payload with no refresh metadata as version zero.
    assert QdrantRepositoryStore._stored_version(candidate,'feature_version')==0
    user=np.frombuffer(frame(0).input_bytes[:1536],dtype='<f4').copy()
    ctx=actual_context(ranker,user=user)
    baseline=ranker.score_batch_compact(user,['python'],[candidate])
    with capture_scored_inputs(ctx) as capture:
        assert ranker.score_batch_compact(user,['python'],[candidate])==baseline
        captured=capture.frames[0]
        assert captured.snapshot.content_version=='1'
        assert captured.snapshot.feature_version=='0'
        packet=packet_for_frame(captured,ctx)
        assert decode_owned_packet(packet,ctx,owner_user_id=ctx.user_id,
            generation_id=ctx.generation_id,repo_id=candidate['id'])==captured.input_bytes
    assert candidate.get('feature_version')==version_overlay.get('feature_version')
    assert ('feature_version' in candidate)==('feature_version' in version_overlay)


@pytest.mark.parametrize('invalid_version',[False,-1,1.5,'01'])
def test_capture_does_not_default_malformed_supplied_refresh_versions(actual_capture_ranker,invalid_version):
    import numpy as np
    from inference.scoring_provenance import capture_scored_inputs
    ranker=actual_capture_ranker;user=np.zeros(384,dtype=np.float32)
    candidate=actual_candidates(ranker,1)[0];candidate['feature_version']=invalid_version
    with capture_scored_inputs(actual_context(ranker,user=user)) as capture:
        with pytest.raises(ValueError):ranker.score_batch_compact(user,[],[candidate])
        assert capture.frames==()


def test_capture_keeps_content_version_required(actual_capture_ranker):
    import numpy as np
    from inference.scoring_provenance import capture_scored_inputs
    ranker=actual_capture_ranker;user=np.zeros(384,dtype=np.float32)
    candidate=actual_candidates(ranker,1)[0];del candidate['content_version']
    with capture_scored_inputs(actual_context(ranker,user=user)) as capture:
        with pytest.raises(KeyError,match='content_version'):ranker.score_batch_compact(user,[],[candidate])
        assert capture.frames==()


def test_concurrent_actual_ranker_scopes_remain_owner_isolated(actual_capture_ranker):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from inference.scoring_provenance import capture_scored_inputs
    ranker=actual_capture_ranker;barrier=Barrier(2)
    def run(owner):
        user=np.full(384,owner/100,dtype=np.float32)
        ctx=actual_context(ranker,user_id=identifier(owner),generation_id=identifier(10+owner),user=user)
        with capture_scored_inputs(ctx) as capture:
            barrier.wait(timeout=5)
            ranker.score_batch_compact(user,[],actual_candidates(ranker,1))
            frame=capture.frames[0]
            assert frame.user_id==ctx.user_id and frame.generation_id==ctx.generation_id
            assert frame.input_bytes[:1536]==user.astype('<f4').tobytes()
        assert capture.frames==()
        return frame
    with ThreadPoolExecutor(max_workers=2) as pool:
        a,b=list(pool.map(run,[1,2]))
    assert a.user_id!=b.user_id and a.input_bytes!=b.input_bytes


def test_capture_rejects_false_artifacts_or_unknown_actual_metadata_and_resets_scope(actual_capture_ranker):
    import numpy as np
    from inference.scoring_provenance import capture_scored_inputs
    ranker=actual_capture_ranker;user=np.zeros(384,dtype=np.float32);ctx=actual_context(ranker,user=user)
    with pytest.raises(ValueError,match='actual loaded'):
        with capture_scored_inputs(ctx.model_copy(update={'model_sha256':'0'*64})):
            ranker.score_batch_compact(user,[],actual_candidates(ranker,1))
    candidates=actual_candidates(ranker,1);del candidates[0]['source_hash']
    with pytest.raises(KeyError):
        with capture_scored_inputs(ctx) as capture:
            ranker.score_batch_compact(user,[],candidates)
    assert capture.frames==()
    # No leaked failed scope changes the ordinary compact score API afterward.
    assert len(ranker.score_batch_compact(user,[],actual_candidates(ranker,1)))==1


def test_capture_is_atomic_bounded_and_never_coerces_other_input_types(actual_capture_ranker):
    import numpy as np
    from inference.scoring_provenance import capture_scored_inputs
    ranker=actual_capture_ranker;ctx=actual_context(ranker,user=np.zeros(384,dtype=np.float32))
    with capture_scored_inputs(ctx) as capture:
        with pytest.raises(ValueError,match='Nested'):
            with capture_scored_inputs(ctx):pass
        with pytest.raises(ValueError,match='float32'):
            capture.record(ranker,actual_candidates(ranker,1),np.zeros((1,778),dtype=np.float64))
        with pytest.raises(ValueError,match='bounded'):
            capture.record(ranker,actual_candidates(ranker,151),np.zeros((151,778),dtype=np.float32))
        candidates=actual_candidates(ranker,2);del candidates[1]['source_hash']
        with pytest.raises(KeyError):capture.record(ranker,candidates,np.zeros((2,778),dtype=np.float32))
        assert capture.frames==()


def test_copied_context_cannot_mutate_another_threads_capture_but_snapshots_remain_readable(actual_capture_ranker):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    from inference.scoring_provenance import capture_scored_inputs
    ranker=actual_capture_ranker;user=np.zeros(384,dtype=np.float32);ctx=actual_context(ranker,user=user)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with capture_scored_inputs(ctx) as capture:
            copied=copy_context()
            mutation=pool.submit(copied.run,ranker.score_batch_compact,user,[],actual_candidates(ranker,1))
            with pytest.raises(ValueError,match='execution thread'):
                mutation.result(timeout=5)
            assert capture.frames==(), 'Copied-context mutation must leave the owner collector untouched.'
            ranker.score_batch_compact(user,[],actual_candidates(ranker,1))
            snapshots=capture.frames
            packet=packet_for_frame(snapshots[0],ctx)
        assert capture.frames==()
        # Finalized tuples and their immutable bytes remain valid after scope
        # exit and can be checked by another thread with explicit owner selectors.
        decoded=pool.submit(decode_owned_packet,packet,ctx,owner_user_id=ctx.user_id,
            generation_id=ctx.generation_id,repo_id=snapshots[0].snapshot.repo_id).result(timeout=5)
        assert decoded==snapshots[0].input_bytes


@pytest.mark.parametrize('field,value', [
    ('profile_version','9'), ('feedback_version','8'), ('feedback_state_revision','7'),
    ('model_sha256','0'*64), ('scaler_sha256','1'*64), ('code_commit','b'*40),
])
def test_saved_frame_cannot_be_rebound_to_another_valid_scoring_context(field,value):
    ctx=context();item=frame(0,ctx)
    changed=ScoringContext.model_validate({**ctx.model_dump(),field:value})
    with pytest.raises(ValueError,match='context'):
        packet_for_frame(item,changed)
    with pytest.raises(ValueError,match='context'):
        seal_scoring_response(response(1),changed,(item,))
    assert decode_owned_packet(packet_for_frame(item,ctx),ctx,owner_user_id=ctx.user_id,
        generation_id=ctx.generation_id,repo_id=item.snapshot.repo_id)==item.input_bytes


def test_actual_captured_frame_keeps_capture_time_context(actual_capture_ranker):
    import numpy as np
    from inference.scoring_provenance import capture_scored_inputs
    ranker=actual_capture_ranker;user=np.zeros(384,dtype=np.float32)
    ctx=actual_context(ranker,user=user)
    with capture_scored_inputs(ctx) as capture:
        ranker.score_batch_compact(user,[],actual_candidates(ranker,1))
        item=capture.frames[0]
        assert item.context is ctx
    changed=ScoringContext.model_validate({**ctx.model_dump(),'profile_version':'9'})
    with pytest.raises(ValueError,match='context'):
        packet_for_frame(item,changed)
    assert packet_for_frame(item,ctx).snapshot.repo_id==item.snapshot.repo_id


def test_copied_context_on_owner_thread_cannot_record_after_scope_exit(actual_capture_ranker):
    import numpy as np
    from contextvars import copy_context
    from inference.scoring_provenance import capture_scored_inputs
    ranker=actual_capture_ranker;user=np.zeros(384,dtype=np.float32)
    ctx=actual_context(ranker,user=user)
    with capture_scored_inputs(ctx) as capture:
        copied=copy_context()
        ranker.score_batch_compact(user,[],actual_candidates(ranker,1))
        saved=capture.frames
    assert capture.frames==()
    with pytest.raises(ValueError,match='closed'):
        copied.run(ranker.score_batch_compact,user,[],actual_candidates(ranker,1))
    assert capture.frames==()
    assert packet_for_frame(saved[0],ctx).snapshot.repo_id==saved[0].snapshot.repo_id
    assert len(ranker.score_batch_compact(user,[],actual_candidates(ranker,1)))==1


@pytest.mark.parametrize('content_hash',['legacy-content-v1','A'*128,'\x00'*128])
def test_private_capture_preserves_authoritative_bounded_content_hash(actual_capture_ranker,content_hash):
    import numpy as np
    from inference.scoring_provenance import capture_scored_inputs
    ranker=actual_capture_ranker
    candidate=actual_candidates(ranker,1)[0]
    candidate['content_hash']=content_hash
    user=np.frombuffer(frame(0).input_bytes[:1536],dtype='<f4').copy()
    ctx=actual_context(ranker,user=user)
    baseline=ranker.score_batch_compact(user,['python'],[candidate])
    with capture_scored_inputs(ctx) as capture:
        assert ranker.score_batch_compact(user,['python'],[candidate])==baseline
        assert capture.frames[0].snapshot.content_hash==content_hash
        packet=packet_for_frame(capture.frames[0],ctx)
        assert packet.snapshot.content_hash==content_hash
        assert decode_owned_packet(packet,ctx,owner_user_id=ctx.user_id,
            generation_id=ctx.generation_id,repo_id=candidate['id'])==capture.frames[0].input_bytes


@pytest.mark.parametrize('content_hash', ['', 'x'*129, None, 123])
def test_content_hash_contract_keeps_strict_nonempty_bounded_strings(content_hash):
    snapshot=frame(0).snapshot.model_dump()
    snapshot['content_hash']=content_hash
    with pytest.raises(ValueError):
        RepositorySnapshot.model_validate(snapshot)
