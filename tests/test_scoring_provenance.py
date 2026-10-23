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


def frame(n):
    values = [float(i)/100 for i in range(778)]
    values[400] = n/100
    return ScoredFrame(identifier(1),identifier(2),RepositorySnapshot(repo_id=identifier(100+n),
        content_version='1',feature_version='0',content_hash='a'*64,source_hash='b'*64,embedding_version_index=0),
        struct.pack('<778f',*values))


def context():
    return ScoringContext(user_id=identifier(1),generation_id=identifier(2),scored_at='2026-10-23T00:00:00Z',feed_version='1',
        profile_version='2',feedback_version='3',feedback_state_revision='4',
        embedding_model='all-MiniLM-L6-v2',embedding_revision='c'*40,user_embedding_version=None,
        repository_embedding_versions=('repo-embedding-v2',),
        user_vector_sha256=hashlib.sha256(frame(1).input_bytes[:1536]).hexdigest(),
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
def test_exact_wire_budget_holds_for_every_allowed_output_size_including_max_versions_and_escaping(character):
    ctx=context().model_copy(update={'model_version':character*512,'embedding_model':character*128,
        'feed_version':str(2**63-1),'profile_version':str(2**63-1),'feedback_version':str(2**63-1),'feedback_state_revision':str(2**63-1),
        'user_embedding_version':character*128,'scored_at':'2026-10-23T00:00:00.'+'1'*19+'Z',
        'repository_embedding_versions':tuple(character*127+str(i) for i in range(8))})
    frames=tuple(replace(frame(i),snapshot=frame(i).snapshot.model_copy(update={
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
        wire=seal_scoring_response(base,ctx,frames)
        assert len(wire)<=1_000_000
        assert len(wire)==len(json.dumps(json.loads(wire),ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':')).encode())
        sizes.append(len(wire))
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
