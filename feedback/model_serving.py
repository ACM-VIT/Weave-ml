"""Owned applied point to one effective float32 recommendation input."""
from dataclasses import dataclass, field
from copy import deepcopy
import hashlib
import struct
import json
import uuid
from feedback import long_session as reducer

@dataclass(frozen=True)
class Context:
    user_id:str
    session_id:str
    epoch:int
    policy_sha256:str
    minimum_applied_operation_version:int
    required_profile_version:int
    as_of_ms:int

@dataclass(frozen=True)
class EffectiveInput:
    vector:tuple[float,...]
    float32_le:bytes
    effective_vector_sha256:str
    user_id:str
    session_id:str
    epoch:int
    profile_version:int
    applied_operation_version:int
    applied_feedback_version:int
    state_sha256:str
    head:str
    policy_sha256:str
    as_of_ms:int
    valid_until_ms:int
    explicit_negative_repo_ids:tuple[str,...]
    session_witness_sha256:str
    profile_payload:dict = field(repr=False, compare=False)

def owned_input(point,users,policy,context):
    generation_freshness_ms = policy.generation_freshness_ms
    # Caller passes the actual sole retrieval read. Own nested payload/vector
    # before validation and never re-query or inspect the caller's object.
    point=deepcopy(point)
    if not isinstance(context,Context):raise reducer.CompatibilityError('typed context required')
    for key in ('user_id','session_id'):
        value=getattr(context,key)
        if type(value) is not str or str(uuid.UUID(value))!=value:raise reducer.CompatibilityError('canonical owner/session required')
    for value,minimum in ((context.epoch,1),(context.minimum_applied_operation_version,1),
            (context.required_profile_version,1),(context.as_of_ms,0),(generation_freshness_ms,1)):
        if type(value) is not int or not minimum<=value<=2**63-1:raise reducer.CompatibilityError('bounded explicit context clock/revision required')
    if context.policy_sha256!=policy.sha256:raise reducer.CompatibilityError('context policy differs')
    snapshot=users.decode_point(point,context.user_id)
    state=snapshot.state
    if isinstance(state,reducer.BootstrapState):
        raise reducer.OperationGap('genuine applied profile basis is pending')
    if state.operation_version<context.minimum_applied_operation_version or state.baseline.content_version!=context.required_profile_version:
        raise reducer.OperationGap('required applied model/profile revision pending')
    if type(snapshot.head) is not str or len(snapshot.head)!=64 or any(c not in '0123456789abcdef' for c in snapshot.head):raise reducer.CompatibilityError('committed model head required')
    view=reducer.recommendation_view(state,policy,session_id=context.session_id,epoch=context.epoch,as_of_ms=context.as_of_ms)
    if view.session_status!='active' or view.epoch!=context.epoch:raise reducer.EpochConflict('requested live epoch is not applied')
    session=next(s for s in state.sessions if s.session_id==context.session_id)
    valid_until=min(context.as_of_ms+generation_freshness_ms,session.last_activity_ms+policy.idle_expiry_ms)
    if not context.as_of_ms<valid_until<=2**63-1:raise reducer.CompatibilityError('generation validity overflow/expiry')
    raw=struct.pack('<384f',*view.blended)
    vector=struct.unpack('<384f',raw)
    source_payload = json.loads(snapshot.complete)['legacy']['payload']
    profile_fields = ('skills','tech_stack','interests','topics','embedding_dim','embedding_model',
                      'embedding_model_revision','profile_version','last_feedback_version','feedback_state_revision','embedding_version')
    profile_payload = {key: source_payload[key] for key in profile_fields if key in source_payload}
    return EffectiveInput(vector,raw,hashlib.sha256(raw).hexdigest(),context.user_id,context.session_id,context.epoch,
        state.baseline.content_version,state.operation_version,state.feedback_version,reducer.state_sha256(state),snapshot.head,
        policy.sha256,context.as_of_ms,valid_until,view.explicit_negative_repo_ids,session.source_witness_sha256,profile_payload)


def input_identity(effective):
    """JSON-safe exact cursor/clock identity without exposing vector bytes."""
    from dataclasses import fields
    values = {item.name: getattr(effective, item.name) for item in fields(effective)
              if item.name not in {"vector", "float32_le", "profile_payload"}}
    for key, value in list(values.items()):
        if type(value) is int:
            values[key] = str(value)
    values['schema_version'] = 1
    return values
