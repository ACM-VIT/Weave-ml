"""Immutable model journal and complete-state CAS coordinator.

No configuration defaults, runtime wiring, resource creation or retention delete.
The caller owns renewed user locking, trusted operation admission and ACK/DLQ.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields, is_dataclass
import base64
import hashlib
import json
import uuid
from typing import Callable, Protocol

from feedback import long_session as reducer
from utils.deadline import check_deadline


class JournalError(RuntimeError): pass
class JournalConflict(JournalError): pass
class JournalMissing(JournalError): pass
class OriginalIdentityConflict(JournalConflict): pass
class CasConflict(JournalError): pass
class HistoryBudget(JournalError): pass

_TYPES = {cls.__name__: cls for cls in (reducer.EmbeddingIdentity, reducer.Policy,
    reducer.PinnedVector, reducer.Signal, reducer.RepositoryLedger, reducer.SessionLedger,
    reducer.State, reducer.Header, reducer.SessionBinding, reducer.Feedback,
    reducer.Profile, reducer.Begin, reducer.Reset, reducer.End, reducer.BootstrapState, reducer.WithdrawnProfile)}


def _pack(value):
    if is_dataclass(value):
        if type(value).__name__ not in _TYPES or _TYPES[type(value).__name__] is not type(value):
            raise JournalError('unknown journal dataclass')
        return {'type': type(value).__name__, 'fields': {f.name:_pack(getattr(value,f.name)) for f in fields(value)}}
    if type(value) is tuple: return {'tuple': [_pack(x) for x in value]}
    if type(value) in (str, int, float, bool) or value is None: return value
    raise JournalError('unsupported journal value')


def _unpack(value):
    if type(value) is not dict: return value
    if set(value)=={'tuple'} and type(value['tuple']) is list:
        return tuple(_unpack(x) for x in value['tuple'])
    if set(value)!={'type','fields'} or value['type'] not in _TYPES or type(value['fields']) is not dict:
        raise JournalError('invalid journal tagged value')
    cls=_TYPES[value['type']]
    if set(value['fields'])!={f.name for f in fields(cls)}: raise JournalError('journal fields differ from schema')
    return cls(**{k:_unpack(v) for k,v in value['fields'].items()})


def _canonical(value): return json.dumps(value,allow_nan=False,sort_keys=True,separators=(',',':')).encode()
def _sha(raw): return hashlib.sha256(raw).hexdigest()
def _uuid(value):
    if type(value) is not str or str(uuid.UUID(value))!=value: raise JournalError('noncanonical owner')
    return value

def _json(raw,limit):
    if type(raw) is not bytes or not 0<len(raw)<=limit: raise JournalError('journal record byte budget')
    def pairs(items):
        result={}
        for k,v in items:
            if k in result: raise JournalError('duplicate journal JSON key')
            result[k]=v
        return result
    try:
        value=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(JournalError('nonfinite journal JSON')))
        if _canonical(value)!=raw: raise JournalError('noncanonical journal encoding')
        return value
    except (ValueError,TypeError,RecursionError) as exc: raise JournalError('invalid journal JSON') from exc


class ImmutableRecords(Protocol):
    def get(self,key:str)->bytes|None: ...
    def put_once(self,key:str,raw:bytes)->bytes: ...


@dataclass(frozen=True)
class Limits:
    input_bytes:int
    record_bytes:int
    combined_user_bytes:int
    history_operations:int
    def __post_init__(self):
        for name,ceiling in [('input_bytes',128*1024),('record_bytes',2*1024*1024),
                             ('combined_user_bytes',1024*1024),('history_operations',100000)]:
            value=getattr(self,name)
            if type(value) is not int or not 0<value<=ceiling: raise JournalError('invalid explicit journal budget')
        if self.record_bytes < max(2*self.input_bytes+32768, self.combined_user_bytes+256):
            raise JournalError('journal record budget must include original input encoding, vector pins and complete state')


class Journal:
    def __init__(self,records:ImmutableRecords,limits:Limits): self.records,self.limits=records,limits
    def put(self,kind,value):
        if kind not in {'input','prepare','complete','checkpoint','status_index'}: raise JournalError('unknown record kind')
        raw=_canonical({'schema_version':1,'kind':kind,'value':value})
        _json(raw,self.limits.record_bytes)
        digest=_sha(raw); key='blob:'+digest
        if self.records.put_once(key,raw)!=raw: raise JournalConflict('content-addressed record differs')
        return digest
    def get(self,digest,kind):
        check_deadline()
        if type(digest) is not str or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise JournalError('invalid content address')
        raw=self.records.get('blob:'+digest)
        if raw is None: raise JournalMissing('referenced journal record missing')
        if _sha(raw)!=digest: raise JournalConflict('journal digest mismatch')
        body=_json(raw,self.limits.record_bytes)
        if set(body)!={'schema_version','kind','value'} or body['schema_version']!=1 or body['kind']!=kind:
            raise JournalError('journal record schema/kind mismatch')
        return body['value']
    def pin_input(self,header,kind,policy,source_envelope,resolve:Callable):
        if type(header) is not reducer.Header or kind not in {'feedback','profile','begin','reset','end'}: raise JournalError('invalid operation identity')
        if type(source_envelope) is not bytes or len(source_envelope)>self.limits.input_bytes or _sha(source_envelope)!=header.source_envelope_sha256:
            raise JournalConflict('original envelope bytes differ')
        alias='operation:'+_uuid(header.user_id)+':'+str(header.operation_version)
        identity={'header':_pack(header),'kind':kind,'policy_sha256':policy.sha256,
                  'source_base64':base64.b64encode(source_envelope).decode()}
        existing=self.records.get(alias)
        if existing is None:
            operation=resolve()
            if type(operation).__name__.lower()!=kind or operation.header!=header: raise JournalConflict('resolved operation identity differs')
            candidate=_canonical({'identity':identity,'operation':_pack(operation)})
            _json(candidate,self.limits.record_bytes)
            existing=self.records.put_once(alias,candidate)
        stored=_json(existing,self.limits.record_bytes)
        if set(stored)!={'identity','operation'} or stored['identity']!=identity:
            raise JournalConflict('operation version or immutable envelope conflict')
        operation=_unpack(stored['operation'])
        if operation.header!=header or type(operation).__name__.lower()!=kind:
            raise JournalConflict('stored operation identity differs')
        digest=self.put('input',stored)
        return operation,digest

    def pin_withdrawal(self, wire, policy):
        operation = reducer.WithdrawnProfile(wire.header, wire.profile_job_id, wire.profile_version,
            wire.allow_genesis, wire.revoked_at_ms, wire.revocation_witness_sha256, wire.withdrawal_sha256)
        raw = wire.original_json.encode('utf-8')
        if _sha(raw) != wire.withdrawal_sha256 or len(raw) > self.limits.input_bytes:
            raise JournalConflict('withdrawal attestation bytes differ')
        value = {'identity': {'header': _pack(wire.header), 'kind': 'profile_withdrawal',
            'policy_sha256': policy.sha256, 'withdrawal_base64': base64.b64encode(raw).decode()}, 'operation': _pack(operation)}
        candidate = _canonical(value)
        _json(candidate, self.limits.record_bytes)
        key = 'withdrawal:' + wire.header.user_id + ':' + str(wire.header.operation_version)
        stored = self.records.put_once(key, candidate)
        if stored != candidate:
            raise JournalConflict('withdrawal attestation identity differs')
        return operation, self.put('input', value)


@dataclass(frozen=True)
class Snapshot:
    user_id:str
    state:reducer.State | reducer.BootstrapState
    head:str|None
    # Exact canonical complete product-independent ML vector/payload bytes.
    # Includes legacy compatibility state and every CAS token, not a delta.
    complete:bytes
    def __post_init__(self):
        _uuid(self.user_id)
        if self.state.user_id!=self.user_id: raise JournalConflict('snapshot owner differs')


class _StatusIndex:
    """Immutable 16-operation blocks and a binary merge frontier.

    The frontier belongs to a committed prepare, never a mutable operation
    alias. Leaves name existing prepare hashes; branches authenticate children.
    A new block costs one record per 16 ancestors, plus amortized merge records.
    """
    BLOCK=16
    def __init__(self,journal,user_id,policy_sha256):
        self.journal,self.user_id,self.policy=journal,user_id,policy_sha256
    @staticmethod
    def _digest(value):
        if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise JournalConflict('status index digest differs')
        return value
    def empty(self,first=1):
        return {'schema_version':1,'first_version':first,'last_version':first-1,'peaks':[],'tail':[]}
    def validate(self,value,last):
        if (type(value) is not dict or set(value)!={'schema_version','first_version','last_version','peaks','tail'}
                or type(value['schema_version']) is not int or value['schema_version']!=1
                or type(value['first_version']) is not int or type(value['last_version']) is not int
                or not 1<=value['first_version']<=last+1 or value['last_version']!=last
                or type(value['peaks']) is not list or len(value['peaks'])>63
                or type(value['tail']) is not list or len(value['tail'])>=self.BLOCK):
            raise JournalConflict('status index frontier differs')
        cursor=value['first_version'];previous_height=63
        for peak in value['peaks']:
            if (type(peak) is not dict or set(peak)!={'first_version','height','head'}
                    or type(peak['height']) is not int or not 0<=peak['height']<previous_height
                    or type(peak['first_version']) is not int or peak['first_version']!=cursor):
                raise JournalConflict('status index peak differs')
            self._digest(peak['head']);previous_height=peak['height']
            cursor+=self.BLOCK*(1<<peak['height'])
            if cursor>last+1:raise JournalConflict('status index range differs')
        for head in value['tail']:self._digest(head)
        if cursor+len(value['tail'])!=last+1:raise JournalConflict('status index coverage differs')
        return value
    def _put(self,value,assert_lock):
        assert_lock();check_deadline()
        return self.journal.put('status_index',{'schema_version':1,'user_id':self.user_id,'policy_sha256':self.policy,**value})
    def append(self,value,version,head,assert_lock):
        self.validate(value,version-1);self._digest(head)
        result={**value,'last_version':version,'peaks':[dict(p) for p in value['peaks']],'tail':[*value['tail'],head]}
        if len(result['tail'])<self.BLOCK:return result
        first=version-self.BLOCK+1
        digest=self._put({'first_version':first,'height':0,'heads':result['tail']},assert_lock)
        result['tail']=[];result['peaks'].append({'first_version':first,'height':0,'head':digest})
        while len(result['peaks'])>1 and result['peaks'][-1]['height']==result['peaks'][-2]['height']:
            right=result['peaks'].pop();left=result['peaks'].pop();height=left['height']+1
            digest=self._put({'first_version':left['first_version'],'height':height,'left':left['head'],'right':right['head']},assert_lock)
            result['peaks'].append({'first_version':left['first_version'],'height':height,'head':digest})
        return self.validate(result,version)
    def lookup(self,value,last,version):
        self.validate(value,last)
        if version<value['first_version']:raise JournalMissing('operation predates indexed committed horizon')
        for peak in value['peaks']:
            first,height,head=peak['first_version'],peak['height'],peak['head']
            if not first<=version<first+self.BLOCK*(1<<height):continue
            while True:
                check_deadline();node=self.journal.get(head,'status_index')
                expected={'schema_version','user_id','policy_sha256','first_version','height'}|({'heads'} if height==0 else {'left','right'})
                if (type(node) is not dict or set(node)!=expected or type(node['schema_version']) is not int or node['schema_version']!=1
                        or node['user_id']!=self.user_id or node['policy_sha256']!=self.policy
                        or type(node['first_version']) is not int or node['first_version']!=first
                        or type(node['height']) is not int or node['height']!=height):
                    raise JournalConflict('status index node identity differs')
                if height==0:
                    if type(node['heads']) is not list or len(node['heads'])!=self.BLOCK:raise JournalConflict('status index block differs')
                    for digest in node['heads']:self._digest(digest)
                    return node['heads'][version-first]
                self._digest(node['left']);self._digest(node['right'])
                height-=1;middle=first+self.BLOCK*(1<<height)
                if version<middle:head=node['left']
                else:head=node['right'];first=middle
        tail_first=last-len(value['tail'])+1
        if not tail_first<=version<=last:raise JournalConflict('status index target differs')
        return value['tail'][version-tail_first]


def validate_committed_head(snapshot, journal):
    if snapshot.state.operation_version == 0:
        return _validate_committed_head(snapshot, journal, None, None)
    return _validate_committed_head(snapshot, journal, reducer.state_sha256(snapshot.state), json.loads(snapshot.complete))


def _validate_committed_head(snapshot, journal, state_digest, complete_body):
    if snapshot.state.operation_version == 0:
        if snapshot.head is not None:
            raise JournalConflict('genesis seed has an unexpected committed head')
        return
    row = journal.get(snapshot.head, 'prepare')
    complete_digest = _sha(_canonical({'schema_version': 1, 'kind': 'complete', 'value': complete_body}))
    if (row.get('user_id') != snapshot.user_id or row.get('operation_version') != snapshot.state.operation_version
            or row.get('state_sha256') != state_digest
            or row.get('complete_sha256') != complete_digest):
        raise JournalConflict('committed head does not bind the complete user point')
    return row


class AtomicUserStore(Protocol):
    def read(self,user_id:str)->Snapshot: ...
    def compare_and_set(self,expected:Snapshot,desired:Snapshot)->bool: ...


class Coordinator:
    def __init__(self,journal:Journal,users:AtomicUserStore,policy:reducer.Policy,*,reject_capacity=False,legacy_reject=None,dead_letter_prepare=None):
        if type(reject_capacity) is not bool or (reject_capacity and (not callable(legacy_reject) or not callable(dead_letter_prepare))):
            raise JournalError('explicit rejection preparation/DLQ callbacks required')
        self.journal,self.users,self.policy=journal,users,policy
        self.reject_capacity,self.legacy_reject,self.dead_letter_prepare=reject_capacity,legacy_reject,dead_letter_prepare
    def _validate_snapshot(self,snapshot):
        body=_json(snapshot.complete,self.journal.limits.combined_user_bytes)
        if set(body)!={'schema_version','user_id','state','legacy'} or body['schema_version']!=1 or body['user_id']!=snapshot.user_id or body['state']!=_pack(snapshot.state):
            raise JournalConflict('complete snapshot identity differs')
        if isinstance(snapshot.state, reducer.BootstrapState):
            reducer.check_bootstrap(snapshot.state, self.policy)
        else:
            reducer.recommendation_view(snapshot.state,self.policy,session_id=None,epoch=None,as_of_ms=snapshot.state.effective_ms)
        return validate_committed_head(snapshot, self.journal)
    def _index_before(self,snapshot,assert_lock):
        index=_StatusIndex(self.journal,snapshot.user_id,self.policy.sha256)
        version=snapshot.state.operation_version
        if version==0:return index.empty()
        row=self.journal.get(snapshot.head,'prepare')
        expected=reducer.state_sha256(snapshot.state)
        if row.get('user_id')!=snapshot.user_id or row.get('operation_version')!=version or row.get('state_sha256')!=expected:
            raise JournalConflict('index predecessor differs from committed head')
        if 'status_index' in row:
            return index.append(row['status_index'],version,snapshot.head,assert_lock)
        # One legacy conversion, confined to the existing proof horizon. It runs
        # in the renewed-lock worker before CAS; old read-only heads still work.
        entries=[];head=snapshot.head
        for _ in range(self.journal.limits.history_operations):
            assert_lock();check_deadline()
            if head is None:break
            if row.get('user_id')!=snapshot.user_id or row.get('operation_version')!=version or row.get('state_sha256')!=expected:
                raise JournalConflict('legacy index ancestry differs')
            entries.append((version,head));head,expected,version=row['previous_head'],row['previous_state_sha256'],version-1
            if head is None or len(entries)==self.journal.limits.history_operations:break
            row=self.journal.get(head,'prepare')
        result=index.empty(entries[-1][0])
        for version,head in reversed(entries):result=index.append(result,version,head,assert_lock)
        return result
    def _empty_prefix(self, header):
        if header.previous_operation_version != 0:
            raise reducer.OperationGap('missing prefix cannot be inferred from a cursor')
        seed = reducer.bootstrap_state(header.user_id, self.policy, accepted_ms=header.accepted_ms)
        legacy = {'vector': {}, 'payload': {'user_id': header.user_id, 'last_feedback_version': 0, 'feedback_state_revision': 0}}
        complete = _canonical({'schema_version': 1, 'user_id': header.user_id, 'state': _pack(seed), 'legacy': legacy})
        return Snapshot(header.user_id, seed, None, complete)

    def _checkpoint_snapshot(self, snapshot):
        self._validate_snapshot(snapshot)
        return self.journal.put('checkpoint', {'user_id': snapshot.user_id, 'head': snapshot.head,
            'state_sha256': reducer.state_sha256(snapshot.state), 'state': _pack(snapshot.state),
            'complete_base64': base64.b64encode(snapshot.complete).decode(), 'policy_sha256': self.policy.sha256})

    def _complete_for(self, state, legacy_bytes):
        limit = self.journal.limits.combined_user_bytes
        if type(legacy_bytes) is bytes and len(legacy_bytes) > limit:
            raise reducer.StateOverflow('combined compatibility state exceeds byte budget')
        legacy = _json(legacy_bytes, limit)
        complete = _canonical({'schema_version': 1, 'user_id': state.user_id, 'state': _pack(state), 'legacy': legacy})
        if len(complete) > limit:
            raise reducer.StateOverflow('combined complete state exceeds byte budget')
        _json(complete, limit)
        validator = getattr(self.users, 'validate_desired', None)
        if validator is not None:
            # A real SHA256 head always has exactly these 64 unescaped bytes.
            # It is not committed authority; this IO-free probe checks the SAME
            # full vector/payload/base64 encoding used by the final CAS.
            validator(Snapshot(state.user_id, state, '0' * 64, complete))
        return complete

    def _commit_transition(self, before, operation, input_sha, legacy, assert_lock, *, insert=False, disposition='applied'):
        after = reducer.reduce_operation(before.state, operation, self.policy)
        complete = self._complete_for(after, legacy)
        complete_sha = self.journal.put('complete', json.loads(complete))
        row = {'user_id': before.user_id, 'previous_head': before.head,
            'previous_state_sha256': reducer.state_sha256(before.state), 'operation_version': after.operation_version,
            'input_sha256': input_sha, 'disposition': disposition,
            'error_code': 'PROFILE_CONSENT_WITHDRAWN' if disposition == 'withdrawn_profile' else None,
            'state_sha256': reducer.state_sha256(after), 'complete_sha256': complete_sha}
        if insert:
            row['genesis_checkpoint_sha256'] = self._checkpoint_snapshot(before)
        row['status_index']=self._index_before(before,assert_lock)
        head = self.journal.put('prepare', row)
        if disposition == 'withdrawn_profile':
            self.dead_letter_prepare(operation.header, input_sha, head, 'PROFILE_CONSENT_WITHDRAWN')
        desired = Snapshot(before.user_id, after, head, complete)
        assert_lock()
        applied = self.users.insert_once(desired) if insert else self.users.compare_and_set(before, desired)
        if not applied:
            raise CasConflict('original/profile-withdrawal state race; retry committed proof')
        assert_lock()
        return disposition

    def initialize(self, header, source_envelope, resolve, legacy, assert_lock):
        operation, input_sha = self.journal.pin_input(header, 'profile', self.policy, source_envelope, resolve)
        if not isinstance(operation, reducer.Profile) or not operation.allow_genesis:
            raise JournalConflict('explicit genesis permission required')
        assert_lock()
        before = self.users.read_optional(header.user_id)
        insert = before is None
        if insert:
            before = self._empty_prefix(header)
        self._validate_snapshot(before)
        if header.operation_version <= before.state.operation_version:
            return self.original_status(before, header)['status']
        if not isinstance(before.state, reducer.BootstrapState):
            raise JournalConflict('existing genuine basis requires ordinary profile update')
        body = _json(legacy, self.journal.limits.combined_user_bytes)
        body['payload']['feedback_state_revision'] = json.loads(before.complete)['legacy']['payload']['feedback_state_revision'] + 1
        return self._commit_transition(before, operation, input_sha, _canonical(body), assert_lock, insert=insert)

    def apply_withdrawal(self, wire, assert_lock):
        operation, input_sha = self.journal.pin_withdrawal(wire, self.policy)
        assert_lock()
        before = self.users.read_optional(wire.header.user_id)
        insert = before is None
        if insert:
            if not wire.allow_genesis:
                raise JournalMissing('activated/legacy user cannot acquire a bootstrap prefix')
            before = self._empty_prefix(wire.header)
        self._validate_snapshot(before)
        if wire.header.operation_version <= before.state.operation_version:
            result = self.original_status(before, wire.header)
            committed = result['operation']
            if isinstance(committed, reducer.Profile):
                source = self.journal.get(result['row']['input_sha256'], 'input')
                envelope = json.loads(base64.b64decode(source['identity']['source_base64'], validate=True))
                payload = envelope['operation']
                if (payload['job_id'] != wire.profile_job_id or int(payload['profile_version']) != wire.profile_version
                        or payload['allow_genesis'] != wire.allow_genesis):
                    raise OriginalIdentityConflict('withdrawal names a different original profile')
                return 'already_applied' if result['status'] == 'applied' else 'already_rejected'
            if not isinstance(committed, reducer.WithdrawnProfile) or committed.withdrawal_sha256 != wire.withdrawal_sha256:
                raise JournalConflict('committed withdrawal differs')
            return 'withdrawn_duplicate'
        legacy = self.legacy_reject(before.complete, operation)
        return self._commit_transition(before, operation, input_sha, legacy, assert_lock,
                                       insert=insert, disposition='withdrawn_profile')

    def original_status(self, snapshot, header):
        return self.status_for_identity(snapshot, operation_id=header.operation_id,
            operation_version=header.operation_version, source_sha256=header.source_envelope_sha256, expected_header=header)

    def status_for_identity(self, snapshot, *, operation_id, operation_version, source_sha256, expected_header=None):
        current=self._validate_snapshot(snapshot)
        if snapshot.state.operation_version-operation_version>=self.journal.limits.history_operations:
            raise HistoryBudget('bounded original-operation proof exhausted')
        def result(head,row):
            frozen=self.journal.get(row['input_sha256'],'input');operation=_unpack(frozen['operation'])
            if (operation.header.operation_id!=operation_id or operation.header.user_id!=snapshot.user_id
                    or operation.header.operation_version!=operation_version or operation.header.source_envelope_sha256!=source_sha256
                    or (expected_header is not None and operation.header!=expected_header)):
                raise OriginalIdentityConflict('committed original UUID/version/source bytes differ')
            if row.get('disposition') not in {'applied','withdrawn_profile','rejected'}:
                raise JournalConflict('committed disposition differs')
            check_deadline()
            return {'status':'withdrawn_profile' if row['disposition']=='withdrawn_profile' else
                ('rejected_capacity' if row.get('error_code')=='MODEL_STATE_CAPACITY' else 'rejected_session_dependency')
                if row['disposition']=='rejected' else 'applied','operation_head':head,'operation':operation,'row':row}
        head=snapshot.head;expected=reducer.state_sha256(snapshot.state);version=snapshot.state.operation_version
        for _ in range(self.journal.limits.history_operations):
            check_deadline()
            if head is None:raise JournalMissing('original operation predates committed proof horizon')
            row=current if head==snapshot.head else self.journal.get(head,'prepare')
            if row.get('user_id')!=snapshot.user_id or row.get('state_sha256')!=expected or row.get('operation_version')!=version:
                raise JournalConflict('committed original-operation chain differs')
            if version==operation_version:return result(head,row)
            if version<operation_version:raise reducer.OperationGap('requested original operation is not applied')
            if head==snapshot.head and 'status_index' in row:
                index=_StatusIndex(self.journal,snapshot.user_id,self.policy.sha256)
                indexed_head=index.lookup(row['status_index'],version-1,operation_version)
                indexed=self.journal.get(indexed_head,'prepare')
                if indexed.get('user_id')!=snapshot.user_id or indexed.get('operation_version')!=operation_version:
                    raise JournalConflict('indexed committed operation differs')
                return result(indexed_head,indexed)
            head,expected,version=row['previous_head'],row['previous_state_sha256'],version-1
        raise HistoryBudget('bounded original-operation proof exhausted')

    def apply(self,header,kind,source_envelope,resolve,legacy_prepare,assert_lock):
        # Pin the derived input once. A stale CAS retry never fetches today's
        # repository/profile vector or rebuilds the original outbox envelope.
        operation,input_sha=self.journal.pin_input(header,kind,self.policy,source_envelope,resolve)
        assert_lock()
        before=self.users.read(header.user_id)
        self._validate_snapshot(before)
        if header.operation_version<=before.state.operation_version:
            committed=self._prove_committed(before,input_sha,header.operation_version)
            return 'rejected_duplicate' if committed.get('disposition')=='rejected' else 'duplicate'
        disposition = 'applied'
        error_code = None
        causal_head = None
        try:
            after_state = reducer.reduce_operation(before.state, operation, self.policy)
            complete = self._complete_for(after_state, legacy_prepare(before.complete, operation))
        except reducer.StateOverflow:
            if not self.reject_capacity:
                raise
            after_state = reducer.reject_capacity(before.state, operation, self.policy)
            # This rejection is checked against the very same complete/persisted
            # byte bound. If it cannot fit, no prepare/DLQ/CAS/ACK is authorized.
            complete = self._complete_for(after_state, self.legacy_reject(before.complete, operation))
            disposition = 'rejected'
            error_code = 'MODEL_STATE_CAPACITY'
        except reducer.EpochConflict:
            if not self.reject_capacity:
                raise
            # A missing/gapped/corrupt session remains retryable. Only committed
            # rejected ancestry for this exact epoch may terminalize a dependent.
            causal_head, control = self._rejected_session_control(before, operation)
            after_state = reducer.reject_session_dependency(before.state, operation, self.policy, control)
            complete = self._complete_for(after_state, self.legacy_reject(before.complete, operation))
            disposition = 'rejected'
            error_code = 'MODEL_SESSION_DEPENDENCY_REJECTED'
        complete_sha=self.journal.put('complete',json.loads(complete))
        prepared={'user_id':header.user_id,'previous_head':before.head,
                  'previous_state_sha256':reducer.state_sha256(before.state),
                  'operation_version':header.operation_version,'input_sha256':input_sha,
                  'disposition':disposition,'error_code':error_code,
                  'state_sha256':reducer.state_sha256(after_state),'complete_sha256':complete_sha}
        if causal_head is not None:
            prepared['causal_prepare_sha256'] = causal_head
        prepared['status_index']=self._index_before(before,assert_lock)
        head=self.journal.put('prepare',prepared)
        desired=Snapshot(header.user_id,after_state,head,complete)
        if disposition=='rejected':
            # This is explicitly a prepared rejection, not an applied outcome.
            # A crash leaves an orphan named by immutable prepare SHA. Only the
            # committed user head proves that its cursor/disposition took effect.
            self.dead_letter_prepare(header,input_sha,head,error_code)
        assert_lock()
        if not self.users.compare_and_set(before,desired): raise CasConflict('complete user state changed; retry pinned input')
        # No separate applied marker is written. The atomic point's head is
        # the proof. Caller ACKs only after this return and its lock check.
        assert_lock()
        return disposition

    def _rejected_session_control(self, snapshot, operation):
        if isinstance(operation, reducer.Feedback):
            session_id = operation.session.session_id if operation.session is not None else None
        elif isinstance(operation, (reducer.Reset, reducer.End)):
            session_id = operation.session_id
        else:
            session_id = None
        if session_id is None:
            raise reducer.EpochConflict('no provable rejected session dependency')
        head, expected, version = snapshot.head, reducer.state_sha256(snapshot.state), snapshot.state.operation_version
        controls = []
        for _ in range(self.journal.limits.history_operations):
            if head is None:
                raise JournalMissing('rejected Begin is outside committed proof horizon')
            row = self.journal.get(head, 'prepare')
            if (row.get('user_id') != snapshot.user_id or row.get('state_sha256') != expected or
                    row.get('operation_version') != version):
                raise JournalConflict('rejected session committed ancestry differs')
            frozen = self.journal.get(row['input_sha256'], 'input')
            control = _unpack(frozen['operation'])
            if control.header.user_id != snapshot.user_id or control.header.operation_version != version:
                raise JournalConflict('rejected session input owner/version differs')
            if isinstance(control, (reducer.Begin, reducer.Reset, reducer.End)) and control.session_id == session_id:
                if isinstance(control, reducer.Begin):
                    if row.get('disposition') != 'rejected' or row.get('error_code') != 'MODEL_STATE_CAPACITY':
                        raise reducer.EpochConflict('session Begin was not a committed capacity rejection')
                    proof_head, proof = head, control
                    for next_head, next_row, next_control in reversed(controls):
                        if (next_row.get('disposition') != 'rejected' or
                                next_row.get('error_code') != 'MODEL_SESSION_DEPENDENCY_REJECTED' or
                                next_row.get('causal_prepare_sha256') != proof_head):
                            raise reducer.EpochConflict('session control is not causally rejected')
                        epoch = 1 if isinstance(proof, reducer.Begin) else proof.new_epoch
                        started = proof.epoch_started_ms if isinstance(proof, reducer.Begin) else proof.header.accepted_ms
                        if isinstance(next_control, reducer.Reset):
                            if next_control.old_epoch != epoch or next_control.new_epoch != epoch + 1:
                                raise JournalConflict('rejected Reset ancestry differs')
                            proof_head, proof = next_head, next_control
                        else:
                            if next_control.epoch != epoch or next_control.epoch_started_ms != started:
                                raise JournalConflict('rejected End ancestry differs')
                            raise reducer.EpochConflict('rejected session was already retired')
                    return proof_head, proof
                controls.append((head, row, control))
            head, expected, version = row['previous_head'], row['previous_state_sha256'], version - 1
        raise HistoryBudget('bounded rejected session proof exhausted')
    def _prove_committed(self,snapshot,input_sha,version):
        head=snapshot.head; expected=reducer.state_sha256(snapshot.state); expected_version=snapshot.state.operation_version
        for _ in range(self.journal.limits.history_operations):
            if head is None: raise JournalMissing('operation predates available committed checkpoint')
            row=self.journal.get(head,'prepare')
            if row['user_id']!=snapshot.user_id or row['state_sha256']!=expected or row['operation_version']!=expected_version: raise JournalConflict('committed chain owner/state differs')
            if row['operation_version']==version:
                if row['input_sha256']!=input_sha: raise JournalConflict('committed operation differs')
                return row
            if row['operation_version']<version: raise JournalConflict('committed operation version missing')
            head,expected=row['previous_head'],row['previous_state_sha256']; expected_version-=1
        raise HistoryBudget('bounded committed history proof exhausted')
    def rejection_status(self, prepare_sha):
        """A DLQ prepare has no authority until reachable from the user head."""
        prepared = self.journal.get(prepare_sha, 'prepare')
        if prepared.get('disposition') not in {'rejected', 'withdrawn_profile'}:
            raise JournalConflict('requested prepare is not a rejection')
        snapshot = self.users.read(prepared['user_id'])
        self._validate_snapshot(snapshot)
        if snapshot.state.operation_version < prepared['operation_version']:
            return 'prepared_not_committed'
        committed = self._prove_committed(snapshot, prepared['input_sha256'], prepared['operation_version'])
        return ('committed_withdrawal' if prepared['disposition'] == 'withdrawn_profile' else 'committed_rejection') if committed == prepared else 'prepared_not_committed'

    def checkpoint(self,user_id):
        return self._checkpoint_snapshot(self.users.read(_uuid(user_id)))
    def rebuild(self,checkpoint,expected_head,*,expected_user_id):
        seed=self.journal.get(checkpoint,'checkpoint')
        if seed['user_id']!=_uuid(expected_user_id) or seed['policy_sha256']!=self.policy.sha256: raise JournalConflict('checkpoint owner/policy differs')
        state=_unpack(seed['state']); complete=base64.b64decode(seed['complete_base64'],validate=True)
        if reducer.state_sha256(state)!=seed['state_sha256']: raise JournalConflict('checkpoint state differs')
        _json(complete,self.journal.limits.combined_user_bytes)
        self._validate_snapshot(Snapshot(expected_user_id,state,seed['head'],complete))
        rows=[]; head=expected_head
        while head!=seed['head']:
            if len(rows)>=self.journal.limits.history_operations: raise HistoryBudget('rebuild history budget')
            if head is None: raise JournalMissing('checkpoint is not reachable')
            row=self.journal.get(head,'prepare')
            if row['user_id']!=expected_user_id: raise JournalConflict('chain owner differs')
            rows.append(row); head=row['previous_head']
        for row in reversed(rows):
            if reducer.state_sha256(state)!=row['previous_state_sha256']: raise JournalConflict('replay predecessor differs')
            frozen=self.journal.get(row['input_sha256'],'input'); operation=_unpack(frozen['operation'])
            if row.get('disposition','applied')=='rejected':
                if row.get('error_code') == 'MODEL_STATE_CAPACITY':
                    state=reducer.reject_capacity(state,operation,self.policy)
                elif row.get('error_code') == 'MODEL_SESSION_DEPENDENCY_REJECTED':
                    proof_head, control = self._rejected_session_control(
                        Snapshot(expected_user_id, state, row['previous_head'], complete), operation)
                    if row.get('causal_prepare_sha256') != proof_head:
                        raise JournalConflict('replayed rejection causal proof differs')
                    state = reducer.reject_session_dependency(state, operation, self.policy, control)
                else:
                    raise JournalConflict('unknown recorded rejection')
            else:
                state=reducer.reduce_operation(state,operation,self.policy)
            if reducer.state_sha256(state)!=row['state_sha256']: raise JournalConflict('replayed state differs')
            body=self.journal.get(row['complete_sha256'],'complete'); complete=_canonical(body)
            if body['user_id']!=expected_user_id or _unpack(body['state'])!=state: raise JournalConflict('complete state differs')
            _json(complete,self.journal.limits.combined_user_bytes)
        if isinstance(state, reducer.BootstrapState):
            reducer.check_bootstrap(state, self.policy)
        else:
            reducer.recommendation_view(state,self.policy,session_id=None,epoch=None,as_of_ms=state.effective_ms)
        return Snapshot(expected_user_id,state,expected_head,complete)


class QdrantRecords:
    """Existing explicitly provisioned vectorless collection, no create/delete.

    A UUID point ID is a deterministic locator; the stored full key and full
    bytes are verified, so its shortened identity cannot hide a collision.
    """
    def __init__(self,client,models,*,collection,record_bytes):
        if not collection or type(record_bytes) is not int or not 0<record_bytes<=2*1024*1024:
            raise JournalError('explicit journal collection/budget required')
        self.client,self.models,self.collection,self.limit=client,models,collection,record_bytes
    def _id(self,key): return str(uuid.uuid5(uuid.UUID('f9013b12-7bbc-4f1a-b63e-26a0190a13d0'),key))
    def get(self,key):
        check_deadline()
        rows=self.client.retrieve(collection_name=self.collection,ids=[self._id(key)],with_payload=True,with_vectors=False)
        if not rows: return None
        if len(rows)!=1: raise JournalConflict('ambiguous journal locator')
        payload=rows[0].payload
        if type(payload) is not dict or set(payload)!={'key','raw_base64','sha256'} or payload['key']!=key:
            raise JournalConflict('journal locator collision/schema mismatch')
        raw=base64.b64decode(payload['raw_base64'],validate=True)
        if not 0<len(raw)<=self.limit or _sha(raw)!=payload['sha256']: raise JournalConflict('journal bytes/hash mismatch')
        return raw
    def put_once(self,key,raw):
        check_deadline()
        if type(key) is not str or len(key)>160 or type(raw) is not bytes or not 0<len(raw)<=self.limit:
            raise JournalError('immutable record budget')
        self.client.upsert(collection_name=self.collection,points=[self.models.PointStruct(
            id=self._id(key),vector={},payload={'key':key,'raw_base64':base64.b64encode(raw).decode(),'sha256':_sha(raw)})],
            wait=True,ordering=self.models.WriteOrdering.STRONG,update_mode=self.models.UpdateMode.INSERT_ONLY)
        found=self.get(key)
        if found is None: raise JournalMissing('write did not become readable')
        return found


    def delete_verified(self, key, expected):
        """Explicit retention-only delete; caller must publish an archive first.

        Runtime never calls this. Immutable INSERT_ONLY ownership is required:
        no concurrent admin overwrite or collection/Redis restore may run.
        """
        found = self.get(key)
        if found is None:
            return
        if found != expected:
            raise JournalConflict('retention deletion bytes differ')
        self.client.delete(collection_name=self.collection,
            points_selector=self.models.PointIdsList(points=[self._id(key)]),
            wait=True, ordering=self.models.WriteOrdering.STRONG)
        if self.get(key) is not None:
            raise JournalError('retention deletion not durably confirmed')


LEGACY_FLOAT_ENCODING = 'float64-hex-v1'


def _legacy_float_projection(value, path=()):
    """Encode only duplicated vector leaves; canonical complete bytes stay raw.

    Qdrant REST JSON can round tiny binary64 numbers by multiple ULPs. Hex
    strings preserve every bit without tolerating corruption or changing the
    vector/ledger used by model inference and replay.
    """
    if type(value) is dict:
        return {key: _legacy_float_projection(item, path + (key,)) for key, item in value.items()}
    if type(value) is list:
        return [_legacy_float_projection(item, path + (index,)) for index, item in enumerate(value)]
    vector_leaf = (len(path) == 2 and path[0] in {
        'preference_accumulator', 'profile_baseline_vector', 'feedback_latent_vector'} and type(path[-1]) is int)
    delta_leaf = (len(path) == 5 and path[0] == 'feedback_adjustments' and path[-2] == 'delta'
                  and type(path[-1]) is int)
    if type(value) is float and (vector_leaf or delta_leaf):
        import math
        if not math.isfinite(value):
            raise JournalConflict('nonfinite compatibility vector')
        return value.hex()
    return value


def _exact_payload_matches(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return set(actual) == set(expected) and all(_exact_payload_matches(actual[key], value)
                                                   for key, value in expected.items())
    if type(expected) is list:
        return len(actual) == len(expected) and all(_exact_payload_matches(a, b) for a, b in zip(actual, expected))
    if type(expected) is float:
        import math
        return math.isfinite(actual) and math.isfinite(expected) and actual.hex() == expected.hex()
    return actual == expected


@dataclass(frozen=True)
class _DecodedPoint:
    snapshot: Snapshot
    state_sha256: str
    base_view: reducer.View | None
    legacy_payload: dict


class QdrantAtomicUsers:
    """One point CAS commits legacy vector/payload and full new state/head.

    `snapshot_filter` is the existing embedding.qdrant_cas helper passed by the
    runtime. Collection must already have the canonical unnamed384 vector.
    Existing points require an explicitly installed migration checkpoint.
    """
    FIELDS=('profile_version','job_id','last_feedback_version','last_feedback_event_id','feedback_state_revision',
            'model_operation_version','model_journal_head','model_state_sha256','model_policy_sha256','model_basis_status','model_legacy_vector_encoding')
    INTERNAL={'model_complete_base64','model_operation_version','model_journal_head','model_state_sha256','model_policy_sha256','model_basis_status','model_legacy_vector_encoding'}
    def __init__(self,client,models,snapshot_filter,*,collection,combined_user_bytes,policy,journal=None,ensure_initialized=None,refuse_missing=None):
        if not collection or not 0<combined_user_bytes<=1024*1024: raise JournalError('explicit user collection/budget required')
        self.client,self.models,self.snapshot_filter=client,models,snapshot_filter
        self.collection,self.limit,self.policy=collection,combined_user_bytes,policy
        self.journal = journal
        self.ensure_initialized = ensure_initialized
        self.refuse_missing = refuse_missing
    def _point(self,snapshot):
        body,state=self._complete(snapshot.complete,snapshot.user_id)
        if state!=snapshot.state: raise JournalConflict('complete ledger differs')
        return self._point_body(snapshot,body,reducer.state_sha256(snapshot.state))

    def _complete(self,complete,user_id):
        body=_json(complete,self.limit)
        if type(body) is not dict or set(body)!={'schema_version','user_id','state','legacy'} or body['schema_version']!=1 or body['user_id']!=user_id:
            raise JournalConflict('complete point schema differs')
        return body,_unpack(body['state'])

    def _point_body(self,snapshot,body,state_digest):
        legacy=body['legacy']
        if type(legacy) is not dict or set(legacy)!={'payload','vector'} or type(legacy['payload']) is not dict:
            raise JournalError('legacy complete vector/payload required')
        if self.INTERNAL.intersection(legacy['payload']): raise JournalConflict('legacy callback overwrote journal fields')
        if (legacy['payload'].get('user_id') != snapshot.user_id
                or legacy['payload'].get('last_feedback_version') != snapshot.state.feedback_version):
            raise JournalConflict('compatibility owner/feedback identity differs from model state')
        vector = legacy['vector']
        if isinstance(snapshot.state, reducer.BootstrapState):
            reducer.check_bootstrap(snapshot.state, self.policy)
            if vector != {} or set(legacy['payload']) != {'user_id', 'last_feedback_version', 'feedback_state_revision'}:
                raise JournalConflict('uninitialized prefix contains an invented basis/profile')
        else:
            if legacy['payload'].get('profile_version') != snapshot.state.baseline.content_version:
                raise JournalConflict('compatibility profile version differs')
            if type(vector) is not list or len(vector)!=384 or any(type(x) not in (int,float) for x in vector):
                raise JournalError('legacy vector dimension/type')
            import math
            if any(not math.isfinite(x) for x in vector) or not math.isclose(math.sqrt(sum(x*x for x in vector)),1.,abs_tol=1e-5):
                raise JournalError('legacy vector finite/unit validation')
        payload=_legacy_float_projection(legacy['payload'])
        payload.update(model_legacy_vector_encoding=LEGACY_FLOAT_ENCODING,model_complete_base64=base64.b64encode(snapshot.complete).decode(),
            model_operation_version=snapshot.state.operation_version,model_journal_head=snapshot.head,
            model_state_sha256=state_digest,model_policy_sha256=self.policy.sha256,
            model_basis_status='awaiting_profile' if isinstance(snapshot.state,reducer.BootstrapState) else 'ready')
        # Count the ACTUAL persisted payload, including compatibility duplication
        # and base64 expansion, against the one shared user-state budget.
        if len(_canonical({'vector':vector,'payload':payload}))>self.limit: raise reducer.StateOverflow('combined persisted user point exceeds byte budget')
        return vector,payload
    def validate_desired(self, snapshot):
        self._point(snapshot)

    def read_optional(self, user_id):
        check_deadline()
        rows = self.client.retrieve(collection_name=self.collection, ids=[_uuid(user_id)], with_payload=True, with_vectors=True)
        if not rows:
            if self.refuse_missing is not None:
                self.refuse_missing(user_id)
            return None
        if len(rows) != 1:
            raise JournalConflict('ambiguous canonical user point')
        return self.decode_point(rows[0], user_id)

    def read(self, user_id):
        snapshot = self.read_optional(user_id)
        if snapshot is None:
            raise JournalMissing('canonical checkpointed user missing')
        return snapshot
    def decode_point(self,point,user_id):
        """Validate a supplied owned point; never query another user revision."""
        return self._decode_point(point,user_id).snapshot

    def decode_for_serving(self,point,user_id,policy):
        """Request-local validated components; never accept a caller's decoded state."""
        if policy.sha256!=self.policy.sha256: raise reducer.CompatibilityError('serving policy differs')
        return self._decode_point(point,user_id)

    def _decode_point(self,point,user_id):
        if str(point.id)!=_uuid(user_id): raise JournalConflict('canonical snapshot owner differs')
        payload=dict(point.payload or {})
        if 'model_complete_base64' not in payload: raise JournalMissing('explicit migration checkpoint required')
        if payload.get('model_legacy_vector_encoding') != LEGACY_FLOAT_ENCODING:
            raise JournalConflict('unknown compatibility vector encoding')
        complete=base64.b64decode(payload['model_complete_base64'],validate=True)
        body,state=self._complete(complete,user_id)
        snapshot=Snapshot(user_id,state,payload.get('model_journal_head'),complete)
        state_digest=reducer.state_sha256(state)
        vector,expected=self._point_body(snapshot,body,state_digest)
        if not _exact_payload_matches(payload, expected):
            raise JournalConflict('stored complete state/payload mismatch')
        actual = point.vector
        base_view=None
        if isinstance(state, reducer.BootstrapState):
            if actual != {}:
                raise JournalConflict('withdrawn-only prefix contains a real vector')
            reducer.check_bootstrap(state, self.policy)
        else:
            import math
            if type(actual) is not list or len(actual)!=384 or any(not math.isfinite(x) or not math.isclose(x,y,abs_tol=1e-6,rel_tol=1e-6) for x,y in zip(actual,vector)):
                raise JournalConflict('stored compatibility vector mismatch')
            base_view=reducer.recommendation_view(state,self.policy,session_id=None,epoch=None,as_of_ms=state.effective_ms)
        if self.journal is not None:
            _validate_committed_head(snapshot,self.journal,state_digest,body)
        if not isinstance(state, reducer.BootstrapState) and self.ensure_initialized is not None:
            # Negative guard must become durable before a basis is observable
            # as ready, not only before the first worker ACK.
            self.ensure_initialized(user_id)
        return _DecodedPoint(snapshot,state_digest,base_view,body['legacy']['payload'])
    def insert_once(self, desired):
        if desired.state.operation_version != 1 or desired.state.feedback_version != 0:
            raise JournalConflict('only initial profile may create a user')
        vector, payload = self._point(desired)
        self.client.upsert(collection_name=self.collection, points=[self.models.PointStruct(
            id=desired.user_id, vector=vector, payload=payload)], wait=True,
            ordering=self.models.WriteOrdering.STRONG, update_mode=self.models.UpdateMode.INSERT_ONLY)
        committed = self.read(desired.user_id)
        return committed.head == desired.head and committed.complete == desired.complete

    def compare_and_set(self,expected,desired):
        if expected.user_id!=desired.user_id or desired.state.operation_version!=expected.state.operation_version+1:
            raise JournalConflict('CAS owner/order differs')
        _,old_payload=self._point(expected); vector,payload=self._point(desired)
        if payload.get('last_feedback_version')!=desired.state.feedback_version:
            raise JournalConflict('legacy and model feedback cursors differ')
        if payload.get('feedback_state_revision')!=old_payload.get('feedback_state_revision',0)+1:
            raise JournalConflict('every new-policy write must fence legacy writers')
        self.client.upsert(collection_name=self.collection,points=[self.models.PointStruct(id=desired.user_id,vector=vector,payload=payload)],
            wait=True,ordering=self.models.WriteOrdering.STRONG,update_mode=self.models.UpdateMode.UPDATE_ONLY,
            update_filter=self.snapshot_filter(self.models,point_id=expected.user_id,payload=old_payload,fields=self.FIELDS))
        committed=self.read(desired.user_id)
        return committed.head==desired.head and committed.complete==desired.complete
