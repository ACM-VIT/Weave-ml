"""Pure long-session-v1 mathematics; no serving, persistence or activation.

Every numeric policy value is explicit. The caller must authenticate canonical
operation/session attribution, retain journal inputs, apply CAS/ordering and keep
legacy state atomically current. This module cannot prove those integrations.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from typing import Callable, Literal
from types import MappingProxyType
import uuid

VERSION = 'long-session-v1'
DIMENSIONS = 384
MAX_INTEGER = 2**63 - 1


class ModelError(ValueError): pass
class CompatibilityError(ModelError): pass
class OperationConflict(ModelError): pass
class OperationGap(ModelError): pass
class JournalProofRequired(ModelError): pass
class StateOverflow(ModelError): pass
class EpochConflict(ModelError): pass


def _integer(value, minimum=0):
    if type(value) is not int or not minimum <= value <= MAX_INTEGER:
        raise ModelError('expected exact bounded integer')
    return value


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ModelError('expected finite scalar')
    return float(value)


def _uuid(value):
    if type(value) is not str or str(uuid.UUID(value)) != value:
        raise ModelError('expected canonical UUID')
    return value


def _text(value, maximum=128):
    if type(value) is not str or not 1 <= len(value) <= maximum:
        raise ModelError('expected bounded nonempty text')
    return value


def _hex(value, size=64):
    if type(value) is not str or len(value) != size or any(x not in '0123456789abcdef' for x in value):
        raise ModelError('expected canonical digest/revision')
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def _zero(): return (0.0,) * DIMENSIONS


def _scale(vector, factor):
    result = tuple(float(x * factor) for x in vector)
    if any(not math.isfinite(x) for x in result):
        raise ModelError('vector arithmetic overflow')
    return result


def _sum(vectors):
    rows = tuple(vectors)
    try:
        return tuple(math.fsum(row[i] for row in rows) for i in range(DIMENSIONS))
    except (OverflowError, ValueError) as exc:
        raise ModelError('vector reduction overflow') from exc


def norm(vector):
    value = math.hypot(*vector)
    if not math.isfinite(value): raise ModelError('nonfinite vector norm')
    return value


def _cap(vector, cap):
    length = norm(vector)
    return _scale(vector, cap / length) if length > cap else vector


def _normalize(vector, fallback, epsilon):
    length = norm(vector)
    selected = fallback if length <= epsilon else vector
    length = norm(selected)
    if length <= epsilon: raise ModelError('declared normalization basis has zero norm')
    return _scale(selected, 1.0 / length)


@dataclass(frozen=True)
class EmbeddingIdentity:
    model: str
    revision: str
    dimensions: int

    def __post_init__(self):
        _text(self.model, 256); _hex(self.revision, 40)
        if self.dimensions != DIMENSIONS or type(self.dimensions) is not int:
            raise CompatibilityError('long-session-v1 requires384dimensions')


@dataclass(frozen=True)
class Policy:
    version: str
    embedding: EmbeddingIdentity
    repository_recipe: str
    baseline_recipe: str
    long_like: float
    long_dislike: float
    long_save: float
    session_like: float
    session_dislike: float
    session_save: float
    session_readme_open: float
    session_github_open: float
    session_share: float
    session_dwell: float
    long_repo_cap: float
    session_repo_cap: float
    session_total_cap: float
    session_blend: float
    generation_freshness_ms: int
    half_life_ms: int
    idle_expiry_ms: int
    minimum_dwell_ms: int
    saturated_dwell_ms: int
    refresh_idle_on_noop: bool
    normalization_epsilon: float
    unit_tolerance: float
    max_repositories: int
    max_sessions: int
    max_state_bytes: int
    max_replay_operations: int
    zero_norm_rule: str

    def __post_init__(self):
        if self.version != VERSION or self.zero_norm_rule != 'long-term-then-baseline':
            raise CompatibilityError('unsupported reducer or normalization rule')
        if not isinstance(self.embedding, EmbeddingIdentity): raise CompatibilityError('missing embedding identity')
        _text(self.repository_recipe); _text(self.baseline_recipe)
        weights = ('long_like','long_dislike','long_save','session_like','session_dislike','session_save',
                   'session_readme_open','session_github_open','session_share','session_dwell')
        for key in (*weights,'long_repo_cap','session_repo_cap','session_total_cap','session_blend',
                    'normalization_epsilon','unit_tolerance'):
            value = _number(getattr(self, key)); object.__setattr__(self, key, value)
            if value < 0: raise ModelError('negative policy magnitude')
        if not 0 <= self.session_blend <= 1: raise ModelError('invalid blend')
        if min(self.long_repo_cap,self.session_repo_cap,self.session_total_cap,self.normalization_epsilon,self.unit_tolerance) <= 0:
            raise ModelError('caps/tolerances must be positive')
        if self.normalization_epsilon >= 1 or self.unit_tolerance > 1e-3:
            raise ModelError('invalid normalization tolerances')
        for key in ('generation_freshness_ms','half_life_ms','idle_expiry_ms','minimum_dwell_ms','saturated_dwell_ms',
                    'max_repositories','max_sessions','max_state_bytes','max_replay_operations'):
            _integer(getattr(self,key),1)
        if self.minimum_dwell_ms < 3000 or self.saturated_dwell_ms < self.minimum_dwell_ms:
            raise ModelError('qualified dwell must retain the3000ms architecture floor')
        if self.max_repositories > 256 or self.max_sessions > 64 or self.max_state_bytes > 1024*1024 or self.max_replay_operations > 100000:
            raise ModelError('policy exceeds explicit implementation work/state ceilings')
        if type(self.refresh_idle_on_noop) is not bool: raise ModelError('idle refresh must be explicit boolean')

    @property
    def sha256(self): return _digest(asdict(self))


@dataclass(frozen=True)
class PinnedVector:
    subject_kind: Literal['profile','repository']
    subject_id: str
    content_version: int
    feature_version: int
    recipe: str
    embedding: EmbeddingIdentity
    vector: tuple[float, ...]
    sha256: str

    def __post_init__(self):
        if self.subject_kind not in ('profile','repository'): raise ModelError('invalid vector subject')
        _uuid(self.subject_id); _integer(self.content_version,1); _integer(self.feature_version); _text(self.recipe)
        if not isinstance(self.embedding,EmbeddingIdentity): raise CompatibilityError('invalid vector embedding identity')
        if type(self.vector) is not tuple or len(self.vector) != DIMENSIONS: raise CompatibilityError('vector dimension/immutability')
        values = tuple(_number(x) for x in self.vector); object.__setattr__(self,'vector',values)
        _hex(self.sha256)
        body = asdict(self); body.pop('sha256')
        if _digest(body) != self.sha256: raise CompatibilityError('pinned vector identity mismatch')

    @classmethod
    def pin(cls, subject_kind, subject_id, content_version, feature_version, recipe, embedding, vector):
        if not isinstance(embedding,EmbeddingIdentity):raise CompatibilityError('missing pinned embedding identity')
        if type(vector) not in (tuple,list) or len(vector)!=DIMENSIONS:raise CompatibilityError('pinned vector input must be bounded')
        body = dict(subject_kind=subject_kind,subject_id=subject_id,content_version=content_version,
                    feature_version=feature_version,recipe=recipe,embedding=asdict(embedding),vector=tuple(_number(x) for x in vector))
        return cls(subject_kind,subject_id,content_version,feature_version,recipe,embedding,body['vector'],_digest(body))


@dataclass(frozen=True)
class Signal:
    family: str
    action: str
    source: PinnedVector
    weight: float
    accepted_ms: int


@dataclass(frozen=True)
class RepositoryLedger:
    repo_id: str
    signals: tuple[Signal, ...]
    dwell_highwater: float


@dataclass(frozen=True)
class SessionLedger:
    session_id: str
    epoch: int
    sealed_through_epoch: int
    ended: bool
    started_ms: int
    anchor_ms: int
    last_activity_ms: int
    entries: tuple[RepositoryLedger, ...]
    source_witness_sha256: str


@dataclass(frozen=True)
class State:
    version: str
    user_id: str
    policy_sha256: str
    baseline: PinnedVector
    long_entries: tuple[RepositoryLedger, ...]
    sessions: tuple[SessionLedger, ...]
    operation_version: int
    feedback_version: int
    effective_ms: int
    last_operation_id: str | None
    last_operation_sha256: str | None


@dataclass(frozen=True)
class BootstrapState:
    """Proven withdrawn-only prefix. It contains no invented user vector."""
    version: str
    user_id: str
    policy_sha256: str
    operation_version: int
    feedback_version: int
    effective_ms: int
    last_operation_id: str | None
    last_operation_sha256: str | None


@dataclass(frozen=True)
class Header:
    operation_id: str
    user_id: str
    operation_version: int
    previous_operation_version: int
    accepted_ms: int
    source_envelope_sha256: str

    def __post_init__(self):
        _uuid(self.operation_id); _uuid(self.user_id); _integer(self.operation_version,1)
        _integer(self.previous_operation_version); _integer(self.accepted_ms); _hex(self.source_envelope_sha256)
        if self.operation_version != self.previous_operation_version+1: raise OperationGap('invalid operation predecessor')


@dataclass(frozen=True)
class SessionBinding:
    user_id: str
    session_id: str
    epoch: int
    epoch_started_ms: int
    active_at_acceptance: bool
    source_witness_sha256: str

    def __post_init__(self):
        _uuid(self.user_id); _uuid(self.session_id); _integer(self.epoch,1); _integer(self.epoch_started_ms)
        _hex(self.source_witness_sha256)
        if type(self.active_at_acceptance) is not bool: raise ModelError('invalid canonical session status')


@dataclass(frozen=True)
class Feedback:
    header: Header
    event_id: str
    feedback_version: int
    repo_id: str
    event_type: str
    repository: PinnedVector | None
    session: SessionBinding | None
    dwell_ms: int | None
    raw_client_timestamp_ms: int | None


@dataclass(frozen=True)
class Profile:
    header: Header
    baseline: PinnedVector
    allow_genesis: bool = False


@dataclass(frozen=True)
class WithdrawnProfile:
    header: Header
    profile_job_id: str
    profile_version: int
    allow_genesis: bool
    revoked_at_ms: int
    revocation_witness_sha256: str
    withdrawal_sha256: str


@dataclass(frozen=True)
class Begin:
    header: Header
    session_id: str
    epoch_started_ms: int
    source_witness_sha256: str


@dataclass(frozen=True)
class Reset:
    header: Header
    session_id: str
    old_epoch: int
    new_epoch: int
    source_witness_sha256: str


@dataclass(frozen=True)
class End:
    header: Header
    session_id: str
    epoch: int
    epoch_started_ms: int


Operation = Feedback | Profile | Begin | Reset | End | WithdrawnProfile
_ACTIONS = MappingProxyType({'like':('reaction',1),'dislike':('reaction',-1),'save':('save',1),
            'readme_open':('readme_open',1),'github_open':('github_open',1),'share':('share',1),'dwell':('dwell',1)})
_INVERSE = MappingProxyType({'unlike':('reaction','like'),'undislike':('reaction','dislike'),'unsave':('save','save')})


def operation_sha256(operation):
    return _digest({'kind':type(operation).__name__.lower(),'input':asdict(operation)})


def state_bytes(state): return canonical(asdict(state))
def state_sha256(state): return hashlib.sha256(state_bytes(state)).hexdigest()


def _check_vector(source, policy, kind, subject):
    if not isinstance(source,PinnedVector) or source.embedding != policy.embedding or source.subject_kind != kind or source.subject_id != subject:
        raise CompatibilityError('incompatible pinned vector subject/model/revision')
    recipe = policy.baseline_recipe if kind == 'profile' else policy.repository_recipe
    if kind=='profile' and source.feature_version!=0:raise CompatibilityError('baseline has repository feature revision')
    if source.recipe != recipe or abs(norm(source.vector)-1) > policy.unit_tolerance:
        raise CompatibilityError('incompatible recipe or non-unit source vector')


def _contribution(entry): return _sum(_scale(signal.source.vector,signal.weight) for signal in entry.signals)


def _active_anchor(entries,default):
    return max((signal.accepted_ms for entry in entries for signal in entry.signals if signal.weight!=0),default=default)


def _session_repo_at(entry,as_of,policy):
    nonzero=tuple(signal for signal in entry.signals if signal.weight!=0)
    if not nonzero:return _zero()
    anchor=max(signal.accepted_ms for signal in nonzero)
    raw=_sum(_scale(signal.source.vector,signal.weight*_decay(anchor-signal.accepted_ms,policy)) for signal in nonzero)
    return _scale(_cap(raw,policy.session_repo_cap),_decay(as_of-anchor,policy))


def _session_at(session,as_of,policy):
    # Cap at the latest currently active source anchor, then decay magnitude.
    # Undo can move that anchor back; no earlier family's weight is overwritten.
    raw=_sum(_session_repo_at(entry,session.anchor_ms,policy) for entry in session.entries)
    return _scale(_cap(raw,policy.session_total_cap),_decay(as_of-session.anchor_ms,policy))

def _check_state(state,policy):
    if state.version != VERSION or state.policy_sha256 != policy.sha256: raise CompatibilityError('state/policy version mismatch')
    _uuid(state.user_id); _check_vector(state.baseline,policy,'profile',state.user_id)
    for key in ('operation_version','feedback_version','effective_ms'): _integer(getattr(state,key))
    if state.feedback_version > state.operation_version: raise ModelError('feedback cursor exceeds operation cursor')
    if state.operation_version:
        _uuid(state.last_operation_id); _hex(state.last_operation_sha256)
    elif state.last_operation_id is not None or state.last_operation_sha256 is not None: raise ModelError('invalid empty operation witness')
    if type(state.long_entries) is not tuple or type(state.sessions) is not tuple: raise ModelError('mutable state container')
    if len(state.long_entries)>policy.max_repositories or len(state.sessions)>policy.max_sessions:raise StateOverflow('explicit ledger bound exceeded')
    repo_ids=set()
    def entries(values,durable):
        if len(values)>policy.max_repositories:raise StateOverflow('explicit repository ledger bound exceeded')
        if type(values) is not tuple or len({row.repo_id for row in values}) != len(values): raise ModelError('invalid repository ledger identity')
        if tuple(row.repo_id for row in values)!=tuple(sorted(row.repo_id for row in values)):raise ModelError('noncanonical repository ledger order')
        for entry in values:
            if not entry.signals:raise ModelError('empty repository family ledger')
            _uuid(entry.repo_id); repo_ids.add(entry.repo_id)
            if len(entry.signals)>(2 if durable else 6):raise ModelError('action family count exceeds fixed vocabulary')
            if durable and entry.dwell_highwater!=0:raise ModelError('durable ledger contains dwell witness')
            if type(entry.signals) is not tuple or len({x.family for x in entry.signals}) != len(entry.signals): raise ModelError('duplicate/mutable family ledger')
            if not 0 <= _number(entry.dwell_highwater) <= policy.session_dwell: raise ModelError('invalid dwell witness')
            if tuple(signal.family for signal in entry.signals)!=tuple(sorted(signal.family for signal in entry.signals)):raise ModelError('noncanonical family order')
            for signal in entry.signals:
                if signal.action not in _ACTIONS or _ACTIONS[signal.action][0] != signal.family: raise ModelError('invalid action family')
                if durable and signal.family not in ('reaction','save'): raise ModelError('transient durable family')
                _check_vector(signal.source,policy,'repository',entry.repo_id)
                value=_number(signal.weight);_integer(signal.accepted_ms)
                if signal.accepted_ms>state.effective_ms:raise ModelError('signal clock exceeds applied state')
                expected=(_ACTIONS[signal.action][1]*getattr(policy,('long_' if durable else 'session_')+signal.action))
                if signal.action=='dwell':expected=entry.dwell_highwater
                if value!=expected:raise ModelError('family weight differs from declared policy/highwater')
            if not durable and entry.dwell_highwater>0 and not any(s.action=='dwell' for s in entry.signals):raise ModelError('dwell witness without current family')
    entries(state.long_entries,True)
    if len({s.session_id for s in state.sessions}) != len(state.sessions): raise ModelError('duplicate session witness')
    if tuple(s.session_id for s in state.sessions)!=tuple(sorted(s.session_id for s in state.sessions)):raise ModelError('noncanonical session order')
    for session in state.sessions:
        _uuid(session.session_id); _integer(session.epoch,1); _integer(session.sealed_through_epoch); _hex(session.source_witness_sha256)
        if type(session.ended) is not bool or not 0 <= session.sealed_through_epoch <= session.epoch: raise ModelError('invalid sealed epoch')
        if session.ended != (session.sealed_through_epoch == session.epoch): raise ModelError('epoch/status mismatch')
        for stamp in (session.started_ms,session.anchor_ms,session.last_activity_ms): _integer(stamp)
        if not session.started_ms <= session.anchor_ms <= state.effective_ms or not session.started_ms <= session.last_activity_ms <= state.effective_ms:
            raise ModelError('invalid session clock')
        if session.ended and session.entries: raise ModelError('sealed epoch has transient entries')
        entries(session.entries,False)
        if session.anchor_ms!=_active_anchor(session.entries,session.started_ms):raise ModelError('session anchor differs from active source anchors')
        if any(signal.accepted_ms<session.started_ms for entry in session.entries for signal in entry.signals):raise ModelError('signal predates canonical epoch')
        if norm(_session_at(session,session.anchor_ms,policy))>policy.session_total_cap+policy.normalization_epsilon:raise ModelError('derived session cap violated')
    if len(repo_ids)>policy.max_repositories or len(state.sessions)>policy.max_sessions or len(state_bytes(state))>policy.max_state_bytes:
        raise StateOverflow('explicit model state bound exceeded; no entries evicted')
    return state


def check_bootstrap(state, policy):
    if type(state) is not BootstrapState or state.version != VERSION or state.policy_sha256 != policy.sha256:
        raise CompatibilityError('invalid uninitialized policy prefix')
    _uuid(state.user_id); _integer(state.operation_version); _integer(state.effective_ms)
    if state.feedback_version != 0 or type(state.feedback_version) is not int:
        raise ModelError('uninitialized prefix cannot contain applied feedback')
    if state.operation_version:
        _uuid(state.last_operation_id); _hex(state.last_operation_sha256)
    elif state.last_operation_id is not None or state.last_operation_sha256 is not None:
        raise ModelError('uninitialized seed has a false operation witness')
    return state


def bootstrap_state(user_id, policy, *, accepted_ms):
    return check_bootstrap(BootstrapState(VERSION, _uuid(user_id), policy.sha256, 0, 0,
        _integer(accepted_ms), None, None), policy)


def _check_withdrawal(operation):
    _uuid(operation.profile_job_id); _integer(operation.profile_version, 1)
    _integer(operation.revoked_at_ms); _hex(operation.revocation_witness_sha256); _hex(operation.withdrawal_sha256)
    if type(operation.allow_genesis) is not bool:
        raise ModelError('explicit genesis authority required')


def _reduce_bootstrap(state, operation, policy):
    check_bootstrap(state, policy)
    header = operation.header
    if header.user_id != state.user_id:
        raise CompatibilityError('uninitialized operation owner differs')
    fingerprint = operation_sha256(operation)
    if header.operation_version == state.operation_version:
        if header.operation_id == state.last_operation_id and fingerprint == state.last_operation_sha256:
            return state
        raise OperationConflict('uninitialized operation identity differs')
    if header.operation_version < state.operation_version:
        raise JournalProofRequired('older withdrawn-prefix duplicate needs committed proof')
    if header.previous_operation_version != state.operation_version:
        raise OperationGap('uninitialized predecessor is pending')
    now = max(state.effective_ms, header.accepted_ms)
    if isinstance(operation, WithdrawnProfile):
        _check_withdrawal(operation)
        if not operation.allow_genesis:
            raise CompatibilityError('an activated or legacy user cannot enter bootstrap')
        return replace(state, operation_version=header.operation_version,
            effective_ms=max(now, operation.revoked_at_ms), last_operation_id=header.operation_id,
            last_operation_sha256=fingerprint)
    if isinstance(operation, Profile) and operation.allow_genesis:
        _check_vector(operation.baseline, policy, 'profile', state.user_id)
        return _check_state(State(VERSION, state.user_id, policy.sha256, operation.baseline, (), (),
            header.operation_version, 0, now, header.operation_id, fingerprint), policy)
    raise OperationGap('a genuine profile basis must precede feedback and session controls')


def initial_state(user_id,baseline,policy,*,accepted_ms):
    state=State(VERSION,_uuid(user_id),policy.sha256,baseline,(),(),0,0,_integer(accepted_ms),None,None)
    return _check_state(state,policy)


def _sorted_entries(values): return tuple(values[key] for key in sorted(values))
def _signal_map(entry): return {row.family:row for row in entry.signals} if entry is not None else {}
def _entry(repo_id,signals,highwater=0.0): return RepositoryLedger(repo_id,tuple(signals[key] for key in sorted(signals)),float(highwater))
def _decay(milliseconds,policy): return math.exp(-math.log(2.0)*milliseconds/policy.half_life_ms)


def _seal(session):
    return replace(session,sealed_through_epoch=session.epoch,ended=True,anchor_ms=session.started_ms,entries=())


def _session_update(session,feedback,now,policy):
    if session.ended: return session
    entries={row.repo_id:row for row in session.entries};existing=entries.get(feedback.repo_id)
    signals=_signal_map(existing);action=feedback.event_type
    meaningful=False;new_signal=None;remove=None;highwater=existing.dwell_highwater if existing else 0.0
    if action in _INVERSE:
        family,target=_INVERSE[action]
        if family in signals and signals[family].action==target: remove=family;meaningful=True
    else:
        family,sign=_ACTIONS[action]
        if action=='dwell':
            if feedback.dwell_ms >= policy.minimum_dwell_ms:
                candidate=policy.session_dwell*min(feedback.dwell_ms/policy.saturated_dwell_ms,1.0)
                if candidate>highwater: highwater=candidate;new_signal=(family,action,candidate);meaningful=True
        elif family not in signals or (family in ('reaction','save') and signals[family].action != action):
            new_signal=(family,action,sign*getattr(policy,'session_'+action));meaningful=True
    if not meaningful:
        return replace(session,last_activity_ms=now) if policy.refresh_idle_on_noop else session
    # Preserve original source weights/times. Caps are derived at active source
    # anchors, so removing a family does not leave projection-damping residue.
    if remove is not None: signals.pop(remove)
    if new_signal is not None:
        family,action,weight=new_signal
        if feedback.repository is None: raise CompatibilityError('new transient signal requires pinned repository vector')
        signals[family]=Signal(family,action,feedback.repository,weight,now)
    if signals: entries[feedback.repo_id]=_entry(feedback.repo_id,signals,highwater)
    else: entries.pop(feedback.repo_id,None)
    ordered=_sorted_entries(entries)
    return replace(session,anchor_ms=_active_anchor(ordered,session.started_ms),last_activity_ms=now,entries=ordered)



def reduce_operation(state:State,operation:Operation,policy:Policy):
    if isinstance(state, BootstrapState):
        return _reduce_bootstrap(state, operation, policy)
    _check_state(state,policy)
    if not isinstance(operation,(Feedback,Profile,Begin,Reset,End,WithdrawnProfile)): raise ModelError('unknown typed operation')
    header=operation.header
    if not isinstance(header,Header) or header.user_id != state.user_id: raise CompatibilityError('operation owner mismatch')
    fingerprint=operation_sha256(operation)
    if header.operation_version==state.operation_version:
        if header.operation_id==state.last_operation_id and fingerprint==state.last_operation_sha256: return state
        raise OperationConflict('same operation version has conflicting immutable input')
    if header.operation_id==state.last_operation_id:raise OperationConflict('operation identity reused at another version')
    if header.operation_version<state.operation_version: raise JournalProofRequired('older duplicate requires committed journal proof')
    if header.previous_operation_version != state.operation_version: raise OperationGap('next model operation is not available')
    now=max(state.effective_ms,header.accepted_ms)
    sessions={row.session_id:row for row in state.sessions}
    sessions={key:(_seal(row) if not row.ended and now-row.last_activity_ms>=policy.idle_expiry_ms else row) for key,row in sessions.items()}
    long={row.repo_id:row for row in state.long_entries};baseline=state.baseline;feedback_version=state.feedback_version
    if isinstance(operation,Feedback):
        _uuid(operation.event_id);_uuid(operation.repo_id);_integer(operation.feedback_version,1)
        if operation.feedback_version != state.feedback_version+1: raise OperationGap('feedback cursor is not next')
        if operation.event_type not in (*_ACTIONS,*_INVERSE): raise ModelError('neutral/unknown events are not model operations')
        if operation.raw_client_timestamp_ms is not None:
            if type(operation.raw_client_timestamp_ms) is not int or not -MAX_INTEGER<=operation.raw_client_timestamp_ms<=MAX_INTEGER:
                raise ModelError('invalid immutable client timestamp')
        if operation.event_type=='dwell': _integer(operation.dwell_ms)
        elif operation.dwell_ms is not None: raise ModelError('dwell value on another event type')
        if operation.repository is not None: _check_vector(operation.repository,policy,'repository',operation.repo_id)
        signals=_signal_map(long.get(operation.repo_id));action=operation.event_type
        if action in _INVERSE:
            family,target=_INVERSE[action]
            if family in signals and signals[family].action==target: signals.pop(family)
        elif action in ('like','dislike','save'):
            family,sign=_ACTIONS[action]
            if family not in signals or signals[family].action != action:
                if operation.repository is None: raise CompatibilityError('new durable signal requires pinned repository vector')
                signals[family]=Signal(family,action,operation.repository,sign*getattr(policy,'long_'+action),now)
        if signals:long[operation.repo_id]=_entry(operation.repo_id,signals)
        else:long.pop(operation.repo_id,None)
        if operation.session is not None:
            binding=operation.session
            if not isinstance(binding,SessionBinding):raise ModelError('invalid typed session binding')
            if binding.user_id != state.user_id or binding.epoch_started_ms>header.accepted_ms: raise CompatibilityError('invalid session owner/accepted binding')
            session=sessions.get(binding.session_id)
            if session is None:
                if binding.active_at_acceptance:
                    raise EpochConflict('active session requires prior ordered Begin')
                # Authenticated inactive attribution can apply durable state
                # after journaled End compaction, never recreate a ledger.
            else:
                if binding.epoch>session.epoch: raise EpochConflict('future session epoch is not applied')
                if binding.epoch<session.epoch and binding.active_at_acceptance:
                    raise EpochConflict('old active epoch cannot follow ordered reset')
                if binding.epoch==session.epoch:
                    if binding.epoch_started_ms!=session.started_ms or binding.source_witness_sha256!=session.source_witness_sha256:
                        raise EpochConflict('canonical epoch start or ownership witness differs')
                    if not binding.active_at_acceptance or now-session.last_activity_ms>=policy.idle_expiry_ms:session=_seal(session)
                    if not session.ended:session=_session_update(session,operation,now,policy)
                sessions[binding.session_id]=session
        feedback_version=operation.feedback_version
    elif isinstance(operation,Profile):
        _check_vector(operation.baseline,policy,'profile',state.user_id)
        if type(operation.allow_genesis) is not bool:raise ModelError('explicit profile initialization authority required')
        if state.operation_version == 0:
            if not operation.allow_genesis or state.feedback_version != 0 or operation.baseline != baseline:
                raise OperationConflict('initial profile requires exact empty genesis basis')
        elif operation.baseline.content_version<=baseline.content_version:raise OperationConflict('profile revision did not advance')
        baseline=operation.baseline
    elif isinstance(operation,WithdrawnProfile):
        _check_withdrawal(operation)
        now = max(now, operation.revoked_at_ms)
    elif isinstance(operation,Begin):
        _uuid(operation.session_id);_integer(operation.epoch_started_ms);_hex(operation.source_witness_sha256)
        if operation.session_id in sessions:raise EpochConflict('Begin requires a fresh authenticated session')
        if operation.epoch_started_ms!=header.accepted_ms:raise EpochConflict('Begin time differs from trusted acceptance')
        sessions[operation.session_id]=SessionLedger(operation.session_id,1,0,False,header.accepted_ms,header.accepted_ms,header.accepted_ms,(),operation.source_witness_sha256)
    elif isinstance(operation,Reset):
        _uuid(operation.session_id);_integer(operation.old_epoch,1);_integer(operation.new_epoch,1);_hex(operation.source_witness_sha256)
        if operation.new_epoch != operation.old_epoch+1:raise EpochConflict('reset must advance exactly one epoch')
        previous=sessions.get(operation.session_id)
        if previous is None or previous.epoch!=operation.old_epoch:raise EpochConflict('reset predecessor mismatch')
        sessions[operation.session_id]=SessionLedger(operation.session_id,operation.new_epoch,operation.old_epoch,False,header.accepted_ms,header.accepted_ms,now,(),operation.source_witness_sha256)
    elif isinstance(operation,End):
        _uuid(operation.session_id);_integer(operation.epoch,1);_integer(operation.epoch_started_ms)
        if operation.epoch_started_ms>header.accepted_ms:raise EpochConflict('end predates canonical epoch start')
        previous=sessions.get(operation.session_id)
        if previous is None:raise EpochConflict('End requires an applied session witness')
        if operation.epoch!=previous.epoch:raise EpochConflict('End requires current epoch')
        if operation.epoch_started_ms!=previous.started_ms:raise EpochConflict('end epoch start differs from applied witness')
        # This deterministic transition is journaled before the complete-state
        # CAS. Backend retains permanent retired UUID authority; inactive late
        # feedback cannot recreate the compacted transient ledger.
        sessions.pop(operation.session_id)
    result=State(VERSION,state.user_id,policy.sha256,baseline,_sorted_entries(long),tuple(sessions[key] for key in sorted(sessions)),
        header.operation_version,feedback_version,now,header.operation_id,fingerprint)
    return _check_state(result,policy)


def reject_capacity(state,operation,policy):
    """Advance only the validated next operation after explicit capacity refusal.

    Caller must first attempt normal reduction and persist the configured bounded
    DLQ prepare before the complete-state CAS; this helper never skips gaps.
    """
    _check_state(state,policy)
    if not isinstance(operation,(Feedback,Profile,Begin,Reset,End)):raise ModelError('unknown rejected operation')
    header=operation.header
    if header.user_id!=state.user_id:raise CompatibilityError('rejected operation owner mismatch')
    if header.previous_operation_version!=state.operation_version:raise OperationGap('rejected operation predecessor absent')
    feedback_version=state.feedback_version
    if isinstance(operation,Feedback):
        _integer(operation.feedback_version,1)
        if operation.feedback_version!=state.feedback_version+1:raise OperationGap('rejected feedback cursor absent')
        feedback_version=operation.feedback_version
    result=replace(state,operation_version=header.operation_version,feedback_version=feedback_version,
        effective_ms=max(state.effective_ms,header.accepted_ms),last_operation_id=header.operation_id,
        last_operation_sha256=operation_sha256(operation))
    return _check_state(result,policy)


@dataclass(frozen=True)
class View:
    long_term: tuple[float,...]
    session: tuple[float,...]
    blended: tuple[float,...]
    explicit_negative_repo_ids: tuple[str,...]
    session_status: str
    epoch: int | None
    operation_version: int
    feedback_version: int
    profile_version: int
    policy_sha256: str
    as_of_ms: int
    vector_sha256: str


def recommendation_view(state,policy,*,session_id,epoch,as_of_ms):
    if isinstance(state, BootstrapState):
        raise CompatibilityError('withdrawn-only prefix has no real profile basis')
    _check_state(state,policy);_integer(as_of_ms)
    if as_of_ms<state.effective_ms:raise ModelError('generation clock precedes applied effective clock')
    baseline=_normalize(state.baseline.vector,state.baseline.vector,policy.normalization_epsilon)
    durable=_sum((state.baseline.vector,*(_cap(_contribution(row),policy.long_repo_cap) for row in state.long_entries)))
    long_term=_normalize(durable,baseline,policy.normalization_epsilon)
    transient=_zero();status='none';selected_epoch=None
    if session_id is not None:
        _uuid(session_id);_integer(epoch,1)
        found=next((s for s in state.sessions if s.session_id==session_id),None)
        if found is None:raise EpochConflict('session requires applied Begin and current epoch witness')
        if found is not None:
            selected_epoch=found.epoch
            if epoch>found.epoch:raise EpochConflict('requested future epoch is not applied')
            if epoch<found.epoch or found.ended:status='sealed'
            elif as_of_ms-found.last_activity_ms>=policy.idle_expiry_ms:status='expired'
            else:
                status='active';transient=_session_at(found,as_of_ms,policy)
    elif epoch is not None:raise ModelError('epoch without session')
    blended=_normalize(_sum((_scale(long_term,1-policy.session_blend),_scale(transient,policy.session_blend))),long_term,policy.normalization_epsilon)
    negatives=tuple(row.repo_id for row in state.long_entries if any(s.action=='dislike' for s in row.signals))
    vector_hash=_digest({'policy':policy.sha256,'as_of_ms':as_of_ms,'operation_version':state.operation_version,
        'profile_version':state.baseline.content_version,'feedback_version':state.feedback_version,'session_id':session_id,'epoch':epoch,'vector':blended})
    return View(long_term,transient,blended,negatives,status,selected_epoch,state.operation_version,state.feedback_version,
        state.baseline.content_version,policy.sha256,as_of_ms,vector_hash)


def replay(seed,operations,policy):
    state=seed
    for count,operation in enumerate(operations,1):
        if count>policy.max_replay_operations:raise StateOverflow('explicit replay work limit exceeded')
        state=reduce_operation(state,operation,policy)
    return state


def dispatch_mode(mode,legacy_prepare:Callable,new_prepare:Callable):
    """Pure computations only: caller must commit both outputs atomically.

    This boundary neither implements nor verifies legacy learning/persistence.
    """
    if mode not in ('disabled','shadow','enabled'):raise ModelError('unknown explicit feature mode')
    legacy=legacy_prepare()
    return (legacy,None) if mode=='disabled' else (legacy,new_prepare())
