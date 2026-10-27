"""Durable exact-input admission and reclaimable ordered operation delivery."""
from __future__ import annotations

from dataclasses import dataclass
import uuid
import re
from itertools import islice

from feedback.model_wire import parse_operation, parse_withdrawal


class OperationCapacity(RuntimeError):
    pass


class OperationBytesConflict(RuntimeError):
    pass


# No MAXLEN trimming or TTL: accepted work and duplicate proofs are durable.
# Retention requires a separately verified checkpoint, never a producer timer.
_ADMIT = """
local old = redis.call('hget', KEYS[2], ARGV[1])
local old_id = redis.call('hget', KEYS[2], ARGV[2])
if old or old_id then
  if old ~= ARGV[3] or old_id ~= ARGV[4] then return -1 end
  if redis.call('hget', KEYS[3], ARGV[1]) then return 2 end
  return 0
end
local extra = 2
local newguard = redis.call('hexists', KEYS[4], ARGV[8]) == 0
if newguard and redis.call('hlen', KEYS[4]) >= tonumber(ARGV[9]) then return -2 end
if redis.call('xlen', KEYS[1]) >= tonumber(ARGV[6]) + tonumber(ARGV[10]) or
   redis.call('hlen', KEYS[2]) + extra > tonumber(ARGV[7]) then return -2 end
redis.call('xadd', KEYS[1], '*', 'operation_json', ARGV[5])
redis.call('hset', KEYS[2], ARGV[1], ARGV[3], ARGV[2], ARGV[4])
redis.call('hsetnx', KEYS[4], ARGV[8], 'pending')
return 1
"""
_WITHDRAW = """
local old = redis.call('hget', KEYS[2], ARGV[1])
local old_id = redis.call('hget', KEYS[2], ARGV[2])
if (old and old ~= ARGV[3]) or (old_id and old_id ~= ARGV[4]) then return -1 end
local withdrawn = redis.call('hget', KEYS[3], ARGV[1])
if withdrawn then return withdrawn == ARGV[5] and 0 or -1 end
local extra = old and 0 or 2
if redis.call('hexists', KEYS[4], ARGV[8]) == 0 and
   redis.call('hlen', KEYS[4]) >= tonumber(ARGV[9]) then return -2 end
if redis.call('xlen', KEYS[1]) >= tonumber(ARGV[6]) + tonumber(ARGV[10]) or
   redis.call('hlen', KEYS[2]) + extra > tonumber(ARGV[7]) or
   redis.call('hlen', KEYS[3]) >= tonumber(ARGV[7]) then return -2 end
redis.call('xadd', KEYS[1], '*', 'withdrawal_json', ARGV[5])
redis.call('hset', KEYS[2], ARGV[1], ARGV[3], ARGV[2], ARGV[4])
redis.call('hsetnx', KEYS[4], ARGV[8], 'pending')
redis.call('hset', KEYS[3], ARGV[1], ARGV[5])
return 1
"""
_GUARD = """
local old = redis.call('hget', KEYS[1], ARGV[1])
if old == 'initialized' then return 0 end
if old and old ~= 'pending' then return -1 end
if not old and redis.call('hlen', KEYS[1]) >= tonumber(ARGV[2]) then return -2 end
redis.call('hset', KEYS[1], ARGV[1], 'initialized')
return 1
"""

_ACK = """
redis.call('xack', KEYS[1], ARGV[1], ARGV[2])
redis.call('xdel', KEYS[1], ARGV[2])
return 1
"""
_DLQ = """
local old = redis.call('hget', KEYS[2], ARGV[1])
if old then
  if old == ARGV[2] then return 0 end
  return -1
end
if redis.call('xlen', KEYS[1]) >= tonumber(ARGV[3]) then return -2 end
redis.call('xadd', KEYS[1], '*', 'prepare_sha256', ARGV[1], 'prepared_json', ARGV[2], 'status', 'prepared_not_committed')
redis.call('hset', KEYS[2], ARGV[1], ARGV[2])
return 1
"""


@dataclass(frozen=True)
class StreamLimits:
    """Normal entries plus exactly one global, ready-only recovery entry.

    The absolute stream ceiling is max_pending + 1, independent of the number
    of users and missing versions. Duplicate/guard/journal/DLQ limits still fail
    closed; the recovery entry never bypasses any of those durable budgets.
    """
    namespace: str
    max_pending: int
    max_identities: int
    input_bytes: int
    reclaim_ms: int
    max_dead_letters: int

    def __post_init__(self):
        if not isinstance(self.namespace, str) or not self.namespace or len(self.namespace) > 128:
            raise ValueError('explicit operation namespace required')
        for name, ceiling in [('max_pending', 100000), ('max_identities', 1000000),
                              ('input_bytes', 128 * 1024), ('reclaim_ms', 3600000), ('max_dead_letters', 100000)]:
            if type(getattr(self, name)) is not int or not 0 < getattr(self, name) <= ceiling:
                raise ValueError('invalid explicit operation resource budget')


class OperationStream:
    def __init__(self, redis, limits, policies, *, initialization_scope, max_guard_users):
        if type(initialization_scope) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}', initialization_scope):
            raise ValueError('explicit stable logical deployment initialization scope required')
        if type(max_guard_users) is not int or not 0 < max_guard_users <= 1000000:
            raise ValueError('explicit initialization guard user capacity required')
        self.redis, self.limits, self.policies = redis, limits, policies
        # This stable scope is independent of model policy, transport namespace
        # and physical Qdrant endpoint. Changing it requires an audited boundary.
        self.guard_key = 'ml:model:initialization-guard:v1:' + initialization_scope
        self.max_guard_users = max_guard_users
        self.stream = limits.namespace + ':stream'
        self.identities = limits.namespace + ':identities'
        self.group = limits.namespace + ':workers'
        self.withdrawals = limits.namespace + ':withdrawals'
        self.profile_preflight = None
        self.ready_predecessor = None

    def _admission(self, script, operation, *args):
        result = self.redis.eval(script, 4, self.stream, self.identities,
            self.withdrawals, self.guard_key, *args, 0)
        return self._recover(script, operation, result, args)

    def _recover(self, script, operation, result, args):
        if type(result) is not int or result not in {-2, -1, 0, 1, 2}:
            raise RuntimeError('unknown operation admission response; retry original bytes')
        if result == -2 and self.ready_predecessor is not None and self.ready_predecessor(operation):
            # Exactly one global recovery slot, never one per user or gap.
            # A validated committed cursor proves this input can make progress.
            # The Lua capacity check remains atomic across replicas. After its
            # terminal CAS/ACK the same slot can admit the next missing prefix,
            # draining even legacy accepted futures with arbitrarily long gaps.
            result = self.redis.eval(script, 4, self.stream, self.identities,
                self.withdrawals, self.guard_key, *args, 1)
            if type(result) is not int or result not in {-2, -1, 0, 1, 2}:
                raise RuntimeError('unknown operation admission response; retry original bytes')
        return result

    def admit(self, original_json):
        operation = parse_operation(original_json, maximum_bytes=self.limits.input_bytes, allowed_policies=self.policies)
        return self._admit_parsed(operation)

    def _arguments(self, operation):
        h = operation.header
        version_key = 'version:' + h.user_id + ':' + str(h.operation_version)
        return (version_key, 'id:' + h.operation_id, h.source_envelope_sha256,
                version_key + ':' + h.source_envelope_sha256, operation.original_json,
                self.limits.max_pending, self.limits.max_identities, h.user_id, self.max_guard_users)

    def admit_batch(self, originals):
        """Validate all inputs, then submit bounded, individually atomic scripts.

        A pipeline is not a transaction: any subset may commit before a response
        is lost. Callers must retry the exact complete batch on failure. Profiles
        are barriers, and recovery is resolved before another input for that
        user can be sent. There is no claim of global ordering across producers.
        """
        from feedback.model_wire import WireError
        originals = list(islice(originals, 51))
        if not 1 <= len(originals) <= 50:
            raise WireError('operation batch must contain 1 to 50 envelopes')
        operations = [parse_operation(raw, maximum_bytes=self.limits.input_bytes,
                                      allowed_policies=self.policies) for raw in originals]
        responses, pending = [], []
        users, byte_count = set(), 0

        def flush():
            if not pending:
                return
            if len(pending) == 1:
                responses.append(self._admit_parsed(pending[0]))
                return
            with self.redis.pipeline(transaction=False) as pipeline:
                for operation in pending:
                    pipeline.eval(_ADMIT, 4, self.stream, self.identities,
                                  self.withdrawals, self.guard_key, *self._arguments(operation), 0)
                results = pipeline.execute(raise_on_error=False)
            if len(results) != len(pending):
                raise RuntimeError('incomplete operation admission response; retry original bytes')
            # A command error does not roll back other commands. Do not expose a
            # successful batch acknowledgement unless every result is proven.
            if any(isinstance(result, Exception) for result in results):
                raise next(result for result in results if isinstance(result, Exception))
            for operation, result in zip(pending, results):
                result = self._recover(_ADMIT, operation, result, self._arguments(operation))
                responses.append(self._response(operation, result))

        for operation in operations:
            # Include the complete RESP command, including repeated Lua source,
            # keys and UTF-8 payload, in the one-MiB pipeline wire budget.
            size = self._command_bytes(operation)
            if (operation.kind == 'profile' or len(pending) >= 50 or
                    byte_count + size > 1024 * 1024 or
                    (self.ready_predecessor is not None and operation.header.user_id in users)):
                flush()
                pending, users, byte_count = [], set(), 0
            if operation.kind == 'profile':
                responses.append(self._admit_parsed(operation))
            else:
                pending.append(operation)
                users.add(operation.header.user_id)
                byte_count += size
        flush()
        return responses

    def _command_bytes(self, operation):
        values = ('EVAL', _ADMIT, 4, self.stream, self.identities, self.withdrawals,
                  self.guard_key, *self._arguments(operation), 0)
        lengths = [len(str(value).encode('utf-8')) for value in values]
        return len(str(len(values))) + 3 + sum(n + len(str(n)) + 5 for n in lengths)

    def _admit_parsed(self, operation):
        if operation.kind == 'profile' and self.profile_preflight is not None:
            from feedback.model_journal import OriginalIdentityConflict
            try:
                proven = self.profile_preflight(operation)
            except OriginalIdentityConflict as exc:
                raise OperationBytesConflict('original profile identity conflict') from exc
            if proven is not None:
                return proven
        result = self._admission(_ADMIT, operation, *self._arguments(operation))
        return self._response(operation, result)

    def _response(self, operation, result):
        h = operation.header
        if type(result) is not int or result not in {-2, -1, 0, 1, 2}:
            raise RuntimeError('unknown operation admission response; retry original bytes')
        if result == -1:
            raise OperationBytesConflict('operation identity was reused with different original bytes')
        if result == -2:
            raise OperationCapacity('operation stream or durable duplicate-proof capacity exhausted')
        response = {'operation_id': h.operation_id, 'model_operation_version': str(h.operation_version),
                'source_envelope_sha256': h.source_envelope_sha256,
                'status': 'withdrawal_accepted' if result == 2 else 'accepted' if result else 'duplicate'}
        if result == 2:
            withdrawal = self.lookup_withdrawal(h.user_id, h.operation_version)
            if withdrawal is None or withdrawal.header.source_envelope_sha256 != h.source_envelope_sha256:
                raise OperationBytesConflict('withdrawal proof unavailable or differs')
            response['withdrawal_sha256'] = withdrawal.withdrawal_sha256
        return response

    def lookup_withdrawal(self, user_id, version):
        raw = self.redis.hget(self.withdrawals, 'version:' + user_id + ':' + str(version))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8', errors='strict')
        return parse_withdrawal(raw, maximum_bytes=self.limits.input_bytes, allowed_policies=self.policies)

    def admit_withdrawal(self, original_json):
        operation = parse_withdrawal(original_json, maximum_bytes=self.limits.input_bytes, allowed_policies=self.policies)
        h = operation.header
        version_key = 'version:' + h.user_id + ':' + str(h.operation_version)
        result = self._admission(_WITHDRAW, operation,
            version_key, 'id:' + h.operation_id, h.source_envelope_sha256, version_key + ':' + h.source_envelope_sha256,
            original_json, self.limits.max_pending, self.limits.max_identities, h.user_id, self.max_guard_users)
        if result == -1:
            raise OperationBytesConflict('withdrawal or original operation identity differs')
        if result == -2:
            raise OperationCapacity('withdrawal stream or identity capacity exhausted')
        return {'operation_id': h.operation_id, 'model_operation_version': str(h.operation_version),
                'source_envelope_sha256': h.source_envelope_sha256, 'withdrawal_sha256': operation.withdrawal_sha256,
                'status': 'accepted' if result else 'duplicate'}

    def ensure_initialized(self, user_id):
        result = int(self.redis.eval(_GUARD, 1, self.guard_key, user_id, self.max_guard_users))
        if result == -2:
            raise OperationCapacity('initialization guard capacity exhausted')
        if result == -1:
            raise OperationBytesConflict('initialization deny-bootstrap guard is corrupt')
        if self.redis.hget(self.guard_key, user_id) not in {'initialized', b'initialized'}:
            raise OperationCapacity('initialization guard was not durably readable')

    def refuse_missing_initialized(self, user_id):
        value = self.redis.hget(self.guard_key, user_id)
        if value in {'initialized', b'initialized'}:
            raise OperationBytesConflict('initialized user model is missing; explicit verified rebuild required')
        if value not in {None, 'pending', b'pending'}:
            raise OperationBytesConflict('initialization deny-bootstrap guard is corrupt')

    def prepare_dead_letter(self, header, input_sha, prepare_sha, code):
        from feedback.model_wire import canonical_json
        raw = canonical_json({'schema_version': 1, 'user_id': header.user_id,
                              'operation_id': header.operation_id, 'model_operation_version': str(header.operation_version),
                              'input_sha256': input_sha, 'prepare_sha256': prepare_sha,
                              'error_code': code, 'status': 'prepared_not_committed'})
        result = int(self.redis.eval(_DLQ, 2, self.limits.namespace + ':dlq', self.limits.namespace + ':dlq-identities',
                                     prepare_sha, raw, self.limits.max_dead_letters))
        if result == -2:
            raise OperationCapacity('prepared rejection capacity exhausted; cursor remains pending')
        if result == -1:
            raise OperationBytesConflict('prepared rejection identity differs')

    def create_group(self):
        try:
            self.redis.xgroup_create(self.stream, self.group, id='0', mkstream=True)
        except Exception as exc:
            if 'BUSYGROUP' not in str(exc):
                raise

    def acknowledge(self, message_id):
        self.redis.eval(_ACK, 1, self.stream, self.group, message_id)


class OperationConsumer:
    """Bounded synchronous polling; one renewable shared user lock per message.

    Delivery exceptions leave the original stream entry pending. A greater
    applied cursor alone never authorizes ACK; the processor proves reachability.
    """
    def __init__(self, stream, processor, lock_factory):
        self.stream, self.processor, self.lock_factory = stream, processor, lock_factory
        self.consumer = 'model-' + uuid.uuid4().hex
        self.cursor = '0-0'
        self.prefer_new = True
        stream.create_group()

    def run_once(self):
        def fresh():
            batches = self.stream.redis.xreadgroup(self.stream.group, self.consumer, {self.stream.stream: '>'}, count=1)
            return batches[0][1] if batches else []
        def reclaim():
            result = self.stream.redis.xautoclaim(self.stream.stream, self.stream.group, self.consumer,
                min_idle_time=self.stream.limits.reclaim_ms, start_id=self.cursor, count=1)
            self.cursor = result[0]
            return result[1]
        # A pending dependency/gap for one owner must not starve new work for
        # another owner (or the missing predecessor arriving later).
        first, second = (fresh, reclaim) if self.prefer_new else (reclaim, fresh)
        self.prefer_new = not self.prefer_new
        messages = first() or second()
        for message_id, fields in messages:
            normal = fields.get('operation_json', fields.get(b'operation_json'))
            withdrawal = fields.get('withdrawal_json', fields.get(b'withdrawal_json'))
            if (normal is None) == (withdrawal is None):
                raise OperationBytesConflict('ambiguous operation stream message')
            raw = normal if normal is not None else withdrawal
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8', errors='strict')
            parser = parse_operation if normal is not None else parse_withdrawal
            operation = parser(raw, maximum_bytes=self.stream.limits.input_bytes, allowed_policies=self.stream.policies)
            with self.lock_factory(operation.header.user_id) as lock:
                self.processor(operation, lock.assert_owned)
                lock.assert_owned()
                self.stream.acknowledge(message_id)
        return len(messages)
