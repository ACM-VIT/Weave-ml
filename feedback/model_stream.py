"""Durable exact-input admission and reclaimable ordered operation delivery."""
from __future__ import annotations

from dataclasses import dataclass
import uuid

from feedback.model_wire import parse_operation


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
  if old == ARGV[3] and old_id == ARGV[4] then return 0 end
  return -1
end
if redis.call('xlen', KEYS[1]) >= tonumber(ARGV[6]) or
   redis.call('hlen', KEYS[2]) + 2 > tonumber(ARGV[7]) then return -2 end
redis.call('xadd', KEYS[1], '*', 'operation_json', ARGV[5])
redis.call('hset', KEYS[2], ARGV[1], ARGV[3], ARGV[2], ARGV[4])
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
    def __init__(self, redis, limits, policies):
        self.redis, self.limits, self.policies = redis, limits, policies
        self.stream = limits.namespace + ':stream'
        self.identities = limits.namespace + ':identities'
        self.group = limits.namespace + ':workers'

    def admit(self, original_json):
        operation = parse_operation(original_json, maximum_bytes=self.limits.input_bytes, allowed_policies=self.policies)
        h = operation.header
        version_key = 'version:' + h.user_id + ':' + str(h.operation_version)
        id_key = 'id:' + h.operation_id
        result = int(self.redis.eval(_ADMIT, 2, self.stream, self.identities, version_key, id_key,
                                     h.source_envelope_sha256, version_key + ':' + h.source_envelope_sha256,
                                     original_json, self.limits.max_pending, self.limits.max_identities))
        if result == -1:
            raise OperationBytesConflict('operation identity was reused with different original bytes')
        if result == -2:
            raise OperationCapacity('operation stream or durable duplicate-proof capacity exhausted')
        return {'operation_id': h.operation_id, 'model_operation_version': str(h.operation_version),
                'source_envelope_sha256': h.source_envelope_sha256, 'status': 'accepted' if result else 'duplicate'}

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
            raw = fields.get('operation_json', fields.get(b'operation_json'))
            if isinstance(raw, bytes):
                raw = raw.decode('utf-8', errors='strict')
            operation = parse_operation(raw, maximum_bytes=self.stream.limits.input_bytes,
                                        allowed_policies=self.stream.policies)
            with self.lock_factory(operation.header.user_id) as lock:
                self.processor(operation, lock.assert_owned)
                lock.assert_owned()
                self.stream.acknowledge(message_id)
        return len(messages)
