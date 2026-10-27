"""Opt-in, exact-byte journal tiering protocol; never wired by runtime startup.

This relocates proofs, not their meaning. Every archived byte remains necessary
and quota charged. It does not reclaim operation admission identities or approve
retention policy. See docs/MODEL_HISTORY_RETENTION.md before any integration.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time

from feedback.model_runtime import BudgetedRecords, ModelLaneUnavailable, _RESERVE
from feedback.model_stream import OperationCapacity

# Routing is installed before deletion, but quota remains charged until deletion
# has been confirmed. A route also closes the race with a new reservation.
_RESERVE_HOT = """
if redis.call('hexists', KEYS[3], ARGV[1]) == 1 then return -4 end
""" + _RESERVE
_ENTER = """
if redis.call('hexists', KEYS[1], ARGV[1]) == 1 then return -1 end
local readers = tonumber(redis.call('hget', KEYS[2], ARGV[1]) or '0')
if readers >= tonumber(ARGV[2]) or
   (readers == 0 and redis.call('hlen', KEYS[2]) >= tonumber(ARGV[2])) then return -2 end
return redis.call('hincrby', KEYS[2], ARGV[1], 1)
"""
_LEAVE = """
local readers = tonumber(redis.call('hget', KEYS[1], ARGV[1]) or '0')
if readers < 1 then return -1 end
if readers == 1 then redis.call('hdel', KEYS[1], ARGV[1])
else redis.call('hincrby', KEYS[1], ARGV[1], -1) end
return 1
"""
_PUBLISH = """
local old = redis.call('hget', KEYS[3], ARGV[1])
if old then return old == ARGV[3] and 0 or -1 end
if redis.call('hexists', KEYS[4], ARGV[1]) == 1 then return -3 end
if redis.call('hget', KEYS[1], ARGV[1]) ~= ARGV[2] or
   redis.call('hexists', KEYS[2], ARGV[1]) == 1 then return -1 end
if redis.call('hlen', KEYS[3]) >= tonumber(ARGV[4]) then return -2 end
redis.call('hset', KEYS[3], ARGV[1], ARGV[3])
return 1
"""
_RELEASE = """
if redis.call('hget', KEYS[3], ARGV[1]) ~= ARGV[3] then return -1 end
local old = redis.call('hget', KEYS[1], ARGV[1])
if not old then return 0 end
if old ~= ARGV[2] or redis.call('hexists', KEYS[2], ARGV[1]) == 1 then return -1 end
local count = tonumber(redis.call('hget', KEYS[1], '_count') or '0')
local bytes = tonumber(redis.call('hget', KEYS[1], '_bytes') or '0')
if count < 1 or bytes < tonumber(ARGV[4]) then return -1 end
redis.call('hdel', KEYS[1], ARGV[1])
redis.call('hset', KEYS[1], '_count', count - 1, '_bytes', bytes - tonumber(ARGV[4]))
return 1
"""


class VerifiedJournalTier(BudgetedRecords):
    """Developer API for a dedicated hot/archive pair, disabled in ModelRuntime.

    Archive must itself be BudgetedRecords over a dedicated immutable store.
    Its reservations are permanent. Redis and both stores require the same
    durable backup/recovery boundary. The caller supplies an approved decision
    digest and a secret signing key; neither is provisioned or persisted here.
    A digest binds receipts to a decision; it is NOT evidence of human approval.
    """
    def __init__(self, records, redis, *, key, max_records, max_bytes, archive,
                 signing_key, decision_sha256):
        super().__init__(records, redis, key=key, max_records=max_records, max_bytes=max_bytes)
        if (type(archive) is not BudgetedRecords or archive.redis is not redis
                or archive.key == key or archive.records is records
                or archive.key.startswith(key + ':') or key.startswith(archive.key + ':')
                or (getattr(records, 'collection', None) is not None
                    and records.collection == getattr(archive.records, 'collection', None))):
            raise ValueError('dedicated bounded archive and shared durable Redis required')
        if type(signing_key) is not bytes or len(signing_key) < 32:
            raise ValueError('explicit archive signing key of at least 32 bytes required')
        if (type(decision_sha256) is not str or len(decision_sha256) != 64
                or any(c not in '0123456789abcdef' for c in decision_sha256)):
            raise ValueError('explicit retention decision digest required')
        self.archive, self.signing_key, self.decision = archive, signing_key, decision_sha256
        self.routes = key + ':archive-routes'
        self.readers = key + ':retention-readers'

    @staticmethod
    def _key(key):
        if type(key) is not str or not key or len(key.encode('utf-8')) > 160 or key.startswith('_'):
            raise ValueError('bounded non-reserved journal key required')

    def _receipt(self, key, raw):
        # Fixed-size authenticated identity; no original user bytes in Redis.
        message = json.dumps([1, self.key, self.archive.key, self.decision, key,
                              self._identity(raw)], separators=(',', ':')).encode()
        return self._identity(raw) + ':' + hmac.new(self.signing_key, message, hashlib.sha256).hexdigest()

    def _archive_record(self, key):
        raw = self.archive.get(key)
        if raw is not None:
            reserved = self.redis.hget(self.archive.key, key)
            if isinstance(reserved, bytes):
                reserved = reserved.decode('ascii')
            if reserved != self._identity(raw):
                raise ModelLaneUnavailable('archive reservation missing or differs')
        return raw

    def _archived(self, key):
        receipt = self.redis.hget(self.routes, key)
        if receipt is None:
            return None
        if isinstance(receipt, bytes):
            receipt = receipt.decode('ascii')
        raw = self._archive_record(key)
        if raw is None or not hmac.compare_digest(receipt, self._receipt(key, raw)):
            raise ModelLaneUnavailable('archive routing proof or exact bytes differ')
        return raw

    def get(self, key):
        self._key(key)
        entered = int(self.redis.eval(_ENTER, 2, self.routes, self.readers, key, self.max_records))
        if entered == -1:
            return self._archived(key)
        if entered == -2:
            raise OperationCapacity('retention reader capacity exhausted; inspect interrupted readers')
        completed = False
        try:
            found = super().get(key)
            # A lost route must not turn an archived key into a new identity.
            if found is None and self.archive.get(key) is not None:
                raise ModelLaneUnavailable('archive exists without routing proof; verified recovery required')
            completed = True
            return found
        finally:
            # A failed IO call may still complete remotely after its timeout.
            # Keep its durable fence until controlled recovery proves quiescence.
            if completed and int(self.redis.eval(_LEAVE, 1, self.readers, key)) != 1:
                raise ModelLaneUnavailable('retention reader accounting differs')

    def put_once(self, key, raw):
        self._key(key)
        if type(raw) is not bytes or not raw:
            raise ValueError('nonempty exact journal bytes required')
        found = self.get(key)
        if found is not None:
            return found
        result = int(self.redis.eval(_RESERVE_HOT, 3, self.key, self.candidates, self.routes,
            key, self._identity(raw), len(raw), self.max_records, self.max_bytes, raw))
        if result == -2:
            raise OperationCapacity('immutable journal capacity exhausted')
        if result == -3:
            raise ModelLaneUnavailable('journal reservation byte accounting differs')
        found = self.get(key)
        if found is None:
            raise ModelLaneUnavailable('reserved or archived journal record missing')
        return found

    def compact(self, key):
        """Copy, authenticate, route, delete, then refund. Retry the same key.

        Only explicitly named existing journal records are moved. No checkpoint
        age, producer TTL, or cursor is used as evidence to discard history.
        An interrupted route is repaired by repeating this operation; failures
        propagate and retain reservations. No caller should swallow them.
        """
        self._key(key)
        started = time.perf_counter()
        raw = self.get(key)
        if raw is None:
            raise ModelLaneUnavailable('cannot archive a missing journal record')
        if self.archive.put_once(key, raw) != raw or self._archive_record(key) != raw:
            raise ModelLaneUnavailable('archive verification differs from original bytes')
        receipt = self._receipt(key, raw)
        result = int(self.redis.eval(_PUBLISH, 4, self.key, self.candidates, self.routes, self.readers,
            key, self._identity(raw), receipt, self.archive.max_records))
        if result == -2:
            raise OperationCapacity('archive routing proof capacity exhausted')
        if result == -3:
            raise ModelLaneUnavailable('active or interrupted hot reader prevents compaction')
        if result < 0:
            raise ModelLaneUnavailable('archive publication reservation differs')
        # Verify the published routing path before crossing the deletion boundary.
        if self._archived(key) != raw:
            raise ModelLaneUnavailable('published archive is not readable')
        self.records.delete_verified(key, raw)
        if self.records.get(key) is not None:
            raise ModelLaneUnavailable('hot journal deletion not confirmed; quota remains charged')
        result = int(self.redis.eval(_RELEASE, 3, self.key, self.candidates, self.routes,
            key, self._identity(raw), receipt, len(raw)))
        if result < 0:
            raise ModelLaneUnavailable('archive release accounting differs')
        return {'released_records': result, 'released_bytes': result * len(raw),
                'elapsed_seconds': time.perf_counter() - started}

    def metrics(self):
        """Constant-round-trip logical payload/reservation counts, not disk sizes."""
        with self.redis.pipeline(transaction=True) as pipe:
            pipe.hmget(self.key, '_count', '_bytes', '_recovery_bytes')
            pipe.hmget(self.archive.key, '_count', '_bytes', '_recovery_bytes')
            pipe.hlen(self.routes)
            pipe.hlen(self.readers)
            hot, cold, routes, readers = pipe.execute()
        hot, cold = [int(x or 0) for x in hot], [int(x or 0) for x in cold]
        return {'hot_records': hot[0], 'hot_bytes': hot[1], 'hot_recovery_bytes': hot[2],
                'archive_records': cold[0], 'archive_bytes': cold[1], 'archive_recovery_bytes': cold[2],
                'routing_records': routes, 'reader_records': readers,
                'routing_metadata_max_bytes': routes * (160 + 143),
                'reader_metadata_max_bytes': readers * (160 + 7),
                'hot_record_headroom': self.max_records - hot[0],
                'hot_byte_headroom': self.max_bytes - hot[1] - hot[2],
                'archive_record_headroom': self.archive.max_records - cold[0],
                'archive_byte_headroom': self.archive.max_bytes - cold[1] - cold[2]}
