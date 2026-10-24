"""Explicitly configured model-operation provider and separate worker entrypoint.

No policy or collection is provisioned by import/startup. An absent configuration
keeps the existing wire and legacy worker unchanged. Collections must already
exist; production activation and migration are separate controlled operations.
"""
from __future__ import annotations

from dataclasses import fields
from functools import lru_cache
import os
from pathlib import Path
import signal
import threading

from feedback.long_session import EmbeddingIdentity, Policy
from feedback.model_journal import Journal, Limits, QdrantAtomicUsers, QdrantRecords
from feedback.model_processor import ModelProcessor
from feedback.model_stream import OperationCapacity, OperationConsumer, OperationStream, StreamLimits
from feedback.model_wire import exact_keys, strict_json


class ModelLaneUnavailable(RuntimeError):
    pass


_RESERVE = """
if string.len(ARGV[6]) ~= tonumber(ARGV[3]) then return -3 end
local old = redis.call('hget', KEYS[1], ARGV[1])
local count = tonumber(redis.call('hget', KEYS[1], '_count') or '0')
local bytes = tonumber(redis.call('hget', KEYS[1], '_bytes') or '0')
local recovery = tonumber(redis.call('hget', KEYS[1], '_recovery_bytes') or '0')
if count < 0 or bytes < 0 or recovery < 0 then return -3 end
if old then
  if old ~= ARGV[2] then return -1 end
  if redis.call('hexists', KEYS[2], ARGV[1]) == 0 then
    if bytes + recovery + tonumber(ARGV[3]) > tonumber(ARGV[5]) then return -2 end
    redis.call('hset', KEYS[2], ARGV[1], ARGV[6])
    redis.call('hset', KEYS[1], '_recovery_bytes', recovery + tonumber(ARGV[3]))
  end
  return 0
end
if count + 1 > tonumber(ARGV[4]) or bytes + recovery + 2 * tonumber(ARGV[3]) > tonumber(ARGV[5]) then return -2 end
redis.call('hset', KEYS[1], ARGV[1], ARGV[2], '_count', count + 1, '_bytes', bytes + tonumber(ARGV[3]))
redis.call('hset', KEYS[1], '_recovery_bytes', recovery + tonumber(ARGV[3]))
redis.call('hset', KEYS[2], ARGV[1], ARGV[6])
return 1
"""
_LOOKUP_RECOVERY = """
return {redis.call('hget', KEYS[1], ARGV[1]), redis.call('hget', KEYS[2], ARGV[1])}
"""
_FINISH_RECOVERY = """
local candidate = redis.call('hget', KEYS[2], ARGV[1])
if not candidate then return 0 end
if candidate ~= ARGV[4] or redis.call('hget', KEYS[1], ARGV[1]) ~= ARGV[2] then return -1 end
local remaining = tonumber(redis.call('hget', KEYS[1], '_recovery_bytes') or '0') - tonumber(ARGV[3])
if remaining < 0 then return -1 end
redis.call('hdel', KEYS[2], ARGV[1])
redis.call('hset', KEYS[1], '_recovery_bytes', remaining)
return 1
"""


def _validate_record_budget(max_records, max_bytes):
    if type(max_records) is not int or not 0 < max_records <= 1000000:
        raise ValueError('explicit journal record ceiling required')
    if type(max_bytes) is not int or not 0 < max_bytes <= 2**40:
        raise ValueError('explicit journal total byte ceiling required')


class BudgetedRecords:
    """Reserve durable total storage before an immutable Qdrant write.

    Each reservation atomically freezes its candidate before Qdrant IO. Final
    reserved bytes plus outstanding recovery-copy bytes share the same ceiling.
    Only the temporary copy is refunded after identical Qdrant readback; durable
    identities, counts and final-byte reservations are never released here.
    """
    def __init__(self, records, redis, *, key, max_records, max_bytes):
        _validate_record_budget(max_records, max_bytes)
        self.records, self.redis, self.key = records, redis, key
        self.candidates = key + ':candidates'
        self.max_records, self.max_bytes = max_records, max_bytes

    @staticmethod
    def _identity(raw):
        import hashlib
        return hashlib.sha256(raw).hexdigest() + ':' + str(len(raw))

    def get(self, key):
        # Sample both Redis fields atomically BEFORE reading Qdrant. A worker
        # may finish/refund concurrently; the subsequent Qdrant read then sees
        # its completed write, never a false legacy missing-candidate diagnosis.
        reserved, candidate = self.redis.eval(_LOOKUP_RECOVERY, 2, self.key, self.candidates, key)
        found = self.records.get(key)
        if isinstance(reserved, bytes):
            reserved = reserved.decode('ascii')
        if found is not None and reserved is not None and self._identity(found) != reserved:
            raise ModelLaneUnavailable('Qdrant journal bytes differ from permanent reservation')
        if candidate is None:
            if found is None and reserved is not None:
                raise ModelLaneUnavailable('immutable reservation lacks its frozen candidate; verified recovery required')
            return found
        if isinstance(candidate, str):
            candidate = candidate.encode('utf-8')
        identity = self._identity(candidate)
        if reserved != identity:
            raise ModelLaneUnavailable('frozen journal candidate differs from reservation')
        if found is None:
            found = self.records.put_once(key, candidate)
        if found != candidate:
            raise ModelLaneUnavailable('frozen journal candidate differs from Qdrant readback')
        result = int(self.redis.eval(_FINISH_RECOVERY, 2, self.key, self.candidates,
            key, identity, len(candidate), candidate))
        if result < 0:
            raise ModelLaneUnavailable('journal recovery accounting differs')
        return found

    def put_once(self, key, raw):
        found = self.get(key)
        if found is not None:
            return found
        identity = self._identity(raw)
        result = int(self.redis.eval(_RESERVE, 2, self.key, self.candidates,
            key, identity, len(raw), self.max_records, self.max_bytes, raw))
        if result == -2:
            raise OperationCapacity('immutable journal capacity exhausted')
        if result == -3:
            raise ModelLaneUnavailable('journal reservation byte accounting differs')
        # A competing preparer reuses the FIRST frozen bytes. Legacy hash-only
        # reservations can recover only an identical candidate, never guessed
        # replacement bytes; absent original bytes require verified recovery.
        found = self.get(key)
        if found is None:
            raise ModelLaneUnavailable('immutable reservation lacks its frozen candidate; verified recovery required')
        return found


class ModelRuntime:
    def __init__(self, config, *, redis=None, qdrant=None, encoder=None, settings=None):
        exact_keys(config, ('schema_version', 'policy', 'stream', 'journal', 'journal_collection',
                            'max_journal_records', 'max_journal_bytes', 'initialization_scope', 'max_initialization_guard_users'))
        if type(config['schema_version']) is not int or config['schema_version'] != 1:
            raise ValueError('unsupported model-lane configuration')
        policy_data = dict(config['policy'])
        exact_keys(policy_data, tuple(field.name for field in fields(Policy)))
        policy_data['embedding'] = EmbeddingIdentity(**policy_data['embedding'])
        self.policy = Policy(**policy_data)
        from config import EMBEDDING_MODEL_REVISION, REPOSITORY_EMBEDDING_MODEL, REPOSITORY_EMBEDDING_VERSION
        if self.policy.embedding != EmbeddingIdentity(REPOSITORY_EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION, 384):
            raise ValueError('model lane must retain the frozen production encoder identity')
        if self.policy.repository_recipe != REPOSITORY_EMBEDDING_VERSION:
            raise ValueError('model policy must identify the frozen repository embedding recipe')
        from feedback.v2 import OrderedFeedbackApplier, _redis_client
        from feedback.v2_settings import V2FeedbackSettings
        from embedding.qdrant_cas import payload_snapshot_filter
        from qdrant_client import QdrantClient, models
        self.settings = settings if settings is not None else V2FeedbackSettings.from_env()
        stream_limits = StreamLimits(**config['stream'])
        limits = Limits(**config['journal'])
        if limits.input_bytes != stream_limits.input_bytes:
            raise ValueError('admission and journal input budgets must agree')
        if not isinstance(config['journal_collection'], str) or not config['journal_collection'] or config['journal_collection'] == self.settings.user_collection:
            raise ValueError('explicit separate journal collection required')
        _validate_record_budget(config['max_journal_records'], config['max_journal_bytes'])
        self.redis = redis if redis is not None else _redis_client(self.settings)
        self.qdrant = qdrant if qdrant is not None else QdrantClient(url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key, timeout=self.settings.qdrant_timeout_seconds)
        self.stream = OperationStream(self.redis, stream_limits, {self.policy.sha256},
            initialization_scope=config['initialization_scope'], max_guard_users=config['max_initialization_guard_users'])
        records = QdrantRecords(self.qdrant, models, collection=config['journal_collection'], record_bytes=limits.record_bytes)
        self.journal = Journal(BudgetedRecords(records, self.redis, key=self.stream.limits.namespace + ':journal-budget',
            max_records=config['max_journal_records'], max_bytes=config['max_journal_bytes']), limits)
        self.users = QdrantAtomicUsers(self.qdrant, models, payload_snapshot_filter, collection=self.settings.user_collection,
                                       combined_user_bytes=limits.combined_user_bytes, policy=self.policy, journal=self.journal,
            ensure_initialized=self.stream.ensure_initialized, refuse_missing=self.stream.refuse_missing_initialized)
        self.legacy = OrderedFeedbackApplier(qdrant=self.qdrant, settings=self.settings)
        if encoder is None:
            from embedding.runtime import user_onboarding_pipeline
            encoder = lambda profile: user_onboarding_pipeline().generate_interest_vector(profile)
        self.processor = ModelProcessor(self.journal, self.users, self.policy, self.legacy, encoder,
                                         dead_letter_prepare=self.stream.prepare_dead_letter, withdrawal_lookup=self.stream.lookup_withdrawal)
        self.stream.profile_preflight = self.processor.preflight_original
        self.stream.ready_predecessor = self.processor.ready_predecessor

    def consumer(self):
        from feedback.user_lock import user_vector_lock
        return OperationConsumer(self.stream, self.processor,
                                  lambda user: user_vector_lock(self.redis, user, settings=self.settings))

    def close(self):
        self.qdrant.close()
        self.redis.close()


@lru_cache(maxsize=1)
def _cached_model_runtime():
    name = os.getenv('WEAVE_MODEL_LANE_CONFIG')
    if not name:
        raise ModelLaneUnavailable('negotiated model-operation lane is not configured')
    with Path(name).open('rb') as source:
        raw = source.read(256 * 1024 + 1)
    config = strict_json(raw.decode('utf-8'), maximum_bytes=256 * 1024)
    return ModelRuntime(config)


_model_runtime_lock = threading.RLock()


def model_runtime():
    # functools.lru_cache alone may call the constructor concurrently on its
    # first miss. Serialize construction so one configured client set is owned.
    with _model_runtime_lock:
        return _cached_model_runtime()


def _clear_runtime_cache():
    with _model_runtime_lock:
        _cached_model_runtime.cache_clear()


model_runtime.cache_info = _cached_model_runtime.cache_info
model_runtime.cache_clear = _clear_runtime_cache


def main():
    from dotenv import load_dotenv
    load_dotenv()
    runtime = model_runtime()
    consumer = runtime.consumer()
    stop = threading.Event()
    def halt(signum, frame):
        stop.set()
    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    try:
        while not stop.is_set():
            try:
                if not consumer.run_once():
                    stop.wait(0.1)
            except Exception as exc:
                # Do not print payloads or acknowledge pending work. One failed
                # user does not block new delivery/reclaim for another owner.
                import logging
                logging.getLogger(__name__).warning('model operation remains pending (%s)', type(exc).__name__,
                                                    extra={'error_type': type(exc).__name__}, exc_info=False)
                stop.wait(0.1)
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
