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
local old = redis.call('hget', KEYS[1], ARGV[1])
if old then return old == ARGV[2] and 0 or -1 end
local count = tonumber(redis.call('hget', KEYS[1], '_count') or '0')
local bytes = tonumber(redis.call('hget', KEYS[1], '_bytes') or '0')
if count + 1 > tonumber(ARGV[4]) or bytes + tonumber(ARGV[3]) > tonumber(ARGV[5]) then return -2 end
redis.call('hset', KEYS[1], ARGV[1], ARGV[2], '_count', count + 1, '_bytes', bytes + tonumber(ARGV[3]))
return 1
"""


def _validate_record_budget(max_records, max_bytes):
    if type(max_records) is not int or not 0 < max_records <= 1000000:
        raise ValueError('explicit journal record ceiling required')
    if type(max_bytes) is not int or not 0 < max_bytes <= 2**40:
        raise ValueError('explicit journal total byte ceiling required')


class BudgetedRecords:
    """Reserve durable total storage before an immutable Qdrant write.

    A failed write may leave a reservation. Keeping that capacity unavailable is
    intentional: only a controlled reconciliation can prove it safe to release.
    """
    def __init__(self, records, redis, *, key, max_records, max_bytes):
        _validate_record_budget(max_records, max_bytes)
        self.records, self.redis, self.key = records, redis, key
        self.max_records, self.max_bytes = max_records, max_bytes

    def get(self, key):
        return self.records.get(key)

    def put_once(self, key, raw):
        import hashlib
        identity = hashlib.sha256(raw).hexdigest() + ':' + str(len(raw))
        result = int(self.redis.eval(_RESERVE, 1, self.key, key, identity, len(raw), self.max_records, self.max_bytes))
        if result == -2:
            raise OperationCapacity('immutable journal capacity exhausted')
        # Existing operation aliases may legitimately have a different winning
        # pin from a stale preparer; readback below decides the frozen winner.
        if result == -1:
            found = self.records.get(key)
            if found is None:
                raise ModelLaneUnavailable('immutable reservation write remains unresolved')
            return found
        return self.records.put_once(key, raw)


class ModelRuntime:
    def __init__(self, config, *, redis=None, qdrant=None, encoder=None, settings=None):
        exact_keys(config, ('schema_version', 'policy', 'stream', 'journal', 'journal_collection',
                            'max_journal_records', 'max_journal_bytes'))
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
        self.stream = OperationStream(self.redis, stream_limits, {self.policy.sha256})
        records = QdrantRecords(self.qdrant, models, collection=config['journal_collection'], record_bytes=limits.record_bytes)
        self.journal = Journal(BudgetedRecords(records, self.redis, key=self.stream.limits.namespace + ':journal-budget',
            max_records=config['max_journal_records'], max_bytes=config['max_journal_bytes']), limits)
        self.users = QdrantAtomicUsers(self.qdrant, models, payload_snapshot_filter, collection=self.settings.user_collection,
                                       combined_user_bytes=limits.combined_user_bytes, policy=self.policy, journal=self.journal)
        self.legacy = OrderedFeedbackApplier(qdrant=self.qdrant, settings=self.settings)
        if encoder is None:
            from embedding.runtime import user_onboarding_pipeline
            encoder = lambda profile: user_onboarding_pipeline().generate_interest_vector(profile)
        self.processor = ModelProcessor(self.journal, self.users, self.policy, self.legacy, encoder,
                                         dead_letter_prepare=self.stream.prepare_dead_letter)

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
                logging.getLogger(__name__).warning('model operation remains pending', extra={'error_type': type(exc).__name__}, exc_info=False)
                stop.wait(0.1)
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
