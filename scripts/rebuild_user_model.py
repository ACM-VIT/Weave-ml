"""Verify an owned user's immutable model journal, or stage a verified copy.

Dry-run is the default. Staging requires an explicit existing target collection
and --stage-copy. It never changes serving configuration, overwrites a user,
creates a collection, deletes history or writes back to the source collection.
"""
from __future__ import annotations

import argparse
import hashlib
import json

from feedback.model_journal import JournalConflict, QdrantAtomicUsers
from feedback.model_runtime import model_runtime
from feedback.model_wire import canonical_uuid, digest
from feedback.user_lock import user_vector_lock


def verify_and_stage(runtime, *, user_id, checkpoint_sha256, expected_head, target_collection=None, stage_copy=False):
    user_id = canonical_uuid(user_id)
    digest(checkpoint_sha256); digest(expected_head)
    if stage_copy != (target_collection is not None):
        raise ValueError('staging requires both explicit target collection and stage-copy')
    if target_collection in {runtime.settings.user_collection, runtime.journal.records.records.collection}:
        raise ValueError('staging must use a separate existing user-vector collection')
    if stage_copy:
        # Resolve source aliases and refuse any alias target. A verified target
        # must be a distinct physical collection, never another name for source.
        aliases={item.alias_name:item.collection_name for item in runtime.qdrant.get_aliases().aliases}
        sources={aliases.get(name,name) for name in (runtime.settings.user_collection,
            runtime.journal.records.records.collection)}
        collections={item.name for item in runtime.qdrant.get_collections().collections}
        if target_collection in aliases or target_collection in sources or target_collection not in collections:
            raise ValueError('staging must use a separate existing physical collection; aliases are forbidden')
    from embedding.qdrant_cas import payload_snapshot_filter
    from qdrant_client import models
    with user_vector_lock(runtime.redis, user_id, settings=runtime.settings) as lock:
        observed = runtime.users.read(user_id)
        if observed.head != expected_head:
            raise JournalConflict('source head changed before rebuild')
        rebuilt = runtime.processor.coordinator.rebuild(checkpoint_sha256, expected_head, expected_user_id=user_id)
        if observed != rebuilt:
            raise JournalConflict('independent rebuild differs from source point')
        lock.assert_owned()
        status = 'verified_dry_run'
        if stage_copy:
            target = QdrantAtomicUsers(runtime.qdrant, models, payload_snapshot_filter, collection=target_collection,
                combined_user_bytes=runtime.users.limit, policy=runtime.policy, journal=runtime.journal)
            vector, payload = target._point(rebuilt)
            # INSERT_ONLY cannot overwrite a conflicting pre-existing point.
            runtime.qdrant.upsert(collection_name=target_collection, points=[models.PointStruct(
                id=user_id, vector=vector, payload=payload)], wait=True, ordering=models.WriteOrdering.STRONG,
                update_mode=models.UpdateMode.INSERT_ONLY)
            if target.read(user_id) != rebuilt:
                raise JournalConflict('staged target differs; no serving cutover performed')
            if runtime.users.read(user_id) != observed:
                raise JournalConflict('source changed during staging; target is not approved for serving')
            lock.assert_owned()
            status = 'verified_staged_copy'
        return {'schema_version': 1, 'status': status, 'user_id': user_id, 'head': expected_head,
                'policy_sha256': runtime.policy.sha256, 'operation_version': str(rebuilt.state.operation_version),
                'feedback_version': str(rebuilt.state.feedback_version),
                'complete_sha256': hashlib.sha256(rebuilt.complete).hexdigest(), 'serving_cutover': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--user-id', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--target-collection')
    parser.add_argument('--stage-copy', action='store_true')
    args = parser.parse_args(argv)
    runtime = model_runtime()
    try:
        result = verify_and_stage(runtime, user_id=args.user_id, checkpoint_sha256=args.checkpoint_sha256,
            expected_head=args.expected_head, target_collection=args.target_collection, stage_copy=args.stage_copy)
        print(json.dumps(result, sort_keys=True))
        return 0
    finally:
        runtime.close()


if __name__ == '__main__':
    raise SystemExit(main())
