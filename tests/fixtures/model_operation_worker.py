"""Test-only abrupt-death worker. Receives owned loopback service settings."""
import json
import os
from pathlib import Path
import sys

from qdrant_client import QdrantClient
import redis

from feedback.model_runtime import ModelRuntime
from feedback.v2_settings import V2FeedbackSettings


def main():
    values=json.loads(Path(sys.argv[1]).read_text());stage=sys.argv[2]
    settings=V2FeedbackSettings(**values['settings'])
    runtime=ModelRuntime(values['config'],settings=settings,
        redis=redis.Redis.from_url(settings.redis_url,decode_responses=True),
        qdrant=QdrantClient(url=settings.qdrant_url,api_key=None,timeout=3),
        encoder=lambda context:[1.]+[0.]*383)
    if stage=='before_prepare':
        put=runtime.journal.put
        def fail(kind,value):
            if kind=='prepare':os._exit(71)
            return put(kind,value)
        runtime.journal.put=fail
    elif stage=='before_cas':
        runtime.users.compare_and_set=lambda *_:os._exit(71)
    elif stage=='after_cas_before_ack':
        runtime.stream.acknowledge=lambda *_:os._exit(71)
    else:raise ValueError('unknown fixed test crash boundary')
    runtime.consumer().run_once()
    raise RuntimeError('crash hook was not reached')


if __name__=='__main__':main()
