"""Disposable loopback server for the Node/Python wire-contract test.

Routes, Pydantic contracts, HTTP authentication and admission are real. External
model/storage boundaries are replaced; this is not distributed persistence QA.
"""
import json
import os
import uuid
import socket
from types import SimpleNamespace

os.environ['INTERNAL_API_SECRET'] = 'local-contract-secret'
os.environ['APP_ENV'] = 'development'

import uvicorn
from api import v2
from api.main import app
from feedback.v2 import FeedbackEventIdConflictError
from retrieval.v2_retriever import RankedRepository, RecommendationBatch


def onboard(request):
    return {'accepted': True, 'status': 'applied', 'user_id': str(request.user_id),
            'profile_version': request.profile_version}


def embed(request, deadline=None):
    return {'schema_version': 2, 'accepted': True, 'status': 'applied',
            'job_id': str(request.job_id), 'repo_id': str(request.repo_id),
            'content_version': request.content_version, 'embedding_version': 'contract-embedding-v2',
            'card_summary': {'summary': request.repository.description, 'highlights': [],
                             'prompt_version': 'repo-card-summary-v1',
                             'model_version': 'repository-description-fallback-v1',
                             'format_version': 'repo-card-summary-json-v1',
                             'source': 'description_fallback'}}


def recommend(*args):
    return RecommendationBatch(
        items=[RankedRepository('00000000-0000-4000-8000-000000000003', .8, 'semantic')],
        model_version='contract-ranker-v2', embedding_version='contract-embedding-v2',
        ranker_applied=True, served_ranker='heavy', heavy_ranker_selected=True,
    )


accepted_events = {}

def feedback(events):
    accepted = duplicates = 0
    for event in events:
        assert isinstance(event['feedback_version'], int)
        assert event['event_id'] == str(uuid.UUID(event['event_id']))
        fingerprint = json.dumps(event, sort_keys=True)
        if event['event_id'] in accepted_events:
            if accepted_events[event['event_id']] != fingerprint:
                raise FeedbackEventIdConflictError(accepted=accepted, duplicates=duplicates,
                                                   failed_event_id=event['event_id'])
            duplicates += 1
        else:
            accepted_events[event['event_id']] = fingerprint
            accepted += 1
    return accepted, duplicates


v2._onboard_user_job = onboard
v2._embed_repository_job = embed
v2._refresh_repository_job = lambda request: {
    'accepted': True, 'status': 'applied', 'repo_id': str(request.repo_id),
    'feature_version': request.feature_version,
}
v2._enqueue_feedback = feedback
v2.retriever = lambda: SimpleNamespace(recommend_batch=recommend)
listener = socket.socket()
listener.bind(('127.0.0.1', 0))
listener.listen(128)
print(f'CONTRACT_PORT={listener.getsockname()[1]}', flush=True)
uvicorn.Server(uvicorn.Config(app, lifespan='off', log_level='error')).run(sockets=[listener])
