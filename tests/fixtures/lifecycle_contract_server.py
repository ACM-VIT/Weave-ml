"""Loopback lifecycle wire fixture: real routes, CAS, local Qdrant and retrieval.

Only the encoder and distributed lease are replaced with deterministic local
boundaries. No network model, Redis, product database or GitHub is contacted.
"""
import os
import socket
from contextlib import nullcontext

os.environ['V2_REPOSITORY_LIFECYCLE_POLICY'] = 'verified-status-v1'
os.environ['INTERNAL_API_SECRET'] = 'local-contract-secret'
os.environ['APP_ENV'] = 'development'

import uvicorn
from qdrant_client import QdrantClient, models
from api import v2
from api.main import app
from embedding.embedding_pipeline import RepositoryEmbeddingPipeline
from embedding.qdrant_store import QdrantRepositoryStore
from retrieval.v2_retriever import QdrantV2Retriever
from config import QDRANT_VECTOR_NAME, REPOSITORY_EMBEDDING_DIM

class Encoder:
    def embed_texts(self, texts, *, normalize=True):
        return [[1.0] + [0.0] * (REPOSITORY_EMBEDDING_DIM - 1) for _ in texts]

class Lock:
    def assert_owned(self):
        pass

client = QdrantClient(':memory:')
store = QdrantRepositoryStore(client=client, collection_name='lifecycle-fixture', corpus_authority=None)
store.ensure_collection()
client.create_collection('lifecycle-users', vectors_config=models.VectorParams(
    size=REPOSITORY_EMBEDDING_DIM, distance=models.Distance.COSINE))
pipeline = RepositoryEmbeddingPipeline(embedder=Encoder())
retriever = QdrantV2Retriever(client=client, repository_collection=store.collection_name,
    user_collection='lifecycle-users', heavy_ranker_enabled=False)
v2.repository_store = lambda: store
v2.repository_embedding_pipeline = lambda: pipeline
v2._repository_job_lock = lambda _: nullcontext(Lock())
v2.retriever = lambda: retriever
listener = socket.socket()
listener.bind(('127.0.0.1', 0))
listener.listen(128)
print(f'CONTRACT_PORT={listener.getsockname()[1]}', flush=True)
uvicorn.Server(uvicorn.Config(app, lifespan='off', log_level='error')).run(sockets=[listener])
