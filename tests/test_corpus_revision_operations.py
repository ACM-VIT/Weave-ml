"""Managed writer and API acceptance separate from the neutral authority protocol."""
import json
from types import SimpleNamespace
from uuid import UUID, uuid4
import pytest
from tests.test_corpus_revision import authority, isolated_redis, proof
from tests.test_v2_retrieval import local_recommendation_corpus, _corpus_retriever








@pytest.mark.integration
def test_real_authority_invalidates_local_qdrant_cached_pool_after_delete_and_replace(authority,local_recommendation_corpus):
    from qdrant_client import models
    from config import QDRANT_VECTOR_NAME, REPOSITORY_EMBEDDING_DIM
    from retrieval.discovery_cache import DiscoveryRevision
    corpus=local_recommendation_corpus
    retriever=_corpus_retriever(corpus,heavy=False)
    def revision():
        current=authority.read()
        return DiscoveryRevision(*current) if current else None
    retriever._discovery_revision_provider=revision
    authority.bootstrap(proof())
    # Four overlapping 25-point channel heads satisfy20; requesting50 would
    # correctly require exclusion-driven refill, covered independently.
    first=retriever._discovery(20,include_vectors=False,exclude_repo_ids=set())
    assert len(first)==20 and corpus.client.scroll.call_count==4
    repeated=retriever._discovery(20,include_vectors=False,exclude_repo_ids=set())
    assert {row[0].repo_id for row in first}=={row[0].repo_id for row in repeated}
    assert corpus.client.scroll.call_count==4
    removed=first[0][0].repo_id
    authority.mutate(lambda: corpus.client.delete(corpus.repository_collection,
        points_selector=models.PointIdsList(points=[removed]),wait=True))
    after_delete=retriever._discovery(20,include_vectors=False,exclude_repo_ids=set())
    assert removed not in {row[0].repo_id for row in after_delete}
    assert corpus.client.scroll.call_count==8
    def replace():
        assert authority.read() is None
        assert corpus.client.delete_collection(corpus.repository_collection) is True
        assert corpus.client.create_collection(corpus.repository_collection,vectors_config={
            QDRANT_VECTOR_NAME:models.VectorParams(size=REPOSITORY_EMBEDDING_DIM,distance=models.Distance.COSINE)
        }) is True
        return True
    authority.mutate(replace,synchronous_boolean=True)
    assert retriever._discovery(20,include_vectors=False,exclude_repo_ids=set())==[]
    assert corpus.client.scroll.call_count==12



@pytest.mark.integration
def test_repository_store_fences_every_supported_mutation(authority):
    from qdrant_client import QdrantClient
    from embedding.qdrant_store import QdrantRepositoryStore
    from tests.test_v2_qdrant_fencing import _repository_result, REPO_ID
    from summarization.pipeline import description_fallback_artifact
    client=QdrantClient(':memory:')
    store=QdrantRepositoryStore(client=client,corpus_authority=authority)
    authority.bootstrap(proof())
    try:
        previous=authority.read()
        def changed():
            nonlocal previous
            current=authority.read()
            assert current is not None and current!=previous
            previous=current
        store.ensure_collection(); changed()
        first=_repository_result(1,str(uuid4()))
        store.upsert([first]); changed()
        point=client.retrieve(store.collection_name,[REPO_ID],with_payload=True,with_vectors=True)[0]
        point=store.compare_and_set_content(_repository_result(2,str(uuid4())),expected_point=point); changed()
        point=store.compare_and_set_features(expected_point=point,feature_payload={
            'feature_version':1,'feature_job_id':str(uuid4()),'star_count':20,
        }); changed()
        artifact=description_fallback_artifact({'description':'A current isolated card summary.'})
        store.compare_and_set_card_summary(expected_point=point,artifact=artifact); changed()
        store.delete_repositories([REPO_ID]); changed()
        assert client.retrieve(store.collection_name,[REPO_ID])==[]
        store.upsert([_repository_result(3,str(uuid4()))]); changed()
        store.replace_empty_collection(proof=proof()); changed()
        assert client.count(store.collection_name).count==0
    finally:
        client.close()



@pytest.fixture
def owned_authority_services():
    """Fresh owned-bridge Redis/Qdrant, loopback ephemeral ports only."""
    import subprocess
    import time
    import os
    if os.getenv("RUN_CORPUS_AUTHORITY_INTEGRATION") != "1":
        pytest.skip("explicit owned-service authority integration required")
    import redis
    from qdrant_client import QdrantClient
    suffix=uuid4().hex[:10]
    network='weave-authority-'+suffix
    names=['weave-authority-redis-'+suffix,'weave-authority-qdrant-'+suffix]
    subprocess.run(['docker','network','create',network],check=True,capture_output=True)
    try:
        for name,image,port in [(names[0],'redis:7.4.5-alpine',6379),(names[1],'qdrant/qdrant:v1.18.2',6333)]:
            command=['docker','run','--detach','--pull=never','--name',name,'--network',network,
                     '--publish',f'127.0.0.1::{port}',image]
            if port==6379: command+=['redis-server','--save','','--appendonly','no']
            subprocess.run(command,check=True,capture_output=True)
        def endpoint(name,port,scheme):
            binding=subprocess.run(['docker','port',name,str(port)+'/tcp'],check=True,capture_output=True,text=True).stdout.strip()
            assert binding.startswith('127.0.0.1:')
            return scheme+'://'+binding
        redis_url=endpoint(names[0],6379,'redis')
        qdrant_url=endpoint(names[1],6333,'http')
        redis_client=redis.from_url(redis_url,socket_timeout=1,socket_connect_timeout=1)
        qdrant=QdrantClient(url=qdrant_url,timeout=1,check_compatibility=False)
        try:
            deadline=time.monotonic()+20
            while True:
                try:
                    redis_client.ping(); qdrant.get_collections(); break
                except Exception:
                    if time.monotonic()>=deadline: raise
                    time.sleep(.05)
            yield redis_url,qdrant_url
        finally:
            redis_client.close(); qdrant.close()
    finally:
        for name in names:
            subprocess.run(['docker','rm','--force',name],check=False,capture_output=True)
        subprocess.run(['docker','network','rm',network],check=False,capture_output=True)


@pytest.mark.integration
def test_actual_api_factories_and_server_write_settling_after_lost_ack(owned_authority_services,monkeypatch):
    import threading
    import time
    import hashlib
    from pathlib import Path
    from dataclasses import replace
    from unittest.mock import Mock
    from api import v2
    import retrieval.v2_retriever as retrieval_module
    from embedding.repository_embedding import RepositoryEmbeddingConfig
    from tests.test_v2_qdrant_fencing import _repository_result, REPO_ID
    from embedding.corpus_revision import CorpusAuthorityUnavailable
    redis_url,qdrant_url=owned_authority_services
    for name,value in {
        'CORPUS_AUTHORITY_ENABLED':'true',
        'CORPUS_AUTHORITY_MANAGED_BOUNDARY':'single-endpoint-fenced-admin-v1',
        'CORPUS_AUTHORITY_REDIS_URL':redis_url,'QDRANT_URL':qdrant_url,
        'APP_ENV':'development','MIN_ELIGIBLE_REPOSITORIES':'0',
        'V2_HEAVY_RANKER_ENABLED':'false','V2_HEAVY_RANKER_REQUIRED':'false',
        'V2_HEAVY_RANKER_TRAFFIC_PERCENT':'0',
    }.items(): monkeypatch.setenv(name,value)
    monkeypatch.setattr(retrieval_module,'QDRANT_URL',qdrant_url)
    monkeypatch.setattr(retrieval_module,'QDRANT_API_KEY',None)
    monkeypatch.setattr(v2,'repository_embedding_pipeline',lambda: SimpleNamespace(config=RepositoryEmbeddingConfig()))
    for factory in [v2.repository_store,v2.retriever,v2.corpus_authority]: factory.cache_clear()
    release=threading.Event(); settled=threading.Event(); thread=None
    authority=None; reader=None
    try:
        authority=v2.corpus_authority()
        assert authority.read() is None
        authority.bootstrap(proof())
        reader=v2.retriever(); store=v2.repository_store()
        assert store.corpus_authority is authority
        assert reader.discovery_cache_status()['revision_provider_configured'] is True
        points=[_repository_result(1,str(uuid4()))]
        for index in range(1,20):
            result=_repository_result(1,str(uuid4()))
            repo_id=str(UUID(int=711+index))
            payload={**result.payload,'repo_id':repo_id,'full_name':f'fixture/repository-{index}',
                     'primary_language':['Python','Rust','Go','JavaScript'][index%4]}
            points.append(replace(result,repo_id=repo_id,payload=payload))
        store.upsert(points)
        reader.client.scroll=Mock(wraps=reader.client.scroll)
        authority.client.eval=Mock(wraps=authority.client.eval)
        began=time.perf_counter()
        first=reader._discovery(10,include_vectors=False,exclude_repo_ids=set())
        cold_ms=(time.perf_counter()-began)*1000
        cold_authority_reads=authority.client.eval.call_count
        began=time.perf_counter()
        repeat=reader._discovery(10,include_vectors=False,exclude_repo_ids=set())
        warm_ms=(time.perf_counter()-began)*1000
        warm_authority_reads=authority.client.eval.call_count-cold_authority_reads
        warm_status=reader.discovery_cache_status()
        assert len(first)==len(repeat)==10
        assert [row[0].repo_id for row in first]==[row[0].repo_id for row in repeat]
        assert warm_status['hits']==1 and warm_status['misses']==1
        assert 0<=warm_status['oldest_pool_age_seconds']<15
        assert first[0][0].repo_id==repeat[0][0].repo_id==REPO_ID
        assert reader.client.scroll.call_count==4
        raw_upsert=store.client.upsert
        outcomes=[]
        def delayed_server_write(**kwargs):
            nonlocal thread
            def later():
                try:
                    assert release.wait(5)
                    outcomes.append(raw_upsert(**kwargs))
                finally: settled.set()
            thread=threading.Thread(target=later)
            thread.start()
            raise TimeoutError('client acknowledgment lost while submitted work can still settle')
        monkeypatch.setattr(store.client,'upsert',delayed_server_write)
        with pytest.raises(TimeoutError): store.upsert([_repository_result(2,str(uuid4()))])
        assert authority.read() is None and not settled.is_set()
        # Recovery never happens from a quiet timeout; the old writer token lives.
        assert authority.status()['writers']==1
        release.set(); assert settled.wait(5)
        thread.join(2); assert not thread.is_alive()
        assert outcomes and getattr(outcomes[0].status,'value',outcomes[0].status)=='completed'
        assert authority.read() is None  # Even actual settlement does not guess recovery.
        current=reader._discovery(10,include_vectors=False,exclude_repo_ids=set())
        assert current[0][0].payload['content_version']==2
        assert reader.client.scroll.call_count==8
        authority.bootstrap(proof())  # All fixture writers exited; completed outcome observed.
        after=reader._discovery(10,include_vectors=False,exclude_repo_ids=set())
        assert after[0][0].payload['content_version']==2
        assert reader.client.scroll.call_count==12
        print(json.dumps({
            'scope':'20 synthetic canonical points,10 retained; actual API factories/Redis7.4.5/Qdrant1.18.2; no production performance or recommendation quality',
            'cold_discovery_reads':4,'warm_additional_reads':0,'dirty_additional_reads':4,
            'recovered_additional_reads':4,'cold_ms':cold_ms,'warm_ms':warm_ms,
            'cold_authority_evals':cold_authority_reads,'warm_authority_evals':warm_authority_reads,
            'warm_hit_rate':warm_status['hits']/(warm_status['hits']+warm_status['misses']),
            'warm_pool_age_seconds':warm_status['oldest_pool_age_seconds'],
            'retained_languages':len({row[0].payload['primary_language'] for row in after}),
            'updated_repository_content_version':after[0][0].payload['content_version'],
            'retriever_source_sha256':hashlib.sha256(Path(retrieval_module.__file__).read_bytes()).hexdigest(),
        },sort_keys=True))
    finally:
        release.set()
        if thread is not None: thread.join(5)
        if reader is not None: reader.client.close()
        if authority is not None: authority.close()
        for factory in [v2.repository_store,v2.retriever,v2.corpus_authority]: factory.cache_clear()
