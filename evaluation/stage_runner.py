"""Offline production-component replay of bounded, frozen vector snapshots.

Qdrant local mode performs exact search, not server ANN. Timings measure this
replay, not the deployed pipeline. Loading supplied vectors does not encode text
or authenticate a snapshot's declared judgment provenance.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version as package_version
import hashlib
import json
import os
import math
import platform
from pathlib import Path
import subprocess
import time
from typing import Annotated, Literal
import uuid

from pydantic import Field, JsonValue, field_validator, model_validator
from evaluation.recommendation_benchmark import (
    Bundle, Commit, Digest, Number, Run, StrictModel, aware_timestamp, canonical_id, digest,
)

ROOT = Path(__file__).resolve().parents[1]
MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024


class FrozenProfile(StrictModel):
    profile_id: str
    vector: Annotated[list[Number], Field(min_length=384, max_length=384)]
    payload: dict[str, JsonValue]
    _id = field_validator('profile_id')(canonical_id)


class FrozenRepository(StrictModel):
    repo_id: str
    payload: dict[str, JsonValue]
    _id = field_validator('repo_id')(canonical_id)


class ArtifactIdentity(StrictModel):
    model_sha256: Digest
    scaler_sha256: Digest
    manifest_sha256: Digest


class FrozenSnapshot(StrictModel):
    schema_version: Literal[1]
    bundle: Bundle
    artifacts: ArtifactIdentity
    profiles: Annotated[list[FrozenProfile], Field(min_length=1, max_length=32)]
    repositories: Annotated[list[FrozenRepository], Field(min_length=1, max_length=256)]
    as_of: str
    limit: Annotated[int, Field(ge=1, le=100)] = 15
    seed: Annotated[str, Field(min_length=1, max_length=128)]
    _time = field_validator('as_of')(aware_timestamp)

    @model_validator(mode='after')
    def coherent(self):
        from embedding.vector_contract import validate_repository_payload
        by_repo = {row.repo_id: row for row in self.bundle.repositories}
        by_profile = {row.profile_id: row for row in self.bundle.profiles}
        if (len(self.repositories) != len(by_repo) or len(self.profiles) != len(by_profile)
                or {row.repo_id for row in self.repositories} != set(by_repo)
                or {row.profile_id for row in self.profiles} != set(by_profile)):
            raise ValueError('snapshot requires every unique bundle repository/profile exactly once')
        embedding = self.bundle.embedding
        from config import EMBEDDING_MODEL_REVISION, REPOSITORY_EMBEDDING_MODEL, REPOSITORY_EMBEDDING_VERSION
        if (embedding.model != REPOSITORY_EMBEDDING_MODEL or embedding.revision != EMBEDDING_MODEL_REVISION
                or embedding.dimension != 384 or embedding.pooling != 'mean'
                or embedding.query_prompt or embedding.document_prompt or embedding.recipe != REPOSITORY_EMBEDDING_VERSION):
            raise ValueError('frozen vectors must identify the serving embedding recipe/model/revision')
        for row in self.profiles:
            norm = math.sqrt(sum(x*x for x in row.vector))
            if not math.isfinite(norm) or abs(norm-1) > 1e-5:
                raise ValueError('frozen user vector must be finite normalized')
            p = row.payload
            for key, expected in [('embedding_dim',384), ('embedding_model',embedding.model),
                                  ('embedding_model_revision',embedding.revision), ('embedding_version',embedding.recipe)]:
                if p.get(key) != expected:
                    raise ValueError('frozen user payload embedding identity mismatch')
            for key in ('profile_version','last_feedback_version','feedback_state_revision'):
                value = p.get(key)
                if type(value) is not int or not (1 if key == 'profile_version' else 0) <= value <= 2**63-1:
                    raise ValueError('frozen user revisions must be exact bounded integers')
            for key in ('skills','tech_stack','interests','topics'):
                values=p.get(key)
                if (not isinstance(values,list) or len(values)>100
                        or any(not isinstance(v,str) or not 1<=len(v)<=128 for v in values)):
                    raise ValueError('frozen profile terms must be explicitly bounded lists')
            profile=by_profile[row.profile_id]
            if len(set(profile.seen)|set(profile.disliked))>500:
                raise ValueError('frozen exclusions exceed backend/ML contract')
        for row in self.repositories:
            p=row.payload; repo=by_repo[row.repo_id]
            validate_repository_payload(p)
            if (p['repo_id']!=row.repo_id or p['embedding_model']!=embedding.model
                    or p['embedding_model_revision']!=embedding.revision or p['embedding_version']!=embedding.recipe
                    or p['model_version']!=embedding.model
                    or p['feature_spec_version']!='v2' or p['content_version']<1):
                raise ValueError('frozen repository serving identity mismatch')
            # Zero is the authoritative absent feature-revision value.
            version=p.get('feature_version',0)
            if type(version) is not int or not 0<=version<=2**63-1:
                raise ValueError('frozen feature revision is invalid')
            if (p['star_count']!=repo.popularity or p['topics']!=repo.topics
                    or p.get('owner_github_id')!=repo.owner or p.get('fork_family_github_id')!=repo.fork_family):
                raise ValueError('bundle metrics metadata and frozen payload disagree')
        if self.bundle.dataset is not None:
            sources={row.repo_id:row for row in self.bundle.dataset.repositories}
            from evaluation.dataset_identity import common_ranking_material
            for row in self.repositories:
                source=sources[row.repo_id]
                material=common_ranking_material(row.payload,content_hash_origin=source.content_hash_origin)
                expected=common_ranking_material(source.ranking_material,content_hash_origin=source.content_hash_origin)
                if expected != material:
                    raise ValueError('raw dataset and actual ranking material differ')
            profiles={row.profile_id:row for row in self.bundle.dataset.profiles}
            for row in self.profiles:
                for key in ('skills','tech_stack','interests','topics'):
                    if getattr(profiles[row.profile_id],key) != row.payload[key]:
                        raise ValueError('raw dataset and actual profile context differ')
        return self


class StageRunExport(StrictModel):
    format: Literal['frozen-model-stages-v1'] = 'frozen-model-stages-v1'
    bundle_sha256: Digest
    snapshot_sha256: Digest
    artifacts: ArtifactIdentity
    source_sha256: dict[str, Digest]
    code_commit: Commit
    source_tree_dirty: bool
    as_of: str
    seed: str
    execution_scope: Literal['offline-local-qdrant-exact-cpu'] = 'offline-local-qdrant-exact-cpu'
    latency_scope: Literal['retrieval includes profile/vector hydration; ranking is heavy scoring; assembly is shape_batch; setup excluded'] = 'retrieval includes profile/vector hydration; ranking is heavy scoring; assembly is shape_batch; setup excluded'
    total_ms: dict[str, Number]
    source_scores: dict[str, dict[str, dict[str, Number]]]
    configuration: dict[str, JsonValue]
    libraries: dict[str, str]
    run: Run
    _time = field_validator('as_of')(aware_timestamp)

    @model_validator(mode='after')
    def bound_artifacts(self):
        if (self.run.model_sha256!=self.artifacts.model_sha256 or self.run.scaler_sha256!=self.artifacts.scaler_sha256
                or self.run.code_commit!=self.code_commit or self.run.timing_kind!='measured'
                or (self.run.manifest_sha256 is not None and self.run.manifest_sha256!=self.artifacts.manifest_sha256)):
            raise ValueError('stage export and actual run identity disagree')
        if set(self.source_scores)!={row.profile_id for row in self.run.outputs}:
            raise ValueError('source score/profile identity mismatch')
        for output in self.run.outputs:
            evidence=self.source_scores[output.profile_id]
            if set(evidence)!={row.repo_id for row in output.stages['retrieval']}:
                raise ValueError('source scores must cover the actual retrieved pool')
            for stage in output.stages.values():
                if any(row.repo_id not in evidence or evidence[row.repo_id].get(row.source)!=row.source_score for row in stage):
                    raise ValueError('stage primary source score differs from captured retrieval evidence')
        if set(self.total_ms)!={row.profile_id for row in self.run.outputs} or any(v<0 for v in self.total_ms.values()):
            raise ValueError('stage total timing/profile identity mismatch')
        return self


class OfflineGitUnavailable(RuntimeError):
    """The installed Git cannot enforce local-only source inspection."""


def require_offline_git():
    message = 'Frozen replay requires Git supporting --no-lazy-fetch; install a compatible Git before replay.'
    try:
        result = subprocess.run(['git', '--no-lazy-fetch', '--version'], cwd=ROOT,
            capture_output=True, timeout=5, check=False,
            env={**os.environ, 'GIT_NO_LAZY_FETCH':'1', 'GIT_TERMINAL_PROMPT':'0', 'GIT_OPTIONAL_LOCKS':'0'})
    except (OSError, subprocess.TimeoutExpired) as error:
        raise OfflineGitUnavailable(message) from error
    if result.returncode:
        raise OfflineGitUnavailable(message)


def source_identity():
    require_offline_git()
    def local_git(*args):
        return subprocess.check_output(
            ['git','--no-lazy-fetch','--no-pager',*args],cwd=ROOT,timeout=5,
            env={**os.environ,'GIT_NO_LAZY_FETCH':'1','GIT_TERMINAL_PROMPT':'0','GIT_OPTIONAL_LOCKS':'0'},
        )
    tracked=local_git('ls-files','-z','--','*.py','pyproject.toml','uv.lock').decode().split('\0')
    paths={ROOT/name for name in tracked if name}
    paths.update({Path(__file__),ROOT/'scripts/run_recommendation_stages.py'})
    for name in ('evaluation/variant_policy.py','evaluation/variant_comparison.py','scripts/compare_recommendation_variants.py'):
        if (ROOT/name).exists():paths.add(ROOT/name)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    commit=local_git('rev-parse','HEAD').decode().strip()
    dirty=bool(local_git('status','--porcelain','--untracked-files=normal'))
    return commit, dirty, hashes



def execute_snapshot(snapshot: FrozenSnapshot, artifact_dir: Path, *, treatment=None, social_evidence=None) -> StageRunExport:
    """Run actual components only against owned in-memory collections.

    No encoder download, remote Qdrant/Redis/SQL/GitHub, live social projection lookup, model
    promotion or production environment mutation. The CLI bounds worker lifetime.
    """
    # Own every nested list/payload, not just the outer Pydantic object. The
    # caller may mutate and restore its copy while actual workers are running.
    raw=json.dumps(snapshot.model_dump(mode='json'),sort_keys=True,allow_nan=False,separators=(',',':')).encode()
    if len(raw)>MAX_SNAPSHOT_BYTES:
        raise ValueError('frozen snapshot exceeds replay byte bound')
    snapshot=FrozenSnapshot.model_validate_json(raw)
    frozen_snapshot_sha=digest(snapshot)
    from evaluation.variant_policy import FrozenTreatment,FrozenVariantContext
    selected=FrozenTreatment(name='baseline',diversity=None,social=None) if treatment is None else treatment
    context=FrozenVariantContext(treatment=selected,social_evidence=[] if social_evidence is None else social_evidence)
    # Independently own every treatment/evidence container before any workers.
    context=FrozenVariantContext.model_validate_json(json.dumps(context.model_dump(mode='json'),sort_keys=True,allow_nan=False))
    context.validate_for(snapshot)
    context_sha=digest(context)
    projections={row.viewer_id:row for row in context.social_evidence}
    import torch
    from qdrant_client import QdrantClient, models
    from inference.ranker_service import RankerService
    from inference.feed_assembly import FeedAssemblySystem
    from inference.diversity import DiversityPolicy
    from retrieval.v2_retriever import QdrantV2Retriever, QDRANT_VECTOR_NAME
    from retrieval.social_candidates import SocialPolicy
    from utils.deadline import RequestDeadline, deadline_scope, check_deadline
    names={'model_sha256':'heavy_ranker.pt','scaler_sha256':'feature_scaler.json','manifest_sha256':'model_manifest.json'}
    for key,name in names.items():
        maximum=128*1024*1024 if key=='model_sha256' else 2*1024*1024
        if (artifact_dir/name).stat().st_size>maximum:
            raise ValueError('frozen artifact exceeds offline replay bound')
    before={key:hashlib.sha256((artifact_dir/name).read_bytes()).hexdigest() for key,name in names.items()}
    if before!=snapshot.artifacts.model_dump():
        raise ValueError('requested frozen artifact identity does not match files')
    ranker=RankerService(model_path=str(artifact_dir/'heavy_ranker.pt'),scaler_path=str(artifact_dir/'feature_scaler.json'),
        manifest_path=str(artifact_dir/'model_manifest.json'), require_production_manifest=True, device='cpu')
    if (ranker.embedding_model!=snapshot.bundle.embedding.model or ranker.embedding_model_revision!=snapshot.bundle.embedding.revision
            or ranker.embedding_version!=snapshot.bundle.embedding.recipe):
        raise ValueError('loaded ranker embedding identity differs from frozen vectors')
    if ranker.loaded_artifact_sha256!=before:
        raise ValueError('loaded artifact bytes differ from frozen identities')
    commit,dirty,hashes=source_identity()
    reference=datetime.fromisoformat(snapshot.as_of.replace('Z','+00:00'))

    class RecordingAssembler(FeedAssemblySystem):
        def shape_batch(self, ranked, **kwargs):
            started=time.perf_counter()
            shaped=super().shape_batch(ranked,**{**kwargs,'reference_time':reference})
            self.elapsed_ms+=(time.perf_counter()-started)*1000
            self.calls+=1
            return shaped

    def candidates(items):
        result=[]
        for item in items:
            # Keep the primary source score separate from the stage rank score.
            # The complete source map is also retained in the export.
            if item.source not in item.retrieval_scores:
                raise ValueError('primary candidate source score is missing')
            result.append(dict(repo_id=item.repo_id,score=float(item.score),source=item.source,
                               source_score=float(item.retrieval_scores[item.source])))
        return result

    class RecordingRetriever(QdrantV2Retriever):
        def _heavy_rank(self, **kwargs):
            started=time.perf_counter()
            self.retrieval_ms=(started-self.began)*1000
            self.retrieved=candidates(kwargs['ranked'])
            self.source_scores={item.repo_id:dict(item.retrieval_scores) for item in kwargs['ranked']}
            scoring_started=time.perf_counter()
            value=super()._heavy_rank(**kwargs)
            self.ranking_ms=(time.perf_counter()-scoring_started)*1000
            if value[0] is None:
                raise ValueError('actual heavy ranking did not produce a complete stage')
            self.ranked=candidates(value[0])
            return value

    class FrozenClient(QdrantClient):
        def __init__(self):
            super().__init__(location=':memory:')
            self.read_errors=[]
        def _read(self, name, function, *args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception:
                self.read_errors.append(name)
                raise
        def retrieve(self,*args,**kwargs):
            return self._read('retrieve',super().retrieve,*args,**kwargs)
        def query_points(self,*args,**kwargs):
            return self._read('query_points',super().query_points,*args,**kwargs)
        def scroll(self,*args,**kwargs):
            return self._read('scroll',super().scroll,*args,**kwargs)

    with closing(FrozenClient()) as client:
        client.create_collection('frozen_repos',vectors_config={QDRANT_VECTOR_NAME:models.VectorParams(size=384,distance=models.Distance.COSINE)})
        client.create_collection('frozen_users',vectors_config=models.VectorParams(size=384,distance=models.Distance.COSINE))
        repos={r.repo_id:r for r in snapshot.bundle.repositories}
        client.upsert('frozen_repos',points=[models.PointStruct(id=r.repo_id,vector={QDRANT_VECTOR_NAME:repos[r.repo_id].vector},payload=r.payload)
            for r in snapshot.repositories if not repos[r.repo_id].archived])
        client.upsert('frozen_users',points=[models.PointStruct(id=r.profile_id,vector=r.vector,payload=r.payload) for r in snapshot.profiles])
        assembler=RecordingAssembler()
        engine=RecordingRetriever(client=client, repository_collection='frozen_repos',user_collection='frozen_users',
            assembler=assembler,ranker=ranker,heavy_ranker_enabled=True,heavy_ranker_required=True,heavy_ranker_traffic_percent=100,
            allow_unqualified_ranker=False,minimum_eligible_repositories=0,max_candidates=150,
            compatible_embedding_versions={snapshot.bundle.embedding.recipe},social_policy=SocialPolicy(mode='disabled'),candidate_vector_mode='deferred')
        # This explicitly selected offline policy changes only our owned engine.
        engine.diversity_mode='enabled' if context.treatment.diversity is not None else 'disabled'
        engine.diversity_traffic_percent=100 if context.treatment.diversity is not None else 0
        if context.treatment.diversity is not None:
            engine.diversity_policy=DiversityPolicy(**context.treatment.diversity)
        if context.treatment.social is not None:
            engine.social_policy=SocialPolicy(**context.treatment.social)
        engine.required_content_version=1;engine.required_feature_spec_version='v2'
        engine.allow_missing_embedding_revision=False
        engine.embedding_model_revision=snapshot.bundle.embedding.revision
        configuration={'retrieval':{key:getattr(engine,key) for key in (
            'max_candidates','candidate_vector_mode','qdrant_timeout_seconds','required_content_version',
            'required_feature_spec_version','compatible_embedding_versions','allow_missing_embedding_revision',
            'heavy_ranker_enabled','heavy_ranker_required','heavy_ranker_traffic_percent','diversity_mode',
            'diversity_traffic_percent')},'assembly':{key:getattr(assembler,key) for key in (
                'freshness_window_hours','freshness_boost','explore_fraction','max_same_language',
                'generation_reference_capacity','generation_reference_ttl_seconds')},
            'social':asdict(engine.social_policy),'diversity':asdict(engine.diversity_policy),
            'runtime':{'repository_vector_name':QDRANT_VECTOR_NAME,'overall_cooperative_deadline_seconds':120,'max_candidate_pool':150,
                       'external_social_evidence':bool(context.social_evidence),'device':str(ranker.device),'requested_limit':snapshot.limit,
                       'warmup':'none; first profile includes first actual forward',
                       'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
                       'archived_policy':'excluded from local eligible catalogue before retrieval'}}
        configuration['variant_context_sha256']=context_sha
        configuration['treatment']=context.treatment.model_dump(mode='json')
        configuration['social_projection_declarations']=[{'viewer_id':row.viewer_id,'observed_at':row.observed_at,
            'declared_origin':row.declared_origin,'declared_policy':row.declared_policy,'reference':row.reference,
            'candidate_count':len(row.candidates)} for row in context.social_evidence]
        configuration['retrieval']['compatible_embedding_versions']=sorted(engine.compatible_embedding_versions)
        from retrieval.runtime import retrieval_runtime_status
        from retrieval.v2_retriever import _CANDIDATE_READ_MULTIPLIER, _MAX_RETRIEVAL_ROUNDS
        configuration['runtime'].update(candidate_read_multiplier=_CANDIDATE_READ_MULTIPLIER,max_retrieval_rounds=_MAX_RETRIEVAL_ROUNDS)
        outputs=[];totals={};source_scores={};shape_calls={}
        with deadline_scope(RequestDeadline(120)):
            for profile in snapshot.bundle.profiles:
                check_deadline()
                generation=str(uuid.uuid5(uuid.NAMESPACE_URL,snapshot.seed+':'+profile.profile_id))
                assembler.elapsed_ms=0.;assembler.calls=0
                engine.began=time.perf_counter()
                engine.retrieved=engine.ranked=None
                projection=projections.get(profile.profile_id)
                batch=engine.recommend_batch(profile.profile_id,snapshot.limit,sorted(set(profile.seen)|set(profile.disliked)),generation,
                    social_candidates=projection.candidates if projection is not None else [])
                elapsed=(time.perf_counter()-engine.began)*1000
                if client.read_errors or not batch.ranker_applied or batch.served_ranker!='heavy' or batch.fallback_code or engine.retrieved is None:
                    raise ValueError('offline stage replay cannot export partial/fallback evidence')
                outputs.append(dict(profile_id=profile.profile_id,stages={'retrieval':engine.retrieved,'ranking':engine.ranked,'assembly':candidates(batch.items)},
                    latency_ms={'retrieval':engine.retrieval_ms,'ranking':engine.ranking_ms,'assembly':assembler.elapsed_ms}))
                totals[profile.profile_id]=elapsed
                source_scores[profile.profile_id]=engine.source_scores
                shape_calls[profile.profile_id]=assembler.calls
        configuration['runtime']['retrieval_executor']=retrieval_runtime_status()
        configuration['runtime']['shape_calls_by_profile']=shape_calls
    if digest(context)!=context_sha:
        raise ValueError('owned frozen treatment/evidence changed during replay')
    if digest(snapshot)!=frozen_snapshot_sha:
        raise ValueError('owned frozen snapshot changed during replay')
    if before!={key:hashlib.sha256((artifact_dir/name).read_bytes()).hexdigest() for key,name in names.items()} or source_identity()!=(commit,dirty,hashes):
        raise ValueError('source/artifact identity changed during replay')
    run=Run.model_validate(dict(schema_version=1,embedding=snapshot.bundle.embedding.model_dump(),model_id=ranker.model_version,
        manifest_sha256=before['manifest_sha256'],model_sha256=before['model_sha256'],scaler_sha256=before['scaler_sha256'],feature_schema=ranker.feature_spec_version,
        code_commit=commit,created_at=datetime.now(timezone.utc).isoformat(),timing_kind='measured',outputs=outputs))
    return StageRunExport(bundle_sha256=digest(snapshot.bundle),snapshot_sha256=frozen_snapshot_sha,artifacts=snapshot.artifacts,
        source_sha256=hashes,code_commit=commit,source_tree_dirty=dirty,as_of=snapshot.as_of,seed=snapshot.seed,total_ms=totals,source_scores=source_scores,configuration=configuration,
        libraries={'python':platform.python_version(),**{name:package_version(name) for name in ('torch','numpy','qdrant-client','pydantic')}},run=run)
