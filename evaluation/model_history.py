"""Frozen single-vector versus long/session diagnostics using actual stages.

History, corpus, baseline vectors, policy, artifacts and as-of are common input.
Derived user vectors differ deliberately. Synthetic labels stay synthetic; this
runner never promotes a policy, trains a model or writes a live user point.
"""
from __future__ import annotations

from dataclasses import asdict, fields
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Literal

from pydantic import Field, model_validator

from evaluation.recommendation_benchmark import StrictModel, digest, evaluate
from evaluation.stage_runner import FrozenSnapshot, StageRunExport, execute_snapshot
from feedback import long_session as model
from feedback.model_wire import parse_operation, decimal
from feedback.v2 import OrderedFeedbackApplier, ALPHAS


class FrozenLegacyPolicy(StrictModel):
    dwell_min_ms: Annotated[int, Field(ge=3000)]
    dwell_full_credit_ms: Annotated[int, Field(ge=3000)]
    dwell_max_alpha: Annotated[float, Field(ge=0, le=1)]
    max_tracked_repositories: Annotated[int, Field(ge=1, le=256)]
    max_user_state_bytes: Annotated[int, Field(ge=1, le=1048576)]
    vector_dimension: Literal[384]
    user_vector_name: None
    repository_vector_name: Literal['repo_embedding']



class FrozenModelHistory(StrictModel):
    schema_version: Literal[1]
    snapshot: FrozenSnapshot
    policy: dict
    legacy_policy: FrozenLegacyPolicy
    operations: Annotated[list[str], Field(min_length=1, max_length=1000)]
    sessions: dict[str, dict]
    k: Annotated[int, Field(ge=1, le=100)]

    def explicit_policy(self):
        values = dict(self.policy)
        if set(values) != {field.name for field in fields(model.Policy)}:
            raise ValueError('every model policy value must be explicit')
        values['embedding'] = model.EmbeddingIdentity(**values['embedding'])
        policy = model.Policy(**values)
        if (policy.embedding.model != self.snapshot.bundle.embedding.model
                or policy.embedding.revision != self.snapshot.bundle.embedding.revision
                or policy.repository_recipe != self.snapshot.bundle.embedding.recipe):
            raise ValueError('history policy and frozen embedding identity differ')
        return policy

    @model_validator(mode='after')
    def coherent_history(self):
        policy = self.explicit_policy()
        owners = {profile.profile_id for profile in self.snapshot.profiles}
        if set(self.sessions) != owners or self.k > self.snapshot.limit:
            raise ValueError('one explicit session selection per frozen profile required')
        for profile in self.snapshot.profiles:
            if profile.payload['last_feedback_version'] or profile.payload['feedback_state_revision']:
                raise ValueError('history requires an explicit zero-feedback profile genesis basis')
        seen = set()
        for raw in self.operations:
            wire = parse_operation(raw, maximum_bytes=131072, allowed_policies={policy.sha256})
            if wire.header.user_id not in owners or wire.header.operation_id in seen:
                raise ValueError('history owner or operation identity differs')
            seen.add(wire.header.operation_id)
        return self


def derive_snapshots(experiment):
    policy = experiment.explicit_policy()
    original = experiment.snapshot
    profiles = {row.profile_id: row for row in original.profiles}
    repositories = {row.repo_id: row for row in original.bundle.repositories}
    repository_payloads = {row.repo_id: row.payload for row in original.repositories}
    states, legacy = {}, {}
    applier = OrderedFeedbackApplier(qdrant=SimpleNamespace(), settings=SimpleNamespace(**experiment.legacy_policy.model_dump()))
    as_of_ms = int(datetime.fromisoformat(original.as_of.replace('Z', '+00:00')).timestamp() * 1000)
    for raw in experiment.operations:
        wire = parse_operation(raw, maximum_bytes=131072, allowed_policies={policy.sha256})
        h, payload, owner = wire.header, wire.payload, wire.header.user_id
        if h.accepted_ms > as_of_ms:
            raise ValueError('history operation is later than frozen as-of')
        if wire.kind == 'profile':
            if owner in states or not payload['allow_genesis'] or decimal(payload['profile_version'], 1) != profiles[owner].payload['profile_version']:
                raise ValueError('each profile requires exactly one explicit frozen genesis')
            if any(profiles[owner].payload.get(key) != value for key, value in payload['original_profile_context'].items()):
                raise ValueError('original profile context differs from frozen encoded baseline')
            pinned = model.PinnedVector.pin('profile', owner, decimal(payload['profile_version'], 1), 0,
                                            policy.baseline_recipe, policy.embedding, profiles[owner].vector)
            states[owner] = model.initial_state(owner, pinned, policy, accepted_ms=h.accepted_ms)
            operation = model.Profile(h, pinned, True)
            legacy[owner] = SimpleNamespace(id=owner, vector=list(profiles[owner].vector), payload=dict(profiles[owner].payload))
        elif owner not in states:
            raise ValueError('operation precedes explicit profile genesis')
        elif wire.kind == 'begin':
            operation = model.Begin(h, payload['session_id'], decimal(payload['epoch_started_ms']), payload['witness_sha256'])
        elif wire.kind == 'reset':
            operation = model.Reset(h, payload['session_id'], decimal(payload['old_epoch'], 1), decimal(payload['new_epoch'], 1), payload['witness_sha256'])
        elif wire.kind == 'end':
            operation = model.End(h, payload['session_id'], decimal(payload['epoch'], 1), decimal(payload['epoch_started_ms']))
        else:
            event = applier.normalize_event(wire.feedback)
            repo = repositories.get(event['repo_id'])
            source = repository_payloads.get(event['repo_id'])
            if repo is None or source is None or repo.archived:
                raise ValueError('history repository missing from frozen eligible corpus')
            pinned = None if event['event_type'] in {'unlike', 'undislike', 'unsave'} else model.PinnedVector.pin(
                'repository', repo.repo_id, source['content_version'], source.get('feature_version', 0), policy.repository_recipe,
                policy.embedding, repo.vector)
            binding = payload.get('canonical_attribution')
            session = None if binding is None else model.SessionBinding(owner, binding['session_id'], decimal(binding['epoch'], 1),
                decimal(binding['epoch_started_ms']), binding['active_at_acceptance'], binding['witness_sha256'])
            operation = model.Feedback(h, event['event_id'], event['feedback_version'], repo.repo_id, event['event_type'],
                                      pinned, session, event.get('dwell_ms'), None)
            result, point = applier.prepare(wire.feedback, user=legacy[owner], repository_vector=lambda _id, _dim: repo.vector)
            if result.status != 'applied' or point is None:
                raise ValueError('legacy history did not advance its exact feedback predecessor')
            legacy[owner] = point
        states[owner] = model.reduce_operation(states[owner], operation, policy)
    if set(states) != set(profiles):
        raise ValueError('history omitted a frozen profile genesis')
    baseline, treatment = original.model_dump(mode='json'), original.model_dump(mode='json')
    storage = {}
    for kind, snapshot in [('single_vector', baseline), ('long_session', treatment)]:
        for row in snapshot['profiles']:
            owner = row['profile_id']; state = states[owner]; selection = experiment.sessions[owner]
            if set(selection) != {'session_id', 'epoch'}:
                raise ValueError('explicit session ID/epoch selection required')
            view = model.recommendation_view(state, policy, session_id=selection['session_id'],
                epoch=decimal(selection['epoch'], 1), as_of_ms=as_of_ms)
            row['vector'] = list(legacy[owner].vector if kind == 'single_vector' else view.blended)
            row['payload']['last_feedback_version'] = state.feedback_version
            row['payload']['feedback_state_revision'] = state.operation_version
            # Enforce the SAME current product dislikes in both conditions.
            for profile in snapshot['bundle']['profiles']:
                if profile['profile_id'] == owner:
                    profile['disliked'] = sorted(set(profile['disliked']) | set(view.explicit_negative_repo_ids))
                    if snapshot['bundle'].get('dataset') is not None:
                        for source in snapshot['bundle']['dataset']['profiles']:
                            if source['profile_id'] == owner:
                                source['disliked'] = list(profile['disliked'])
            storage[owner] = {'legacy_bytes': len(model.canonical({'vector':list(legacy[owner].vector),'payload':dict(legacy[owner].payload)})),
                              'model_state_bytes': len(model.state_bytes(state)), 'feedback_version': str(state.feedback_version),
                              'operation_version': str(state.operation_version), 'state_sha256': model.state_sha256(state)}
    return {'single_vector': FrozenSnapshot.model_validate(baseline), 'long_session': FrozenSnapshot.model_validate(treatment)}, storage


def model_source_identity():
    root = Path(__file__).resolve().parents[1]
    paths = ['evaluation/model_history.py', 'feedback/long_session.py', 'feedback/model_processor.py',
             'feedback/model_serving.py', 'feedback/model_wire.py', 'feedback/v2.py']
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}


def compare_model_history(experiment, artifact_dir):
    snapshots, storage = derive_snapshots(experiment)
    results = {}
    for name, snapshot in snapshots.items():
        # This module runs only in the owned offline worker. Drain/forget its
        # idle executor between conditions so observations are per-condition,
        # not cumulative statistics from the baseline or a previous replay.
        from retrieval.runtime import retrieval_runtime_cleanup_actions, retrieval_runtime_status
        status = retrieval_runtime_status()
        if any(status.get(key, 0) for key in ('outstanding', 'running', 'waiting_submitters')):
            raise RuntimeError('offline model comparison requires an idle owned retrieval executor')
        for _, action in retrieval_runtime_cleanup_actions():
            action()
        exported = execute_snapshot(snapshot, Path(artifact_dir))
        results[name] = {'execution': exported.model_dump(mode='json'),
                         'evaluation': evaluate(snapshot.bundle, exported.run, k=experiment.k)}
    return {'schema_version': 1, 'format': 'frozen-user-model-comparison-v1', 'qualified': False,
            'experiment_sha256': digest(experiment), 'original_snapshot_sha256': digest(experiment.snapshot),
            'model_source_sha256': model_source_identity(),
            'policy_sha256': experiment.explicit_policy().sha256,
            'legacy_policy': {**experiment.legacy_policy.model_dump(mode='json'), 'action_alphas': dict(ALPHAS)},
            'original_operation_sha256': [hashlib.sha256(raw.encode('utf-8')).hexdigest() for raw in experiment.operations],
            'as_of': experiment.snapshot.as_of, 'label_provenance': experiment.snapshot.bundle.provenance.model_dump(mode='json'),
            'scope': 'same frozen corpus/baseline/events/artifacts/as-of; actual local exact retrieval/heavy ranking/assembly; no real-user uplift or server ANN claim',
            'storage': storage, 'results': results}


def validate_model_comparison(value, experiment):
    snapshots, storage = derive_snapshots(experiment)
    if type(value) is not dict or set(value.get('results', {})) != set(snapshots):
        raise ValueError('worker model comparison conditions differ')
    expected = {'schema_version': 1, 'format': 'frozen-user-model-comparison-v1', 'qualified': False,
        'experiment_sha256': digest(experiment), 'original_snapshot_sha256': digest(experiment.snapshot),
            'model_source_sha256': model_source_identity(),
        'policy_sha256': experiment.explicit_policy().sha256,
        'legacy_policy': {**experiment.legacy_policy.model_dump(mode='json'), 'action_alphas': dict(ALPHAS)},
        'original_operation_sha256': [hashlib.sha256(raw.encode('utf-8')).hexdigest() for raw in experiment.operations],
        'as_of': experiment.snapshot.as_of, 'label_provenance': experiment.snapshot.bundle.provenance.model_dump(mode='json'),
        'scope': 'same frozen corpus/baseline/events/artifacts/as-of; actual local exact retrieval/heavy ranking/assembly; no real-user uplift or server ANN claim',
        'storage': storage, 'results': {}}
    for name, snapshot in snapshots.items():
        if set(value['results'][name]) != {'execution', 'evaluation'}:
            raise ValueError('worker model condition schema differs')
        execution = StageRunExport.model_validate(value['results'][name]['execution'])
        if (execution.snapshot_sha256 != digest(snapshot) or execution.bundle_sha256 != digest(snapshot.bundle)
                or execution.artifacts != snapshot.artifacts or execution.as_of != snapshot.as_of or execution.seed != snapshot.seed):
            raise ValueError('worker executed a different frozen model input')
        calls = execution.configuration.get('runtime', {}).get('retrieval_executor', {}).get('stages', {}).get('profile', {}).get('calls_total')
        if type(calls) is not int or calls != len(snapshot.profiles):
            raise ValueError('worker model condition includes cumulative or missing profile measurements')
        expected['results'][name] = {'execution': execution.model_dump(mode='json'),
                                      'evaluation': evaluate(snapshot.bundle, execution.run, k=experiment.k)}
    if json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')) != json.dumps(expected,sort_keys=True,allow_nan=False,separators=(',',':')):
        raise ValueError('worker derived model comparison differs')
    return expected
