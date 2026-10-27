"""Apply negotiated operations with once-pinned sources and one complete CAS."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from feedback import long_session as model
from feedback.model_journal import Coordinator, JournalConflict, OriginalIdentityConflict, _canonical, _json
from feedback.model_wire import decimal, WireWithdrawal
from feedback.v2 import FeedbackStateError, MissingVectorError, FeedbackStateLimitError, TrackedRepositoryLimitError
from scripts.user_onboarding import _has_feedback_state, _is_feedback_payload_key
from config import REPOSITORY_EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION


class ModelProcessor:
    def __init__(self, journal, users, policy, legacy_applier, profile_encoder, *, dead_letter_prepare, withdrawal_lookup=None):
        self.journal, self.users, self.policy = journal, users, policy
        self.legacy, self.profile_encoder = legacy_applier, profile_encoder
        self.withdrawal_lookup = withdrawal_lookup or (lambda user, version: None)
        self.coordinator = Coordinator(journal, users, policy, reject_capacity=True,
                                       legacy_reject=self._legacy_reject, dead_letter_prepare=dead_letter_prepare)

    def _resolve(self, wire):
        h, payload = wire.header, wire.payload
        if wire.kind == 'profile':
            vector = self.profile_encoder(payload['original_profile_context'])
            pinned = model.PinnedVector.pin('profile', h.user_id, decimal(payload['profile_version'], 1), 0,
                                            self.policy.baseline_recipe, self.policy.embedding, list(vector))
            return model.Profile(h, pinned, payload['allow_genesis'])
        if wire.kind == 'begin':
            return model.Begin(h, payload['session_id'], decimal(payload['epoch_started_ms']), payload['witness_sha256'])
        if wire.kind == 'reset':
            return model.Reset(h, payload['session_id'], decimal(payload['old_epoch'], 1), decimal(payload['new_epoch'], 1), payload['witness_sha256'])
        if wire.kind == 'end':
            return model.End(h, payload['session_id'], decimal(payload['epoch'], 1), decimal(payload['epoch_started_ms']))
        event = self.legacy.normalize_event(wire.feedback)
        pinned = None
        if event['event_type'] not in {'unlike', 'undislike', 'unsave'}:
            # Batch context freezes the exact same repository record for both
            # revision attribution and the established legacy eligibility check.
            points = self.legacy.qdrant.retrieve(collection_name=self.legacy.settings.repository_collection,
                ids=[event['repo_id']], with_vectors=True, with_payload=[
                    'repo_id', 'serving_eligibility_version', 'content_version', 'feature_version',
                    'embedding_model', 'embedding_model_revision', 'embedding_version', 'embedding_dim', 'feature_spec_version'])
            point = next((p for p in points if str(p.id) == event['repo_id']), None)
            if point is None:
                raise MissingVectorError('repository')
            content_version = point.payload.get('content_version')
            feature_version = point.payload.get('feature_version', 0)
            model._integer(content_version, 1)
            model._integer(feature_version)
            if point.payload.get('embedding_version') != self.policy.repository_recipe:
                raise model.CompatibilityError('pinned repository recipe differs from explicit model policy')
            self.legacy._repository_batches.points = {event['repo_id']: point}
            try:
                vector = self.legacy._repository_vector(event['repo_id'], 384)
            finally:
                del self.legacy._repository_batches.points
            pinned = model.PinnedVector.pin('repository', event['repo_id'], content_version,
                                            feature_version, self.policy.repository_recipe, self.policy.embedding, vector.tolist())
        binding = payload.get('canonical_attribution')
        session = None if binding is None else model.SessionBinding(binding['user_id'], binding['session_id'],
            decimal(binding['epoch'], 1), decimal(binding['epoch_started_ms']), binding['active_at_acceptance'], binding['witness_sha256'])
        return model.Feedback(h, event['event_id'], event['feedback_version'], event['repo_id'], event['event_type'],
                              pinned, session, event.get('dwell_ms'), None)

    def _profile_legacy(self, wire, operation, previous=None):
        payload = deepcopy(wire.payload['original_profile_context'])
        payload.update(job_id=wire.payload['job_id'], profile_version=operation.baseline.content_version,
                       user_id=wire.header.user_id, embedding_dim=384, embedding_model=REPOSITORY_EMBEDDING_MODEL,
                       embedding_model_revision=EMBEDDING_MODEL_REVISION)
        vector = list(operation.baseline.vector)
        old = previous or {'payload': {}, 'vector': vector}
        old_payload = old['payload']
        if _has_feedback_state(old_payload):
            vector = list(old['vector'])
            for key in old_payload:
                if _is_feedback_payload_key(key):
                    payload[key] = deepcopy(old_payload[key])
            payload['profile_baseline_vector'] = list(operation.baseline.vector)
        else:
            payload['preference_accumulator'] = list(operation.baseline.vector)
            payload['last_feedback_version'] = 0
        payload['feedback_state_revision'] = old_payload.get('feedback_state_revision', 0) + (1 if previous else 0)
        return {'vector': vector, 'payload': payload}

    def _legacy_prepare(self, wire, complete, operation):
        body = _json(complete, self.journal.limits.combined_user_bytes)
        legacy = deepcopy(body['legacy'])
        if isinstance(operation, model.Feedback):
            user = SimpleNamespace(id=wire.header.user_id, **legacy)
            def pinned_vector(repo_id, dimension):
                if operation.repository is None or repo_id != operation.repository.subject_id or dimension != 384:
                    raise JournalConflict('legacy computation requested an unpinned repository')
                return operation.repository.vector
            try:
                result, point = self.legacy.prepare(wire.feedback, user=user, repository_vector=pinned_vector)
            except (FeedbackStateLimitError, TrackedRepositoryLimitError) as exc:
                raise model.StateOverflow('legacy compatibility capacity exceeded') from exc
            if result.status != 'applied' or point is None:
                raise FeedbackStateError('legacy feedback predecessor was not applied')
            legacy = {'vector': list(point.vector), 'payload': dict(point.payload)}
        elif isinstance(operation, model.Profile):
            legacy = self._profile_legacy(wire, operation, legacy)
        else:
            legacy['payload']['feedback_state_revision'] = legacy['payload'].get('feedback_state_revision', 0) + 1
        return _canonical(legacy)

    def _legacy_reject(self, complete, operation):
        legacy = deepcopy(_json(complete, self.journal.limits.combined_user_bytes)['legacy'])
        payload = legacy['payload']
        payload['feedback_state_revision'] = payload.get('feedback_state_revision', 0) + 1
        if isinstance(operation, model.Feedback):
            if operation.feedback_version != payload.get('last_feedback_version', 0) + 1:
                raise model.OperationGap('legacy rejection predecessor missing')
            payload.update(last_feedback_version=operation.feedback_version, last_feedback_event_id=operation.event_id,
                           last_feedback_status='rejected')
        return _canonical(legacy)

    def _matching_withdrawal(self, wire):
        withdrawn = self.withdrawal_lookup(wire.header.user_id, wire.header.operation_version)
        if withdrawn is not None and (withdrawn.header != wire.header
                or withdrawn.profile_job_id != wire.payload['job_id']
                or withdrawn.profile_version != decimal(wire.payload['profile_version'], 1)
                or withdrawn.allow_genesis != wire.payload['allow_genesis']):
            raise OriginalIdentityConflict('withdrawal attestation names a different original profile')
        return withdrawn

    def preflight_original(self, wire):
        """Recognize a committed original without re-encoding private context."""
        snapshot = self.users.read_optional(wire.header.user_id)
        if snapshot is None or snapshot.state.operation_version < wire.header.operation_version:
            return None
        result = self.coordinator.original_status(snapshot, wire.header)
        operation = result['operation']
        response = {'operation_id': wire.header.operation_id,
            'model_operation_version': str(wire.header.operation_version),
            'source_envelope_sha256': wire.header.source_envelope_sha256, 'status': 'duplicate'}
        if isinstance(operation, model.WithdrawnProfile):
            if (operation.profile_job_id != wire.payload['job_id']
                    or operation.profile_version != decimal(wire.payload['profile_version'], 1)
                    or operation.allow_genesis != wire.payload['allow_genesis']):
                raise OriginalIdentityConflict('late original profile differs from committed withdrawal')
            response.update(status='withdrawal_accepted', withdrawal_sha256=operation.withdrawal_sha256)
        elif not isinstance(operation, model.Profile):
            raise OriginalIdentityConflict('original profile identity belongs to another operation kind')
        return response

    def ready_predecessor(self, wire):
        snapshot = self.users.read_optional(wire.header.user_id)
        if snapshot is None or isinstance(snapshot.state, model.BootstrapState):
            # Missing initialized users are refused by read_optional. Only a
            # genuine profile or its authenticated withdrawal can start a lane.
            genesis = wire.allow_genesis if isinstance(wire, WireWithdrawal) else (
                wire.kind == 'profile' and wire.payload['allow_genesis'])
            previous = 0 if snapshot is None else snapshot.state.operation_version
            if snapshot is not None:
                self.coordinator._validate_snapshot(snapshot)
            return wire.header.previous_operation_version == previous and genesis
        self.coordinator._validate_snapshot(snapshot)
        return wire.header.previous_operation_version == snapshot.state.operation_version

    def __call__(self, wire, assert_lock):
        if wire.policy_sha256 != self.policy.sha256:
            raise JournalConflict('processor policy differs')
        if isinstance(wire, WireWithdrawal):
            return self.coordinator.apply_withdrawal(wire, assert_lock)
        raw = wire.original_json.encode('utf-8')
        resolve = lambda: self._resolve(wire)
        snapshot = self.users.read_optional(wire.header.user_id)
        if wire.kind == 'profile':
            proven = self.preflight_original(wire)
            if proven is not None:
                return 'withdrawn_duplicate' if proven['status'] == 'withdrawal_accepted' else 'duplicate'
            withdrawn = self._matching_withdrawal(wire)
            if withdrawn is not None:
                return self.coordinator.apply_withdrawal(withdrawn, assert_lock)
            def profile_lock():
                assert_lock()
                if self._matching_withdrawal(wire) is not None:
                    raise JournalConflict('profile withdrawal is pending; retry its terminal disposition')
            if snapshot is None or isinstance(snapshot.state, model.BootstrapState):
                if not wire.payload['allow_genesis']:
                    raise model.OperationGap('real profile basis is not applied')
                previous = 0 if snapshot is None else snapshot.state.operation_version
                if wire.header.previous_operation_version != previous:
                    raise model.OperationGap('profile genesis predecessor is pending')
                operation, _ = self.journal.pin_input(wire.header, wire.kind, self.policy, raw, resolve)
                return self.coordinator.initialize(wire.header, raw, lambda: operation,
                    _canonical(self._profile_legacy(wire, operation)), profile_lock)
            return self.coordinator.apply(wire.header, wire.kind, raw, resolve,
                lambda complete, operation: self._legacy_prepare(wire, complete, operation), profile_lock)
        if snapshot is None or isinstance(snapshot.state, model.BootstrapState):
            raise model.OperationGap('real profile basis must precede model feedback and controls')
        return self.coordinator.apply(wire.header, wire.kind, raw, resolve,
            lambda complete, operation: self._legacy_prepare(wire, complete, operation), assert_lock)
