"""Strict exact-byte transport for the negotiated, ordered user-model lane.

Only the backend allocates these operations. Parsing never rewrites the opaque
legacy feedback, and a parsed object is never used as the admission fingerprint.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
import uuid

from feedback.long_session import Header, MAX_INTEGER, VERSION, ModelError


class WireError(ValueError):
    pass


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def strict_json(text, *, maximum_bytes):
    if type(text) is not str:
        raise WireError('JSON input must be an exact string')
    try:
        raw = text.encode('utf-8', errors='strict')
    except UnicodeError as exc:
        raise WireError('invalid UTF-8 input') from exc
    if not 0 < len(raw) <= maximum_bytes:
        raise WireError('JSON input byte budget exceeded')

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise WireError('duplicate JSON key')
            result[key] = value
        return result

    def number(value):
        result = int(value)
        if abs(result) > 2**53 - 1:
            raise WireError('JSON integers must be safely representable; use decimal strings for cursors')
        return result

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise WireError("nonfinite JSON value")
        return result

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_int=number, parse_float=finite_float,
                          parse_constant=lambda _: (_ for _ in ()).throw(WireError('nonfinite JSON value')))
    except (ValueError, TypeError, RecursionError) as exc:
        raise WireError('invalid strict JSON input') from exc


def exact_keys(value, required, optional=()):
    if type(value) is not dict or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        raise WireError('unexpected or missing operation fields')


def decimal(value, minimum=0):
    if type(value) is not str or not re.fullmatch(r'0|[1-9][0-9]{0,18}', value):
        raise WireError('canonical int64 decimal string required')
    result = int(value)
    if not minimum <= result <= MAX_INTEGER:
        raise WireError('decimal outside int64 range')
    return result


def canonical_uuid(value):
    try:
        if type(value) is str and str(uuid.UUID(value)) == value:
            return value
    except (ValueError, AttributeError):
        pass
    raise WireError('canonical UUID required')


def digest(value):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise WireError('canonical SHA256 required')
    return value


@dataclass(frozen=True)
class WireOperation:
    header: Header
    kind: str
    policy_sha256: str
    original_json: str
    payload: dict
    feedback: dict | None


def parse_operation(text, *, maximum_bytes, allowed_policies):
    envelope = strict_json(text, maximum_bytes=maximum_bytes)
    exact_keys(envelope, ('schema_version', 'operation_id', 'user_id', 'model_operation_version',
                         'previous_model_operation_version', 'accepted_at_ms', 'policy_version',
                         'policy_sha256', 'kind', 'operation'))
    if type(envelope['schema_version']) is not int or envelope['schema_version'] != 1:
        raise WireError('unsupported model-operation schema')
    if canonical_json(envelope) != text:
        raise WireError('operation JSON must use canonical UTF-8 encoding')
    if envelope['policy_version'] != VERSION or digest(envelope['policy_sha256']) not in allowed_policies:
        raise WireError('model policy is not explicitly registered')
    user = canonical_uuid(envelope['user_id'])
    try:
        header = Header(canonical_uuid(envelope['operation_id']), user,
                        decimal(envelope['model_operation_version'], 1),
                        decimal(envelope['previous_model_operation_version']),
                        decimal(envelope['accepted_at_ms']), hashlib.sha256(text.encode('utf-8')).hexdigest())
    except ModelError as exc:
        raise WireError("invalid operation order or identity") from exc
    payload, kind, feedback = envelope['operation'], envelope['kind'], None
    if kind == 'profile':
        exact_keys(payload, ('job_id', 'profile_version', 'allow_genesis', 'original_profile_context'))
        canonical_uuid(payload['job_id']); decimal(payload['profile_version'], 1)
        if type(payload['allow_genesis']) is not bool or type(payload['original_profile_context']) is not dict:
            raise WireError('explicit profile initialization and context required')
        from api.contracts import OnboardingProfile
        try:
            OnboardingProfile.model_validate(payload['original_profile_context'])
        except ValueError as exc:
            raise WireError('invalid bounded original profile context') from exc
    elif kind == 'feedback':
        exact_keys(payload, ('original_feedback_json',), ('canonical_attribution',))
        feedback = strict_json(payload['original_feedback_json'], maximum_bytes=maximum_bytes)
        exact_keys(feedback, ('event_id', 'user_id', 'repo_id', 'feedback_version', 'event_type'),
                   ('dwell_ms', 'occurred_at'))
        canonical_uuid(feedback['event_id']); canonical_uuid(feedback['repo_id'])
        if canonical_uuid(feedback['user_id']) != user:
            raise WireError('feedback owner differs from operation owner')
        if type(feedback['feedback_version']) is int:
            if feedback['feedback_version'] < 1:
                raise WireError('positive original feedback version required')
        else:
            decimal(feedback['feedback_version'], 1)
        if feedback['event_type'] not in {'like', 'dislike', 'unlike', 'undislike', 'save', 'unsave',
                                           'readme_open', 'github_open', 'share', 'dwell'}:
            raise WireError('unsupported online feedback type')
        if feedback['event_type'] == 'dwell':
            if type(feedback.get('dwell_ms')) is not int or not 3000 <= feedback['dwell_ms'] <= 2**53 - 1:
                raise WireError('qualified integer dwell required')
        elif feedback.get('dwell_ms') is not None:
            raise WireError('only dwell may supply dwell_ms')
        from api.contracts import FeedbackEvent
        try:
            FeedbackEvent.model_validate(feedback)
        except ValueError as exc:
            raise WireError('invalid original feedback contract') from exc
        if 'canonical_attribution' in payload:
            binding = payload['canonical_attribution']
            exact_keys(binding, ('user_id', 'session_id', 'epoch', 'epoch_started_ms',
                                 'active_at_acceptance', 'witness_sha256'))
            if canonical_uuid(binding['user_id']) != user:
                raise WireError('session owner differs')
            canonical_uuid(binding['session_id']); decimal(binding['epoch'], 1)
            started = decimal(binding['epoch_started_ms']); digest(binding['witness_sha256'])
            if type(binding['active_at_acceptance']) is not bool or started > header.accepted_ms:
                raise WireError('invalid canonical session attribution')
    elif kind in {'begin', 'reset', 'end'}:
        required = {'begin': ('session_id', 'epoch', 'epoch_started_ms', 'witness_sha256'),
                    'reset': ('session_id', 'old_epoch', 'new_epoch', 'new_epoch_started_ms', 'witness_sha256'),
                    'end': ('session_id', 'epoch', 'epoch_started_ms', 'retired', 'witness_sha256')}[kind]
        exact_keys(payload, required); canonical_uuid(payload['session_id']); digest(payload['witness_sha256'])
        if kind == 'reset':
            if decimal(payload['new_epoch'], 1) != decimal(payload['old_epoch'], 1) + 1:
                raise WireError('reset must advance exactly one epoch')
            if decimal(payload['new_epoch_started_ms']) != header.accepted_ms:
                raise WireError('reset must use backend acceptance time')
        else:
            epoch, started = decimal(payload['epoch'], 1), decimal(payload['epoch_started_ms'])
            if started > header.accepted_ms or (kind == 'begin' and (epoch != 1 or started != header.accepted_ms)):
                raise WireError('invalid control epoch/time')
            if kind == 'end' and payload['retired'] is not True:
                raise WireError('End must permanently retire the session')
    else:
        raise WireError('unsupported model operation kind')
    return WireOperation(header, kind, envelope['policy_sha256'], text, payload, feedback)


@dataclass(frozen=True)
class WireWithdrawal:
    header: Header
    policy_sha256: str
    profile_job_id: str
    profile_version: int
    allow_genesis: bool
    revoked_at_ms: int
    revocation_witness_sha256: str
    withdrawal_sha256: str
    original_json: str


def parse_withdrawal(text, *, maximum_bytes, allowed_policies):
    """Authenticated backend attestation; never contains revoked context."""
    value = strict_json(text, maximum_bytes=maximum_bytes)
    exact_keys(value, ('schema_version', 'operation_id', 'user_id', 'model_operation_version',
        'previous_model_operation_version', 'accepted_at_ms', 'policy_version', 'policy_sha256',
        'source_envelope_sha256', 'profile_job_id', 'profile_version', 'allow_genesis',
        'revoked_at_ms', 'revocation_witness_sha256', 'reason'))
    if type(value['schema_version']) is not int or value['schema_version'] != 1 or canonical_json(value) != text:
        raise WireError('unsupported or noncanonical withdrawal attestation')
    if value['policy_version'] != VERSION or digest(value['policy_sha256']) not in allowed_policies:
        raise WireError('withdrawal policy is not registered')
    if value['reason'] != 'github_consent_revoked' or type(value['allow_genesis']) is not bool:
        raise WireError('explicit profile-consent revocation required')
    try:
        header = Header(canonical_uuid(value['operation_id']), canonical_uuid(value['user_id']),
            decimal(value['model_operation_version'], 1), decimal(value['previous_model_operation_version']),
            decimal(value['accepted_at_ms']), digest(value['source_envelope_sha256']))
    except ModelError as exc:
        raise WireError('invalid original operation identity') from exc
    return WireWithdrawal(header, value['policy_sha256'], canonical_uuid(value['profile_job_id']),
        decimal(value['profile_version'], 1), value['allow_genesis'], decimal(value['revoked_at_ms']),
        digest(value['revocation_witness_sha256']), hashlib.sha256(text.encode('utf-8')).hexdigest(), text)
