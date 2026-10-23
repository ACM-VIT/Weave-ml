"""Fresh pair proof never acts as an eligibility or cross-request vector cache."""
from types import SimpleNamespace
from uuid import UUID

import pytest

from retrieval.candidate_vectors import InlineCandidateSnapshots

ID = str(UUID(int=1))


def tracker(capacity=2):
    return InlineCandidateSnapshots(core_fields=['content_version', 'feature_version'],
                                    extra_fields=['topics'], max_entries=capacity)


def point(*, content=1, feature=1, topics=None, vector=None):
    return SimpleNamespace(payload={'content_version': content, 'feature_version': feature,
        'topics': ['python'] if topics is None else topics}, vector=[1., 0.] if vector is None else vector)


def test_every_snapshot_requires_validation_without_retaining_points():
    proof = tracker()
    proof[ID, True] = point()
    proof[ID, True] = point()
    assert (ID, True) not in proof
    assert dict(proof) == {}  # No old point can satisfy normal memo lookup.
    assert proof.has_pair(ID) and proof.tracked_ids == 1
    assert tracker().has_pair(ID) is False


@pytest.mark.parametrize('changed', [point(content=2), point(feature=2),
                                    point(topics=['rust']), point(vector=[0., 1.]), None])
def test_conflicting_or_invalid_snapshot_cannot_reuse_a_prior_pair(changed):
    proof = tracker()
    proof[ID, True] = point()
    proof[ID, True] = changed
    assert not proof.has_pair(ID)
    assert ID in proof.conflicts


def test_metadata_only_snapshot_needs_its_own_matching_validated_vector():
    proof = tracker()
    metadata = point(topics=[])
    proof[ID, False] = metadata
    assert not proof.has_pair(ID)
    proof[ID, True] = point()
    assert proof.has_pair(ID)
    proof[ID, False] = point(feature=2)
    assert not proof.has_pair(ID)


def test_input_mutation_cannot_rewrite_previously_observed_digest():
    proof = tracker()
    value = point()
    proof[ID, True] = value
    value.payload['topics'].append('changed')
    proof[ID, True] = value
    assert not proof.has_pair(ID)


def test_snapshot_overflow_is_bounded_and_never_authorizes_a_pair():
    proof = tracker(capacity=1)
    proof[ID, True] = point()
    proof[str(UUID(int=2)), True] = point()
    assert proof.overflowed and proof.tracked_ids == 1
    assert not proof.has_pair(ID)
