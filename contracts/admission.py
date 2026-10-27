"""Shared, non-enforcing repository admission proposal for owner review.

This is deliberately separate from schema/vector validation and from the
existing path-specific gates. No caller can enable rejection through this API.
"""

from collections.abc import Mapping
from typing import Any


ADMISSION_POLICY_VERSION = "repository-admission-proposal-v1"
ADMISSION_OBSERVATION_FIELD = "admission_observation"
PROPOSED_MIN_README_CHARS = 200


def _has_labels(value: Any) -> bool:
    # Acquisition can supply a language-to-byte-count mapping; V2 uses lists.
    if not isinstance(value, (list, tuple, Mapping)):
        return False
    return any(isinstance(label, str) and label.strip() for label in value)


def observe_repository_admission(repo: Mapping[str, Any]) -> dict[str, Any]:
    """Return deterministic content observations, never quality approval.

    Raw README is authoritative when supplied; otherwise use its observed
    length. Missing/invalid evidence requests review rather than inventing it.
    Star count and recency intentionally have no rejection threshold.
    """
    readme = repo.get("readme")
    length = len(readme) if isinstance(readme, str) else repo.get("readme_length")
    reasons: list[str] = []
    if type(length) is not int or length < 0:
        length = None
        reasons.append("readme_length_unknown")
    elif length == 0:
        reasons.append("readme_missing")
    elif length < PROPOSED_MIN_README_CHARS:
        reasons.append("readme_thin")

    description = repo.get("description")
    if not (
        (isinstance(description, str) and description.strip())
        or _has_labels(repo.get("languages"))
        or _has_labels(repo.get("topics"))
    ):
        reasons.append("repository_metadata_empty")

    return {
        "policy_version": ADMISSION_POLICY_VERSION,
        "mode": "observe_only",
        "approval_status": "pending_owner_decision",
        "proposed_min_readme_chars": PROPOSED_MIN_README_CHARS,
        "content_candidate": "review" if reasons else "pass",
        "reason_codes": reasons,
        "readme_length": length,
        "acquisition_acceptance": "existing_path_gate",
        "quality_approval": "not_evaluated",
        "schema_validity": "independent_boundary_check_required",
        "indexing_eligibility": "requires_schema_and_valid_vector",
        "serving_eligibility": "requires_backend_policy",
        "star_rule": "no_floor_proposed",
        "recency_rule": "no_rejection_threshold_proposed",
    }
