"""Strict, bounded request contracts for the v2 backend boundary."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from functools import lru_cache
import os
from typing import Annotated, Any, Literal, get_args
from urllib.parse import urlsplit

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from feedback.v2_settings import V2FeedbackSettings
from embedding.vector_contract import validate_diversity_metadata
from summarization.contracts import CardSummaryArtifact


ShortText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]
MediumText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=512),
]
ListText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]
DescriptionText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, max_length=10_000),
]
ReadmeText = Annotated[str, StringConstraints(max_length=1_000_000)]
ParagraphText = Annotated[str, StringConstraints(max_length=10_000)]
NonNegativeCount = Annotated[int, Field(ge=0, le=9_223_372_036_854_775_807)]
DeltaCount = Annotated[int, Field(ge=-2_147_483_648, le=2_147_483_647)]
UnitScore = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RequestValidationDetail(StrictModel):
    path: str
    message: str
    type: str


class RequestValidationResponse(StrictModel):
    """Safe validation envelope emitted by the integrated application handler."""

    detail: str
    code: Literal["REQUEST_VALIDATION_FAILED"]
    message: str
    retryable: Literal[False]
    # Compatibility response IDs also accept bounded legacy backend IDs.
    # Trace correlation separately requires UUIDs and never adopts these IDs.
    request_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    details: list[RequestValidationDetail]


def _utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be an ISO-8601 UTC timestamp")
    return value


@lru_cache(maxsize=16)
def _feedback_dwell_bounds(
    minimum_seconds: str | None,
    full_credit_seconds: str | None,
) -> tuple[int, int]:
    # The raw values are cache keys so tests and local processes that update
    # configuration before constructing a contract cannot retain stale bounds.
    settings = V2FeedbackSettings.from_env()
    return settings.dwell_min_ms, settings.dwell_full_credit_ms


def feedback_dwell_bounds() -> tuple[int, int]:
    return _feedback_dwell_bounds(
        os.getenv("FEEDBACK_DWELL_MIN_SECONDS"),
        os.getenv("FEEDBACK_DWELL_FULL_CREDIT_SECONDS"),
    )


class RecommendationContext(StrictModel):
    cold_start: bool = False
    # Locale is reserved for deterministic filtering/experiments; retrieval
    # currently reports it as reserved rather than silently changing ranking.
    locale: str | None = Field(
        default=None,
        min_length=2,
        max_length=32,
        pattern=r"^[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})*$",
    )


from retrieval.social_candidates import SocialCandidate, validate_social_candidates


class ScoringCaptureRequest(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    schema_version: Literal[1]
    encoding: Literal["float32-le-v1"]

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_schema(cls, value):
        if type(value) is not int:
            raise ValueError("Capture schema must be an integer")
        return value


def _bounded_decimal_pattern(maximum: int) -> str:
    """Canonical positive decimal <= maximum, using portable digit ranges."""
    digits = str(maximum)
    alternatives = [rf"[1-9][0-9]{{0,{len(digits)-2}}}"] if len(digits) > 1 else []
    for index, digit in enumerate(digits):
        lower, upper = (1 if index == 0 else 0), int(digit) - 1
        if lower <= upper:
            choice = str(lower) if lower == upper else f"[{lower}-{upper}]"
            remaining = len(digits) - index - 1
            alternatives.append(digits[:index] + choice + (rf"[0-9]{{{remaining}}}" if remaining else ""))
    alternatives.append(digits)
    return "^(?:" + "|".join(alternatives) + ")$"


_CAPTURE_FEED_VERSION_PATTERN = _bounded_decimal_pattern(2**63 - 1)


def _recommendation_request_schema(schema: dict) -> None:
    # The common properties still enforce the complete capture object schema.
    # Branches express negotiation without changing the public Python model.
    capture_reference = next(choice["$ref"] for choice in
        schema["properties"]["scoring_capture"]["anyOf"] if "$ref" in choice)
    schema["oneOf"] = [
        {"required": ["feed_version"], "properties": {
            "feed_version": {"type": "integer", "minimum": 1, "maximum": 2**53-1},
            "scoring_capture": {"type": "null"}, "model_context": {"type": "null"}}},
        {"required": ["feed_version", "scoring_capture"], "properties": {
            "feed_version": {"type": "string", "minLength": 1, "maxLength": 19,
                "pattern": _CAPTURE_FEED_VERSION_PATTERN,
                "not": {"pattern": "[^0-9]"}},
            "scoring_capture": {"$ref": capture_reference}}},
        {"required": ["feed_version", "model_context"], "properties": {
            "feed_version": {"type": "string", "minLength": 1, "maxLength": 19,
                "pattern": _CAPTURE_FEED_VERSION_PATTERN, "not": {"pattern": "[^0-9]"}},
            "scoring_capture": {"type": "null"},
            "model_context": {"$ref": next(choice["$ref"] for choice in schema["properties"]["model_context"]["anyOf"] if "$ref" in choice)}}},
    ]


PositiveModelDecimal = Annotated[str, StringConstraints(strict=True, max_length=19, pattern=_CAPTURE_FEED_VERSION_PATTERN),
                                 Field(json_schema_extra={"not": {"pattern": "[^0-9]"}})]
NonnegativeModelDecimal = Annotated[str, StringConstraints(strict=True, max_length=19,
    pattern=_CAPTURE_FEED_VERSION_PATTERN.replace("^(?:", "^(?:0|", 1)),
    Field(json_schema_extra={"not": {"pattern": "[^0-9]"}})]


class ModelRecommendationContext(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=False, hide_input_in_errors=True)
    schema_version: Literal[1]
    session_id: Annotated[str, StringConstraints(strict=True, min_length=36, max_length=36,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")]
    epoch: PositiveModelDecimal
    policy_sha256: Annotated[str, StringConstraints(strict=True, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]
    minimum_applied_operation_version: PositiveModelDecimal
    required_profile_version: PositiveModelDecimal
    as_of_ms: NonnegativeModelDecimal

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_schema(cls, value):
        if type(value) is not int:
            raise ValueError("Model context schema must be an integer")
        return value

    @model_validator(mode="after")
    def exact_context(self):
        from feedback.model_wire import decimal, digest, canonical_uuid
        canonical_uuid(self.session_id)
        decimal(self.epoch, 1); decimal(self.minimum_applied_operation_version, 1)
        decimal(self.required_profile_version, 1); decimal(self.as_of_ms); digest(self.policy_sha256)
        return self


class RecommendationRequest(StrictModel):
    model_config = ConfigDict(json_schema_extra=_recommendation_request_schema)
    schema_version: Literal[2]
    generation_id: uuid.UUID
    user_id: uuid.UUID
    feed_version: Annotated[int, Field(strict=True, ge=1, le=2**53-1)] | Annotated[str, StringConstraints(strict=True, max_length=19, pattern=_CAPTURE_FEED_VERSION_PATTERN)]
    scoring_capture: ScoringCaptureRequest | None = None
    model_context: ModelRecommendationContext | None = None
    limit: int = Field(ge=1, le=100)

    @field_validator("feed_version", mode="before")
    @classmethod
    def exact_feed_version(cls, value):
        import re
        if type(value) is int and 1 <= value <= 2**53-1:
            return value
        if type(value) is str and re.fullmatch(r"[1-9][0-9]{0,18}", value) and int(value) <= 2**63-1:
            return value
        raise ValueError("Feed version must be a safe integer or canonical int64 decimal")

    @model_validator(mode="after")
    def negotiated_feed_version(self):
        if (self.scoring_capture is not None or self.model_context is not None) != (type(self.feed_version) is str):
            raise ValueError("Captured requests require an exact decimal string; ordinary requests require an integer")
        return self
    exclude_repo_ids: list[uuid.UUID] = Field(default_factory=list, max_length=500)
    social_repo_ids: list[uuid.UUID] = Field(default_factory=list, max_length=500)
    social_candidates: list[SocialCandidate] = Field(default_factory=list, max_length=100, json_schema_extra={"uniqueItems": True})

    @field_validator("social_candidates")
    @classmethod
    def unique_social_evidence(cls, value: list[SocialCandidate]) -> list[SocialCandidate]:
        return validate_social_candidates(value)
    context: RecommendationContext

    @field_validator("exclude_repo_ids", "social_repo_ids")
    @classmethod
    def unique_repository_ids(cls, value: list[uuid.UUID]) -> list[uuid.UUID]:
        if len(set(value)) != len(value):
            raise ValueError("repository ID lists must be unique")
        return value


EventType = Literal[
    "impression",
    "dwell",
    "readme_open",
    "github_open",
    "like",
    "unlike",
    "dislike",
    "undislike",
    "save",
    "unsave",
    "share",
]


class FeedbackEvent(StrictModel):
    event_id: uuid.UUID
    user_id: uuid.UUID
    repo_id: uuid.UUID
    feedback_version: int = Field(ge=1)
    event_type: EventType = Field(json_schema_extra={
        "enum": [event for event in get_args(EventType) if event != "impression"],
    })
    dwell_ms: int | None = None
    occurred_at: datetime

    @field_validator("feedback_version", mode="before", json_schema_input_type=Annotated[int, Field(ge=1)] | Annotated[str, StringConstraints(pattern=r"^[1-9][0-9]*$")])
    @classmethod
    def feedback_version_wire(cls, value):
        # Node sends decimal strings to avoid losing 64-bit version precision.
        return value

    @field_validator("occurred_at")
    @classmethod
    def utc_occurred_at(cls, value: datetime) -> datetime:
        return _utc(value, field_name="occurred_at")

    @model_validator(mode="after")
    def validate_dwell(self):
        if self.event_type == "impression":
            raise ValueError("impressions are offline-only and must not be sent to ML")
        if self.event_type == "dwell":
            minimum, maximum = feedback_dwell_bounds()
            if self.dwell_ms is None or not minimum <= self.dwell_ms <= maximum:
                raise ValueError(
                    f"dwell_ms must be between {minimum} and {maximum}"
                )
        elif self.dwell_ms is not None:
            raise ValueError("only dwell events may carry dwell_ms")
        return self


class FeedbackBatch(StrictModel):
    schema_version: Literal[2]
    events: list[FeedbackEvent] = Field(min_length=1, max_length=100)

    @field_validator("events")
    @classmethod
    def unique_events(cls, value: list[FeedbackEvent]) -> list[FeedbackEvent]:
        if len({event.event_id for event in value}) != len(value):
            raise ValueError("event_id values must be unique within a batch")
        return value


class RepositorySource(StrictModel):
    # These identity/source fields are supplied by the canonical Node v2
    # ingestion outbox. ``repo_id`` is checked against the job envelope below;
    # owner/name/node ID remain source metadata and are not used as Qdrant IDs.
    repo_id: uuid.UUID | None = None
    github_id: str | None = Field(default=None, pattern=r"^[0-9]{1,32}$")
    owner_github_id: str | None = Field(default=None, pattern=r'^[1-9][0-9]{0,18}$')
    is_fork: Annotated[bool, Field(strict=True)] | None = None
    fork_parent_github_id: str | None = Field(default=None, pattern=r'^[1-9][0-9]{0,18}$')
    fork_family_github_id: str | None = Field(default=None, pattern=r'^[1-9][0-9]{0,18}$')
    lineage_observed_at: datetime | None = None
    lineage_source: Literal['github-graphql-repository', 'github-rest-repository'] | None = None
    github_node_id: MediumText | None = None
    full_name: str = Field(min_length=3, max_length=256, pattern=r"^[^/\s]+/[^/\s]+$")
    owner: ShortText | None = None
    name: ShortText | None = None
    html_url: str | None = Field(
        default=None,
        max_length=2_048,
        validation_alias=AliasChoices("html_url", "url"),
    )
    description: DescriptionText | None = None
    primary_language: ShortText | None = None
    languages: list[ListText] = Field(default_factory=list, max_length=100)
    topics: list[ListText] = Field(default_factory=list, max_length=200)
    readme: ReadmeText | None = None
    readme_source_path: str | None = Field(default=None, min_length=1, max_length=1_024)
    readme_default_branch: str | None = Field(default=None, min_length=1, max_length=256)
    readme_base_url: str | None = Field(default=None, min_length=1, max_length=2_048)
    extracted_paragraphs: list[ParagraphText] = Field(
        default_factory=list, max_length=512
    )
    readme_length: NonNegativeCount = 0
    star_count: NonNegativeCount = 0
    fork_count: NonNegativeCount = 0
    open_issues_count: NonNegativeCount = 0
    pushed_days_ago: Annotated[int, Field(ge=0, le=100_000)] = 999
    delta_3d: DeltaCount = 0
    delta_7d: DeltaCount = 0
    delta_30d: DeltaCount = 0
    mentionable_users_count: NonNegativeCount = 0
    readme_to_codebase_ratio: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    recent_commits: list[datetime] = Field(default_factory=list, max_length=100)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    pushed_at: datetime | None = None
    observed_at: datetime | None = None
    discovery_category: MediumText | None = None
    discovery_band: MediumText | None = None
    content_hash: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode='after')
    def observed_diversity_identity(self):
        validate_diversity_metadata(self.model_dump())
        return self

    @field_validator("created_at", "updated_at", "pushed_at", "observed_at")
    @classmethod
    def utc_timestamps(cls, value: datetime | None, info):
        return None if value is None else _utc(value, field_name=info.field_name)

    @field_validator("recent_commits")
    @classmethod
    def utc_commits(cls, value: list[datetime]) -> list[datetime]:
        return [_utc(item, field_name="recent_commits") for item in value]

    @field_validator("readme_source_path")
    @classmethod
    def safe_readme_source_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value.startswith("/") or "\\" in value or ".." in value.split("/"):
            raise ValueError("readme_source_path must be a repository-relative POSIX path")
        return value

    @field_validator("readme_default_branch")
    @classmethod
    def safe_readme_default_branch(cls, value: str | None) -> str | None:
        if value is None:
            return None
        segments = value.split("/")
        if "\\" in value or any(segment in {"", ".", ".."} for segment in segments):
            raise ValueError(
                "readme_default_branch must be a safe slash-separated Git ref"
            )
        return value

    @field_validator("readme_base_url")
    @classmethod
    def safe_readme_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "raw.githubusercontent.com"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "readme_base_url must be an HTTPS raw.githubusercontent.com directory URL"
            )
        return value if value.endswith("/") else f"{value}/"

    @model_validator(mode="after")
    def bounded_readme(self):
        paragraph_size = sum(len(item) for item in self.extracted_paragraphs)
        if paragraph_size > 1_000_000:
            raise ValueError("extracted_paragraphs exceed the 1 MB content limit")
        if self.readme is not None and self.extracted_paragraphs:
            raise ValueError("send readme or extracted_paragraphs, not both")
        return self


class RepositoryJob(StrictModel):
    schema_version: Literal[2]
    job_id: uuid.UUID
    repo_id: uuid.UUID
    content_version: int = Field(ge=1)
    repository: RepositorySource

    @model_validator(mode="after")
    def matching_repository_identity(self):
        nested_repo_id = self.repository.repo_id
        if nested_repo_id is not None and nested_repo_id != self.repo_id:
            raise ValueError("repository.repo_id must match repo_id")
        if not self.repository.content_hash:
            raise ValueError("repository.content_hash is required for repository jobs")
        return self


class RepositoryFeaturePatch(StrictModel):
    """Mutable observations plus optional, validated derived-score echoes.

    ``pushed_days_ago`` is the producer's age at its observation anchor. The
    refresh service never advances it from wall-clock time or infers it from
    ``pushed_at``. Ranking scores are server-owned; callers may include them
    for compatibility, but the refresh service requires exact agreement with
    the effective score. It recomputes a score only when one of that score's
    source observations changes.
    """

    star_count: NonNegativeCount | None = None
    fork_count: NonNegativeCount | None = None
    open_issues_count: NonNegativeCount | None = None
    pushed_days_ago: Annotated[
        int,
        Field(
            ge=0,
            le=100_000,
            description="Repository push age at the producer's observation anchor.",
        ),
    ] | None = None
    delta_3d: DeltaCount | None = None
    delta_7d: DeltaCount | None = None
    delta_30d: DeltaCount | None = None
    mentionable_users_count: NonNegativeCount | None = None
    doc_quality: UnitScore | None = None
    code_health: Annotated[
        UnitScore,
        Field(description="Optional echo of the server-derived code health score."),
    ] | None = None
    activity_score: Annotated[
        UnitScore,
        Field(description="Optional echo of the server-derived activity score."),
    ] | None = None
    trend_velocity: Annotated[
        UnitScore,
        Field(description="Optional echo of the server-derived trend velocity."),
    ] | None = None
    updated_at: datetime | None = None
    pushed_at: datetime | None = None
    discovery_category: MediumText | None = None
    discovery_band: MediumText | None = None

    @field_validator("updated_at", "pushed_at")
    @classmethod
    def utc_timestamps(cls, value: datetime | None, info):
        return None if value is None else _utc(value, field_name=info.field_name)

    @model_validator(mode="after")
    def not_empty(self):
        if not self.model_fields_set:
            raise ValueError("features must contain at least one refreshable field")
        return self


class RepositoryRefreshJob(StrictModel):
    schema_version: Literal[2]
    job_id: uuid.UUID
    repo_id: uuid.UUID
    feature_version: int = Field(ge=1)
    features: RepositoryFeaturePatch


class OnboardingProfile(StrictModel):
    github_username: ShortText | None = None
    username: ShortText | None = None
    full_name: MediumText | None = None
    bio: str | None = Field(default=None, max_length=2_000)
    interests: list[ListText] = Field(default_factory=list, max_length=100)
    topics: list[ListText] = Field(default_factory=list, max_length=100)
    skills: list[ListText] = Field(default_factory=list, max_length=100)
    tech_stack: list[ListText] = Field(default_factory=list, max_length=100)
    avatar_url: str | None = Field(default=None, max_length=2_048)

    @model_validator(mode="after")
    def has_embedding_context(self):
        if not (
            self.bio
            or self.interests
            or self.topics
            or self.skills
            or self.tech_stack
        ):
            raise ValueError(
                "profile must include bio, interests, topics, skills, or tech_stack"
            )
        return self


class OnboardingJob(StrictModel):
    schema_version: Literal[2]
    job_id: uuid.UUID
    user_id: uuid.UUID
    profile_version: int = Field(ge=1)
    profile: OnboardingProfile


class RepositorySearchFilters(StrictModel):
    # Language is the indexed primary language; topics use canonical slugs.
    languages: list[Annotated[str, StringConstraints(min_length=1, max_length=64)]] = Field(default_factory=list, max_length=10)
    topics: list[Annotated[str, StringConstraints(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9-]*$")]] = Field(default_factory=list, max_length=10)


class RepositorySearchRequest(StrictModel):
    schema_version: Literal[2]
    search_id: uuid.UUID
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=512)]
    limit: int = Field(ge=1, le=100, strict=True)
    exclude_repo_ids: list[uuid.UUID] = Field(default_factory=list, max_length=500)
    filters: RepositorySearchFilters = Field(default_factory=RepositorySearchFilters)

    @field_validator("exclude_repo_ids")
    @classmethod
    def unique_search_exclusions(cls, values: list[uuid.UUID]) -> list[uuid.UUID]:
        if len(set(values)) != len(values):
            raise ValueError("repository exclusions must be unique")
        return values


class RepositorySearchHit(StrictModel):
    repo_id: uuid.UUID
    score: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)


class RepositorySearchResponse(StrictModel):
    schema_version: Literal[2]
    search_id: uuid.UUID
    embedding_version: ShortText
    items: list[RepositorySearchHit] = Field(max_length=100)


# Response schemas are the wire contract used by the generated backend client.
# Keep derived summary artifacts separate from canonical README source.
class RecommendationItem(StrictModel):
    repo_id: uuid.UUID
    score: float = Field(allow_inf_nan=False)
    source: ShortText
    retrieval_scores: dict[
        Annotated[str, StringConstraints(min_length=1, max_length=32, pattern=r"^[a-z_]+$")],
        Annotated[float, Field(allow_inf_nan=False)],
    ] = Field(default_factory=dict, max_length=16)


class RecommendationContextStatus(StrictModel):
    cold_start_applied: bool
    locale: Literal["reserved_unused", "not_provided"]


class AppliedModelContext(StrictModel):
    schema_version: Literal[1]
    user_id: uuid.UUID
    session_id: uuid.UUID
    epoch: PositiveModelDecimal
    profile_version: PositiveModelDecimal
    applied_operation_version: PositiveModelDecimal
    applied_feedback_version: NonnegativeModelDecimal
    state_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    head: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    policy_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    effective_vector_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    session_witness_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    as_of_ms: NonnegativeModelDecimal
    valid_until_ms: PositiveModelDecimal
    explicit_negative_repo_ids: list[uuid.UUID] = Field(max_length=256)


class RecommendationResponse(StrictModel):
    schema_version: Literal[2]
    generation_id: uuid.UUID
    user_id: uuid.UUID
    feed_version: int = Field(ge=1)
    model_version: str = Field(min_length=1, max_length=512)
    # Retrieval supports eight compatible 128-character identities plus the
    # mixed-version prefix and seven separating commas.
    embedding_version: str = Field(min_length=1, max_length=len("compatible-mixed:") + 8 * 128 + 7)
    embedding_versions: list[str]
    served_ranker: Literal["heavy", "hybrid"]
    heavy_ranker_selected: bool
    ranker_applied: bool
    fallback_code: str | None
    fallback_reason: str | None
    retrieval_mode: str
    context_status: RecommendationContextStatus
    model_context: AppliedModelContext | None = None
    items: list[RecommendationItem] = Field(max_length=150)

    @field_validator("items")
    @classmethod
    def unique_items(cls, value: list[RecommendationItem]) -> list[RecommendationItem]:
        if len({item.repo_id for item in value}) != len(value):
            raise ValueError("recommendation IDs must be unique")
        return value


from inference.scoring_provenance import ScoringContext, ScoredPacket, decode_owned_packet


class CapturedRecommendationItem(RecommendationItem):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    raw_heavy_score: float = Field(allow_inf_nan=False)
    scored_input: ScoredPacket = Field(repr=False)


class CapturedScoringStatus(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    schema_version: Literal[1]
    status: Literal["captured"]

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_schema(cls, value):
        if type(value) is not int:
            raise ValueError("Capture schema must be an integer")
        return value


class EmptyScoringStatus(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    schema_version: Literal[1]
    status: Literal["not_scored_empty"]

    _exact_schema = field_validator("schema_version", mode="before")(CapturedScoringStatus.exact_schema.__func__)


class CapturedRecommendationResponse(RecommendationResponse):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    feed_version: Annotated[str, StringConstraints(strict=True, pattern=r"^[1-9][0-9]{0,18}$")]
    scoring_capture: CapturedScoringStatus
    scoring_context: ScoringContext = Field(repr=False)
    scoring_metadata_complete: bool = Field(strict=True)
    served_ranker: Literal["heavy"]
    heavy_ranker_selected: Literal[True]
    ranker_applied: Literal[True]
    fallback_code: None
    fallback_reason: None
    retrieval_mode: Literal["personalized", "cold_start_profile"]
    items: list[CapturedRecommendationItem] = Field(min_length=1, max_length=150, repr=False)

    @model_validator(mode="after")
    def exact_captured_bindings(self):
        context = self.scoring_context
        if (context.user_id != str(self.user_id) or context.generation_id != str(self.generation_id)
                or context.feed_version != self.feed_version or context.model_version != self.model_version
                or self.scoring_metadata_complete != (context.user_embedding_version is not None)):
            raise ValueError("Captured response context binding mismatch")
        versions = []
        for item in self.items:
            decode_owned_packet(item.scored_input, context, owner_user_id=str(self.user_id),
                                generation_id=str(self.generation_id), repo_id=str(item.repo_id))
            versions.append(context.repository_embedding_versions[item.scored_input.snapshot.embedding_version_index])
        versions = sorted(set(versions))
        version = versions[0] if len(versions) == 1 else "compatible-mixed:" + ",".join(versions)
        if self.embedding_versions != versions or self.embedding_version != version:
            raise ValueError("Captured response embedding binding mismatch")
        return self


class UnscoredEmptyRecommendationResponse(RecommendationResponse):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    feed_version: Annotated[str, StringConstraints(strict=True, pattern=r"^[1-9][0-9]{0,18}$")]
    scoring_capture: EmptyScoringStatus
    fallback_code: None
    fallback_reason: None
    items: list[RecommendationItem] = Field(max_length=0)

    @field_validator("feed_version")
    @classmethod
    def bounded_empty_version(cls, value):
        if int(value) > 2**63-1:
            raise ValueError("Captured feed version exceeds int64")
        return value


RecommendationWireResponse = RecommendationResponse | CapturedRecommendationResponse | UnscoredEmptyRecommendationResponse


class FeedbackResponse(StrictModel):
    accepted: int = Field(ge=0, le=100, strict=True)
    duplicates: int = Field(ge=0, le=100, strict=True)
    durable: Literal[True]
    acknowledged_event_ids: list[uuid.UUID] = Field(
        min_length=1, max_length=100, json_schema_extra={"uniqueItems": True},
    )

    @model_validator(mode="after")
    def complete_acknowledgement(self):
        if len(set(self.acknowledged_event_ids)) != len(self.acknowledged_event_ids):
            raise ValueError("acknowledged event IDs must be unique")
        if self.accepted + self.duplicates != len(self.acknowledged_event_ids):
            raise ValueError("accepted plus duplicates must match acknowledged event IDs")
        return self


class RepositoryIndexResponse(StrictModel):
    schema_version: Literal[2]
    accepted: Literal[True]
    status: Literal["applied", "duplicate", "current"]
    job_id: uuid.UUID
    repo_id: uuid.UUID
    content_version: int = Field(ge=1)
    embedding_version: str = Field(min_length=1, max_length=512)
    card_summary: CardSummaryArtifact


class RepositoryRefreshResponse(StrictModel):
    accepted: Literal[True]
    status: Literal["applied", "duplicate", "current"]
    repo_id: uuid.UUID
    feature_version: int = Field(ge=1)


class OnboardingResponse(StrictModel):
    accepted: Literal[True]
    status: Literal["applied", "duplicate", "current"]
    user_id: uuid.UUID
    profile_version: int = Field(ge=1)


class ModelOperationInput(StrictModel):
    # Whitespace and original Unicode are part of the immutable source hash.
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=False, hide_input_in_errors=True)
    operation_json: str = Field(min_length=1, max_length=131072)


class ModelOperationBatch(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    schema_version: Literal[1]
    operations: list[ModelOperationInput] = Field(min_length=1, max_length=50)

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_schema(cls, value):
        if type(value) is not int:
            raise ValueError("Model operation schema must be an integer")
        return value


class ModelOperationAcknowledgement(StrictModel):
    operation_id: uuid.UUID
    model_operation_version: str
    source_envelope_sha256: str
    status: Literal["accepted", "duplicate", "withdrawal_accepted"]
    withdrawal_sha256: str | None = None


class ModelOperationResponse(StrictModel):
    schema_version: Literal[1]
    operations: list[ModelOperationAcknowledgement]


class ModelAuthorityRequest(StrictModel):
    schema_version: Literal[1]
    user_id: uuid.UUID
    model_context: ModelRecommendationContext


class ModelAuthorityResponse(StrictModel):
    schema_version: Literal[1]
    model_context: AppliedModelContext


class ProfileWithdrawalInput(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=False, hide_input_in_errors=True)
    withdrawal_json: str = Field(min_length=1, max_length=131072)


class ProfileWithdrawalBatch(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    schema_version: Literal[1]
    withdrawals: list[ProfileWithdrawalInput] = Field(min_length=1, max_length=50)

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_schema(cls, value):
        if type(value) is not int:
            raise ValueError("Withdrawal schema must be an integer")
        return value


class ProfileWithdrawalAcknowledgement(StrictModel):
    operation_id: uuid.UUID
    model_operation_version: PositiveModelDecimal
    source_envelope_sha256: str
    withdrawal_sha256: str
    status: Literal["accepted", "duplicate"]


class ProfileWithdrawalResponse(StrictModel):
    schema_version: Literal[1]
    withdrawals: list[ProfileWithdrawalAcknowledgement]


class ModelOperationStatusRequest(StrictModel):
    schema_version: Literal[1]
    user_id: uuid.UUID
    operation_id: uuid.UUID
    model_operation_version: PositiveModelDecimal
    source_envelope_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")


class ModelOperationStatusResponse(StrictModel):
    schema_version: Literal[1]
    user_id: uuid.UUID
    operation_id: uuid.UUID
    model_operation_version: PositiveModelDecimal
    source_envelope_sha256: str
    status: Literal["pending", "applied", "withdrawn_profile", "rejected_capacity"]
    basis_status: Literal["missing", "awaiting_profile", "ready"]
    applied_operation_version: PositiveModelDecimal | None = None
    applied_profile_version: PositiveModelDecimal | None = None
    applied_feedback_version: NonnegativeModelDecimal | None = None
    current_head: str | None = None
    operation_head: str | None = None
    withdrawal_sha256: str | None = None
