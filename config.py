import hmac
import os

from dotenv import load_dotenv


# Load local development configuration before freezing module-level contracts.
# Existing process environment variables still take precedence.
load_dotenv()

V2_REPOSITORY_COLLECTION_NAME = "osiris_research_corpus_v2_20260722_r1"

def internal_api_header_name() -> str:
    """Return the normalized service-auth header shared by every API layer."""
    return os.getenv("INTERNAL_API_HEADER", "x-internal-secret").strip().lower() or "x-internal-secret"


def constant_time_secret_matches(supplied: str | None, expected: str) -> bool:
    """Compare service credentials without str-only compare_digest failures.

    ``hmac.compare_digest`` rejects non-ASCII ``str`` values.  Comparing UTF-8
    bytes keeps authentication fail-closed for malformed headers and avoids a
    request-triggered ``TypeError``.
    """

    if not supplied:
        return False
    try:
        return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))
    except (AttributeError, UnicodeEncodeError):
        return False


def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    value_text = raw_value.strip()
    if not value_text:
        raise ValueError(f"{name} must be a positive integer; got an empty value.")
    try:
        value = int(value_text)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer; got {raw_value!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value}.")
    return value


def _non_negative_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    value_text = raw_value.strip()
    if not value_text:
        raise ValueError(f"{name} must be a non-negative integer; got an empty value.")
    try:
        value = int(value_text)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer; got {raw_value!r}.") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer; got {value}.")
    return value

NOVELTY_THRESHOLD               = 0.35
TOP_K_COMPARISONS               = 5
EMBEDDING_MODEL                 = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM                   = _positive_int_env("VECTOR_DIMENSION", 384)
COLLECTION_NAME                 = V2_REPOSITORY_COLLECTION_NAME
QDRANT_VECTOR_NAME              = os.getenv("QDRANT_VECTOR_NAME", "repo_embedding")
USER_PROFILES_COLLECTION_NAME   = os.getenv(
    "USER_PROFILES_COLLECTION", "user_profiles"
)
MAX_DOC_SCORE                   = 100
GATE_APPROVAL_THRESHOLD         = 0.60
MIN_STARS_PREFILTER             = 50
MIN_README_PREFILTER            = 200


DUPLICATE_SIMILARITY_THRESHOLD  = 0.94
WRAPPER_SIMILARITY_THRESHOLD    = 0.85

HYBRID_WEIGHTS = {
    "readme":       0.40,
    "description":  0.25,
    "topics":       0.20,
    "category":     0.10,
    "language":     0.05,
}

NOVELTY_WEIGHTS = {
    "semantic":   0.60,
    "tech_stack": 0.20,
    "category":   0.10,
    "activity":   0.10,
}


# The below configuration is for the repository embedding pipeline. Environment
# variables are used here so deployments can change models, chunking, and Qdrant
# targets without editing source code.
REPOSITORY_EMBEDDING_MODEL = EMBEDDING_MODEL
REPOSITORY_EMBEDDING_DIM = EMBEDDING_DIM
REPOSITORY_EMBEDDING_VERSION = os.getenv("REPOSITORY_EMBEDDING_VERSION", "repo-embedding-v2")
REPOSITORY_FEATURE_SPEC_VERSION = os.getenv(
    "REPOSITORY_FEATURE_SPEC_VERSION", "v2"
).strip()
if not REPOSITORY_FEATURE_SPEC_VERSION:
    raise ValueError("REPOSITORY_FEATURE_SPEC_VERSION must be non-empty")
EMBEDDING_MODEL_REVISION = os.getenv(
    "EMBEDDING_MODEL_REVISION",
    "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
).strip()
if not EMBEDDING_MODEL_REVISION:
    raise ValueError("EMBEDDING_MODEL_REVISION must be a non-empty immutable revision")
README_CHUNK_CHARS = _positive_int_env("README_CHUNK_CHARS", 2500)
README_CHUNK_OVERLAP_CHARS = _non_negative_int_env("README_CHUNK_OVERLAP_CHARS", 250)
if README_CHUNK_OVERLAP_CHARS >= README_CHUNK_CHARS:
    raise ValueError(
        "README_CHUNK_OVERLAP_CHARS must be smaller than README_CHUNK_CHARS; "
        f"got {README_CHUNK_OVERLAP_CHARS} >= {README_CHUNK_CHARS}."
    )

REPO_TOWER_WEIGHTS = {
    "readme": 0.60,
    "metadata": 0.25,
    "topics": 0.15,
}

# The below Qdrant settings are for collection bootstrap, indexing, and CLI
# validation commands.
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION_NAME", COLLECTION_NAME
).strip()
if QDRANT_COLLECTION_NAME != V2_REPOSITORY_COLLECTION_NAME:
    raise ValueError(
        "QDRANT_COLLECTION_NAME must target the strict V2 repository corpus "
        f"{V2_REPOSITORY_COLLECTION_NAME!r}; got {QDRANT_COLLECTION_NAME!r}"
    )
QDRANT_DISTANCE = os.getenv("QDRANT_DISTANCE", "Cosine")
QDRANT_PAYLOAD_INDEX_SCHEMA = {
    "repo_id": "keyword",
    "serving_eligibility_version": "keyword",
    "embedding_model": "keyword",
    "embedding_model_revision": "keyword",
    "embedding_version": "keyword",
    "embedding_dim": "integer",
    "feature_spec_version": "keyword",
    "model_version": "keyword",
    "primary_language": "keyword",
    "topics": "keyword",
    "category": "keyword",
    "discovery_category": "keyword",
    "discovery_band": "keyword",
    "star_count": "integer",
    "trend_velocity": "float",
    "activity_score": "float",
    "doc_quality": "float",
    "code_health": "float",
    "pushed_days_ago": "integer",
    "content_version": "integer",
    "updated_at": "datetime",
    "pushed_at": "datetime",
}
QDRANT_PAYLOAD_INDEX_FIELDS = list(QDRANT_PAYLOAD_INDEX_SCHEMA)

# ── Dwell-time signal configuration ──────────────────────────────────────────
# These constants control how observed dwell time on a repository card is
# translated into an embedding learning-rate (alpha) for the vector shift.
#
#   MIN_DWELL_SECONDS  — dwells shorter than this are treated as accidental
#                        scrolls and silently ignored (no embedding update).
#   MAX_DWELL_SECONDS  — cap for the log-linear mapping; dwells beyond this
#                        saturate at DWELL_BASE_ALPHA.
#   DWELL_BASE_ALPHA   — maximum shift strength applied when a user reads a
#                        repo card to completion (analogous to ACTION_WEIGHTS).

def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a float; got {raw!r}.") from exc

MIN_DWELL_SECONDS: float = _float_env("MIN_DWELL_SECONDS", 3.0)
MAX_DWELL_SECONDS: float = _float_env("MAX_DWELL_SECONDS", 300.0)
DWELL_BASE_ALPHA:  float = _float_env("DWELL_BASE_ALPHA",  0.15)

if MIN_DWELL_SECONDS < 0:
    raise ValueError("MIN_DWELL_SECONDS must be non-negative.")
if MAX_DWELL_SECONDS <= MIN_DWELL_SECONDS:
    raise ValueError("MAX_DWELL_SECONDS must be greater than MIN_DWELL_SECONDS.")
if not (0.0 < DWELL_BASE_ALPHA <= 1.0):
    raise ValueError("DWELL_BASE_ALPHA must be in the range (0, 1].")
