"""Production API for feedback ingestion and ML operations."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import re
import time
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from config import constant_time_secret_matches, internal_api_header_name
from api.metrics import record_api_request

load_dotenv()

logger = logging.getLogger("pipeline.api")
MAX_REQUEST_BODY_BYTES = 256 * 1024
REPOSITORY_EMBED_MAX_REQUEST_BODY_BYTES = 8 * 1024 * 1024
MAX_VALIDATION_ERROR_DETAILS = 50

def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "") or uuid.uuid4())


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    retryable: bool,
    headers: dict[str, str] | None = None,
    details: Any | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    body: dict[str, Any] = {
        # Keep FastAPI's historical detail field for backend compatibility.
        "detail": message,
        "code": code,
        "message": message,
        "retryable": retryable,
        "request_id": request_id,
    }
    if details is not None:
        body["details"] = details
    response_headers = dict(headers or {})
    response_headers["x-request-id"] = request_id
    return JSONResponse(
        status_code=status_code,
        content=body,
        headers=response_headers,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start and validate the V2-only runtime."""
    from api.v2 import shutdown_v2_runtime, validate_v2_runtime_configuration
    from embedding.runtime import embedding_warmup_enabled, warm_embedding_runtime
    from scripts.validate_production_config import validate_production_config

    if os.getenv("APP_ENV", "development").strip().casefold() == "production":
        production_issues = validate_production_config()
        if production_issues:
            issue_names = ", ".join(sorted({issue.name for issue in production_issues}))
            raise RuntimeError(f"Production configuration is invalid: {issue_names}")
    settings = validate_v2_runtime_configuration()
    secret = os.getenv("INTERNAL_API_SECRET", "")
    if settings.production and not re.fullmatch(r"[0-9a-f]{64}", secret):
        raise RuntimeError(
            "INTERNAL_API_SECRET must be 64 lowercase hexadecimal characters in production"
        )
    warmup_enabled = embedding_warmup_enabled()
    if settings.production and not warmup_enabled:
        raise RuntimeError("EMBEDDING_WARMUP_ON_STARTUP must be enabled in production")
    app.state.feedback_settings = settings
    if warmup_enabled:
        app.state.embedding_runtime = await asyncio.to_thread(warm_embedding_runtime)
    app.state.feedback_settings = settings
    try:
        yield
    finally:
        await asyncio.to_thread(shutdown_v2_runtime)


app = FastAPI(
    title="Git Social ML API",
    description="Authenticated ML operations and durable feedback ingestion.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def request_validation_error(request: Request, exc: RequestValidationError):
    validation_errors = exc.errors()
    details = [
        {
            "path": ".".join(str(item) for item in error.get("loc", ())),
            "message": error.get("msg", "Invalid value"),
            "type": error.get("type", "validation_error"),
        }
        # FastAPI's validation wrapper supports a narrower ``errors`` signature
        # than Pydantic on some supported versions.  We copy only safe fields
        # below, so raw inputs are never returned or logged.
        for error in validation_errors[:MAX_VALIDATION_ERROR_DETAILS]
    ]
    if len(validation_errors) > MAX_VALIDATION_ERROR_DETAILS:
        details.append(
            {
                "path": "body",
                "message": "Additional validation errors were omitted.",
                "type": "validation_errors_omitted",
            }
        )
    return _error_response(
        request,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        code="REQUEST_VALIDATION_FAILED",
        message="Request validation failed.",
        retryable=False,
        details=details,
    )


@app.exception_handler(StarletteHTTPException)
async def service_http_error(request: Request, exc: StarletteHTTPException):
    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    code_by_status = {
        500: "INTERNAL_ERROR",
        401: "UNAUTHORIZED",
        404: "NOT_FOUND",
        409: "VERSION_CONFLICT",
        422: "REQUEST_VALIDATION_FAILED",
        429: "RATE_LIMITED",
        503: "DEPENDENCY_UNAVAILABLE",
    }
    explicit_code = getattr(exc, "error_code", None)
    explicit_retryable = getattr(exc, "retryable", None)
    explicit_details = getattr(exc, "safe_details", None)
    return _error_response(
        request,
        status_code=exc.status_code,
        code=(
            explicit_code
            if isinstance(explicit_code, str) and explicit_code
            else code_by_status.get(exc.status_code, f"HTTP_{exc.status_code}")
        ),
        message=message,
        retryable=(
            explicit_retryable
            if isinstance(explicit_retryable, bool)
            else exc.status_code in {408, 425, 429, 502, 503, 504}
        ),
        headers=dict(exc.headers or {}),
        details=explicit_details if isinstance(explicit_details, dict) else None,
    )


@app.exception_handler(Exception)
async def unhandled_service_error(request: Request, exc: Exception):
    logger.error(
        "Unhandled ML API error request_id=%s path=%s error_type=%s",
        _request_id(request),
        request.url.path,
        type(exc).__name__,
        extra={
            "request_context": {
                "request_id": _request_id(request),
                "path": request.url.path,
                "error_type": type(exc).__name__,
            }
        },
    )
    return _error_response(
        request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="INTERNAL_ERROR",
        message="The ML service could not complete the request.",
        retryable=False,
    )


@app.middleware("http")
async def authenticate_non_health_routes(request: Request, call_next):
    """Fail closed for every route except the single health endpoint."""
    supplied_request_id = request.headers.get("x-request-id", "").strip()
    request.state.request_id = (
        supplied_request_id
        if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", supplied_request_id)
        else str(uuid.uuid4())
    )
    if request.method in {"POST", "PUT", "PATCH"}:
        raw_content_length = request.headers.get("content-length")
        if raw_content_length is None:
            return _error_response(
                request,
                status_code=status.HTTP_411_LENGTH_REQUIRED,
                code="CONTENT_LENGTH_REQUIRED",
                message="Content-Length is required for request bodies.",
                retryable=False,
            )
        try:
            content_length = int(raw_content_length)
        except ValueError:
            content_length = -1
        if content_length < 0:
            return _error_response(
                request,
                status_code=status.HTTP_400_BAD_REQUEST,
                code="INVALID_CONTENT_LENGTH",
                message="Content-Length is invalid.",
                retryable=False,
            )
        request_limit = (
            REPOSITORY_EMBED_MAX_REQUEST_BODY_BYTES
            if request.url.path == "/api/v2/repositories/embed"
            else MAX_REQUEST_BODY_BYTES
        )
        if content_length > request_limit:
            return _error_response(
                request,
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                code="REQUEST_TOO_LARGE",
                message="Request body exceeds the service limit.",
                retryable=False,
            )
    secret = os.getenv("INTERNAL_API_SECRET")
    if not secret:
        return _error_response(
            request,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="AUTH_NOT_CONFIGURED",
            message="Internal API authentication is not configured.",
            retryable=False,
        )
    supplied = request.headers.get(internal_api_header_name())
    if not constant_time_secret_matches(supplied, secret):
        return _error_response(
            request,
            status_code=status.HTTP_401_UNAUTHORIZED,
            code="UNAUTHORIZED",
            message="Unauthorized.",
            retryable=False,
        )
    response = await call_next(request)
    response.headers["x-request-id"] = request.state.request_id
    return response


@app.middleware("http")
async def observe_api_requests(request: Request, call_next):
    """Record bounded, fixed-cardinality request metrics for the V2 API."""

    started = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        if request.url.path.startswith("/api/v2/"):
            record_api_request(
                path=request.url.path,
                method=request.method,
                status_code=response.status_code if response is not None else 500,
                duration_seconds=time.perf_counter() - started,
            )


from api.v2 import router as v2_router

app.include_router(v2_router)
