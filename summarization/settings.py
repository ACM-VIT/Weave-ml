"""Validated, summary-specific runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from urllib.parse import urlsplit

from .contracts import CARD_SUMMARY_FORMAT_VERSION, CARD_SUMMARY_PROMPT_VERSION


_SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._/-]{0,255}$")
OPENAI_SUMMARY_API_URL = "https://api.openai.com/v1/chat/completions"


def _integer(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    try:
        value = default if raw is None else int(raw.strip())
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _number(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    try:
        value = default if raw is None else float(raw.strip())
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return value


@dataclass(frozen=True, slots=True)
class SummarySettings:
    """All provider and artifact identity settings for card summaries."""

    provider: str = "openrouter"
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    api_key: str | None = None
    model_id: str = "meta-llama/llama-3.3-70b-instruct"
    prompt_version: str = CARD_SUMMARY_PROMPT_VERSION
    format_version: str = CARD_SUMMARY_FORMAT_VERSION
    input_max_chars: int = 12_000
    temperature: float = 0.1
    max_output_tokens: int = 240
    request_timeout_seconds: float = 5.0
    rpm_limit: float = 60.0
    tpm_limit: int = 120_000
    failure_threshold: int = 3
    cooldown_seconds: float = 30.0
    max_retries: int = 2
    retry_base_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.provider not in {"openrouter", "openai"}:
            raise ValueError("SUMMARY_PROVIDER must be 'openrouter' or 'openai'")
        if self.provider == "openai" and self.api_url != OPENAI_SUMMARY_API_URL:
            raise ValueError("SUMMARY_API_URL must be the official OpenAI endpoint")
        parsed = urlsplit(self.api_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "SUMMARY_API_URL must be an HTTPS URL without credentials, query, or fragment"
            )
        for field_name, value in (
            ("SUMMARY_MODEL_ID", self.model_id),
            ("SUMMARY_PROMPT_VERSION", self.prompt_version),
            ("SUMMARY_FORMAT_VERSION", self.format_version),
        ):
            if not _SAFE_VERSION.fullmatch(value):
                raise ValueError(f"{field_name} contains unsupported characters")
        if self.prompt_version != CARD_SUMMARY_PROMPT_VERSION:
            raise ValueError(
                f"SUMMARY_PROMPT_VERSION must be {CARD_SUMMARY_PROMPT_VERSION!r}"
            )
        if self.format_version != CARD_SUMMARY_FORMAT_VERSION:
            raise ValueError(
                f"SUMMARY_FORMAT_VERSION must be {CARD_SUMMARY_FORMAT_VERSION!r}"
            )
        if not 1_000 <= self.input_max_chars <= 30_000:
            raise ValueError("SUMMARY_INPUT_MAX_CHARS must be between 1000 and 30000")
        if not 0 <= self.temperature <= 0.2:
            raise ValueError("SUMMARY_TEMPERATURE must be between 0 and 0.2")
        if not 64 <= self.max_output_tokens <= 512:
            raise ValueError("SUMMARY_MAX_OUTPUT_TOKENS must be between 64 and 512")
        if not 0.1 <= self.request_timeout_seconds <= 30:
            raise ValueError("SUMMARY_REQUEST_TIMEOUT_SECONDS must be between 0.1 and 30")
        if not 0 < self.rpm_limit <= 600:
            raise ValueError("SUMMARY_RPM_LIMIT must be greater than 0 and at most 600")
        if type(self.tpm_limit) is not int or not 1 <= self.tpm_limit <= 10_000_000:
            raise ValueError("SUMMARY_TPM_LIMIT must be between 1 and 10000000")
        if type(self.failure_threshold) is not int or not 1 <= self.failure_threshold <= 20:
            raise ValueError("SUMMARY_FAILURE_THRESHOLD must be between 1 and 20")
        if not 1 <= self.cooldown_seconds <= 300:
            raise ValueError("SUMMARY_COOLDOWN_SECONDS must be between 1 and 300")
        if not 0 <= self.max_retries <= 2:
            raise ValueError("SUMMARY_MAX_RETRIES must be between 0 and 2")
        if not 0.1 <= self.retry_base_seconds <= 1:
            raise ValueError("SUMMARY_RETRY_BASE_SECONDS must be between 0.1 and 1")

    @property
    def provider_enabled(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    @classmethod
    def from_env(cls) -> "SummarySettings":
        provider = os.getenv("SUMMARY_PROVIDER", "openrouter").strip().casefold()
        return cls(
            provider=provider,
            api_url=os.getenv(
                "SUMMARY_API_URL",
                OPENAI_SUMMARY_API_URL if provider == "openai" else "https://openrouter.ai/api/v1/chat/completions",
            ).strip(),
            api_key=os.getenv("SUMMARY_API_KEY") or None,
            model_id=os.getenv(
                "SUMMARY_MODEL_ID",
                "gpt-6-luna" if provider == "openai" else "meta-llama/llama-3.3-70b-instruct",
            ).strip(),
            prompt_version=os.getenv(
                "SUMMARY_PROMPT_VERSION", CARD_SUMMARY_PROMPT_VERSION
            ).strip(),
            format_version=os.getenv(
                "SUMMARY_FORMAT_VERSION", CARD_SUMMARY_FORMAT_VERSION
            ).strip(),
            input_max_chars=_integer(
                "SUMMARY_INPUT_MAX_CHARS", 12_000, minimum=1_000, maximum=30_000
            ),
            temperature=_number(
                "SUMMARY_TEMPERATURE", 0.1, minimum=0, maximum=0.2
            ),
            max_output_tokens=_integer(
                "SUMMARY_MAX_OUTPUT_TOKENS", 240, minimum=64, maximum=512
            ),
            request_timeout_seconds=_number(
                "SUMMARY_REQUEST_TIMEOUT_SECONDS",
                5.0,
                minimum=0.1,
                maximum=30,
            ),
            rpm_limit=_number(
                "SUMMARY_RPM_LIMIT", 60.0, minimum=0.01, maximum=600
            ),
            tpm_limit=_integer("SUMMARY_TPM_LIMIT",120_000,minimum=1,maximum=10_000_000),
            failure_threshold=_integer("SUMMARY_FAILURE_THRESHOLD",3,minimum=1,maximum=20),
            cooldown_seconds=_number("SUMMARY_COOLDOWN_SECONDS",30,minimum=1,maximum=300),
            max_retries=_integer(
                "SUMMARY_MAX_RETRIES", 2, minimum=0, maximum=2
            ),
            retry_base_seconds=_number(
                "SUMMARY_RETRY_BASE_SECONDS", 1.0, minimum=0.1, maximum=1
            ),
        )
