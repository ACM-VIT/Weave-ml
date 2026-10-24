"""Bounded OpenRouter and direct OpenAI transport for card summaries."""

from __future__ import annotations

import logging
import json
from copy import deepcopy
import threading
import time
from typing import Any, Callable, Protocol

import requests

from .prompt import RESPONSE_JSON_SCHEMA, SYSTEM_PROMPT, user_prompt
from .settings import SummarySettings
from .transport import BoundedSummarySession
from .admission import AdmissionDenied, LocalSummaryAdmission, fallback_key


logger = logging.getLogger(__name__)
MAX_RETRY_DELAY_SECONDS = 2.0
MAX_RATE_LIMIT_WAIT_SECONDS = 2.0
# Keep enough of the embedding job deadline for deterministic fallback,
# embedding, and the durable Qdrant write.  The provider is an enhancement,
# never a reason to abandon an otherwise valid repository job.
SUMMARY_COMPLETION_RESERVE_SECONDS = 10.0


class SummaryProviderError(RuntimeError):
    """Provider or transport failure safe for description fallback."""

    def __init__(self, message: str, *, defer_upgrade: bool = True):
        super().__init__(message)
        self.defer_upgrade = defer_upgrade


class SummaryProvider(Protocol):
    def generate(self, source: str, *, repair_feedback: str | None = None, deadline: float | None = None) -> str: ...


class SummaryRateLimiter:
    """Thread-safe minimum-spacing limiter shared by provider requests."""

    def __init__(
        self,
        rpm_limit: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if rpm_limit <= 0:
            raise ValueError("rpm_limit must be positive")
        self._spacing = 60.0 / rpm_limit
        self._clock = clock
        self._sleeper = sleeper
        self._last_request: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if self._last_request is not None:
                remaining = self._spacing - (now - self._last_request)
                if remaining > 0:
                    if remaining > MAX_RATE_LIMIT_WAIT_SECONDS:
                        raise SummaryProviderError(
                            "summary rate-limit wait exceeds the request budget"
                        )
                    self._sleeper(remaining)
                    now = self._clock()
            self._last_request = now


class OpenRouterSummaryProvider:
    def __init__(
        self,
        settings: SummarySettings,
        *,
        session: Any | None = None,
        limiter: SummaryRateLimiter | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        admission: Any | None = None,
    ) -> None:
        if not settings.provider_enabled:
            raise ValueError("SUMMARY_API_KEY is required to enable the summary provider")
        self.settings = settings
        self.session = session or BoundedSummarySession()
        self.limiter = limiter or SummaryRateLimiter(settings.rpm_limit)
        self.sleeper = sleeper
        self.admission = admission or LocalSummaryAdmission(settings)

    def defer_fallback(self, repo: Any) -> None:
        key = fallback_key(repo, self.settings)
        if key is not None: self.admission.defer(key)

    def fallback_deferred(self, repo: Any) -> bool:
        key = fallback_key(repo, self.settings)
        return key is not None and self.admission.deferred(key)

    def close(self) -> None:
        close = getattr(self.session, "close", None)
        if callable(close):
            close()
        admission_close = getattr(self.admission, 'close', None)
        if callable(admission_close): admission_close()

    def _request_payload(
        self,
        source: str,
        *,
        repair_feedback: str | None,
    ) -> dict[str, Any]:
        response_schema = RESPONSE_JSON_SCHEMA
        if self.settings.provider == "openai":
            response_schema = deepcopy(RESPONSE_JSON_SCHEMA)
            # OpenAI strict output requires every property, including an empty
            # highlights array. Keep the shared public/OpenRouter schema intact.
            response_schema["schema"]["required"] = list(response_schema["schema"]["properties"])
        payload = {
            "model": self.settings.model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_prompt(source, repair_feedback=repair_feedback),
                },
            ],
            "temperature": self.settings.temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": response_schema,
            },
        }
        if self.settings.provider == "openai":
            payload["max_completion_tokens"] = self.settings.max_output_tokens
            if self.settings.model_id == "gpt-6-luna" or self.settings.model_id.startswith("gpt-6-luna-"):
                payload["reasoning_effort"] = "none"
        else:
            payload["max_tokens"] = self.settings.max_output_tokens
            payload["provider"] = {"require_parameters": True}
        return payload

    def generate(self, source: str, *, repair_feedback: str | None = None, deadline: float | None = None) -> str:
        if not source or len(source) > self.settings.input_max_chars:
            raise ValueError("summary source is empty or exceeds the configured bound")
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        if self.settings.provider == "openrouter":
            headers["X-Title"] = "Weave Repository Summaries"
        payload = self._request_payload(source, repair_feedback=repair_feedback)
        attempts = self.settings.max_retries + 1
        for attempt in range(attempts):
            self._request_budget(deadline)
            self.limiter.wait()
            # UTF-8 bytes upper-bound input tokens; charge the output ceiling
            # for every HTTP attempt, including repair and failed requests.
            tokens = len(json.dumps(payload, ensure_ascii=False).encode('utf-8')) + self.settings.max_output_tokens
            try:
                permit = self.admission.acquire(tokens)
            except AdmissionDenied as exc:
                raise SummaryProviderError(str(exc)) from exc
            try:
                request_timeout = min(self.settings.request_timeout_seconds, self._request_budget(deadline))
            except SummaryProviderError:
                self.admission.report(permit, None)
                raise
            try:
                response = self.session.post(
                    self.settings.api_url,
                    json=payload,
                    headers=headers,
                    timeout=request_timeout,
                )
            except requests.RequestException as exc:
                self.admission.report(permit, False)
                if attempt + 1 >= attempts:
                    raise SummaryProviderError("summary provider transport failed") from exc
                self.sleeper(
                    min(
                        self.settings.retry_base_seconds * (2**attempt),
                        MAX_RETRY_DELAY_SECONDS,
                    )
                )
                continue

            status_code = int(response.status_code)
            # A model's invalid content goes through validation/repair; it does
            # not open an availability circuit on a successful HTTP response.
            if status_code == 200:
                try:
                    body = response.json()
                    content = body["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    self.admission.report(permit, False)
                    raise SummaryProviderError(
                        "summary provider returned an invalid response envelope"
                    ) from exc
                if not isinstance(content, str) or len(content) > 8_192:
                    self.admission.report(permit, False)
                    raise SummaryProviderError(
                        "summary provider returned invalid bounded content"
                    )
                self.admission.report(permit, True)
                return content.strip()

            self.admission.report(permit, False)
            retryable = status_code == 429 or 500 <= status_code <= 599
            if not retryable or attempt + 1 >= attempts:
                raise SummaryProviderError(
                    f"summary provider request failed with HTTP {status_code}"
                )
            retry_after = response.headers.get("Retry-After")
            try:
                requested_delay = float(retry_after) if retry_after is not None else 0.0
            except ValueError:
                requested_delay = 0.0
            if requested_delay > MAX_RETRY_DELAY_SECONDS:
                raise SummaryProviderError(
                    "summary provider retry delay exceeds the request budget"
                )
            delay = max(
                requested_delay,
                self.settings.retry_base_seconds * (2**attempt),
            )
            self.sleeper(min(delay, MAX_RETRY_DELAY_SECONDS))

        raise SummaryProviderError("summary provider retry budget was exhausted")

    @staticmethod
    def _request_budget(deadline: float | None) -> float:
        """Return provider time that cannot consume the job's completion reserve."""

        if deadline is None:
            return float("inf")
        remaining = deadline - time.monotonic() - SUMMARY_COMPLETION_RESERVE_SECONDS
        if remaining < 0.1:
            raise SummaryProviderError(
                "summary generation would consume the embedding completion reserve",
                defer_upgrade=False,
            )
        return remaining
