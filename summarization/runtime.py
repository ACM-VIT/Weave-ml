"""Process-scoped summary provider runtime."""

from __future__ import annotations

from functools import lru_cache

from .pipeline import CardSummaryPipeline
from .provider import OpenRouterSummaryProvider
from .settings import SummarySettings


@lru_cache(maxsize=1)
def card_summary_pipeline() -> CardSummaryPipeline:
    settings = SummarySettings.from_env()
    provider = OpenRouterSummaryProvider(settings) if settings.provider_enabled else None
    return CardSummaryPipeline(settings=settings, provider=provider)


def shutdown_card_summary_runtime() -> None:
    if card_summary_pipeline.cache_info().currsize:
        provider = card_summary_pipeline().provider
        close = getattr(provider, "close", None)
        if callable(close):
            close()
    card_summary_pipeline.cache_clear()
