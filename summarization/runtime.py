"""Process-scoped summary provider runtime."""

from __future__ import annotations

from functools import lru_cache
import os
import redis

from .admission import RedisSummaryAdmission

from .pipeline import CardSummaryPipeline
from .provider import OpenRouterSummaryProvider
from .settings import SummarySettings


_admission_counts = {}


@lru_cache(maxsize=1)
def card_summary_pipeline() -> CardSummaryPipeline:
    global _admission_counts
    settings = SummarySettings.from_env()
    provider = None
    if settings.provider_enabled:
        url = os.getenv('REDIS_URL')
        client = redis.Redis.from_url(url, socket_timeout=0.25, socket_connect_timeout=0.25) if url else None
        # Shared admission rejects immediately. No process-local pacing sleep
        # holds the indexing executor or the shared encoder's semaphore.
        class NoWait:
            def wait(self): pass
        provider = OpenRouterSummaryProvider(settings, limiter=NoWait(),
            admission=RedisSummaryAdmission(settings, client, owns_client=True))
    _admission_counts = provider.admission.metrics if provider is not None else {}
    return CardSummaryPipeline(settings=settings, provider=provider)


def card_summary_cleanup_actions():
    """Capture the provider close independently of its pipeline cache clear."""
    actions = []
    if card_summary_pipeline.cache_info().currsize:
        close = getattr(card_summary_pipeline().provider, "close", None)
        if callable(close):
            actions.append(("summary.provider.close", close))
    return [*actions, ("summary.pipeline.cache.clear", card_summary_pipeline.cache_clear)]


def shutdown_card_summary_runtime() -> None:
    if card_summary_pipeline.cache_info().currsize:
        provider = card_summary_pipeline().provider
        close = getattr(provider, "close", None)
        if callable(close):
            close()
    card_summary_pipeline.cache_clear()


def summary_admission_metrics() -> dict[str, int]:
    """Network-free scrape; never initialize a provider solely for metrics."""
    outcomes = ('admitted', 'available', 'failed', 'abandoned', 'probes',
                'circuit_rejected', 'quota_rejected', 'coordination_unavailable', 'fallback_deferred')
    counts = _admission_counts
    return {name: max(0, min(2**63-1, counts.get(name, 0))) for name in outcomes}
