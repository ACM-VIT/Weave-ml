"""Failure categories shared by trending storage and scheduling."""


class IncompleteTrendingSnapshot(RuntimeError):
    """Missing enrichment may succeed on a bounded same-snapshot retry."""
