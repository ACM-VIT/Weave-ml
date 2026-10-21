"""Deterministic acquisition quota and identity regression fixtures."""

from collections import Counter

import pytest

from acquisition.github_discovery import DiscoveryConfig, GitHubDiscoveryEngine


class SearchFixture:
    def __init__(self, *, duplicate_only=False, empty=False):
        self.queries = []
        self.duplicate_only = duplicate_only
        self.empty = empty

    def execute(self, _query, variables):
        self.queries.append(variables["query"])
        if self.empty:
            nodes = []
        elif self.duplicate_only:
            nodes = [{"nameWithOwner": "Owner/One"}, {"nameWithOwner": "owner/one"}] * 10
        else:
            nodes = [
                {"nameWithOwner": f"owner/query-{len(self.queries)}-repo-{index}"}
                for index in range(100)
            ]
        return {"data": {"search": {"nodes": nodes}}}


def engine(client, **kwargs):
    result = GitHubDiscoveryEngine(client, config=DiscoveryConfig(random_seed=7, **kwargs))
    result._query_for_band = lambda category, _terms, band: f"{band}:{category}"
    return result


@pytest.mark.parametrize("limit", [0, 1, 2, 3, 4, 5, 10, 170])
def test_discovery_respects_each_band_and_stops_searching_full_bands(limit):
    client = SearchFixture()
    discovery = engine(client)

    results = discovery.discover(limit=limit)

    allocations = discovery._allocations(limit)
    assert len(results) == limit
    assert len({item["full_name"].casefold() for item in results}) == limit
    counts = Counter(item["_discovery_band"] for item in results)
    for band, quota in allocations.items():
        assert counts[band] == quota
        # Each fixture query can fill up to ten rows at these target sizes.
        assert sum(query.startswith(f"{band}:") for query in client.queries) == (quota + 9) // 10


def test_zero_configured_target_does_not_search():
    client = SearchFixture()
    assert engine(client, total_limit=0).discover() == []
    assert client.queries == []


def test_bands_continue_category_cycle_instead_of_reusing_first_category():
    client = SearchFixture()
    results = engine(client).discover(limit=10)
    assert len({item["_discovery_category"] for item in results}) == 4
    assert len({query.split(":", 1)[1] for query in client.queries}) == 4


def test_duplicate_heavy_results_preserve_identity_and_exhaust_finite_categories():
    client = SearchFixture(duplicate_only=True)
    discovery = engine(client)
    results = discovery.discover(limit=170)

    assert [item["full_name"] for item in results] == ["Owner/One"]
    assert len(client.queries) == 4 * 23


def test_exhausted_categories_return_partial_results_without_refill_loop():
    client = SearchFixture(empty=True)
    assert engine(client).discover(limit=170) == []
    assert len(client.queries) == 4 * 23


@pytest.mark.parametrize("limit", [-1, 1.5, True])
def test_invalid_target_is_rejected_before_search(limit):
    client = SearchFixture()
    with pytest.raises(ValueError, match="non-negative integer"):
        engine(client).discover(limit=limit)
    assert client.queries == []
