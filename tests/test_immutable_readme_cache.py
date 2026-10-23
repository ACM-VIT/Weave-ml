"""Real immutable ref/manifest/blob paths, bounded caching and cancellation."""
from concurrent.futures import CancelledError, ThreadPoolExecutor
import threading
import time

import pytest
from requests.auth import AuthBase

from acquisition.github_graphql_client import GitHubClientError
from acquisition.github_rate_limit import GitHubRequestBudget, GitHubRetryLater
from acquisition.readme_cache import BoundedReadmeCache, SharedReadmeLoadUnavailable
import acquisition.repository_enricher as enrichment
from scripts.benchmark_readme_cache import GraphQLFixture, isolated_caches, run_benchmark

pytestmark = pytest.mark.unit


@pytest.fixture
def stores():
    with isolated_caches() as values:
        yield values


def wait_until(check):
    deadline = time.monotonic() + 2
    while not check():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not become true before deadline")
        time.sleep(.005)


def test_unchanged_content_revalidates_access_but_reuses_immutable_layers(stores):
    with GraphQLFixture() as fixture:
        client = fixture.client()
        try:
            first = client.get_readme("owner", "repo")
            second = client.get_readme("owner", "repo")
            assert first == second
            assert first.source_commit == fixture.commit and first.blob_oid
            assert first.source_path == "README.md"
            assert fixture.calls == {"ref": 2, "manifest": 1, "blob": 1}
            assert stores["README_BLOB_CACHE"].snapshot()["hits"] == 1
            assert stores["README_BLOB_CACHE"].snapshot()["bytes"] == len(first.encode("utf-8"))
            fixture.denied = True
            with pytest.raises(GitHubClientError):
                client.get_readme("owner", "repo")
            assert fixture.calls["ref"] == 3
            assert stores["README_BLOB_CACHE"].snapshot()["entries"] == 1
        finally:
            client.session.close()


def test_branch_path_content_and_commit_invalidate_only_correct_layers(stores):
    with GraphQLFixture("# Canonical\n\nOriginal text.") as fixture:
        client = fixture.client()
        try:
            first = client.get_readme("owner", "repo")
            fixture.branch = "release/v2"
            renamed = client.get_readme("owner", "repo")
            assert renamed.default_branch == "release/v2" and renamed.blob_oid == first.blob_oid
            assert fixture.calls == {"ref": 2, "manifest": 1, "blob": 1}
            fixture.set_files({"README.md": str(first)})
            moved = client.get_readme("owner", "repo")
            assert moved.source_commit != first.source_commit and moved.blob_oid == first.blob_oid
            assert fixture.calls == {"ref": 3, "manifest": 2, "blob": 1}
            fixture.set_files({"docs/README.md": str(first)})
            path = client.get_readme("owner", "repo")
            assert path.source_path == "docs/README.md" and path.base_url.endswith(f"/{fixture.commit}/docs/")
            assert fixture.calls["blob"] == 2
            fixture.set_files({"docs/README.md": str(first) + "\nUpdated."})
            updated = client.get_readme("owner", "repo")
            assert updated.blob_oid != first.blob_oid and updated.endswith("Updated.")
            assert fixture.calls["blob"] == 3
        finally:
            client.session.close()


def test_credentials_and_repository_identity_never_share_cached_private_artifacts(stores):
    with GraphQLFixture() as fixture:
        clients = [fixture.client(token) for token in ("scope-one", "scope-two")]
        try:
            for client in clients:
                client.get_readme("owner", "repo")
            clients[0].get_readme("other-owner", "repo")
            assert fixture.calls == {"ref": 3, "manifest": 3, "blob": 3}
            for store in stores.values():
                assert "scope-one" not in repr(store._values) and "scope-two" not in repr(store._values)
        finally:
            for client in clients:
                client.session.close()


def test_opaque_auth_bypasses_shared_content_and_document_caches(stores):
    class OpaqueAuth(AuthBase):
        value = "opaque-one"

        def __call__(self, request):
            request.headers["Authorization"] = self.value
            return request

    with GraphQLFixture() as fixture:
        client = fixture.client()
        client.session.auth = OpaqueAuth()
        try:
            first = client.get_readme("owner", "repo")
            client.session.auth.value = "opaque-two"
            second = client.get_readme("owner", "repo")
            assert first == second and first.authorization_scope is None
            assert fixture.calls == {"ref": 2, "manifest": 2, "blob": 2}
            assert all(store.snapshot()["entries"] == 0 for store in stores.values())
        finally:
            client.session.close()


@pytest.mark.parametrize("failure", ["tamper_blob", "truncate_blob"])
def test_invalid_hash_or_byte_count_is_retryable_and_never_retained(stores, failure):
    with GraphQLFixture() as fixture:
        client = fixture.client()
        setattr(fixture, failure, True)
        try:
            with pytest.raises(GitHubClientError, match="immutable identity"):
                client.get_readme("owner", "repo")
            assert stores["README_BLOB_CACHE"].snapshot()["entries"] == 0
            setattr(fixture, failure, False)
            assert client.get_readme("owner", "repo")
            assert fixture.calls == {"ref": 2, "manifest": 1, "blob": 2}
            assert stores["README_BLOB_CACHE"].snapshot()["inflight"] == 0
        finally:
            client.session.close()


def test_transient_manifest_is_not_negative_cached_but_confirmed_missing_is(stores):
    with GraphQLFixture() as fixture:
        fixture.set_files({})
        fixture.failures["manifest"] = 1
        client = fixture.client()
        try:
            with pytest.raises(GitHubClientError):
                client.get_readme("owner", "repo")
            assert stores["README_MANIFEST_CACHE"].snapshot()["entries"] == 0
            assert client.get_readme("owner", "repo") == ""
            assert client.get_readme("owner", "repo") == ""
            assert fixture.calls == {"ref": 3, "manifest": 2}
            fixture.set_files({"README.md": "# Newly present README"})
            assert client.get_readme("owner", "repo") == "# Newly present README"
        finally:
            client.session.close()


def test_partial_candidate_selection_is_not_retained_as_canonical_manifest(stores):
    with GraphQLFixture() as fixture:
        fixture.set_files({".github/README.md": "# Preferred", "README.md": "# Lower priority"})
        fixture.partial_manifest = True
        client = fixture.client()
        try:
            assert client.get_readme("owner", "repo") == "# Lower priority"
            assert stores["README_MANIFEST_CACHE"].snapshot()["entries"] == 0
            fixture.partial_manifest = False
            assert client.get_readme("owner", "repo") == "# Preferred"
            assert fixture.calls["manifest"] == 2
        finally:
            client.session.close()


def test_real_concurrent_clients_share_manifest_and_blob_work(stores):
    with GraphQLFixture() as fixture, ThreadPoolExecutor(max_workers=6) as pool:
        fixture.release_blob.clear()
        clients = [fixture.client() for _ in range(6)]
        futures = [pool.submit(client.get_readme, "owner", "repo") for client in clients]
        try:
            assert fixture.blob_started.wait(2)
            wait_until(lambda: stores["README_BLOB_CACHE"].snapshot().get("coalesced", 0) >= 5)
        finally:
            fixture.release_blob.set()
        try:
            values = [future.result(3) for future in futures]
            assert len(set(values)) == 1
            assert fixture.calls == {"ref": 6, "manifest": 1, "blob": 1}
            assert stores["README_BLOB_CACHE"].snapshot()["inflight"] == 0
        finally:
            for client in clients:
                client.session.close()


def test_processed_artifacts_invalidate_parser_context_and_cannot_be_poisoned(stores, monkeypatch):
    with GraphQLFixture() as fixture:
        client = fixture.client()
        original = enrichment.process_markdown
        calls = []

        def process(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(enrichment, "process_markdown", process)
        try:
            readme = client.get_readme("owner", "repo")
            enricher = enrichment.RepositoryEnricher(client)
            def parsed(branch="main"):
                return enricher._process_readme(readme, source_path=readme.source_path,
                    default_branch=branch, base_url=readme.base_url)
            first = parsed()
            first.clean_text = "poisoned"
            assert parsed().clean_text != "poisoned" and len(calls) == 1
            monkeypatch.setattr(enrichment, "README_PROCESSING_VERSION", "acceptance-parser-v2")
            assert parsed().raw_markdown == str(readme) and len(calls) == 2
            assert parsed("release/v2").default_branch == "release/v2" and len(calls) == 3
        finally:
            client.session.close()


def test_bounds_oversized_bypass_and_inflight_admission():
    cache = BoundedReadmeCache(max_entries=2, max_bytes=5, max_inflight=1)
    def load(key, value):
        budget = GitHubRequestBudget()
        try:
            return cache.get_or_load(key, lambda: value, budget, size=len)
        finally:
            budget.finish()
    for key, value in (("a", "aa"), ("b", "bb"), ("c", "ccc")):
        assert load(key, value) == value
    assert cache.snapshot()["bytes"] == 5 and cache.snapshot()["entries"] == 2
    assert cache.snapshot()["evictions"] == 1
    assert load("oversize", "xxxxxx") == "xxxxxx"
    assert cache.snapshot()["bytes"] == 5 and cache.snapshot()["entries"] == 2
    owner = GitHubRequestBudget()
    follower = GitHubRequestBudget(defer_retries=True)
    started, release = threading.Event(), threading.Event()
    def blocked():
        started.set()
        assert release.wait(3)
        return "x"
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(cache.get_or_load, "held", blocked, owner, size=len)
        try:
            assert started.wait(2)
            with pytest.raises(GitHubRetryLater):
                cache.get_or_load("next", lambda: pytest.fail("producer bound bypassed"), follower, size=len)
            assert cache.snapshot()["inflight"] == 1
        finally:
            release.set()
        assert future.result(2) == "x"
    owner.finish()
    follower.finish()


def test_producer_cancellation_is_retryable_for_healthy_followers():
    cache = BoundedReadmeCache(max_entries=2, max_bytes=100)
    cancelled, started, release = threading.Event(), threading.Event(), threading.Event()
    owner = GitHubRequestBudget(cancelled=cancelled.is_set, defer_retries=False)
    follower = GitHubRequestBudget(defer_retries=False)
    def blocked():
        started.set()
        assert release.wait(3)
        return "canonical"
    with ThreadPoolExecutor(max_workers=2) as pool:
        producer = pool.submit(cache.get_or_load, "same", blocked, owner, size=len)
        assert started.wait(2)
        waiting = pool.submit(cache.get_or_load, "same", lambda: pytest.fail("duplicate producer"), follower, size=len)
        try:
            wait_until(lambda: cache.snapshot().get("coalesced", 0) == 1)
            cancelled.set()
        finally:
            release.set()
        with pytest.raises(CancelledError):
            producer.result(2)
        with pytest.raises(SharedReadmeLoadUnavailable):
            waiting.result(2)
    owner.finish()
    follower.finish()
    assert cache.snapshot()["entries"] == 0 and cache.snapshot()["inflight"] == 0
    budget = GitHubRequestBudget()
    assert cache.get_or_load("same", lambda: "retried", budget, size=len) == "retried"
    budget.finish()


def test_cancelled_follower_does_not_cancel_or_release_healthy_producer():
    cache = BoundedReadmeCache(max_entries=2, max_bytes=100)
    cancelled, started, release = threading.Event(), threading.Event(), threading.Event()
    owner = GitHubRequestBudget(defer_retries=False)
    follower = GitHubRequestBudget(cancelled=cancelled.is_set, defer_retries=False)
    def blocked():
        started.set()
        assert release.wait(3)
        return "canonical"
    with ThreadPoolExecutor(max_workers=2) as pool:
        producer = pool.submit(cache.get_or_load, "same", blocked, owner, size=len)
        assert started.wait(2)
        waiting = pool.submit(cache.get_or_load, "same", lambda: pytest.fail("duplicate producer"), follower, size=len)
        try:
            wait_until(lambda: cache.snapshot().get("coalesced", 0) == 1)
            cancelled.set()
            with pytest.raises(CancelledError):
                waiting.result(2)
            follower.finish()
            assert cache.snapshot()["inflight"] == 1
        finally:
            release.set()
        assert producer.result(2) == "canonical"
    owner.finish()
    assert cache.snapshot()["entries"] == 1 and cache.snapshot()["inflight"] == 0


def test_benchmark_counts_actual_wire_savings_without_claiming_latency_thresholds():
    report = run_benchmark(iterations=3, delay=0)
    assert report["results"]["uncached"]["requests"] == {"ref": 3, "manifest": 3, "blob": 3}
    assert report["results"]["cached"]["requests"] == {"ref": 3, "manifest": 1, "blob": 1}
    assert report["avoided_requests"] == 4 and report["avoided_response_body_bytes"] > 0
    assert report["results"]["cached"]["cache"]["README_BLOB_CACHE"]["hits"] == 2
    assert report["results"]["cached"]["cache"]["README_BLOB_CACHE"]["stored_value_hit_ratio"] == pytest.approx(2 / 3)
