"""Transform GitHub repository data into Osiris-compatible payloads."""

from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from copy import deepcopy
import hashlib
from datetime import datetime, timezone
import math
import re
from typing import Any, TYPE_CHECKING
import logging

from .github_graphql_client import (
    FetchedReadme,
    GitHubGraphQLClient,
    GitHubGraphQLClientError,
    build_readme_base_url,
)
from .graphql_queries import README_CANDIDATES
from .models import AcquisitionFailure, github_failure_disposition
from .repository_lineage import lineage_from_graphql
from utils.readme_processor import README_PROCESSING_VERSION, ReadmeDocument, process_markdown
from .github_rate_limit import GitHubRequestBudget
from .readme_cache import README_DOCUMENT_CACHE
from .identity import normalize_repository_name, repository_identity_key
from .config import metadata_batch_size as validated_metadata_batch_size

if TYPE_CHECKING:
    from .readme_scheduler import ReadmeScheduler

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class EnrichmentResult:
    repo_id: str
    payload: dict[str, Any]
    raw_repository: dict[str, Any]
    readme: ReadmeDocument
    topics: list[str]
    languages: dict[str, int]
    warnings: list[str] = field(default_factory=list)
    failure_disposition: str = "retryable"


class EnrichmentBatch(list[EnrichmentResult]):
    """List-compatible successful results with explicit failed identities."""

    def __init__(self, *, failures: list[AcquisitionFailure]) -> None:
        super().__init__()
        self.failures = failures


class RepositoryEnricher:
    def __init__(
        self,
        graphql_client: GitHubGraphQLClient | None = None,
        *,
        metadata_batch_size: int | None = None,
        metadata_projection: str = "legacy",
        readme_scheduler: ReadmeScheduler | None = None,
    ) -> None:
        self.metadata_batch_size = validated_metadata_batch_size(metadata_batch_size)
        if metadata_projection not in {"legacy", "delivery"}:
            raise ValueError("Unknown metadata projection")
        self.metadata_projection = metadata_projection
        self.readme_scheduler = readme_scheduler
        self.graphql_client = graphql_client or GitHubGraphQLClient()

    def _process_readme(
        self, value: str, *, source_path: str | None,
        default_branch: str | None, base_url: str | None,
    ) -> ReadmeDocument:
        source_commit = getattr(value, "source_commit", None)
        blob_oid = getattr(value, "blob_oid", None)
        scope = getattr(value, "authorization_scope", None)
        repository = getattr(value, "source_repository", None)

        def process() -> ReadmeDocument:
            document = process_markdown(
                str(value), source_path=source_path,
                default_branch=default_branch, base_url=base_url,
            )
            document.source_commit = source_commit
            document.blob_oid = blob_oid
            return document

        if not (scope and repository and source_path and blob_oid):
            return process()
        # Include media resolution context and processing version; the same
        # blob can appear on a renamed branch/path or under a different parser.
        key = (
            scope, repository, source_path, blob_oid, README_PROCESSING_VERSION,
            source_commit, default_branch, base_url,
            hashlib.sha256(str(value).encode("utf-8")).hexdigest(),
        )
        deadline = getattr(self.graphql_client, "deadline", None)
        deadline = deadline if type(deadline) in {int, float} else None
        budget = GitHubRequestBudget(
            deadline=deadline, defer_retries=False,
            cancelled=lambda: bool(self.readme_scheduler is not None and self.readme_scheduler.cancelled),
        )
        try:
            document = README_DOCUMENT_CACHE.get_or_load(
                key, process, budget,
                size=lambda result: len(result.raw_markdown.encode("utf-8"))
                + len(result.clean_text.encode("utf-8"))
                + sum(len(paragraph.encode("utf-8")) for paragraph in result.extracted_paragraphs),
            )
            # Downstream summary stages may mutate ReadmeDocument. They must
            # never modify a canonical/shared artifact seen by another caller.
            return deepcopy(document)
        finally:
            budget.finish()

    def _check_cancelled(self) -> None:
        if self.readme_scheduler is not None and self.readme_scheduler.cancelled:
            raise CancelledError("Acquisition cancelled")

    def enrich(self, repository: dict[str, Any] | str) -> EnrichmentResult | None:
        full_name = normalize_repository_name(
            repository if isinstance(repository, str) else repository.get("full_name")
        )
        if not full_name:
            return None
        
        # Determine discovery metadata to carry over
        discovery_category = None
        discovery_band = None
        if isinstance(repository, dict):
            discovery_category = repository.get("_discovery_category")
            discovery_band = repository.get("_discovery_band")

        owner, _, name = full_name.partition("/")
        if owner and name:
            try:
                result = self._enrich_graphql(owner, name, discovery_category, discovery_band)
                if result:
                    return result
            except (CancelledError, TimeoutError):
                raise
            except Exception as e:
                logger.warning(f"GraphQL enrichment failed for {full_name}: {e}.")

        return None

    def get_repositories_batch(self, repositories: list[dict[str, Any] | str]) -> list[EnrichmentResult]:
        """Fetch bounded metadata batches, followed by individual READMEs."""
        targets = []
        repo_metadata = {}
        
        for repository in repositories:
            full_name = normalize_repository_name(
                repository if isinstance(repository, str) else repository.get("full_name")
            )
            if not full_name:
                continue
            owner, _, name = full_name.partition("/")
            if owner and name:
                targets.append((owner, name))
                if isinstance(repository, dict):
                    repo_metadata[repository_identity_key(full_name)] = {
                        "_discovery_category": repository.get("_discovery_category"),
                        "_discovery_band": repository.get("_discovery_band")
                    }
        
        results = []
        metadata_failures: list[AcquisitionFailure] = []

        def failed_rows(rows: list[tuple[str, str]], error: GitHubGraphQLClientError) -> None:
            metadata_failures.extend(
                AcquisitionFailure(f"{owner}/{name}", "metadata_enrichment", error.safe_diagnostic,
                                   disposition=github_failure_disposition(error))
                for owner, name in rows
            )
            logger.warning("Metadata chunk failed: %s", error.safe_diagnostic)

        def observed_rows(rows):
            # Freeze provenance at each completed GitHub call, before fallback
            # siblings/README processing can delay this response's application.
            received_at = datetime.now(timezone.utc).isoformat()
            return {name: ({**data, "_lifecycle_observed_at": received_at} if data else data)
                    for name, data in rows.items()}

        # Pass 1: batch metadata (no readme — avoids 502 on large repos)
        for i in range(0, len(targets), self.metadata_batch_size):
            self._check_cancelled()
            batch = targets[i:i+self.metadata_batch_size]
            batch_res = {}
            try:
                if self.metadata_projection == "legacy":
                    batch_res = observed_rows(self.graphql_client.get_repositories_batch(batch))
                else:
                    batch_res = observed_rows(self.graphql_client.get_repositories_batch(batch, projection=self.metadata_projection))
            except CancelledError:
                raise
            except GitHubGraphQLClientError as exc:
                # The client already exhausted/classified this request. Keep
                # prior chunks and record only the failed identities; never
                # fan out a transport/auth/rate failure into new retry loops.
                if exc.category != "response" or len(batch) == 1:
                    failed_rows(batch, exc)
                    continue
                logger.warning("Metadata response failed; isolating repository rows")
                for position, (owner, name) in enumerate(batch):
                    self._check_cancelled()
                    try:
                        retry_res = self.graphql_client.get_repositories_batch(
                            [(owner, name)], projection=self.metadata_projection,
                        )
                        batch_res.update(observed_rows(retry_res))
                    except CancelledError:
                        raise
                    except GitHubGraphQLClientError as retry_exc:
                        if retry_exc.category != "response":
                            # Preserve already resolved singleton rows too;
                            # unqueried siblings stay retryable, not amplified.
                            failed_rows(batch[position:], retry_exc)
                            break
                        failed_rows([(owner, name)], retry_exc)

            for full_name, data in batch_res.items():
                if not data:
                    continue
                meta = repo_metadata.get(repository_identity_key(full_name), {})
                try:
                    result = self._process_graphql_data(
                        data,
                        meta.get("_discovery_category"),
                        meta.get("_discovery_band"),
                        readme_text=None,  # fetched in Pass 2
                    )
                    if result:
                        results.append(result)
                except Exception as exc:
                    logger.warning("Failed to process metadata for %s: %s", full_name, exc)

        # Pass 2: fetch README individually for each result
        owner_name_map = {
            r.repo_id: tuple(r.repo_id.split("/", 1))
            for r in results
        }
        enriched = EnrichmentBatch(failures=metadata_failures)
        readme_futures = (
            self.readme_scheduler.ordered(owner_name_map[result.repo_id] for result in results)
            if self.readme_scheduler is not None else None
        )
        for result in results:
            repo_id = result.repo_id
            owner, name = owner_name_map.get(repo_id, (None, None))
            readme_text: str | FetchedReadme = ""
            if owner and name:
                try:
                    readme_text = (
                        next(readme_futures).result()
                        if readme_futures is not None
                        else self.graphql_client.get_readme(owner, name)
                    )
                except CancelledError:
                    raise
                except Exception as exc:
                    diagnostic = exc.safe_diagnostic if isinstance(exc, GitHubGraphQLClientError) else str(exc)
                    warning = f"README fetch failed: {type(exc).__name__}: {diagnostic}"
                    result.warnings.append(warning[:500])
                    result.failure_disposition = github_failure_disposition(exc)
                    logger.warning("README fetch failed for %s: %s", repo_id, diagnostic)

            if readme_text or isinstance(readme_text, FetchedReadme):
                source_path = getattr(readme_text, "source_path", None)
                default_branch = (
                    getattr(readme_text, "default_branch", None)
                    or result.raw_repository.get("default_branch")
                )
                base_url = getattr(readme_text, "base_url", None)
                if not base_url and source_path:
                    base_url = build_readme_base_url(
                        owner, name, default_branch, source_path
                    )
                readme = self._process_readme(
                    readme_text,
                    source_path=source_path,
                    default_branch=default_branch,
                    base_url=base_url,
                )
                # Patch the result with the real README data
                result.readme = readme
                result.payload["readme_length"] = readme.readme_length
                result.payload["readme_md"] = readme.readme_md
                result.payload["readme_source_path"] = readme.source_path
                result.payload["readme_default_branch"] = readme.default_branch
                result.payload["readme_base_url"] = readme.base_url
                result.payload["extracted_paragraphs"] = readme.extracted_paragraphs
                result.payload["readme_to_codebase_ratio"] = self._readme_to_codebase_ratio(
                    readme.readme_length, int(result.raw_repository.get("size") or 0)
                )
            enriched.append(result)

        return enriched

    def _enrich_graphql(self, owner: str, name: str, discovery_category: str | None, discovery_band: str | None) -> EnrichmentResult | None:
        data = self.graphql_client.get_repository(owner, name)
        if not data:
            return None
        data = {**data, "_lifecycle_observed_at": datetime.now(timezone.utc).isoformat()}
        return self._process_graphql_data(data, discovery_category, discovery_band)

    def _process_graphql_data(
        self,
        data: dict[str, Any],
        discovery_category: str | None,
        discovery_band: str | None,
        readme_text: str | None = None,
    ) -> EnrichmentResult | None:
        full_name = normalize_repository_name(data.get("nameWithOwner"))
        if not full_name:
            return None

        # Reconstruct topics
        topics = []
        topic_nodes = data.get("repositoryTopics", {}).get("nodes", [])
        for node in topic_nodes:
            if "topic" in node and "name" in node["topic"]:
                topics.append(node["topic"]["name"])

        # Reconstruct languages
        languages = {}
        lang_edges = data.get("languages", {}).get("edges", [])
        for edge in lang_edges:
            size = edge.get("size", 0)
            lang_name = edge.get("node", {}).get("name")
            if lang_name:
                languages[lang_name] = size

        # GitHub's first ten languages may omit bytes. Only totalSize is an
        # authoritative percentage denominator, never the returned edge sum.
        raw_total = data.get("languages", {}).get("totalSize")
        total = raw_total if type(raw_total) is int and 0 <= raw_total <= 2**53 - 1 else None
        breakdown = []
        for name, size in languages.items():
            edge = next(edge for edge in lang_edges if edge.get("node", {}).get("name") == name)
            observed_size = size if type(size) is int and 0 <= size <= 2**53 - 1 else None
            color = edge.get("node", {}).get("color")
            breakdown.append(dict(name=name, color=color if isinstance(color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", color) else None,
                size=observed_size, percentage=100 * observed_size / total
                if total and observed_size is not None and observed_size <= total else None))
        language_details = dict(language_breakdown=breakdown, language_total_size=total,
            language_breakdown_complete=(all(row["size"] is not None for row in breakdown)
                and sum(row["size"] for row in breakdown) == total) if total is not None else None)

        # Primary language
        primary_language = None
        if languages:
            primary_language = max(languages.items(), key=lambda item: item[1])[0]

        default_branch = (data.get("defaultBranchRef") or {}).get("name")

        # README — use provided text, or check inline fields, or fall back to empty.
        # Cleaning is derived from a copy; canonical Markdown remains untouched.
        source_path = getattr(readme_text, "source_path", None)
        readme_default_branch = (
            getattr(readme_text, "default_branch", None) or default_branch
        )
        readme_base_url = getattr(readme_text, "base_url", None)
        if readme_text is None:
            readme_text = ""
            for key, candidate_path in README_CANDIDATES:
                blob = data.get(key)
                if blob and blob.get("text"):
                    readme_text = blob["text"]
                    source_path = candidate_path
                    break
        owner, _, name = full_name.partition("/")
        if not readme_base_url and source_path:
            readme_base_url = build_readme_base_url(
                owner, name, readme_default_branch, source_path
            )
        readme = self._process_readme(
            readme_text or "",
            source_path=source_path,
            default_branch=readme_default_branch,
            base_url=readme_base_url,
        )

        # Star history and events approximation
        stargazers = [{"starred_at": edge.get("starredAt")} for edge in data.get("stargazers", {}).get("edges", [])]
        
        # We approximate events with commits to the default branch
        events = []
        commits = data.get("defaultBranchRef", {}).get("target", {}).get("history", {}).get("nodes", [])
        for commit in commits:
            events.append({
                "type": "PushEvent",
                "created_at": commit.get("committedDate")
            })

        # Contributor/community evidence is not collected by this bounded query.
        # Keep the compatibility list empty; never infer counts from its owner.
        contributors = []

        # Construct raw_repository (REST equivalent structure for downstream compatibility)
        raw_repository = {
            "github_id": str(data.get("databaseId") or ""),
            "github_node_id": data.get("id"),
            "full_name": full_name,
            "name": data.get("name"),
            "description": data.get("description"),
            "html_url": data.get("url"),
            "homepage": data.get("homepageUrl"),
            "created_at": data.get("createdAt"),
            "updated_at": data.get("updatedAt"),
            "pushed_at": data.get("pushedAt"),
            "default_branch": default_branch,
            "size": 0, # Cannot get size from GraphQL repo directly easily without languages sum
            "stargazers_count": data.get("stargazerCount", 0),
            "watchers_count": data.get("watchers", {}).get("totalCount", 0),
            "language": primary_language,
            "forks_count": data.get("forkCount", 0),
            "open_issues_count": data.get("issues", {}).get("totalCount", 0),
            "pull_requests_count": data.get("pullRequests", {}).get("totalCount", 0),
            "owner": {
                "login": data.get("owner", {}).get("login"),
                "github_id": str(data.get("owner", {}).get("databaseId") or ""),
            } if data.get("owner") else None,
            "_discovery_category": discovery_category,
            "_discovery_band": discovery_band,
        }
        
        if isinstance(data.get('isArchived'), bool) and isinstance(data.get('isDisabled'), bool):
            raw_repository['lifecycle'] = {
                'status': 'verified', 'source': 'github-graphql-repository',
                'observed_at': data.get('_lifecycle_observed_at') or datetime.now(timezone.utc).isoformat(),
                'archived': data['isArchived'], 'disabled': data['isDisabled'],
            }

        if 'isFork' in data:
            lineage = lineage_from_graphql(data, observed_at=datetime.now(timezone.utc))
            raw_repository.update({
                'is_fork': lineage.is_fork,
                'fork_parent_github_id': lineage.fork_parent_github_id,
                'fork_family_github_id': lineage.fork_family_github_id,
                'lineage_source': lineage.source,
                'lineage_observed_at': lineage.observed_at,
            })

        # Calculate size from languages sum as fallback
        if languages:
            raw_repository["size"] = sum(languages.values()) // 1024

        recent_commits = [commit.get("committedDate") for commit in commits if commit.get("committedDate")]
        payload = self.to_osiris_payload(
            raw_repository,
            readme=readme,
            topics=topics,
            languages=languages,
            contributors=contributors,
            events=events,
            stargazers=stargazers,
            recent_commits=recent_commits,
        )
        
        payload.update(language_details)
        return EnrichmentResult(
            repo_id=payload["id"],
            payload=payload,
            raw_repository=raw_repository,
            readme=readme,
            topics=topics,
            languages=languages,
        )




    def to_osiris_payload(
        self,
        repository: dict[str, Any],
        *,
        readme: ReadmeDocument,
        topics: list[str],
        languages: dict[str, int],
        contributors: list[dict[str, Any]],
        events: list[dict[str, Any]],
        stargazers: list[dict[str, Any]],
        recent_commits: list[str] | None = None,
    ) -> dict[str, Any]:
        full_name = repository.get("full_name") or repository.get("name") or "unknown/repository"
        size_kb = int(repository.get("size") or 0)
        primary_language = repository.get("language") or self._primary_language(languages)
        pushed_days_ago = self._days_since(repository.get("pushed_at"))
        deltas = self._estimate_star_deltas(repository, stargazers=stargazers, events=events)

        primary_lang_str = primary_language or "Unknown"
        special_label = None
        if primary_lang_str == "Unknown":
            special_label = self._classify_unknown_repo(repository.get("description"), topics)

        return {
            "id": full_name,
            "full_name": full_name,
            "github_id": repository.get("github_id"),
            "github_node_id": repository.get("github_node_id"),
            "owner_github_id": (repository.get("owner") or {}).get("github_id"),
            **{field: repository[field] for field in ('is_fork', 'fork_parent_github_id',
                'fork_family_github_id', 'lineage_source', 'lineage_observed_at') if field in repository},
            "star_count": int(repository.get("stargazers_count") or repository.get("watchers_count") or 0),
            "primary_language": primary_lang_str,
            "special_label": special_label,
            "readme_length": readme.readme_length,
            "readme_md": readme.readme_md,
            "readme_source_path": readme.source_path,
            "readme_default_branch": readme.default_branch,
            "readme_base_url": readme.base_url,
            "readme_to_codebase_ratio": self._readme_to_codebase_ratio(readme.readme_length, size_kb),
            "delta_3d": deltas[3],
            "delta_7d": deltas[7],
            "delta_30d": deltas[30],
            "extracted_paragraphs": readme.extracted_paragraphs,
            "pushed_days_ago": pushed_days_ago,
            "topics": topics,
            "languages": list(languages.keys()),
            "fork_count": int(repository.get("forks_count") or 0),
            "open_issues_count": int(repository.get("open_issues_count") or 0),
            "description": repository.get("description") or "",
            "html_url": repository.get("html_url"),
            "created_at": repository.get("created_at"),
            "updated_at": repository.get("updated_at"),
            "pushed_at": repository.get("pushed_at"),
            **({"lifecycle": repository["lifecycle"]} if "lifecycle" in repository else {}),
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "discovery_category": repository.get("_discovery_category"),
            "discovery_band": repository.get("_discovery_band"),
            "recent_commits": recent_commits or [],
        }


    @staticmethod
    def _classify_unknown_repo(description: str | None, topics: list[str]) -> str:
        text_to_search = f"{description or ''} {' '.join(topics or [])}".lower()
        
        heuristics = [
            ("list", r"\b(list|awesome|collection|resources|books)\b"),
            ("roadmap", r"\b(roadmap|path|curriculum|syllabus|career|interview)\b"),
            ("tutorial", r"\b(tutorial|build-your-own|step-by-step)\b"),
            ("explainer", r"\b(explain|how-to|notes|learn|course|guide|concepts|primer)\b"),
            ("cheatsheet", r"\b(cheatsheet|cheat-sheet|reference|quick-look)\b"),
            ("dataset", r"\b(dataset|data|corpus|apis)\b"),
            ("template", r"\b(template|boilerplate|starter|scaffold|scaffolding)\b"),
            ("spec", r"\b(rfc|spec|specification|standard|protocol)\b"),
            ("showcase", r"\b(showcase|portfolio|gallery|examples|demos)\b"),
        ]
        
        for label, pattern in heuristics:
            if re.search(pattern, text_to_search):
                return label
                
        return "other"

    @staticmethod
    def _primary_language(languages: dict[str, int]) -> str | None:
        if not languages:
            return None
        return max(languages.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _readme_to_codebase_ratio(readme_length: int, size_kb: int) -> float:
        codebase_bytes = max(size_kb * 1024, 1)
        return round(readme_length / codebase_bytes, 8)

    def _estimate_star_deltas(
        self,
        repository: dict[str, Any],
        *,
        stargazers: list[dict[str, Any]],
        events: list[dict[str, Any]],
    ) -> dict[int, int]:
        windows = {3: 0, 7: 0, 30: 0}
        now = datetime.now(timezone.utc)
        timestamps = [self._parse_datetime(item.get("starred_at")) for item in stargazers if item.get("starred_at")]
        timestamps = sorted([value for value in timestamps if value])
        total_stars = int(repository.get("stargazers_count") or repository.get("watchers_count") or 0)

        if timestamps:
            if len(timestamps) >= total_stars or len(timestamps) < 100:
                for days in windows:
                    windows[days] = sum(1 for value in timestamps if (now - value).days <= days)
                return windows

            n = len(timestamps)
            if n >= 2:
                oldest_ts = timestamps[0]
                newest_ts = timestamps[-1]
                span_seconds = (newest_ts - oldest_ts).total_seconds()
                span_days = span_seconds / 86400.0
                if span_days < 0.1:
                    span_days = 0.1

                recent_rate = n / span_days
                oldest_days_ago = (now - oldest_ts).total_seconds() / 86400.0

                for W in windows:
                    if W <= oldest_days_ago:
                        windows[W] = sum(1 for ts in timestamps if (now - ts).total_seconds() / 86400.0 <= W)
                    else:
                        observed = n
                        t_diff = W - oldest_days_ago
                        decay_constant = 0.05
                        extrapolated = recent_rate * (1.0 - math.exp(-decay_constant * t_diff)) / decay_constant
                        windows[W] = min(int(round(observed + extrapolated)), total_stars)
                return windows

        push_events = [event for event in events if event.get("type") in {"PushEvent", "CreateEvent", "PullRequestEvent", "IssuesEvent"}]
        pushed_days_ago = self._days_since(repository.get("pushed_at"))
        stars = total_stars
        activity_multiplier = min(len(push_events) / 30.0, 1.0)
        recency_multiplier = 1.0 if pushed_days_ago <= 3 else 0.6 if pushed_days_ago <= 7 else 0.25 if pushed_days_ago <= 30 else 0.05
        baseline_monthly = max(int((stars ** 0.5) * activity_multiplier * recency_multiplier), 0)
        windows[30] = baseline_monthly
        windows[7] = min(windows[30], max(int(baseline_monthly * 0.35), 0))
        windows[3] = min(windows[7], max(int(baseline_monthly * 0.18), 0))
        return windows

    @staticmethod
    def _days_since(value: str | None) -> int:
        parsed = RepositoryEnricher._parse_datetime(value)
        if not parsed:
            return 999
        return max((datetime.now(timezone.utc) - parsed).days, 0)

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        normalized = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
