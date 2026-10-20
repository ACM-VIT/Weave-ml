"""Transform GitHub repository data into Osiris-compatible payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import re
from typing import Any
import logging

from .github_graphql_client import (
    FetchedReadme,
    GitHubGraphQLClient,
    build_readme_base_url,
)
from .graphql_queries import README_CANDIDATES
from utils.readme_processor import ReadmeDocument, process_markdown
from .identity import normalize_repository_name, repository_identity_key

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


class RepositoryEnricher:
    def __init__(self, graphql_client: GitHubGraphQLClient | None = None) -> None:
        self.graphql_client = graphql_client or GitHubGraphQLClient()

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
            except Exception as e:
                logger.warning(f"GraphQL enrichment failed for {full_name}: {e}.")

        return None

    def get_repositories_batch(self, repositories: list[dict[str, Any] | str]) -> list[EnrichmentResult]:
        """Fetch multiple repositories using GraphQL batching, falling back to sequential REST if needed."""
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
        # Pass 1: batch metadata (no readme — avoids 502 on large repos)
        for i in range(0, len(targets), 10):
            batch = targets[i:i+10]
            batch_res = {}
            try:
                batch_res = self.graphql_client.get_repositories_batch(batch)
            except Exception as e:
                logger.warning(f"Batch {i//10 + 1} failed, retrying one-by-one: {e}")
                # Retry the batch one repo at a time
                for owner, name in batch:
                    try:
                        data = self.graphql_client.get_repository(owner, name)
                        if data:
                            batch_res[f"{owner}/{name}"] = data
                    except Exception as e2:
                        logger.warning(f"Individual fetch failed for {owner}/{name}: {e2}")

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
        enriched = []
        for result in results:
            repo_id = result.repo_id
            owner, name = owner_name_map.get(repo_id, (None, None))
            readme_text: str | FetchedReadme = ""
            if owner and name:
                try:
                    readme_text = self.graphql_client.get_readme(owner, name)
                except Exception as exc:
                    warning = f"README fetch failed: {type(exc).__name__}: {exc}"
                    result.warnings.append(warning[:500])
                    logger.warning("README fetch failed for %s: %s", repo_id, exc)

            if readme_text:
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
                readme = process_markdown(
                    str(readme_text),
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
        readme = process_markdown(
            str(readme_text or ""),
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

        # Contributor data is not required downstream; returning an empty list as audited.
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
            "star_count": int(repository.get("stargazers_count") or repository.get("watchers_count") or 0),
            "primary_language": primary_lang_str,
            "special_label": special_label,
            "readme_length": readme.readme_length,
            "readme_md": readme.readme_md,
            "readme_source_path": readme.source_path,
            "readme_default_branch": readme.default_branch,
            "readme_base_url": readme.base_url,
            "readme_to_codebase_ratio": self._readme_to_codebase_ratio(readme.readme_length, size_kb),
            "mentionable_users_count": self._mentionable_users_count(contributors, repository),
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

    @staticmethod
    def _mentionable_users_count(contributors: list[dict[str, Any]], repository: dict[str, Any]) -> int:
        if contributors:
            return min(len(contributors), 100)
        return 1 if repository.get("owner") else 0

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
