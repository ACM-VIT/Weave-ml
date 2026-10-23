import math
import random
import logging
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

from inference.diversity import DiversityCandidate, DiversityPolicy, DiversityResult, rerank_diverse
from utils.deadline import check_deadline

logger = logging.getLogger(__name__)


def _utc_hour(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


class DiversityUnavailableError(RuntimeError):
    """Configured diversity cannot serve a fully qualified selection."""


class FeedAssemblySystem:
    def __init__(
        self,
        *,
        freshness_window_hours=48.0,
        freshness_boost=0.25,
        explore_fraction=1 / 3,
        max_same_language=5,
        generation_reference_capacity=65_536,
        generation_reference_ttl_seconds=6 * 60 * 60,
    ):
        self.freshness_window_hours = freshness_window_hours
        self.freshness_boost = freshness_boost
        self.explore_fraction = explore_fraction
        self.max_same_language = max_same_language
        self.generation_reference_capacity = int(generation_reference_capacity)
        self.generation_reference_ttl_seconds = float(
            generation_reference_ttl_seconds
        )

        if not 0.0 <= float(self.explore_fraction) <= 0.5:
            raise ValueError("explore_fraction must be between 0.0 and 0.5")
        if int(self.max_same_language) < 1:
            raise ValueError("max_same_language must be at least 1")
        if self.generation_reference_capacity < 1:
            raise ValueError("generation_reference_capacity must be at least 1")
        if self.generation_reference_ttl_seconds <= 0:
            raise ValueError("generation_reference_ttl_seconds must be positive")

        # A generation retry must not change merely because wall time crossed
        # an hour while the request was in flight.  This process-local FIFO is
        # deliberately bounded by both capacity and time: it avoids a Redis
        # round trip on the recommendation hot path while keeping retry anchors
        # long enough for normal backend retry windows.
        self._generation_references: OrderedDict[
            str, tuple[float, datetime]
        ] = OrderedDict()
        self._generation_reference_lock = threading.Lock()

    def _freshness_reference(
        self,
        *,
        generation_id: str | None,
        observed_at: datetime,
    ) -> datetime:
        observed_hour = _utc_hour(observed_at)
        if generation_id is None:
            return observed_hour
        generation_key = str(generation_id).strip()
        if not generation_key:
            return observed_hour

        now = time.monotonic()
        with self._generation_reference_lock:
            while self._generation_references:
                oldest_key, (expires_at, _) = next(
                    iter(self._generation_references.items())
                )
                if expires_at > now:
                    break
                self._generation_references.pop(oldest_key, None)

            cached = self._generation_references.get(generation_key)
            if cached is not None and cached[0] > now:
                return cached[1]
            self._generation_references.pop(generation_key, None)
            self._generation_references[generation_key] = (
                now + self.generation_reference_ttl_seconds,
                observed_hour,
            )
            while (
                len(self._generation_references)
                > self.generation_reference_capacity
            ):
                self._generation_references.popitem(last=False)
        return observed_hour

    def shape_batch(
        self,
        ranked: list[dict],
        *,
        seen_repo_ids: set[str] | None = None,
        randomizer: random.Random | None = None,
        reference_time: datetime | None = None,
        generation_id: str | None = None,
        target_size: int | None = None,
        input_is_unique: bool = False,
        diversity_policy: DiversityPolicy | None = None,
        diversity_vectors: dict[str, list[float] | None] | None = None,
        diversity_seed: str = 'default',
        diversity_report: Callable[[DiversityResult], None] | None = None,
    ) -> list[dict]:
        """Shape a ranked pool with in-page exploration.

        ``target_size`` is applied *before* exploration is selected.  This is
        important for the serving path: shuffling the bottom third of a
        150-item pool and subsequently returning its first 15 items does not
        explore anything.  The exploitation prefix remains stable while the
        returned tail is sampled deterministically from the lower-ranked pool.
        """
        if target_size is not None and target_size <= 0:
            return []

        seen_repo_ids = seen_repo_ids or set()
        if input_is_unique:
            # The retriever constructs candidates in a repo-id keyed mapping,
            # so its hot path can avoid paying for redundant deduplication.
            shaped = [
                dict(item)
                for item in ranked
                if item.get("repo_id") not in seen_repo_ids
            ]
        else:
            shaped = []
            included_repo_ids = set(seen_repo_ids)
            for item in ranked:
                raw_repo_id = item.get("repo_id")
                if raw_repo_id is None:
                    # Contract validation is owned by the caller. Keeping the
                    # malformed item here lets that boundary return its useful
                    # error instead of silently dropping it.
                    shaped.append(dict(item))
                    continue
                repo_id = str(raw_repo_id)
                if repo_id in included_repo_ids:
                    continue
                included_repo_ids.add(repo_id)
                copied = dict(item)
                copied["repo_id"] = repo_id
                shaped.append(copied)

        # New generations use the current UTC hour so freshness keeps advancing.
        # Retries reuse the generation's bounded first-seen anchor, including
        # when a retry crosses into the next UTC hour.
        current_time = self._freshness_reference(
            generation_id=generation_id,
            observed_at=reference_time or datetime.now(timezone.utc),
        )
        for item in shaped:
            base_score = item.get("final_score")
            if base_score is None:
                base_score = item.get("score")
            if base_score is None:
                base_score = 0.5
            item["final_score"] = base_score

            raw_created_at = item.get("created_at")
            if not raw_created_at:
                continue

            try:
                if isinstance(raw_created_at, str):
                    created_date = datetime.fromisoformat(
                        raw_created_at.replace("Z", "+00:00")
                    )
                elif isinstance(raw_created_at, datetime):
                    created_date = raw_created_at
                else:
                    continue

                created_date = _utc_hour(created_date)

                if created_date <= current_time:
                    age_hours = max(
                        0.0,
                        (current_time - created_date).total_seconds() / 3600.0,
                    )
                    if age_hours < self.freshness_window_hours:
                        boost = self.freshness_boost * (
                            1.1 / (1.0 + math.log1p(age_hours))
                        )
                        item["final_score"] += boost
            except Exception as exc:
                logger.error(
                    "Freshness parsing failed for repo %s: %s",
                    item.get("repo_id"),
                    exc,
                )

        shaped.sort(
            key=lambda item: (
                item.get("final_score")
                if item.get("final_score") is not None
                else 0.5
            ),
            reverse=True,
        )

        if diversity_policy is not None:
            vectors = diversity_vectors or {}
            candidates = []
            for item in shaped:
                vector = vectors.get(item['repo_id'])
                if vector is None:
                    raise DiversityUnavailableError('Diversity requires fresh compatible repository vectors')
                raw_topics = item.get('topics') or []
                if (not isinstance(raw_topics, (list, tuple)) or len(raw_topics) > 200
                        or any(not isinstance(topic, str) or not 1 <= len(topic) <= 128 for topic in raw_topics)
                        or (item.get('category') is not None and not isinstance(item['category'], str))):
                    raise DiversityUnavailableError('Diversity requires bounded topic/category metadata')
                candidates.append(DiversityCandidate(
                    repo_id=item['repo_id'], score=float(item['final_score']), vector=tuple(vector),
                    owner_github_id=item.get('owner_github_id'), fork_family_github_id=item.get('fork_family_github_id'),
                    topics=tuple(sorted(set(topic.strip().casefold() for topic in raw_topics))),
                    category=str(item['category']).strip().casefold() if item.get('category') else None,
                    bucket=item.get('candidate_bucket', 'personalized'),
                ))
            desired = len(shaped) if target_size is None else int(target_size)
            personalized = frozenset(item['repo_id'] for item in shaped
                if item.get('candidate_bucket', 'personalized') == 'personalized')
            result = rerank_diverse(candidates, target_size=desired, seed=diversity_seed,
                policy=diversity_policy, recent_repo_ids=frozenset(seen_repo_ids),
                personalized_top_ids=personalized, check_work=check_deadline)
            if diversity_report is not None:
                diversity_report(result)
            if result.unknown_identity or result.exploration_shortfall or (desired and not result.candidates):
                raise DiversityUnavailableError('Diversity metadata or required exploration is unavailable')
            by_id = {item['repo_id']: item for item in shaped}
            return [by_id[candidate.repo_id] for candidate in result.candidates]

        language_counts: dict[str, int] = {}
        diverse: list[dict] = []
        overflow: list[dict] = []
        for item in shaped:
            language = item.get("primary_language")
            if not language:
                diverse.append(item)
                continue

            language_key = str(language).casefold()
            count = language_counts.get(language_key, 0)
            if count >= self.max_same_language:
                overflow.append(item)
                continue

            language_counts[language_key] = count + 1
            diverse.append(item)

        ordered = diverse + overflow
        target_count = min(
            len(ordered),
            len(ordered) if target_size is None else int(target_size),
        )
        if target_count < 3 or self.explore_fraction <= 0:
            return ordered[:target_count]

        explore_count = max(1, int(target_count * self.explore_fraction))
        # Always retain a non-empty, stable exploitation tier.
        explore_count = min(explore_count, target_count - 1)
        exploit_count = target_count - explore_count
        exploitation_tier = ordered[:exploit_count]

        exploration_pool = ordered[exploit_count:]
        random_source = randomizer or random.Random(0)
        pool_size = len(exploration_pool)
        start = random_source.randrange(pool_size)
        step = 1
        if pool_size > 1:
            step = random_source.randrange(1, pool_size)
            while math.gcd(step, pool_size) != 1:
                step = 1 if step + 1 == pool_size else step + 1

        # Preserve the language cap when enough alternatives exist, then use
        # deferred candidates only to honour the exact response-size contract.
        language_counts: dict[str, int] = {}
        for item in exploitation_tier:
            language = item.get("primary_language")
            if language:
                key = str(language).casefold()
                language_counts[key] = language_counts.get(key, 0) + 1

        exploration_tier: list[dict] = []
        deferred: list[dict] = []
        for offset in range(pool_size):
            item = exploration_pool[(start + offset * step) % pool_size]
            language = item.get("primary_language")
            key = str(language).casefold() if language else None
            if key and language_counts.get(key, 0) >= self.max_same_language:
                deferred.append(item)
                continue
            exploration_tier.append(item)
            if key:
                language_counts[key] = language_counts.get(key, 0) + 1
            if len(exploration_tier) == explore_count:
                break

        if len(exploration_tier) < explore_count:
            selected_ids = {
                str(item.get("repo_id")) for item in exploration_tier
            }
            exploration_tier.extend(
                item
                for item in deferred
                if str(item.get("repo_id")) not in selected_ids
            )
            exploration_tier = exploration_tier[:explore_count]

        return exploitation_tier + exploration_tier

    @staticmethod
    def process_feed_assembly(candidates: List[Dict[str, Any]], target_size: int = 15) -> List[str]:
        """Compatibility wrapper returning only repository IDs."""
        if target_size <= 0:
            return []

        stable_seed = "\x1f".join(
            str(item.get("repo_id") or "") for item in candidates
        )
        final_pool = FeedAssemblySystem().shape_batch(
            candidates,
            target_size=target_size,
            randomizer=random.Random(stable_seed),
        )

        # Strip internal temporary scores and return clean ordered string IDs
        ordered_ids: list[str] = []
        for item in final_pool:
            repo_id = item.get("repo_id")
            if not repo_id:
                raise ValueError("Feed candidate missing required repo_id")
            ordered_ids.append(str(repo_id))

        return ordered_ids
