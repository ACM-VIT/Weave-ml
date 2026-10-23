"""Non-waiting provider admission, shared quotas and fenced recovery probes."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import uuid
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Any

from redis.exceptions import RedisError

from .settings import SummarySettings


class AdmissionDenied(RuntimeError):
    def __init__(self, reason: str, retry_after: float = 30) -> None:
        super().__init__(reason)
        self.retry_after = retry_after


@dataclass(frozen=True)
class Permit:
    epoch: int
    probe: str = ""


# Redis time owns both the minute window and cooldown/probe leases. Circuit
# epochs fence late outcomes from requests admitted before an open/recovery.
ADMIT_LUA = """
local clock=redis.call('TIME'); local now=tonumber(clock[1])*1000+math.floor(tonumber(clock[2])/1000)
local opened=tonumber(redis.call('HGET',KEYS[1],'until') or '0')
local epoch=tonumber(redis.call('HGET',KEYS[1],'epoch') or '0')
if opened>now then return {0,1,opened-now} end
if opened>0 and tonumber(redis.call('HGET',KEYS[1],'probe_until') or '0')>now then return {0,1,1000} end
local window=math.max(60000,math.ceil(60000/tonumber(ARGV[1])))
local minute=math.floor(now/window)
local sig=ARGV[1]..':'..ARGV[2]
local previous=redis.call('HGET',KEYS[2],'config')
if previous and previous~=sig then return {0,3,30000} end
local count=0; local tokens=0
local prior_minute=tonumber(redis.call('HGET',KEYS[2],'minute') or '-1')
if prior_minute>=minute then
 minute=prior_minute
 count=tonumber(redis.call('HGET',KEYS[2],'requests') or '0')
 tokens=tonumber(redis.call('HGET',KEYS[2],'tokens') or '0')
end
if count+1>math.max(1,math.floor(tonumber(ARGV[1]))) or tokens+tonumber(ARGV[3])>tonumber(ARGV[2]) then return {0,2,window-now%window} end
redis.call('HSET',KEYS[2],'minute',minute,'requests',count+1,'tokens',tokens+tonumber(ARGV[3]),'config',sig)
redis.call('PEXPIRE',KEYS[2],window*2+120000)
redis.call('HSET',KEYS[1],'epoch',epoch)
redis.call('PEXPIRE',KEYS[1],tonumber(ARGV[6]))
local probe=''
if opened>0 then
 probe=ARGV[4];redis.call('HSET',KEYS[1],'probe',probe,'probe_until',now+tonumber(ARGV[5]))
end
return {1,epoch,probe}
"""
REPORT_LUA = """
local epoch=tonumber(redis.call('HGET',KEYS[1],'epoch') or '-1')
if epoch~=tonumber(ARGV[1]) then return 0 end
local probe=redis.call('HGET',KEYS[1],'probe') or ''
if probe~=ARGV[2] then return 0 end
local clock=redis.call('TIME');local now=tonumber(clock[1])*1000+math.floor(tonumber(clock[2])/1000)
if probe~='' and tonumber(redis.call('HGET',KEYS[1],'probe_until') or '0')<=now then return 0 end
if ARGV[3]=='2' then
 if probe~='' then redis.call('HSET',KEYS[1],'probe','','probe_until',0,'epoch',epoch+1) end
elseif ARGV[3]=='1' then
 redis.call('HSET',KEYS[1],'failures',0)
 if probe~='' then redis.call('HSET',KEYS[1],'until',0,'probe','','probe_until',0,'epoch',epoch+1) end
else
 local failures=redis.call('HINCRBY',KEYS[1],'failures',1)
 if probe~='' or failures>=tonumber(ARGV[4]) then
  redis.call('HSET',KEYS[1],'until',now+tonumber(ARGV[5]),'epoch',epoch+1,'probe','','probe_until',0)
 end
end
redis.call('PEXPIRE',KEYS[1],tonumber(ARGV[6]));return 1
"""


class LocalSummaryAdmission:
    """Instance-local policy for injected/offline providers; never a fleet cap."""

    def __init__(self, settings: SummarySettings, *, clock=time.monotonic) -> None:
        self.settings = settings
        self.clock = clock
        self._lock = threading.Lock()
        self._epoch = 0
        self._failures = 0
        self._until = 0.0
        self._probe = ""
        self._probe_until = 0.0
        self._minute = -1
        self._requests = self._tokens = 0
        self._deferred: OrderedDict[str, float] = OrderedDict()
        self.metrics: Counter[str] = Counter()

    def acquire(self, tokens: int) -> Permit:
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("summary token reservation must be a positive integer")
        with self._lock:
            now = self.clock()
            if self._until > now or (self._until and self._probe_until > now):
                self.metrics["circuit_rejected"] += 1
                raise AdmissionDenied(
                    "summary provider circuit is cooling down",
                    max(1, self._until - now),
                )
            window = max(60, math.ceil(60 / self.settings.rpm_limit))
            minute = int(now // window)
            if minute > self._minute:
                self._minute = minute
                self._requests = self._tokens = 0
            if (
                self._requests + 1 > max(1, math.floor(self.settings.rpm_limit))
                or self._tokens + tokens > self.settings.tpm_limit
            ):
                self.metrics["quota_rejected"] += 1
                raise AdmissionDenied(
                    "summary provider quota is unavailable", window - now % window
                )
            self._requests += 1
            self._tokens += tokens
            if self._until:
                self._probe = uuid.uuid4().hex
                self._probe_until = now + self.settings.request_timeout_seconds + 1
                self.metrics["probes"] += 1
            self.metrics["admitted"] += 1
            return Permit(self._epoch, self._probe)

    def report(self, permit: Permit, available: bool | None) -> None:
        with self._lock:
            if permit.epoch != self._epoch or permit.probe != self._probe:
                return
            if permit.probe and self.clock() >= self._probe_until:
                return
            if available is None:
                self.metrics["abandoned"] += 1
                if permit.probe:
                    self._epoch += 1
                    self._probe = ""
                    self._probe_until = 0
                return
            self.metrics[
                "abandoned"
                if available is None
                else "available"
                if available
                else "failed"
            ] += 1
            self._failures = 0 if available else self._failures + 1
            if not available and (
                permit.probe or self._failures >= self.settings.failure_threshold
            ):
                self._until = self.clock() + self.settings.cooldown_seconds
                self._epoch += 1
                self._probe = ""
                self._probe_until = 0
            elif available and permit.probe:
                self._until = 0
                self._epoch += 1
                self._probe = ""
                self._probe_until = 0

    def defer(self, key: str) -> None:
        with self._lock:
            self._deferred[key] = self.clock() + self.settings.cooldown_seconds
            self._deferred.move_to_end(key)
            while len(self._deferred) > 1024:
                self._deferred.popitem(last=False)

    def deferred(self, key: str) -> bool:
        with self._lock:
            until = self._deferred.get(key, 0)
            if until <= self.clock():
                self._deferred.pop(key, None)
                return False
            self.metrics["fallback_deferred"] += 1
            return True


class RedisSummaryAdmission:
    """Atomic process-shared request/token budgets and provider circuit state."""

    def __init__(
        self, settings: SummarySettings, client: Any, *, owns_client: bool = False
    ) -> None:
        self.settings, self.client = settings, client
        self.owns_client = owns_client
        account = hashlib.sha256(
            f"{settings.api_url}\0{settings.api_key}".encode()
        ).hexdigest()
        model = hashlib.sha256(settings.model_id.encode()).hexdigest()
        self.prefix = f"weave:summary:{{{account}}}"
        self.state = f"{self.prefix}:circuit:{model}"
        self.quota = f"{self.prefix}:quota"
        self.metrics: Counter[str] = Counter()

    def close(self) -> None:
        if self.owns_client and self.client is not None:
            self.client.close()

    @property
    def ttl_ms(self) -> int:
        return int(
            (
                2 * self.settings.cooldown_seconds
                + self.settings.request_timeout_seconds
                + 120
            )
            * 1000
        )

    def acquire(self, tokens: int) -> Permit:
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("summary token reservation must be a positive integer")
        if self.client is None:
            self.metrics["coordination_unavailable"] += 1
            raise AdmissionDenied("summary quota coordination is unavailable")
        try:
            result = self.client.eval(
                ADMIT_LUA,
                2,
                self.state,
                self.quota,
                float(self.settings.rpm_limit),
                self.settings.tpm_limit,
                tokens,
                uuid.uuid4().hex,
                int((self.settings.request_timeout_seconds + 1) * 1000),
                self.ttl_ms,
            )
        except (RedisError, OSError) as exc:
            self.metrics["coordination_unavailable"] += 1
            raise AdmissionDenied("summary quota coordination is unavailable") from exc
        if int(result[0]) != 1:
            reason = {1: "circuit", 2: "quota", 3: "quota configuration"}[
                int(result[1])
            ]
            self.metrics[
                "circuit_rejected" if int(result[1]) == 1 else "quota_rejected"
            ] += 1
            raise AdmissionDenied(
                f"summary provider {reason} is unavailable", int(result[2]) / 1000
            )
        probe = result[2].decode() if isinstance(result[2], bytes) else str(result[2])
        self.metrics["admitted"] += 1
        if probe:
            self.metrics["probes"] += 1
        return Permit(int(result[1]), probe)

    def report(self, permit: Permit, available: bool | None) -> None:
        self.metrics[
            "abandoned" if available is None else "available" if available else "failed"
        ] += 1
        try:
            self.client.eval(
                REPORT_LUA,
                1,
                self.state,
                permit.epoch,
                permit.probe,
                (2 if available is None else int(available)),
                self.settings.failure_threshold,
                int(self.settings.cooldown_seconds * 1000),
                self.ttl_ms,
            )
        except (RedisError, OSError):
            self.metrics["coordination_unavailable"] += 1

    def defer(self, key: str) -> None:
        if self.client is None:
            return
        try:
            self.client.set(
                f"{self.prefix}:fallback:{key}",
                1,
                px=int(self.settings.cooldown_seconds * 1000),
            )
        except (RedisError, OSError):
            self.metrics["coordination_unavailable"] += 1

    def deferred(self, key: str) -> bool:
        if self.client is None:
            return True
        try:
            result = bool(self.client.exists(f"{self.prefix}:fallback:{key}"))
            if result:
                self.metrics["fallback_deferred"] += 1
            return result
        except (RedisError, OSError):
            return True  # Keep an accepted durable fallback during Redis failure.


def fallback_key(repo: Any, settings: SummarySettings) -> str | None:
    """Canonical content identity; no raw README or provider credential retained."""
    if not repo.get("repo_id") or not repo.get("content_hash"):
        return None
    return hashlib.sha256(
        json.dumps(
            [
                repo["repo_id"],
                repo["content_hash"],
                settings.model_id,
                settings.prompt_version,
                settings.format_version,
            ],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
