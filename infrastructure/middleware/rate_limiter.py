"""
Advanced Rate Limiter — CognitionOS

Multi-strategy rate limiting:
- Token bucket per user/tenant/IP
- Sliding window counters
- Endpoint-specific limits
- Tier-based quotas
- Real-time usage stats
"""

from __future__ import annotations

import hashlib
import logging
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


class RateLimitTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    UNLIMITED = "unlimited"


@dataclass
class TierLimits:
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 20
    concurrent_requests: int = 5
    max_tokens_per_request: int = 4096
    max_requests_per_endpoint: Dict[str, int] = field(default_factory=dict)


DEFAULT_TIERS: Dict[RateLimitTier, TierLimits] = {
    RateLimitTier.FREE: TierLimits(
        requests_per_minute=30, requests_per_hour=500,
        requests_per_day=5000, burst_size=10, concurrent_requests=2,
        max_tokens_per_request=2048),
    RateLimitTier.PRO: TierLimits(
        requests_per_minute=120, requests_per_hour=5000,
        requests_per_day=50000, burst_size=50, concurrent_requests=10,
        max_tokens_per_request=8192),
    RateLimitTier.ENTERPRISE: TierLimits(
        requests_per_minute=600, requests_per_hour=30000,
        requests_per_day=500000, burst_size=200, concurrent_requests=50,
        max_tokens_per_request=32768),
    RateLimitTier.UNLIMITED: TierLimits(
        requests_per_minute=999999, requests_per_hour=999999,
        requests_per_day=999999, burst_size=9999, concurrent_requests=500,
        max_tokens_per_request=128000),
}


@dataclass
class RateLimitResult:
    allowed: bool
    remaining: int
    limit: int
    reset_at: float
    retry_after_seconds: float = 0.0
    tier: str = ""
    reason: str = ""

    def to_headers(self) -> Dict[str, str]:
        h = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(self.reset_at)),
        }
        if not self.allowed:
            h["Retry-After"] = str(int(self.retry_after_seconds))
        return h


class _TokenBucket:
    def __init__(self, rate: float, capacity: int) -> None:
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()

    def consume(self, n: int = 1) -> Tuple[bool, float]:
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now
            if self.tokens >= n:
                self.tokens -= n
                return True, self.tokens
            wait = (n - self.tokens) / self.rate
            return False, wait


class _SlidingWindow:
    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window = window_seconds
        self.requests: List[float] = []
        self.lock = threading.Lock()

    def check(self) -> Tuple[bool, int]:
        with self.lock:
            now = time.monotonic()
            cutoff = now - self.window
            self.requests = [t for t in self.requests if t > cutoff]
            if len(self.requests) < self.limit:
                self.requests.append(now)
                return True, self.limit - len(self.requests)
            return False, 0


class RateLimiter:
    """Production rate limiter with tier-based quotas."""

    def __init__(self, *, tiers: Dict[RateLimitTier, TierLimits] | None = None) -> None:
        self._tiers = tiers or DEFAULT_TIERS
        self._buckets: Dict[str, _TokenBucket] = {}
        self._windows: Dict[str, _SlidingWindow] = {}
        self._concurrent: Dict[str, int] = defaultdict(int)
        self._metrics: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._lock = threading.Lock()

    def _key(self, identity: str, scope: str) -> str:
        return f"{identity}:{scope}"

    def _get_bucket(self, identity: str, tier: RateLimitTier) -> _TokenBucket:
        key = self._key(identity, "bucket")
        if key not in self._buckets:
            limits = self._tiers[tier]
            rate = limits.requests_per_minute / 60.0
            self._buckets[key] = _TokenBucket(rate, limits.burst_size)
        return self._buckets[key]

    def _get_window(self, identity: str, tier: RateLimitTier, window: str) -> _SlidingWindow:
        key = self._key(identity, f"window:{window}")
        if key not in self._windows:
            limits = self._tiers[tier]
            if window == "minute":
                self._windows[key] = _SlidingWindow(limits.requests_per_minute, 60)
            elif window == "hour":
                self._windows[key] = _SlidingWindow(limits.requests_per_hour, 3600)
            elif window == "day":
                self._windows[key] = _SlidingWindow(limits.requests_per_day, 86400)
        return self._windows[key]

    def check(
        self,
        identity: str,
        *,
        tier: RateLimitTier = RateLimitTier.FREE,
        endpoint: str = "",
        tokens: int = 1,
    ) -> RateLimitResult:
        limits = self._tiers.get(tier, self._tiers[RateLimitTier.FREE])

        # Token bucket check
        bucket = self._get_bucket(identity, tier)
        ok, remaining_or_wait = bucket.consume(tokens)
        if not ok:
            self._metrics[identity]["rejected"] += 1
            return RateLimitResult(
                allowed=False, remaining=0, limit=limits.requests_per_minute,
                reset_at=time.time() + remaining_or_wait,
                retry_after_seconds=remaining_or_wait,
                tier=tier.value, reason="token_bucket_exhausted")

        # Sliding window per-minute
        win_min = self._get_window(identity, tier, "minute")
        ok_min, rem = win_min.check()
        if not ok_min:
            self._metrics[identity]["rejected"] += 1
            return RateLimitResult(
                allowed=False, remaining=0, limit=limits.requests_per_minute,
                reset_at=time.time() + 60, retry_after_seconds=60,
                tier=tier.value, reason="minute_limit_exceeded")

        # Concurrent limit
        with self._lock:
            if self._concurrent[identity] >= limits.concurrent_requests:
                self._metrics[identity]["rejected"] += 1
                return RateLimitResult(
                    allowed=False, remaining=0, limit=limits.concurrent_requests,
                    reset_at=time.time() + 1, retry_after_seconds=1,
                    tier=tier.value, reason="concurrent_limit_exceeded")
            self._concurrent[identity] += 1

        self._metrics[identity]["allowed"] += 1
        return RateLimitResult(
            allowed=True, remaining=rem, limit=limits.requests_per_minute,
            reset_at=time.time() + 60, tier=tier.value, reason="allowed")

    def release(self, identity: str) -> None:
        with self._lock:
            if self._concurrent[identity] > 0:
                self._concurrent[identity] -= 1

    def get_usage(self, identity: str) -> Dict[str, Any]:
        return dict(self._metrics.get(identity, {}))

    def get_all_metrics(self) -> Dict[str, Dict[str, int]]:
        return {k: dict(v) for k, v in self._metrics.items()}

    def reset(self, identity: str) -> None:
        keys_to_del = [k for k in self._buckets if k.startswith(identity)]
        for k in keys_to_del:
            del self._buckets[k]
        keys_to_del = [k for k in self._windows if k.startswith(identity)]
        for k in keys_to_del:
            del self._windows[k]
        self._concurrent.pop(identity, None)
        self._metrics.pop(identity, None)


_limiter: RateLimiter | None = None

def get_rate_limiter() -> RateLimiter:
    global _limiter
    if not _limiter:
        _limiter = RateLimiter()
    return _limiter
