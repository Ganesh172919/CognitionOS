"""
Advanced Rate Limiter

Multi-algorithm rate limiting with per-tenant quotas:
- Token bucket (burst-friendly)
- Sliding window counter (precise)
- Fixed window (simple, cheap)
- Leaky bucket (smooth output rate)

Supports:
- Per-tenant, per-user, per-API-key namespacing
- Global and per-endpoint limits
- Soft limits (warn) vs hard limits (reject)
- Quota inheritance (tenant → user → API key)
- In-memory backend (swap for Redis for distributed)
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class Algorithm(str, Enum):
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class LimitType(str, Enum):
    REQUESTS = "requests"
    TOKENS = "tokens"
    COST_USD = "cost_usd"
    DATA_BYTES = "data_bytes"


@dataclass
class RateLimit:
    """Definition of a rate limit rule"""
    name: str
    limit_type: LimitType
    max_value: float          # Maximum units per window
    window_seconds: int       # Time window in seconds
    algorithm: Algorithm = Algorithm.TOKEN_BUCKET
    burst_multiplier: float = 1.5     # Allowed burst ratio (token bucket only)
    soft_limit_pct: float = 0.8       # Warn at this % of limit
    priority: int = 0                 # Higher priority rules take precedence
    scope: str = "global"             # "global", "tenant", "user", "api_key"


@dataclass
class RateLimitState:
    """Mutable state for a single rate-limited identity"""
    # Token bucket state
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.time)
    # Sliding window state
    request_timestamps: List[float] = field(default_factory=list)
    # Fixed window state
    window_start: float = field(default_factory=time.time)
    window_count: float = 0.0
    # Leaky bucket state
    queue_size: float = 0.0
    last_leak: float = field(default_factory=time.time)
    # Totals
    total_requests: int = 0
    total_rejected: int = 0
    total_cost_usd: float = 0.0
    last_seen: float = field(default_factory=time.time)


@dataclass
class RateLimitResult:
    """Result of a rate limit check"""
    allowed: bool
    limit_name: str
    current_value: float
    max_value: float
    remaining: float
    reset_at: float
    retry_after_s: Optional[float] = None
    is_soft_limit: bool = False
    algorithm: Algorithm = Algorithm.TOKEN_BUCKET

    @property
    def utilization_pct(self) -> float:
        if self.max_value == 0:
            return 100.0
        return (self.current_value / self.max_value) * 100

    def to_headers(self) -> Dict[str, str]:
        """Return as HTTP rate-limit response headers"""
        headers = {
            "X-RateLimit-Limit": str(int(self.max_value)),
            "X-RateLimit-Remaining": str(max(0, int(self.remaining))),
            "X-RateLimit-Reset": str(int(self.reset_at)),
        }
        if self.retry_after_s is not None:
            headers["Retry-After"] = str(math.ceil(self.retry_after_s))
        return headers

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "limit_name": self.limit_name,
            "current_value": round(self.current_value, 4),
            "max_value": self.max_value,
            "remaining": round(self.remaining, 4),
            "reset_at": self.reset_at,
            "retry_after_s": self.retry_after_s,
            "utilization_pct": round(self.utilization_pct, 1),
        }


class RateLimiter:
    """
    Multi-algorithm rate limiter with per-tenant quota management.

    Usage::

        limiter = RateLimiter()
        limiter.add_rule(RateLimit(
            name="api_default",
            limit_type=LimitType.REQUESTS,
            max_value=100,
            window_seconds=60,
            algorithm=Algorithm.TOKEN_BUCKET,
        ))

        result = limiter.check("tenant:abc123", "api_default", consume=1.0)
        if not result.allowed:
            raise HTTPException(429, headers=result.to_headers())
    """

    def __init__(self) -> None:
        self._rules: Dict[str, RateLimit] = {}
        self._states: Dict[str, Dict[str, RateLimitState]] = {}
        # Override limits per identity (tenant/user overrides)
        self._overrides: Dict[str, Dict[str, float]] = {}

    # ──────────────────────────────────────────────
    # Configuration
    # ──────────────────────────────────────────────

    def add_rule(self, rule: RateLimit) -> None:
        self._rules[rule.name] = rule

    def remove_rule(self, name: str) -> None:
        self._rules.pop(name, None)

    def set_override(
        self,
        identity: str,
        rule_name: str,
        max_value: float,
    ) -> None:
        """Override the limit for a specific identity (e.g. premium tenant)"""
        self._overrides.setdefault(identity, {})[rule_name] = max_value

    def remove_override(self, identity: str, rule_name: Optional[str] = None) -> None:
        if rule_name:
            self._overrides.get(identity, {}).pop(rule_name, None)
        else:
            self._overrides.pop(identity, None)

    # ──────────────────────────────────────────────
    # Rate checking
    # ──────────────────────────────────────────────

    def check(
        self,
        identity: str,
        rule_name: str,
        consume: float = 1.0,
    ) -> RateLimitResult:
        """
        Check and optionally consume quota.
        Returns RateLimitResult with allowed=True/False.
        """
        rule = self._rules.get(rule_name)
        if rule is None:
            return RateLimitResult(
                allowed=True,
                limit_name=rule_name,
                current_value=0,
                max_value=math.inf,
                remaining=math.inf,
                reset_at=time.time() + 3600,
            )

        effective_max = self._overrides.get(identity, {}).get(rule_name, rule.max_value)
        state = self._get_state(identity, rule_name)
        state.last_seen = time.time()

        if rule.algorithm == Algorithm.TOKEN_BUCKET:
            return self._token_bucket_check(rule, state, consume, effective_max)
        if rule.algorithm == Algorithm.SLIDING_WINDOW:
            return self._sliding_window_check(rule, state, consume, effective_max)
        if rule.algorithm == Algorithm.FIXED_WINDOW:
            return self._fixed_window_check(rule, state, consume, effective_max)
        if rule.algorithm == Algorithm.LEAKY_BUCKET:
            return self._leaky_bucket_check(rule, state, consume, effective_max)
        return RateLimitResult(allowed=True, limit_name=rule_name, current_value=0,
                               max_value=effective_max, remaining=effective_max,
                               reset_at=time.time() + rule.window_seconds)

    def check_all(
        self,
        identity: str,
        rule_names: List[str],
        consume: float = 1.0,
    ) -> Tuple[bool, List[RateLimitResult]]:
        """
        Check multiple rules. Returns (all_allowed, [results]).
        Consumes quota only if ALL rules pass.
        """
        # Dry-run first
        results = []
        all_allowed = True
        for rule_name in rule_names:
            result = self.check(identity, rule_name, consume=0)  # dry check
            results.append(result)
            if not result.allowed:
                all_allowed = False

        if all_allowed:
            # Actually consume
            results = []
            for rule_name in rule_names:
                results.append(self.check(identity, rule_name, consume=consume))

        return all_allowed, results

    def get_quota_status(self, identity: str) -> Dict[str, Any]:
        """Return full quota status for an identity"""
        status: Dict[str, Any] = {}
        for rule_name, rule in self._rules.items():
            state = self._get_state(identity, rule_name)
            effective_max = self._overrides.get(identity, {}).get(rule_name, rule.max_value)
            result = self.check(identity, rule_name, consume=0)
            status[rule_name] = {
                "max_value": effective_max,
                "remaining": round(result.remaining, 2),
                "utilization_pct": round(result.utilization_pct, 1),
                "total_requests": state.total_requests,
                "total_rejected": state.total_rejected,
            }
        return status

    # ──────────────────────────────────────────────
    # Algorithm implementations
    # ──────────────────────────────────────────────

    def _token_bucket_check(
        self,
        rule: RateLimit,
        state: RateLimitState,
        consume: float,
        effective_max: float,
    ) -> RateLimitResult:
        now = time.time()
        burst_max = effective_max * rule.burst_multiplier
        refill_rate = effective_max / rule.window_seconds

        # Refill tokens since last check
        elapsed = now - state.last_refill
        state.tokens = min(burst_max, state.tokens + elapsed * refill_rate)
        state.last_refill = now

        if consume == 0:
            # Dry-run
            remaining = state.tokens
            allowed = remaining >= 0
            return RateLimitResult(
                allowed=allowed,
                limit_name=rule.name,
                current_value=burst_max - remaining,
                max_value=effective_max,
                remaining=remaining,
                reset_at=now + (burst_max - state.tokens) / max(refill_rate, 1e-9),
                algorithm=Algorithm.TOKEN_BUCKET,
            )

        allowed = state.tokens >= consume
        if allowed:
            state.tokens -= consume
            state.total_requests += 1
        else:
            state.total_rejected += 1

        retry_after = (consume - state.tokens) / max(refill_rate, 1e-9) if not allowed else None

        return RateLimitResult(
            allowed=allowed,
            limit_name=rule.name,
            current_value=burst_max - state.tokens,
            max_value=effective_max,
            remaining=max(0.0, state.tokens),
            reset_at=now + (burst_max - state.tokens) / max(refill_rate, 1e-9),
            retry_after_s=retry_after,
            is_soft_limit=(state.tokens / burst_max) < (1 - rule.soft_limit_pct),
            algorithm=Algorithm.TOKEN_BUCKET,
        )

    def _sliding_window_check(
        self,
        rule: RateLimit,
        state: RateLimitState,
        consume: float,
        effective_max: float,
    ) -> RateLimitResult:
        now = time.time()
        cutoff = now - rule.window_seconds
        state.request_timestamps = [t for t in state.request_timestamps if t > cutoff]
        current = len(state.request_timestamps)

        allowed = current + consume <= effective_max
        if allowed and consume > 0:
            for _ in range(int(consume)):
                state.request_timestamps.append(now)
            state.total_requests += 1
        elif not allowed and consume > 0:
            state.total_rejected += 1

        oldest = state.request_timestamps[0] if state.request_timestamps else now
        reset_at = oldest + rule.window_seconds
        retry_after = max(0.0, reset_at - now) if not allowed else None

        return RateLimitResult(
            allowed=allowed,
            limit_name=rule.name,
            current_value=current,
            max_value=effective_max,
            remaining=max(0.0, effective_max - current),
            reset_at=reset_at,
            retry_after_s=retry_after,
            algorithm=Algorithm.SLIDING_WINDOW,
        )

    def _fixed_window_check(
        self,
        rule: RateLimit,
        state: RateLimitState,
        consume: float,
        effective_max: float,
    ) -> RateLimitResult:
        now = time.time()
        if now - state.window_start >= rule.window_seconds:
            state.window_start = now
            state.window_count = 0.0

        allowed = state.window_count + consume <= effective_max
        if allowed and consume > 0:
            state.window_count += consume
            state.total_requests += 1
        elif not allowed and consume > 0:
            state.total_rejected += 1

        reset_at = state.window_start + rule.window_seconds
        retry_after = max(0.0, reset_at - now) if not allowed else None

        return RateLimitResult(
            allowed=allowed,
            limit_name=rule.name,
            current_value=state.window_count,
            max_value=effective_max,
            remaining=max(0.0, effective_max - state.window_count),
            reset_at=reset_at,
            retry_after_s=retry_after,
            algorithm=Algorithm.FIXED_WINDOW,
        )

    def _leaky_bucket_check(
        self,
        rule: RateLimit,
        state: RateLimitState,
        consume: float,
        effective_max: float,
    ) -> RateLimitResult:
        now = time.time()
        leak_rate = effective_max / rule.window_seconds
        elapsed = now - state.last_leak
        leaked = elapsed * leak_rate
        state.queue_size = max(0.0, state.queue_size - leaked)
        state.last_leak = now

        allowed = state.queue_size + consume <= effective_max
        if allowed and consume > 0:
            state.queue_size += consume
            state.total_requests += 1
        elif not allowed and consume > 0:
            state.total_rejected += 1

        drain_time = state.queue_size / max(leak_rate, 1e-9)
        retry_after = drain_time if not allowed else None

        return RateLimitResult(
            allowed=allowed,
            limit_name=rule.name,
            current_value=state.queue_size,
            max_value=effective_max,
            remaining=max(0.0, effective_max - state.queue_size),
            reset_at=now + drain_time,
            retry_after_s=retry_after,
            algorithm=Algorithm.LEAKY_BUCKET,
        )

    # ──────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────

    def _get_state(self, identity: str, rule_name: str) -> RateLimitState:
        if identity not in self._states:
            self._states[identity] = {}
        if rule_name not in self._states[identity]:
            rule = self._rules.get(rule_name)
            initial_tokens = (rule.max_value * rule.burst_multiplier) if rule else 0.0
            self._states[identity][rule_name] = RateLimitState(tokens=initial_tokens)
        return self._states[identity][rule_name]

    def evict_stale(self, max_age_s: int = 3600) -> int:
        """Remove states for identities that haven't been seen recently"""
        cutoff = time.time() - max_age_s
        count = 0
        for identity in list(self._states.keys()):
            for rule_name, state in list(self._states[identity].items()):
                if state.last_seen < cutoff:
                    del self._states[identity][rule_name]
                    count += 1
            if not self._states[identity]:
                del self._states[identity]
        return count


# Pre-built tier configurations
def build_free_tier_limiter() -> RateLimiter:
    limiter = RateLimiter()
    limiter.add_rule(RateLimit("requests_per_minute", LimitType.REQUESTS, 30, 60, Algorithm.TOKEN_BUCKET))
    limiter.add_rule(RateLimit("requests_per_day", LimitType.REQUESTS, 500, 86400, Algorithm.FIXED_WINDOW))
    limiter.add_rule(RateLimit("tokens_per_day", LimitType.TOKENS, 50000, 86400, Algorithm.FIXED_WINDOW))
    limiter.add_rule(RateLimit("cost_per_day", LimitType.COST_USD, 0.10, 86400, Algorithm.FIXED_WINDOW))
    return limiter


def build_pro_tier_limiter() -> RateLimiter:
    limiter = RateLimiter()
    limiter.add_rule(RateLimit("requests_per_minute", LimitType.REQUESTS, 300, 60, Algorithm.TOKEN_BUCKET))
    limiter.add_rule(RateLimit("requests_per_day", LimitType.REQUESTS, 50000, 86400, Algorithm.SLIDING_WINDOW))
    limiter.add_rule(RateLimit("tokens_per_day", LimitType.TOKENS, 5000000, 86400, Algorithm.FIXED_WINDOW))
    limiter.add_rule(RateLimit("cost_per_day", LimitType.COST_USD, 10.0, 86400, Algorithm.FIXED_WINDOW))
    return limiter


def build_enterprise_tier_limiter() -> RateLimiter:
    limiter = RateLimiter()
    limiter.add_rule(RateLimit("requests_per_minute", LimitType.REQUESTS, 3000, 60, Algorithm.TOKEN_BUCKET, burst_multiplier=2.0))
    limiter.add_rule(RateLimit("requests_per_day", LimitType.REQUESTS, 1000000, 86400, Algorithm.SLIDING_WINDOW))
    limiter.add_rule(RateLimit("tokens_per_day", LimitType.TOKENS, 100000000, 86400, Algorithm.FIXED_WINDOW))
    limiter.add_rule(RateLimit("cost_per_day", LimitType.COST_USD, 1000.0, 86400, Algorithm.FIXED_WINDOW))
    return limiter
