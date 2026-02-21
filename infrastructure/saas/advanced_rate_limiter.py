"""
Advanced Rate Limiting with Dynamic Quotas
Intelligent rate limiting system with adaptive quotas, burst handling, and cost-based throttling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
import hashlib
from collections import defaultdict


class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    ADAPTIVE = "adaptive"
    COST_BASED = "cost_based"


class LimitScope(Enum):
    """Scope of rate limit"""
    GLOBAL = "global"
    PER_TENANT = "per_tenant"
    PER_USER = "per_user"
    PER_IP = "per_ip"
    PER_ENDPOINT = "per_endpoint"
    PER_RESOURCE = "per_resource"


@dataclass
class RateLimit:
    """Rate limit configuration"""
    limit_id: str
    name: str
    strategy: RateLimitStrategy
    scope: LimitScope
    max_requests: int
    window_seconds: int
    burst_size: int = 0  # Allow bursts above limit
    cost_per_request: Decimal = Decimal("0")
    max_cost_per_window: Decimal = Decimal("0")
    priority_enabled: bool = False
    dynamic_adjustment: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitBucket:
    """Token bucket for rate limiting"""
    bucket_id: str
    capacity: int
    tokens: float
    refill_rate: float  # Tokens per second
    last_refill: datetime
    burst_capacity: int = 0


@dataclass
class RateLimitViolation:
    """Rate limit violation record"""
    violation_id: str
    limit_id: str
    scope_key: str
    timestamp: datetime
    requested: int
    allowed: int
    retry_after_seconds: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DynamicQuota:
    """Dynamic quota that adjusts based on usage patterns"""
    quota_id: str
    base_limit: int
    current_limit: int
    min_limit: int
    max_limit: int
    adjustment_factor: float = 1.0
    last_adjusted: datetime = field(default_factory=datetime.utcnow)
    adjustment_history: List[Dict[str, Any]] = field(default_factory=list)


class TokenBucketLimiter:
    """Token bucket rate limiter"""

    def __init__(self):
        self.buckets: Dict[str, RateLimitBucket] = {}

    async def check_limit(
        self,
        bucket_id: str,
        capacity: int,
        refill_rate: float,
        tokens_requested: int = 1,
        burst_capacity: int = 0
    ) -> tuple[bool, int]:
        """
        Check if request is allowed under token bucket limit
        Returns: (allowed, retry_after_seconds)
        """
        now = datetime.utcnow()

        # Get or create bucket
        if bucket_id not in self.buckets:
            self.buckets[bucket_id] = RateLimitBucket(
                bucket_id=bucket_id,
                capacity=capacity,
                tokens=capacity,
                refill_rate=refill_rate,
                last_refill=now,
                burst_capacity=burst_capacity
            )

        bucket = self.buckets[bucket_id]

        # Refill tokens
        time_passed = (now - bucket.last_refill).total_seconds()
        tokens_to_add = time_passed * bucket.refill_rate
        bucket.tokens = min(bucket.capacity + bucket.burst_capacity, bucket.tokens + tokens_to_add)
        bucket.last_refill = now

        # Check if enough tokens
        if bucket.tokens >= tokens_requested:
            bucket.tokens -= tokens_requested
            return True, 0
        else:
            # Calculate retry after
            tokens_needed = tokens_requested - bucket.tokens
            retry_after = int(tokens_needed / bucket.refill_rate) + 1
            return False, retry_after

    async def reset_bucket(self, bucket_id: str):
        """Reset bucket to full capacity"""
        if bucket_id in self.buckets:
            bucket = self.buckets[bucket_id]
            bucket.tokens = bucket.capacity
            bucket.last_refill = datetime.utcnow()


class SlidingWindowLimiter:
    """Sliding window rate limiter"""

    def __init__(self):
        self.request_log: Dict[str, List[datetime]] = defaultdict(list)

    async def check_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> tuple[bool, int]:
        """
        Check if request is allowed under sliding window limit
        Returns: (allowed, retry_after_seconds)
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=window_seconds)

        # Clean old requests
        self.request_log[key] = [
            ts for ts in self.request_log[key]
            if ts > cutoff
        ]

        # Check limit
        if len(self.request_log[key]) < max_requests:
            self.request_log[key].append(now)
            return True, 0
        else:
            # Calculate retry after
            oldest_request = min(self.request_log[key])
            retry_after = int((oldest_request - cutoff).total_seconds()) + 1
            return False, retry_after

    async def get_current_count(self, key: str, window_seconds: int) -> int:
        """Get current request count in window"""
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=window_seconds)

        self.request_log[key] = [
            ts for ts in self.request_log[key]
            if ts > cutoff
        ]

        return len(self.request_log[key])


class CostBasedLimiter:
    """Cost-based rate limiter"""

    def __init__(self):
        self.cost_tracker: Dict[str, List[tuple[datetime, Decimal]]] = defaultdict(list)

    async def check_limit(
        self,
        key: str,
        cost: Decimal,
        max_cost: Decimal,
        window_seconds: int
    ) -> tuple[bool, int]:
        """
        Check if request is allowed under cost limit
        Returns: (allowed, retry_after_seconds)
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=window_seconds)

        # Clean old costs
        self.cost_tracker[key] = [
            (ts, c) for ts, c in self.cost_tracker[key]
            if ts > cutoff
        ]

        # Calculate current cost
        current_cost = sum(c for _, c in self.cost_tracker[key])

        # Check if adding this cost would exceed limit
        if current_cost + cost <= max_cost:
            self.cost_tracker[key].append((now, cost))
            return True, 0
        else:
            # Calculate retry after (when oldest cost expires)
            if self.cost_tracker[key]:
                oldest_time = min(ts for ts, _ in self.cost_tracker[key])
                retry_after = int((oldest_time - cutoff).total_seconds()) + 1
            else:
                retry_after = window_seconds

            return False, retry_after

    async def get_current_cost(self, key: str, window_seconds: int) -> Decimal:
        """Get current cost in window"""
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=window_seconds)

        self.cost_tracker[key] = [
            (ts, c) for ts, c in self.cost_tracker[key]
            if ts > cutoff
        ]

        return sum(c for _, c in self.cost_tracker[key])


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on usage patterns"""

    def __init__(self):
        self.dynamic_quotas: Dict[str, DynamicQuota] = {}
        self.token_bucket_limiter = TokenBucketLimiter()
        self.usage_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    async def check_limit(
        self,
        key: str,
        base_limit: int,
        window_seconds: int,
        min_limit: int,
        max_limit: int
    ) -> tuple[bool, int]:
        """Check adaptive rate limit"""
        # Get or create dynamic quota
        if key not in self.dynamic_quotas:
            self.dynamic_quotas[key] = DynamicQuota(
                quota_id=key,
                base_limit=base_limit,
                current_limit=base_limit,
                min_limit=min_limit,
                max_limit=max_limit
            )

        quota = self.dynamic_quotas[key]

        # Check using token bucket with current limit
        refill_rate = quota.current_limit / window_seconds
        allowed, retry_after = await self.token_bucket_limiter.check_limit(
            bucket_id=key,
            capacity=quota.current_limit,
            refill_rate=refill_rate,
            tokens_requested=1
        )

        # Record usage
        self.usage_history[key].append({
            "timestamp": datetime.utcnow(),
            "allowed": allowed,
            "current_limit": quota.current_limit
        })

        # Periodically adjust quota
        await self._adjust_quota(key)

        return allowed, retry_after

    async def _adjust_quota(self, key: str):
        """Adjust quota based on usage patterns"""
        quota = self.dynamic_quotas.get(key)
        if not quota:
            return

        # Only adjust every 5 minutes
        if (datetime.utcnow() - quota.last_adjusted).total_seconds() < 300:
            return

        # Analyze recent usage
        recent_usage = [
            u for u in self.usage_history.get(key, [])
            if u["timestamp"] >= datetime.utcnow() - timedelta(minutes=10)
        ]

        if len(recent_usage) < 10:
            return

        # Calculate metrics
        total_requests = len(recent_usage)
        blocked_requests = sum(1 for u in recent_usage if not u["allowed"])
        block_rate = blocked_requests / total_requests

        old_limit = quota.current_limit

        # Adjust based on block rate
        if block_rate > 0.2:  # More than 20% blocked
            # Increase limit if under max
            quota.current_limit = min(
                quota.max_limit,
                int(quota.current_limit * 1.2)
            )
        elif block_rate < 0.05:  # Less than 5% blocked
            # Decrease limit if above min
            quota.current_limit = max(
                quota.min_limit,
                int(quota.current_limit * 0.9)
            )

        # Record adjustment
        if old_limit != quota.current_limit:
            quota.adjustment_history.append({
                "timestamp": datetime.utcnow(),
                "old_limit": old_limit,
                "new_limit": quota.current_limit,
                "block_rate": block_rate,
                "reason": "high_block_rate" if block_rate > 0.2 else "low_utilization"
            })
            quota.last_adjusted = datetime.utcnow()


class RateLimitManager:
    """Main rate limiting manager"""

    def __init__(self):
        self.limits: Dict[str, RateLimit] = {}
        self.token_bucket_limiter = TokenBucketLimiter()
        self.sliding_window_limiter = SlidingWindowLimiter()
        self.cost_based_limiter = CostBasedLimiter()
        self.adaptive_limiter = AdaptiveRateLimiter()
        self.violations: List[RateLimitViolation] = []

    async def create_limit(
        self,
        name: str,
        strategy: RateLimitStrategy,
        scope: LimitScope,
        max_requests: int,
        window_seconds: int,
        burst_size: int = 0,
        cost_per_request: Decimal = Decimal("0"),
        max_cost_per_window: Decimal = Decimal("0"),
        dynamic_adjustment: bool = False
    ) -> RateLimit:
        """Create new rate limit"""
        import uuid

        limit_id = str(uuid.uuid4())

        limit = RateLimit(
            limit_id=limit_id,
            name=name,
            strategy=strategy,
            scope=scope,
            max_requests=max_requests,
            window_seconds=window_seconds,
            burst_size=burst_size,
            cost_per_request=cost_per_request,
            max_cost_per_window=max_cost_per_window,
            dynamic_adjustment=dynamic_adjustment
        )

        self.limits[limit_id] = limit

        return limit

    async def check_rate_limit(
        self,
        limit_id: str,
        scope_key: str,
        tokens_requested: int = 1,
        request_cost: Optional[Decimal] = None
    ) -> tuple[bool, Optional[int]]:
        """
        Check if request is allowed
        Returns: (allowed, retry_after_seconds)
        """
        limit = self.limits.get(limit_id)
        if not limit:
            # No limit configured - allow
            return True, None

        # Build key for this scope
        key = f"{limit_id}:{scope_key}"

        # Apply strategy
        if limit.strategy == RateLimitStrategy.TOKEN_BUCKET:
            refill_rate = limit.max_requests / limit.window_seconds
            allowed, retry_after = await self.token_bucket_limiter.check_limit(
                bucket_id=key,
                capacity=limit.max_requests,
                refill_rate=refill_rate,
                tokens_requested=tokens_requested,
                burst_capacity=limit.burst_size
            )

        elif limit.strategy == RateLimitStrategy.SLIDING_WINDOW:
            allowed, retry_after = await self.sliding_window_limiter.check_limit(
                key=key,
                max_requests=limit.max_requests,
                window_seconds=limit.window_seconds
            )

        elif limit.strategy == RateLimitStrategy.COST_BASED:
            cost = request_cost or limit.cost_per_request
            allowed, retry_after = await self.cost_based_limiter.check_limit(
                key=key,
                cost=cost,
                max_cost=limit.max_cost_per_window,
                window_seconds=limit.window_seconds
            )

        elif limit.strategy == RateLimitStrategy.ADAPTIVE:
            min_limit = int(limit.max_requests * 0.5)
            max_limit = int(limit.max_requests * 2.0)
            allowed, retry_after = await self.adaptive_limiter.check_limit(
                key=key,
                base_limit=limit.max_requests,
                window_seconds=limit.window_seconds,
                min_limit=min_limit,
                max_limit=max_limit
            )

        else:
            # Default to sliding window
            allowed, retry_after = await self.sliding_window_limiter.check_limit(
                key=key,
                max_requests=limit.max_requests,
                window_seconds=limit.window_seconds
            )

        # Record violation if not allowed
        if not allowed:
            await self._record_violation(
                limit_id=limit_id,
                scope_key=scope_key,
                requested=tokens_requested,
                retry_after=retry_after
            )

        return allowed, retry_after if not allowed else None

    async def _record_violation(
        self,
        limit_id: str,
        scope_key: str,
        requested: int,
        retry_after: int
    ):
        """Record rate limit violation"""
        import uuid

        violation = RateLimitViolation(
            violation_id=str(uuid.uuid4()),
            limit_id=limit_id,
            scope_key=scope_key,
            timestamp=datetime.utcnow(),
            requested=requested,
            allowed=0,
            retry_after_seconds=retry_after
        )

        self.violations.append(violation)

    async def get_current_usage(
        self,
        limit_id: str,
        scope_key: str
    ) -> Dict[str, Any]:
        """Get current usage for a limit"""
        limit = self.limits.get(limit_id)
        if not limit:
            return {}

        key = f"{limit_id}:{scope_key}"

        if limit.strategy == RateLimitStrategy.SLIDING_WINDOW:
            count = await self.sliding_window_limiter.get_current_count(
                key=key,
                window_seconds=limit.window_seconds
            )
            return {
                "current": count,
                "limit": limit.max_requests,
                "percentage": (count / limit.max_requests * 100) if limit.max_requests > 0 else 0
            }

        elif limit.strategy == RateLimitStrategy.COST_BASED:
            cost = await self.cost_based_limiter.get_current_cost(
                key=key,
                window_seconds=limit.window_seconds
            )
            return {
                "current_cost": float(cost),
                "max_cost": float(limit.max_cost_per_window),
                "percentage": (float(cost) / float(limit.max_cost_per_window) * 100) if limit.max_cost_per_window > 0 else 0
            }

        return {}

    async def get_violation_report(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get violation report"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_violations = [
            v for v in self.violations
            if v.timestamp >= cutoff
        ]

        if not recent_violations:
            return {"total_violations": 0}

        # Aggregate by limit
        by_limit = defaultdict(int)
        by_scope = defaultdict(int)

        for v in recent_violations:
            by_limit[v.limit_id] += 1
            by_scope[v.scope_key] += 1

        return {
            "total_violations": len(recent_violations),
            "by_limit": dict(by_limit),
            "by_scope": dict(by_scope),
            "time_range_hours": hours
        }
