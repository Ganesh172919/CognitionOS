"""
API Key Management and Rate Limiting

Comprehensive API key management with tier-based rate limiting and quota enforcement.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import hashlib
import secrets
import logging
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)


class APIKeyScope(str, Enum):
    """API key permission scopes"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    BILLING = "billing"


class RateLimitWindow(str, Enum):
    """Rate limit time windows"""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    MONTH = "month"


@dataclass
class APIKey:
    """API key model"""
    id: str
    tenant_id: str
    key_hash: str
    name: str
    scopes: Set[APIKeyScope]
    created_at: datetime
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True

    # Rate limiting
    rate_limit_per_minute: Optional[int] = None
    rate_limit_per_hour: Optional[int] = None
    rate_limit_per_day: Optional[int] = None

    # Usage tracking
    total_requests: int = 0

    # Metadata
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_scope(self, scope: APIKeyScope) -> bool:
        """Check if key has a specific scope"""
        return scope in self.scopes or APIKeyScope.ADMIN in self.scopes

    def is_valid(self) -> bool:
        """Check if key is valid"""
        if not self.is_active:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True


@dataclass
class RateLimit:
    """Rate limit configuration"""
    window: RateLimitWindow
    limit: int

    def get_window_seconds(self) -> int:
        """Get window duration in seconds"""
        windows = {
            RateLimitWindow.SECOND: 1,
            RateLimitWindow.MINUTE: 60,
            RateLimitWindow.HOUR: 3600,
            RateLimitWindow.DAY: 86400,
            RateLimitWindow.MONTH: 2592000
        }
        return windows.get(self.window, 60)


@dataclass
class RateLimitStatus:
    """Current rate limit status"""
    limit: int
    remaining: int
    reset_at: datetime
    window: RateLimitWindow

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers"""
        return {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_at.timestamp())),
            "X-RateLimit-Window": self.window.value
        }


class APIKeyManager:
    """
    Manages API keys for tenants

    Handles creation, validation, rotation, and lifecycle management of API keys.
    """

    def __init__(self, storage_backend: Optional[Any] = None):
        self.storage = storage_backend
        self._key_cache: Dict[str, APIKey] = {}
        self._cache_ttl = timedelta(minutes=5)

    def generate_key(self, prefix: str = "sk") -> tuple[str, str]:
        """
        Generate API key and its hash

        Returns:
            Tuple of (plain_key, key_hash)
        """
        # Generate random key
        random_part = secrets.token_urlsafe(32)
        plain_key = f"{prefix}_{random_part}"

        # Hash for storage
        key_hash = hashlib.sha256(plain_key.encode()).hexdigest()

        return plain_key, key_hash

    async def create_key(
        self,
        tenant_id: str,
        name: str,
        scopes: Set[APIKeyScope],
        expires_in_days: Optional[int] = None,
        rate_limits: Optional[Dict[RateLimitWindow, int]] = None,
        created_by: Optional[str] = None
    ) -> tuple[APIKey, str]:
        """
        Create new API key

        Args:
            tenant_id: Tenant identifier
            name: Human-readable key name
            scopes: Permission scopes
            expires_in_days: Optional expiration in days
            rate_limits: Optional custom rate limits
            created_by: User who created the key

        Returns:
            Tuple of (APIKey object, plain_key_string)
        """
        plain_key, key_hash = self.generate_key()

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        api_key = APIKey(
            id=f"key_{secrets.token_hex(8)}",
            tenant_id=tenant_id,
            key_hash=key_hash,
            name=name,
            scopes=scopes,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            created_by=created_by,
            rate_limit_per_minute=rate_limits.get(RateLimitWindow.MINUTE) if rate_limits else None,
            rate_limit_per_hour=rate_limits.get(RateLimitWindow.HOUR) if rate_limits else None,
            rate_limit_per_day=rate_limits.get(RateLimitWindow.DAY) if rate_limits else None
        )

        if self.storage:
            await self.storage.save_key(api_key)

        self._key_cache[key_hash] = api_key
        logger.info(f"Created API key {api_key.id} for tenant {tenant_id}")

        return api_key, plain_key

    async def validate_key(self, plain_key: str) -> Optional[APIKey]:
        """
        Validate API key

        Args:
            plain_key: Plain text API key

        Returns:
            APIKey object if valid, None otherwise
        """
        key_hash = hashlib.sha256(plain_key.encode()).hexdigest()

        # Check cache first
        if key_hash in self._key_cache:
            api_key = self._key_cache[key_hash]
            if api_key.is_valid():
                return api_key

        # Check storage
        if self.storage:
            api_key = await self.storage.get_key_by_hash(key_hash)
            if api_key and api_key.is_valid():
                self._key_cache[key_hash] = api_key
                return api_key

        return None

    async def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key"""
        if self.storage:
            success = await self.storage.revoke_key(key_id)
            if success:
                # Invalidate cache
                self._key_cache = {
                    k: v for k, v in self._key_cache.items()
                    if v.id != key_id
                }
                logger.info(f"Revoked API key {key_id}")
                return True
        return False

    async def rotate_key(self, key_id: str, grace_period_hours: int = 24) -> tuple[APIKey, str]:
        """
        Rotate an API key

        Creates a new key and schedules old key for deletion.

        Args:
            key_id: ID of key to rotate
            grace_period_hours: Hours before old key expires

        Returns:
            Tuple of (new_key, plain_key)
        """
        if not self.storage:
            raise ValueError("Storage backend required for key rotation")

        old_key = await self.storage.get_key(key_id)
        if not old_key:
            raise ValueError(f"Key {key_id} not found")

        # Create new key with same permissions
        new_key, plain_key = await self.create_key(
            tenant_id=old_key.tenant_id,
            name=f"{old_key.name} (rotated)",
            scopes=old_key.scopes,
            rate_limits={
                RateLimitWindow.MINUTE: old_key.rate_limit_per_minute,
                RateLimitWindow.HOUR: old_key.rate_limit_per_hour,
                RateLimitWindow.DAY: old_key.rate_limit_per_day
            } if old_key.rate_limit_per_minute else None
        )

        # Schedule old key expiration
        old_key.expires_at = datetime.utcnow() + timedelta(hours=grace_period_hours)
        await self.storage.update_key(old_key)

        logger.info(f"Rotated key {key_id} with {grace_period_hours}h grace period")
        return new_key, plain_key

    async def list_keys(self, tenant_id: str) -> List[APIKey]:
        """List all keys for a tenant"""
        if not self.storage:
            return []
        return await self.storage.list_keys(tenant_id)

    async def update_last_used(self, key_id: str):
        """Update last used timestamp"""
        if self.storage:
            await self.storage.update_last_used(key_id, datetime.utcnow())


class RateLimiter:
    """
    Token bucket rate limiter

    Implements sliding window rate limiting with multiple time windows.
    """

    def __init__(self, storage_backend: Optional[Any] = None):
        self.storage = storage_backend
        self._local_cache: Dict[str, Dict[str, Any]] = defaultdict(dict)

    async def check_rate_limit(
        self,
        key: str,
        limits: List[RateLimit]
    ) -> tuple[bool, Optional[RateLimitStatus]]:
        """
        Check if request is within rate limits

        Args:
            key: Identifier (tenant_id, api_key, etc.)
            limits: List of rate limits to check

        Returns:
            Tuple of (allowed, status)
        """
        now = datetime.utcnow()

        # Check each window
        for rate_limit in limits:
            window_key = f"{key}:{rate_limit.window.value}"

            # Get current count
            count, window_start = await self._get_window_count(window_key, rate_limit)

            # Calculate reset time
            window_seconds = rate_limit.get_window_seconds()
            reset_at = window_start + timedelta(seconds=window_seconds)

            # Check if limit exceeded
            if count >= rate_limit.limit:
                status = RateLimitStatus(
                    limit=rate_limit.limit,
                    remaining=0,
                    reset_at=reset_at,
                    window=rate_limit.window
                )
                return False, status

        # All limits passed, increment counters
        for rate_limit in limits:
            window_key = f"{key}:{rate_limit.window.value}"
            await self._increment_window_count(window_key, rate_limit)

        # Return status for primary window (usually minute)
        primary_limit = limits[0]
        count, window_start = await self._get_window_count(
            f"{key}:{primary_limit.window.value}",
            primary_limit
        )

        status = RateLimitStatus(
            limit=primary_limit.limit,
            remaining=primary_limit.limit - count,
            reset_at=window_start + timedelta(seconds=primary_limit.get_window_seconds()),
            window=primary_limit.window
        )

        return True, status

    async def _get_window_count(
        self,
        window_key: str,
        rate_limit: RateLimit
    ) -> tuple[int, datetime]:
        """Get current count for a time window"""
        now = datetime.utcnow()
        window_seconds = rate_limit.get_window_seconds()

        # Calculate window start
        timestamp = int(now.timestamp())
        window_start_timestamp = (timestamp // window_seconds) * window_seconds
        window_start = datetime.fromtimestamp(window_start_timestamp)

        if self.storage:
            # Use distributed storage (Redis)
            count = await self.storage.get_window_count(window_key, window_start)
        else:
            # Use local cache
            cache_entry = self._local_cache.get(window_key, {})
            cache_start = cache_entry.get("window_start")

            if cache_start and cache_start >= window_start:
                count = cache_entry.get("count", 0)
            else:
                count = 0
                self._local_cache[window_key] = {
                    "window_start": window_start,
                    "count": 0
                }

        return count, window_start

    async def _increment_window_count(self, window_key: str, rate_limit: RateLimit):
        """Increment count for a time window"""
        now = datetime.utcnow()
        window_seconds = rate_limit.get_window_seconds()

        timestamp = int(now.timestamp())
        window_start_timestamp = (timestamp // window_seconds) * window_seconds
        window_start = datetime.fromtimestamp(window_start_timestamp)

        if self.storage:
            await self.storage.increment_window_count(
                window_key,
                window_start,
                ttl_seconds=window_seconds * 2
            )
        else:
            cache_entry = self._local_cache[window_key]
            cache_entry["count"] = cache_entry.get("count", 0) + 1

    async def reset_limits(self, key: str):
        """Reset all limits for a key"""
        if self.storage:
            await self.storage.reset_limits(key)
        else:
            # Clear local cache for this key
            keys_to_remove = [k for k in self._local_cache.keys() if k.startswith(f"{key}:")]
            for k in keys_to_remove:
                del self._local_cache[k]


class TierBasedLimiter:
    """
    Tier-based rate limiting

    Applies different rate limits based on subscription tier.
    """

    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self._tier_limits: Dict[str, List[RateLimit]] = {}
        self._initialize_default_limits()

    def _initialize_default_limits(self):
        """Initialize default tier-based limits"""
        # Free tier
        self._tier_limits["free"] = [
            RateLimit(RateLimitWindow.MINUTE, 10),
            RateLimit(RateLimitWindow.HOUR, 100),
            RateLimit(RateLimitWindow.DAY, 1000)
        ]

        # Starter tier
        self._tier_limits["starter"] = [
            RateLimit(RateLimitWindow.MINUTE, 100),
            RateLimit(RateLimitWindow.HOUR, 1000),
            RateLimit(RateLimitWindow.DAY, 10000)
        ]

        # Pro tier
        self._tier_limits["pro"] = [
            RateLimit(RateLimitWindow.MINUTE, 1000),
            RateLimit(RateLimitWindow.HOUR, 10000),
            RateLimit(RateLimitWindow.DAY, 100000)
        ]

        # Enterprise tier (very high limits)
        self._tier_limits["enterprise"] = [
            RateLimit(RateLimitWindow.MINUTE, 10000),
            RateLimit(RateLimitWindow.HOUR, 100000),
            RateLimit(RateLimitWindow.DAY, 1000000)
        ]

    def set_tier_limits(self, tier: str, limits: List[RateLimit]):
        """Set custom limits for a tier"""
        self._tier_limits[tier] = limits

    async def check_limit(
        self,
        tenant_id: str,
        tier: str
    ) -> tuple[bool, Optional[RateLimitStatus]]:
        """
        Check rate limit based on tier

        Args:
            tenant_id: Tenant identifier
            tier: Subscription tier

        Returns:
            Tuple of (allowed, status)
        """
        limits = self._tier_limits.get(tier.lower())
        if not limits:
            logger.warning(f"No rate limits defined for tier {tier}, using free tier")
            limits = self._tier_limits["free"]

        return await self.rate_limiter.check_rate_limit(tenant_id, limits)


class QuotaEnforcer:
    """
    Monthly quota enforcement

    Enforces monthly quotas for various resources.
    """

    def __init__(self, storage_backend: Optional[Any] = None):
        self.storage = storage_backend
        self._quota_cache: Dict[str, Dict[str, int]] = defaultdict(dict)

    async def check_quota(
        self,
        tenant_id: str,
        resource: str,
        amount: int,
        monthly_limit: Optional[int]
    ) -> tuple[bool, int]:
        """
        Check if quota allows operation

        Args:
            tenant_id: Tenant identifier
            resource: Resource type
            amount: Amount to consume
            monthly_limit: Monthly limit (None = unlimited)

        Returns:
            Tuple of (allowed, remaining)
        """
        if monthly_limit is None:
            return True, -1  # Unlimited

        # Get current month usage
        current_usage = await self._get_monthly_usage(tenant_id, resource)

        # Check if operation would exceed limit
        new_usage = current_usage + amount
        if new_usage > monthly_limit:
            return False, monthly_limit - current_usage

        return True, monthly_limit - new_usage

    async def consume_quota(
        self,
        tenant_id: str,
        resource: str,
        amount: int
    ):
        """Consume quota for a resource"""
        month_key = datetime.utcnow().strftime("%Y-%m")
        quota_key = f"{tenant_id}:{resource}:{month_key}"

        if self.storage:
            await self.storage.increment_quota(quota_key, amount)
        else:
            self._quota_cache[tenant_id][resource] = \
                self._quota_cache[tenant_id].get(resource, 0) + amount

    async def _get_monthly_usage(self, tenant_id: str, resource: str) -> int:
        """Get current month usage for a resource"""
        month_key = datetime.utcnow().strftime("%Y-%m")
        quota_key = f"{tenant_id}:{resource}:{month_key}"

        if self.storage:
            return await self.storage.get_quota(quota_key)
        else:
            return self._quota_cache[tenant_id].get(resource, 0)

    async def get_quota_status(
        self,
        tenant_id: str,
        resource: str,
        monthly_limit: Optional[int]
    ) -> Dict[str, Any]:
        """Get quota status for a resource"""
        usage = await self._get_monthly_usage(tenant_id, resource)

        status = {
            "resource": resource,
            "usage": usage,
            "limit": monthly_limit,
            "remaining": (monthly_limit - usage) if monthly_limit else None,
            "percentage": (usage / monthly_limit * 100) if monthly_limit else 0,
            "is_exceeded": usage >= monthly_limit if monthly_limit else False
        }

        return status
