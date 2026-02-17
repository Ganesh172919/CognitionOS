"""
Redis-based rate limiting for horizontal scalability.

Uses Redis for distributed rate limiting with sliding window algorithm.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
from uuid import UUID

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class RedisRateLimiter:
    """
    Redis-based rate limiter using sliding window algorithm.
    
    Supports horizontal scaling by using Redis as centralized state.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "ratelimit:",
    ):
        """
        Initialize Redis rate limiter.
        
        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for all rate limit keys
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._redis: Optional[aioredis.Redis] = None
    
    async def connect(self):
        """Establish Redis connection."""
        if self._redis is None:
            self._redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            logger.info(f"Redis rate limiter connected to {self.redis_url}")
    
    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            logger.info("Redis rate limiter connection closed")
    
    def _get_key(self, tenant_id: UUID, resource_key: str, window_start: datetime) -> str:
        """
        Generate Redis key for rate limit tracking.
        
        Args:
            tenant_id: Tenant UUID
            resource_key: Resource identifier (e.g., 'api_calls')
            window_start: Window start timestamp
            
        Returns:
            Redis key string
        """
        # Format: ratelimit:{tenant_id}:{resource}:{timestamp}
        timestamp = int(window_start.timestamp())
        return f"{self.key_prefix}{tenant_id}:{resource_key}:{timestamp}"
    
    async def check_rate_limit(
        self,
        tenant_id: UUID,
        resource_key: str,
        limit: int,
        window_seconds: int = 60,
    ) -> Tuple[bool, int, datetime]:
        """
        Check if request is within rate limit using Redis.
        
        Uses sliding window counter with automatic key expiration.
        
        Args:
            tenant_id: Tenant UUID
            resource_key: Resource being rate limited
            limit: Maximum requests allowed in window
            window_seconds: Time window in seconds
            
        Returns:
            Tuple of (is_allowed, current_count, reset_time)
        """
        await self.connect()
        
        now = datetime.utcnow()
        window_start = now.replace(second=0, microsecond=0)  # Align to minute
        reset_time = window_start + timedelta(seconds=window_seconds)
        
        key = self._get_key(tenant_id, resource_key, window_start)
        
        try:
            # Use Redis pipeline for atomic operations
            pipe = self._redis.pipeline()
            
            # Increment counter
            pipe.incr(key)
            
            # Set expiration (window + buffer for grace period)
            pipe.expire(key, window_seconds + 300)
            
            # Execute pipeline
            results = await pipe.execute()
            current_count = results[0]
            
            # Check if within limit
            is_allowed = current_count <= limit
            
            if not is_allowed:
                logger.warning(
                    f"Rate limit exceeded for tenant {tenant_id}, "
                    f"resource {resource_key}: {current_count}/{limit}"
                )
            
            return is_allowed, current_count, reset_time
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}", exc_info=True)
            # Fail open - allow request if Redis is down
            return True, 0, reset_time
    
    async def get_current_count(
        self,
        tenant_id: UUID,
        resource_key: str,
    ) -> int:
        """
        Get current request count for a tenant/resource.
        
        Args:
            tenant_id: Tenant UUID
            resource_key: Resource identifier
            
        Returns:
            Current request count
        """
        await self.connect()
        
        now = datetime.utcnow()
        window_start = now.replace(second=0, microsecond=0)
        key = self._get_key(tenant_id, resource_key, window_start)
        
        try:
            count = await self._redis.get(key)
            return int(count) if count else 0
        except Exception as e:
            logger.error(f"Failed to get rate limit count: {e}")
            return 0
    
    async def reset_limit(
        self,
        tenant_id: UUID,
        resource_key: str,
    ):
        """
        Reset rate limit counter for a tenant/resource.
        
        Args:
            tenant_id: Tenant UUID
            resource_key: Resource identifier
        """
        await self.connect()
        
        now = datetime.utcnow()
        window_start = now.replace(second=0, microsecond=0)
        key = self._get_key(tenant_id, resource_key, window_start)
        
        try:
            await self._redis.delete(key)
            logger.info(f"Reset rate limit for tenant {tenant_id}, resource {resource_key}")
        except Exception as e:
            logger.error(f"Failed to reset rate limit: {e}")
    
    async def cleanup_expired_keys(self):
        """
        Clean up expired rate limit keys (maintenance operation).
        
        Note: Redis automatically expires keys, but this can be used
        for manual cleanup if needed.
        """
        await self.connect()
        
        try:
            # Scan for keys matching our prefix
            cursor = 0
            deleted_count = 0
            
            while True:
                cursor, keys = await self._redis.scan(
                    cursor,
                    match=f"{self.key_prefix}*",
                    count=100
                )
                
                for key in keys:
                    # Check if key has TTL
                    ttl = await self._redis.ttl(key)
                    if ttl == -1:  # No expiration set
                        # Set default expiration
                        await self._redis.expire(key, 300)
                        deleted_count += 1
                
                if cursor == 0:
                    break
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} rate limit keys")
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


# Global rate limiter instance
_rate_limiter: Optional[RedisRateLimiter] = None


async def get_redis_rate_limiter(
    redis_url: str = "redis://localhost:6379/0"
) -> RedisRateLimiter:
    """
    Get or create global Redis rate limiter instance.
    
    Args:
        redis_url: Redis connection URL
        
    Returns:
        RedisRateLimiter instance
    """
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = RedisRateLimiter(redis_url=redis_url)
        await _rate_limiter.connect()
    
    return _rate_limiter
