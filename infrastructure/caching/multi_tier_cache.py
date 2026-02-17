"""
Multi-tier caching strategy with L1 (memory) + L2 (Redis) layers.

Provides high-performance caching with automatic failover and cache warming.
"""

import asyncio
import hashlib
import logging
import pickle
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Optional, Callable, Dict
from collections import OrderedDict

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache for L1 memory layer."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]['value']
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache with TTL."""
        async with self._lock:
            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self.cache.popitem(last=False)
            
            self.cache[key] = {
                'value': value,
                'expires_at': datetime.utcnow() + timedelta(seconds=ttl)
            }
            self.cache.move_to_end(key)
    
    async def delete(self, key: str):
        """Delete key from cache."""
        async with self._lock:
            self.cache.pop(key, None)
    
    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            self.cache.clear()


class MultiTierCache:
    """
    Multi-tier cache with L1 (memory) and L2 (Redis) layers.
    
    Features:
    - Automatic failover if Redis is unavailable
    - Cache warming from L2 to L1
    - TTL support at both layers
    - Namespace support for logical separation
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/1",
        l1_max_size: int = 1000,
        namespace: str = "cache",
    ):
        self.redis_url = redis_url
        self.namespace = namespace
        
        # L1 in-memory cache
        self.l1_cache = LRUCache(max_size=l1_max_size)
        
        # L2 Redis cache
        self._redis: Optional[aioredis.Redis] = None
        self._redis_available = False
    
    async def connect(self):
        """Connect to Redis for L2 cache."""
        try:
            self._redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # Use bytes for pickle
            )
            await self._redis.ping()
            self._redis_available = True
            logger.info(f"Multi-tier cache connected to Redis: {self.redis_url}")
        except Exception as e:
            logger.warning(f"Redis L2 cache unavailable: {e}. Using L1 only.")
            self._redis_available = False
    
    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            self._redis_available = False
    
    def _make_key(self, key: str) -> str:
        """Create namespaced cache key."""
        return f"{self.namespace}:{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (L1 first, then L2).
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        namespaced_key = self._make_key(key)
        
        # Try L1 first (fastest)
        value = await self.l1_cache.get(namespaced_key)
        if value is not None:
            logger.debug(f"L1 cache hit: {key}")
            return value
        
        # Try L2 (Redis) if available
        if self._redis_available:
            try:
                data = await self._redis.get(namespaced_key)
                if data:
                    value = pickle.loads(data)
                    # Warm L1 cache
                    await self.l1_cache.set(namespaced_key, value)
                    logger.debug(f"L2 cache hit: {key}")
                    return value
            except Exception as e:
                logger.warning(f"L2 cache read failed: {e}")
                self._redis_available = False
        
        logger.debug(f"Cache miss: {key}")
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 300,
        l1_only: bool = False,
    ):
        """
        Set value in cache (both L1 and L2).
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            l1_only: If True, only cache in L1 (for ephemeral data)
        """
        namespaced_key = self._make_key(key)
        
        # Always set in L1
        await self.l1_cache.set(namespaced_key, value, ttl)
        
        # Set in L2 if available and not L1-only
        if not l1_only and self._redis_available:
            try:
                data = pickle.dumps(value)
                await self._redis.setex(namespaced_key, ttl, data)
                logger.debug(f"Cached in L1+L2: {key}")
            except Exception as e:
                logger.warning(f"L2 cache write failed: {e}")
                self._redis_available = False
        else:
            logger.debug(f"Cached in L1 only: {key}")
    
    async def delete(self, key: str):
        """Delete key from both cache layers."""
        namespaced_key = self._make_key(key)
        
        # Delete from L1
        await self.l1_cache.delete(namespaced_key)
        
        # Delete from L2
        if self._redis_available:
            try:
                await self._redis.delete(namespaced_key)
            except Exception as e:
                logger.warning(f"L2 cache delete failed: {e}")
    
    async def clear(self, pattern: Optional[str] = None):
        """
        Clear cache entries.
        
        Args:
            pattern: Optional pattern to match keys (e.g., "user:*")
        """
        # Clear L1
        await self.l1_cache.clear()
        
        # Clear L2
        if self._redis_available and pattern:
            try:
                cursor = 0
                while True:
                    cursor, keys = await self._redis.scan(
                        cursor,
                        match=f"{self.namespace}:{pattern}",
                        count=100
                    )
                    if keys:
                        await self._redis.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.warning(f"L2 cache clear failed: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "l1_size": len(self.l1_cache.cache),
            "l1_max_size": self.l1_cache.max_size,
            "l2_available": self._redis_available,
        }
        
        if self._redis_available:
            try:
                info = await self._redis.info("memory")
                stats["l2_used_memory_mb"] = int(info.get("used_memory", 0)) / 1024 / 1024
            except Exception as e:
                logger.warning(f"Failed to get L2 stats: {e}")
        
        return stats


# Global cache instance
_cache: Optional[MultiTierCache] = None


async def get_cache(
    redis_url: str = "redis://localhost:6379/1",
    namespace: str = "cache",
) -> MultiTierCache:
    """Get or create global cache instance."""
    global _cache
    
    if _cache is None:
        _cache = MultiTierCache(redis_url=redis_url, namespace=namespace)
        await _cache.connect()
    
    return _cache


def cache_result(
    ttl: int = 300,
    key_prefix: str = "",
    l1_only: bool = False,
):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        l1_only: If True, only use L1 cache
    
    Example:
        @cache_result(ttl=600, key_prefix="user")
        async def get_user(user_id: str):
            return await db.get_user(user_id)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_parts = [key_prefix or func.__name__]
            
            # Add arguments to key
            for arg in args:
                key_parts.append(str(arg))
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}={v}")
            
            # Hash the key for consistent length
            key_str = ":".join(key_parts)
            cache_key = hashlib.md5(key_str.encode()).hexdigest()
            
            # Try to get from cache
            cache = await get_cache()
            cached_value = await cache.get(cache_key)
            
            if cached_value is not None:
                return cached_value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(cache_key, result, ttl=ttl, l1_only=l1_only)
            
            return result
        
        return wrapper
    return decorator
