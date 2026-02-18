"""
Repository Caching Decorator

Provides easy-to-use caching decorators for repository methods.
"""

import hashlib
import json
import logging
from functools import wraps
from typing import Any, Callable, Optional

from infrastructure.caching.multi_tier_cache import MultiTierCache
from core.config import get_config

logger = logging.getLogger(__name__)

# Global cache instance
_cache_instance: Optional[MultiTierCache] = None


async def get_cache() -> MultiTierCache:
    """Get or create global cache instance."""
    global _cache_instance
    
    if _cache_instance is None:
        config = get_config()
        _cache_instance = MultiTierCache(
            redis_url=config.redis.url,
            l1_max_size=1000,
            namespace="repository",
        )
        await _cache_instance.initialize()
    
    return _cache_instance


def generate_cache_key(prefix: str, *args, **kwargs) -> str:
    """
    Generate cache key from function arguments.
    
    Args:
        prefix: Cache key prefix (usually function name)
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Cache key string
    """
    # Convert args to strings
    arg_strs = [str(arg) for arg in args if arg is not None]
    
    # Sort kwargs for consistent ordering
    kwarg_strs = [f"{k}={v}" for k, v in sorted(kwargs.items()) if v is not None]
    
    # Combine all parts
    key_parts = [prefix] + arg_strs + kwarg_strs
    key_str = ":".join(key_parts)
    
    # Use blake2b for fast non-cryptographic hashing
    key_hash = hashlib.blake2b(key_str.encode(), digest_size=16).hexdigest()
    
    return f"{prefix}:{key_hash}"


def cache_result(
    ttl: int = 300,
    key_prefix: Optional[str] = None,
    skip_self: bool = True,
):
    """
    Decorator to cache repository method results.
    
    Usage:
        @cache_result(ttl=600, key_prefix="workflow")
        async def get_by_id(self, workflow_id: UUID) -> Optional[Workflow]:
            # Method implementation
            pass
    
    Args:
        ttl: Time to live in seconds (default: 300 = 5 minutes)
        key_prefix: Custom cache key prefix (default: function name)
        skip_self: Skip 'self' parameter in cache key generation (default: True)
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            prefix = key_prefix or func.__name__
            
            # Skip 'self' parameter if requested
            cache_args = args[1:] if skip_self and len(args) > 0 else args
            cache_key = generate_cache_key(prefix, *cache_args, **kwargs)
            
            # Try to get from cache
            try:
                cache = await get_cache()
                cached_value = await cache.get(cache_key)
                
                if cached_value is not None:
                    logger.debug(
                        f"Cache hit for {func.__name__}",
                        extra={"cache_key": cache_key}
                    )
                    return cached_value
                
                logger.debug(
                    f"Cache miss for {func.__name__}",
                    extra={"cache_key": cache_key}
                )
                
            except Exception as e:
                logger.warning(
                    f"Cache get failed for {func.__name__}: {e}",
                    extra={"cache_key": cache_key}
                )
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result if not None
            if result is not None:
                try:
                    cache = await get_cache()
                    await cache.set(cache_key, result, ttl=ttl)
                    
                    logger.debug(
                        f"Cached result for {func.__name__}",
                        extra={"cache_key": cache_key, "ttl": ttl}
                    )
                    
                except Exception as e:
                    logger.warning(
                        f"Cache set failed for {func.__name__}: {e}",
                        extra={"cache_key": cache_key}
                    )
            
            return result
        
        return wrapper
    
    return decorator


def invalidate_cache(key_prefix: str, *args, **kwargs):
    """
    Invalidate cache entries matching the given prefix and parameters.
    
    Usage:
        await invalidate_cache("workflow", workflow_id=workflow_id)
    
    Args:
        key_prefix: Cache key prefix
        *args: Positional arguments
        **kwargs: Keyword arguments
    """
    async def _invalidate():
        cache_key = generate_cache_key(key_prefix, *args, **kwargs)
        
        try:
            cache = await get_cache()
            await cache.delete(cache_key)
            
            logger.debug(
                f"Invalidated cache",
                extra={"cache_key": cache_key}
            )
            
        except Exception as e:
            logger.warning(
                f"Cache invalidation failed: {e}",
                extra={"cache_key": cache_key}
            )
    
    return _invalidate()


def cache_on_success(
    ttl: int = 300,
    key_prefix: Optional[str] = None,
):
    """
    Decorator to cache results only on successful operations.
    
    Useful for write operations that should invalidate cache on success.
    
    Usage:
        @cache_on_success(ttl=600)
        async def save(self, workflow: Workflow) -> None:
            # Save implementation
            pass
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Custom cache key prefix
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # On successful write, invalidate related caches
            # Extract ID from args/kwargs if available
            entity_id = None
            if len(args) > 1:
                entity = args[1]
                if hasattr(entity, 'id'):
                    entity_id = entity.id
            
            if entity_id:
                prefix = key_prefix or func.__name__.replace('save', 'get_by_id')
                await invalidate_cache(prefix, entity_id)
            
            return result
        
        return wrapper
    
    return decorator


# Pre-configured decorators for common TTL values

# Short-lived cache (1 minute) - for frequently changing data
cache_short = lambda func: cache_result(ttl=60)(func)

# Standard cache (5 minutes) - for normal data
cache_standard = lambda func: cache_result(ttl=300)(func)

# Long-lived cache (1 hour) - for infrequently changing data
cache_long = lambda func: cache_result(ttl=3600)(func)

# Extended cache (24 hours) - for rarely changing data
cache_extended = lambda func: cache_result(ttl=86400)(func)
