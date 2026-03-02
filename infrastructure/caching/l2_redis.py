"""
L2 Redis Cache - Distributed cache for tenant/subscription lookups.

Uses Redis for cross-process/cross-instance caching.
"""

import json
import logging
from typing import Any, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


class L2RedisCache:
    """
    Redis-backed cache layer.
    Uses RedisPoolManager for connection pooling.
    """

    def __init__(self, namespace: str = "cognitionos", default_ttl: int = 600):
        self._namespace = namespace
        self._default_ttl = default_ttl
        self._redis = None

    def _make_key(self, key: str) -> str:
        return f"{self._namespace}:{key}"

    async def _get_redis(self):
        if self._redis is None:
            from infrastructure.persistence.redis_pool import RedisPoolManager
            pool = await RedisPoolManager.get_instance()
            self._redis = pool.get_client()
        return self._redis

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        try:
            redis = await self._get_redis()
            full_key = self._make_key(key)
            raw = await redis.get(full_key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.warning("L2 cache get failed: %s", e)
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set value in Redis."""
        try:
            redis = await self._get_redis()
            full_key = self._make_key(key)
            ttl = ttl if ttl is not None else self._default_ttl
            await redis.setex(
                full_key,
                ttl,
                json.dumps(value, default=str),
            )
        except Exception as e:
            logger.warning("L2 cache set failed: %s", e)

    async def delete(self, key: str) -> None:
        """Delete key from Redis."""
        try:
            redis = await self._get_redis()
            await redis.delete(self._make_key(key))
        except Exception as e:
            logger.warning("L2 cache delete failed: %s", e)

    async def delete_tenant_keys(self, tenant_id: UUID, prefix: str = "tenant") -> int:
        """Delete all keys matching tenant pattern. Returns count deleted."""
        try:
            redis = await self._get_redis()
            pattern = self._make_key(f"{prefix}:{tenant_id}*")
            keys = []
            async for k in redis.scan_iter(match=pattern):
                keys.append(k)
            if keys:
                await redis.delete(*keys)
            return len(keys)
        except Exception as e:
            logger.warning("L2 cache delete_tenant_keys failed: %s", e)
            return 0
