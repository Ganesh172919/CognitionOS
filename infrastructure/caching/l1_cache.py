"""
L1 In-Memory Cache - Fast local cache for tenant/subscription lookups.

Thread-safe LRU cache with tenant-scoped keys.
"""

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


def _tenant_key(prefix: str, tenant_id: UUID, *suffixes: str) -> str:
    """Build tenant-scoped cache key."""
    parts = [prefix, str(tenant_id)] + list(suffixes)
    return ":".join(parts)


class L1Cache:
    """
    In-memory LRU cache with tenant-aware keys.
    Suitable for tenant/subscription lookups.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._lock = asyncio.Lock()

    def _evict_if_needed(self, key: str) -> None:
        if len(self._cache) >= self._max_size and key not in self._cache:
            self._cache.popitem(last=False)

    def _expired(self, entry: dict) -> bool:
        expires = entry.get("expires_at")
        if expires is None:
            return False
        return time.monotonic() > expires

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key not in self._cache:
                return None
            entry = self._cache[key]
            if self._expired(entry):
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return entry["value"]

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set value in cache."""
        ttl = ttl if ttl is not None else self._default_ttl
        async with self._lock:
            self._evict_if_needed(key)
            self._cache[key] = {
                "value": value,
                "expires_at": time.monotonic() + ttl,
            }
            self._cache.move_to_end(key)

    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        async with self._lock:
            self._cache.pop(key, None)

    async def delete_tenant(self, tenant_id: UUID, prefix: str = "tenant") -> int:
        """Delete all keys for a tenant (prefix:tenant_id:*). Returns count deleted."""
        prefix_key = f"{prefix}:{tenant_id}"
        count = 0
        async with self._lock:
            keys_to_del = [k for k in self._cache if k.startswith(prefix_key)]
            for k in keys_to_del:
                self._cache.pop(k, None)
                count += 1
        return count

    async def clear(self) -> None:
        """Clear all entries."""
        async with self._lock:
            self._cache.clear()

    @staticmethod
    def tenant_subscription_key(tenant_id: UUID) -> str:
        """Cache key for tenant subscription lookup."""
        return _tenant_key("subscription", tenant_id)

    @staticmethod
    def tenant_key(tenant_id: UUID, slug: Optional[str] = None) -> str:
        """Cache key for tenant lookup."""
        if slug:
            return f"tenant:slug:{slug}"
        return f"tenant:id:{tenant_id}"
