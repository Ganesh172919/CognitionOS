"""
Distributed Cache Layer — CognitionOS

Multi-tier caching system with:
- In-memory LRU cache (L1)
- Distributed cache abstraction (L2 — Redis compatible)
- Cache-aside, read-through, write-through patterns
- TTL and eviction policies
- Cache invalidation with pub/sub
- Namespace isolation per tenant
- Cache warming strategies
- Performance metrics
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class EvictionPolicy(str, Enum):
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    FIFO = "fifo"


class CachePattern(str, Enum):
    CACHE_ASIDE = "cache_aside"
    READ_THROUGH = "read_through"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"


@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    namespace: str = ""
    tags: List[str] = field(default_factory=list)


class LRUCache:
    """In-memory LRU cache (L1)."""

    def __init__(self, *, max_size: int = 10000,
                 default_ttl: int = 300):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None

        # Check expiry
        if entry.expires_at and time.time() > entry.expires_at:
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.access_count += 1
        entry.last_accessed = time.time()
        self._hits += 1
        return entry.value

    def set(self, key: str, value: Any, *, ttl: Optional[int] = None,
              namespace: str = "", tags: Optional[List[str]] = None):
        ttl = ttl or self._default_ttl
        expires_at = time.time() + ttl if ttl > 0 else None

        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key].value = value
            self._cache[key].expires_at = expires_at
            return

        # Evict if full
        while len(self._cache) >= self._max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            self._evictions += 1

        self._cache[key] = CacheEntry(
            key=key, value=value, expires_at=expires_at,
            namespace=namespace, tags=tags or [],
        )

    def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def invalidate_by_tag(self, tag: str) -> int:
        to_remove = [
            k for k, v in self._cache.items() if tag in v.tags
        ]
        for k in to_remove:
            del self._cache[k]
        return len(to_remove)

    def invalidate_by_namespace(self, namespace: str) -> int:
        to_remove = [
            k for k, v in self._cache.items() if v.namespace == namespace
        ]
        for k in to_remove:
            del self._cache[k]
        return len(to_remove)

    def clear(self):
        self._cache.clear()

    def cleanup_expired(self) -> int:
        now = time.time()
        expired = [
            k for k, v in self._cache.items()
            if v.expires_at and v.expires_at <= now
        ]
        for k in expired:
            del self._cache[k]
        return len(expired)

    def get_stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(self._hits / max(total, 1) * 100, 1),
            "evictions": self._evictions,
        }


class DistributedCacheLayer:
    """
    Production multi-tier cache with namespace isolation,
    cache patterns, and comprehensive metrics.
    """

    def __init__(self, *, l1_size: int = 10000,
                 default_ttl: int = 300,
                 l2_adapter: Optional[Any] = None):
        self._l1 = LRUCache(max_size=l1_size, default_ttl=default_ttl)
        self._l2 = l2_adapter  # Redis adapter placeholder
        self._read_through_fns: Dict[str, Callable[..., Awaitable[Any]]] = {}
        self._write_through_fns: Dict[str, Callable[..., Awaitable[None]]] = {}
        self._invalidation_callbacks: List[Callable[[str], None]] = []
        self._warmup_keys: List[Dict[str, Any]] = []

    # ── Get / Set ──

    async def get(self, key: str, *, namespace: str = "") -> Optional[Any]:
        full_key = f"{namespace}:{key}" if namespace else key

        # L1
        value = self._l1.get(full_key)
        if value is not None:
            return value

        # L2
        if self._l2 and hasattr(self._l2, "get"):
            value = await self._l2.get(full_key)
            if value is not None:
                self._l1.set(full_key, value, namespace=namespace)
                return value

        return None

    async def set(self, key: str, value: Any, *,
                    ttl: Optional[int] = None,
                    namespace: str = "",
                    tags: Optional[List[str]] = None,
                    pattern: CachePattern = CachePattern.CACHE_ASIDE):
        full_key = f"{namespace}:{key}" if namespace else key

        # L1
        self._l1.set(full_key, value, ttl=ttl, namespace=namespace, tags=tags)

        # L2
        if self._l2 and hasattr(self._l2, "set"):
            await self._l2.set(full_key, value, ttl=ttl)

        # Write-through
        if pattern == CachePattern.WRITE_THROUGH:
            fn = self._write_through_fns.get(namespace, self._write_through_fns.get("*"))
            if fn:
                await fn(key, value)

    async def delete(self, key: str, *, namespace: str = ""):
        full_key = f"{namespace}:{key}" if namespace else key
        self._l1.delete(full_key)
        if self._l2 and hasattr(self._l2, "delete"):
            await self._l2.delete(full_key)
        for cb in self._invalidation_callbacks:
            cb(full_key)

    # ── Read-Through ──

    async def get_or_load(self, key: str, loader: Callable[..., Awaitable[Any]], *,
                            ttl: Optional[int] = None,
                            namespace: str = "") -> Any:
        """Get from cache, or load via loader function and cache."""
        value = await self.get(key, namespace=namespace)
        if value is not None:
            return value

        value = await loader()
        if value is not None:
            await self.set(key, value, ttl=ttl, namespace=namespace)
        return value

    # ── Registration ──

    def register_read_through(self, namespace: str,
                                fn: Callable[..., Awaitable[Any]]):
        self._read_through_fns[namespace] = fn

    def register_write_through(self, namespace: str,
                                 fn: Callable[..., Awaitable[None]]):
        self._write_through_fns[namespace] = fn

    def on_invalidation(self, callback: Callable[[str], None]):
        self._invalidation_callbacks.append(callback)

    # ── Bulk Operations ──

    async def invalidate_namespace(self, namespace: str) -> int:
        count = self._l1.invalidate_by_namespace(namespace)
        return count

    async def invalidate_tag(self, tag: str) -> int:
        count = self._l1.invalidate_by_tag(tag)
        return count

    # ── Warming ──

    async def warm(self, keys: List[Dict[str, Any]]):
        """Pre-populate cache with specified keys."""
        for spec in keys:
            key = spec.get("key", "")
            namespace = spec.get("namespace", "")
            fn = self._read_through_fns.get(namespace)
            if fn:
                try:
                    value = await fn(key)
                    if value is not None:
                        await self.set(key, value, namespace=namespace,
                                        ttl=spec.get("ttl"))
                except Exception as exc:
                    logger.warning("Cache warm failed for %s: %s", key, exc)

    # ── Cache Key Generation ──

    @staticmethod
    def make_key(*parts: str) -> str:
        return ":".join(str(p) for p in parts)

    @staticmethod
    def hash_key(*parts: str) -> str:
        raw = ":".join(str(p) for p in parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ── Stats ──

    def get_stats(self) -> Dict[str, Any]:
        return {
            "l1": self._l1.get_stats(),
            "l2_connected": self._l2 is not None,
            "read_through_namespaces": len(self._read_through_fns),
            "write_through_namespaces": len(self._write_through_fns),
        }


# ── Singleton ──
_cache: Optional[DistributedCacheLayer] = None


def get_cache_layer(**kwargs) -> DistributedCacheLayer:
    global _cache
    if not _cache:
        _cache = DistributedCacheLayer(**kwargs)
    return _cache
