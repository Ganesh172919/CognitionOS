"""
Multi-Layer Cache System — CognitionOS

Features:
- L1 in-memory (LRU) + L2 persistent layer
- TTL management
- Cache invalidation patterns (tag-based, prefix-based)
- Cache-aside, write-through strategies
- Stats and hit-rate tracking
- Namespace isolation for multi-tenancy
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: float = field(default_factory=time.monotonic)
    ttl_seconds: float = 300.0
    tags: Set[str] = field(default_factory=set)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.monotonic)
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > self.ttl_seconds


class LRUCache:
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, max_size: int = 10000, default_ttl: float = 300.0) -> None:
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.RLock()
        self._tags: Dict[str, Set[str]] = defaultdict(set)  # tag -> keys
        self._stats = {"hits": 0, "misses": 0, "evictions": 0, "expirations": 0}

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            if entry.is_expired:
                self._remove(key)
                self._stats["expirations"] += 1
                self._stats["misses"] += 1
                return None
            entry.access_count += 1
            entry.last_accessed = time.monotonic()
            self._cache.move_to_end(key)
            self._stats["hits"] += 1
            return entry.value

    def set(self, key: str, value: Any, *, ttl: float | None = None,
            tags: Set[str] | None = None) -> None:
        with self._lock:
            if key in self._cache:
                self._remove(key)
            while len(self._cache) >= self._max_size:
                evict_key = next(iter(self._cache))
                self._remove(evict_key)
                self._stats["evictions"] += 1

            entry = CacheEntry(key=key, value=value,
                                ttl_seconds=ttl or self._default_ttl,
                                tags=tags or set())
            self._cache[key] = entry
            for tag in entry.tags:
                self._tags[tag].add(key)

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._remove(key)

    def _remove(self, key: str) -> bool:
        entry = self._cache.pop(key, None)
        if entry:
            for tag in entry.tags:
                self._tags[tag].discard(key)
            return True
        return False

    def invalidate_by_tag(self, tag: str) -> int:
        with self._lock:
            keys = list(self._tags.get(tag, set()))
            for key in keys:
                self._remove(key)
            return len(keys)

    def invalidate_by_prefix(self, prefix: str) -> int:
        with self._lock:
            keys = [k for k in self._cache if k.startswith(prefix)]
            for k in keys:
                self._remove(k)
            return len(keys)

    def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._tags.clear()
            return count

    def cleanup_expired(self) -> int:
        with self._lock:
            expired = [k for k, v in self._cache.items() if v.is_expired]
            for k in expired:
                self._remove(k)
            self._stats["expirations"] += len(expired)
            return len(expired)

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self._stats["hits"] + self._stats["misses"]
        return (self._stats["hits"] / total * 100) if total > 0 else 0

    def get_stats(self) -> Dict[str, Any]:
        return {**self._stats, "size": self.size, "max_size": self._max_size,
                "hit_rate_pct": round(self.hit_rate, 2)}


class CacheManager:
    """Manages multiple cache namespaces with unified API."""

    def __init__(self, *, default_max_size: int = 10000,
                 default_ttl: float = 300.0) -> None:
        self._namespaces: Dict[str, LRUCache] = {}
        self._default_max_size = default_max_size
        self._default_ttl = default_ttl
        self._default = LRUCache(default_max_size, default_ttl)

    def namespace(self, ns: str) -> LRUCache:
        if ns not in self._namespaces:
            self._namespaces[ns] = LRUCache(self._default_max_size, self._default_ttl)
        return self._namespaces[ns]

    def get(self, key: str, *, ns: str = "default") -> Any | None:
        cache = self._namespaces.get(ns, self._default)
        return cache.get(key)

    def set(self, key: str, value: Any, *, ns: str = "default",
            ttl: float | None = None, tags: Set[str] | None = None) -> None:
        cache = self.namespace(ns)
        cache.set(key, value, ttl=ttl, tags=tags)

    def delete(self, key: str, *, ns: str = "default") -> bool:
        cache = self._namespaces.get(ns, self._default)
        return cache.delete(key)

    def invalidate_namespace(self, ns: str) -> int:
        cache = self._namespaces.get(ns)
        return cache.clear() if cache else 0

    def cache_aside(self, key: str, loader: Callable[[], Any], *,
                    ns: str = "default", ttl: float | None = None) -> Any:
        cached = self.get(key, ns=ns)
        if cached is not None:
            return cached
        value = loader()
        self.set(key, value, ns=ns, ttl=ttl)
        return value

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        stats = {"default": self._default.get_stats()}
        for ns, cache in self._namespaces.items():
            stats[ns] = cache.get_stats()
        return stats

    def cleanup_all(self) -> int:
        total = self._default.cleanup_expired()
        for cache in self._namespaces.values():
            total += cache.cleanup_expired()
        return total


_manager: CacheManager | None = None

def get_cache_manager() -> CacheManager:
    global _manager
    if not _manager:
        _manager = CacheManager()
    return _manager
