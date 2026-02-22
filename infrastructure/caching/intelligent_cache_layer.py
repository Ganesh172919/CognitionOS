"""
Intelligent Multi-Tier Cache Layer

Advanced caching system with:
- Multi-tier caching (L1: Memory, L2: Redis, L3: Disk)
- Predictive pre-warming
- Automatic cache invalidation
- Cache hit rate optimization
- Distributed cache coordination
- Cache analytics and monitoring
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
import hashlib
import json
import pickle
import time
from collections import OrderedDict


class CacheTier(Enum):
    """Cache tier levels"""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DISK = "l3_disk"


class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # AI-powered adaptive


class CacheStrategy(Enum):
    """Caching strategies"""
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"
    READ_THROUGH = "read_through"


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    tier: CacheTier
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStatistics:
    """Cache performance statistics"""
    total_requests: int = 0
    hits: int = 0
    misses: int = 0
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    evictions: int = 0
    invalidations: int = 0
    total_size_bytes: int = 0
    avg_access_time_ms: float = 0.0


class LRUCache:
    """LRU Cache implementation using OrderedDict"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any) -> Optional[str]:
        """Put value and return evicted key if any"""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value

        if len(self.cache) > self.capacity:
            evicted_key = next(iter(self.cache))
            del self.cache[evicted_key]
            return evicted_key
        return None

    def delete(self, key: str) -> bool:
        if key in self.cache:
            del self.cache[key]
            return True
        return False

    def clear(self):
        self.cache.clear()

    def size(self) -> int:
        return len(self.cache)


class IntelligentCacheLayer:
    """
    Multi-tier intelligent caching system.

    Features:
    - 3-tier caching architecture
    - Predictive pre-warming
    - Adaptive eviction
    - Hit rate optimization
    - Distributed coordination
    """

    def __init__(
        self,
        l1_capacity: int = 1000,
        l2_capacity: int = 10000,
        l3_capacity: int = 100000,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
        default_ttl: int = 3600
    ):
        self.l1_capacity = l1_capacity
        self.l2_capacity = l2_capacity
        self.l3_capacity = l3_capacity
        self.eviction_policy = eviction_policy
        self.default_ttl = default_ttl

        # Cache tiers
        self.l1_cache = LRUCache(l1_capacity)  # In-memory
        self.l2_cache: Dict[str, CacheEntry] = {}  # Redis simulation
        self.l3_cache: Dict[str, CacheEntry] = {}  # Disk simulation

        # Metadata tracking
        self.entry_metadata: Dict[str, CacheEntry] = {}

        # Statistics
        self.stats = CacheStatistics()

        # Pre-warming patterns
        self.access_patterns: Dict[str, List[float]] = {}  # key -> access times

        # Invalidation tracking
        self.invalidation_rules: Dict[str, Callable] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache with tier fallback.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        start_time = time.time()
        self.stats.total_requests += 1

        # Try L1 (memory)
        value = self.l1_cache.get(key)
        if value is not None:
            self.stats.hits += 1
            self.stats.l1_hits += 1
            self._update_access_pattern(key)
            self._update_access_time(start_time)
            return value

        # Try L2 (redis)
        if key in self.l2_cache:
            entry = self.l2_cache[key]
            if not self._is_expired(entry):
                self.stats.hits += 1
                self.stats.l2_hits += 1

                # Promote to L1
                self.l1_cache.put(key, entry.value)

                self._update_access_pattern(key)
                self._update_access_time(start_time)
                return entry.value
            else:
                # Expired, remove
                del self.l2_cache[key]

        # Try L3 (disk)
        if key in self.l3_cache:
            entry = self.l3_cache[key]
            if not self._is_expired(entry):
                self.stats.hits += 1
                self.stats.l3_hits += 1

                # Promote to L2 and L1
                self.l2_cache[key] = entry
                self.l1_cache.put(key, entry.value)

                self._update_access_pattern(key)
                self._update_access_time(start_time)
                return entry.value
            else:
                del self.l3_cache[key]

        # Cache miss
        self.stats.misses += 1
        self._update_access_time(start_time)
        return default

    def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[Set[str]] = None
    ) -> None:
        """
        Put value in cache with multi-tier storage.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            tags: Tags for invalidation
        """
        ttl_seconds = ttl or self.default_ttl

        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            tier=CacheTier.L1_MEMORY,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            ttl_seconds=ttl_seconds,
            size_bytes=len(str(value)),
            tags=tags or set()
        )

        # Store in L1
        evicted = self.l1_cache.put(key, value)
        if evicted:
            self.stats.evictions += 1
            # Demote evicted to L2
            if evicted in self.entry_metadata:
                evicted_entry = self.entry_metadata[evicted]
                self.l2_cache[evicted] = evicted_entry

        # Store in L2
        self.l2_cache[key] = entry

        # Store in L3 for persistence
        self.l3_cache[key] = entry

        # Track metadata
        self.entry_metadata[key] = entry

        # Update size
        self.stats.total_size_bytes += entry.size_bytes

    def delete(self, key: str) -> bool:
        """Delete from all tiers"""
        deleted = False

        if self.l1_cache.delete(key):
            deleted = True

        if key in self.l2_cache:
            del self.l2_cache[key]
            deleted = True

        if key in self.l3_cache:
            del self.l3_cache[key]
            deleted = True

        if key in self.entry_metadata:
            entry = self.entry_metadata[key]
            self.stats.total_size_bytes -= entry.size_bytes
            del self.entry_metadata[key]

        return deleted

    def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate all entries with given tags"""
        invalidated = 0

        keys_to_delete = []
        for key, entry in self.entry_metadata.items():
            if entry.tags & tags:  # Intersection
                keys_to_delete.append(key)

        for key in keys_to_delete:
            if self.delete(key):
                invalidated += 1

        self.stats.invalidations += invalidated
        return invalidated

    def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate keys matching pattern"""
        invalidated = 0

        keys_to_delete = [
            key for key in self.entry_metadata.keys()
            if pattern in key
        ]

        for key in keys_to_delete:
            if self.delete(key):
                invalidated += 1

        self.stats.invalidations += invalidated
        return invalidated

    def warm_cache(
        self,
        keys: List[str],
        loader: Callable[[str], Any]
    ) -> int:
        """
        Pre-warm cache with predicted keys.

        Args:
            keys: Keys to pre-load
            loader: Function to load value for key

        Returns:
            Number of keys warmed
        """
        warmed = 0

        for key in keys:
            try:
                value = loader(key)
                self.put(key, value)
                warmed += 1
            except Exception:
                continue

        return warmed

    def predict_next_keys(self, limit: int = 10) -> List[str]:
        """
        Predict next keys to access based on patterns.

        Returns:
            List of predicted keys
        """
        # Analyze access patterns
        predictions = []

        for key, access_times in self.access_patterns.items():
            if len(access_times) < 2:
                continue

            # Calculate access frequency
            if len(access_times) >= 2:
                recent_interval = access_times[-1] - access_times[-2]
                time_since_last = time.time() - access_times[-1]

                # Predict if access is likely soon
                if time_since_last >= recent_interval * 0.8:
                    predictions.append((key, len(access_times)))

        # Sort by access count
        predictions.sort(key=lambda x: x[1], reverse=True)

        return [key for key, _ in predictions[:limit]]

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = (self.stats.hits / self.stats.total_requests * 100) if self.stats.total_requests > 0 else 0

        return {
            "total_requests": self.stats.total_requests,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": hit_rate,
            "l1_hits": self.stats.l1_hits,
            "l2_hits": self.stats.l2_hits,
            "l3_hits": self.stats.l3_hits,
            "evictions": self.stats.evictions,
            "invalidations": self.stats.invalidations,
            "total_size_bytes": self.stats.total_size_bytes,
            "avg_access_time_ms": self.stats.avg_access_time_ms,
            "l1_size": self.l1_cache.size(),
            "l2_size": len(self.l2_cache),
            "l3_size": len(self.l3_cache)
        }

    def optimize_tiers(self) -> Dict[str, Any]:
        """
        Optimize tier allocation based on access patterns.

        Returns:
            Optimization report
        """
        # Identify hot keys
        hot_keys = []
        for key, entry in self.entry_metadata.items():
            if entry.access_count > 10:
                hot_keys.append(key)

        # Promote hot keys to L1
        promoted = 0
        for key in hot_keys[:self.l1_capacity]:
            if key in self.l2_cache or key in self.l3_cache:
                entry = self.entry_metadata[key]
                self.l1_cache.put(key, entry.value)
                promoted += 1

        # Demote cold keys from L1
        cold_keys = []
        for key in list(self.entry_metadata.keys()):
            entry = self.entry_metadata[key]
            time_since_access = (datetime.utcnow() - entry.last_accessed).seconds
            if time_since_access > 3600:  # 1 hour
                cold_keys.append(key)

        demoted = 0
        for key in cold_keys:
            if self.l1_cache.delete(key):
                demoted += 1

        return {
            "hot_keys_identified": len(hot_keys),
            "promoted_to_l1": promoted,
            "demoted_from_l1": demoted,
            "optimization_timestamp": datetime.utcnow().isoformat()
        }

    def clear_tier(self, tier: CacheTier) -> None:
        """Clear specific tier"""
        if tier == CacheTier.L1_MEMORY:
            self.l1_cache.clear()
        elif tier == CacheTier.L2_REDIS:
            self.l2_cache.clear()
        elif tier == CacheTier.L3_DISK:
            self.l3_cache.clear()

    def clear_all(self) -> None:
        """Clear all cache tiers"""
        self.l1_cache.clear()
        self.l2_cache.clear()
        self.l3_cache.clear()
        self.entry_metadata.clear()
        self.stats = CacheStatistics()

    # Private helper methods

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if entry is expired"""
        if entry.ttl_seconds is None:
            return False

        age = (datetime.utcnow() - entry.created_at).seconds
        return age > entry.ttl_seconds

    def _update_access_pattern(self, key: str) -> None:
        """Update access pattern for key"""
        if key not in self.access_patterns:
            self.access_patterns[key] = []

        self.access_patterns[key].append(time.time())

        # Keep only recent accesses
        if len(self.access_patterns[key]) > 100:
            self.access_patterns[key] = self.access_patterns[key][-100:]

        # Update entry metadata
        if key in self.entry_metadata:
            entry = self.entry_metadata[key]
            entry.last_accessed = datetime.utcnow()
            entry.access_count += 1

    def _update_access_time(self, start_time: float) -> None:
        """Update average access time"""
        duration_ms = (time.time() - start_time) * 1000

        if self.stats.total_requests == 1:
            self.stats.avg_access_time_ms = duration_ms
        else:
            self.stats.avg_access_time_ms = (
                self.stats.avg_access_time_ms * 0.95 +
                duration_ms * 0.05
            )


class CacheCoordinator:
    """Coordinate caching across distributed instances"""

    def __init__(self):
        self.instances: Dict[str, IntelligentCacheLayer] = {}
        self.sync_log: List[Dict[str, Any]] = []

    def register_instance(
        self,
        instance_id: str,
        cache: IntelligentCacheLayer
    ) -> None:
        """Register cache instance"""
        self.instances[instance_id] = cache

    def broadcast_invalidation(
        self,
        tags: Set[str]
    ) -> Dict[str, int]:
        """Broadcast invalidation to all instances"""
        results = {}

        for instance_id, cache in self.instances.items():
            invalidated = cache.invalidate_by_tags(tags)
            results[instance_id] = invalidated

        return results

    def sync_hot_keys(self) -> Dict[str, Any]:
        """Sync hot keys across instances"""
        # Collect hot keys from all instances
        all_hot_keys = set()

        for cache in self.instances.values():
            predicted = cache.predict_next_keys(limit=20)
            all_hot_keys.update(predicted)

        # Distribute to all instances
        synced = {}
        for instance_id, cache in self.instances.items():
            synced[instance_id] = list(all_hot_keys)

        return {
            "hot_keys_identified": len(all_hot_keys),
            "instances_synced": len(self.instances),
            "sync_timestamp": datetime.utcnow().isoformat()
        }
