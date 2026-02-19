"""
Adaptive ML-Based Caching System
Intelligent cache with ML-based eviction and prefetching.
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field


class CacheStrategy(str, Enum):
    """Cache eviction strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    ADAPTIVE = "adaptive"  # ML-based adaptive
    TTL = "ttl"  # Time To Live


class AccessPattern(str, Enum):
    """Access pattern types"""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    PERIODIC = "periodic"
    BURST = "burst"


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    size_bytes: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    hit_count: int = 0
    importance_score: float = 1.0  # ML-computed importance
    tags: Set[str] = field(default_factory=set)


class CacheMetrics(BaseModel):
    """Cache performance metrics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    current_size_bytes: int = 0
    max_size_bytes: int
    hit_rate: float = 0.0
    avg_access_time_ms: float = 0.0


class PredictionModel(BaseModel):
    """Simple prediction model for cache"""
    key_patterns: Dict[str, AccessPattern] = Field(default_factory=dict)
    access_frequencies: Dict[str, List[datetime]] = Field(default_factory=dict)
    correlation_matrix: Dict[str, List[str]] = Field(default_factory=dict)


class AdaptiveCacheSystem:
    """
    ML-based adaptive caching with intelligent eviction and prefetching.
    """

    def __init__(
        self,
        max_size_mb: int = 1000,
        default_ttl_seconds: int = 3600,
        enable_prefetch: bool = True
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl_seconds = default_ttl_seconds
        self.enable_prefetch = enable_prefetch

        self.cache: Dict[str, CacheEntry] = {}
        self.metrics = CacheMetrics(max_size_bytes=self.max_size_bytes)
        self.prediction_model = PredictionModel()

        # Access tracking
        self.access_history: List[Tuple[str, datetime]] = []
        self.key_relationships: Dict[str, Set[str]] = defaultdict(set)

        # Performance tracking
        self.access_times: List[float] = []

    async def get(
        self,
        key: str,
        tags: Optional[Set[str]] = None
    ) -> Optional[Any]:
        """
        Get value from cache with ML-based tracking
        """
        start_time = time.time()

        self.metrics.total_requests += 1

        # Track access
        self._track_access(key)

        # Check if key exists
        if key in self.cache:
            entry = self.cache[key]

            # Check TTL
            if self._is_expired(entry):
                await self._evict(key)
                self.metrics.cache_misses += 1
                elapsed_ms = (time.time() - start_time) * 1000
                self.access_times.append(elapsed_ms)
                return None

            # Update access metadata
            entry.last_accessed = datetime.utcnow()
            entry.access_count += 1
            entry.hit_count += 1

            # Update importance score
            entry.importance_score = self._calculate_importance(entry)

            self.metrics.cache_hits += 1

            # Trigger prefetch for related keys
            if self.enable_prefetch:
                await self._prefetch_related(key)

            elapsed_ms = (time.time() - start_time) * 1000
            self.access_times.append(elapsed_ms)

            return entry.value

        self.metrics.cache_misses += 1

        elapsed_ms = (time.time() - start_time) * 1000
        self.access_times.append(elapsed_ms)

        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[Set[str]] = None
    ) -> bool:
        """
        Set value in cache with adaptive management
        """
        # Calculate size (simplified)
        size_bytes = self._estimate_size(value)

        # Check if eviction needed
        while (
            self.metrics.current_size_bytes + size_bytes > self.max_size_bytes
            and len(self.cache) > 0
        ):
            await self._evict_adaptive()

        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            size_bytes=size_bytes,
            ttl_seconds=ttl_seconds or self.default_ttl_seconds,
            tags=tags or set()
        )

        # Update or insert
        if key in self.cache:
            old_entry = self.cache[key]
            self.metrics.current_size_bytes -= old_entry.size_bytes

        self.cache[key] = entry
        self.metrics.current_size_bytes += size_bytes

        # Learn relationships
        await self._learn_relationships(key)

        return True

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if key in self.cache:
            entry = self.cache[key]
            self.metrics.current_size_bytes -= entry.size_bytes
            del self.cache[key]
            return True
        return False

    async def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate all entries with matching tags"""
        count = 0
        keys_to_delete = []

        for key, entry in self.cache.items():
            if entry.tags & tags:  # Intersection
                keys_to_delete.append(key)

        for key in keys_to_delete:
            await self.delete(key)
            count += 1

        return count

    def _track_access(self, key: str) -> None:
        """Track access for ML learning"""
        now = datetime.utcnow()
        self.access_history.append((key, now))

        # Track frequency
        if key not in self.prediction_model.access_frequencies:
            self.prediction_model.access_frequencies[key] = []

        self.prediction_model.access_frequencies[key].append(now)

        # Keep only recent history (last 1000 accesses)
        if len(self.access_history) > 1000:
            self.access_history = self.access_history[-1000:]

        # Detect access pattern
        if len(self.prediction_model.access_frequencies[key]) >= 5:
            pattern = self._detect_access_pattern(key)
            self.prediction_model.key_patterns[key] = pattern

    def _detect_access_pattern(self, key: str) -> AccessPattern:
        """Detect access pattern using ML"""
        accesses = self.prediction_model.access_frequencies[key]

        if len(accesses) < 5:
            return AccessPattern.RANDOM

        # Calculate inter-arrival times
        intervals = []
        for i in range(1, len(accesses)):
            interval = (accesses[i] - accesses[i-1]).total_seconds()
            intervals.append(interval)

        # Calculate statistics
        avg_interval = sum(intervals) / len(intervals)
        variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = variance ** 0.5

        # Classify pattern
        if std_dev / avg_interval < 0.2:  # Low variance
            return AccessPattern.PERIODIC
        elif len(accesses) > 10 and all(i < 5 for i in intervals[-10:]):  # Recent burst
            return AccessPattern.BURST
        elif std_dev / avg_interval > 1.0:  # High variance
            return AccessPattern.RANDOM
        else:
            return AccessPattern.SEQUENTIAL

    async def _learn_relationships(self, key: str) -> None:
        """Learn relationships between keys"""
        # Look at recent access history
        recent_keys = [k for k, _ in self.access_history[-10:]]

        # Keys accessed together are likely related
        for other_key in recent_keys:
            if other_key != key:
                self.key_relationships[key].add(other_key)
                self.key_relationships[other_key].add(key)

                # Update correlation matrix
                if key not in self.prediction_model.correlation_matrix:
                    self.prediction_model.correlation_matrix[key] = []

                if other_key not in self.prediction_model.correlation_matrix[key]:
                    self.prediction_model.correlation_matrix[key].append(other_key)

    async def _prefetch_related(self, key: str) -> None:
        """Prefetch related keys (simplified - would fetch from DB in production)"""
        # Get related keys that might be accessed next
        related_keys = self.key_relationships.get(key, set())

        for related_key in list(related_keys)[:3]:  # Limit to top 3
            if related_key not in self.cache:
                # In production, would fetch from database
                # For now, just record the intent
                pass

    def _calculate_importance(self, entry: CacheEntry) -> float:
        """
        Calculate importance score for adaptive eviction
        Uses multiple signals: recency, frequency, access pattern
        """
        now = datetime.utcnow()

        # Recency factor (0-1, higher = more recent)
        time_since_access = (now - entry.last_accessed).total_seconds()
        recency_score = 1.0 / (1.0 + time_since_access / 3600)  # Half-life of 1 hour

        # Frequency factor (normalized by total accesses)
        frequency_score = min(1.0, entry.access_count / 100)

        # Hit rate factor
        hit_rate = entry.hit_count / max(entry.access_count, 1)

        # Pattern factor
        pattern = self.prediction_model.key_patterns.get(entry.key, AccessPattern.RANDOM)
        pattern_weights = {
            AccessPattern.PERIODIC: 1.2,  # Likely to be accessed again
            AccessPattern.BURST: 0.9,  # May not be accessed soon
            AccessPattern.SEQUENTIAL: 1.0,
            AccessPattern.RANDOM: 0.8
        }
        pattern_score = pattern_weights[pattern]

        # Combine scores (weighted average)
        importance = (
            0.3 * recency_score +
            0.3 * frequency_score +
            0.2 * hit_rate +
            0.2 * pattern_score
        )

        return importance

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if entry is expired"""
        if entry.ttl_seconds is None:
            return False

        age_seconds = (datetime.utcnow() - entry.created_at).total_seconds()
        return age_seconds > entry.ttl_seconds

    async def _evict_adaptive(self) -> None:
        """Evict entry using adaptive ML-based strategy"""
        if not self.cache:
            return

        # Find entry with lowest importance score
        min_importance = float('inf')
        key_to_evict = None

        for key, entry in self.cache.items():
            # Skip recently accessed entries
            if (datetime.utcnow() - entry.last_accessed).total_seconds() < 60:
                continue

            importance = entry.importance_score

            if importance < min_importance:
                min_importance = importance
                key_to_evict = key

        if key_to_evict:
            await self._evict(key_to_evict)

    async def _evict(self, key: str) -> None:
        """Evict specific key"""
        if key in self.cache:
            entry = self.cache[key]
            self.metrics.current_size_bytes -= entry.size_bytes
            del self.cache[key]
            self.metrics.evictions += 1

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes (simplified)"""
        import sys

        try:
            return sys.getsizeof(value)
        except:
            return 1024  # Default 1KB

    async def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        # Calculate hit rate
        if self.metrics.total_requests > 0:
            self.metrics.hit_rate = self.metrics.cache_hits / self.metrics.total_requests

        # Calculate average access time
        if self.access_times:
            self.metrics.avg_access_time_ms = sum(self.access_times) / len(self.access_times)

        return {
            "total_requests": self.metrics.total_requests,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "hit_rate": round(self.metrics.hit_rate * 100, 2),
            "evictions": self.metrics.evictions,
            "current_size_mb": round(self.metrics.current_size_bytes / (1024 * 1024), 2),
            "max_size_mb": round(self.metrics.max_size_bytes / (1024 * 1024), 2),
            "utilization_pct": round(
                (self.metrics.current_size_bytes / self.metrics.max_size_bytes) * 100, 2
            ),
            "avg_access_time_ms": round(self.metrics.avg_access_time_ms, 3),
            "total_keys": len(self.cache),
            "detected_patterns": len(self.prediction_model.key_patterns),
            "learned_relationships": sum(len(v) for v in self.key_relationships.values())
        }

    async def get_key_insights(self, key: str) -> Dict[str, Any]:
        """Get insights about a specific key"""
        if key not in self.cache:
            return {"error": "Key not found"}

        entry = self.cache[key]

        pattern = self.prediction_model.key_patterns.get(key, AccessPattern.RANDOM)
        related_keys = list(self.key_relationships.get(key, set()))

        return {
            "key": key,
            "size_bytes": entry.size_bytes,
            "age_seconds": (datetime.utcnow() - entry.created_at).total_seconds(),
            "access_count": entry.access_count,
            "hit_count": entry.hit_count,
            "importance_score": round(entry.importance_score, 3),
            "access_pattern": pattern.value,
            "related_keys": related_keys[:5],  # Top 5
            "tags": list(entry.tags)
        }

    async def optimize(self) -> Dict[str, Any]:
        """Run optimization pass"""
        optimizations = {
            "evicted_expired": 0,
            "evicted_low_value": 0
        }

        keys_to_check = list(self.cache.keys())

        for key in keys_to_check:
            entry = self.cache[key]

            # Evict expired entries
            if self._is_expired(entry):
                await self._evict(key)
                optimizations["evicted_expired"] += 1
                continue

            # Evict low-value entries if cache is > 90% full
            if self.metrics.current_size_bytes > self.max_size_bytes * 0.9:
                if entry.importance_score < 0.3:
                    await self._evict(key)
                    optimizations["evicted_low_value"] += 1

        return optimizations

    async def clear(self) -> int:
        """Clear entire cache"""
        count = len(self.cache)
        self.cache.clear()
        self.metrics.current_size_bytes = 0
        return count

    async def warm_cache(self, key_value_pairs: List[Tuple[str, Any]]) -> int:
        """Warm cache with initial data"""
        count = 0
        for key, value in key_value_pairs:
            await self.set(key, value)
            count += 1
        return count
