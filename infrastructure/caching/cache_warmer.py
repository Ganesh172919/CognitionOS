"""
Distributed Cache Warming Strategies

Intelligent cache warming and preloading for optimal performance.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Callable
from enum import Enum
from dataclasses import dataclass, field
import hashlib

logger = logging.getLogger(__name__)


class WarmingStrategy(str, Enum):
    """Cache warming strategies."""
    EAGER = "eager"  # Warm immediately on startup
    LAZY = "lazy"  # Warm on first access
    PREDICTIVE = "predictive"  # Warm based on usage patterns
    SCHEDULED = "scheduled"  # Warm on schedule
    ADAPTIVE = "adaptive"  # Dynamically adjust based on metrics


class WarmingPriority(str, Enum):
    """Priority levels for cache warming."""
    CRITICAL = "critical"  # Must be warm (e.g., config)
    HIGH = "high"  # Should be warm (e.g., hot data)
    MEDIUM = "medium"  # Nice to warm
    LOW = "low"  # Optional


@dataclass
class WarmingPattern:
    """Pattern for cache warming."""
    name: str
    priority: WarmingPriority
    strategy: WarmingStrategy
    key_pattern: str  # Pattern to match keys
    data_loader: Callable  # Function to load data
    ttl: int = 3600  # TTL in seconds
    refresh_interval: int = 300  # Refresh every N seconds
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True
    
    def matches_key(self, key: str) -> bool:
        """Check if key matches this pattern."""
        import re
        return bool(re.match(self.key_pattern, key))


@dataclass
class WarmingStats:
    """Statistics for cache warming."""
    total_keys_warmed: int = 0
    successful_loads: int = 0
    failed_loads: int = 0
    total_time_seconds: float = 0.0
    last_warm_time: Optional[datetime] = None
    average_load_time: float = 0.0
    
    def update(self, success: bool, load_time: float):
        """Update statistics."""
        self.total_keys_warmed += 1
        if success:
            self.successful_loads += 1
        else:
            self.failed_loads += 1
        self.total_time_seconds += load_time
        self.last_warm_time = datetime.utcnow()
        self.average_load_time = self.total_time_seconds / self.total_keys_warmed


class DistributedCacheWarmer:
    """
    Distributed cache warming system.
    
    Features:
    - Multiple warming strategies
    - Priority-based warming
    - Predictive preloading based on patterns
    - Dependency resolution
    - Health monitoring
    """
    
    def __init__(
        self,
        cache: Any,  # Cache instance
        redis_client: Optional[Any] = None,  # For distributed coordination
    ):
        """
        Initialize cache warmer.
        
        Args:
            cache: Cache instance to warm
            redis_client: Optional Redis client for distributed coordination
        """
        self.cache = cache
        self.redis_client = redis_client
        
        self.patterns: Dict[str, WarmingPattern] = {}
        self.stats: Dict[str, WarmingStats] = {}
        self.warming_tasks: Dict[str, asyncio.Task] = {}
        self.access_patterns: Dict[str, List[datetime]] = {}
        
        self._running = False
        
        logger.info("Distributed cache warmer initialized")
    
    def register_pattern(self, pattern: WarmingPattern):
        """
        Register a cache warming pattern.
        
        Args:
            pattern: Warming pattern to register
        """
        self.patterns[pattern.name] = pattern
        self.stats[pattern.name] = WarmingStats()
        
        logger.info(f"Registered warming pattern: {pattern.name} "
                   f"(strategy={pattern.strategy}, priority={pattern.priority})")
    
    async def start(self):
        """Start cache warming."""
        if self._running:
            logger.warning("Cache warmer already running")
            return
        
        self._running = True
        logger.info("Starting cache warming")
        
        # Start warming patterns based on strategy
        for pattern_name, pattern in self.patterns.items():
            if not pattern.enabled:
                continue
            
            if pattern.strategy == WarmingStrategy.EAGER:
                # Warm immediately
                await self._warm_pattern(pattern)
            
            elif pattern.strategy == WarmingStrategy.SCHEDULED:
                # Schedule periodic warming
                task = asyncio.create_task(
                    self._scheduled_warming(pattern)
                )
                self.warming_tasks[pattern_name] = task
            
            elif pattern.strategy == WarmingStrategy.ADAPTIVE:
                # Start adaptive warming
                task = asyncio.create_task(
                    self._adaptive_warming(pattern)
                )
                self.warming_tasks[pattern_name] = task
        
        logger.info(f"Cache warmer started with {len(self.patterns)} patterns")
    
    async def stop(self):
        """Stop cache warming."""
        self._running = False
        
        # Cancel all warming tasks
        for task in self.warming_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.warming_tasks:
            await asyncio.gather(*self.warming_tasks.values(), return_exceptions=True)
        
        self.warming_tasks.clear()
        
        logger.info("Cache warmer stopped")
    
    async def warm_key(
        self,
        key: str,
        force: bool = False,
    ) -> bool:
        """
        Warm a specific cache key.
        
        Args:
            key: Cache key to warm
            force: Force warming even if already cached
            
        Returns:
            True if successfully warmed
        """
        # Find matching pattern
        pattern = self._find_pattern_for_key(key)
        if not pattern:
            logger.warning(f"No warming pattern found for key: {key}")
            return False
        
        # Check if already cached
        if not force:
            cached = await self.cache.get(key)
            if cached is not None:
                logger.debug(f"Key already cached: {key}")
                return True
        
        # Load data
        start_time = datetime.utcnow()
        try:
            data = await pattern.data_loader(key)
            await self.cache.set(key, data, ttl=pattern.ttl)
            
            load_time = (datetime.utcnow() - start_time).total_seconds()
            self.stats[pattern.name].update(success=True, load_time=load_time)
            
            logger.debug(f"Warmed key: {key} (time={load_time:.3f}s)")
            return True
            
        except Exception as e:
            load_time = (datetime.utcnow() - start_time).total_seconds()
            self.stats[pattern.name].update(success=False, load_time=load_time)
            
            logger.error(f"Failed to warm key {key}: {e}")
            return False
    
    async def warm_pattern(self, pattern_name: str) -> Dict[str, Any]:
        """
        Warm all keys matching a pattern.
        
        Args:
            pattern_name: Name of pattern to warm
            
        Returns:
            Warming result statistics
        """
        pattern = self.patterns.get(pattern_name)
        if not pattern:
            raise ValueError(f"Unknown warming pattern: {pattern_name}")
        
        return await self._warm_pattern(pattern)
    
    def track_access(self, key: str):
        """
        Track cache access for predictive warming.
        
        Args:
            key: Cache key that was accessed
        """
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        self.access_patterns[key].append(datetime.utcnow())
        
        # Keep only recent accesses (last hour)
        cutoff = datetime.utcnow() - timedelta(hours=1)
        self.access_patterns[key] = [
            ts for ts in self.access_patterns[key]
            if ts > cutoff
        ]
    
    async def predict_and_warm(self, count: int = 10):
        """
        Predict hot keys and warm them proactively.
        
        Args:
            count: Number of keys to warm
        """
        # Calculate access frequency
        key_scores = []
        for key, accesses in self.access_patterns.items():
            score = len(accesses)  # Simple frequency-based scoring
            key_scores.append((key, score))
        
        # Sort by score and warm top keys
        key_scores.sort(key=lambda x: x[1], reverse=True)
        hot_keys = [k for k, _ in key_scores[:count]]
        
        logger.info(f"Predictive warming {len(hot_keys)} hot keys")
        
        results = await asyncio.gather(
            *[self.warm_key(key) for key in hot_keys],
            return_exceptions=True
        )
        
        success_count = sum(1 for r in results if r is True)
        logger.info(f"Warmed {success_count}/{len(hot_keys)} predicted keys")
    
    async def _warm_pattern(self, pattern: WarmingPattern) -> Dict[str, Any]:
        """Warm all keys for a pattern."""
        logger.info(f"Warming pattern: {pattern.name}")
        
        start_time = datetime.utcnow()
        warmed_count = 0
        failed_count = 0
        
        # Get keys to warm (would integrate with data source)
        # For now, this is a placeholder
        keys_to_warm = await self._get_keys_for_pattern(pattern)
        
        # Warm keys with priority ordering
        for key in keys_to_warm:
            success = await self.warm_key(key)
            if success:
                warmed_count += 1
            else:
                failed_count += 1
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        result = {
            "pattern": pattern.name,
            "warmed": warmed_count,
            "failed": failed_count,
            "duration_seconds": duration,
        }
        
        logger.info(f"Pattern warming complete: {result}")
        
        return result
    
    async def _scheduled_warming(self, pattern: WarmingPattern):
        """Periodically warm a pattern."""
        while self._running:
            try:
                await self._warm_pattern(pattern)
                await asyncio.sleep(pattern.refresh_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduled warming failed for {pattern.name}: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _adaptive_warming(self, pattern: WarmingPattern):
        """Adaptively warm based on access patterns."""
        while self._running:
            try:
                # Analyze access patterns
                hot_keys = self._identify_hot_keys(pattern)
                
                # Warm hot keys
                for key in hot_keys:
                    await self.warm_key(key)
                
                # Adjust warming frequency based on hit rate
                await asyncio.sleep(pattern.refresh_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Adaptive warming failed for {pattern.name}: {e}")
                await asyncio.sleep(60)
    
    def _identify_hot_keys(self, pattern: WarmingPattern) -> List[str]:
        """Identify hot keys for a pattern."""
        # Find keys matching pattern
        hot_keys = []
        for key, accesses in self.access_patterns.items():
            if pattern.matches_key(key) and len(accesses) >= 5:  # Threshold
                hot_keys.append(key)
        
        return hot_keys
    
    async def _get_keys_for_pattern(self, pattern: WarmingPattern) -> List[str]:
        """Get keys to warm for a pattern."""
        # This would integrate with data sources
        # For now, return empty list as placeholder
        return []
    
    def _find_pattern_for_key(self, key: str) -> Optional[WarmingPattern]:
        """Find warming pattern for a key."""
        for pattern in self.patterns.values():
            if pattern.matches_key(key):
                return pattern
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get warming statistics."""
        return {
            "patterns": {
                name: {
                    "total_warmed": stats.total_keys_warmed,
                    "successful": stats.successful_loads,
                    "failed": stats.failed_loads,
                    "average_time": stats.average_load_time,
                    "last_warm": stats.last_warm_time.isoformat() if stats.last_warm_time else None,
                }
                for name, stats in self.stats.items()
            },
            "access_patterns_tracked": len(self.access_patterns),
            "active_tasks": len(self.warming_tasks),
            "running": self._running,
        }
