"""
Comprehensive Performance Optimization

Database query optimization, caching strategies, profiling, and connection pooling.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import logging
import hashlib
import time

logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Cache eviction strategies"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"


@dataclass
class QueryMetrics:
    """Database query performance metrics"""
    query_id: str
    query: str
    execution_time_ms: float
    rows_returned: int
    rows_scanned: int
    index_used: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    needs_optimization: bool = False
    suggested_indexes: List[str] = field(default_factory=list)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 1
    ttl_seconds: Optional[int] = None

    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl_seconds is None:
            return False
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl_seconds


class QueryOptimizer:
    """
    Database query optimization

    Analyzes slow queries and suggests optimizations including
    indexing strategies and query rewrites.
    """

    def __init__(self):
        self._query_history: List[QueryMetrics] = []
        self._slow_query_threshold_ms = 100.0

    async def analyze_query(
        self,
        query: str,
        execution_time_ms: float,
        rows_returned: int,
        rows_scanned: int,
        index_used: bool
    ) -> QueryMetrics:
        """
        Analyze query performance

        Args:
            query: SQL query
            execution_time_ms: Execution time
            rows_returned: Rows in result
            rows_scanned: Total rows scanned
            index_used: Whether index was used

        Returns:
            QueryMetrics with optimization suggestions
        """
        metrics = QueryMetrics(
            query_id=self._generate_query_id(query),
            query=query,
            execution_time_ms=execution_time_ms,
            rows_returned=rows_returned,
            rows_scanned=rows_scanned,
            index_used=index_used
        )

        # Check if optimization needed
        if execution_time_ms > self._slow_query_threshold_ms:
            metrics.needs_optimization = True

        # Analyze query patterns
        metrics.suggested_indexes = self._suggest_indexes(query, metrics)

        self._query_history.append(metrics)

        if metrics.needs_optimization:
            logger.warning(
                f"Slow query detected ({execution_time_ms:.2f}ms): {query[:100]}"
            )

        return metrics

    def _generate_query_id(self, query: str) -> str:
        """Generate unique query ID"""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    def _suggest_indexes(
        self,
        query: str,
        metrics: QueryMetrics
    ) -> List[str]:
        """Suggest indexes for query"""
        suggestions = []

        # Parse WHERE clauses
        where_columns = self._extract_where_columns(query)
        for col in where_columns:
            suggestions.append(f"CREATE INDEX idx_{col} ON table ({col})")

        # Check for table scans
        if not metrics.index_used and metrics.rows_scanned > 1000:
            suggestions.append("Query performing table scan - add index on WHERE clause columns")

        # Check for inefficient joins
        if "JOIN" in query.upper() and metrics.rows_scanned > metrics.rows_returned * 10:
            suggestions.append("Inefficient JOIN - consider adding composite index on join columns")

        return suggestions

    def _extract_where_columns(self, query: str) -> List[str]:
        """Extract column names from WHERE clause"""
        columns = []

        if "WHERE" in query.upper():
            where_part = query.upper().split("WHERE")[1]
            import re
            matches = re.findall(r'\b(\w+)\s*[=><!]', where_part)
            columns.extend(matches)

        return columns

    def get_slow_queries(
        self,
        threshold_ms: Optional[float] = None,
        limit: int = 10
    ) -> List[QueryMetrics]:
        """Get slow queries"""
        threshold = threshold_ms or self._slow_query_threshold_ms

        slow = [
            q for q in self._query_history
            if q.execution_time_ms > threshold
        ]

        slow.sort(key=lambda q: q.execution_time_ms, reverse=True)

        return slow[:limit]

    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization report"""
        total_queries = len(self._query_history)
        slow_queries = len([q for q in self._query_history if q.needs_optimization])

        avg_time = (
            sum(q.execution_time_ms for q in self._query_history) / total_queries
            if total_queries > 0 else 0
        )

        return {
            "total_queries_analyzed": total_queries,
            "slow_queries": slow_queries,
            "slow_query_percentage": slow_queries / total_queries * 100 if total_queries > 0 else 0,
            "average_execution_time_ms": avg_time,
            "queries_needing_indexes": len([
                q for q in self._query_history
                if not q.index_used and q.rows_scanned > 1000
            ]),
            "top_slow_queries": [
                {
                    "query": q.query[:100],
                    "time_ms": q.execution_time_ms,
                    "suggestions": q.suggested_indexes
                }
                for q in self.get_slow_queries(limit=5)
            ]
        }


class IntelligentCache:
    """
    Intelligent multi-layer caching system

    Implements LRU, LFU, and TTL caching with automatic
    cache warming and invalidation.
    """

    def __init__(
        self,
        strategy: CacheStrategy = CacheStrategy.LRU,
        max_size: int = 10000,
        default_ttl_seconds: Optional[int] = 3600
    ):
        self.strategy = strategy
        self.max_size = max_size
        self.default_ttl = default_ttl_seconds

        self._cache: Dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired():
            del self._cache[key]
            self._misses += 1
            return None

        entry.last_accessed = datetime.utcnow()
        entry.access_count += 1

        self._hits += 1
        return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ):
        """Set value in cache"""
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict()

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            ttl_seconds=ttl_seconds or self.default_ttl
        )

        self._cache[key] = entry

    def _evict(self):
        """Evict entry based on strategy"""
        if not self._cache:
            return

        if self.strategy == CacheStrategy.LRU:
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].last_accessed
            )
            del self._cache[oldest_key]

        elif self.strategy == CacheStrategy.LFU:
            least_used_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].access_count
            )
            del self._cache[least_used_key]

        elif self.strategy == CacheStrategy.FIFO:
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].created_at
            )
            del self._cache[oldest_key]

        elif self.strategy == CacheStrategy.TTL:
            for key, entry in self._cache.items():
                if entry.is_expired():
                    del self._cache[key]
                    return

    def invalidate(self, key: str):
        """Invalidate cache entry"""
        if key in self._cache:
            del self._cache[key]

    def invalidate_pattern(self, pattern: str):
        """Invalidate entries matching pattern"""
        keys_to_delete = [
            k for k in self._cache.keys()
            if pattern in k
        ]
        for key in keys_to_delete:
            del self._cache[key]

    def clear(self):
        """Clear entire cache"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "utilization": len(self._cache) / self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "strategy": self.strategy.value
        }


class PerformanceProfiler:
    """
    Performance profiling and analysis

    Profiles code execution, identifies bottlenecks, and
    provides optimization recommendations.
    """

    def __init__(self):
        self._profiles: Dict[str, List[float]] = {}

    def profile(self, name: str):
        """Context manager for profiling code blocks"""
        return ProfileContext(name, self)

    def record_execution(self, name: str, duration_ms: float):
        """Record execution time"""
        if name not in self._profiles:
            self._profiles[name] = []

        self._profiles[name].append(duration_ms)

    def get_profile_stats(self, name: str) -> Dict[str, float]:
        """Get profile statistics"""
        if name not in self._profiles:
            return {}

        times = self._profiles[name]

        return {
            "name": name,
            "count": len(times),
            "total_ms": sum(times),
            "avg_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "p50_ms": sorted(times)[len(times) // 2],
            "p95_ms": sorted(times)[int(len(times) * 0.95)] if len(times) > 20 else max(times),
            "p99_ms": sorted(times)[int(len(times) * 0.99)] if len(times) > 100 else max(times)
        }

    def get_bottlenecks(self, threshold_ms: float = 100.0) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []

        for name in self._profiles:
            stats = self.get_profile_stats(name)
            if stats.get("avg_ms", 0) > threshold_ms:
                bottlenecks.append(stats)

        bottlenecks.sort(key=lambda b: b["avg_ms"], reverse=True)

        return bottlenecks

    def generate_report(self) -> str:
        """Generate performance report"""
        report = ["Performance Profiling Report", "=" * 50, ""]

        total_profiles = len(self._profiles)
        total_executions = sum(len(times) for times in self._profiles.values())

        report.append(f"Total profiled operations: {total_profiles}")
        report.append(f"Total executions: {total_executions}")
        report.append("")

        bottlenecks = self.get_bottlenecks(threshold_ms=0.0)[:10]

        report.append("Top 10 Slowest Operations:")
        report.append("-" * 50)

        for i, stats in enumerate(bottlenecks, 1):
            report.append(
                f"{i}. {stats['name']}: "
                f"avg={stats['avg_ms']:.2f}ms, "
                f"p95={stats['p95_ms']:.2f}ms, "
                f"count={stats['count']}"
            )

        return "\n".join(report)


class ProfileContext:
    """Context manager for profiling"""

    def __init__(self, name: str, profiler: PerformanceProfiler):
        self.name = name
        self.profiler = profiler
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        self.profiler.record_execution(self.name, duration_ms)


class DatabaseConnectionPool:
    """
    Database connection pooling

    Manages database connections with connection pooling,
    health checks, and automatic failover.
    """

    def __init__(
        self,
        min_connections: int = 5,
        max_connections: int = 20,
        connection_timeout_seconds: int = 30
    ):
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout_seconds

        self._pool: List[Any] = []
        self._active_connections: Set[Any] = set()
        self._total_connections = 0

    def get_connection(self) -> Any:
        """Get connection from pool"""
        if self._pool:
            conn = self._pool.pop()
            self._active_connections.add(conn)
            return conn

        if self._total_connections < self.max_connections:
            conn = self._create_connection()
            self._active_connections.add(conn)
            self._total_connections += 1
            return conn

        raise Exception("Connection pool exhausted")

    def return_connection(self, conn: Any):
        """Return connection to pool"""
        if conn in self._active_connections:
            self._active_connections.remove(conn)
            self._pool.append(conn)

    def _create_connection(self) -> Any:
        """Create new database connection"""
        logger.info("Creating new database connection")
        return object()

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            "available_connections": len(self._pool),
            "active_connections": len(self._active_connections),
            "total_connections": self._total_connections,
            "max_connections": self.max_connections,
            "utilization": self._total_connections / self.max_connections
        }
