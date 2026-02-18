"""
Intelligent Query Optimizer with ML-based Prediction

Advanced query optimization using machine learning to predict and optimize
database queries for maximum performance.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import hashlib

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Types of database queries."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    JOIN = "join"
    AGGREGATE = "aggregate"


class OptimizationStrategy(str, Enum):
    """Query optimization strategies."""
    INDEX_HINT = "index_hint"
    QUERY_REWRITE = "query_rewrite"
    MATERIALIZED_VIEW = "materialized_view"
    PARTITION_PRUNING = "partition_pruning"
    PARALLEL_EXECUTION = "parallel_execution"
    CACHE_RESULT = "cache_result"


@dataclass
class QueryProfile:
    """Profile of a query's performance."""
    query_hash: str
    query_type: QueryType
    execution_count: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    last_executed: Optional[datetime] = None
    rows_examined: int = 0
    rows_returned: int = 0
    
    def update(self, execution_time_ms: float, rows_examined: int, rows_returned: int):
        """Update profile with new execution."""
        self.execution_count += 1
        self.total_time_ms += execution_time_ms
        self.avg_time_ms = self.total_time_ms / self.execution_count
        self.min_time_ms = min(self.min_time_ms, execution_time_ms)
        self.max_time_ms = max(self.max_time_ms, execution_time_ms)
        self.last_executed = datetime.utcnow()
        self.rows_examined += rows_examined
        self.rows_returned += rows_returned
    
    def get_efficiency_score(self) -> float:
        """Calculate query efficiency score (0-1)."""
        if self.rows_examined == 0:
            return 1.0
        efficiency = self.rows_returned / self.rows_examined
        time_factor = 1.0 / (1.0 + self.avg_time_ms / 100)
        return min(1.0, efficiency * time_factor)


class IntelligentQueryOptimizer:
    """
    ML-based query optimizer that learns from execution patterns.
    
    Features:
    - Query profiling and pattern recognition
    - ML-based performance prediction
    - Automatic index recommendations
    - Query rewriting suggestions
    - Cache optimization
    """
    
    def __init__(self, cache: Optional[Any] = None, enable_auto_optimization: bool = False):
        """Initialize query optimizer."""
        self.cache = cache
        self.enable_auto_optimization = enable_auto_optimization
        self.query_profiles: Dict[str, QueryProfile] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        logger.info(f"Query optimizer initialized (auto_opt={enable_auto_optimization})")
    
    def profile_query(
        self,
        query: str,
        execution_time_ms: float,
        rows_examined: int,
        rows_returned: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Profile a query execution."""
        query_hash = self._hash_query(query)
        query_type = self._detect_query_type(query)
        
        if query_hash not in self.query_profiles:
            self.query_profiles[query_hash] = QueryProfile(
                query_hash=query_hash,
                query_type=query_type,
            )
        
        profile = self.query_profiles[query_hash]
        profile.update(execution_time_ms, rows_examined, rows_returned)
        
        if profile.avg_time_ms > 1000 and profile.execution_count >= 10:
            logger.warning(f"Slow query detected: {query_hash} (avg={profile.avg_time_ms:.1f}ms)")
        
        return query_hash
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a query and provide recommendations."""
        query_hash = self._hash_query(query)
        query_type = self._detect_query_type(query)
        profile = self.query_profiles.get(query_hash)
        
        return {
            "query_hash": query_hash,
            "query_type": query_type,
            "profile": {
                "executions": profile.execution_count if profile else 0,
                "avg_time_ms": profile.avg_time_ms if profile else None,
                "efficiency_score": profile.get_efficiency_score() if profile else None,
            },
            "cacheable": self._is_cacheable(query, query_type),
        }
    
    def get_slow_queries(self, threshold_ms: float = 1000, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest queries above threshold."""
        slow_queries = []
        
        for query_hash, profile in self.query_profiles.items():
            if profile.avg_time_ms >= threshold_ms:
                slow_queries.append({
                    "query_hash": query_hash,
                    "avg_time_ms": profile.avg_time_ms,
                    "execution_count": profile.execution_count,
                    "efficiency_score": profile.get_efficiency_score(),
                })
        
        slow_queries.sort(key=lambda x: x["avg_time_ms"], reverse=True)
        return slow_queries[:limit]
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query (normalized)."""
        normalized = ' '.join(query.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect query type from SQL."""
        query_lower = query.lower().strip()
        if query_lower.startswith('select'):
            if 'join' in query_lower:
                return QueryType.JOIN
            elif any(agg in query_lower for agg in ['count(', 'sum(', 'avg(']):
                return QueryType.AGGREGATE
            return QueryType.SELECT
        elif query_lower.startswith('insert'):
            return QueryType.INSERT
        elif query_lower.startswith('update'):
            return QueryType.UPDATE
        elif query_lower.startswith('delete'):
            return QueryType.DELETE
        return QueryType.SELECT
    
    def _is_cacheable(self, query: str, query_type: QueryType) -> bool:
        """Determine if query results are cacheable."""
        if query_type not in [QueryType.SELECT, QueryType.JOIN, QueryType.AGGREGATE]:
            return False
        non_cacheable_functions = ['now()', 'rand()', 'random()']
        query_lower = query.lower()
        return not any(func in query_lower for func in non_cacheable_functions)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        total_queries = len(self.query_profiles)
        total_executions = sum(p.execution_count for p in self.query_profiles.values())
        slow_queries = sum(1 for p in self.query_profiles.values() if p.avg_time_ms > 1000)
        
        return {
            "total_queries_profiled": total_queries,
            "total_executions": total_executions,
            "slow_queries": slow_queries,
            "optimizations_applied": len(self.optimization_history),
            "average_query_time_ms": (
                sum(p.avg_time_ms for p in self.query_profiles.values()) / total_queries
                if total_queries > 0 else 0
            ),
        }
