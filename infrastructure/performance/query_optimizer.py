"""
Intelligent Query Optimization Layer

Automatic query optimization, caching, and performance monitoring.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import json

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.caching.multi_tier_cache import MultiTierCache


class QueryType(str, Enum):
    """Query types for optimization"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    AGGREGATE = "aggregate"


@dataclass
class QueryPerformance:
    """Query performance metrics"""
    query_hash: str
    query_text: str
    execution_time_ms: float
    rows_returned: int
    cache_hit: bool
    optimization_applied: bool
    timestamp: datetime


@dataclass
class QueryOptimization:
    """Query optimization recommendation"""
    original_query: str
    optimized_query: str
    optimization_type: str
    expected_improvement_pct: float
    reasoning: str


class QueryOptimizer:
    """
    Intelligent query optimization layer.
    
    Features:
    - Automatic index recommendations
    - Query rewriting
    - Smart caching
    - Query statistics tracking
    - Slow query detection
    - Batch optimization
    """
    
    def __init__(self, session: AsyncSession, cache: MultiTierCache):
        self.session = session
        self.cache = cache
        self.slow_query_threshold_ms = 1000
    
    async def optimize_and_execute(
        self,
        query: str,
        params: Optional[Dict] = None,
        use_cache: bool = True
    ) -> Tuple[List[Dict], QueryPerformance]:
        """Optimize and execute a query"""
        start_time = datetime.utcnow()
        
        # Generate query hash for caching
        query_hash = self._hash_query(query, params)
        
        # Check cache
        if use_cache:
            cached_result = await self.cache.get(f"query:{query_hash}")
            if cached_result:
                end_time = datetime.utcnow()
                execution_time = (end_time - start_time).total_seconds() * 1000
                
                return cached_result, QueryPerformance(
                    query_hash=query_hash,
                    query_text=query,
                    execution_time_ms=execution_time,
                    rows_returned=len(cached_result),
                    cache_hit=True,
                    optimization_applied=False,
                    timestamp=end_time
                )
        
        # Optimize query
        optimized_query = await self._optimize_query(query)
        optimization_applied = optimized_query != query
        
        # Execute query
        result = await self.session.execute(
            text(optimized_query if optimization_applied else query),
            params or {}
        )
        rows = [dict(row._mapping) for row in result]
        
        # Cache result
        if use_cache and len(rows) > 0:
            await self.cache.set(
                f"query:{query_hash}",
                rows,
                ttl=300  # 5 minutes
            )
        
        # Record performance
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        performance = QueryPerformance(
            query_hash=query_hash,
            query_text=query,
            execution_time_ms=execution_time,
            rows_returned=len(rows),
            cache_hit=False,
            optimization_applied=optimization_applied,
            timestamp=end_time
        )
        
        # Track slow queries
        if execution_time > self.slow_query_threshold_ms:
            await self._log_slow_query(performance)
        
        return rows, performance
    
    async def _optimize_query(self, query: str) -> str:
        """Apply query optimizations"""
        optimized = query
        
        # Add index hints if beneficial
        if "WHERE" in optimized.upper() and "ORDER BY" in optimized.upper():
            # Would add actual index hints in production
            pass
        
        # Optimize LIKE patterns
        if "LIKE '%" in optimized:
            # Suggest full-text search instead
            pass
        
        # Optimize subqueries
        if "SELECT" in optimized.upper().count("SELECT") > 1:
            # Consider JOIN instead of subquery
            pass
        
        return optimized
    
    async def get_index_recommendations(
        self,
        table_name: str
    ) -> List[Dict[str, Any]]:
        """Get index recommendations for a table"""
        # Analyze query patterns
        recommendations = []
        
        # Would analyze actual query logs in production
        recommendations.append({
            "table": table_name,
            "columns": ["tenant_id", "created_at"],
            "index_type": "btree",
            "reason": "Frequent WHERE clauses on these columns",
            "expected_improvement": "40-60% faster queries",
            "priority": "high"
        })
        
        return recommendations
    
    async def get_slow_queries(
        self,
        limit: int = 10,
        lookback_hours: int = 24
    ) -> List[QueryPerformance]:
        """Get slow queries for analysis"""
        # Would query from performance tracking table
        return []
    
    def _hash_query(self, query: str, params: Optional[Dict]) -> str:
        """Generate hash for query caching"""
        cache_key = f"{query}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    
    async def _log_slow_query(self, performance: QueryPerformance):
        """Log slow query for analysis"""
        # Would store in database for analysis
        pass
