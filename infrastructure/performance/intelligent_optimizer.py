"""
Intelligent Query Optimizer
ML-based query optimization with adaptive learning and performance tracking.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Types of queries"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    AGGREGATE = "aggregate"
    JOIN = "join"


class OptimizationStrategy(str, Enum):
    """Query optimization strategies"""
    INDEX_RECOMMENDATION = "index_recommendation"
    QUERY_REWRITE = "query_rewrite"
    CACHING = "caching"
    PARTITIONING = "partitioning"
    DENORMALIZATION = "denormalization"
    MATERIALIZED_VIEW = "materialized_view"


@dataclass
class QueryPattern:
    """Detected query pattern"""
    pattern_id: str
    pattern_signature: str
    query_type: QueryType
    tables: Set[str] = field(default_factory=set)
    columns: Set[str] = field(default_factory=set)
    where_conditions: List[str] = field(default_factory=list)
    frequency: int = 0
    avg_execution_time_ms: float = 0.0
    last_seen: datetime = field(default_factory=datetime.utcnow)


class QueryExecution(BaseModel):
    """Query execution record"""
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    query_type: QueryType
    execution_time_ms: float
    rows_affected: int = 0
    cache_hit: bool = False
    optimized: bool = False
    optimization_applied: Optional[OptimizationStrategy] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class OptimizationRecommendation(BaseModel):
    """Query optimization recommendation"""
    recommendation_id: str = Field(default_factory=lambda: str(uuid4()))
    query_pattern_id: str
    strategy: OptimizationStrategy
    description: str
    estimated_improvement_pct: float
    implementation_complexity: str  # low, medium, high
    priority: int = Field(ge=1, le=10)  # 1 lowest, 10 highest
    applied: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


class IndexRecommendation(BaseModel):
    """Specific index recommendation"""
    table_name: str
    columns: List[str]
    index_type: str = "btree"  # btree, hash, gin, gist
    reason: str
    estimated_improvement: float
    cost_analysis: Dict[str, Any] = Field(default_factory=dict)


class IntelligentQueryOptimizer:
    """
    ML-based query optimizer that learns from execution patterns
    and provides intelligent optimization recommendations.
    """

    def __init__(self):
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.executions: List[QueryExecution] = []
        self.recommendations: Dict[str, OptimizationRecommendation] = {}
        self.applied_optimizations: Dict[str, Any] = {}
        self.performance_baselines: Dict[str, float] = {}

    async def track_query(
        self,
        query: str,
        execution_time_ms: float,
        rows_affected: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> QueryExecution:
        """
        Track query execution and learn patterns
        """
        # Extract query type
        query_type = self._detect_query_type(query)

        # Create execution record
        execution = QueryExecution(
            query=query,
            query_type=query_type,
            execution_time_ms=execution_time_ms,
            rows_affected=rows_affected
        )

        self.executions.append(execution)

        # Extract and update pattern
        pattern = await self._extract_pattern(query, query_type)
        await self._update_pattern_stats(pattern, execution_time_ms)

        # Analyze for optimization opportunities
        if pattern.frequency >= 10:  # Minimum frequency threshold
            await self._analyze_optimization_opportunity(pattern)

        return execution

    def _detect_query_type(self, query: str) -> QueryType:
        """Detect query type from SQL"""
        query_upper = query.strip().upper()

        if query_upper.startswith("SELECT"):
            if "JOIN" in query_upper:
                return QueryType.JOIN
            elif any(agg in query_upper for agg in ["COUNT", "SUM", "AVG", "MAX", "MIN"]):
                return QueryType.AGGREGATE
            else:
                return QueryType.SELECT
        elif query_upper.startswith("INSERT"):
            return QueryType.INSERT
        elif query_upper.startswith("UPDATE"):
            return QueryType.UPDATE
        elif query_upper.startswith("DELETE"):
            return QueryType.DELETE
        else:
            return QueryType.SELECT

    async def _extract_pattern(self, query: str, query_type: QueryType) -> QueryPattern:
        """Extract query pattern signature"""
        # Normalize query for pattern matching
        normalized = self._normalize_query(query)
        pattern_signature = self._create_signature(normalized)

        # Check if pattern exists
        if pattern_signature in self.query_patterns:
            return self.query_patterns[pattern_signature]

        # Create new pattern
        tables = self._extract_tables(query)
        columns = self._extract_columns(query)
        conditions = self._extract_where_conditions(query)

        pattern = QueryPattern(
            pattern_id=str(uuid4()),
            pattern_signature=pattern_signature,
            query_type=query_type,
            tables=tables,
            columns=columns,
            where_conditions=conditions
        )

        self.query_patterns[pattern_signature] = pattern

        return pattern

    def _normalize_query(self, query: str) -> str:
        """Normalize query by removing literals and formatting"""
        import re

        # Remove string literals
        normalized = re.sub(r"'[^']*'", "'?'", query)

        # Remove numeric literals
        normalized = re.sub(r'\b\d+\b', '?', normalized)

        # Remove extra whitespace
        normalized = ' '.join(normalized.split())

        return normalized.upper()

    def _create_signature(self, normalized_query: str) -> str:
        """Create pattern signature from normalized query"""
        import hashlib
        return hashlib.md5(normalized_query.encode()).hexdigest()

    def _extract_tables(self, query: str) -> Set[str]:
        """Extract table names from query"""
        import re
        tables = set()

        # Simple extraction - in production would use SQL parser
        from_match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
        if from_match:
            tables.add(from_match.group(1).lower())

        join_matches = re.findall(r'JOIN\s+(\w+)', query, re.IGNORECASE)
        for match in join_matches:
            tables.add(match.lower())

        return tables

    def _extract_columns(self, query: str) -> Set[str]:
        """Extract column names from query"""
        import re
        columns = set()

        # Extract from SELECT clause
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
        if select_match:
            columns_str = select_match.group(1)
            if '*' not in columns_str:
                for col in columns_str.split(','):
                    col_name = col.strip().split()[-1]  # Get last word (handles AS aliases)
                    columns.add(col_name.lower())

        return columns

    def _extract_where_conditions(self, query: str) -> List[str]:
        """Extract WHERE conditions"""
        import re
        conditions = []

        where_match = re.search(r'WHERE\s+(.+?)(?:ORDER BY|GROUP BY|LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1).strip()
            # Split by AND/OR
            parts = re.split(r'\s+(?:AND|OR)\s+', where_clause, flags=re.IGNORECASE)
            conditions = [p.strip() for p in parts]

        return conditions

    async def _update_pattern_stats(
        self,
        pattern: QueryPattern,
        execution_time_ms: float
    ) -> None:
        """Update pattern statistics"""
        pattern.frequency += 1
        pattern.last_seen = datetime.utcnow()

        # Update moving average
        if pattern.avg_execution_time_ms == 0:
            pattern.avg_execution_time_ms = execution_time_ms
        else:
            # Exponential moving average
            alpha = 0.2
            pattern.avg_execution_time_ms = (
                alpha * execution_time_ms +
                (1 - alpha) * pattern.avg_execution_time_ms
            )

    async def _analyze_optimization_opportunity(
        self,
        pattern: QueryPattern
    ) -> None:
        """Analyze pattern for optimization opportunities"""

        # Skip if already optimized
        if pattern.pattern_id in self.applied_optimizations:
            return

        # Check for slow queries
        if pattern.avg_execution_time_ms > 100:  # 100ms threshold

            # Recommend index if query has WHERE conditions
            if pattern.where_conditions:
                await self._recommend_index(pattern)

            # Recommend caching for read-heavy queries
            if pattern.query_type in [QueryType.SELECT, QueryType.AGGREGATE]:
                if pattern.frequency > 100:  # High frequency
                    await self._recommend_caching(pattern)

            # Recommend materialized view for complex aggregates
            if pattern.query_type == QueryType.AGGREGATE:
                if pattern.avg_execution_time_ms > 500:
                    await self._recommend_materialized_view(pattern)

    async def _recommend_index(self, pattern: QueryPattern) -> None:
        """Recommend index creation"""
        for table in pattern.tables:
            # Extract columns from WHERE conditions
            index_columns = self._extract_index_columns(pattern.where_conditions)

            if index_columns:
                recommendation = OptimizationRecommendation(
                    query_pattern_id=pattern.pattern_id,
                    strategy=OptimizationStrategy.INDEX_RECOMMENDATION,
                    description=f"Create index on {table}({', '.join(index_columns)})",
                    estimated_improvement_pct=40.0,  # Estimated
                    implementation_complexity="low",
                    priority=8
                )

                self.recommendations[recommendation.recommendation_id] = recommendation

    def _extract_index_columns(self, conditions: List[str]) -> List[str]:
        """Extract columns that would benefit from indexing"""
        import re
        columns = []

        for condition in conditions:
            # Extract column name from condition (before operator)
            match = re.match(r'(\w+)\s*[=<>]', condition)
            if match:
                columns.append(match.group(1))

        return columns[:3]  # Limit to 3 columns for composite index

    async def _recommend_caching(self, pattern: QueryPattern) -> None:
        """Recommend query result caching"""
        recommendation = OptimizationRecommendation(
            query_pattern_id=pattern.pattern_id,
            strategy=OptimizationStrategy.CACHING,
            description=f"Cache query results (freq: {pattern.frequency})",
            estimated_improvement_pct=70.0,
            implementation_complexity="low",
            priority=9
        )

        self.recommendations[recommendation.recommendation_id] = recommendation

    async def _recommend_materialized_view(self, pattern: QueryPattern) -> None:
        """Recommend materialized view"""
        recommendation = OptimizationRecommendation(
            query_pattern_id=pattern.pattern_id,
            strategy=OptimizationStrategy.MATERIALIZED_VIEW,
            description=f"Create materialized view for aggregate query",
            estimated_improvement_pct=80.0,
            implementation_complexity="medium",
            priority=7
        )

        self.recommendations[recommendation.recommendation_id] = recommendation

    async def get_optimization_recommendations(
        self,
        min_priority: int = 5,
        limit: int = 10
    ) -> List[OptimizationRecommendation]:
        """Get top optimization recommendations"""
        recommendations = [
            rec for rec in self.recommendations.values()
            if rec.priority >= min_priority and not rec.applied
        ]

        # Sort by priority desc
        recommendations.sort(key=lambda x: x.priority, reverse=True)

        return recommendations[:limit]

    async def apply_optimization(
        self,
        recommendation_id: str
    ) -> Dict[str, Any]:
        """
        Apply optimization (in production would execute actual changes)
        """
        if recommendation_id not in self.recommendations:
            return {"success": False, "error": "Recommendation not found"}

        recommendation = self.recommendations[recommendation_id]

        # Mark as applied
        recommendation.applied = True

        # Store applied optimization
        self.applied_optimizations[recommendation.query_pattern_id] = {
            "recommendation_id": recommendation_id,
            "strategy": recommendation.strategy,
            "applied_at": datetime.utcnow(),
            "description": recommendation.description
        }

        return {
            "success": True,
            "recommendation_id": recommendation_id,
            "strategy": recommendation.strategy.value,
            "description": recommendation.description
        }

    async def get_performance_report(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get performance analysis report
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        # Filter recent executions
        recent_executions = [
            e for e in self.executions
            if e.timestamp >= cutoff
        ]

        if not recent_executions:
            return {"error": "No data for specified period"}

        # Calculate statistics
        total_queries = len(recent_executions)
        avg_execution_time = sum(e.execution_time_ms for e in recent_executions) / total_queries

        # Group by query type
        by_type: Dict[QueryType, List[float]] = {}
        for execution in recent_executions:
            if execution.query_type not in by_type:
                by_type[execution.query_type] = []
            by_type[execution.query_type].append(execution.execution_time_ms)

        type_stats = {
            qt.value: {
                "count": len(times),
                "avg_ms": sum(times) / len(times),
                "max_ms": max(times),
                "min_ms": min(times)
            }
            for qt, times in by_type.items()
        }

        # Find slow queries
        slow_threshold = 200  # ms
        slow_queries = [
            {
                "query": e.query[:100],  # Truncate
                "execution_time_ms": e.execution_time_ms,
                "timestamp": e.timestamp.isoformat()
            }
            for e in recent_executions
            if e.execution_time_ms > slow_threshold
        ]

        # Sort slow queries by execution time
        slow_queries.sort(key=lambda x: x["execution_time_ms"], reverse=True)

        return {
            "period_hours": hours,
            "total_queries": total_queries,
            "avg_execution_time_ms": round(avg_execution_time, 2),
            "by_type": type_stats,
            "slow_queries_count": len(slow_queries),
            "top_slow_queries": slow_queries[:10],
            "patterns_detected": len(self.query_patterns),
            "recommendations_count": len([r for r in self.recommendations.values() if not r.applied])
        }

    async def get_index_recommendations(self) -> List[IndexRecommendation]:
        """Get specific index recommendations"""
        index_recs = []

        for rec in self.recommendations.values():
            if rec.strategy == OptimizationStrategy.INDEX_RECOMMENDATION and not rec.applied:
                # Parse recommendation description to extract details
                # In production would store structured data
                pattern = self.query_patterns.get(rec.query_pattern_id)
                if pattern:
                    for table in pattern.tables:
                        columns = self._extract_index_columns(pattern.where_conditions)
                        if columns:
                            index_recs.append(IndexRecommendation(
                                table_name=table,
                                columns=columns,
                                index_type="btree",
                                reason=rec.description,
                                estimated_improvement=rec.estimated_improvement_pct
                            ))

        return index_recs

    def clear_old_data(self, days: int = 30) -> int:
        """Clear old execution data"""
        cutoff = datetime.utcnow() - timedelta(days=days)

        original_count = len(self.executions)
        self.executions = [
            e for e in self.executions
            if e.timestamp >= cutoff
        ]

        return original_count - len(self.executions)
