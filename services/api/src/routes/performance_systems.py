"""
API Routes for Performance Optimization Systems
Exposes intelligent query optimization and adaptive caching.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from infrastructure.performance.intelligent_optimizer import (
    IntelligentQueryOptimizer,
    QueryType
)
from infrastructure.performance.adaptive_cache import AdaptiveCacheSystem

router = APIRouter(prefix="/api/v3/performance", tags=["Performance"])

# Initialize systems
query_optimizer = IntelligentQueryOptimizer()
cache_system = AdaptiveCacheSystem(max_size_mb=1000, enable_prefetch=True)


# Query Optimization endpoints

class TrackQueryRequest(BaseModel):
    """Request to track query execution"""
    query: str
    execution_time_ms: float
    rows_affected: int = 0


@router.post("/optimizer/track")
async def track_query(request: TrackQueryRequest):
    """
    Track query execution for optimization learning
    """
    try:
        execution = await query_optimizer.track_query(
            query=request.query,
            execution_time_ms=request.execution_time_ms,
            rows_affected=request.rows_affected
        )

        return {
            "success": True,
            "execution_id": execution.execution_id,
            "query_type": execution.query_type.value,
            "execution_time_ms": execution.execution_time_ms
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/optimizer/recommendations")
async def get_optimization_recommendations(
    min_priority: int = Query(5, ge=1, le=10),
    limit: int = Query(10, ge=1, le=50)
):
    """
    Get optimization recommendations

    Returns ML-generated recommendations for:
    - Index creation
    - Query rewriting
    - Caching opportunities
    - Materialized views
    """
    try:
        recommendations = await query_optimizer.get_optimization_recommendations(
            min_priority=min_priority,
            limit=limit
        )

        return {
            "success": True,
            "recommendations": [
                {
                    "recommendation_id": rec.recommendation_id,
                    "strategy": rec.strategy.value,
                    "description": rec.description,
                    "estimated_improvement_pct": rec.estimated_improvement_pct,
                    "implementation_complexity": rec.implementation_complexity,
                    "priority": rec.priority,
                    "applied": rec.applied
                }
                for rec in recommendations
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/optimizer/recommendations/{recommendation_id}/apply")
async def apply_optimization(recommendation_id: str):
    """
    Apply an optimization recommendation
    """
    try:
        result = await query_optimizer.apply_optimization(recommendation_id)

        return {
            "success": result["success"],
            **result
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/optimizer/report")
async def get_performance_report(
    hours: int = Query(24, ge=1, le=168)
):
    """
    Get performance analysis report
    """
    try:
        report = await query_optimizer.get_performance_report(hours=hours)

        return {
            "success": True,
            **report
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/optimizer/indexes")
async def get_index_recommendations():
    """
    Get specific index recommendations
    """
    try:
        recommendations = await query_optimizer.get_index_recommendations()

        return {
            "success": True,
            "recommendations": [
                {
                    "table_name": rec.table_name,
                    "columns": rec.columns,
                    "index_type": rec.index_type,
                    "reason": rec.reason,
                    "estimated_improvement": rec.estimated_improvement
                }
                for rec in recommendations
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Cache endpoints

class CacheSetRequest(BaseModel):
    """Request to set cache value"""
    key: str
    value: dict  # JSON-serializable
    ttl_seconds: Optional[int] = None
    tags: Optional[list[str]] = None


@router.post("/cache/set")
async def cache_set(request: CacheSetRequest):
    """
    Set value in adaptive cache
    """
    try:
        success = await cache_system.set(
            key=request.key,
            value=request.value,
            ttl_seconds=request.ttl_seconds,
            tags=set(request.tags) if request.tags else None
        )

        return {
            "success": success,
            "key": request.key
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/cache/get/{key}")
async def cache_get(key: str):
    """
    Get value from cache
    """
    try:
        value = await cache_system.get(key)

        if value is None:
            return {
                "success": True,
                "found": False,
                "key": key
            }

        return {
            "success": True,
            "found": True,
            "key": key,
            "value": value
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/cache/{key}")
async def cache_delete(key: str):
    """
    Delete key from cache
    """
    try:
        success = await cache_system.delete(key)

        return {
            "success": True,
            "deleted": success,
            "key": key
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/cache/statistics")
async def get_cache_statistics():
    """
    Get cache performance statistics

    Returns:
    - Hit rate
    - Total requests
    - Cache size and utilization
    - Average access time
    - Detected patterns
    """
    try:
        stats = await cache_system.get_statistics()

        return {
            "success": True,
            **stats
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/cache/insights/{key}")
async def get_key_insights(key: str):
    """
    Get insights about a specific cache key

    Returns access patterns, importance score, and related keys
    """
    try:
        insights = await cache_system.get_key_insights(key)

        return {
            "success": True,
            **insights
        }

    except Exception as e:
        if "error" in insights:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=insights["error"]
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/cache/optimize")
async def optimize_cache():
    """
    Run optimization pass on cache

    Evicts expired and low-value entries
    """
    try:
        result = await cache_system.optimize()

        return {
            "success": True,
            **result
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/cache/clear")
async def clear_cache():
    """
    Clear entire cache
    """
    try:
        count = await cache_system.clear()

        return {
            "success": True,
            "cleared_keys": count
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
