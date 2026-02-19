"""
Plugin Marketplace API Routes

Routes for plugin marketplace, discovery, and management.
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.src.dependencies.injection import get_db_session
from infrastructure.marketplace.plugin_marketplace import (
    PluginMarketplace,
    PluginCategory
)


router = APIRouter(prefix="/api/v3/marketplace", tags=["marketplace"])


@router.get("/plugins/search")
async def search_plugins(
    query: str = Query("", min_length=0, max_length=100),
    category: Optional[PluginCategory] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_db_session)
):
    """Search plugins in marketplace"""
    marketplace = PluginMarketplace(session)
    
    plugins = await marketplace.search_plugins(
        query=query,
        category=category,
        limit=limit,
        offset=offset
    )
    
    return {
        "query": query,
        "category": category.value if category else None,
        "total": len(plugins),
        "limit": limit,
        "offset": offset,
        "plugins": plugins
    }


@router.get("/plugins/{plugin_id}")
async def get_plugin_details(
    plugin_id: str,
    session: AsyncSession = Depends(get_db_session)
):
    """Get detailed plugin information"""
    marketplace = PluginMarketplace(session)
    
    try:
        details = await marketplace.get_plugin_details(plugin_id)
        return details
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/plugins/{plugin_id}/install")
async def install_plugin(
    plugin_id: str,
    tenant_id: str,
    session: AsyncSession = Depends(get_db_session)
):
    """Install a plugin for a tenant"""
    marketplace = PluginMarketplace(session)
    
    try:
        result = await marketplace.install_plugin(plugin_id, tenant_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/plugins/{plugin_id}/rate")
async def rate_plugin(
    plugin_id: str,
    user_id: str,
    rating: float = Query(..., ge=1.0, le=5.0),
    review: Optional[str] = None,
    session: AsyncSession = Depends(get_db_session)
):
    """Submit a rating and review for a plugin"""
    marketplace = PluginMarketplace(session)
    
    try:
        rating_obj = await marketplace.rate_plugin(plugin_id, user_id, rating, review)
        return {
            "plugin_id": rating_obj.plugin_id,
            "user_id": rating_obj.user_id,
            "rating": rating_obj.rating,
            "review": rating_obj.review,
            "created_at": rating_obj.created_at.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/plugins/{plugin_id}/reviews")
async def get_plugin_reviews(
    plugin_id: str,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_db_session)
):
    """Get reviews for a plugin"""
    marketplace = PluginMarketplace(session)
    
    reviews = await marketplace.get_plugin_reviews(plugin_id, limit, offset)
    
    return {
        "plugin_id": plugin_id,
        "total": len(reviews),
        "reviews": [
            {
                "user_id": r.user_id,
                "rating": r.rating,
                "review": r.review,
                "created_at": r.created_at.isoformat(),
                "helpful_count": r.helpful_count
            }
            for r in reviews
        ]
    }


@router.get("/recommendations")
async def get_plugin_recommendations(
    tenant_id: str,
    limit: int = Query(5, ge=1, le=20),
    session: AsyncSession = Depends(get_db_session)
):
    """Get personalized plugin recommendations"""
    marketplace = PluginMarketplace(session)
    
    recommendations = await marketplace.get_recommendations(tenant_id, limit)
    
    return {
        "tenant_id": tenant_id,
        "recommendations": [
            {
                "plugin_id": r.plugin_id,
                "plugin_name": r.plugin_name,
                "score": r.score,
                "reasoning": r.reasoning,
                "category": r.category.value
            }
            for r in recommendations
        ]
    }


@router.get("/trending")
async def get_trending_plugins(
    days: int = Query(7, ge=1, le=30),
    limit: int = Query(10, ge=1, le=50),
    session: AsyncSession = Depends(get_db_session)
):
    """Get trending plugins"""
    marketplace = PluginMarketplace(session)
    
    trending = await marketplace.get_trending_plugins(days, limit)
    
    return {
        "period_days": days,
        "trending_plugins": trending
    }


@router.get("/developer/{developer_id}/dashboard")
async def get_developer_dashboard(
    developer_id: str,
    session: AsyncSession = Depends(get_db_session)
):
    """Get dashboard data for plugin developer"""
    marketplace = PluginMarketplace(session)
    
    dashboard = await marketplace.get_developer_dashboard(developer_id)
    
    return dashboard
