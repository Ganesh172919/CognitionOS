"""
Advanced Analytics API Routes

Routes for usage analytics, forecasting, and pricing recommendations.
"""
from typing import Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.src.dependencies.injection import get_db_session
from infrastructure.analytics.usage_analytics import (
    UsageAnalyticsEngine,
    ForecastMethod
)
from infrastructure.intelligence.pricing_optimizer import PricingOptimizer
from infrastructure.intelligence.recommendation_engine import RecommendationEngine


router = APIRouter(prefix="/api/v3/analytics", tags=["analytics"])


@router.get("/usage")
async def get_usage_analytics(
    tenant_id: str,
    resource_type: str,
    days: int = Query(30, ge=1, le=365),
    session: AsyncSession = Depends(get_db_session)
):
    """Get comprehensive usage analytics"""
    engine = UsageAnalyticsEngine(session)
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    analytics = await engine.get_usage_analytics(
        tenant_id=tenant_id,
        resource_type=resource_type,
        start_date=start_date,
        end_date=end_date
    )
    
    return {
        "tenant_id": analytics.tenant_id,
        "resource_type": analytics.resource_type,
        "period_start": analytics.period_start.isoformat(),
        "period_end": analytics.period_end.isoformat(),
        "total_usage": analytics.total_usage,
        "avg_daily_usage": analytics.avg_daily_usage,
        "peak_usage": analytics.peak_usage,
        "peak_time": analytics.peak_time.isoformat(),
        "trend": analytics.trend,
        "growth_rate": analytics.growth_rate,
        "percentile_rank": analytics.percentile_rank,
        "cost_efficiency_score": analytics.cost_efficiency_score
    }


@router.get("/forecast")
async def get_usage_forecast(
    tenant_id: str,
    resource_type: str,
    days_ahead: int = Query(30, ge=1, le=90),
    method: ForecastMethod = ForecastMethod.EXPONENTIAL_SMOOTHING,
    session: AsyncSession = Depends(get_db_session)
):
    """Get usage forecast"""
    engine = UsageAnalyticsEngine(session)
    
    forecasts = await engine.forecast_usage(
        tenant_id=tenant_id,
        resource_type=resource_type,
        days_ahead=days_ahead,
        method=method
    )
    
    return {
        "tenant_id": tenant_id,
        "resource_type": resource_type,
        "forecast_method": method.value,
        "days_ahead": days_ahead,
        "forecasts": [
            {
                "date": f.forecast_date.isoformat(),
                "predicted_quantity": f.predicted_quantity,
                "confidence_lower": f.confidence_lower,
                "confidence_upper": f.confidence_upper
            }
            for f in forecasts
        ]
    }


@router.get("/anomalies")
async def detect_usage_anomalies(
    tenant_id: str,
    resource_type: str,
    lookback_days: int = Query(7, ge=1, le=30),
    session: AsyncSession = Depends(get_db_session)
):
    """Detect usage anomalies"""
    engine = UsageAnalyticsEngine(session)
    
    anomalies = await engine.detect_anomalies(
        tenant_id=tenant_id,
        resource_type=resource_type,
        lookback_days=lookback_days
    )
    
    return {
        "tenant_id": tenant_id,
        "resource_type": resource_type,
        "anomalies_detected": len(anomalies),
        "anomalies": [
            {
                "detected_at": a.detected_at.isoformat(),
                "type": a.anomaly_type.value,
                "severity": a.severity,
                "expected_value": a.expected_value,
                "actual_value": a.actual_value,
                "description": a.description,
                "suggested_action": a.suggested_action
            }
            for a in anomalies
        ]
    }


@router.get("/cost-optimization")
async def get_cost_optimization_recommendations(
    tenant_id: str,
    session: AsyncSession = Depends(get_db_session)
):
    """Get cost optimization recommendations"""
    engine = UsageAnalyticsEngine(session)
    
    recommendations = await engine.get_cost_optimization_recommendations(tenant_id)
    
    return {
        "tenant_id": tenant_id,
        "recommendations_count": len(recommendations),
        "recommendations": recommendations
    }


@router.get("/pricing/recommendation")
async def get_pricing_recommendation(
    tenant_id: str,
    analysis_days: int = Query(30, ge=7, le=90),
    session: AsyncSession = Depends(get_db_session)
):
    """Get AI-powered tier recommendation"""
    optimizer = PricingOptimizer(session)
    
    recommendation = await optimizer.get_tier_recommendation(
        tenant_id=tenant_id,
        analysis_days=analysis_days
    )
    
    return {
        "tenant_id": recommendation.tenant_id,
        "current_tier": recommendation.current_tier.value,
        "recommended_tier": recommendation.recommended_tier.value,
        "recommendation_type": recommendation.recommendation_type.value,
        "confidence_score": recommendation.confidence_score,
        "estimated_savings_monthly": recommendation.estimated_savings_monthly,
        "estimated_revenue_impact": recommendation.estimated_revenue_impact,
        "reasoning": recommendation.reasoning,
        "factors": recommendation.factors,
        "effective_date": recommendation.effective_date.isoformat()
    }


@router.get("/pricing/customer-value")
async def get_customer_value(
    tenant_id: str,
    session: AsyncSession = Depends(get_db_session)
):
    """Get customer lifetime value analysis"""
    optimizer = PricingOptimizer(session)
    
    value = await optimizer.calculate_customer_value(tenant_id)
    
    return {
        "tenant_id": value.tenant_id,
        "ltv": float(value.ltv),
        "cac": float(value.cac),
        "ltv_cac_ratio": value.ltv_cac_ratio,
        "churn_probability": value.churn_probability,
        "engagement_score": value.engagement_score,
        "value_segment": value.value_segment,
        "retention_priority": value.retention_priority
    }


@router.get("/recommendations")
async def get_recommendations(
    tenant_id: str,
    limit: int = Query(5, ge=1, le=20),
    session: AsyncSession = Depends(get_db_session)
):
    """Get personalized recommendations"""
    engine = RecommendationEngine(session)
    
    recommendations = await engine.get_recommendations(tenant_id, limit)
    
    return {
        "tenant_id": tenant_id,
        "recommendations": [
            {
                "item_id": r.item_id,
                "item_type": r.item_type.value,
                "score": r.score,
                "title": r.title,
                "description": r.description,
                "reasoning": r.reasoning,
                "benefit": r.benefit,
                "estimated_impact": r.estimated_impact
            }
            for r in recommendations
        ]
    }
