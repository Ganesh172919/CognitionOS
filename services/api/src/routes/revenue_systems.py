"""
API Routes for Revenue Systems
Exposes usage billing and feature gating via REST API.
"""

from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from infrastructure.revenue import (
    UsageBasedBillingEngine,
    DynamicFeatureGate,
    UsageMetricType,
    SubscriptionTier,
    QuotaType
)

router = APIRouter(prefix="/api/v3/revenue", tags=["Revenue Systems"])

# Initialize systems (in production, would be dependency injection)
billing_engine = UsageBasedBillingEngine()
feature_gate = DynamicFeatureGate()


# Billing endpoints

class RecordUsageRequest(BaseModel):
    """Request to record usage"""
    tenant_id: str
    metric_type: UsageMetricType
    quantity: float
    user_id: Optional[str] = None
    metadata: Optional[dict] = None


@router.post("/billing/usage")
async def record_usage(request: RecordUsageRequest):
    """
    Record usage event for billing

    Real-time usage tracking with automatic cost calculation
    """
    try:
        record = await billing_engine.record_usage(
            tenant_id=request.tenant_id,
            metric_type=request.metric_type,
            quantity=Decimal(str(request.quantity)),
            user_id=request.user_id,
            metadata=request.metadata
        )

        return {
            "success": True,
            "record_id": record.record_id,
            "metric_type": record.metric_type.value,
            "quantity": float(record.quantity),
            "unit_cost": float(record.unit_cost),
            "total_cost": float(record.total_cost),
            "timestamp": record.timestamp.isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/billing/usage/{tenant_id}/current")
async def get_current_usage(tenant_id: str):
    """
    Get current month-to-date usage and estimated cost
    """
    try:
        usage = await billing_engine.get_current_usage_cost(tenant_id)

        return {
            "success": True,
            **usage
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/billing/invoices/{tenant_id}")
async def get_invoices(
    tenant_id: str,
    limit: int = Query(10, ge=1, le=100)
):
    """
    Get invoices for tenant
    """
    try:
        invoices = await billing_engine.get_tenant_invoices(tenant_id, limit)

        return {
            "success": True,
            "invoices": [
                {
                    "invoice_id": inv.invoice_id,
                    "billing_period": inv.billing_period.value,
                    "period_start": inv.period_start.isoformat(),
                    "period_end": inv.period_end.isoformat(),
                    "line_items": inv.line_items,
                    "subtotal": float(inv.subtotal),
                    "tax": float(inv.tax),
                    "total": float(inv.total),
                    "status": inv.status,
                    "due_date": inv.due_date.isoformat() if inv.due_date else None
                }
                for inv in invoices
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/billing/pricing/{metric_type}")
async def get_pricing_config(metric_type: UsageMetricType):
    """
    Get pricing configuration for a metric
    """
    config = billing_engine.get_pricing_config(metric_type)

    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pricing configuration not found"
        )

    return {
        "success": True,
        "metric_type": config.metric_type.value,
        "pricing_model": config.pricing_model.value,
        "tiers": [
            {
                "tier_name": tier.tier_name,
                "start_quantity": tier.start_quantity,
                "end_quantity": tier.end_quantity,
                "price_per_unit": float(tier.price_per_unit),
                "flat_fee": float(tier.flat_fee)
            }
            for tier in config.tiers
        ],
        "minimum_charge": float(config.minimum_charge),
        "maximum_charge": float(config.maximum_charge) if config.maximum_charge else None
    }


# Feature gating endpoints

@router.get("/features/check/{tenant_id}/{feature_id}")
async def check_feature_access(tenant_id: str, feature_id: str):
    """
    Check if tenant has access to a feature
    """
    result = await feature_gate.check_feature_access(tenant_id, feature_id)

    return {
        "success": True,
        "allowed": result.allowed,
        "reason": result.reason,
        "upgrade_required": result.upgrade_required.value if result.upgrade_required else None,
        "quota_exceeded": result.quota_exceeded.value if result.quota_exceeded else None
    }


@router.get("/features/quota/{tenant_id}/{quota_type}")
async def check_quota(
    tenant_id: str,
    quota_type: QuotaType,
    requested_amount: int = Query(1, ge=1)
):
    """
    Check if tenant has quota available
    """
    result = await feature_gate.check_quota(tenant_id, quota_type, requested_amount)

    return {
        "success": True,
        "allowed": result.allowed,
        "reason": result.reason,
        "quota_exceeded": result.quota_exceeded.value if result.quota_exceeded else None
    }


@router.post("/features/quota/{tenant_id}/{quota_type}/consume")
async def consume_quota(
    tenant_id: str,
    quota_type: QuotaType,
    amount: int = Query(1, ge=1)
):
    """
    Consume quota for tenant
    """
    success = await feature_gate.consume_quota(tenant_id, quota_type, amount)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to consume quota"
        )

    return {
        "success": True,
        "quota_type": quota_type.value,
        "amount_consumed": amount
    }


@router.get("/features/available/{tenant_id}")
async def get_available_features(tenant_id: str):
    """
    Get list of features available to tenant
    """
    features = feature_gate.get_available_features(tenant_id)

    return {
        "success": True,
        "features": [
            {
                "feature_id": f.feature_id,
                "name": f.name,
                "description": f.description,
                "category": f.category.value,
                "required_tier": f.required_tier.value,
                "beta": f.beta,
                "deprecated": f.deprecated
            }
            for f in features
        ]
    }


@router.get("/subscription/tiers")
async def get_tier_comparison():
    """
    Get comparison of all subscription tiers
    """
    comparison = feature_gate.get_tier_comparison()

    return {
        "success": True,
        "tiers": comparison
    }


class CreateSubscriptionRequest(BaseModel):
    """Request to create subscription"""
    tenant_id: str
    tier: SubscriptionTier
    is_trial: bool = False
    trial_days: int = 14


@router.post("/subscription/create")
async def create_subscription(request: CreateSubscriptionRequest):
    """
    Create new subscription for tenant
    """
    subscription = feature_gate.create_subscription(
        tenant_id=request.tenant_id,
        tier=request.tier,
        is_trial=request.is_trial,
        trial_days=request.trial_days
    )

    return {
        "success": True,
        "tenant_id": subscription.tenant_id,
        "tier": subscription.tier.value,
        "is_trial": subscription.is_trial,
        "trial_end": subscription.trial_end.isoformat() if subscription.trial_end else None,
        "quotas": {
            qt.value: {
                "limit": q.limit,
                "current_usage": q.current_usage
            }
            for qt, q in subscription.quotas.items()
        }
    }


class UpgradeSubscriptionRequest(BaseModel):
    """Request to upgrade subscription"""
    tenant_id: str
    new_tier: SubscriptionTier


@router.post("/subscription/upgrade")
async def upgrade_subscription(request: UpgradeSubscriptionRequest):
    """
    Upgrade subscription to new tier
    """
    subscription = feature_gate.upgrade_subscription(
        tenant_id=request.tenant_id,
        new_tier=request.new_tier
    )

    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to upgrade subscription (not found or not an upgrade)"
        )

    return {
        "success": True,
        "tenant_id": subscription.tenant_id,
        "tier": subscription.tier.value,
        "updated_at": subscription.updated_at.isoformat()
    }
