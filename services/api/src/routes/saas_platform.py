"""
SaaS Platform API Routes
API endpoints for subscription management, tenant management, usage metering, and rate limiting
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from decimal import Decimal

from infrastructure.saas.advanced_subscription_manager import (
    SubscriptionManager,
    SubscriptionTier,
    BillingCycle,
    PaymentMethod
)
from infrastructure.saas.multi_tenant_manager import (
    MultiTenantManager,
    TenantType,
    IsolationLevel
)
from infrastructure.saas.usage_metering_engine import (
    UsageMeteringEngine,
    UsageEventType,
    TokenProvider
)
from infrastructure.saas.advanced_rate_limiter import (
    RateLimitManager,
    RateLimitStrategy,
    LimitScope
)

# Initialize managers
subscription_manager = SubscriptionManager()
tenant_manager = MultiTenantManager()
metering_engine = UsageMeteringEngine()
rate_limit_manager = RateLimitManager()

router = APIRouter(prefix="/api/v3/saas", tags=["saas-platform"])


# ============================================================================
# SUBSCRIPTION MANAGEMENT
# ============================================================================

@router.post("/subscriptions")
async def create_subscription(
    tenant_id: str,
    tier: str,
    billing_cycle: str,
    payment_method: Optional[str] = None,
    trial_days: int = 0
):
    """Create new subscription"""
    try:
        tier_enum = SubscriptionTier[tier.upper()]
        cycle_enum = BillingCycle[billing_cycle.upper()]
        payment_enum = PaymentMethod[payment_method.upper()] if payment_method else None

        subscription = await subscription_manager.create_subscription(
            tenant_id=tenant_id,
            tier=tier_enum,
            billing_cycle=cycle_enum,
            payment_method=payment_enum,
            trial_days=trial_days
        )

        return {
            "subscription_id": subscription.subscription_id,
            "tenant_id": subscription.tenant_id,
            "tier": subscription.tier.value,
            "status": subscription.status.value,
            "current_period_start": subscription.current_period_start.isoformat(),
            "current_period_end": subscription.current_period_end.isoformat(),
            "trial_end": subscription.trial_end.isoformat() if subscription.trial_end else None
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/subscriptions/{subscription_id}/upgrade")
async def upgrade_subscription(
    subscription_id: str,
    new_tier: str,
    immediate: bool = True
):
    """Upgrade subscription"""
    try:
        tier_enum = SubscriptionTier[new_tier.upper()]

        result = await subscription_manager.upgrade_subscription(
            subscription_id=subscription_id,
            new_tier=tier_enum,
            immediate=immediate
        )

        return {
            "success": result["success"],
            "subscription_id": subscription_id,
            "new_tier": new_tier,
            "upgrade_cost": float(result["upgrade_cost"]),
            "applied_immediately": immediate
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/subscriptions/{subscription_id}/cancel")
async def cancel_subscription(
    subscription_id: str,
    immediate: bool = False,
    reason: Optional[str] = None
):
    """Cancel subscription"""
    try:
        subscription = await subscription_manager.cancel_subscription(
            subscription_id=subscription_id,
            immediate=immediate,
            reason=reason
        )

        return {
            "subscription_id": subscription.subscription_id,
            "status": subscription.status.value,
            "cancelled_at": subscription.cancelled_at.isoformat() if subscription.cancelled_at else None,
            "cancellation_reason": subscription.cancellation_reason
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/subscriptions/{tenant_id}/metrics")
async def get_subscription_metrics(tenant_id: str):
    """Get subscription metrics"""
    try:
        metrics = await subscription_manager.get_subscription_metrics(tenant_id)
        return metrics

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# TENANT MANAGEMENT
# ============================================================================

@router.post("/tenants")
async def create_tenant(
    name: str,
    type: str,
    owner_user_id: str,
    isolation_level: str = "shared",
    trial_days: int = 14
):
    """Create new tenant"""
    try:
        type_enum = TenantType[type.upper()]
        isolation_enum = IsolationLevel[isolation_level.upper()]

        tenant = await tenant_manager.create_tenant(
            name=name,
            type=type_enum,
            owner_user_id=owner_user_id,
            isolation_level=isolation_enum,
            trial_days=trial_days
        )

        return {
            "tenant_id": tenant.tenant_id,
            "name": tenant.name,
            "type": tenant.type.value,
            "status": tenant.status.value,
            "isolation_level": tenant.config.isolation_level.value,
            "trial_end_date": tenant.trial_end_date.isoformat() if tenant.trial_end_date else None
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/tenants/{tenant_id}/users")
async def add_user_to_tenant(
    tenant_id: str,
    user_id: str,
    email: str,
    role: str,
    permissions: Optional[List[str]] = None
):
    """Add user to tenant"""
    try:
        user = await tenant_manager.add_user_to_tenant(
            tenant_id=tenant_id,
            user_id=user_id,
            email=email,
            role=role,
            permissions=permissions
        )

        return {
            "user_id": user.user_id,
            "tenant_id": user.tenant_id,
            "email": user.email,
            "role": user.role,
            "is_active": user.is_active
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/tenants/{tenant_id}/analytics")
async def get_tenant_analytics(tenant_id: str):
    """Get tenant analytics"""
    try:
        analytics = await tenant_manager.get_tenant_analytics(tenant_id)
        return analytics

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/tenants/{tenant_id}/suspend")
async def suspend_tenant(tenant_id: str, reason: str):
    """Suspend tenant"""
    try:
        tenant = await tenant_manager.suspend_tenant(tenant_id, reason)
        return {
            "tenant_id": tenant.tenant_id,
            "status": tenant.status.value,
            "suspended_reason": tenant.suspended_reason,
            "suspended_at": tenant.suspended_at.isoformat() if tenant.suspended_at else None
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# USAGE METERING
# ============================================================================

@router.post("/usage/record")
async def record_usage(
    tenant_id: str,
    user_id: str,
    event_type: str,
    quantity: float = 1.0,
    unit: str = "count",
    input_tokens: int = 0,
    output_tokens: int = 0,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    duration_ms: Optional[int] = None,
    success: bool = True
):
    """Record usage event"""
    try:
        event_type_enum = UsageEventType[event_type.upper()]
        provider_enum = TokenProvider[provider.upper()] if provider else None

        event = await metering_engine.record_usage(
            tenant_id=tenant_id,
            user_id=user_id,
            event_type=event_type_enum,
            quantity=Decimal(str(quantity)),
            unit=unit,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider=provider_enum,
            model_name=model_name,
            duration_ms=duration_ms,
            success=success
        )

        return {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "cost": float(event.cost),
            "total_tokens": event.total_tokens
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/usage/{tenant_id}/summary")
async def get_usage_summary(
    tenant_id: str,
    period_hours: int = 24
):
    """Get usage summary"""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=period_hours)

        summary = await metering_engine.get_usage_summary(
            tenant_id=tenant_id,
            period_start=start_time,
            period_end=end_time
        )

        return {
            "tenant_id": summary.tenant_id,
            "period_start": summary.period_start.isoformat(),
            "period_end": summary.period_end.isoformat(),
            "total_tokens": summary.total_tokens,
            "total_cost": float(summary.total_cost),
            "by_provider": {
                k: {
                    "total_tokens": v["total_tokens"],
                    "cost": float(v["cost"])
                }
                for k, v in summary.by_provider.items()
            },
            "by_model": {
                k: {
                    "total_tokens": v["total_tokens"],
                    "cost": float(v["cost"])
                }
                for k, v in summary.by_model.items()
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/usage/{tenant_id}/realtime")
async def get_realtime_metrics(
    tenant_id: str,
    window_minutes: int = 5
):
    """Get real-time usage metrics"""
    try:
        metrics = await metering_engine.get_real_time_metrics(
            tenant_id=tenant_id,
            window_minutes=window_minutes
        )
        return metrics

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# RATE LIMITING
# ============================================================================

@router.post("/rate-limits")
async def create_rate_limit(
    name: str,
    strategy: str,
    scope: str,
    max_requests: int,
    window_seconds: int,
    burst_size: int = 0,
    dynamic_adjustment: bool = False
):
    """Create rate limit"""
    try:
        strategy_enum = RateLimitStrategy[strategy.upper()]
        scope_enum = LimitScope[scope.upper()]

        limit = await rate_limit_manager.create_limit(
            name=name,
            strategy=strategy_enum,
            scope=scope_enum,
            max_requests=max_requests,
            window_seconds=window_seconds,
            burst_size=burst_size,
            dynamic_adjustment=dynamic_adjustment
        )

        return {
            "limit_id": limit.limit_id,
            "name": limit.name,
            "strategy": limit.strategy.value,
            "scope": limit.scope.value,
            "max_requests": limit.max_requests,
            "window_seconds": limit.window_seconds
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/rate-limits/{limit_id}/check")
async def check_rate_limit(
    limit_id: str,
    scope_key: str,
    tokens_requested: int = 1
):
    """Check rate limit"""
    try:
        allowed, retry_after = await rate_limit_manager.check_rate_limit(
            limit_id=limit_id,
            scope_key=scope_key,
            tokens_requested=tokens_requested
        )

        if not allowed:
            return {
                "allowed": False,
                "retry_after_seconds": retry_after
            }

        return {
            "allowed": True
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/rate-limits/{limit_id}/usage")
async def get_rate_limit_usage(
    limit_id: str,
    scope_key: str
):
    """Get current usage for rate limit"""
    try:
        usage = await rate_limit_manager.get_current_usage(
            limit_id=limit_id,
            scope_key=scope_key
        )
        return usage

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/rate-limits/violations")
async def get_violation_report(hours: int = 24):
    """Get rate limit violation report"""
    try:
        report = await rate_limit_manager.get_violation_report(hours=hours)
        return report

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# INVOICING & BILLING
# ============================================================================

@router.post("/invoices/generate")
async def generate_invoice(subscription_id: str):
    """Generate invoice for subscription"""
    try:
        invoice = await subscription_manager.generate_invoice(subscription_id)

        return {
            "invoice_id": invoice.invoice_id,
            "subscription_id": invoice.subscription_id,
            "subtotal": float(invoice.subtotal),
            "discount": float(invoice.discount),
            "tax": float(invoice.tax),
            "total": float(invoice.total),
            "status": invoice.status,
            "due_date": invoice.due_date.isoformat(),
            "line_items": invoice.line_items
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/invoices/{invoice_id}/pay")
async def pay_invoice(
    invoice_id: str,
    payment_method: str
):
    """Process payment for invoice"""
    try:
        payment_enum = PaymentMethod[payment_method.upper()]

        result = await subscription_manager.process_payment(
            invoice_id=invoice_id,
            payment_method=payment_enum
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
