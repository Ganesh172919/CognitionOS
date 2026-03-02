"""
Billing v4 API.

Stable, tenant-scoped billing endpoints:
- Create checkout sessions (Stripe)
- Create billing portal sessions (Stripe)
- Read current subscription and usage summary
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.domain.billing.entities import SubscriptionTier
from core.domain.billing.services import EntitlementService
from infrastructure.billing.orchestrator import BillingOrchestrator
from infrastructure.persistence.billing_repository import (
    PostgreSQLSubscriptionRepository,
    PostgreSQLUsageRecordRepository,
)
from services.api.src.dependencies.injection import get_db_session


router = APIRouter(prefix="/api/v4/billing", tags=["Billing (v4)"])


def _require_tenant_id(request: Request) -> UUID:
    tenant_id = getattr(request.state, "tenant_id", None)
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context required (provide X-Tenant-Slug or authenticate via API key).",
        )
    return tenant_id


class CheckoutSessionRequest(BaseModel):
    tier: SubscriptionTier = Field(..., description="Target tier")
    billing_cycle: str = Field(default="monthly", description="Billing cycle (monthly/yearly)")
    success_url: Optional[str] = Field(default=None, description="Override default success URL")
    cancel_url: Optional[str] = Field(default=None, description="Override default cancel URL")
    trial_days: Optional[int] = Field(default=None, ge=0, le=365, description="Optional trial period in days")


class SessionResponse(BaseModel):
    id: str
    url: Optional[str] = None
    provider: str


class PortalSessionRequest(BaseModel):
    return_url: Optional[str] = Field(default=None, description="Override default return URL")


class SubscriptionSummary(BaseModel):
    tier: str
    status: str
    current_period_start: datetime
    current_period_end: datetime
    amount_cents: int
    currency: str
    billing_cycle: str
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None


class UsageMetric(BaseModel):
    resource_type: str
    current_usage: Decimal
    limit: Optional[Decimal] = None
    remaining: Optional[Decimal] = None


class UsageSummary(BaseModel):
    tenant_id: UUID
    period_start: datetime
    period_end: datetime
    tier: str
    metrics: Dict[str, UsageMetric]


@router.post("/checkout", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_checkout_session(
    request: Request,
    payload: CheckoutSessionRequest,
    session: AsyncSession = Depends(get_db_session),
) -> SessionResponse:
    tenant_id = _require_tenant_id(request)
    orchestrator = BillingOrchestrator(session)

    try:
        result = await orchestrator.create_checkout_session(
            tenant_id=tenant_id,
            tier=payload.tier,
            billing_cycle=payload.billing_cycle,
            success_url=payload.success_url,
            cancel_url=payload.cancel_url,
            trial_days=payload.trial_days,
        )
        return SessionResponse(id=str(result.get("id")), url=result.get("url"), provider=str(result.get("provider")))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/portal", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_billing_portal_session(
    request: Request,
    payload: PortalSessionRequest,
    session: AsyncSession = Depends(get_db_session),
) -> SessionResponse:
    tenant_id = _require_tenant_id(request)
    orchestrator = BillingOrchestrator(session)

    try:
        result = await orchestrator.create_billing_portal_session(tenant_id=tenant_id, return_url=payload.return_url)
        return SessionResponse(id=str(result.get("id")), url=result.get("url"), provider=str(result.get("provider")))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/subscription", response_model=Optional[SubscriptionSummary])
async def get_current_subscription(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> Optional[SubscriptionSummary]:
    tenant_id = _require_tenant_id(request)
    subscription_repo = PostgreSQLSubscriptionRepository(session)

    subscription = await subscription_repo.get_by_tenant(tenant_id)
    if not subscription:
        return None

    return SubscriptionSummary(
        tier=subscription.tier.value,
        status=subscription.status.value,
        current_period_start=subscription.current_period_start,
        current_period_end=subscription.current_period_end,
        amount_cents=subscription.amount_cents,
        currency=subscription.currency,
        billing_cycle=subscription.billing_cycle,
        stripe_customer_id=subscription.stripe_customer_id,
        stripe_subscription_id=subscription.stripe_subscription_id,
    )


@router.get("/usage", response_model=UsageSummary)
async def get_usage_summary(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> UsageSummary:
    tenant_id = _require_tenant_id(request)

    subscription_repo = PostgreSQLSubscriptionRepository(session)
    usage_repo = PostgreSQLUsageRecordRepository(session)
    entitlement = EntitlementService(subscription_repository=subscription_repo, usage_repository=usage_repo)

    subscription = await subscription_repo.get_by_tenant(tenant_id)
    if subscription and subscription.is_active():
        tier = subscription.tier
        period_start = subscription.current_period_start
        period_end = subscription.current_period_end
    else:
        # Fallback if subscription is not present/active.
        tier_value = getattr(getattr(request.state, "tenant", None), "subscription_tier", None) or "free"
        try:
            tier = SubscriptionTier(str(tier_value))
        except Exception:
            tier = SubscriptionTier.FREE
        now = datetime.utcnow()
        period_start = datetime(now.year, now.month, 1)
        period_end = now

    metrics: Dict[str, UsageMetric] = {}
    for resource_type in ["executions", "tokens", "api_calls", "storage_mb", "agents", "workflows"]:
        check = await entitlement.check_entitlement(
            tenant_id=tenant_id,
            resource_type=resource_type,
            quantity=Decimal("0"),
        )
        metrics[resource_type] = UsageMetric(
            resource_type=resource_type,
            current_usage=check.current_usage or Decimal("0"),
            limit=check.limit,
            remaining=check.remaining,
        )

    return UsageSummary(
        tenant_id=tenant_id,
        period_start=period_start,
        period_end=period_end,
        tier=tier.value,
        metrics=metrics,
    )
