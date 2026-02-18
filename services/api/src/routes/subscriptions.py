"""
Subscription API Routes

Provides REST endpoints for subscription and billing management.
"""

import os
from datetime import datetime
from decimal import Decimal
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.src.auth.dependencies import CurrentUser, get_current_user
from services.api.src.dependencies.injection import get_db_session
from infrastructure.middleware.tenant_context import get_current_tenant
from core.domain.billing.entities import Subscription, SubscriptionStatus, SubscriptionTier, Invoice, InvoiceStatus


router = APIRouter(prefix="/api/v3/subscriptions", tags=["Subscriptions"])


# ==================== Request/Response Schemas ====================

class PaymentMethodSchema(BaseModel):
    """Payment method schema"""
    id: str = Field(..., description="Payment method ID")
    type: str = Field(..., description="Payment method type")
    last4: str = Field(..., description="Last 4 digits")
    brand: Optional[str] = Field(default=None, description="Card brand")
    exp_month: Optional[int] = Field(default=None, description="Expiration month")
    exp_year: Optional[int] = Field(default=None, description="Expiration year")
    is_default: bool = Field(default=False, description="Is default payment method")


class SubscriptionResponse(BaseModel):
    """Subscription response"""
    id: str = Field(..., description="Subscription ID")
    tenant_id: str = Field(..., description="Tenant ID")
    tier: str = Field(..., description="Subscription tier")
    status: str = Field(..., description="Subscription status")
    current_period_start: datetime = Field(..., description="Current billing period start")
    current_period_end: datetime = Field(..., description="Current billing period end")
    trial_start: Optional[datetime] = Field(default=None, description="Trial start date")
    trial_end: Optional[datetime] = Field(default=None, description="Trial end date")
    canceled_at: Optional[datetime] = Field(default=None, description="Cancellation date")
    cancel_at_period_end: bool = Field(..., description="Cancel at period end flag")
    amount_cents: int = Field(..., description="Amount in cents")
    currency: str = Field(..., description="Currency code")
    billing_cycle: str = Field(..., description="Billing cycle")
    payment_method: Optional[PaymentMethodSchema] = Field(default=None, description="Payment method")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class UpgradeTierRequest(BaseModel):
    """Request to upgrade subscription tier"""
    tier: str = Field(..., description="Target tier (pro, team, enterprise)")
    billing_cycle: str = Field(default="monthly", description="Billing cycle (monthly, yearly)")


class DowngradeTierRequest(BaseModel):
    """Request to downgrade subscription tier"""
    tier: str = Field(..., description="Target tier (free, pro, team)")
    reason: Optional[str] = Field(default=None, description="Downgrade reason")


class UsageMetricSchema(BaseModel):
    """Usage metric schema"""
    metric_name: str = Field(..., description="Metric name")
    current_value: int = Field(..., description="Current usage")
    limit: int = Field(..., description="Usage limit")
    percentage: float = Field(..., description="Usage percentage")
    unit: str = Field(..., description="Measurement unit")


class UsageResponse(BaseModel):
    """Current usage response"""
    tenant_id: str = Field(..., description="Tenant ID")
    period_start: datetime = Field(..., description="Current period start")
    period_end: datetime = Field(..., description="Current period end")
    metrics: List[UsageMetricSchema] = Field(..., description="Usage metrics")
    total_cost_cents: int = Field(..., description="Total cost in cents")


class InvoiceLineItemSchema(BaseModel):
    """Invoice line item schema"""
    description: str = Field(..., description="Item description")
    quantity: int = Field(..., description="Quantity")
    unit_amount_cents: int = Field(..., description="Unit amount in cents")
    amount_cents: int = Field(..., description="Total amount in cents")


class InvoiceResponse(BaseModel):
    """Invoice response"""
    id: str = Field(..., description="Invoice ID")
    subscription_id: str = Field(..., description="Subscription ID")
    tenant_id: str = Field(..., description="Tenant ID")
    status: str = Field(..., description="Invoice status")
    number: str = Field(..., description="Invoice number")
    total_cents: int = Field(..., description="Total amount in cents")
    subtotal_cents: int = Field(..., description="Subtotal in cents")
    tax_cents: int = Field(..., description="Tax amount in cents")
    currency: str = Field(..., description="Currency code")
    due_date: datetime = Field(..., description="Due date")
    paid_at: Optional[datetime] = Field(default=None, description="Payment date")
    line_items: List[InvoiceLineItemSchema] = Field(..., description="Invoice line items")
    created_at: datetime = Field(..., description="Creation timestamp")
    pdf_url: Optional[str] = Field(default=None, description="PDF download URL")


class InvoiceListResponse(BaseModel):
    """Invoice list response"""
    invoices: List[InvoiceResponse] = Field(..., description="List of invoices")
    total: int = Field(..., description="Total invoice count")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Page size")


class SuccessResponse(BaseModel):
    """Generic success response"""
    success: bool = Field(default=True, description="Success flag")
    message: str = Field(..., description="Success message")


# ==================== Database Dependencies ====================

async def get_subscription_repository(session: AsyncSession = Depends(get_db_session)):
    """Get subscription repository"""
    from infrastructure.persistence.billing_repository import PostgreSQLSubscriptionRepository
    return PostgreSQLSubscriptionRepository(session)


async def get_invoice_repository(session: AsyncSession = Depends(get_db_session)):
    """Get invoice repository"""
    from infrastructure.persistence.billing_repository import PostgreSQLInvoiceRepository
    return PostgreSQLInvoiceRepository(session)


async def get_usage_repository(session: AsyncSession = Depends(get_db_session)):
    """Get usage repository"""
    from infrastructure.persistence.billing_repository import PostgreSQLUsageRecordRepository
    return PostgreSQLUsageRecordRepository(session)


# ==================== Subscription Management Endpoints ====================

@router.get(
    "/current",
    response_model=SubscriptionResponse,
    summary="Get current subscription",
    description="Retrieve the current subscription for the authenticated tenant",
)
async def get_current_subscription(
    current_user: CurrentUser = Depends(get_current_user),
    subscription_repo = Depends(get_subscription_repository),
) -> SubscriptionResponse:
    """Get current subscription"""
    
    tenant = get_current_tenant()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context is required"
        )
    
    subscription = await subscription_repo.get_by_tenant(tenant.id)
    
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active subscription found for this tenant"
        )
    
    # Convert payment method if exists
    payment_method_schema = None
    if subscription.payment_method:
        payment_method_schema = PaymentMethodSchema(
            id=subscription.payment_method.id,
            type=subscription.payment_method.type,
            last4=subscription.payment_method.last4,
            brand=subscription.payment_method.brand,
            exp_month=subscription.payment_method.exp_month,
            exp_year=subscription.payment_method.exp_year,
            is_default=subscription.payment_method.is_default,
        )
    
    return SubscriptionResponse(
        id=str(subscription.id),
        tenant_id=str(subscription.tenant_id),
        tier=subscription.tier.value,
        status=subscription.status.value,
        current_period_start=subscription.current_period_start,
        current_period_end=subscription.current_period_end,
        trial_start=subscription.trial_start,
        trial_end=subscription.trial_end,
        canceled_at=subscription.canceled_at,
        cancel_at_period_end=subscription.cancel_at_period_end,
        amount_cents=subscription.amount_cents,
        currency=subscription.currency,
        billing_cycle=subscription.billing_cycle,
        payment_method=payment_method_schema,
        created_at=subscription.created_at,
        updated_at=subscription.updated_at,
    )


@router.post(
    "/upgrade",
    response_model=SubscriptionResponse,
    summary="Upgrade subscription tier",
    description="Upgrade to a higher subscription tier",
)
async def upgrade_subscription(
    request: UpgradeTierRequest,
    current_user: CurrentUser = Depends(get_current_user),
    subscription_repo = Depends(get_subscription_repository),
    session: AsyncSession = Depends(get_db_session),
) -> SubscriptionResponse:
    """Upgrade subscription tier"""
    
    tenant = get_current_tenant()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context is required"
        )
    
    subscription = await subscription_repo.get_by_tenant(tenant.id)
    
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active subscription found"
        )
    
    # Validate tier upgrade
    tier_order = {"free": 0, "pro": 1, "team": 2, "enterprise": 3}
    current_tier_level = tier_order.get(subscription.tier.value, 0)
    target_tier_level = tier_order.get(request.tier, 0)
    
    if target_tier_level <= current_tier_level:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Target tier must be higher than current tier"
        )
    
    # Update subscription (in real implementation, this would call Stripe)
    subscription.tier = SubscriptionTier(request.tier)
    subscription.billing_cycle = request.billing_cycle
    subscription.updated_at = datetime.utcnow()
    
    # Update amount based on tier (simplified pricing)
    tier_pricing = {
        "pro": 2900 if request.billing_cycle == "monthly" else 29000,
        "team": 9900 if request.billing_cycle == "monthly" else 99000,
        "enterprise": 49900 if request.billing_cycle == "monthly" else 499000,
    }
    subscription.amount_cents = tier_pricing.get(request.tier, 0)
    
    # Save changes
    updated_subscription = await subscription_repo.update(subscription)
    await session.commit()
    
    return SubscriptionResponse(
        id=str(updated_subscription.id),
        tenant_id=str(updated_subscription.tenant_id),
        tier=updated_subscription.tier.value,
        status=updated_subscription.status.value,
        current_period_start=updated_subscription.current_period_start,
        current_period_end=updated_subscription.current_period_end,
        trial_start=updated_subscription.trial_start,
        trial_end=updated_subscription.trial_end,
        canceled_at=updated_subscription.canceled_at,
        cancel_at_period_end=updated_subscription.cancel_at_period_end,
        amount_cents=updated_subscription.amount_cents,
        currency=updated_subscription.currency,
        billing_cycle=updated_subscription.billing_cycle,
        payment_method=None,
        created_at=updated_subscription.created_at,
        updated_at=updated_subscription.updated_at,
    )


@router.post(
    "/downgrade",
    response_model=SubscriptionResponse,
    summary="Downgrade subscription tier",
    description="Downgrade to a lower subscription tier",
)
async def downgrade_subscription(
    request: DowngradeTierRequest,
    current_user: CurrentUser = Depends(get_current_user),
    subscription_repo = Depends(get_subscription_repository),
    session: AsyncSession = Depends(get_db_session),
) -> SubscriptionResponse:
    """Downgrade subscription tier"""
    
    tenant = get_current_tenant()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context is required"
        )
    
    subscription = await subscription_repo.get_by_tenant(tenant.id)
    
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active subscription found"
        )
    
    # Validate tier downgrade
    tier_order = {"free": 0, "pro": 1, "team": 2, "enterprise": 3}
    current_tier_level = tier_order.get(subscription.tier.value, 0)
    target_tier_level = tier_order.get(request.tier, 0)
    
    if target_tier_level >= current_tier_level:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Target tier must be lower than current tier"
        )
    
    # Schedule downgrade at period end (don't downgrade immediately)
    subscription.tier = SubscriptionTier(request.tier)
    subscription.cancel_at_period_end = True
    subscription.updated_at = datetime.utcnow()
    
    # Update amount based on tier
    tier_pricing = {
        "free": 0,
        "pro": 2900,
        "team": 9900,
    }
    subscription.amount_cents = tier_pricing.get(request.tier, 0)
    
    # Save changes
    updated_subscription = await subscription_repo.update(subscription)
    await session.commit()
    
    return SubscriptionResponse(
        id=str(updated_subscription.id),
        tenant_id=str(updated_subscription.tenant_id),
        tier=updated_subscription.tier.value,
        status=updated_subscription.status.value,
        current_period_start=updated_subscription.current_period_start,
        current_period_end=updated_subscription.current_period_end,
        trial_start=updated_subscription.trial_start,
        trial_end=updated_subscription.trial_end,
        canceled_at=updated_subscription.canceled_at,
        cancel_at_period_end=updated_subscription.cancel_at_period_end,
        amount_cents=updated_subscription.amount_cents,
        currency=updated_subscription.currency,
        billing_cycle=updated_subscription.billing_cycle,
        payment_method=None,
        created_at=updated_subscription.created_at,
        updated_at=updated_subscription.updated_at,
    )


@router.post(
    "/cancel",
    response_model=SuccessResponse,
    summary="Cancel subscription",
    description="Cancel the current subscription",
)
async def cancel_subscription(
    current_user: CurrentUser = Depends(get_current_user),
    subscription_repo = Depends(get_subscription_repository),
    session: AsyncSession = Depends(get_db_session),
) -> SuccessResponse:
    """Cancel subscription"""
    
    tenant = get_current_tenant()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context is required"
        )
    
    subscription = await subscription_repo.get_by_tenant(tenant.id)
    
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active subscription found"
        )
    
    if subscription.status == SubscriptionStatus.CANCELED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Subscription is already canceled"
        )
    
    # Cancel at period end
    subscription.cancel_at_period_end = True
    subscription.canceled_at = datetime.utcnow()
    subscription.updated_at = datetime.utcnow()
    
    # Save changes
    await subscription_repo.update(subscription)
    await session.commit()
    
    return SuccessResponse(
        success=True,
        message=f"Subscription will be canceled at the end of the billing period ({subscription.current_period_end.strftime('%Y-%m-%d')})"
    )


@router.get(
    "/usage",
    response_model=UsageResponse,
    summary="Get current usage",
    description="Retrieve current resource usage for the billing period",
)
async def get_current_usage(
    current_user: CurrentUser = Depends(get_current_user),
    subscription_repo = Depends(get_subscription_repository),
    usage_repo = Depends(get_usage_repository),
) -> UsageResponse:
    """Get current usage"""
    
    tenant = get_current_tenant()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context is required"
        )
    
    subscription = await subscription_repo.get_by_tenant(tenant.id)
    
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active subscription found"
        )
    
    # Get usage records for current period
    usage_records = await usage_repo.get_by_tenant(
        tenant.id,
        start_time=subscription.current_period_start,
        end_time=subscription.current_period_end,
    )
    
    # Aggregate usage by resource type
    usage_by_type = {}
    for record in usage_records:
        if record.resource_type not in usage_by_type:
            usage_by_type[record.resource_type] = 0
        usage_by_type[record.resource_type] += int(record.quantity)
    
    # Build metrics based on tenant settings
    metrics = [
        UsageMetricSchema(
            metric_name="workflow_executions",
            current_value=usage_by_type.get("workflow_execution", 0),
            limit=tenant.settings.max_executions_per_month,
            percentage=(usage_by_type.get("workflow_execution", 0) / tenant.settings.max_executions_per_month * 100) if tenant.settings.max_executions_per_month > 0 else 0,
            unit="executions"
        ),
        UsageMetricSchema(
            metric_name="storage",
            current_value=usage_by_type.get("storage", 0),
            limit=tenant.settings.max_storage_gb * 1024 * 1024 * 1024,
            percentage=(usage_by_type.get("storage", 0) / (tenant.settings.max_storage_gb * 1024 * 1024 * 1024) * 100) if tenant.settings.max_storage_gb > 0 else 0,
            unit="bytes"
        ),
        UsageMetricSchema(
            metric_name="api_calls",
            current_value=usage_by_type.get("api_call", 0),
            limit=tenant.settings.api_rate_limit_per_minute * 60 * 24 * 30,
            percentage=(usage_by_type.get("api_call", 0) / (tenant.settings.api_rate_limit_per_minute * 60 * 24 * 30) * 100) if tenant.settings.api_rate_limit_per_minute > 0 else 0,
            unit="calls"
        ),
    ]
    
    return UsageResponse(
        tenant_id=str(tenant.id),
        period_start=subscription.current_period_start,
        period_end=subscription.current_period_end,
        metrics=metrics,
        total_cost_cents=subscription.amount_cents,
    )


@router.get(
    "/invoices",
    response_model=InvoiceListResponse,
    summary="List invoices",
    description="List all invoices for the current tenant",
)
async def list_invoices(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    current_user: CurrentUser = Depends(get_current_user),
    invoice_repo = Depends(get_invoice_repository),
) -> InvoiceListResponse:
    """List invoices"""
    
    tenant = get_current_tenant()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context is required"
        )
    
    # Get invoices for tenant
    offset = (page - 1) * page_size
    invoices = await invoice_repo.get_by_tenant(
        tenant.id,
        limit=page_size,
        offset=offset,
    )
    
    # Convert to response schemas
    invoice_responses = []
    for invoice in invoices:
        # Build line items
        line_items = [
            InvoiceLineItemSchema(
                description=item["description"],
                quantity=item["quantity"],
                unit_amount_cents=item["unit_amount_cents"],
                amount_cents=item["amount_cents"],
            )
            for item in invoice.line_items
        ]
        
        invoice_responses.append(
            InvoiceResponse(
                id=str(invoice.id),
                subscription_id=str(invoice.subscription_id),
                tenant_id=str(invoice.tenant_id),
                status=invoice.status.value,
                number=invoice.number,
                total_cents=invoice.total_cents,
                subtotal_cents=invoice.subtotal_cents,
                tax_cents=invoice.tax_cents,
                currency=invoice.currency,
                due_date=invoice.due_date,
                paid_at=invoice.paid_at,
                line_items=line_items,
                created_at=invoice.created_at,
                pdf_url=invoice.pdf_url,
            )
        )
    
    # Get total count (simplified - in real implementation, use a count query)
    total = len(invoices)
    
    return InvoiceListResponse(
        invoices=invoice_responses,
        total=total,
        page=page,
        page_size=page_size,
    )
