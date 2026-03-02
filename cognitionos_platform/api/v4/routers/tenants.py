"""
Tenants v4 API.

Self-serve tenant onboarding endpoints intended for the SaaS control plane:
- Create a tenant for the authenticated user
- List tenants owned by the authenticated user

This module is additive and does not remove or alter existing v3 behavior.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_config
from core.domain.billing.entities import Subscription, SubscriptionTier
from core.domain.tenant.entities import Tenant
from infrastructure.persistence.billing_repository import PostgreSQLSubscriptionRepository
from infrastructure.persistence.tenant_repository import PostgreSQLTenantRepository
from services.api.src.auth.dependencies import CurrentUser, get_current_user
from services.api.src.dependencies.injection import get_db_session


router = APIRouter(prefix="/api/v4/tenants", tags=["Tenants (v4)"])
config = get_config()


def _safe_uuid(value: str) -> Optional[UUID]:
    try:
        return UUID(str(value))
    except Exception:
        return None


def _require_owner_or_admin(current_user: CurrentUser, tenant: Tenant) -> None:
    if current_user.has_role("admin"):
        return
    user_uuid = _safe_uuid(current_user.user_id)
    if not user_uuid or not tenant.owner_user_id or tenant.owner_user_id != user_uuid:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied to this tenant.")


class CreateTenantRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(
        ...,
        min_length=3,
        max_length=100,
        pattern=r"^[a-z][a-z0-9-]{2,99}$",
        description="URL-friendly slug (lowercase letters, digits, hyphens).",
    )
    billing_email: Optional[str] = Field(default=None, max_length=255)
    create_trial: bool = Field(default=True, description="Create a trial subscription if enabled by server policy.")
    trial_tier: SubscriptionTier = Field(default=SubscriptionTier.PRO, description="Trial tier when create_trial=true.")
    trial_days: Optional[int] = Field(default=None, ge=1, le=365)


class TenantResponse(BaseModel):
    id: UUID
    name: str
    slug: str
    status: str
    subscription_tier: str
    owner_user_id: Optional[UUID] = None
    billing_email: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    trial_ends_at: Optional[datetime] = None

    @staticmethod
    def from_entity(tenant: Tenant) -> "TenantResponse":
        return TenantResponse(
            id=tenant.id,
            name=tenant.name,
            slug=tenant.slug,
            status=tenant.status.value,
            subscription_tier=str(tenant.subscription_tier),
            owner_user_id=tenant.owner_user_id,
            billing_email=tenant.billing_email,
            created_at=tenant.created_at,
            updated_at=tenant.updated_at,
            trial_ends_at=tenant.trial_ends_at,
        )


@router.post("", response_model=TenantResponse, status_code=status.HTTP_201_CREATED)
async def create_tenant(
    request: Request,
    payload: CreateTenantRequest,
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> TenantResponse:
    tenant_repo = PostgreSQLTenantRepository(session)
    subscription_repo = PostgreSQLSubscriptionRepository(session)

    if await tenant_repo.exists_slug(payload.slug):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tenant with slug '{payload.slug}' already exists.",
        )

    owner_user_id = _safe_uuid(current_user.user_id)
    if not owner_user_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user id in token.")

    billing_email = payload.billing_email or current_user.email
    if not billing_email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="billing_email is required.")

    # Default tier is free; a trial subscription can elevate effective tier via entitlements.
    tenant = Tenant.create(
        name=payload.name,
        slug=payload.slug,
        owner_user_id=owner_user_id,
        billing_email=billing_email,
        subscription_tier=SubscriptionTier.FREE.value,
    )
    created = await tenant_repo.create(tenant)

    # Optionally create a trial subscription at signup.
    if payload.create_trial and config.billing.enable_trials:
        trial_days = int(payload.trial_days or config.billing.default_trial_days or 14)
        trial_subscription = Subscription.create_trial(
            tenant_id=created.id,
            tier=payload.trial_tier,
            trial_days=trial_days,
        )
        await subscription_repo.create(trial_subscription)

        # Keep tenant tier and trial metadata in sync for fast UI access.
        created.upgrade_tier(trial_subscription.tier.value)
        created.trial_ends_at = trial_subscription.trial_end
        created.updated_at = datetime.utcnow()
        created = await tenant_repo.update(created)

    await session.commit()
    # Attach tenant context for the rest of the request lifecycle (useful for immediate follow-on calls).
    request.state.tenant = created
    request.state.tenant_id = created.id

    return TenantResponse.from_entity(created)


@router.get("", response_model=List[TenantResponse])
async def list_tenants(
    current_user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    all: bool = False,
) -> List[TenantResponse]:
    tenant_repo = PostgreSQLTenantRepository(session)

    if all and current_user.has_role("admin"):
        tenants = await tenant_repo.list_all(limit=200)
        return [TenantResponse.from_entity(t) for t in tenants]

    owner_user_id = _safe_uuid(current_user.user_id)
    if not owner_user_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user id in token.")

    tenants = await tenant_repo.get_by_owner(owner_user_id)
    # Enforce owner/admin access even if repository returns unexpected rows.
    for t in tenants:
        _require_owner_or_admin(current_user, t)

    return [TenantResponse.from_entity(t) for t in tenants]
