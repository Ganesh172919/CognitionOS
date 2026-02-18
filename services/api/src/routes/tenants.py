"""
Tenant API Routes

Provides REST endpoints for tenant management in multi-tenant architecture.
"""

import os
from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.src.auth.dependencies import CurrentUser, get_current_user, require_role
from services.api.src.dependencies.injection import get_db_session
from infrastructure.persistence.tenant_repository import PostgreSQLTenantRepository
from infrastructure.middleware.tenant_context import get_current_tenant
from core.domain.tenant.entities import Tenant, TenantStatus, TenantSettings


router = APIRouter(prefix="/api/v3/tenants", tags=["Tenants"])


# ==================== Request/Response Schemas ====================

class TenantSettingsSchema(BaseModel):
    """Tenant settings schema"""
    max_users: int = Field(default=5, description="Maximum number of users")
    max_agents: int = Field(default=10, description="Maximum number of agents")
    max_workflows: int = Field(default=50, description="Maximum number of workflows")
    max_executions_per_month: int = Field(default=1000, description="Maximum monthly executions")
    max_storage_gb: int = Field(default=10, description="Maximum storage in GB")
    api_rate_limit_per_minute: int = Field(default=60, description="API rate limit per minute")
    enable_plugins: bool = Field(default=False, description="Enable plugin marketplace")
    enable_custom_models: bool = Field(default=False, description="Enable custom models")
    enable_priority_execution: bool = Field(default=False, description="Enable priority execution")
    custom_domain: Optional[str] = Field(default=None, description="Custom domain")
    webhook_url: Optional[str] = Field(default=None, description="Webhook URL")


class CreateTenantRequest(BaseModel):
    """Request to create a new tenant"""
    name: str = Field(..., min_length=1, max_length=255, description="Tenant name")
    slug: str = Field(..., min_length=1, max_length=100, description="URL-friendly slug")
    owner_user_id: str = Field(..., description="Owner user ID")
    billing_email: str = Field(..., description="Billing email address")
    subscription_tier: str = Field(default="free", description="Subscription tier")


class UpdateTenantRequest(BaseModel):
    """Request to update tenant"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=255, description="Tenant name")
    billing_email: Optional[str] = Field(default=None, description="Billing email address")


class UpdateSettingsRequest(BaseModel):
    """Request to update tenant settings"""
    settings: TenantSettingsSchema = Field(..., description="Tenant settings")


class SuspendTenantRequest(BaseModel):
    """Request to suspend tenant"""
    reason: str = Field(..., min_length=1, description="Suspension reason")


class TenantResponse(BaseModel):
    """Tenant response"""
    id: str = Field(..., description="Tenant ID")
    name: str = Field(..., description="Tenant name")
    slug: str = Field(..., description="Tenant slug")
    status: str = Field(..., description="Tenant status")
    subscription_tier: str = Field(..., description="Subscription tier")
    owner_user_id: Optional[str] = Field(default=None, description="Owner user ID")
    billing_email: Optional[str] = Field(default=None, description="Billing email")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    trial_ends_at: Optional[datetime] = Field(default=None, description="Trial end date")
    suspended_at: Optional[datetime] = Field(default=None, description="Suspension timestamp")
    suspended_reason: Optional[str] = Field(default=None, description="Suspension reason")


class TenantSettingsResponse(BaseModel):
    """Tenant settings response"""
    tenant_id: str = Field(..., description="Tenant ID")
    settings: TenantSettingsSchema = Field(..., description="Tenant settings")


class SuccessResponse(BaseModel):
    """Generic success response"""
    success: bool = Field(default=True, description="Success flag")
    message: str = Field(..., description="Success message")


# ==================== Database Dependency ====================

async def get_tenant_repository(session: AsyncSession = Depends(get_db_session)) -> PostgreSQLTenantRepository:
    """Get tenant repository"""
    return PostgreSQLTenantRepository(session)


# ==================== Tenant Management Endpoints ====================

@router.post(
    "",
    response_model=TenantResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create tenant (admin only)",
    description="Create a new tenant with specified configuration",
)
async def create_tenant(
    request: CreateTenantRequest,
    current_user: CurrentUser = Depends(require_role("admin")),
    tenant_repo: PostgreSQLTenantRepository = Depends(get_tenant_repository),
    session: AsyncSession = Depends(get_db_session),
) -> TenantResponse:
    """Create a new tenant"""
    
    # Check if slug already exists
    if await tenant_repo.exists_slug(request.slug):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tenant with slug '{request.slug}' already exists"
        )
    
    # Create tenant
    tenant = Tenant.create(
        name=request.name,
        slug=request.slug,
        owner_user_id=UUID(request.owner_user_id),
        billing_email=request.billing_email,
        subscription_tier=request.subscription_tier,
    )
    
    # Save to database
    created_tenant = await tenant_repo.create(tenant)
    await session.commit()
    
    return TenantResponse(
        id=str(created_tenant.id),
        name=created_tenant.name,
        slug=created_tenant.slug,
        status=created_tenant.status.value,
        subscription_tier=created_tenant.subscription_tier,
        owner_user_id=str(created_tenant.owner_user_id) if created_tenant.owner_user_id else None,
        billing_email=created_tenant.billing_email,
        created_at=created_tenant.created_at,
        updated_at=created_tenant.updated_at,
        trial_ends_at=created_tenant.trial_ends_at,
        suspended_at=created_tenant.suspended_at,
        suspended_reason=created_tenant.suspended_reason,
    )


@router.get(
    "/{tenant_id}",
    response_model=TenantResponse,
    summary="Get tenant",
    description="Retrieve tenant information by ID",
)
async def get_tenant(
    tenant_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    tenant_repo: PostgreSQLTenantRepository = Depends(get_tenant_repository),
) -> TenantResponse:
    """Get tenant by ID"""
    
    tenant = await tenant_repo.get_by_id(UUID(tenant_id))
    
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found"
        )
    
    # Check if user has access to this tenant
    current_tenant = get_current_tenant()
    if not current_user.has_role("admin") and (not current_tenant or str(current_tenant.id) != tenant_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this tenant"
        )
    
    return TenantResponse(
        id=str(tenant.id),
        name=tenant.name,
        slug=tenant.slug,
        status=tenant.status.value,
        subscription_tier=tenant.subscription_tier,
        owner_user_id=str(tenant.owner_user_id) if tenant.owner_user_id else None,
        billing_email=tenant.billing_email,
        created_at=tenant.created_at,
        updated_at=tenant.updated_at,
        trial_ends_at=tenant.trial_ends_at,
        suspended_at=tenant.suspended_at,
        suspended_reason=tenant.suspended_reason,
    )


@router.put(
    "/{tenant_id}",
    response_model=TenantResponse,
    summary="Update tenant",
    description="Update tenant information",
)
async def update_tenant(
    tenant_id: str,
    request: UpdateTenantRequest,
    current_user: CurrentUser = Depends(get_current_user),
    tenant_repo: PostgreSQLTenantRepository = Depends(get_tenant_repository),
    session: AsyncSession = Depends(get_db_session),
) -> TenantResponse:
    """Update tenant"""
    
    tenant = await tenant_repo.get_by_id(UUID(tenant_id))
    
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found"
        )
    
    # Check if user has access to this tenant
    current_tenant = get_current_tenant()
    if not current_user.has_role("admin") and (not current_tenant or str(current_tenant.id) != tenant_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this tenant"
        )
    
    # Update fields
    if request.name is not None:
        tenant.name = request.name
    if request.billing_email is not None:
        tenant.billing_email = request.billing_email
    
    tenant.updated_at = datetime.utcnow()
    
    # Save changes
    updated_tenant = await tenant_repo.update(tenant)
    await session.commit()
    
    return TenantResponse(
        id=str(updated_tenant.id),
        name=updated_tenant.name,
        slug=updated_tenant.slug,
        status=updated_tenant.status.value,
        subscription_tier=updated_tenant.subscription_tier,
        owner_user_id=str(updated_tenant.owner_user_id) if updated_tenant.owner_user_id else None,
        billing_email=updated_tenant.billing_email,
        created_at=updated_tenant.created_at,
        updated_at=updated_tenant.updated_at,
        trial_ends_at=updated_tenant.trial_ends_at,
        suspended_at=updated_tenant.suspended_at,
        suspended_reason=updated_tenant.suspended_reason,
    )


@router.get(
    "/{tenant_id}/settings",
    response_model=TenantSettingsResponse,
    summary="Get tenant settings",
    description="Retrieve tenant-specific configuration settings",
)
async def get_tenant_settings(
    tenant_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    tenant_repo: PostgreSQLTenantRepository = Depends(get_tenant_repository),
) -> TenantSettingsResponse:
    """Get tenant settings"""
    
    tenant = await tenant_repo.get_by_id(UUID(tenant_id))
    
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found"
        )
    
    # Check if user has access to this tenant
    current_tenant = get_current_tenant()
    if not current_user.has_role("admin") and (not current_tenant or str(current_tenant.id) != tenant_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this tenant"
        )
    
    return TenantSettingsResponse(
        tenant_id=str(tenant.id),
        settings=TenantSettingsSchema(
            max_users=tenant.settings.max_users,
            max_agents=tenant.settings.max_agents,
            max_workflows=tenant.settings.max_workflows,
            max_executions_per_month=tenant.settings.max_executions_per_month,
            max_storage_gb=tenant.settings.max_storage_gb,
            api_rate_limit_per_minute=tenant.settings.api_rate_limit_per_minute,
            enable_plugins=tenant.settings.enable_plugins,
            enable_custom_models=tenant.settings.enable_custom_models,
            enable_priority_execution=tenant.settings.enable_priority_execution,
            custom_domain=tenant.settings.custom_domain,
            webhook_url=tenant.settings.webhook_url,
        )
    )


@router.put(
    "/{tenant_id}/settings",
    response_model=TenantSettingsResponse,
    summary="Update tenant settings",
    description="Update tenant-specific configuration settings",
)
async def update_tenant_settings(
    tenant_id: str,
    request: UpdateSettingsRequest,
    current_user: CurrentUser = Depends(require_role("admin")),
    tenant_repo: PostgreSQLTenantRepository = Depends(get_tenant_repository),
    session: AsyncSession = Depends(get_db_session),
) -> TenantSettingsResponse:
    """Update tenant settings"""
    
    tenant = await tenant_repo.get_by_id(UUID(tenant_id))
    
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found"
        )
    
    # Update settings
    tenant.settings = TenantSettings(
        max_users=request.settings.max_users,
        max_agents=request.settings.max_agents,
        max_workflows=request.settings.max_workflows,
        max_executions_per_month=request.settings.max_executions_per_month,
        max_storage_gb=request.settings.max_storage_gb,
        api_rate_limit_per_minute=request.settings.api_rate_limit_per_minute,
        enable_plugins=request.settings.enable_plugins,
        enable_custom_models=request.settings.enable_custom_models,
        enable_priority_execution=request.settings.enable_priority_execution,
        custom_domain=request.settings.custom_domain,
        webhook_url=request.settings.webhook_url,
    )
    
    tenant.updated_at = datetime.utcnow()
    
    # Save changes
    updated_tenant = await tenant_repo.update(tenant)
    await session.commit()
    
    return TenantSettingsResponse(
        tenant_id=str(updated_tenant.id),
        settings=TenantSettingsSchema(
            max_users=updated_tenant.settings.max_users,
            max_agents=updated_tenant.settings.max_agents,
            max_workflows=updated_tenant.settings.max_workflows,
            max_executions_per_month=updated_tenant.settings.max_executions_per_month,
            max_storage_gb=updated_tenant.settings.max_storage_gb,
            api_rate_limit_per_minute=updated_tenant.settings.api_rate_limit_per_minute,
            enable_plugins=updated_tenant.settings.enable_plugins,
            enable_custom_models=updated_tenant.settings.enable_custom_models,
            enable_priority_execution=updated_tenant.settings.enable_priority_execution,
            custom_domain=updated_tenant.settings.custom_domain,
            webhook_url=updated_tenant.settings.webhook_url,
        )
    )


@router.post(
    "/{tenant_id}/suspend",
    response_model=SuccessResponse,
    summary="Suspend tenant",
    description="Suspend a tenant and block access",
)
async def suspend_tenant(
    tenant_id: str,
    request: SuspendTenantRequest,
    current_user: CurrentUser = Depends(require_role("admin")),
    tenant_repo: PostgreSQLTenantRepository = Depends(get_tenant_repository),
    session: AsyncSession = Depends(get_db_session),
) -> SuccessResponse:
    """Suspend tenant"""
    
    tenant = await tenant_repo.get_by_id(UUID(tenant_id))
    
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found"
        )
    
    # Suspend tenant
    tenant.status = TenantStatus.SUSPENDED
    tenant.suspended_at = datetime.utcnow()
    tenant.suspended_reason = request.reason
    tenant.updated_at = datetime.utcnow()
    
    # Save changes
    await tenant_repo.update(tenant)
    await session.commit()
    
    return SuccessResponse(
        success=True,
        message=f"Tenant '{tenant.name}' has been suspended"
    )


@router.post(
    "/{tenant_id}/reactivate",
    response_model=SuccessResponse,
    summary="Reactivate tenant",
    description="Reactivate a suspended tenant",
)
async def reactivate_tenant(
    tenant_id: str,
    current_user: CurrentUser = Depends(require_role("admin")),
    tenant_repo: PostgreSQLTenantRepository = Depends(get_tenant_repository),
    session: AsyncSession = Depends(get_db_session),
) -> SuccessResponse:
    """Reactivate tenant"""
    
    tenant = await tenant_repo.get_by_id(UUID(tenant_id))
    
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found"
        )
    
    # Reactivate tenant
    tenant.status = TenantStatus.ACTIVE
    tenant.suspended_at = None
    tenant.suspended_reason = None
    tenant.updated_at = datetime.utcnow()
    
    # Save changes
    await tenant_repo.update(tenant)
    await session.commit()
    
    return SuccessResponse(
        success=True,
        message=f"Tenant '{tenant.name}' has been reactivated"
    )
