"""
Tenant Repository Implementation

PostgreSQL implementation of TenantRepository.
"""

import logging
from datetime import datetime
from typing import Optional, List
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.domain.tenant.entities import Tenant, TenantStatus, TenantSettings
from core.domain.tenant.repositories import TenantRepository
from infrastructure.persistence.tenant_models import TenantModel


logger = logging.getLogger(__name__)


class PostgreSQLTenantRepository(TenantRepository):
    """PostgreSQL implementation of TenantRepository"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, tenant: Tenant) -> Tenant:
        """Create a new tenant"""
        try:
            tenant_model = self._to_model(tenant)
            self.session.add(tenant_model)
            await self.session.flush()
            await self.session.refresh(tenant_model)
            
            logger.info(f"Created tenant: {tenant.id}")
            return self._to_entity(tenant_model)
        except Exception as e:
            logger.error(f"Error creating tenant {tenant.id}: {e}")
            raise
    
    async def get_by_id(self, tenant_id: UUID) -> Optional[Tenant]:
        """Get tenant by ID"""
        try:
            stmt = select(TenantModel).where(TenantModel.id == tenant_id)
            result = await self.session.execute(stmt)
            tenant_model = result.scalar_one_or_none()
            
            if tenant_model is None:
                return None
            
            return self._to_entity(tenant_model)
        except Exception as e:
            logger.error(f"Error fetching tenant {tenant_id}: {e}")
            raise
    
    async def get_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get tenant by slug"""
        try:
            stmt = select(TenantModel).where(TenantModel.slug == slug)
            result = await self.session.execute(stmt)
            tenant_model = result.scalar_one_or_none()
            
            if tenant_model is None:
                return None
            
            return self._to_entity(tenant_model)
        except Exception as e:
            logger.error(f"Error fetching tenant by slug {slug}: {e}")
            raise
    
    async def get_by_owner(self, owner_user_id: UUID) -> List[Tenant]:
        """Get all tenants owned by a user"""
        try:
            stmt = select(TenantModel).where(TenantModel.owner_user_id == owner_user_id)
            result = await self.session.execute(stmt)
            tenant_models = result.scalars().all()
            
            return [self._to_entity(model) for model in tenant_models]
        except Exception as e:
            logger.error(f"Error fetching tenants for owner {owner_user_id}: {e}")
            raise
    
    async def update(self, tenant: Tenant) -> Tenant:
        """Update an existing tenant"""
        try:
            stmt = select(TenantModel).where(TenantModel.id == tenant.id)
            result = await self.session.execute(stmt)
            tenant_model = result.scalar_one_or_none()
            
            if tenant_model is None:
                raise ValueError(f"Tenant not found: {tenant.id}")
            
            # Update fields
            tenant_model.name = tenant.name
            tenant_model.slug = tenant.slug
            tenant_model.status = tenant.status
            tenant_model.subscription_tier = tenant.subscription_tier
            tenant_model.settings = self._settings_to_dict(tenant.settings)
            tenant_model.owner_user_id = tenant.owner_user_id
            tenant_model.billing_email = tenant.billing_email
            tenant_model.trial_ends_at = tenant.trial_ends_at
            tenant_model.suspended_at = tenant.suspended_at
            tenant_model.suspended_reason = tenant.suspended_reason
            tenant_model.updated_at = tenant.updated_at
            tenant_model.metadata = tenant.metadata
            
            await self.session.flush()
            await self.session.refresh(tenant_model)
            
            logger.info(f"Updated tenant: {tenant.id}")
            return self._to_entity(tenant_model)
        except Exception as e:
            logger.error(f"Error updating tenant {tenant.id}: {e}")
            raise
    
    async def delete(self, tenant_id: UUID) -> bool:
        """Delete a tenant (soft delete recommended)"""
        try:
            stmt = select(TenantModel).where(TenantModel.id == tenant_id)
            result = await self.session.execute(stmt)
            tenant_model = result.scalar_one_or_none()
            
            if tenant_model is None:
                return False
            
            # Soft delete by setting status to churned
            tenant_model.status = TenantStatus.CHURNED
            tenant_model.updated_at = datetime.utcnow()
            
            await self.session.flush()
            
            logger.info(f"Deleted (soft) tenant: {tenant_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting tenant {tenant_id}: {e}")
            raise
    
    async def list_all(self, skip: int = 0, limit: int = 100) -> List[Tenant]:
        """List all tenants with pagination"""
        try:
            stmt = (
                select(TenantModel)
                .offset(skip)
                .limit(limit)
                .order_by(TenantModel.created_at.desc())
            )
            result = await self.session.execute(stmt)
            tenant_models = result.scalars().all()
            
            return [self._to_entity(model) for model in tenant_models]
        except Exception as e:
            logger.error(f"Error listing tenants: {e}")
            raise
    
    async def exists_slug(self, slug: str) -> bool:
        """Check if slug already exists"""
        try:
            stmt = select(TenantModel.id).where(TenantModel.slug == slug)
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none() is not None
        except Exception as e:
            logger.error(f"Error checking slug existence {slug}: {e}")
            raise
    
    def _to_entity(self, model: TenantModel) -> Tenant:
        """Convert model to entity"""
        settings = self._dict_to_settings(model.settings)
        
        return Tenant(
            id=model.id,
            name=model.name,
            slug=model.slug,
            status=model.status,
            settings=settings,
            subscription_tier=model.subscription_tier,
            created_at=model.created_at,
            updated_at=model.updated_at,
            trial_ends_at=model.trial_ends_at,
            suspended_at=model.suspended_at,
            suspended_reason=model.suspended_reason,
            owner_user_id=model.owner_user_id,
            billing_email=model.billing_email,
            metadata=model.metadata or {},
        )
    
    def _to_model(self, entity: Tenant) -> TenantModel:
        """Convert entity to model"""
        return TenantModel(
            id=entity.id,
            name=entity.name,
            slug=entity.slug,
            status=entity.status,
            subscription_tier=entity.subscription_tier,
            settings=self._settings_to_dict(entity.settings),
            owner_user_id=entity.owner_user_id,
            billing_email=entity.billing_email,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            trial_ends_at=entity.trial_ends_at,
            suspended_at=entity.suspended_at,
            suspended_reason=entity.suspended_reason,
            metadata=entity.metadata,
        )
    
    def _settings_to_dict(self, settings: TenantSettings) -> dict:
        """Convert TenantSettings to dict for JSON storage"""
        return {
            "max_users": settings.max_users,
            "max_agents": settings.max_agents,
            "max_workflows": settings.max_workflows,
            "max_executions_per_month": settings.max_executions_per_month,
            "max_storage_gb": settings.max_storage_gb,
            "api_rate_limit_per_minute": settings.api_rate_limit_per_minute,
            "enable_plugins": settings.enable_plugins,
            "enable_custom_models": settings.enable_custom_models,
            "enable_priority_execution": settings.enable_priority_execution,
            "custom_domain": settings.custom_domain,
            "webhook_url": settings.webhook_url,
            "metadata": settings.metadata,
        }
    
    def _dict_to_settings(self, data: dict) -> TenantSettings:
        """Convert dict to TenantSettings"""
        return TenantSettings(
            max_users=data.get("max_users", 5),
            max_agents=data.get("max_agents", 10),
            max_workflows=data.get("max_workflows", 50),
            max_executions_per_month=data.get("max_executions_per_month", 1000),
            max_storage_gb=data.get("max_storage_gb", 10),
            api_rate_limit_per_minute=data.get("api_rate_limit_per_minute", 60),
            enable_plugins=data.get("enable_plugins", False),
            enable_custom_models=data.get("enable_custom_models", False),
            enable_priority_execution=data.get("enable_priority_execution", False),
            custom_domain=data.get("custom_domain"),
            webhook_url=data.get("webhook_url"),
            metadata=data.get("metadata", {}),
        )
