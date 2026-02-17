"""Tenant repository interface."""

from abc import ABC, abstractmethod
from typing import Optional, List
from uuid import UUID

from .entities import Tenant


class TenantRepository(ABC):
    """Abstract repository for tenant persistence."""
    
    @abstractmethod
    async def create(self, tenant: Tenant) -> Tenant:
        """Create a new tenant."""
        pass
    
    @abstractmethod
    async def get_by_id(self, tenant_id: UUID) -> Optional[Tenant]:
        """Get tenant by ID."""
        pass
    
    @abstractmethod
    async def get_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get tenant by slug."""
        pass
    
    @abstractmethod
    async def get_by_owner(self, owner_user_id: UUID) -> List[Tenant]:
        """Get all tenants owned by a user."""
        pass
    
    @abstractmethod
    async def update(self, tenant: Tenant) -> Tenant:
        """Update an existing tenant."""
        pass
    
    @abstractmethod
    async def delete(self, tenant_id: UUID) -> bool:
        """Delete a tenant (soft delete recommended)."""
        pass
    
    @abstractmethod
    async def list_all(self, skip: int = 0, limit: int = 100) -> List[Tenant]:
        """List all tenants with pagination."""
        pass
    
    @abstractmethod
    async def exists_slug(self, slug: str) -> bool:
        """Check if slug already exists."""
        pass
