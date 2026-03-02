"""
API Key Repository Implementation

PostgreSQL implementation for API key management.
"""

from typing import Optional
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.persistence.auth_models import APIKeyModel


class PostgresAPIKeyRepository:
    """PostgreSQL implementation of API Key Repository"""
    
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        *,
        tenant_id: UUID,
        name: str,
        key_hash: str,
        key_prefix: str,
        scopes: list[str],
        rate_limit_per_minute: int = 60,
        expires_at: Optional[datetime] = None,
        created_by: Optional[UUID] = None,
    ) -> APIKeyModel:
        """Create and persist a new API key record (hash only)."""
        api_key = APIKeyModel(
            tenant_id=tenant_id,
            name=name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            scopes=scopes,
            rate_limit_per_minute=rate_limit_per_minute,
            expires_at=expires_at,
            created_by=created_by,
            is_active=True,
        )
        self.session.add(api_key)
        await self.session.flush()
        await self.session.refresh(api_key)
        return api_key

    async def get_by_id(self, api_key_id: UUID) -> Optional[APIKeyModel]:
        """Retrieve API key by ID (active or inactive)."""
        stmt = select(APIKeyModel).where(APIKeyModel.id == api_key_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_tenant(
        self,
        tenant_id: UUID,
        *,
        include_inactive: bool = False,
        limit: int = 100,
    ) -> list[APIKeyModel]:
        """List API keys for a tenant."""
        stmt = select(APIKeyModel).where(APIKeyModel.tenant_id == tenant_id)
        if not include_inactive:
            stmt = stmt.where(APIKeyModel.is_active == True)
        stmt = stmt.order_by(APIKeyModel.created_at.desc()).limit(limit)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def revoke(self, api_key_id: UUID, *, tenant_id: Optional[UUID] = None) -> bool:
        """Revoke (deactivate) an API key by ID."""
        stmt = select(APIKeyModel).where(APIKeyModel.id == api_key_id)
        if tenant_id is not None:
            stmt = stmt.where(APIKeyModel.tenant_id == tenant_id)

        result = await self.session.execute(stmt)
        api_key = result.scalar_one_or_none()
        if not api_key:
            return False

        api_key.is_active = False
        await self.session.flush()
        return True
    
    async def get_by_hash(self, key_hash: str) -> Optional[APIKeyModel]:
        """
        Find API key by its hash.
        
        Args:
            key_hash: The hashed API key
            
        Returns:
            APIKeyModel if found and active, None otherwise
        """
        stmt = select(APIKeyModel).where(
            APIKeyModel.key_hash == key_hash,
            APIKeyModel.is_active == True
        )
        result = await self.session.execute(stmt)
        api_key = result.scalar_one_or_none()
        
        # Check expiration
        if api_key and api_key.expires_at:
            from datetime import timezone
            if datetime.now(timezone.utc) > api_key.expires_at:
                return None
        
        return api_key
    
    async def get_by_prefix(self, key_prefix: str) -> Optional[APIKeyModel]:
        """
        Find API key by its prefix.
        
        Args:
            key_prefix: The key prefix (first chars after cog_)
            
        Returns:
            APIKeyModel if found, None otherwise
        """
        stmt = select(APIKeyModel).where(
            APIKeyModel.key_prefix == key_prefix,
            APIKeyModel.is_active == True
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def update_last_used(self, api_key_id: UUID) -> None:
        """
        Update the last_used_at timestamp for an API key.
        
        Args:
            api_key_id: The API key ID
        """
        stmt = select(APIKeyModel).where(APIKeyModel.id == api_key_id)
        result = await self.session.execute(stmt)
        api_key = result.scalar_one_or_none()
        
        if api_key:
            from datetime import timezone
            api_key.last_used_at = datetime.now(timezone.utc)
            await self.session.flush()
