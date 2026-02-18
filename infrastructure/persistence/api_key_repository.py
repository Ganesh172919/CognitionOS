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
            if datetime.utcnow() > api_key.expires_at:
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
            api_key.last_used_at = datetime.utcnow()
            await self.session.flush()
