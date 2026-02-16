"""
Authentication Repository Implementation

PostgreSQL implementation of UserRepository.
"""

from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.domain.auth.entities import User, UserStatus
from core.domain.auth.repositories import UserRepository
from infrastructure.persistence.auth_models import UserModel


class PostgresUserRepository(UserRepository):
    """PostgreSQL implementation of UserRepository"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def find_by_id(self, user_id: UUID) -> Optional[User]:
        """Find user by ID"""
        stmt = select(UserModel).where(UserModel.user_id == user_id)
        result = await self.session.execute(stmt)
        user_model = result.scalar_one_or_none()
        
        if user_model is None:
            return None
        
        return self._to_entity(user_model)
    
    async def find_by_email(self, email: str) -> Optional[User]:
        """Find user by email"""
        stmt = select(UserModel).where(UserModel.email == email.lower())
        result = await self.session.execute(stmt)
        user_model = result.scalar_one_or_none()
        
        if user_model is None:
            return None
        
        return self._to_entity(user_model)
    
    async def create(self, user: User) -> User:
        """Create a new user"""
        user_model = self._to_model(user)
        self.session.add(user_model)
        await self.session.flush()
        await self.session.refresh(user_model)
        
        return self._to_entity(user_model)
    
    async def update(self, user: User) -> User:
        """Update an existing user"""
        stmt = select(UserModel).where(UserModel.user_id == user.user_id)
        result = await self.session.execute(stmt)
        user_model = result.scalar_one_or_none()
        
        if user_model is None:
            raise ValueError(f"User not found: {user.user_id}")
        
        # Update fields
        user_model.email = user.email
        user_model.password_hash = user.password_hash
        user_model.full_name = user.full_name
        user_model.roles = user.roles
        user_model.status = user.status
        user_model.email_verified = user.email_verified
        user_model.failed_login_attempts = user.failed_login_attempts
        user_model.locked_until = user.locked_until
        user_model.last_login_at = user.last_login_at
        user_model.updated_at = user.updated_at
        
        await self.session.flush()
        await self.session.refresh(user_model)
        
        return self._to_entity(user_model)
    
    async def delete(self, user_id: UUID) -> bool:
        """Delete a user"""
        stmt = select(UserModel).where(UserModel.user_id == user_id)
        result = await self.session.execute(stmt)
        user_model = result.scalar_one_or_none()
        
        if user_model is None:
            return False
        
        await self.session.delete(user_model)
        await self.session.flush()
        
        return True
    
    async def exists_by_email(self, email: str) -> bool:
        """Check if user exists by email"""
        stmt = select(UserModel.user_id).where(UserModel.email == email.lower())
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none() is not None
    
    def _to_entity(self, model: UserModel) -> User:
        """Convert model to entity"""
        return User(
            user_id=model.user_id,
            email=model.email,
            password_hash=model.password_hash,
            full_name=model.full_name,
            roles=model.roles or ["user"],
            status=model.status,
            email_verified=model.email_verified,
            failed_login_attempts=model.failed_login_attempts,
            locked_until=model.locked_until,
            last_login_at=model.last_login_at,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
    
    def _to_model(self, entity: User) -> UserModel:
        """Convert entity to model"""
        return UserModel(
            user_id=entity.user_id,
            email=entity.email,
            password_hash=entity.password_hash,
            full_name=entity.full_name,
            roles=entity.roles,
            status=entity.status,
            email_verified=entity.email_verified,
            failed_login_attempts=entity.failed_login_attempts,
            locked_until=entity.locked_until,
            last_login_at=entity.last_login_at,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )
