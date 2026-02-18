"""
Authentication SQLAlchemy Models

Database models for authentication entities.
"""

from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Integer, ARRAY, Enum as SQLEnum, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
import uuid

from infrastructure.persistence.base import Base
from core.domain.auth.entities import UserStatus


class UserModel(Base):
    """SQLAlchemy model for User entity"""
    
    __tablename__ = "users"
    
    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    roles = Column(ARRAY(String), nullable=False, default=["user"])
    status = Column(
        SQLEnum(UserStatus, name="user_status_enum", create_type=True),
        nullable=False,
        default=UserStatus.ACTIVE,
        index=True
    )
    email_verified = Column(Boolean, nullable=False, default=False)
    failed_login_attempts = Column(Integer, nullable=False, default=0)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<UserModel(user_id={self.user_id}, email={self.email}, status={self.status})>"


class APIKeyModel(Base):
    """SQLAlchemy model for API Key entity"""
    
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey('tenants.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Key details
    key_hash = Column(String(255), nullable=False, unique=True, index=True)
    key_prefix = Column(String(20), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    
    # Permissions
    scopes = Column(ARRAY(String), nullable=False, default=list)
    
    # Rate limiting
    rate_limit_per_minute = Column(Integer, default=60)
    
    # Status
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    
    # Expiration
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Audit
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True), nullable=True)
    
    def __repr__(self):
        return f"<APIKeyModel(id={self.id}, name={self.name}, tenant_id={self.tenant_id}, is_active={self.is_active})>"
