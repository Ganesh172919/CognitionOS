"""
Tenant SQLAlchemy Models

Database models for tenant entities.
"""

from datetime import datetime
from sqlalchemy import Column, String, DateTime, Enum as SQLEnum, JSON
from sqlalchemy.dialects.postgresql import UUID
import uuid

from infrastructure.persistence.base import Base
from core.domain.tenant.entities import TenantStatus


class TenantModel(Base):
    """SQLAlchemy model for Tenant entity"""
    
    __tablename__ = "tenants"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(255), nullable=False, unique=True, index=True)
    status = Column(
        SQLEnum(TenantStatus, name="tenant_status_enum", create_type=False),
        nullable=False,
        index=True
    )
    subscription_tier = Column(String(50), nullable=False, index=True)
    settings = Column(JSON, nullable=False, default={})
    owner_user_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    billing_email = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    trial_ends_at = Column(DateTime(timezone=True), nullable=True)
    suspended_at = Column(DateTime(timezone=True), nullable=True)
    suspended_reason = Column(String, nullable=True)
    tenant_metadata = Column("metadata", JSON, nullable=False, default={})
    
    def __repr__(self):
        return f"<TenantModel(id={self.id}, name={self.name}, slug={self.slug}, status={self.status})>"
