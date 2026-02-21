"""
Billing SQLAlchemy Models

Database models for billing entities.
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, DateTime, Enum as SQLEnum, Integer, Boolean, JSON, DECIMAL, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID
import uuid

from infrastructure.persistence.base import Base
from core.domain.billing.entities import SubscriptionStatus, InvoiceStatus


class SubscriptionModel(Base):
    """SQLAlchemy model for Subscription entity"""
    
    __tablename__ = "subscriptions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    tier = Column(String(50), nullable=False)
    status = Column(
        SQLEnum(SubscriptionStatus, name="subscription_status_enum", create_type=False),
        nullable=False,
        index=True
    )
    stripe_subscription_id = Column(String(255), nullable=True, unique=True, index=True)
    stripe_customer_id = Column(String(255), nullable=True)
    current_period_start = Column(DateTime(timezone=True), nullable=False)
    current_period_end = Column(DateTime(timezone=True), nullable=False, index=True)
    trial_start = Column(DateTime(timezone=True), nullable=True)
    trial_end = Column(DateTime(timezone=True), nullable=True, index=True)
    canceled_at = Column(DateTime(timezone=True), nullable=True)
    cancel_at_period_end = Column(Boolean, nullable=False, default=False)
    amount_cents = Column(Integer, nullable=False)
    currency = Column(String(3), nullable=False, default="usd")
    billing_cycle = Column(String(20), nullable=False)
    payment_method = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    subscription_metadata = Column("metadata", JSON, nullable=False, default={})

    def __repr__(self):
        return f"<SubscriptionModel(id={self.id}, tenant_id={self.tenant_id}, tier={self.tier}, status={self.status})>"


class InvoiceModel(Base):
    """SQLAlchemy model for Invoice entity"""
    
    __tablename__ = "invoices"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    subscription_id = Column(UUID(as_uuid=True), ForeignKey("subscriptions.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(
        SQLEnum(InvoiceStatus, name="invoice_status_enum", create_type=False),
        nullable=False,
        index=True
    )
    invoice_number = Column(String(100), nullable=False, unique=True)
    amount_cents = Column(Integer, nullable=False)
    amount_paid_cents = Column(Integer, nullable=False, default=0)
    amount_due_cents = Column(Integer, nullable=False)
    currency = Column(String(3), nullable=False, default="usd")
    stripe_invoice_id = Column(String(255), nullable=True, unique=True, index=True)
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    due_date = Column(DateTime(timezone=True), nullable=True, index=True)
    paid_at = Column(DateTime(timezone=True), nullable=True, index=True)
    line_items = Column(JSON, nullable=False, default=[])
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    invoice_metadata = Column("metadata", JSON, nullable=False, default={})
    
    def __repr__(self):
        return f"<InvoiceModel(id={self.id}, invoice_number={self.invoice_number}, status={self.status})>"


class UsageRecordModel(Base):
    """SQLAlchemy model for UsageRecord entity"""
    
    __tablename__ = "usage_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    resource_type = Column(String(100), nullable=False, index=True)
    quantity = Column(DECIMAL(20, 6), nullable=False)
    unit = Column(String(50), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True)
    usage_metadata = Column("metadata", JSON, nullable=False, default={})
    
    def __repr__(self):
        return f"<UsageRecordModel(id={self.id}, tenant_id={self.tenant_id}, resource_type={self.resource_type}, quantity={self.quantity})>"



class RateLimitTrackingModel(Base):
    """SQLAlchemy model for Rate Limit Tracking entity"""
    
    __tablename__ = "rate_limit_tracking"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    resource_key = Column(String(255), nullable=False, index=True)
    window_start = Column(DateTime(timezone=True), nullable=False, index=True)
    window_duration_seconds = Column(Integer, nullable=False)
    request_count = Column(Integer, nullable=False, default=0)
    blocked_count = Column(Integer, nullable=False, default=0)
    last_request_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<RateLimitTrackingModel(id={self.id}, tenant_id={self.tenant_id}, resource_key={self.resource_key})>"
