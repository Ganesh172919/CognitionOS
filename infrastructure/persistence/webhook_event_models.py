"""
Webhook Event Database Models

SQLAlchemy models for webhook event persistence and idempotent processing.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any

from sqlalchemy import (
    Column,
    String,
    DateTime,
    Integer,
    Text,
    Enum as SQLEnum,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from infrastructure.persistence.base import Base


class WebhookEventStatus(str, Enum):
    """Status of webhook event processing."""
    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"
    RETRYING = "retrying"


class WebhookEventModel(Base):
    """
    Model for storing webhook events.
    
    Enables idempotent webhook processing and retry management.
    """
    
    __tablename__ = "webhook_events"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Event identification
    event_id = Column(String(255), unique=True, nullable=False, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    
    # Processing status
    status = Column(
        SQLEnum(WebhookEventStatus),
        nullable=False,
        default=WebhookEventStatus.PENDING,
        index=True,
    )
    
    # Event data
    result = Column(JSONB, nullable=True)  # Processing result for successful events
    error_message = Column(Text, nullable=True)  # Error message for failed events
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    processed_at = Column(DateTime, nullable=True)
    failed_at = Column(DateTime, nullable=True)
    last_retry_at = Column(DateTime, nullable=True)
    
    # Retry management
    retry_count = Column(Integer, nullable=False, default=0)
    
    # Indexes for efficient querying
    __table_args__ = (
        Index("idx_webhook_events_status_retry", "status", "retry_count"),
        Index("idx_webhook_events_type_status", "event_type", "status"),
        Index("idx_webhook_events_created_at", "created_at"),
    )
    
    def __repr__(self) -> str:
        return (
            f"<WebhookEvent("
            f"id={self.id}, "
            f"event_id={self.event_id}, "
            f"type={self.event_type}, "
            f"status={self.status}"
            f")>"
        )
