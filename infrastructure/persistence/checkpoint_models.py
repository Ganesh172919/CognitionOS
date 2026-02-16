"""
Checkpoint Infrastructure - SQLAlchemy Models

ORM models for persisting Checkpoint domain entities to PostgreSQL.
"""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, String, DateTime, Integer, BigInteger, Boolean, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB

from infrastructure.persistence.base import Base


class CheckpointModel(Base):
    """
    SQLAlchemy model for Checkpoint entity.

    Maps to 'checkpoints' table.
    """
    __tablename__ = "checkpoints"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    workflow_execution_id = Column(PGUUID(as_uuid=True), ForeignKey("workflow_executions.id", ondelete="CASCADE"), nullable=False)
    checkpoint_number = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    # Execution state snapshot
    execution_state = Column(JSONB, nullable=False)
    dag_progress = Column(JSONB, nullable=False)
    memory_snapshot_ref = Column(String(500), nullable=True)
    active_tasks = Column(JSONB, nullable=True)
    budget_state = Column(JSONB, nullable=True)
    
    # Metadata
    checkpoint_size_bytes = Column(BigInteger, nullable=True)
    compression_enabled = Column(Boolean, default=True)
    checkpoint_metadata = Column(JSONB, nullable=True)
    
    # Constraints
    __table_args__ = (
        {'extend_existing': True}
    )
