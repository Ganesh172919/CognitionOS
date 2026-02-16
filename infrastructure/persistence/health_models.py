"""
Health Monitoring Infrastructure - SQLAlchemy Models

ORM models for persisting Health Monitoring domain entities to PostgreSQL.
"""

from datetime import datetime
from uuid import UUID, uuid4
import enum

from sqlalchemy import (
    Column, String, DateTime, Enum, Integer, Float, Text, Boolean, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB

from infrastructure.persistence.base import Base


class HealthStatusEnum(str, enum.Enum):
    """Health status for database"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


class IncidentSeverityEnum(str, enum.Enum):
    """Incident severity for database"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatusEnum(str, enum.Enum):
    """Incident status for database"""
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    IGNORED = "ignored"


class AgentHealthStatusModel(Base):
    """
    SQLAlchemy model for AgentHealthStatus entity.

    Maps to 'agent_health_status' table.
    """
    __tablename__ = "agent_health_status"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id = Column(PGUUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"), nullable=False)
    workflow_execution_id = Column(PGUUID(as_uuid=True), ForeignKey("workflow_executions.id", ondelete="CASCADE"), nullable=False)
    
    # Health status
    status = Column(Enum(HealthStatusEnum), nullable=False, default=HealthStatusEnum.HEALTHY)
    last_heartbeat = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    # Resource metrics
    memory_usage_mb = Column(Float, nullable=True)
    cpu_usage_percent = Column(Float, nullable=True)
    working_memory_count = Column(Integer, nullable=True)
    episodic_memory_count = Column(Integer, nullable=True)
    
    # Cost metrics
    cost_consumed = Column(Float, nullable=True)
    budget_remaining = Column(Float, nullable=True)
    
    # Task metrics
    active_tasks_count = Column(Integer, default=0)
    completed_tasks_count = Column(Integer, default=0)
    failed_tasks_count = Column(Integer, default=0)
    
    # Overall health score (0-1)
    health_score = Column(Float, nullable=True)
    
    # Metadata
    error_message = Column(Text, nullable=True)
    recovery_attempts = Column(Integer, default=0)
    checkpoint_metadata = Column(JSONB, nullable=True)
    
    # Timing
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    __table_args__ = (
        {'extend_existing': True}
    )


class AgentHealthIncidentModel(Base):
    """
    SQLAlchemy model for AgentHealthIncident entity.

    Maps to 'agent_health_incidents' table.
    """
    __tablename__ = "agent_health_incidents"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id = Column(PGUUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"), nullable=False)
    workflow_execution_id = Column(PGUUID(as_uuid=True), ForeignKey("workflow_executions.id", ondelete="CASCADE"), nullable=False)
    
    # Incident details
    incident_type = Column(String(50), nullable=False)
    severity = Column(Enum(IncidentSeverityEnum), nullable=False)
    description = Column(Text, nullable=False)
    
    # Resolution
    status = Column(Enum(IncidentStatusEnum), default=IncidentStatusEnum.OPEN)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    
    # Recovery
    recovery_action = Column(String(100), nullable=True)
    recovery_successful = Column(Boolean, nullable=True)
    
    # Timing
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    # Metadata
    metrics_snapshot = Column(JSONB, nullable=True)
    checkpoint_metadata = Column(JSONB, nullable=True)
    
    __table_args__ = (
        {'extend_existing': True}
    )
