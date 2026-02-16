"""
Cost Governance Infrastructure - SQLAlchemy Models

ORM models for persisting Cost Governance domain entities to PostgreSQL.
"""

from datetime import datetime
from uuid import UUID, uuid4
import enum

from sqlalchemy import (
    Column, String, DateTime, Enum, Integer, BigInteger, Numeric, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB

from infrastructure.persistence.base import Base


class BudgetStatusEnum(str, enum.Enum):
    """Budget status for database"""
    ACTIVE = "active"
    WARNING = "warning"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"
    COMPLETED = "completed"


class OperationTypeEnum(str, enum.Enum):
    """Operation type for database"""
    LLM_CALL = "llm_call"
    STORAGE = "storage"
    COMPUTE = "compute"
    MEMORY_OPERATION = "memory_operation"


class WorkflowBudgetModel(Base):
    """
    SQLAlchemy model for WorkflowBudget entity.

    Maps to 'workflow_budget' table.
    """
    __tablename__ = "workflow_budget"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    workflow_execution_id = Column(PGUUID(as_uuid=True), ForeignKey("workflow_executions.id", ondelete="CASCADE"), nullable=False)
    
    # Budget allocation
    allocated_budget = Column(Numeric(10, 2), nullable=False)
    consumed_budget = Column(Numeric(10, 2), default=0)
    
    # Thresholds
    warning_threshold = Column(Numeric(10, 2), nullable=True)
    critical_threshold = Column(Numeric(10, 2), nullable=True)
    
    # Status
    status = Column(Enum(BudgetStatusEnum), default=BudgetStatusEnum.ACTIVE)
    
    # Actions taken
    warnings_sent = Column(Integer, default=0)
    halt_triggered_at = Column(DateTime(timezone=True), nullable=True)
    suspended_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timing
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    # Metadata
    checkpoint_metadata = Column(JSONB, nullable=True)
    
    __table_args__ = (
        {'extend_existing': True}
    )


class CostTrackingModel(Base):
    """
    SQLAlchemy model for CostEntry entity.

    Maps to 'cost_tracking' table.
    """
    __tablename__ = "cost_tracking"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    workflow_execution_id = Column(PGUUID(as_uuid=True), ForeignKey("workflow_executions.id", ondelete="CASCADE"), nullable=False)
    agent_id = Column(PGUUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"), nullable=True)
    
    # Operation details
    operation_type = Column(Enum(OperationTypeEnum), nullable=False)
    provider = Column(String(50), nullable=True)
    model = Column(String(100), nullable=True)
    
    # Resource usage
    tokens_used = Column(Integer, nullable=True)
    execution_time_ms = Column(Integer, nullable=True)
    memory_bytes = Column(BigInteger, nullable=True)
    
    # Cost calculation
    cost = Column(Numeric(10, 6), nullable=False)
    currency = Column(String(3), default='USD')
    
    # Context
    task_id = Column(PGUUID(as_uuid=True), nullable=True)
    step_name = Column(String(200), nullable=True)
    
    # Timing
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    # Metadata
    request_payload = Column(JSONB, nullable=True)
    response_metadata = Column(JSONB, nullable=True)
    checkpoint_metadata = Column(JSONB, nullable=True)
    
    __table_args__ = (
        {'extend_existing': True}
    )
