"""
Workflow Infrastructure - SQLAlchemy Models

ORM models for persisting Workflow domain entities to PostgreSQL.
"""

from datetime import datetime
from typing import List
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, String, DateTime, Enum, Integer, JSON, ForeignKey, Text, Boolean
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID, ARRAY
from sqlalchemy.orm import relationship

from infrastructure.persistence.base import Base
import enum


class WorkflowStatusEnum(str, enum.Enum):
    """Workflow status for database"""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ExecutionStatusEnum(str, enum.Enum):
    """Execution status for database"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class WorkflowModel(Base):
    """
    SQLAlchemy model for Workflow entity.

    Maps to 'workflows' table.
    """
    __tablename__ = "workflows"

    id = Column(String(255), primary_key=True)
    version_major = Column(Integer, nullable=False)
    version_minor = Column(Integer, nullable=False)
    version_patch = Column(Integer, nullable=False)
    name = Column(String(512), nullable=False)
    description = Column(Text, nullable=False)
    status = Column(Enum(WorkflowStatusEnum), nullable=False, default=WorkflowStatusEnum.DRAFT)
    schedule = Column(String(255), nullable=True)  # Cron expression
    tags = Column(ARRAY(String), default=list)
    steps = Column(JSON, nullable=False)  # JSON array of steps
    created_by = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    # Relationships
    executions = relationship("WorkflowExecutionModel", back_populates="workflow", cascade="all, delete-orphan")

    # Composite unique constraint on (id, version)
    __table_args__ = (
        {'extend_existing': True}
    )


class WorkflowExecutionModel(Base):
    """
    SQLAlchemy model for WorkflowExecution entity.

    Maps to 'workflow_executions' table.
    """
    __tablename__ = "workflow_executions"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    workflow_id = Column(String(255), ForeignKey("workflows.id"), nullable=False)
    workflow_version_major = Column(Integer, nullable=False)
    workflow_version_minor = Column(Integer, nullable=False)
    workflow_version_patch = Column(Integer, nullable=False)
    status = Column(Enum(ExecutionStatusEnum), nullable=False, default=ExecutionStatusEnum.PENDING)
    inputs = Column(JSON, default=dict)
    outputs = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    user_id = Column(PGUUID(as_uuid=True), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    # Relationships
    workflow = relationship("WorkflowModel", back_populates="executions")
    step_executions = relationship("StepExecutionModel", back_populates="execution", cascade="all, delete-orphan")


class StepExecutionModel(Base):
    """
    SQLAlchemy model for StepExecution entity.

    Maps to 'step_executions' table.
    """
    __tablename__ = "step_executions"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    execution_id = Column(PGUUID(as_uuid=True), ForeignKey("workflow_executions.id"), nullable=False)
    step_id = Column(String(255), nullable=False)
    step_type = Column(String(100), nullable=False)
    status = Column(Enum(ExecutionStatusEnum), nullable=False, default=ExecutionStatusEnum.PENDING)
    output = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    agent_id = Column(PGUUID(as_uuid=True), nullable=True)
    retry_count = Column(Integer, default=0)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    execution = relationship("WorkflowExecutionModel", back_populates="step_executions")
