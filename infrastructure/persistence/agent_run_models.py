"""
Agent Run SQLAlchemy Models

Database models for persisted single-agent runtime executions.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    LargeBinary,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from infrastructure.persistence.base import Base


class AgentRunModel(Base):
    """SQLAlchemy model for AgentRun."""

    __tablename__ = "agent_runs"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    user_id = Column(PGUUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)

    requirement = Column(Text, nullable=False)
    status = Column(String(50), nullable=False, index=True)

    budgets = Column(JSON, nullable=False, default=dict)
    usage = Column(JSON, nullable=False, default=dict)

    error = Column(Text, nullable=True)
    run_metadata = Column("metadata", JSON, nullable=False, default=dict)

    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AgentRunModel(id={self.id}, status={self.status}, tenant_id={self.tenant_id})>"


class AgentRunStepModel(Base):
    """SQLAlchemy model for AgentRunStep."""

    __tablename__ = "agent_run_steps"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    step_index = Column(Integer, nullable=False)
    step_type = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False, index=True)

    input = Column(JSON, nullable=False, default=dict)
    output = Column(JSON, nullable=True)
    tool_calls = Column(JSON, nullable=False, default=list)

    tokens_used = Column(Integer, nullable=False, default=0)
    cost_usd = Column(Numeric(12, 6), nullable=False, default=0)
    duration_ms = Column(Integer, nullable=True)
    error = Column(Text, nullable=True)

    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("run_id", "step_index", name="unique_agent_run_step"),)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<AgentRunStepModel(id={self.id}, run_id={self.run_id}, index={self.step_index}, "
            f"status={self.status})>"
        )


class AgentRunArtifactModel(Base):
    """SQLAlchemy model for AgentRunArtifact."""

    __tablename__ = "agent_run_artifacts"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    step_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("agent_run_steps.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    kind = Column(String(50), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=False, default="text/plain")

    content_text = Column(Text, nullable=True)
    content_bytes = Column(LargeBinary, nullable=True)
    sha256 = Column(String(64), nullable=True)
    size_bytes = Column(Integer, nullable=True)
    storage_url = Column(Text, nullable=True)

    artifact_metadata = Column("metadata", JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AgentRunArtifactModel(id={self.id}, run_id={self.run_id}, kind={self.kind})>"


class AgentRunEvaluationModel(Base):
    """SQLAlchemy model for AgentRunEvaluation."""

    __tablename__ = "agent_run_evaluations"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    success = Column(Boolean, nullable=False, default=False)
    confidence = Column(Numeric(5, 4), nullable=False, default=0)
    quality_scores = Column(JSON, nullable=False, default=dict)
    policy_violations = Column(JSON, nullable=False, default=list)
    retry_plan = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AgentRunEvaluationModel(id={self.id}, run_id={self.run_id}, success={self.success})>"


class AgentRunMemoryLinkModel(Base):
    """SQLAlchemy model for AgentRunMemoryLink."""

    __tablename__ = "agent_run_memory_links"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    memory_id = Column(PGUUID(as_uuid=True), nullable=False)
    memory_tier = Column(String(50), nullable=False)
    relation = Column(String(50), nullable=False, default="used")
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AgentRunMemoryLinkModel(id={self.id}, run_id={self.run_id}, memory_id={self.memory_id})>"

