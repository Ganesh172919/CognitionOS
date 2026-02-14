"""
Task Decomposition Infrastructure - SQLAlchemy Models

ORM models for persisting TaskDecomposition and TaskNode domain entities to PostgreSQL.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    Column, String, DateTime, Integer, Float, Boolean, ForeignKey, Text
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB, ARRAY
from sqlalchemy.orm import relationship

from infrastructure.persistence.base import Base


class TaskDecompositionModel(Base):
    """
    SQLAlchemy model for TaskDecomposition entity.
    
    Maps to 'task_decompositions' table.
    """
    __tablename__ = "task_decompositions"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    workflow_execution_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("workflow_executions.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Root task information
    root_task_name = Column(String(500), nullable=False)
    root_task_description = Column(Text, nullable=True)
    root_node_id = Column(PGUUID(as_uuid=True), nullable=True)
    
    # Strategy and statistics
    strategy = Column(String(50), nullable=False)  # breadth_first, depth_first, hybrid, adaptive
    total_nodes = Column(Integer, default=0)
    max_depth_reached = Column(Integer, default=0)
    leaf_node_count = Column(Integer, default=0)
    
    # All node IDs for quick access
    all_node_ids = Column(ARRAY(PGUUID(as_uuid=True)), default=list)
    
    # Status flags
    is_complete = Column(Boolean, default=False)
    has_cycles = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    metadata = Column(JSONB, nullable=True)
    
    # Relationship to task nodes
    task_nodes = relationship(
        "TaskNodeModel",
        back_populates="decomposition",
        cascade="all, delete-orphan"
    )
    
    __table_args__ = (
        {'extend_existing': True}
    )


class TaskNodeModel(Base):
    """
    SQLAlchemy model for TaskNode entity.
    
    Maps to 'task_nodes' table.
    """
    __tablename__ = "task_nodes"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    decomposition_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("task_decompositions.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Task information
    name = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    
    # Hierarchy
    parent_id = Column(PGUUID(as_uuid=True), nullable=True)
    depth_level = Column(Integer, nullable=False)
    child_node_ids = Column(ARRAY(PGUUID(as_uuid=True)), default=list)
    
    # Task properties
    estimated_complexity = Column(Float, nullable=False)
    is_leaf = Column(Boolean, default=True)
    actual_subtask_count = Column(Integer, default=0)
    
    # Status
    status = Column(String(50), nullable=False, default="pending")
    
    # Dependencies (stored as JSONB array)
    dependencies = Column(JSONB, default=list)
    
    # Tags and metadata
    tags = Column(ARRAY(String), default=list)
    metadata = Column(JSONB, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=True)
    decomposed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationship to decomposition
    decomposition = relationship("TaskDecompositionModel", back_populates="task_nodes")
    
    __table_args__ = (
        {'extend_existing': True}
    )
