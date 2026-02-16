"""
Memory Hierarchy Infrastructure - SQLAlchemy Models

ORM models for persisting memory hierarchy domain entities to PostgreSQL.
Maps to tables from migration 003_phase3_extended_operation.sql
"""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, String, DateTime, Integer, Float, Boolean, ForeignKey, Text
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB, ARRAY
from pgvector.sqlalchemy import Vector

from infrastructure.persistence.base import Base


class WorkingMemoryModel(Base):
    """
    SQLAlchemy model for WorkingMemory entity.
    
    Maps to 'working_memory' table (L1 tier).
    """
    __tablename__ = "working_memory"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id = Column(PGUUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"), nullable=False)
    workflow_execution_id = Column(PGUUID(as_uuid=True), ForeignKey("workflow_executions.id", ondelete="CASCADE"), nullable=True)
    
    # Memory content
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    
    # Importance and lifecycle
    importance_score = Column(Float, nullable=False, default=0.5)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    last_accessed_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    access_count = Column(Integer, nullable=False, default=0)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    memory_type = Column(String(50), nullable=True)
    tags = Column(ARRAY(Text), nullable=True)
    checkpoint_metadata = Column(JSONB, nullable=True)
    
    # Constraints
    __table_args__ = (
        {'extend_existing': True}
    )


class EpisodicMemoryModel(Base):
    """
    SQLAlchemy model for EpisodicMemory entity.
    
    Maps to 'episodic_memory' table (L2 tier).
    """
    __tablename__ = "episodic_memory"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id = Column(PGUUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"), nullable=False)
    
    # Clustering and summarization
    cluster_id = Column(PGUUID(as_uuid=True), nullable=True)
    summary = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    
    # Compression metadata
    compression_ratio = Column(Float, nullable=True)
    source_memory_ids = Column(ARRAY(PGUUID(as_uuid=True)), nullable=True)
    source_memory_count = Column(Integer, nullable=True)
    
    # Importance and lifecycle
    importance_score = Column(Float, nullable=False, default=0.5)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    last_accessed_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    access_count = Column(Integer, nullable=False, default=0)
    
    # Metadata
    temporal_period = Column(JSONB, nullable=True)
    tags = Column(ARRAY(Text), nullable=True)
    checkpoint_metadata = Column(JSONB, nullable=True)
    
    # Constraints
    __table_args__ = (
        {'extend_existing': True}
    )


class LongTermMemoryModel(Base):
    """
    SQLAlchemy model for LongTermMemory entity.
    
    Maps to 'longterm_memory' table (L3 tier).
    """
    __tablename__ = "longterm_memory"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id = Column(PGUUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"), nullable=False)
    
    # Knowledge content
    knowledge_type = Column(String(50), nullable=True)
    title = Column(String(500), nullable=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    
    # Importance and lifecycle
    importance_score = Column(Float, nullable=False, default=0.5)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    last_accessed_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    access_count = Column(Integer, nullable=False, default=0)
    archived_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    source_type = Column(String(50), nullable=True)
    tags = Column(ARRAY(Text), nullable=True)
    checkpoint_metadata = Column(JSONB, nullable=True)
    
    # Constraints
    __table_args__ = (
        {'extend_existing': True}
    )


class MemoryLifecycleEventModel(Base):
    """
    SQLAlchemy model for memory lifecycle events.
    
    Maps to 'memory_lifecycle_events' table.
    """
    __tablename__ = "memory_lifecycle_events"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id = Column(PGUUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"), nullable=False)
    
    # Event details
    event_type = Column(String(50), nullable=False)
    memory_tier = Column(String(10), nullable=False)
    memory_id = Column(PGUUID(as_uuid=True), nullable=False)
    
    # Context
    reason = Column(Text, nullable=True)
    importance_score_before = Column(Float, nullable=True)
    importance_score_after = Column(Float, nullable=True)
    
    # Timing
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    # Metadata
    checkpoint_metadata = Column(JSONB, nullable=True)
    
    # Constraints
    __table_args__ = (
        {'extend_existing': True}
    )
