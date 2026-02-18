"""
Execution Persistence Repository Implementation

PostgreSQL repositories for replay sessions and execution snapshots.
"""

from typing import Optional, List
from uuid import UUID
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Boolean, JSON, ForeignKey, Integer, Text, select
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.persistence.base import Base
from core.domain.execution import (
    ReplaySession,
    ExecutionSnapshot,
    ReplayMode,
    SnapshotType,
)


# ==================== SQLAlchemy Models ====================

class ReplaySessionModel(Base):
    """SQLAlchemy model for ReplaySession entity"""
    
    __tablename__ = "replay_sessions"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True)
    original_execution_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    replay_execution_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    replay_mode = Column(String(50), nullable=False)
    start_from_step = Column(String(255), nullable=True)
    use_cached_outputs = Column(Boolean, nullable=False, default=True)
    triggered_by = Column(PGUUID(as_uuid=True), nullable=True)
    status = Column(String(50), nullable=False, default="pending")
    match_percentage = Column(JSON, nullable=True)
    divergence_details = Column(JSON, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ReplaySessionModel(id={self.id}, mode={self.replay_mode}, status={self.status})>"


class ExecutionSnapshotModel(Base):
    """SQLAlchemy model for ExecutionSnapshot entity"""
    
    __tablename__ = "execution_snapshots"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True)
    execution_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    snapshot_type = Column(String(50), nullable=False)
    workflow_state = Column(JSON, nullable=False)
    step_states = Column(JSON, nullable=False)
    variables = Column(JSON, nullable=False, default=dict)
    completed_steps = Column(JSON, nullable=False, default=list)  # List of step IDs
    pending_steps = Column(JSON, nullable=False, default=list)  # List of step IDs
    failed_steps = Column(JSON, nullable=False, default=list)  # List of step IDs
    created_by = Column(String(255), nullable=False, default="system")
    snapshot_size_bytes = Column(Integer, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ExecutionSnapshotModel(id={self.id}, execution_id={self.execution_id}, type={self.snapshot_type})>"


# ==================== Repository Implementations ====================

class PostgreSQLReplaySessionRepository:
    """PostgreSQL repository for ReplaySession"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, replay_session: ReplaySession) -> None:
        """Save or update replay session"""
        stmt = select(ReplaySessionModel).where(ReplaySessionModel.id == replay_session.id)
        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update existing
            existing.status = replay_session.status
            existing.match_percentage = replay_session.match_percentage
            existing.divergence_details = replay_session.divergence_details
            existing.started_at = replay_session.started_at
            existing.completed_at = replay_session.completed_at
            existing.metadata = replay_session.metadata
        else:
            # Create new
            model = ReplaySessionModel(
                id=replay_session.id,
                original_execution_id=replay_session.original_execution_id,
                replay_execution_id=replay_session.replay_execution_id,
                replay_mode=replay_session.replay_mode.value,
                start_from_step=replay_session.start_from_step,
                use_cached_outputs=replay_session.use_cached_outputs,
                triggered_by=replay_session.triggered_by,
                status=replay_session.status,
                match_percentage=replay_session.match_percentage,
                divergence_details=replay_session.divergence_details,
                started_at=replay_session.started_at,
                completed_at=replay_session.completed_at,
                metadata=replay_session.metadata,
                created_at=replay_session.created_at,
            )
            self.session.add(model)
        
        await self.session.flush()
    
    async def get_by_id(self, session_id: UUID) -> Optional[ReplaySession]:
        """Get replay session by ID"""
        stmt = select(ReplaySessionModel).where(ReplaySessionModel.id == session_id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if not model:
            return None
        
        return self._to_entity(model)
    
    async def get_by_original_execution(self, execution_id: UUID) -> List[ReplaySession]:
        """Get all replay sessions for an execution"""
        stmt = select(ReplaySessionModel).where(
            ReplaySessionModel.original_execution_id == execution_id
        ).order_by(ReplaySessionModel.created_at.desc())
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]
    
    def _to_entity(self, model: ReplaySessionModel) -> ReplaySession:
        """Convert model to entity"""
        return ReplaySession(
            id=model.id,
            original_execution_id=model.original_execution_id,
            replay_execution_id=model.replay_execution_id,
            replay_mode=ReplayMode(model.replay_mode),
            start_from_step=model.start_from_step,
            use_cached_outputs=model.use_cached_outputs,
            triggered_by=model.triggered_by,
            status=model.status,
            match_percentage=model.match_percentage,
            divergence_details=model.divergence_details or {},
            started_at=model.started_at,
            completed_at=model.completed_at,
            metadata=model.metadata or {},
            created_at=model.created_at,
        )


class PostgreSQLExecutionSnapshotRepository:
    """PostgreSQL repository for ExecutionSnapshot"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, snapshot: ExecutionSnapshot) -> None:
        """Save execution snapshot"""
        model = ExecutionSnapshotModel(
            id=snapshot.id,
            execution_id=snapshot.execution_id,
            snapshot_type=snapshot.snapshot_type.value,
            workflow_state=snapshot.workflow_state,
            step_states=snapshot.step_states,
            variables=snapshot.variables,
            completed_steps=snapshot.completed_steps,
            pending_steps=snapshot.pending_steps,
            failed_steps=snapshot.failed_steps,
            created_by=snapshot.created_by,
            snapshot_size_bytes=snapshot.snapshot_size_bytes,
            metadata=snapshot.metadata,
            created_at=snapshot.created_at,
        )
        self.session.add(model)
        await self.session.flush()
    
    async def get_by_id(self, snapshot_id: UUID) -> Optional[ExecutionSnapshot]:
        """Get snapshot by ID"""
        stmt = select(ExecutionSnapshotModel).where(ExecutionSnapshotModel.id == snapshot_id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if not model:
            return None
        
        return self._to_entity(model)
    
    async def get_latest_for_execution(self, execution_id: UUID) -> Optional[ExecutionSnapshot]:
        """Get latest snapshot for an execution"""
        stmt = select(ExecutionSnapshotModel).where(
            ExecutionSnapshotModel.execution_id == execution_id
        ).order_by(ExecutionSnapshotModel.created_at.desc()).limit(1)
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if not model:
            return None
        
        return self._to_entity(model)
    
    async def get_all_for_execution(self, execution_id: UUID) -> List[ExecutionSnapshot]:
        """Get all snapshots for an execution"""
        stmt = select(ExecutionSnapshotModel).where(
            ExecutionSnapshotModel.execution_id == execution_id
        ).order_by(ExecutionSnapshotModel.created_at.asc())
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]
    
    def _to_entity(self, model: ExecutionSnapshotModel) -> ExecutionSnapshot:
        """Convert model to entity"""
        return ExecutionSnapshot(
            id=model.id,
            execution_id=model.execution_id,
            snapshot_type=SnapshotType(model.snapshot_type),
            workflow_state=model.workflow_state,
            step_states=model.step_states,
            variables=model.variables or {},
            completed_steps=model.completed_steps or [],
            pending_steps=model.pending_steps or [],
            failed_steps=model.failed_steps or [],
            created_by=model.created_by,
            snapshot_size_bytes=model.snapshot_size_bytes,
            metadata=model.metadata or {},
            created_at=model.created_at,
        )
