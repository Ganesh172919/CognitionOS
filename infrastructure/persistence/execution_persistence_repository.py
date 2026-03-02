"""
Execution Persistence Repository Implementation

PostgreSQL repositories for deterministic execution persistence:
- Step execution attempts (idempotency + replay comparison)
- Execution snapshots (resume)
- Replay sessions (replay tracking + comparison results)
- Execution locks (distributed locking)
- Execution errors (unified error model persistence)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    and_,
    delete,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID as PGUUID, insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.persistence.base import Base
from core.domain.execution import (
    AttemptStatus,
    ErrorCategory,
    ErrorSeverity,
    ExecutionError,
    ExecutionLock,
    ReplaySession,
    ExecutionSnapshot,
    ReplayMode,
    SnapshotType,
    StepExecutionAttempt,
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
    match_percentage = Column(Numeric(5, 2), nullable=True)
    divergence_details = Column(JSON, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    replay_metadata = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    
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
    completed_steps = Column(ARRAY(Text), nullable=False, default=list)  # List of step IDs
    pending_steps = Column(ARRAY(Text), nullable=False, default=list)  # List of step IDs
    failed_steps = Column(ARRAY(Text), nullable=False, default=list)  # List of step IDs
    created_by = Column(String(255), nullable=False, default="system")
    snapshot_size_bytes = Column(Integer, nullable=True)
    snapshot_metadata = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    
    def __repr__(self):
        return f"<ExecutionSnapshotModel(id={self.id}, execution_id={self.execution_id}, type={self.snapshot_type})>"


class StepExecutionAttemptModel(Base):
    """SQLAlchemy model for StepExecutionAttempt entity"""

    __tablename__ = "step_execution_attempts"

    id = Column(PGUUID(as_uuid=True), primary_key=True)
    step_execution_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("step_executions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attempt_number = Column(Integer, nullable=False)
    idempotency_key = Column(String(255), nullable=False, unique=True, index=True)

    inputs = Column(JSON, nullable=False)
    agent_id = Column(PGUUID(as_uuid=True), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    outputs = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    status = Column(String(50), nullable=False, index=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Integer, nullable=True)

    is_deterministic = Column(Boolean, nullable=False, default=True)
    nondeterminism_flags = Column(ARRAY(Text), nullable=True)

    request_payload = Column(JSON, nullable=True)
    response_payload = Column(JSON, nullable=True)
    response_hash = Column(String(64), nullable=True)

    attempt_metadata = Column("metadata", JSON, nullable=True, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("step_execution_id", "attempt_number", name="unique_step_attempt"),
    )

    def __repr__(self):
        return (
            f"<StepExecutionAttemptModel(id={self.id}, step_execution_id={self.step_execution_id}, "
            f"attempt={self.attempt_number}, status={self.status})>"
        )


class ExecutionErrorModel(Base):
    """SQLAlchemy model for ExecutionError entity"""

    __tablename__ = "execution_errors"

    id = Column(PGUUID(as_uuid=True), primary_key=True)

    error_code = Column(String(100), nullable=False, index=True)
    error_category = Column(String(50), nullable=False, index=True)
    severity = Column(String(50), nullable=False, index=True)
    message = Column(Text, nullable=False)

    correlation_id = Column(PGUUID(as_uuid=True), nullable=False, index=True)
    is_retryable = Column(Boolean, nullable=False, default=False, index=True)

    details = Column(JSON, nullable=False, default=dict)
    stack_trace = Column(Text, nullable=True)
    service_name = Column(String(255), nullable=True)

    execution_id = Column(PGUUID(as_uuid=True), nullable=True, index=True)
    step_execution_id = Column(PGUUID(as_uuid=True), nullable=True, index=True)
    user_id = Column(PGUUID(as_uuid=True), nullable=True)

    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    next_retry_at = Column(DateTime(timezone=True), nullable=True)

    resolved = Column(Boolean, nullable=False, default=False, index=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolution_notes = Column(Text, nullable=True)

    error_metadata = Column("metadata", JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<ExecutionErrorModel(id={self.id}, code={self.error_code}, category={self.error_category})>"


class ExecutionLockModel(Base):
    """SQLAlchemy model for ExecutionLock entity"""

    __tablename__ = "execution_locks"

    id = Column(PGUUID(as_uuid=True), primary_key=True)
    lock_key = Column(String(255), nullable=False, unique=True, index=True)
    lock_holder = Column(String(255), nullable=False)
    acquired_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    lock_metadata = Column("metadata", JSON, nullable=False, default=dict)

    def __repr__(self):
        return f"<ExecutionLockModel(key={self.lock_key}, holder={self.lock_holder}, expires_at={self.expires_at})>"


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
            existing.replay_metadata = replay_session.metadata
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
                replay_metadata=replay_session.metadata,
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
            match_percentage=float(model.match_percentage) if model.match_percentage is not None else None,
            divergence_details=model.divergence_details or {},
            started_at=model.started_at,
            completed_at=model.completed_at,
            metadata=model.replay_metadata or {},
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
            snapshot_metadata=snapshot.metadata,
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
            metadata=model.snapshot_metadata or {},
            created_at=model.created_at,
        )


class PostgreSQLStepExecutionAttemptRepository:
    """PostgreSQL repository for StepExecutionAttempt (idempotency + replay)"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, attempt_id: UUID) -> Optional[StepExecutionAttempt]:
        stmt = select(StepExecutionAttemptModel).where(StepExecutionAttemptModel.id == attempt_id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def get_by_idempotency_key(self, idempotency_key: str) -> Optional[StepExecutionAttempt]:
        stmt = select(StepExecutionAttemptModel).where(StepExecutionAttemptModel.idempotency_key == idempotency_key)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def create(self, attempt: StepExecutionAttempt) -> StepExecutionAttempt:
        model = self._to_model(attempt)
        self.session.add(model)
        try:
            await self.session.flush()
        except IntegrityError:
            await self.session.rollback()
            existing = await self.get_by_idempotency_key(attempt.idempotency_key)
            if existing:
                return existing
            raise
        await self.session.refresh(model)
        return self._to_entity(model)

    async def save(self, attempt: StepExecutionAttempt) -> StepExecutionAttempt:
        stmt = select(StepExecutionAttemptModel).where(StepExecutionAttemptModel.id == attempt.id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()

        if model is None:
            return await self.create(attempt)

        model.step_execution_id = attempt.step_execution_id
        model.attempt_number = attempt.attempt_number
        model.idempotency_key = attempt.idempotency_key
        model.inputs = attempt.inputs
        model.agent_id = attempt.agent_id
        model.started_at = attempt.started_at
        model.outputs = attempt.outputs
        model.error = attempt.error
        model.status = attempt.status.value if isinstance(attempt.status, AttemptStatus) else str(attempt.status)
        model.completed_at = attempt.completed_at
        model.duration_ms = attempt.duration_ms
        model.is_deterministic = bool(attempt.is_deterministic)
        model.nondeterminism_flags = list(attempt.nondeterminism_flags or [])
        model.request_payload = attempt.request_payload
        model.response_payload = attempt.response_payload
        model.response_hash = attempt.response_hash
        model.attempt_metadata = attempt.metadata or {}

        await self.session.flush()
        await self.session.refresh(model)
        return self._to_entity(model)

    async def list_for_step_execution(self, step_execution_id: UUID) -> List[StepExecutionAttempt]:
        stmt = (
            select(StepExecutionAttemptModel)
            .where(StepExecutionAttemptModel.step_execution_id == step_execution_id)
            .order_by(StepExecutionAttemptModel.attempt_number.asc())
        )
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._to_entity(m) for m in models]

    async def get_latest_for_step_execution(self, step_execution_id: UUID) -> Optional[StepExecutionAttempt]:
        stmt = (
            select(StepExecutionAttemptModel)
            .where(StepExecutionAttemptModel.step_execution_id == step_execution_id)
            .order_by(StepExecutionAttemptModel.attempt_number.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def get_latest_success_by_execution_and_step(self, execution_id: UUID, step_id: str) -> Optional[StepExecutionAttempt]:
        from infrastructure.persistence.workflow_models import StepExecutionModel

        stmt = (
            select(StepExecutionAttemptModel)
            .join(StepExecutionModel, StepExecutionModel.id == StepExecutionAttemptModel.step_execution_id)
            .where(
                and_(
                    StepExecutionModel.execution_id == execution_id,
                    StepExecutionModel.step_id == step_id,
                    StepExecutionAttemptModel.status == AttemptStatus.SUCCESS.value,
                )
            )
            .order_by(StepExecutionAttemptModel.attempt_number.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def get_latest_success_by_execution(self, execution_id: UUID) -> Dict[str, StepExecutionAttempt]:
        """
        Return a mapping step_id -> latest successful attempt for the given execution.

        Note: step_id is returned as the workflow step identifier string.
        """
        from infrastructure.persistence.workflow_models import StepExecutionModel

        stmt = (
            select(StepExecutionAttemptModel, StepExecutionModel.step_id)
            .join(StepExecutionModel, StepExecutionModel.id == StepExecutionAttemptModel.step_execution_id)
            .where(
                and_(
                    StepExecutionModel.execution_id == execution_id,
                    StepExecutionAttemptModel.status == AttemptStatus.SUCCESS.value,
                )
            )
            .order_by(StepExecutionModel.step_id.asc(), StepExecutionAttemptModel.attempt_number.desc())
        )
        result = await self.session.execute(stmt)
        rows = result.all()
        latest: Dict[str, StepExecutionAttempt] = {}
        for attempt_model, step_id in rows:
            if step_id not in latest:
                latest[step_id] = self._to_entity(attempt_model)
        return latest

    def _to_model(self, entity: StepExecutionAttempt) -> StepExecutionAttemptModel:
        return StepExecutionAttemptModel(
            id=entity.id,
            step_execution_id=entity.step_execution_id,
            attempt_number=entity.attempt_number,
            idempotency_key=entity.idempotency_key,
            inputs=entity.inputs,
            agent_id=entity.agent_id,
            started_at=entity.started_at,
            outputs=entity.outputs,
            error=entity.error,
            status=entity.status.value if isinstance(entity.status, AttemptStatus) else str(entity.status),
            completed_at=entity.completed_at,
            duration_ms=entity.duration_ms,
            is_deterministic=bool(entity.is_deterministic),
            nondeterminism_flags=list(entity.nondeterminism_flags or []),
            request_payload=entity.request_payload,
            response_payload=entity.response_payload,
            response_hash=entity.response_hash,
            attempt_metadata=entity.metadata or {},
            created_at=entity.created_at,
        )

    def _to_entity(self, model: StepExecutionAttemptModel) -> StepExecutionAttempt:
        return StepExecutionAttempt(
            id=model.id,
            step_execution_id=model.step_execution_id,
            attempt_number=model.attempt_number,
            idempotency_key=model.idempotency_key,
            inputs=model.inputs or {},
            status=AttemptStatus(model.status) if model.status else AttemptStatus.FAILED,
            started_at=model.started_at,
            agent_id=model.agent_id,
            outputs=model.outputs,
            error=model.error,
            completed_at=model.completed_at,
            duration_ms=model.duration_ms,
            is_deterministic=bool(model.is_deterministic),
            nondeterminism_flags=list(model.nondeterminism_flags or []),
            request_payload=model.request_payload,
            response_payload=model.response_payload,
            response_hash=model.response_hash,
            metadata=model.attempt_metadata or {},
            created_at=model.created_at,
        )


class PostgreSQLExecutionLockRepository:
    """PostgreSQL repository for distributed execution locks."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def acquire(
        self,
        *,
        lock_key: str,
        lock_holder: str,
        ttl_seconds: int = 60,
        metadata: Optional[Dict[str, object]] = None,
    ) -> bool:
        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=max(1, int(ttl_seconds)))

        stmt = (
            pg_insert(ExecutionLockModel)
            .values(
                id=func.gen_random_uuid(),
                lock_key=lock_key,
                lock_holder=lock_holder,
                acquired_at=now,
                expires_at=expires_at,
                lock_metadata=metadata or {},
            )
            .on_conflict_do_update(
                index_elements=[ExecutionLockModel.lock_key],
                set_={
                    "lock_holder": lock_holder,
                    "acquired_at": now,
                    "expires_at": expires_at,
                    "metadata": metadata or {},
                },
                where=ExecutionLockModel.expires_at < func.now(),
            )
            .returning(ExecutionLockModel.id)
        )
        result = await self.session.execute(stmt)
        row = result.first()
        return row is not None

    async def release(self, *, lock_key: str, lock_holder: str) -> bool:
        stmt = delete(ExecutionLockModel).where(
            and_(
                ExecutionLockModel.lock_key == lock_key,
                ExecutionLockModel.lock_holder == lock_holder,
            )
        )
        result = await self.session.execute(stmt)
        return bool(getattr(result, "rowcount", 0))

    async def cleanup_expired(self) -> int:
        stmt = delete(ExecutionLockModel).where(ExecutionLockModel.expires_at < func.now())
        result = await self.session.execute(stmt)
        return int(getattr(result, "rowcount", 0) or 0)


class PostgreSQLExecutionErrorRepository:
    """PostgreSQL repository for ExecutionError (unified error model)."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, error: ExecutionError) -> ExecutionError:
        model = self._to_model(error)
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        return self._to_entity(model)

    async def get_by_id(self, error_id: UUID) -> Optional[ExecutionError]:
        stmt = select(ExecutionErrorModel).where(ExecutionErrorModel.id == error_id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def list_by_correlation(self, correlation_id: UUID, limit: int = 100) -> List[ExecutionError]:
        stmt = (
            select(ExecutionErrorModel)
            .where(ExecutionErrorModel.correlation_id == correlation_id)
            .order_by(ExecutionErrorModel.created_at.desc())
            .limit(max(1, min(1000, limit)))
        )
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._to_entity(m) for m in models]

    def _to_model(self, entity: ExecutionError) -> ExecutionErrorModel:
        return ExecutionErrorModel(
            id=entity.id,
            error_code=entity.error_code,
            error_category=entity.error_category.value if isinstance(entity.error_category, ErrorCategory) else str(entity.error_category),
            severity=entity.severity.value if isinstance(entity.severity, ErrorSeverity) else str(entity.severity),
            message=entity.message,
            correlation_id=entity.correlation_id,
            is_retryable=bool(entity.is_retryable),
            details=entity.details or {},
            stack_trace=entity.stack_trace,
            service_name=entity.service_name,
            execution_id=entity.execution_id,
            step_execution_id=entity.step_execution_id,
            user_id=entity.user_id,
            retry_count=int(entity.retry_count or 0),
            max_retries=int(entity.max_retries or 3),
            next_retry_at=entity.next_retry_at,
            resolved=bool(entity.resolved),
            resolved_at=entity.resolved_at,
            resolution_notes=entity.resolution_notes,
            error_metadata=entity.metadata or {},
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )

    def _to_entity(self, model: ExecutionErrorModel) -> ExecutionError:
        return ExecutionError(
            id=model.id,
            error_code=model.error_code,
            error_category=ErrorCategory(model.error_category),
            severity=ErrorSeverity(model.severity),
            message=model.message,
            correlation_id=model.correlation_id,
            is_retryable=bool(model.is_retryable),
            details=model.details or {},
            stack_trace=model.stack_trace,
            service_name=model.service_name,
            execution_id=model.execution_id,
            step_execution_id=model.step_execution_id,
            user_id=model.user_id,
            retry_count=int(model.retry_count or 0),
            max_retries=int(model.max_retries or 3),
            next_retry_at=model.next_retry_at,
            resolved=bool(model.resolved),
            resolved_at=model.resolved_at,
            resolution_notes=model.resolution_notes,
            metadata=model.error_metadata or {},
            created_at=model.created_at,
            updated_at=model.updated_at,
        )
