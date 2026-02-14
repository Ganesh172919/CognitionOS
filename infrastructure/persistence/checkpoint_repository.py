"""
Checkpoint Infrastructure - PostgreSQL Repository Implementation

Concrete implementation of CheckpointRepository using PostgreSQL.
"""

from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, delete, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from core.domain.checkpoint.entities import (
    Checkpoint,
    CheckpointId,
    CheckpointStatus,
    ExecutionSnapshot,
    DAGProgress,
    BudgetSnapshot
)
from core.domain.checkpoint.repositories import CheckpointRepository

from infrastructure.persistence.checkpoint_models import CheckpointModel


class PostgreSQLCheckpointRepository(CheckpointRepository):
    """
    PostgreSQL implementation of CheckpointRepository.

    Maps between domain entities and SQLAlchemy models.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, checkpoint: Checkpoint) -> None:
        """Persist checkpoint to database"""
        model = self._to_model(checkpoint)
        self.session.add(model)
        await self.session.flush()

    async def find_by_id(self, checkpoint_id: UUID) -> Optional[Checkpoint]:
        """Retrieve checkpoint by ID"""
        stmt = select(CheckpointModel).where(CheckpointModel.id == checkpoint_id)
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
        
        return self._to_entity(model)

    async def find_by_workflow_execution(
        self,
        workflow_execution_id: UUID,
        limit: Optional[int] = None,
    ) -> List[Checkpoint]:
        """Find checkpoints for a workflow execution"""
        stmt = (
            select(CheckpointModel)
            .where(CheckpointModel.workflow_execution_id == workflow_execution_id)
            .order_by(CheckpointModel.checkpoint_number.desc())
        )
        
        if limit:
            stmt = stmt.limit(limit)
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]

    async def find_latest_by_workflow_execution(
        self,
        workflow_execution_id: UUID,
    ) -> Optional[Checkpoint]:
        """Find the latest checkpoint for a workflow execution"""
        stmt = (
            select(CheckpointModel)
            .where(CheckpointModel.workflow_execution_id == workflow_execution_id)
            .order_by(CheckpointModel.checkpoint_number.desc())
            .limit(1)
        )
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
        
        return self._to_entity(model)

    async def find_by_workflow_and_number(
        self,
        workflow_execution_id: UUID,
        checkpoint_number: int,
    ) -> Optional[Checkpoint]:
        """Find checkpoint by workflow execution ID and checkpoint number"""
        stmt = select(CheckpointModel).where(
            and_(
                CheckpointModel.workflow_execution_id == workflow_execution_id,
                CheckpointModel.checkpoint_number == checkpoint_number
            )
        )
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
        
        return self._to_entity(model)

    async def delete(self, checkpoint_id: UUID) -> bool:
        """Delete a checkpoint"""
        stmt = select(CheckpointModel).where(CheckpointModel.id == checkpoint_id)
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return False
        
        await self.session.delete(model)
        await self.session.flush()
        return True

    async def delete_old_checkpoints(
        self,
        workflow_execution_id: UUID,
        keep_count: int,
    ) -> int:
        """Delete old checkpoints, keeping only the N most recent"""
        # Find checkpoint numbers to keep
        stmt_keep = (
            select(CheckpointModel.checkpoint_number)
            .where(CheckpointModel.workflow_execution_id == workflow_execution_id)
            .order_by(CheckpointModel.checkpoint_number.desc())
            .limit(keep_count)
        )
        
        result_keep = await self.session.execute(stmt_keep)
        keep_numbers = [row[0] for row in result_keep.all()]
        
        if not keep_numbers:
            return 0
        
        # Delete checkpoints not in keep list
        stmt_delete = delete(CheckpointModel).where(
            and_(
                CheckpointModel.workflow_execution_id == workflow_execution_id,
                CheckpointModel.checkpoint_number.notin_(keep_numbers)
            )
        )
        
        result = await self.session.execute(stmt_delete)
        await self.session.flush()
        
        return result.rowcount

    async def get_checkpoint_count(
        self,
        workflow_execution_id: UUID,
    ) -> int:
        """Get count of checkpoints for a workflow execution"""
        stmt = (
            select(func.count())
            .select_from(CheckpointModel)
            .where(CheckpointModel.workflow_execution_id == workflow_execution_id)
        )
        
        result = await self.session.execute(stmt)
        count = result.scalar_one()
        
        return count

    async def exists(
        self,
        workflow_execution_id: UUID,
        checkpoint_number: int,
    ) -> bool:
        """Check if a checkpoint exists"""
        checkpoint = await self.find_by_workflow_and_number(
            workflow_execution_id,
            checkpoint_number
        )
        return checkpoint is not None

    def _to_model(self, checkpoint: Checkpoint) -> CheckpointModel:
        """Convert domain entity to ORM model"""
        return CheckpointModel(
            id=checkpoint.id.value,
            workflow_execution_id=checkpoint.workflow_execution_id,
            checkpoint_number=checkpoint.checkpoint_number,
            created_at=checkpoint.created_at,
            execution_state=checkpoint.execution_state.to_dict(),
            dag_progress=checkpoint.dag_progress.to_dict(),
            memory_snapshot_ref=checkpoint.memory_snapshot_ref,
            active_tasks=checkpoint.active_tasks,
            budget_state=checkpoint.budget_state.to_dict(),
            checkpoint_size_bytes=checkpoint.checkpoint_size_bytes,
            compression_enabled=checkpoint.compression_enabled,
            metadata=checkpoint.metadata
        )

    def _to_entity(self, model: CheckpointModel) -> Checkpoint:
        """Convert ORM model to domain entity"""
        return Checkpoint(
            id=CheckpointId(value=model.id),
            workflow_execution_id=model.workflow_execution_id,
            checkpoint_number=model.checkpoint_number,
            status=CheckpointStatus.READY,
            execution_state=ExecutionSnapshot.from_dict(model.execution_state),
            dag_progress=DAGProgress.from_dict(model.dag_progress),
            budget_state=BudgetSnapshot.from_dict(model.budget_state or {}),
            memory_snapshot_ref=model.memory_snapshot_ref,
            active_tasks=model.active_tasks or [],
            checkpoint_size_bytes=model.checkpoint_size_bytes,
            compression_enabled=model.compression_enabled,
            metadata=model.metadata or {},
            created_at=model.created_at
        )
