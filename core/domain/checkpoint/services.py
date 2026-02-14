"""
Checkpoint Domain - Services

Domain services for checkpoint orchestration and business logic.
"""

from typing import List, Optional
from uuid import UUID

from .entities import (
    Checkpoint,
    CheckpointStatus,
    ExecutionSnapshot,
    DAGProgress,
    BudgetSnapshot,
)
from .repositories import CheckpointRepository


class CheckpointService:
    """
    Domain service for checkpoint operations.
    
    Orchestrates checkpoint creation, restoration, and lifecycle management.
    """

    def __init__(self, checkpoint_repository: CheckpointRepository):
        """
        Initialize checkpoint service.
        
        Args:
            checkpoint_repository: Checkpoint repository
        """
        self.checkpoint_repository = checkpoint_repository

    async def create_checkpoint(
        self,
        workflow_execution_id: UUID,
        execution_state: ExecutionSnapshot,
        dag_progress: DAGProgress,
        budget_state: BudgetSnapshot,
        memory_snapshot_ref: Optional[str] = None,
        active_tasks: Optional[List[dict]] = None,
        compression_enabled: bool = True,
    ) -> Checkpoint:
        """
        Create a new checkpoint.
        
        Args:
            workflow_execution_id: Workflow execution ID
            execution_state: Current execution state
            dag_progress: DAG progress snapshot
            budget_state: Budget state snapshot
            memory_snapshot_ref: Reference to memory snapshot
            active_tasks: Active tasks snapshot
            compression_enabled: Enable compression
            
        Returns:
            Created checkpoint
        """
        # Determine checkpoint number (next sequential)
        checkpoint_count = await self.checkpoint_repository.get_checkpoint_count(
            workflow_execution_id
        )
        checkpoint_number = checkpoint_count + 1

        # Create checkpoint
        checkpoint = Checkpoint.create(
            workflow_execution_id=workflow_execution_id,
            checkpoint_number=checkpoint_number,
            execution_state=execution_state,
            dag_progress=dag_progress,
            budget_state=budget_state,
            memory_snapshot_ref=memory_snapshot_ref,
            active_tasks=active_tasks,
            compression_enabled=compression_enabled,
        )

        # Mark as ready
        checkpoint.mark_as_ready()

        # Save checkpoint
        await self.checkpoint_repository.save(checkpoint)

        return checkpoint

    async def restore_checkpoint(
        self,
        checkpoint_id: UUID,
    ) -> Checkpoint:
        """
        Restore a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID
            
        Returns:
            Restored checkpoint
            
        Raises:
            ValueError: If checkpoint not found or cannot be restored
        """
        # Find checkpoint
        checkpoint = await self.checkpoint_repository.find_by_id(checkpoint_id)
        if not checkpoint:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        # Verify checkpoint can be restored
        if not checkpoint.can_be_restored():
            raise ValueError(
                f"Checkpoint cannot be restored (status: {checkpoint.status})"
            )

        # Mark as restoring
        checkpoint.mark_as_restoring()
        await self.checkpoint_repository.save(checkpoint)

        # Mark as restored
        checkpoint.mark_as_restored()
        await self.checkpoint_repository.save(checkpoint)

        return checkpoint

    async def find_latest_checkpoint(
        self,
        workflow_execution_id: UUID,
    ) -> Optional[Checkpoint]:
        """
        Find the latest checkpoint for a workflow.
        
        Args:
            workflow_execution_id: Workflow execution ID
            
        Returns:
            Latest checkpoint if found, None otherwise
        """
        return await self.checkpoint_repository.find_latest_by_workflow_execution(
            workflow_execution_id
        )

    async def cleanup_old_checkpoints(
        self,
        workflow_execution_id: UUID,
        max_checkpoints: int = 10,
    ) -> int:
        """
        Cleanup old checkpoints, keeping only the most recent.
        
        Args:
            workflow_execution_id: Workflow execution ID
            max_checkpoints: Maximum checkpoints to keep (default: 10)
            
        Returns:
            Number of checkpoints deleted
        """
        return await self.checkpoint_repository.delete_old_checkpoints(
            workflow_execution_id=workflow_execution_id,
            keep_count=max_checkpoints,
        )

    async def get_checkpoint_history(
        self,
        workflow_execution_id: UUID,
        limit: Optional[int] = None,
    ) -> List[Checkpoint]:
        """
        Get checkpoint history for a workflow.
        
        Args:
            workflow_execution_id: Workflow execution ID
            limit: Maximum number of checkpoints to return
            
        Returns:
            List of checkpoints (newest first)
        """
        return await self.checkpoint_repository.find_by_workflow_execution(
            workflow_execution_id=workflow_execution_id,
            limit=limit,
        )

    async def estimate_checkpoint_overhead(
        self,
        checkpoint: Checkpoint,
    ) -> dict:
        """
        Estimate checkpoint overhead.
        
        Args:
            checkpoint: Checkpoint to analyze
            
        Returns:
            Dict with overhead metrics
        """
        size_bytes = checkpoint.get_memory_footprint_estimate()
        
        return {
            "size_bytes": size_bytes,
            "size_kb": size_bytes / 1024,
            "size_mb": size_bytes / (1024 * 1024),
            "compressed": checkpoint.is_compressed(),
            "completion_percentage": checkpoint.get_completion_percentage(),
            "budget_usage_percentage": checkpoint.get_budget_usage_percentage(),
        }

    def validate_checkpoint_invariants(
        self,
        checkpoint: Checkpoint,
    ) -> List[str]:
        """
        Validate checkpoint business invariants.
        
        Args:
            checkpoint: Checkpoint to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Validate checkpoint number
        if checkpoint.checkpoint_number < 1:
            errors.append("Checkpoint number must be positive")

        # Validate DAG progress
        if checkpoint.dag_progress.total_steps < 0:
            errors.append("Total steps cannot be negative")

        total_accounted = (
            len(checkpoint.dag_progress.completed_steps)
            + len(checkpoint.dag_progress.pending_steps)
            + len(checkpoint.dag_progress.failed_steps)
            + len(checkpoint.dag_progress.skipped_steps)
        )
        if total_accounted > checkpoint.dag_progress.total_steps:
            errors.append("Accounted steps exceed total steps")

        # Validate budget
        if checkpoint.budget_state.consumed < 0:
            errors.append("Budget consumed cannot be negative")

        if checkpoint.budget_state.consumed > checkpoint.budget_state.allocated:
            errors.append("Budget consumed exceeds allocated budget")

        # Validate status
        if checkpoint.status == CheckpointStatus.RESTORING and not checkpoint.restored_at:
            # OK - currently restoring
            pass
        elif checkpoint.status == CheckpointStatus.READY and checkpoint.restored_at:
            # Check if restoration time is reasonable
            pass

        return errors
