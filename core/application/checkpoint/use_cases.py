"""
Checkpoint Application - Use Cases

Application layer use cases for Checkpoint bounded context.
Orchestrates domain entities and coordinates with infrastructure.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID

from core.domain.checkpoint import (
    Checkpoint,
    CheckpointId,
    CheckpointStatus,
    ExecutionSnapshot,
    DAGProgress,
    BudgetSnapshot,
    CheckpointRepository,
    CheckpointService,
    CheckpointCreated,
    CheckpointRestored,
    CheckpointDeleted,
)


# ==================== DTOs (Data Transfer Objects) ====================

@dataclass
class CreateCheckpointCommand:
    """Command to create a checkpoint from workflow execution state"""
    workflow_execution_id: UUID
    execution_variables: Dict[str, Any]
    execution_context: Dict[str, Any]
    current_step_id: Optional[str]
    completed_steps: List[str]
    pending_steps: List[str]
    failed_steps: List[str]
    skipped_steps: List[str]
    total_steps: int
    allocated_budget: float
    consumed_budget: float
    memory_snapshot_ref: Optional[str] = None
    active_tasks: Optional[List[Dict[str, Any]]] = None
    compression_enabled: bool = True
    error_state: Optional[Dict[str, Any]] = None


@dataclass
class RestoreCheckpointCommand:
    """Command to restore workflow from checkpoint"""
    checkpoint_id: UUID
    recovery_reason: str


@dataclass
class ListCheckpointsQuery:
    """Query to list checkpoints for a workflow"""
    workflow_execution_id: UUID
    limit: Optional[int] = None


@dataclass
class CleanupCheckpointsCommand:
    """Command to cleanup old checkpoints"""
    workflow_execution_id: UUID
    max_checkpoints: int = 10


@dataclass
class CheckpointResult:
    """Result of checkpoint operation"""
    checkpoint_id: UUID
    workflow_execution_id: UUID
    checkpoint_number: int
    status: CheckpointStatus
    completion_percentage: float
    budget_consumed: float
    checkpoint_size_bytes: Optional[int] = None
    created_at: str = None


# ==================== Use Cases ====================

class CreateCheckpointUseCase:
    """
    Use Case: Create a checkpoint from workflow execution state.

    Orchestrates:
    1. Build execution snapshot
    2. Build DAG progress snapshot
    3. Build budget snapshot
    4. Create checkpoint via domain service
    5. Publish domain event
    """

    def __init__(
        self,
        checkpoint_service: CheckpointService,
        event_publisher: Optional[Any] = None
    ):
        self.checkpoint_service = checkpoint_service
        self.event_publisher = event_publisher

    async def execute(self, command: CreateCheckpointCommand) -> CheckpointResult:
        """
        Create a new checkpoint.

        Args:
            command: Create checkpoint command

        Returns:
            CheckpointResult with checkpoint details

        Raises:
            ValueError: If checkpoint creation fails
        """
        # Build execution snapshot
        execution_state = ExecutionSnapshot(
            variables=command.execution_variables,
            context=command.execution_context,
            current_step_id=command.current_step_id,
            error_state=command.error_state,
        )

        # Calculate completion percentage
        completion_percentage = 0.0
        if command.total_steps > 0:
            completion_percentage = (len(command.completed_steps) / command.total_steps) * 100

        # Build DAG progress snapshot
        dag_progress = DAGProgress(
            completed_steps=command.completed_steps,
            pending_steps=command.pending_steps,
            failed_steps=command.failed_steps,
            skipped_steps=command.skipped_steps,
            total_steps=command.total_steps,
            completion_percentage=completion_percentage,
        )

        # Calculate budget metrics
        remaining_budget = max(0.0, command.allocated_budget - command.consumed_budget)
        warning_threshold = 0.8
        critical_threshold = 0.95
        
        budget_status = "active"
        if command.allocated_budget > 0:
            usage_percentage = command.consumed_budget / command.allocated_budget
            if usage_percentage >= 1.0:
                budget_status = "exhausted"
            elif usage_percentage >= critical_threshold:
                budget_status = "critical"
            elif usage_percentage >= warning_threshold:
                budget_status = "warning"

        # Build budget snapshot
        budget_state = BudgetSnapshot(
            allocated=command.allocated_budget,
            consumed=command.consumed_budget,
            remaining=remaining_budget,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            status=budget_status,
        )

        # Create checkpoint via domain service
        checkpoint = await self.checkpoint_service.create_checkpoint(
            workflow_execution_id=command.workflow_execution_id,
            execution_state=execution_state,
            dag_progress=dag_progress,
            budget_state=budget_state,
            memory_snapshot_ref=command.memory_snapshot_ref,
            active_tasks=command.active_tasks,
            compression_enabled=command.compression_enabled,
        )

        # Publish event
        if self.event_publisher:
            event = CheckpointCreated.create(
                checkpoint_id=checkpoint.id.value,
                workflow_execution_id=command.workflow_execution_id,
                checkpoint_number=checkpoint.checkpoint_number,
                completion_percentage=completion_percentage,
                budget_consumed=command.consumed_budget,
            )
            await self.event_publisher.publish(event)

        return CheckpointResult(
            checkpoint_id=checkpoint.id.value,
            workflow_execution_id=command.workflow_execution_id,
            checkpoint_number=checkpoint.checkpoint_number,
            status=checkpoint.status,
            completion_percentage=completion_percentage,
            budget_consumed=command.consumed_budget,
            checkpoint_size_bytes=checkpoint.checkpoint_size_bytes,
            created_at=checkpoint.created_at.isoformat(),
        )


class RestoreCheckpointUseCase:
    """
    Use Case: Restore workflow from checkpoint.

    Orchestrates:
    1. Restore checkpoint via domain service
    2. Validate checkpoint can be restored
    3. Publish domain event
    """

    def __init__(
        self,
        checkpoint_service: CheckpointService,
        event_publisher: Optional[Any] = None
    ):
        self.checkpoint_service = checkpoint_service
        self.event_publisher = event_publisher

    async def execute(self, command: RestoreCheckpointCommand) -> CheckpointResult:
        """
        Restore a checkpoint.

        Args:
            command: Restore checkpoint command

        Returns:
            CheckpointResult with restored checkpoint details

        Raises:
            ValueError: If checkpoint cannot be restored
        """
        # Restore checkpoint via domain service
        checkpoint = await self.checkpoint_service.restore_checkpoint(
            checkpoint_id=command.checkpoint_id
        )

        # Publish event
        if self.event_publisher:
            event = CheckpointRestored.create(
                checkpoint_id=checkpoint.id.value,
                workflow_execution_id=checkpoint.workflow_execution_id,
                checkpoint_number=checkpoint.checkpoint_number,
                recovery_reason=command.recovery_reason,
            )
            await self.event_publisher.publish(event)

        return CheckpointResult(
            checkpoint_id=checkpoint.id.value,
            workflow_execution_id=checkpoint.workflow_execution_id,
            checkpoint_number=checkpoint.checkpoint_number,
            status=checkpoint.status,
            completion_percentage=checkpoint.get_completion_percentage(),
            budget_consumed=checkpoint.budget_state.consumed,
            checkpoint_size_bytes=checkpoint.checkpoint_size_bytes,
            created_at=checkpoint.created_at.isoformat(),
        )


class ListCheckpointsUseCase:
    """
    Use Case: List checkpoints for a workflow.

    Retrieves checkpoint history for a workflow execution.
    """

    def __init__(
        self,
        checkpoint_service: CheckpointService
    ):
        self.checkpoint_service = checkpoint_service

    async def execute(self, query: ListCheckpointsQuery) -> List[CheckpointResult]:
        """
        List checkpoints for a workflow.

        Args:
            query: List checkpoints query

        Returns:
            List of checkpoint results
        """
        # Get checkpoint history
        checkpoints = await self.checkpoint_service.get_checkpoint_history(
            workflow_execution_id=query.workflow_execution_id,
            limit=query.limit,
        )

        # Convert to results
        results = []
        for checkpoint in checkpoints:
            result = CheckpointResult(
                checkpoint_id=checkpoint.id.value,
                workflow_execution_id=checkpoint.workflow_execution_id,
                checkpoint_number=checkpoint.checkpoint_number,
                status=checkpoint.status,
                completion_percentage=checkpoint.get_completion_percentage(),
                budget_consumed=checkpoint.budget_state.consumed,
                checkpoint_size_bytes=checkpoint.checkpoint_size_bytes,
                created_at=checkpoint.created_at.isoformat(),
            )
            results.append(result)

        return results


class CleanupOldCheckpointsUseCase:
    """
    Use Case: Cleanup old checkpoints.

    Orchestrates:
    1. Delete old checkpoints keeping N most recent
    2. Publish deletion events
    """

    def __init__(
        self,
        checkpoint_service: CheckpointService,
        event_publisher: Optional[Any] = None
    ):
        self.checkpoint_service = checkpoint_service
        self.event_publisher = event_publisher

    async def execute(self, command: CleanupCheckpointsCommand) -> int:
        """
        Cleanup old checkpoints.

        Args:
            command: Cleanup checkpoints command

        Returns:
            Number of checkpoints deleted
        """
        # Cleanup via domain service
        deleted_count = await self.checkpoint_service.cleanup_old_checkpoints(
            workflow_execution_id=command.workflow_execution_id,
            max_checkpoints=command.max_checkpoints,
        )

        # Note: Individual checkpoint deletion events could be published if needed
        # For now, we just return the count

        return deleted_count
