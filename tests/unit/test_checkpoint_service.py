"""
Unit Tests for Checkpoint Domain Services

Tests for checkpoint service business logic and orchestration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from core.domain.checkpoint.entities import (
    Checkpoint,
    CheckpointStatus,
    ExecutionSnapshot,
    DAGProgress,
    BudgetSnapshot,
)
from core.domain.checkpoint.services import CheckpointService


class TestCheckpointService:
    """Tests for CheckpointService domain service"""
    
    @pytest.fixture
    def mock_checkpoint_repository(self):
        """Create mock checkpoint repository"""
        repository = AsyncMock()
        repository.get_checkpoint_count = AsyncMock(return_value=0)
        repository.save = AsyncMock()
        repository.find_by_id = AsyncMock()
        repository.find_latest_by_workflow_execution = AsyncMock()
        repository.delete_old_checkpoints = AsyncMock(return_value=5)
        return repository
    
    @pytest.fixture
    def checkpoint_service(self, mock_checkpoint_repository):
        """Create checkpoint service instance"""
        return CheckpointService(mock_checkpoint_repository)
    
    @pytest.fixture
    def sample_execution_snapshot(self):
        """Create sample execution snapshot"""
        return ExecutionSnapshot(
            variables={"x": 1, "y": 2},
            context={"user_id": "test-user"},
            current_step_id="step-1"
        )
    
    @pytest.fixture
    def sample_dag_progress(self):
        """Create sample DAG progress"""
        return DAGProgress(
            completed_steps=["step-1", "step-2"],
            pending_steps=["step-3"],
            failed_steps=[],
            skipped_steps=[],
            total_steps=3,
            completion_percentage=66.67
        )
    
    @pytest.fixture
    def sample_budget_snapshot(self):
        """Create sample budget snapshot"""
        return BudgetSnapshot(
            allocated=100.0,
            consumed=25.0,
            remaining=75.0,
            warning_threshold=0.8,
            critical_threshold=0.95,
            status="active"
        )
    
    @pytest.mark.asyncio
    async def test_create_checkpoint(
        self,
        checkpoint_service,
        mock_checkpoint_repository,
        sample_execution_snapshot,
        sample_dag_progress,
        sample_budget_snapshot
    ):
        """Test creating a checkpoint"""
        workflow_id = uuid4()
        
        # Create checkpoint
        checkpoint = await checkpoint_service.create_checkpoint(
            workflow_execution_id=workflow_id,
            execution_state=sample_execution_snapshot,
            dag_progress=sample_dag_progress,
            budget_state=sample_budget_snapshot
        )
        
        # Verify checkpoint was created
        assert checkpoint is not None
        assert checkpoint.workflow_execution_id == workflow_id
        assert checkpoint.checkpoint_number == 1  # First checkpoint
        assert checkpoint.status == CheckpointStatus.READY
        
        # Verify repository was called
        mock_checkpoint_repository.get_checkpoint_count.assert_called_once_with(workflow_id)
        mock_checkpoint_repository.save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_checkpoint_sequential_numbering(
        self,
        checkpoint_service,
        mock_checkpoint_repository,
        sample_execution_snapshot,
        sample_dag_progress,
        sample_budget_snapshot
    ):
        """Test checkpoint sequential numbering"""
        workflow_id = uuid4()
        
        # Simulate existing checkpoints
        mock_checkpoint_repository.get_checkpoint_count.return_value = 5
        
        # Create checkpoint
        checkpoint = await checkpoint_service.create_checkpoint(
            workflow_execution_id=workflow_id,
            execution_state=sample_execution_snapshot,
            dag_progress=sample_dag_progress,
            budget_state=sample_budget_snapshot
        )
        
        # Verify checkpoint number is sequential
        assert checkpoint.checkpoint_number == 6  # Next after 5
    
    @pytest.mark.asyncio
    async def test_create_checkpoint_with_memory_reference(
        self,
        checkpoint_service,
        mock_checkpoint_repository,
        sample_execution_snapshot,
        sample_dag_progress,
        sample_budget_snapshot
    ):
        """Test creating checkpoint with memory snapshot reference"""
        workflow_id = uuid4()
        memory_ref = "s3://bucket/memory-snapshot-123.json"
        
        checkpoint = await checkpoint_service.create_checkpoint(
            workflow_execution_id=workflow_id,
            execution_state=sample_execution_snapshot,
            dag_progress=sample_dag_progress,
            budget_state=sample_budget_snapshot,
            memory_snapshot_ref=memory_ref
        )
        
        assert checkpoint.memory_snapshot_ref == memory_ref
    
    @pytest.mark.asyncio
    async def test_create_checkpoint_with_active_tasks(
        self,
        checkpoint_service,
        mock_checkpoint_repository,
        sample_execution_snapshot,
        sample_dag_progress,
        sample_budget_snapshot
    ):
        """Test creating checkpoint with active tasks"""
        workflow_id = uuid4()
        active_tasks = [
            {"task_id": "task-1", "status": "running"},
            {"task_id": "task-2", "status": "pending"}
        ]
        
        checkpoint = await checkpoint_service.create_checkpoint(
            workflow_execution_id=workflow_id,
            execution_state=sample_execution_snapshot,
            dag_progress=sample_dag_progress,
            budget_state=sample_budget_snapshot,
            active_tasks=active_tasks
        )
        
        assert len(checkpoint.active_tasks) == 2
        assert checkpoint.active_tasks[0]["task_id"] == "task-1"
    
    @pytest.mark.asyncio
    async def test_restore_checkpoint_success(
        self,
        checkpoint_service,
        mock_checkpoint_repository,
        sample_execution_snapshot,
        sample_dag_progress,
        sample_budget_snapshot
    ):
        """Test successfully restoring a checkpoint"""
        checkpoint_id = uuid4()
        workflow_id = uuid4()
        
        # Create a ready checkpoint
        existing_checkpoint = Checkpoint.create(
            workflow_execution_id=workflow_id,
            checkpoint_number=1,
            execution_state=sample_execution_snapshot,
            dag_progress=sample_dag_progress,
            budget_state=sample_budget_snapshot
        )
        existing_checkpoint.mark_as_ready()
        
        # Mock repository to return existing checkpoint
        mock_checkpoint_repository.find_by_id.return_value = existing_checkpoint
        
        # Restore checkpoint
        restored = await checkpoint_service.restore_checkpoint(checkpoint_id)
        
        # Verify restoration
        assert restored is not None
        assert restored.workflow_execution_id == workflow_id
        assert restored.restored_at is not None
        
        # Verify repository was called correctly
        mock_checkpoint_repository.find_by_id.assert_called_once_with(checkpoint_id)
        assert mock_checkpoint_repository.save.call_count == 2  # Once for restoring, once for restored
    
    @pytest.mark.asyncio
    async def test_restore_checkpoint_not_found(
        self,
        checkpoint_service,
        mock_checkpoint_repository
    ):
        """Test restoring a non-existent checkpoint"""
        checkpoint_id = uuid4()
        
        # Mock repository to return None
        mock_checkpoint_repository.find_by_id.return_value = None
        
        # Attempt to restore
        with pytest.raises(ValueError, match="Checkpoint not found"):
            await checkpoint_service.restore_checkpoint(checkpoint_id)
    
    @pytest.mark.asyncio
    async def test_restore_checkpoint_not_ready(
        self,
        checkpoint_service,
        mock_checkpoint_repository,
        sample_execution_snapshot,
        sample_dag_progress,
        sample_budget_snapshot
    ):
        """Test restoring a checkpoint that's not ready"""
        checkpoint_id = uuid4()
        workflow_id = uuid4()
        
        # Create a checkpoint in CREATING state
        existing_checkpoint = Checkpoint.create(
            workflow_execution_id=workflow_id,
            checkpoint_number=1,
            execution_state=sample_execution_snapshot,
            dag_progress=sample_dag_progress,
            budget_state=sample_budget_snapshot
        )
        # Don't mark as ready
        
        mock_checkpoint_repository.find_by_id.return_value = existing_checkpoint
        
        # Attempt to restore
        with pytest.raises(ValueError, match="cannot be restored"):
            await checkpoint_service.restore_checkpoint(checkpoint_id)
    
    @pytest.mark.asyncio
    async def test_find_latest_checkpoint(
        self,
        checkpoint_service,
        mock_checkpoint_repository,
        sample_execution_snapshot,
        sample_dag_progress,
        sample_budget_snapshot
    ):
        """Test finding latest checkpoint for workflow"""
        workflow_id = uuid4()
        
        # Create latest checkpoint
        latest_checkpoint = Checkpoint.create(
            workflow_execution_id=workflow_id,
            checkpoint_number=5,
            execution_state=sample_execution_snapshot,
            dag_progress=sample_dag_progress,
            budget_state=sample_budget_snapshot
        )
        latest_checkpoint.mark_as_ready()
        
        mock_checkpoint_repository.find_latest_by_workflow_execution.return_value = latest_checkpoint
        
        # Find latest
        result = await checkpoint_service.find_latest_checkpoint(workflow_id)
        
        assert result is not None
        assert result.checkpoint_number == 5
        mock_checkpoint_repository.find_latest_by_workflow_execution.assert_called_once_with(workflow_id)
    
    @pytest.mark.asyncio
    async def test_find_latest_checkpoint_none_exist(
        self,
        checkpoint_service,
        mock_checkpoint_repository
    ):
        """Test finding latest checkpoint when none exist"""
        workflow_id = uuid4()
        
        mock_checkpoint_repository.find_latest_by_workflow_execution.return_value = None
        
        result = await checkpoint_service.find_latest_checkpoint(workflow_id)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cleanup_old_checkpoints(
        self,
        checkpoint_service,
        mock_checkpoint_repository
    ):
        """Test cleaning up old checkpoints"""
        workflow_id = uuid4()
        max_checkpoints = 10
        
        # Mock repository to return 5 deleted
        mock_checkpoint_repository.delete_old_checkpoints.return_value = 5
        
        # Cleanup
        deleted_count = await checkpoint_service.cleanup_old_checkpoints(
            workflow_id,
            max_checkpoints
        )
        
        assert deleted_count == 5
        mock_checkpoint_repository.delete_old_checkpoints.assert_called_once_with(
            workflow_execution_id=workflow_id,
            keep_count=max_checkpoints
        )
    
    @pytest.mark.asyncio
    async def test_estimate_checkpoint_overhead(
        self,
        checkpoint_service,
        sample_execution_snapshot,
        sample_dag_progress,
        sample_budget_snapshot
    ):
        """Test estimating checkpoint overhead"""
        workflow_id = uuid4()
        
        checkpoint = Checkpoint.create(
            workflow_execution_id=workflow_id,
            checkpoint_number=1,
            execution_state=sample_execution_snapshot,
            dag_progress=sample_dag_progress,
            budget_state=sample_budget_snapshot
        )
        
        # Estimate overhead
        overhead = await checkpoint_service.estimate_checkpoint_overhead(checkpoint)
        
        assert "size_bytes" in overhead
        assert "size_kb" in overhead
        assert "size_mb" in overhead
        assert "compressed" in overhead
        assert "completion_percentage" in overhead
        assert "budget_usage_percentage" in overhead
        
        assert overhead["compressed"] is True
        assert overhead["completion_percentage"] == 66.67
    
    @pytest.mark.asyncio
    async def test_validate_checkpoint_invariants(
        self,
        checkpoint_service,
        sample_execution_snapshot,
        sample_dag_progress,
        sample_budget_snapshot
    ):
        """Test validating checkpoint business invariants"""
        workflow_id = uuid4()
        
        checkpoint = Checkpoint.create(
            workflow_execution_id=workflow_id,
            checkpoint_number=1,
            execution_state=sample_execution_snapshot,
            dag_progress=sample_dag_progress,
            budget_state=sample_budget_snapshot
        )
        
        # Valid checkpoint should have no errors
        errors = checkpoint_service.validate_checkpoint_invariants(checkpoint)
        assert len(errors) == 0
    
    @pytest.mark.asyncio
    async def test_validate_checkpoint_invariants_invalid_number(
        self,
        checkpoint_service,
        sample_execution_snapshot,
        sample_budget_snapshot
    ):
        """Test validating checkpoint with invalid data"""
        workflow_id = uuid4()
        
        # Create checkpoint with invalid DAG progress
        invalid_dag_progress = DAGProgress(
            completed_steps=["a", "b", "c", "d", "e"],  # 5 steps
            pending_steps=[],
            failed_steps=[],
            skipped_steps=[],
            total_steps=3,  # But total is only 3!
            completion_percentage=166.67  # Invalid!
        )
        
        checkpoint = Checkpoint.create(
            workflow_execution_id=workflow_id,
            checkpoint_number=1,
            execution_state=sample_execution_snapshot,
            dag_progress=invalid_dag_progress,
            budget_state=sample_budget_snapshot
        )
        
        # Should have validation errors
        errors = checkpoint_service.validate_checkpoint_invariants(checkpoint)
        assert len(errors) > 0
        assert any("exceed total steps" in error for error in errors)
