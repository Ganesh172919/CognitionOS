"""
Unit Tests for Checkpoint Domain Entities

Tests for checkpoint/resume domain entities and value objects.
"""

import pytest
from datetime import datetime
from uuid import uuid4, UUID

from core.domain.checkpoint.entities import (
    Checkpoint,
    CheckpointId,
    CheckpointStatus,
    ExecutionSnapshot,
    DAGProgress,
    BudgetSnapshot,
)


class TestCheckpointId:
    """Tests for CheckpointId value object"""
    
    def test_checkpoint_id_creation(self):
        """Test creating a checkpoint ID"""
        checkpoint_id = CheckpointId.generate()
        assert isinstance(checkpoint_id.value, UUID)
    
    def test_checkpoint_id_immutable(self):
        """Test that checkpoint ID is immutable"""
        checkpoint_id = CheckpointId(value=uuid4())
        with pytest.raises(AttributeError):
            checkpoint_id.value = uuid4()


class TestExecutionSnapshot:
    """Tests for ExecutionSnapshot value object"""
    
    def test_execution_snapshot_creation(self):
        """Test creating an execution snapshot"""
        snapshot = ExecutionSnapshot(
            variables={"x": 1, "y": 2},
            context={"user_id": "123"},
            current_step_id="step-1"
        )
        assert snapshot.variables == {"x": 1, "y": 2}
        assert snapshot.context == {"user_id": "123"}
        assert snapshot.current_step_id == "step-1"
    
    def test_execution_snapshot_to_dict(self):
        """Test converting execution snapshot to dictionary"""
        snapshot = ExecutionSnapshot(
            variables={"x": 1},
            context={"test": "data"},
            current_step_id="step-1"
        )
        data = snapshot.to_dict()
        
        assert data["variables"] == {"x": 1}
        assert data["context"] == {"test": "data"}
        assert data["current_step_id"] == "step-1"
    
    def test_execution_snapshot_from_dict(self):
        """Test creating execution snapshot from dictionary"""
        data = {
            "variables": {"a": 1},
            "context": {"b": 2},
            "current_step_id": "step-2",
            "error_state": {"error": "test"}
        }
        snapshot = ExecutionSnapshot.from_dict(data)
        
        assert snapshot.variables == {"a": 1}
        assert snapshot.context == {"b": 2}
        assert snapshot.current_step_id == "step-2"
        assert snapshot.error_state == {"error": "test"}


class TestDAGProgress:
    """Tests for DAGProgress value object"""
    
    def test_dag_progress_creation(self):
        """Test creating DAG progress snapshot"""
        progress = DAGProgress(
            completed_steps=["step-1", "step-2"],
            pending_steps=["step-3"],
            failed_steps=[],
            skipped_steps=[],
            total_steps=3,
            completion_percentage=66.67
        )
        assert len(progress.completed_steps) == 2
        assert len(progress.pending_steps) == 1
        assert progress.total_steps == 3
        assert progress.completion_percentage == 66.67
    
    def test_dag_progress_to_dict(self):
        """Test converting DAG progress to dictionary"""
        progress = DAGProgress(
            completed_steps=["step-1"],
            pending_steps=["step-2"],
            failed_steps=[],
            skipped_steps=[],
            total_steps=2,
            completion_percentage=50.0
        )
        data = progress.to_dict()
        
        assert data["completed_steps"] == ["step-1"]
        assert data["total_steps"] == 2
        assert data["completion_percentage"] == 50.0
    
    def test_dag_progress_from_dict(self):
        """Test creating DAG progress from dictionary"""
        data = {
            "completed_steps": ["a", "b"],
            "pending_steps": ["c"],
            "failed_steps": [],
            "skipped_steps": [],
            "total_steps": 3,
            "completion_percentage": 66.67
        }
        progress = DAGProgress.from_dict(data)
        
        assert progress.completed_steps == ["a", "b"]
        assert progress.total_steps == 3


class TestBudgetSnapshot:
    """Tests for BudgetSnapshot value object"""
    
    def test_budget_snapshot_creation(self):
        """Test creating budget snapshot"""
        budget = BudgetSnapshot(
            allocated=100.0,
            consumed=50.0,
            remaining=50.0,
            warning_threshold=0.8,
            critical_threshold=0.95,
            status="active"
        )
        assert budget.allocated == 100.0
        assert budget.consumed == 50.0
        assert budget.remaining == 50.0
    
    def test_budget_snapshot_to_dict(self):
        """Test converting budget snapshot to dictionary"""
        budget = BudgetSnapshot(
            allocated=100.0,
            consumed=25.0,
            remaining=75.0,
            warning_threshold=0.8,
            critical_threshold=0.95,
            status="active"
        )
        data = budget.to_dict()
        
        assert data["allocated"] == 100.0
        assert data["consumed"] == 25.0
        assert data["status"] == "active"


class TestCheckpoint:
    """Tests for Checkpoint entity"""
    
    def test_checkpoint_creation(self):
        """Test creating a checkpoint"""
        workflow_id = uuid4()
        checkpoint = Checkpoint.create(
            workflow_execution_id=workflow_id,
            checkpoint_number=1,
            execution_state=ExecutionSnapshot(
                variables={},
                context={},
                current_step_id="step-1"
            ),
            dag_progress=DAGProgress(
                completed_steps=["step-1"],
                pending_steps=["step-2"],
                failed_steps=[],
                skipped_steps=[],
                total_steps=2,
                completion_percentage=50.0
            ),
            budget_state=BudgetSnapshot(
                allocated=100.0,
                consumed=10.0,
                remaining=90.0,
                warning_threshold=0.8,
                critical_threshold=0.95,
                status="active"
            )
        )
        
        assert checkpoint.workflow_execution_id == workflow_id
        assert checkpoint.checkpoint_number == 1
        assert checkpoint.status == CheckpointStatus.CREATING
    
    def test_checkpoint_mark_as_ready(self):
        """Test marking checkpoint as ready"""
        checkpoint = Checkpoint.create(
            workflow_execution_id=uuid4(),
            checkpoint_number=1,
            execution_state=ExecutionSnapshot(
                variables={},
                context={},
                current_step_id=None
            ),
            dag_progress=DAGProgress(
                completed_steps=[],
                pending_steps=[],
                failed_steps=[],
                skipped_steps=[],
                total_steps=0,
                completion_percentage=0.0
            ),
            budget_state=BudgetSnapshot(
                allocated=100.0,
                consumed=0.0,
                remaining=100.0,
                warning_threshold=0.8,
                critical_threshold=0.95,
                status="active"
            )
        )
        
        checkpoint.mark_as_ready()
        assert checkpoint.status == CheckpointStatus.READY
    
    def test_checkpoint_can_be_restored(self):
        """Test checking if checkpoint can be restored"""
        checkpoint = Checkpoint.create(
            workflow_execution_id=uuid4(),
            checkpoint_number=1,
            execution_state=ExecutionSnapshot(
                variables={},
                context={},
                current_step_id=None
            ),
            dag_progress=DAGProgress(
                completed_steps=[],
                pending_steps=[],
                failed_steps=[],
                skipped_steps=[],
                total_steps=0,
                completion_percentage=0.0
            ),
            budget_state=BudgetSnapshot(
                allocated=100.0,
                consumed=0.0,
                remaining=100.0,
                warning_threshold=0.8,
                critical_threshold=0.95,
                status="active"
            )
        )
        
        # Creating checkpoint cannot be restored
        assert not checkpoint.can_be_restored()
        
        # Ready checkpoint can be restored
        checkpoint.mark_as_ready()
        assert checkpoint.can_be_restored()
    
    def test_checkpoint_get_completion_percentage(self):
        """Test getting DAG completion percentage"""
        checkpoint = Checkpoint.create(
            workflow_execution_id=uuid4(),
            checkpoint_number=1,
            execution_state=ExecutionSnapshot(
                variables={},
                context={},
                current_step_id=None
            ),
            dag_progress=DAGProgress(
                completed_steps=["a", "b", "c"],
                pending_steps=["d"],
                failed_steps=[],
                skipped_steps=[],
                total_steps=4,
                completion_percentage=75.0
            ),
            budget_state=BudgetSnapshot(
                allocated=100.0,
                consumed=0.0,
                remaining=100.0,
                warning_threshold=0.8,
                critical_threshold=0.95,
                status="active"
            )
        )
        
        assert checkpoint.get_completion_percentage() == 75.0
    
    def test_checkpoint_get_budget_usage_percentage(self):
        """Test getting budget usage percentage"""
        checkpoint = Checkpoint.create(
            workflow_execution_id=uuid4(),
            checkpoint_number=1,
            execution_state=ExecutionSnapshot(
                variables={},
                context={},
                current_step_id=None
            ),
            dag_progress=DAGProgress(
                completed_steps=[],
                pending_steps=[],
                failed_steps=[],
                skipped_steps=[],
                total_steps=0,
                completion_percentage=0.0
            ),
            budget_state=BudgetSnapshot(
                allocated=100.0,
                consumed=30.0,
                remaining=70.0,
                warning_threshold=0.8,
                critical_threshold=0.95,
                status="active"
            )
        )
        
        assert checkpoint.get_budget_usage_percentage() == 30.0
    
    def test_checkpoint_invalid_number(self):
        """Test creating checkpoint with invalid number"""
        with pytest.raises(ValueError, match="Checkpoint number must be non-negative"):
            Checkpoint.create(
                workflow_execution_id=uuid4(),
                checkpoint_number=-1,
                execution_state=ExecutionSnapshot(
                    variables={},
                    context={},
                    current_step_id=None
                ),
                dag_progress=DAGProgress(
                    completed_steps=[],
                    pending_steps=[],
                    failed_steps=[],
                    skipped_steps=[],
                    total_steps=0,
                    completion_percentage=0.0
                ),
                budget_state=BudgetSnapshot(
                    allocated=100.0,
                    consumed=0.0,
                    remaining=100.0,
                    warning_threshold=0.8,
                    critical_threshold=0.95,
                    status="active"
                )
            )
    
    def test_checkpoint_to_dict_and_from_dict(self):
        """Test checkpoint serialization and deserialization"""
        workflow_id = uuid4()
        checkpoint = Checkpoint.create(
            workflow_execution_id=workflow_id,
            checkpoint_number=1,
            execution_state=ExecutionSnapshot(
                variables={"x": 1},
                context={"y": 2},
                current_step_id="step-1"
            ),
            dag_progress=DAGProgress(
                completed_steps=["step-1"],
                pending_steps=[],
                failed_steps=[],
                skipped_steps=[],
                total_steps=1,
                completion_percentage=100.0
            ),
            budget_state=BudgetSnapshot(
                allocated=100.0,
                consumed=10.0,
                remaining=90.0,
                warning_threshold=0.8,
                critical_threshold=0.95,
                status="active"
            )
        )
        checkpoint.mark_as_ready()
        
        # Convert to dict
        data = checkpoint.to_dict()
        
        # Convert back to entity
        restored = Checkpoint.from_dict(data)
        
        assert restored.workflow_execution_id == workflow_id
        assert restored.checkpoint_number == 1
        assert restored.status == CheckpointStatus.READY
        assert restored.execution_state.variables == {"x": 1}
        assert restored.dag_progress.total_steps == 1
        assert restored.budget_state.allocated == 100.0
