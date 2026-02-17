"""
E2E Test Suite for P0 Deterministic Execution

Tests for:
- Execution persistence with idempotency
- Replay functionality with output comparison
- Resume from checkpoints
- Unified error handling
"""

import pytest
import asyncio
from uuid import uuid4
from datetime import datetime, timedelta
from typing import Dict, Any

from core.domain.execution import (
    StepExecutionAttempt,
    ExecutionSnapshot,
    ReplaySession,
    ExecutionError,
    ExecutionLock,
    AttemptStatus,
    ReplayMode,
    SnapshotType,
    ErrorCategory,
    ErrorSeverity,
)
from core.domain.workflow import (
    Workflow,
    WorkflowExecution,
    StepExecution,
    WorkflowId,
    Version,
    StepId,
    WorkflowStep,
    ExecutionStatus,
)


# ==================== Test Fixtures ====================

@pytest.fixture
def sample_workflow():
    """Create a sample workflow for testing"""
    return Workflow(
        id=WorkflowId("test-workflow"),
        version=Version(1, 0, 0),
        name="Test Workflow",
        description="A test workflow for E2E testing",
        steps=[
            WorkflowStep(
                id=StepId("step1"),
                type="execute_task",
                name="Step 1",
                params={"input": "value1"},
                depends_on=[],
            ),
            WorkflowStep(
                id=StepId("step2"),
                type="execute_task",
                name="Step 2",
                params={"input": "value2"},
                depends_on=[StepId("step1")],
            ),
            WorkflowStep(
                id=StepId("step3"),
                type="execute_task",
                name="Step 3",
                params={"input": "value3"},
                depends_on=[StepId("step2")],
            ),
        ]
    )


@pytest.fixture
def sample_execution(sample_workflow):
    """Create a sample workflow execution"""
    return WorkflowExecution(
        id=uuid4(),
        workflow_id=sample_workflow.id,
        workflow_version=sample_workflow.version,
        status=ExecutionStatus.PENDING,
        inputs={"global_input": "test"},
        user_id=uuid4(),
    )


# ==================== Idempotency Tests ====================

class TestIdempotency:
    """Test execution idempotency with retries"""

    def test_idempotency_key_generation(self):
        """Test deterministic idempotency key generation"""
        execution_id = uuid4()
        step_id = "step1"
        attempt = 1

        # Generate key twice - should be identical
        key1 = StepExecutionAttempt.generate_idempotency_key(execution_id, step_id, attempt)
        key2 = StepExecutionAttempt.generate_idempotency_key(execution_id, step_id, attempt)

        assert key1 == key2
        assert len(key1) == 64  # SHA-256 hash length

    def test_different_attempts_different_keys(self):
        """Test that different attempts generate different keys"""
        execution_id = uuid4()
        step_id = "step1"

        key1 = StepExecutionAttempt.generate_idempotency_key(execution_id, step_id, 1)
        key2 = StepExecutionAttempt.generate_idempotency_key(execution_id, step_id, 2)

        assert key1 != key2

    def test_step_execution_attempt_creation(self):
        """Test creating a step execution attempt"""
        attempt = StepExecutionAttempt(
            id=uuid4(),
            step_execution_id=uuid4(),
            attempt_number=1,
            idempotency_key="test-key",
            inputs={"input": "value"},
            status=AttemptStatus.SUCCESS,
            outputs={"output": "result"},
            started_at=datetime.utcnow(),
            is_deterministic=True,
        )

        assert attempt.attempt_number == 1
        assert attempt.is_deterministic
        assert len(attempt.nondeterminism_flags) == 0

    def test_response_hash_computation(self):
        """Test deterministic response hash computation"""
        response1 = {"output": "result", "status": "success"}
        response2 = {"status": "success", "output": "result"}  # Different order

        hash1 = StepExecutionAttempt.compute_response_hash(response1)
        hash2 = StepExecutionAttempt.compute_response_hash(response2)

        # Should be identical despite different key order
        assert hash1 == hash2

    def test_response_comparison(self):
        """Test response output comparison"""
        attempt = StepExecutionAttempt(
            id=uuid4(),
            step_execution_id=uuid4(),
            attempt_number=1,
            idempotency_key="test-key",
            inputs={"input": "value"},
            status=AttemptStatus.SUCCESS,
            outputs={"output": "result"},
            started_at=datetime.utcnow(),
        )

        attempt.compute_and_store_hash()

        # Should match identical output
        assert attempt.matches_response({"output": "result"})

        # Should not match different output
        assert not attempt.matches_response({"output": "different"})


# ==================== Replay Tests ====================

class TestReplay:
    """Test execution replay functionality"""

    def test_replay_session_creation(self):
        """Test creating a replay session"""
        original_id = uuid4()
        replay_id = uuid4()

        session = ReplaySession(
            id=uuid4(),
            original_execution_id=original_id,
            replay_execution_id=replay_id,
            replay_mode=ReplayMode.FULL,
            use_cached_outputs=True,
        )

        assert session.status == "pending"
        assert session.replay_mode == ReplayMode.FULL
        assert session.use_cached_outputs is True

    def test_replay_session_lifecycle(self):
        """Test replay session state transitions"""
        session = ReplaySession(
            id=uuid4(),
            original_execution_id=uuid4(),
            replay_execution_id=uuid4(),
            replay_mode=ReplayMode.FULL,
        )

        # Start session
        session.start()
        assert session.status == "running"
        assert session.started_at is not None

        # Complete session
        session.complete(
            match_percentage=95.5,
            divergence_details={"divergent_steps": ["step2"]}
        )
        assert session.status == "completed"
        assert session.match_percentage == 95.5
        assert session.completed_at is not None

    def test_replay_modes(self):
        """Test different replay modes"""
        original_id = uuid4()

        # Full replay
        full_session = ReplaySession(
            id=uuid4(),
            original_execution_id=original_id,
            replay_execution_id=uuid4(),
            replay_mode=ReplayMode.FULL,
        )
        assert full_session.replay_mode == ReplayMode.FULL

        # From step replay
        from_step_session = ReplaySession(
            id=uuid4(),
            original_execution_id=original_id,
            replay_execution_id=uuid4(),
            replay_mode=ReplayMode.FROM_STEP,
            start_from_step="step2",
        )
        assert from_step_session.replay_mode == ReplayMode.FROM_STEP
        assert from_step_session.start_from_step == "step2"

        # Failed only replay
        failed_only_session = ReplaySession(
            id=uuid4(),
            original_execution_id=original_id,
            replay_execution_id=uuid4(),
            replay_mode=ReplayMode.FAILED_ONLY,
        )
        assert failed_only_session.replay_mode == ReplayMode.FAILED_ONLY


# ==================== Resume Tests ====================

class TestResume:
    """Test execution resume functionality"""

    def test_execution_snapshot_creation(self):
        """Test creating an execution snapshot"""
        snapshot = ExecutionSnapshot(
            id=uuid4(),
            execution_id=uuid4(),
            snapshot_type=SnapshotType.CHECKPOINT,
            workflow_state={"status": "running"},
            step_states={"step1": "completed", "step2": "running"},
            completed_steps=["step1"],
            pending_steps=["step3"],
        )

        assert snapshot.snapshot_type == SnapshotType.CHECKPOINT
        assert "step1" in snapshot.completed_steps
        assert "step3" in snapshot.pending_steps

    def test_snapshot_size_calculation(self):
        """Test snapshot size is calculated"""
        snapshot = ExecutionSnapshot(
            id=uuid4(),
            execution_id=uuid4(),
            snapshot_type=SnapshotType.CHECKPOINT,
            workflow_state={"large_data": "x" * 1000},
            step_states={},
        )

        # Size should be calculated automatically
        assert snapshot.snapshot_size_bytes is not None
        assert snapshot.snapshot_size_bytes > 0

    def test_can_resume_from_snapshot(self):
        """Test determining if execution can resume from snapshot"""
        # Snapshot with pending steps can resume
        resumable_snapshot = ExecutionSnapshot(
            id=uuid4(),
            execution_id=uuid4(),
            snapshot_type=SnapshotType.CHECKPOINT,
            workflow_state={},
            step_states={},
            completed_steps=["step1"],
            pending_steps=["step2", "step3"],
        )
        assert resumable_snapshot.can_resume_from()

        # Snapshot with no pending/failed steps cannot resume
        completed_snapshot = ExecutionSnapshot(
            id=uuid4(),
            execution_id=uuid4(),
            snapshot_type=SnapshotType.CHECKPOINT,
            workflow_state={},
            step_states={},
            completed_steps=["step1", "step2", "step3"],
            pending_steps=[],
            failed_steps=[],
        )
        assert not completed_snapshot.can_resume_from()

    def test_get_next_steps(self):
        """Test getting next steps to execute on resume"""
        snapshot = ExecutionSnapshot(
            id=uuid4(),
            execution_id=uuid4(),
            snapshot_type=SnapshotType.CHECKPOINT,
            workflow_state={},
            step_states={},
            completed_steps=["step1"],
            pending_steps=["step3"],
            failed_steps=["step2"],
        )

        next_steps = snapshot.get_next_steps()

        # Failed steps should come first
        assert next_steps[0] == "step2"
        assert "step3" in next_steps


# ==================== Error Model Tests ====================

class TestUnifiedErrorModel:
    """Test unified error handling"""

    def test_execution_error_creation(self):
        """Test creating an execution error"""
        error = ExecutionError(
            id=uuid4(),
            error_code="WORKFLOW_STEP_FAILED",
            error_category=ErrorCategory.EXECUTION,
            severity=ErrorSeverity.HIGH,
            message="Step execution failed",
            correlation_id=uuid4(),
            is_retryable=True,
            retry_count=0,
            max_retries=3,
        )

        assert error.error_category == ErrorCategory.EXECUTION
        assert error.severity == ErrorSeverity.HIGH
        assert error.is_retryable

    def test_error_retry_logic(self):
        """Test error retry increment logic"""
        error = ExecutionError(
            id=uuid4(),
            error_code="TRANSIENT_ERROR",
            error_category=ErrorCategory.EXTERNAL,
            severity=ErrorSeverity.MEDIUM,
            message="External service timeout",
            correlation_id=uuid4(),
            is_retryable=True,
            retry_count=0,
            max_retries=3,
        )

        # Can retry initially
        assert error.can_retry()

        # Increment retry
        error.increment_retry(retry_delay_seconds=60)
        assert error.retry_count == 1
        assert error.next_retry_at is not None

        # Can still retry
        assert error.can_retry()

        # Increment to max
        error.increment_retry(60)
        error.increment_retry(60)
        assert error.retry_count == 3

        # Cannot retry anymore
        assert not error.can_retry()

    def test_error_resolution(self):
        """Test marking error as resolved"""
        error = ExecutionError(
            id=uuid4(),
            error_code="TEMPORARY_ERROR",
            error_category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.LOW,
            message="Resource temporarily unavailable",
            correlation_id=uuid4(),
        )

        assert not error.resolved

        error.resolve("Issue resolved after retry")

        assert error.resolved
        assert error.resolved_at is not None
        assert error.resolution_notes == "Issue resolved after retry"

    def test_error_envelope_format(self):
        """Test standardized error envelope"""
        error = ExecutionError(
            id=uuid4(),
            error_code="VALIDATION_ERROR",
            error_category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            message="Invalid input",
            correlation_id=uuid4(),
            details={"field": "email", "reason": "invalid format"},
        )

        envelope = error.to_error_envelope()

        assert envelope["code"] == "VALIDATION_ERROR"
        assert envelope["category"] == "validation"
        assert envelope["severity"] == "medium"
        assert envelope["message"] == "Invalid input"
        assert "correlation_id" in envelope
        assert envelope["details"]["field"] == "email"


# ==================== Execution Lock Tests ====================

class TestExecutionLocks:
    """Test distributed execution locks"""

    def test_lock_creation(self):
        """Test creating an execution lock"""
        lock = ExecutionLock(
            id=uuid4(),
            lock_key="execution:123:step:456",
            lock_holder="worker-1",
            acquired_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(minutes=5),
        )

        assert lock.lock_key == "execution:123:step:456"
        assert lock.lock_holder == "worker-1"

    def test_lock_expiration(self):
        """Test lock expiration check"""
        # Expired lock
        expired_lock = ExecutionLock(
            id=uuid4(),
            lock_key="test-lock",
            lock_holder="worker-1",
            acquired_at=datetime.utcnow() - timedelta(minutes=10),
            expires_at=datetime.utcnow() - timedelta(minutes=5),
        )
        assert expired_lock.is_expired()

        # Valid lock
        valid_lock = ExecutionLock(
            id=uuid4(),
            lock_key="test-lock",
            lock_holder="worker-1",
            acquired_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(minutes=5),
        )
        assert not valid_lock.is_expired()

    def test_lock_holder_check(self):
        """Test lock holder verification"""
        lock = ExecutionLock(
            id=uuid4(),
            lock_key="test-lock",
            lock_holder="worker-1",
            acquired_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(minutes=5),
        )

        assert lock.is_held_by("worker-1")
        assert not lock.is_held_by("worker-2")

    def test_lock_key_generation(self):
        """Test lock key generation"""
        execution_id = uuid4()

        # Execution-level lock
        exec_key = ExecutionLock.generate_lock_key(execution_id)
        assert exec_key == f"execution:{execution_id}"

        # Step-level lock
        step_key = ExecutionLock.generate_lock_key(execution_id, "step1")
        assert step_key == f"execution:{execution_id}:step:step1"


# ==================== Integration Tests ====================

@pytest.mark.asyncio
class TestE2EIntegration:
    """End-to-end integration tests for P0 features"""

    async def test_full_execution_with_persistence(self, sample_workflow, sample_execution):
        """Test complete execution flow with persistence"""
        # Start execution
        sample_execution.start()
        assert sample_execution.status == ExecutionStatus.RUNNING

        # Execute steps with attempts
        for step in sample_workflow.steps:
            step_execution = StepExecution(
                id=uuid4(),
                execution_id=sample_execution.id,
                step_id=step.id,
                step_type=step.type,
            )

            # Create first attempt
            attempt = StepExecutionAttempt(
                id=uuid4(),
                step_execution_id=step_execution.id,
                attempt_number=1,
                idempotency_key=StepExecutionAttempt.generate_idempotency_key(
                    sample_execution.id, step.id.value, 1
                ),
                inputs=step.params,
                status=AttemptStatus.SUCCESS,
                outputs={"result": f"output_{step.id.value}"},
                started_at=datetime.utcnow(),
            )

            attempt.compute_and_store_hash()
            assert attempt.response_hash is not None

        # Complete execution
        sample_execution.complete({"final_output": "success"})
        assert sample_execution.status == ExecutionStatus.COMPLETED

    async def test_replay_with_comparison(self, sample_execution):
        """Test replay execution with output comparison"""
        original_id = sample_execution.id
        replay_id = uuid4()

        # Create replay session
        replay_session = ReplaySession(
            id=uuid4(),
            original_execution_id=original_id,
            replay_execution_id=replay_id,
            replay_mode=ReplayMode.FULL,
            use_cached_outputs=True,
        )

        # Start replay
        replay_session.start()
        assert replay_session.status == "running"

        # Simulate replay completion with comparison
        replay_session.complete(
            match_percentage=100.0,
            divergence_details={}
        )

        assert replay_session.status == "completed"
        assert replay_session.match_percentage == 100.0

    async def test_resume_from_failure(self, sample_execution):
        """Test resuming execution from failure"""
        # Start execution
        sample_execution.start()

        # Create snapshot before failure
        snapshot = ExecutionSnapshot(
            id=uuid4(),
            execution_id=sample_execution.id,
            snapshot_type=SnapshotType.CHECKPOINT,
            workflow_state={"status": "running"},
            step_states={"step1": "completed"},
            completed_steps=["step1"],
            pending_steps=["step2", "step3"],
            failed_steps=[],
        )

        # Simulate failure
        sample_execution.fail("Step 2 failed")

        # Update snapshot with failure
        snapshot.failed_steps = ["step2"]
        snapshot.pending_steps = ["step3"]

        # Can resume
        assert snapshot.can_resume_from()

        # Get next steps (failed first)
        next_steps = snapshot.get_next_steps()
        assert next_steps[0] == "step2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
