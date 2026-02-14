"""
Checkpoint Domain - Entities

Pure domain entities for checkpoint/resume functionality.
NO external dependencies except Python stdlib.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class CheckpointStatus(str, Enum):
    """Checkpoint status"""
    CREATING = "creating"
    READY = "ready"
    RESTORING = "restoring"
    FAILED = "failed"
    DELETED = "deleted"


# ==================== Value Objects ====================

@dataclass(frozen=True)
class CheckpointId:
    """Checkpoint identifier value object"""
    value: UUID

    @classmethod
    def generate(cls) -> "CheckpointId":
        """Generate a new checkpoint ID"""
        return cls(value=uuid4())


@dataclass(frozen=True)
class ExecutionSnapshot:
    """Snapshot of execution state at checkpoint time"""
    variables: Dict[str, Any]
    context: Dict[str, Any]
    current_step_id: Optional[str]
    error_state: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "variables": self.variables,
            "context": self.context,
            "current_step_id": self.current_step_id,
            "error_state": self.error_state,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionSnapshot":
        """Create from dictionary"""
        return cls(
            variables=data.get("variables", {}),
            context=data.get("context", {}),
            current_step_id=data.get("current_step_id"),
            error_state=data.get("error_state"),
        )


@dataclass(frozen=True)
class DAGProgress:
    """DAG execution progress snapshot"""
    completed_steps: List[str]
    pending_steps: List[str]
    failed_steps: List[str]
    skipped_steps: List[str]
    total_steps: int
    completion_percentage: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "completed_steps": self.completed_steps,
            "pending_steps": self.pending_steps,
            "failed_steps": self.failed_steps,
            "skipped_steps": self.skipped_steps,
            "total_steps": self.total_steps,
            "completion_percentage": self.completion_percentage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DAGProgress":
        """Create from dictionary"""
        return cls(
            completed_steps=data.get("completed_steps", []),
            pending_steps=data.get("pending_steps", []),
            failed_steps=data.get("failed_steps", []),
            skipped_steps=data.get("skipped_steps", []),
            total_steps=data.get("total_steps", 0),
            completion_percentage=data.get("completion_percentage", 0.0),
        )


@dataclass(frozen=True)
class BudgetSnapshot:
    """Budget state snapshot"""
    allocated: float
    consumed: float
    remaining: float
    warning_threshold: float
    critical_threshold: float
    status: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "allocated": self.allocated,
            "consumed": self.consumed,
            "remaining": self.remaining,
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BudgetSnapshot":
        """Create from dictionary"""
        return cls(
            allocated=data.get("allocated", 0.0),
            consumed=data.get("consumed", 0.0),
            remaining=data.get("remaining", 0.0),
            warning_threshold=data.get("warning_threshold", 0.8),
            critical_threshold=data.get("critical_threshold", 0.95),
            status=data.get("status", "active"),
        )


# ==================== Entities ====================

@dataclass
class Checkpoint:
    """
    Checkpoint entity for workflow state persistence.
    
    Enables 24+ hour workflows through checkpoint/resume.
    Design principles:
    - Idempotent state reconstruction
    - Minimal overhead (<100ms)
    - Snapshot + delta strategy
    """
    id: CheckpointId
    workflow_execution_id: UUID
    checkpoint_number: int
    status: CheckpointStatus
    
    # State snapshots
    execution_state: ExecutionSnapshot
    dag_progress: DAGProgress
    budget_state: BudgetSnapshot
    
    # Memory and tasks
    memory_snapshot_ref: Optional[str] = None
    active_tasks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    checkpoint_size_bytes: Optional[int] = None
    compression_enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    restored_at: Optional[datetime] = None

    def __post_init__(self):
        """Validate checkpoint invariants"""
        if self.checkpoint_number < 0:
            raise ValueError("Checkpoint number must be non-negative")
        
        if self.dag_progress.completion_percentage < 0 or self.dag_progress.completion_percentage > 100:
            raise ValueError("DAG completion percentage must be between 0 and 100")
        
        if self.budget_state.consumed < 0:
            raise ValueError("Budget consumed cannot be negative")

    def mark_as_ready(self) -> None:
        """Mark checkpoint as ready for use"""
        if self.status != CheckpointStatus.CREATING:
            raise ValueError(f"Cannot mark checkpoint as ready from status: {self.status}")
        self.status = CheckpointStatus.READY

    def mark_as_restoring(self) -> None:
        """Mark checkpoint as being restored"""
        if self.status != CheckpointStatus.READY:
            raise ValueError(f"Cannot restore checkpoint from status: {self.status}")
        self.status = CheckpointStatus.RESTORING

    def mark_as_restored(self) -> None:
        """Mark checkpoint as successfully restored"""
        if self.status != CheckpointStatus.RESTORING:
            raise ValueError(f"Cannot mark checkpoint as restored from status: {self.status}")
        self.status = CheckpointStatus.READY
        self.restored_at = datetime.utcnow()

    def mark_as_failed(self, error: str) -> None:
        """Mark checkpoint as failed"""
        self.status = CheckpointStatus.FAILED
        self.metadata["error"] = error
        self.metadata["failed_at"] = datetime.utcnow().isoformat()

    def can_be_restored(self) -> bool:
        """Check if checkpoint can be restored"""
        return self.status == CheckpointStatus.READY

    def get_completion_percentage(self) -> float:
        """Get DAG completion percentage"""
        return self.dag_progress.completion_percentage

    def get_budget_usage_percentage(self) -> float:
        """Get budget usage percentage"""
        if self.budget_state.allocated == 0:
            return 0.0
        return (self.budget_state.consumed / self.budget_state.allocated) * 100

    def is_compressed(self) -> bool:
        """Check if checkpoint is compressed"""
        return self.compression_enabled

    def get_memory_footprint_estimate(self) -> int:
        """Estimate memory footprint in bytes"""
        if self.checkpoint_size_bytes:
            return self.checkpoint_size_bytes
        
        # Rough estimate based on components
        estimate = 0
        estimate += len(str(self.execution_state.to_dict()))
        estimate += len(str(self.dag_progress.to_dict()))
        estimate += len(str(self.budget_state.to_dict()))
        estimate += len(str(self.active_tasks)) if self.active_tasks else 0
        
        return estimate

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary for persistence"""
        return {
            "id": str(self.id.value),
            "workflow_execution_id": str(self.workflow_execution_id),
            "checkpoint_number": self.checkpoint_number,
            "status": self.status.value,
            "execution_state": self.execution_state.to_dict(),
            "dag_progress": self.dag_progress.to_dict(),
            "budget_state": self.budget_state.to_dict(),
            "memory_snapshot_ref": self.memory_snapshot_ref,
            "active_tasks": self.active_tasks,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "compression_enabled": self.compression_enabled,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "restored_at": self.restored_at.isoformat() if self.restored_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Create checkpoint from dictionary"""
        return cls(
            id=CheckpointId(value=UUID(data["id"])),
            workflow_execution_id=UUID(data["workflow_execution_id"]),
            checkpoint_number=data["checkpoint_number"],
            status=CheckpointStatus(data["status"]),
            execution_state=ExecutionSnapshot.from_dict(data["execution_state"]),
            dag_progress=DAGProgress.from_dict(data["dag_progress"]),
            budget_state=BudgetSnapshot.from_dict(data["budget_state"]),
            memory_snapshot_ref=data.get("memory_snapshot_ref"),
            active_tasks=data.get("active_tasks", []),
            checkpoint_size_bytes=data.get("checkpoint_size_bytes"),
            compression_enabled=data.get("compression_enabled", True),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            restored_at=datetime.fromisoformat(data["restored_at"]) if data.get("restored_at") else None,
        )

    @classmethod
    def create(
        cls,
        workflow_execution_id: UUID,
        checkpoint_number: int,
        execution_state: ExecutionSnapshot,
        dag_progress: DAGProgress,
        budget_state: BudgetSnapshot,
        memory_snapshot_ref: Optional[str] = None,
        active_tasks: Optional[List[Dict[str, Any]]] = None,
        compression_enabled: bool = True,
    ) -> "Checkpoint":
        """Factory method to create a new checkpoint"""
        return cls(
            id=CheckpointId.generate(),
            workflow_execution_id=workflow_execution_id,
            checkpoint_number=checkpoint_number,
            status=CheckpointStatus.CREATING,
            execution_state=execution_state,
            dag_progress=dag_progress,
            budget_state=budget_state,
            memory_snapshot_ref=memory_snapshot_ref,
            active_tasks=active_tasks or [],
            compression_enabled=compression_enabled,
        )
