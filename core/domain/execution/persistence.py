"""
Execution Persistence Domain Entities (P0)

New entities for deterministic execution, replay, and resume capabilities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
import hashlib
import json


# ==================== Enums ====================

class AttemptStatus(str, Enum):
    """Status of a step execution attempt"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ReplayMode(str, Enum):
    """Replay execution mode"""
    FULL = "full"  # Replay entire workflow
    FROM_STEP = "from_step"  # Replay from specific step
    FAILED_ONLY = "failed_only"  # Replay only failed steps


class SnapshotType(str, Enum):
    """Type of execution snapshot"""
    CHECKPOINT = "checkpoint"  # Periodic checkpoint
    BEFORE_STEP = "before_step"  # Before step execution
    AFTER_STEP = "after_step"  # After step execution


class ErrorCategory(str, Enum):
    """Error category for unified error model"""
    VALIDATION = "validation"
    EXECUTION = "execution"
    EXTERNAL = "external"
    TIMEOUT = "timeout"
    RESOURCE = "resource"


class ErrorSeverity(str, Enum):
    """Error severity level"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ==================== Entities ====================

@dataclass
class StepExecutionAttempt:
    """
    Step execution attempt with idempotency tracking.

    Tracks every attempt to execute a step, enabling:
    - Idempotent retries
    - Deterministic replay
    - Output comparison
    """
    id: UUID
    step_execution_id: UUID
    attempt_number: int
    idempotency_key: str
    inputs: Dict[str, Any]
    status: AttemptStatus
    started_at: datetime
    agent_id: Optional[UUID] = None
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    is_deterministic: bool = True
    nondeterminism_flags: List[str] = field(default_factory=list)
    request_payload: Optional[Dict[str, Any]] = None
    response_payload: Optional[Dict[str, Any]] = None
    response_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    @staticmethod
    def generate_idempotency_key(execution_id: UUID, step_id: str, attempt: int) -> str:
        """Generate deterministic idempotency key"""
        key_data = f"{execution_id}:{step_id}:{attempt}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    @staticmethod
    def compute_response_hash(response: Dict[str, Any]) -> str:
        """Compute deterministic hash of response for comparison"""
        # Sort keys to ensure deterministic JSON serialization
        json_str = json.dumps(response, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def mark_nondeterministic(self, flags: List[str]) -> None:
        """Mark attempt as nondeterministic with reasons"""
        self.is_deterministic = False
        self.nondeterminism_flags = flags

    def compute_and_store_hash(self) -> None:
        """Compute and store response hash for replay comparison"""
        if self.outputs:
            self.response_hash = self.compute_response_hash(self.outputs)

    def matches_response(self, other_outputs: Dict[str, Any]) -> bool:
        """Check if outputs match another execution"""
        if not self.response_hash or not other_outputs:
            return False
        other_hash = self.compute_response_hash(other_outputs)
        return self.response_hash == other_hash


@dataclass
class ExecutionSnapshot:
    """
    Execution state snapshot for fast resume.

    Captures complete execution state at a point in time.
    """
    id: UUID
    execution_id: UUID
    snapshot_type: SnapshotType
    workflow_state: Dict[str, Any]
    step_states: Dict[str, Any]
    variables: Dict[str, Any] = field(default_factory=dict)
    completed_steps: List[str] = field(default_factory=list)
    pending_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    snapshot_size_bytes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate snapshot size"""
        if self.snapshot_size_bytes is None:
            # Estimate size from JSON serialization
            data = {
                "workflow_state": self.workflow_state,
                "step_states": self.step_states,
                "variables": self.variables,
            }
            self.snapshot_size_bytes = len(json.dumps(data).encode())

    def can_resume_from(self) -> bool:
        """Check if execution can be resumed from this snapshot"""
        return len(self.pending_steps) > 0 or len(self.failed_steps) > 0

    def get_next_steps(self) -> List[str]:
        """Get list of steps to execute on resume"""
        # Resume from failed steps first, then pending
        return self.failed_steps + self.pending_steps


@dataclass
class ReplaySession:
    """
    Replay session for deterministic execution verification.

    Tracks replay of a previous execution and comparison results.
    """
    id: UUID
    original_execution_id: UUID
    replay_execution_id: UUID
    replay_mode: ReplayMode
    status: str = "pending"
    start_from_step: Optional[str] = None
    use_cached_outputs: bool = True
    match_percentage: Optional[float] = None
    divergence_details: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    triggered_by: Optional[UUID] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def start(self) -> None:
        """Start replay session"""
        if self.status != "pending":
            raise ValueError(f"Cannot start replay in {self.status} status")
        self.status = "running"
        self.started_at = datetime.utcnow()

    def complete(self, match_percentage: float, divergence_details: Dict[str, Any]) -> None:
        """Complete replay with comparison results"""
        if self.status != "running":
            raise ValueError(f"Cannot complete replay in {self.status} status")
        self.status = "completed"
        self.match_percentage = match_percentage
        self.divergence_details = divergence_details
        self.completed_at = datetime.utcnow()

    def fail(self, error: str) -> None:
        """Mark replay as failed"""
        if self.status != "running":
            raise ValueError(f"Cannot fail replay in {self.status} status")
        self.status = "failed"
        self.divergence_details = {"error": error}
        self.completed_at = datetime.utcnow()


@dataclass
class ExecutionError:
    """
    Unified error model for all execution errors.

    Standardizes error representation across services.
    """
    id: UUID
    error_code: str
    error_category: ErrorCategory
    severity: ErrorSeverity
    message: str
    correlation_id: UUID
    is_retryable: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    service_name: Optional[str] = None
    execution_id: Optional[UUID] = None
    step_execution_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    retry_count: int = 0
    max_retries: int = 3
    next_retry_at: Optional[datetime] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def can_retry(self) -> bool:
        """Check if error can be retried"""
        return self.is_retryable and self.retry_count < self.max_retries

    def increment_retry(self, retry_delay_seconds: int = 60) -> None:
        """Increment retry count and set next retry time"""
        if not self.can_retry():
            raise ValueError("Cannot retry this error")
        self.retry_count += 1
        self.next_retry_at = datetime.utcnow().replace(microsecond=0)
        # Add exponential backoff
        import datetime as dt
        self.next_retry_at += dt.timedelta(seconds=retry_delay_seconds * (2 ** (self.retry_count - 1)))
        self.updated_at = datetime.utcnow()

    def resolve(self, notes: str) -> None:
        """Mark error as resolved"""
        self.resolved = True
        self.resolved_at = datetime.utcnow()
        self.resolution_notes = notes
        self.updated_at = datetime.utcnow()

    def to_error_envelope(self) -> Dict[str, Any]:
        """Convert to standardized error envelope"""
        return {
            "code": self.error_code,
            "message": self.message,
            "category": self.error_category.value,
            "severity": self.severity.value,
            "retryable": self.is_retryable,
            "correlation_id": str(self.correlation_id),
            "details": self.details,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionLock:
    """
    Distributed execution lock.

    Prevents concurrent execution of same workflow/step.
    """
    id: UUID
    lock_key: str
    lock_holder: str
    acquired_at: datetime
    expires_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if lock has expired"""
        return datetime.utcnow() > self.expires_at

    def is_held_by(self, holder: str) -> bool:
        """Check if lock is held by specific holder"""
        return self.lock_holder == holder and not self.is_expired()

    @staticmethod
    def generate_lock_key(execution_id: UUID, step_id: Optional[str] = None) -> str:
        """Generate lock key for execution or step"""
        if step_id:
            return f"execution:{execution_id}:step:{step_id}"
        return f"execution:{execution_id}"


# ==================== Exports ====================

__all__ = [
    # Enums
    "AttemptStatus",
    "ReplayMode",
    "SnapshotType",
    "ErrorCategory",
    "ErrorSeverity",
    # Entities
    "StepExecutionAttempt",
    "ExecutionSnapshot",
    "ReplaySession",
    "ExecutionError",
    "ExecutionLock",
]
