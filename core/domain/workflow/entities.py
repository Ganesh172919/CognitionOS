"""
Workflow Domain - Entities

Pure domain entities with business logic and invariants.
NO external dependencies except Python stdlib.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4


class WorkflowStatus(str, Enum):
    """Workflow definition status"""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ExecutionStatus(str, Enum):
    """Execution status for workflows and steps"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


# ==================== Value Objects ====================

@dataclass(frozen=True)
class WorkflowId:
    """Workflow identifier value object"""
    value: str

    def __post_init__(self):
        if not self.value or len(self.value) == 0:
            raise ValueError("Workflow ID cannot be empty")
        if len(self.value) > 255:
            raise ValueError("Workflow ID cannot exceed 255 characters")


@dataclass(frozen=True)
class Version:
    """Semantic version value object"""
    major: int
    minor: int
    patch: int

    @classmethod
    def from_string(cls, version_str: str) -> "Version":
        """Parse version from string like '1.2.3'"""
        parts = version_str.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}")
        try:
            return cls(
                major=int(parts[0]),
                minor=int(parts[1]),
                patch=int(parts[2])
            )
        except ValueError:
            raise ValueError(f"Invalid version format: {version_str}")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: "Version") -> bool:
        """Compare versions"""
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        return self.patch < other.patch


@dataclass(frozen=True)
class StepId:
    """Step identifier within a workflow"""
    value: str

    def __post_init__(self):
        if not self.value or len(self.value) == 0:
            raise ValueError("Step ID cannot be empty")
        # Step IDs must be valid identifiers (for template references)
        if not self.value.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Step ID must be alphanumeric with underscores/hyphens: {self.value}")


# ==================== Entities ====================

@dataclass
class WorkflowStep:
    """
    Workflow step entity.

    Represents a single step in a workflow DAG.
    Invariants:
    - Step ID must be unique within workflow
    - Dependencies must reference valid step IDs
    - Timeout must be positive
    """
    id: StepId
    type: str  # Step type (execute_task, http_request, etc.)
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[StepId] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 0
    condition: Optional[str] = None
    agent_role: Optional[str] = None

    def __post_init__(self):
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        if self.retry_count < 0:
            raise ValueError("Retry count cannot be negative")

    def has_dependencies(self) -> bool:
        """Check if step has dependencies"""
        return len(self.depends_on) > 0

    def depends_on_step(self, step_id: StepId) -> bool:
        """Check if this step depends on another step"""
        return step_id in self.depends_on

    def add_dependency(self, step_id: StepId) -> None:
        """Add a dependency (idempotent)"""
        if step_id not in self.depends_on:
            self.depends_on.append(step_id)

    def requires_agent(self) -> bool:
        """Check if step requires agent execution"""
        return self.agent_role is not None


@dataclass
class Workflow:
    """
    Workflow aggregate root.

    Represents a complete workflow definition with steps organized as a DAG.
    Invariants:
    - Must have at least one step
    - Step IDs must be unique
    - Dependencies must reference existing steps
    - Must not have circular dependencies
    - Only ACTIVE workflows can be executed
    """
    id: WorkflowId
    version: Version
    name: str
    description: str
    steps: List[WorkflowStep]
    status: WorkflowStatus = WorkflowStatus.DRAFT
    schedule: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None

    def __post_init__(self):
        """Validate workflow invariants on construction"""
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """Validate all workflow invariants"""
        if len(self.steps) == 0:
            raise ValueError("Workflow must have at least one step")

        # Check step ID uniqueness
        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Workflow step IDs must be unique")

    def can_execute(self) -> bool:
        """
        Business rule: Check if workflow can be executed.
        Only ACTIVE workflows with valid DAG can be executed.
        """
        if self.status != WorkflowStatus.ACTIVE:
            return False
        if len(self.steps) == 0:
            return False
        return self.has_valid_dag()

    def has_valid_dag(self) -> bool:
        """
        Domain logic: Verify workflow has a valid DAG (no cycles).
        Uses topological sort algorithm.
        """
        try:
            _ = self.get_execution_order()
            return True
        except ValueError:
            return False

    def get_execution_order(self) -> List[StepId]:
        """
        Domain logic: Get topological order of steps for execution.
        Raises ValueError if DAG has cycles.
        """
        # Build adjacency list and in-degree map
        step_map = {step.id: step for step in self.steps}
        in_degree = {step.id: 0 for step in self.steps}
        adjacency = {step.id: [] for step in self.steps}

        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_map:
                    raise ValueError(f"Step {step.id.value} depends on non-existent step {dep.value}")
                adjacency[dep].append(step.id)
                in_degree[step.id] += 1

        # Kahn's algorithm for topological sort
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.steps):
            raise ValueError("Workflow contains circular dependencies")

        return result

    def get_step_by_id(self, step_id: StepId) -> Optional[WorkflowStep]:
        """Find step by ID"""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_independent_steps(self) -> List[WorkflowStep]:
        """
        Domain logic: Get steps that have no dependencies.
        These can start execution immediately.
        """
        return [step for step in self.steps if not step.has_dependencies()]

    def get_dependent_steps(self, step_id: StepId) -> List[WorkflowStep]:
        """
        Domain logic: Get all steps that depend on the given step.
        """
        return [step for step in self.steps if step.depends_on_step(step_id)]

    def activate(self) -> None:
        """
        Business rule: Activate workflow.
        Can only activate DRAFT workflows with valid DAG.
        """
        if self.status != WorkflowStatus.DRAFT:
            raise ValueError(f"Cannot activate workflow in {self.status} status")
        if not self.has_valid_dag():
            raise ValueError("Cannot activate workflow with invalid DAG")
        self.status = WorkflowStatus.ACTIVE

    def deprecate(self) -> None:
        """
        Business rule: Deprecate active workflow.
        Deprecated workflows cannot be executed but existing executions continue.
        """
        if self.status != WorkflowStatus.ACTIVE:
            raise ValueError(f"Cannot deprecate workflow in {self.status} status")
        self.status = WorkflowStatus.DEPRECATED

    def archive(self) -> None:
        """
        Business rule: Archive workflow.
        Can archive DEPRECATED or DRAFT workflows.
        """
        if self.status == WorkflowStatus.ARCHIVED:
            raise ValueError("Workflow is already archived")
        self.status = WorkflowStatus.ARCHIVED

    def is_scheduled(self) -> bool:
        """Check if workflow has a schedule (cron)"""
        return self.schedule is not None

    def add_tag(self, tag: str) -> None:
        """Add tag to workflow (idempotent)"""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove tag from workflow"""
        if tag in self.tags:
            self.tags.remove(tag)


@dataclass
class WorkflowExecution:
    """
    Workflow execution entity (aggregate root).

    Represents a single execution instance of a workflow.
    Invariants:
    - Can only transition to valid next states
    - Cannot restart completed/failed/cancelled executions
    - Started time must be before completed time
    """
    id: UUID
    workflow_id: WorkflowId
    workflow_version: Version
    status: ExecutionStatus
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    user_id: Optional[UUID] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate execution invariants"""
        if self.started_at and self.completed_at:
            if self.completed_at < self.started_at:
                raise ValueError("Completion time cannot be before start time")

    def can_start(self) -> bool:
        """Business rule: Check if execution can be started"""
        return self.status == ExecutionStatus.PENDING

    def can_cancel(self) -> bool:
        """Business rule: Check if execution can be cancelled"""
        return self.status in [ExecutionStatus.PENDING, ExecutionStatus.RUNNING]

    def start(self) -> None:
        """
        Business rule: Start execution.
        Can only start PENDING executions.
        """
        if not self.can_start():
            raise ValueError(f"Cannot start execution in {self.status} status")
        self.status = ExecutionStatus.RUNNING
        self.started_at = datetime.utcnow()

    def complete(self, outputs: Dict[str, Any]) -> None:
        """
        Business rule: Complete execution successfully.
        Can only complete RUNNING executions.
        """
        if self.status != ExecutionStatus.RUNNING:
            raise ValueError(f"Cannot complete execution in {self.status} status")
        self.status = ExecutionStatus.COMPLETED
        self.outputs = outputs
        self.completed_at = datetime.utcnow()

    def fail(self, error: str) -> None:
        """
        Business rule: Mark execution as failed.
        Can only fail RUNNING executions.
        """
        if self.status != ExecutionStatus.RUNNING:
            raise ValueError(f"Cannot fail execution in {self.status} status")
        self.status = ExecutionStatus.FAILED
        self.error = error
        self.completed_at = datetime.utcnow()

    def cancel(self) -> None:
        """
        Business rule: Cancel execution.
        Can only cancel PENDING or RUNNING executions.
        """
        if not self.can_cancel():
            raise ValueError(f"Cannot cancel execution in {self.status} status")
        self.status = ExecutionStatus.CANCELLED
        self.completed_at = datetime.utcnow()

    def is_terminal(self) -> bool:
        """Check if execution is in terminal state"""
        return self.status in [
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED
        ]

    def is_running(self) -> bool:
        """Check if execution is currently running"""
        return self.status == ExecutionStatus.RUNNING

    def duration_seconds(self) -> Optional[int]:
        """Calculate execution duration in seconds"""
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds())
        return None


@dataclass
class StepExecution:
    """
    Step execution entity.

    Represents execution of a single workflow step.
    Part of WorkflowExecution aggregate.
    """
    id: UUID
    execution_id: UUID  # Parent workflow execution
    step_id: StepId
    step_type: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    agent_id: Optional[UUID] = None
    retry_count: int = 0

    def can_start(self) -> bool:
        """Check if step execution can be started"""
        return self.status == ExecutionStatus.PENDING

    def can_retry(self, max_retries: int) -> bool:
        """Check if step can be retried"""
        return (
            self.status == ExecutionStatus.FAILED and
            self.retry_count < max_retries
        )

    def start(self, agent_id: Optional[UUID] = None) -> None:
        """Start step execution"""
        if not self.can_start():
            raise ValueError(f"Cannot start step in {self.status} status")
        self.status = ExecutionStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.agent_id = agent_id

    def complete(self, output: Dict[str, Any]) -> None:
        """Complete step execution successfully"""
        if self.status != ExecutionStatus.RUNNING:
            raise ValueError(f"Cannot complete step in {self.status} status")
        self.status = ExecutionStatus.COMPLETED
        self.output = output
        self.completed_at = datetime.utcnow()

    def fail(self, error: str) -> None:
        """Mark step execution as failed"""
        if self.status != ExecutionStatus.RUNNING:
            raise ValueError(f"Cannot fail step in {self.status} status")
        self.status = ExecutionStatus.FAILED
        self.error = error
        self.completed_at = datetime.utcnow()

    def retry(self) -> None:
        """Retry failed step"""
        if self.status != ExecutionStatus.FAILED:
            raise ValueError("Can only retry failed steps")
        self.retry_count += 1
        self.status = ExecutionStatus.PENDING
        self.started_at = None
        self.completed_at = None
        self.error = None

    def skip(self) -> None:
        """Skip step (conditional execution)"""
        if self.status != ExecutionStatus.PENDING:
            raise ValueError("Can only skip pending steps")
        self.status = ExecutionStatus.SKIPPED
        self.completed_at = datetime.utcnow()

    def duration_seconds(self) -> Optional[int]:
        """Calculate step duration in seconds"""
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds())
        return None
