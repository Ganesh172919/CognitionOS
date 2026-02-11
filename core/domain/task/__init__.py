"""
Task Domain - Complete Domain Package

Entities, repositories, services, and events for Task bounded context.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from abc import ABC, abstractmethod


# ==================== Enums ====================

class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Task priority level"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ==================== Value Objects ====================

@dataclass(frozen=True)
class TaskId:
    """Task identifier value object"""
    value: UUID

    def __str__(self) -> str:
        return str(self.value)


# ==================== Entities ====================

@dataclass
class Task:
    """
    Task aggregate root.

    Represents a unit of work to be executed by an agent.

    Invariants:
    - Task must have a goal
    - Cannot cancel completed tasks
    - Dependencies must form a DAG (no cycles)
    """
    id: TaskId
    user_id: UUID
    session_id: UUID
    goal_id: UUID
    name: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    required_capabilities: List[str] = field(default_factory=list)
    dependencies: List[TaskId] = field(default_factory=list)
    assigned_agent_id: Optional[UUID] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)

    def can_start(self) -> bool:
        """Business rule: Check if task can be started"""
        return self.status in [TaskStatus.PENDING, TaskStatus.ASSIGNED]

    def assign_to_agent(self, agent_id: UUID) -> None:
        """Business rule: Assign task to agent"""
        if self.status != TaskStatus.PENDING:
            raise ValueError(f"Cannot assign task in {self.status} status")
        self.assigned_agent_id = agent_id
        self.status = TaskStatus.ASSIGNED

    def start(self) -> None:
        """Business rule: Start task execution"""
        if not self.can_start():
            raise ValueError(f"Cannot start task in {self.status} status")
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()

    def complete(self, output: Dict[str, Any]) -> None:
        """Business rule: Complete task successfully"""
        if self.status != TaskStatus.IN_PROGRESS:
            raise ValueError(f"Cannot complete task in {self.status} status")
        self.status = TaskStatus.COMPLETED
        self.output_data = output
        self.completed_at = datetime.utcnow()

    def fail(self, error: str) -> None:
        """Business rule: Mark task as failed"""
        if self.status != TaskStatus.IN_PROGRESS:
            raise ValueError(f"Cannot fail task in {self.status} status")
        self.status = TaskStatus.FAILED
        self.error_message = error
        self.completed_at = datetime.utcnow()

    def can_retry(self) -> bool:
        """Business rule: Check if task can be retried"""
        return (
            self.status == TaskStatus.FAILED and
            self.retry_count < self.max_retries
        )

    def retry(self) -> None:
        """Business rule: Retry failed task"""
        if not self.can_retry():
            raise ValueError("Task cannot be retried")
        self.retry_count += 1
        self.status = TaskStatus.PENDING
        self.started_at = None
        self.completed_at = None
        self.error_message = None

    def cancel(self) -> None:
        """Business rule: Cancel task"""
        if self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            raise ValueError(f"Cannot cancel task in {self.status} status")
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.utcnow()

    def duration_seconds(self) -> Optional[int]:
        """Calculate task duration"""
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds())
        return None


# ==================== Repository Interfaces ====================

class TaskRepository(ABC):
    """Repository interface for Task aggregate"""

    @abstractmethod
    async def save(self, task: Task) -> None:
        pass

    @abstractmethod
    async def get_by_id(self, task_id: TaskId) -> Optional[Task]:
        pass

    @abstractmethod
    async def get_by_status(self, status: TaskStatus, limit: int = 100) -> List[Task]:
        pass

    @abstractmethod
    async def get_by_agent(self, agent_id: UUID) -> List[Task]:
        pass

    @abstractmethod
    async def get_pending_tasks(self, limit: int = 100) -> List[Task]:
        pass


# ==================== Domain Events ====================

@dataclass(frozen=True)
class DomainEvent:
    occurred_at: datetime
    event_id: UUID


@dataclass(frozen=True)
class TaskCreated(DomainEvent):
    task_id: TaskId
    user_id: UUID
    name: str


@dataclass(frozen=True)
class TaskAssigned(DomainEvent):
    task_id: TaskId
    agent_id: UUID


@dataclass(frozen=True)
class TaskStarted(DomainEvent):
    task_id: TaskId
    agent_id: UUID


@dataclass(frozen=True)
class TaskCompleted(DomainEvent):
    task_id: TaskId
    duration_seconds: int


@dataclass(frozen=True)
class TaskFailed(DomainEvent):
    task_id: TaskId
    error: str


# ==================== Exports ====================

__all__ = [
    # Enums
    "TaskStatus",
    "TaskPriority",
    # Value Objects
    "TaskId",
    # Entities
    "Task",
    # Repositories
    "TaskRepository",
    # Events
    "DomainEvent",
    "TaskCreated",
    "TaskAssigned",
    "TaskStarted",
    "TaskCompleted",
    "TaskFailed",
]
