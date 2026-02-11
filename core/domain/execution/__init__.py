"""
Execution Domain - Complete Domain Package

Entities, repositories, services, and events for Execution tracking.
This context tracks execution of workflows, tasks, and agent actions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID
from abc import ABC, abstractmethod


# ==================== Enums ====================

class ExecutionStatus(str, Enum):
    """Execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionType(str, Enum):
    """Type of execution"""
    WORKFLOW = "workflow"
    TASK = "task"
    AGENT_ACTION = "agent_action"
    TOOL = "tool"


# ==================== Entities ====================

@dataclass
class ExecutionTrace:
    """
    Execution trace aggregate root.

    Tracks detailed execution history for observability.
    """
    id: UUID
    execution_type: ExecutionType
    status: ExecutionStatus
    entity_id: UUID  # ID of workflow/task/agent being executed
    user_id: UUID
    parent_trace_id: Optional[UUID] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def start(self) -> None:
        """Start execution"""
        if self.status != ExecutionStatus.PENDING:
            raise ValueError(f"Cannot start execution in {self.status} status")
        self.status = ExecutionStatus.RUNNING
        self.started_at = datetime.utcnow()

    def complete(self, outputs: Dict[str, Any]) -> None:
        """Complete execution successfully"""
        if self.status != ExecutionStatus.RUNNING:
            raise ValueError(f"Cannot complete execution in {self.status} status")
        self.status = ExecutionStatus.COMPLETED
        self.outputs = outputs
        self.completed_at = datetime.utcnow()

    def fail(self, error: str) -> None:
        """Mark execution as failed"""
        if self.status != ExecutionStatus.RUNNING:
            raise ValueError(f"Cannot fail execution in {self.status} status")
        self.status = ExecutionStatus.FAILED
        self.error = error
        self.completed_at = datetime.utcnow()

    def duration_seconds(self) -> Optional[int]:
        """Calculate execution duration"""
        if self.started_at and self.completed_at:
            return int((self.completed_at - self.started_at).total_seconds())
        return None


# ==================== Repository Interfaces ====================

class ExecutionTraceRepository(ABC):
    """Repository interface for ExecutionTrace aggregate"""

    @abstractmethod
    async def save(self, trace: ExecutionTrace) -> None:
        pass

    @abstractmethod
    async def get_by_id(self, trace_id: UUID) -> Optional[ExecutionTrace]:
        pass

    @abstractmethod
    async def get_by_entity(self, entity_id: UUID, execution_type: ExecutionType) -> List[ExecutionTrace]:
        pass

    @abstractmethod
    async def get_children(self, parent_trace_id: UUID) -> List[ExecutionTrace]:
        pass


# ==================== Domain Events ====================

@dataclass(frozen=True)
class DomainEvent:
    occurred_at: datetime
    event_id: UUID


@dataclass(frozen=True)
class ExecutionStarted(DomainEvent):
    trace_id: UUID
    execution_type: ExecutionType
    entity_id: UUID


@dataclass(frozen=True)
class ExecutionCompleted(DomainEvent):
    trace_id: UUID
    execution_type: ExecutionType
    duration_seconds: int


@dataclass(frozen=True)
class ExecutionFailed(DomainEvent):
    trace_id: UUID
    execution_type: ExecutionType
    error: str


# ==================== Exports ====================

__all__ = [
    # Enums
    "ExecutionStatus",
    "ExecutionType",
    # Entities
    "ExecutionTrace",
    # Repositories
    "ExecutionTraceRepository",
    # Events
    "DomainEvent",
    "ExecutionStarted",
    "ExecutionCompleted",
    "ExecutionFailed",
]
