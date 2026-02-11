"""
Workflow Domain - Domain Events

Events represent things that have happened in the domain.
These are immutable facts about state changes.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from .entities import WorkflowId, Version, StepId, ExecutionStatus


# ==================== Base Event ====================

@dataclass(frozen=True)
class DomainEvent:
    """
    Base class for all domain events.

    Events are immutable records of things that happened.
    """
    occurred_at: datetime
    event_id: UUID


# ==================== Workflow Events ====================

@dataclass(frozen=True)
class WorkflowCreated(DomainEvent):
    """Event: A new workflow was created"""
    workflow_id: WorkflowId
    version: Version
    name: str
    created_by: Optional[str]


@dataclass(frozen=True)
class WorkflowActivated(DomainEvent):
    """Event: A workflow was activated"""
    workflow_id: WorkflowId
    version: Version


@dataclass(frozen=True)
class WorkflowDeprecated(DomainEvent):
    """Event: A workflow was deprecated"""
    workflow_id: WorkflowId
    version: Version
    reason: Optional[str]


@dataclass(frozen=True)
class WorkflowArchived(DomainEvent):
    """Event: A workflow was archived"""
    workflow_id: WorkflowId
    version: Version


# ==================== Execution Events ====================

@dataclass(frozen=True)
class WorkflowExecutionStarted(DomainEvent):
    """Event: A workflow execution started"""
    execution_id: UUID
    workflow_id: WorkflowId
    workflow_version: Version
    inputs: Dict[str, Any]
    user_id: Optional[UUID]


@dataclass(frozen=True)
class WorkflowExecutionCompleted(DomainEvent):
    """Event: A workflow execution completed successfully"""
    execution_id: UUID
    workflow_id: WorkflowId
    workflow_version: Version
    outputs: Dict[str, Any]
    duration_seconds: int


@dataclass(frozen=True)
class WorkflowExecutionFailed(DomainEvent):
    """Event: A workflow execution failed"""
    execution_id: UUID
    workflow_id: WorkflowId
    workflow_version: Version
    error: str
    failed_step_id: Optional[StepId]


@dataclass(frozen=True)
class WorkflowExecutionCancelled(DomainEvent):
    """Event: A workflow execution was cancelled"""
    execution_id: UUID
    workflow_id: WorkflowId
    workflow_version: Version
    cancelled_by: Optional[UUID]
    reason: Optional[str]


# ==================== Step Execution Events ====================

@dataclass(frozen=True)
class StepExecutionStarted(DomainEvent):
    """Event: A step execution started"""
    step_execution_id: UUID
    execution_id: UUID
    step_id: StepId
    step_type: str
    agent_id: Optional[UUID]


@dataclass(frozen=True)
class StepExecutionCompleted(DomainEvent):
    """Event: A step execution completed successfully"""
    step_execution_id: UUID
    execution_id: UUID
    step_id: StepId
    output: Dict[str, Any]
    duration_seconds: int


@dataclass(frozen=True)
class StepExecutionFailed(DomainEvent):
    """Event: A step execution failed"""
    step_execution_id: UUID
    execution_id: UUID
    step_id: StepId
    error: str
    retry_count: int


@dataclass(frozen=True)
class StepExecutionRetried(DomainEvent):
    """Event: A step execution was retried"""
    step_execution_id: UUID
    execution_id: UUID
    step_id: StepId
    retry_count: int


@dataclass(frozen=True)
class StepExecutionSkipped(DomainEvent):
    """Event: A step execution was skipped"""
    step_execution_id: UUID
    execution_id: UUID
    step_id: StepId
    reason: str


# ==================== Workflow Scheduling Events ====================

@dataclass(frozen=True)
class WorkflowScheduled(DomainEvent):
    """Event: A workflow was scheduled with cron"""
    workflow_id: WorkflowId
    version: Version
    schedule: str  # Cron expression


@dataclass(frozen=True)
class WorkflowScheduleRemoved(DomainEvent):
    """Event: A workflow's schedule was removed"""
    workflow_id: WorkflowId
    version: Version


@dataclass(frozen=True)
class ScheduledWorkflowTriggered(DomainEvent):
    """Event: A scheduled workflow was triggered by cron"""
    workflow_id: WorkflowId
    version: Version
    execution_id: UUID
    schedule: str
