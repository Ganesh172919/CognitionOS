"""
Agent Domain - Domain Events

Events representing state changes in the Agent domain.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from .entities import AgentId, AgentRole, AgentStatus


# ==================== Base Event ====================

@dataclass(frozen=True)
class DomainEvent:
    """Base class for all domain events"""
    occurred_at: datetime
    event_id: UUID


# ==================== Agent Instance Events ====================

@dataclass(frozen=True)
class AgentCreated(DomainEvent):
    """Event: A new agent instance was created"""
    agent_id: AgentId
    definition_id: UUID
    role: AgentRole
    capabilities: List[str]


@dataclass(frozen=True)
class AgentAssigned(DomainEvent):
    """Event: An agent was assigned to a task"""
    agent_id: AgentId
    task_id: UUID
    required_capabilities: List[str]


@dataclass(frozen=True)
class AgentStartedReasoning(DomainEvent):
    """Event: An agent started reasoning phase"""
    agent_id: AgentId
    task_id: UUID


@dataclass(frozen=True)
class AgentStartedExecution(DomainEvent):
    """Event: An agent started execution phase"""
    agent_id: AgentId
    task_id: UUID


@dataclass(frozen=True)
class AgentStartedValidation(DomainEvent):
    """Event: An agent started validation phase"""
    agent_id: AgentId
    task_id: UUID


@dataclass(frozen=True)
class AgentCompletedTask(DomainEvent):
    """Event: An agent completed a task"""
    agent_id: AgentId
    task_id: UUID
    duration_seconds: int


@dataclass(frozen=True)
class AgentFailedTask(DomainEvent):
    """Event: An agent failed a task"""
    agent_id: AgentId
    task_id: UUID
    error: str
    failure_strategy: str


@dataclass(frozen=True)
class AgentRetrying(DomainEvent):
    """Event: An agent is retrying a failed task"""
    agent_id: AgentId
    task_id: UUID
    retry_count: int


@dataclass(frozen=True)
class AgentBecameIdle(DomainEvent):
    """Event: An agent became idle"""
    agent_id: AgentId
    previous_status: AgentStatus


@dataclass(frozen=True)
class AgentTerminated(DomainEvent):
    """Event: An agent was terminated"""
    agent_id: AgentId
    reason: Optional[str]


# ==================== Budget Events ====================

@dataclass(frozen=True)
class AgentBudgetExceeded(DomainEvent):
    """Event: An agent exceeded its budget limits"""
    agent_id: AgentId
    task_id: Optional[UUID]
    tokens_used: int
    cost_usd: float
    time_seconds: float


@dataclass(frozen=True)
class AgentApproachingBudgetLimit(DomainEvent):
    """Event: An agent is approaching its budget limits"""
    agent_id: AgentId
    task_id: Optional[UUID]
    budget_percentage_used: float


@dataclass(frozen=True)
class AgentBudgetReset(DomainEvent):
    """Event: An agent's budget was reset"""
    agent_id: AgentId


# ==================== Tool Events ====================

@dataclass(frozen=True)
class AgentToolExecuted(DomainEvent):
    """Event: An agent executed a tool"""
    agent_id: AgentId
    task_id: UUID
    tool_name: str
    duration_seconds: float
    success: bool


@dataclass(frozen=True)
class AgentToolFailed(DomainEvent):
    """Event: An agent's tool execution failed"""
    agent_id: AgentId
    task_id: UUID
    tool_name: str
    error: str


# ==================== Permission Events ====================

@dataclass(frozen=True)
class AgentPermissionGranted(DomainEvent):
    """Event: A permission was granted to an agent"""
    agent_id: AgentId
    permission: str
    granted_by: Optional[UUID]


@dataclass(frozen=True)
class AgentPermissionRevoked(DomainEvent):
    """Event: A permission was revoked from an agent"""
    agent_id: AgentId
    permission: str
    revoked_by: Optional[UUID]


# ==================== Agent Definition Events ====================

@dataclass(frozen=True)
class AgentDefinitionRegistered(DomainEvent):
    """Event: A new agent definition was registered"""
    definition_id: UUID
    name: str
    role: AgentRole
    version: str
    capabilities: List[str]


@dataclass(frozen=True)
class AgentDefinitionUpdated(DomainEvent):
    """Event: An agent definition was updated"""
    definition_id: UUID
    name: str
    old_version: str
    new_version: str


@dataclass(frozen=True)
class AgentDefinitionDeleted(DomainEvent):
    """Event: An agent definition was deleted"""
    definition_id: UUID
    name: str
    version: str


# ==================== Health Events ====================

@dataclass(frozen=True)
class AgentStuck(DomainEvent):
    """Event: An agent appears to be stuck"""
    agent_id: AgentId
    task_id: UUID
    status: AgentStatus
    time_in_state_seconds: int


@dataclass(frozen=True)
class AgentRecovered(DomainEvent):
    """Event: A stuck agent recovered"""
    agent_id: AgentId
    task_id: UUID
