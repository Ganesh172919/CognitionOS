"""
Agent Domain Package

Exports all domain entities, value objects, services, and events
for the Agent bounded context.
"""

# Entities and Value Objects
from .entities import (
    Agent,
    AgentDefinition,
    Tool,
    AgentId,
    Capability,
    BudgetLimits,
    BudgetUsage,
    ModelConfig,
    AgentRole,
    AgentStatus,
    FailureStrategy
)

# Repository Interfaces
from .repositories import (
    AgentRepository,
    AgentDefinitionRepository
)

# Domain Services
from .services import (
    AgentMatcher,
    AgentLoadBalancer,
    AgentHealthMonitor,
    AgentCapabilityRegistry
)

# Domain Events
from .events import (
    DomainEvent,
    AgentCreated,
    AgentAssigned,
    AgentStartedReasoning,
    AgentStartedExecution,
    AgentStartedValidation,
    AgentCompletedTask,
    AgentFailedTask,
    AgentRetrying,
    AgentBecameIdle,
    AgentTerminated,
    AgentBudgetExceeded,
    AgentApproachingBudgetLimit,
    AgentBudgetReset,
    AgentToolExecuted,
    AgentToolFailed,
    AgentPermissionGranted,
    AgentPermissionRevoked,
    AgentDefinitionRegistered,
    AgentDefinitionUpdated,
    AgentDefinitionDeleted,
    AgentStuck,
    AgentRecovered
)

__all__ = [
    # Entities
    "Agent",
    "AgentDefinition",
    "Tool",
    # Value Objects
    "AgentId",
    "Capability",
    "BudgetLimits",
    "BudgetUsage",
    "ModelConfig",
    # Enums
    "AgentRole",
    "AgentStatus",
    "FailureStrategy",
    # Repositories
    "AgentRepository",
    "AgentDefinitionRepository",
    # Services
    "AgentMatcher",
    "AgentLoadBalancer",
    "AgentHealthMonitor",
    "AgentCapabilityRegistry",
    # Events
    "DomainEvent",
    "AgentCreated",
    "AgentAssigned",
    "AgentStartedReasoning",
    "AgentStartedExecution",
    "AgentStartedValidation",
    "AgentCompletedTask",
    "AgentFailedTask",
    "AgentRetrying",
    "AgentBecameIdle",
    "AgentTerminated",
    "AgentBudgetExceeded",
    "AgentApproachingBudgetLimit",
    "AgentBudgetReset",
    "AgentToolExecuted",
    "AgentToolFailed",
    "AgentPermissionGranted",
    "AgentPermissionRevoked",
    "AgentDefinitionRegistered",
    "AgentDefinitionUpdated",
    "AgentDefinitionDeleted",
    "AgentStuck",
    "AgentRecovered",
]
