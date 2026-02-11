"""
Agent Domain - Entities

Pure domain entities for Agent bounded context.
NO external dependencies except Python stdlib.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


# ==================== Enums ====================

class AgentRole(str, Enum):
    """Agent role types"""
    PLANNER = "planner"
    REASONER = "reasoner"
    EXECUTOR = "executor"
    CRITIC = "critic"
    SUMMARIZER = "summarizer"


class AgentStatus(str, Enum):
    """Agent instance status"""
    CREATED = "created"
    IDLE = "idle"
    ASSIGNED = "assigned"
    REASONING = "reasoning"
    EXECUTING = "executing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    RETRY = "retry"
    TERMINATED = "terminated"
    FAILED = "failed"


class FailureStrategy(str, Enum):
    """Agent failure handling strategy"""
    RETRY = "retry"
    ESCALATE = "escalate"
    ABORT = "abort"
    FALLBACK = "fallback"


# ==================== Value Objects ====================

@dataclass(frozen=True)
class AgentId:
    """Agent identifier value object"""
    value: UUID

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Capability:
    """Agent capability value object"""
    name: str
    version: str = "1.0.0"

    def __post_init__(self):
        if not self.name or len(self.name) == 0:
            raise ValueError("Capability name cannot be empty")

    def matches(self, required: str) -> bool:
        """Check if this capability matches a required capability"""
        # Simple string matching for now
        # Can be extended to support semantic matching
        return self.name == required


@dataclass(frozen=True)
class BudgetLimits:
    """Resource budget limits for agent execution"""
    max_tokens: int = 32000
    max_cost_usd: float = 1.0
    max_time_seconds: int = 300
    max_tool_executions: int = 10

    def __post_init__(self):
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")
        if self.max_cost_usd <= 0:
            raise ValueError("Max cost must be positive")
        if self.max_time_seconds <= 0:
            raise ValueError("Max time must be positive")
        if self.max_tool_executions <= 0:
            raise ValueError("Max tool executions must be positive")


@dataclass
class BudgetUsage:
    """Tracks budget consumption (mutable)"""
    tokens_used: int = 0
    cost_usd: float = 0.0
    time_seconds: float = 0.0
    tool_executions: int = 0

    def within_limits(self, limits: BudgetLimits) -> bool:
        """Check if usage is within budget limits"""
        return (
            self.tokens_used <= limits.max_tokens and
            self.cost_usd <= limits.max_cost_usd and
            self.time_seconds <= limits.max_time_seconds and
            self.tool_executions <= limits.max_tool_executions
        )

    def is_approaching_limit(self, limits: BudgetLimits, threshold: float = 0.8) -> bool:
        """Check if any resource is approaching limit (80% by default)"""
        return (
            self.tokens_used >= limits.max_tokens * threshold or
            self.cost_usd >= limits.max_cost_usd * threshold or
            self.time_seconds >= limits.max_time_seconds * threshold or
            self.tool_executions >= limits.max_tool_executions * threshold
        )

    def add_token_usage(self, tokens: int, cost: float) -> None:
        """Record token usage"""
        if tokens < 0 or cost < 0:
            raise ValueError("Token usage and cost must be non-negative")
        self.tokens_used += tokens
        self.cost_usd += cost

    def add_tool_execution(self, duration_seconds: float) -> None:
        """Record tool execution"""
        if duration_seconds < 0:
            raise ValueError("Duration must be non-negative")
        self.tool_executions += 1
        self.time_seconds += duration_seconds

    def reset(self) -> None:
        """Reset all usage counters"""
        self.tokens_used = 0
        self.cost_usd = 0.0
        self.time_seconds = 0.0
        self.tool_executions = 0


@dataclass(frozen=True)
class ModelConfig:
    """LLM model configuration"""
    provider: str  # "openai", "anthropic", "local"
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0

    def __post_init__(self):
        if not self.provider:
            raise ValueError("Provider cannot be empty")
        if not self.model_name:
            raise ValueError("Model name cannot be empty")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError("Top-p must be between 0 and 1")


# ==================== Entities ====================

@dataclass
class Tool:
    """
    Tool entity that agents can use.

    Represents a capability that allows agents to interact with external systems.
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    required_permissions: List[str] = field(default_factory=list)
    timeout_seconds: int = 30
    is_destructive: bool = False

    def __post_init__(self):
        if not self.name:
            raise ValueError("Tool name cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")

    def requires_permission(self, permission: str) -> bool:
        """Check if tool requires a specific permission"""
        return permission in self.required_permissions

    def can_execute(self, granted_permissions: List[str]) -> bool:
        """Check if tool can be executed with given permissions"""
        return all(perm in granted_permissions for perm in self.required_permissions)


@dataclass
class Agent:
    """
    Agent aggregate root.

    Represents an AI agent with capabilities, budget, and execution state.

    Invariants:
    - Agent must have at least one capability
    - Budget usage cannot exceed budget limits
    - Agent can only execute tasks matching its capabilities
    - Only IDLE agents can be assigned tasks
    """
    id: AgentId
    role: AgentRole
    name: str
    description: str
    capabilities: List[Capability]
    tools: List[Tool]
    model_config: ModelConfig
    budget_limits: BudgetLimits
    status: AgentStatus = AgentStatus.CREATED
    budget_usage: BudgetUsage = field(default_factory=BudgetUsage)
    failure_strategy: FailureStrategy = FailureStrategy.RETRY
    current_task_id: Optional[UUID] = None
    granted_permissions: List[str] = field(default_factory=list)
    system_prompt: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate agent invariants"""
        if len(self.capabilities) == 0:
            raise ValueError("Agent must have at least one capability")

    def can_execute_task(self, required_capabilities: List[str]) -> bool:
        """
        Business rule: Check if agent can execute a task.

        Agent can execute if:
        1. Has all required capabilities
        2. Is in IDLE status
        3. Has budget remaining
        """
        if self.status != AgentStatus.IDLE:
            return False

        if not self.budget_usage.within_limits(self.budget_limits):
            return False

        # Check all required capabilities are present
        agent_capability_names = [cap.name for cap in self.capabilities]
        return all(req in agent_capability_names for req in required_capabilities)

    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability"""
        return any(cap.name == capability_name for cap in self.capabilities)

    def assign_task(self, task_id: UUID) -> None:
        """
        Business rule: Assign task to agent.

        Can only assign to IDLE or CREATED agents.
        """
        if self.status not in [AgentStatus.IDLE, AgentStatus.CREATED]:
            raise ValueError(f"Cannot assign task to agent in {self.status} status")

        self.current_task_id = task_id
        self.status = AgentStatus.ASSIGNED
        self.last_active = datetime.utcnow()

    def start_reasoning(self) -> None:
        """
        Business rule: Start reasoning phase.

        Can only start reasoning when ASSIGNED.
        """
        if self.status != AgentStatus.ASSIGNED:
            raise ValueError(f"Cannot start reasoning in {self.status} status")

        self.status = AgentStatus.REASONING
        self.last_active = datetime.utcnow()

    def start_execution(self) -> None:
        """
        Business rule: Start execution phase.

        Can transition from REASONING or RETRY states.
        """
        if self.status not in [AgentStatus.REASONING, AgentStatus.RETRY]:
            raise ValueError(f"Cannot start execution in {self.status} status")

        self.status = AgentStatus.EXECUTING
        self.last_active = datetime.utcnow()

    def start_validation(self) -> None:
        """
        Business rule: Start validation phase.

        Can only validate after EXECUTING.
        """
        if self.status != AgentStatus.EXECUTING:
            raise ValueError(f"Cannot start validation in {self.status} status")

        self.status = AgentStatus.VALIDATING
        self.last_active = datetime.utcnow()

    def complete_task(self) -> None:
        """
        Business rule: Complete current task.

        Can complete from EXECUTING or VALIDATING states.
        """
        if self.status not in [AgentStatus.EXECUTING, AgentStatus.VALIDATING]:
            raise ValueError(f"Cannot complete task in {self.status} status")

        self.status = AgentStatus.COMPLETED
        self.current_task_id = None
        self.last_active = datetime.utcnow()

    def fail_task(self, error: str) -> None:
        """
        Business rule: Mark task as failed.

        Can fail from any active status (ASSIGNED, REASONING, EXECUTING, VALIDATING).
        """
        if self.status in [AgentStatus.CREATED, AgentStatus.IDLE, AgentStatus.TERMINATED]:
            raise ValueError(f"Cannot fail task in {self.status} status")

        self.status = AgentStatus.FAILED
        self.last_active = datetime.utcnow()

    def retry_task(self) -> None:
        """
        Business rule: Retry failed task.

        Can only retry from FAILED status.
        """
        if self.status != AgentStatus.FAILED:
            raise ValueError("Can only retry from FAILED status")

        self.status = AgentStatus.RETRY
        self.last_active = datetime.utcnow()

    def make_idle(self) -> None:
        """
        Business rule: Transition to IDLE state.

        Can transition from COMPLETED, FAILED, or CREATED states.
        """
        if self.status not in [AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.CREATED]:
            raise ValueError(f"Cannot make agent idle from {self.status} status")

        self.status = AgentStatus.IDLE
        self.current_task_id = None
        self.last_active = datetime.utcnow()

    def terminate(self) -> None:
        """
        Business rule: Terminate agent.

        Can terminate from any status except TERMINATED.
        """
        if self.status == AgentStatus.TERMINATED:
            raise ValueError("Agent is already terminated")

        self.status = AgentStatus.TERMINATED
        self.current_task_id = None
        self.last_active = datetime.utcnow()

    def is_active(self) -> bool:
        """Check if agent is in an active state"""
        return self.status in [
            AgentStatus.ASSIGNED,
            AgentStatus.REASONING,
            AgentStatus.EXECUTING,
            AgentStatus.VALIDATING
        ]

    def is_idle_or_available(self) -> bool:
        """Check if agent is available for new tasks"""
        return self.status in [AgentStatus.IDLE, AgentStatus.CREATED]

    def record_token_usage(self, tokens: int, cost: float) -> None:
        """Record LLM token usage"""
        self.budget_usage.add_token_usage(tokens, cost)
        self.last_active = datetime.utcnow()

        # Check if budget exceeded
        if not self.budget_usage.within_limits(self.budget_limits):
            raise ValueError("Budget limits exceeded")

    def record_tool_execution(self, duration_seconds: float) -> None:
        """Record tool execution"""
        self.budget_usage.add_tool_execution(duration_seconds)
        self.last_active = datetime.utcnow()

        # Check if budget exceeded
        if not self.budget_usage.within_limits(self.budget_limits):
            raise ValueError("Budget limits exceeded")

    def reset_budget(self) -> None:
        """Reset budget usage counters"""
        self.budget_usage.reset()

    def can_use_tool(self, tool_name: str) -> bool:
        """Check if agent can use a specific tool"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.can_execute(self.granted_permissions)
        return False

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get tool by name"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def grant_permission(self, permission: str) -> None:
        """Grant a permission to the agent"""
        if permission not in self.granted_permissions:
            self.granted_permissions.append(permission)

    def revoke_permission(self, permission: str) -> None:
        """Revoke a permission from the agent"""
        if permission in self.granted_permissions:
            self.granted_permissions.remove(permission)

    def uptime_seconds(self) -> int:
        """Calculate agent uptime in seconds"""
        return int((datetime.utcnow() - self.created_at).total_seconds())


@dataclass
class AgentDefinition:
    """
    Agent definition (template/blueprint).

    Represents a reusable agent configuration.
    """
    id: UUID
    name: str
    role: AgentRole
    version: str
    description: str
    default_capabilities: List[Capability]
    default_tools: List[Tool]
    model_config: ModelConfig
    default_budget: BudgetLimits
    system_prompt_template: str
    created_at: datetime = field(default_factory=datetime.utcnow)

    def instantiate(
        self,
        agent_id: AgentId,
        custom_budget: Optional[BudgetLimits] = None,
        custom_permissions: Optional[List[str]] = None
    ) -> Agent:
        """
        Factory method: Create agent instance from definition.

        Args:
            agent_id: Unique agent ID
            custom_budget: Override default budget
            custom_permissions: Initial permissions

        Returns:
            New Agent instance
        """
        return Agent(
            id=agent_id,
            role=self.role,
            name=self.name,
            description=self.description,
            capabilities=self.default_capabilities.copy(),
            tools=self.default_tools.copy(),
            model_config=self.model_config,
            budget_limits=custom_budget or self.default_budget,
            status=AgentStatus.CREATED,
            granted_permissions=custom_permissions or [],
            system_prompt=self.system_prompt_template
        )

    def is_compatible_with(self, required_capabilities: List[str]) -> bool:
        """Check if this agent definition can fulfill required capabilities"""
        capability_names = [cap.name for cap in self.default_capabilities]
        return all(req in capability_names for req in required_capabilities)
