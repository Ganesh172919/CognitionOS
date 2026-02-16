"""
Agent Application - Use Cases

Application layer use cases for Agent bounded context.
"""

from dataclasses import dataclass
from typing import Any, List, Optional
from uuid import UUID, uuid4

from core.domain.agent import (
    Agent,
    AgentDefinition,
    AgentId,
    AgentRole,
    AgentStatus,
    Capability,
    BudgetLimits,
    AgentRepository,
    AgentDefinitionRepository,
    AgentMatcher,
    AgentLoadBalancer,
    AgentCreated,
    AgentAssigned,
    AgentCompletedTask
)


# ==================== DTOs ====================

@dataclass
class RegisterAgentDefinitionCommand:
    """Command to register a new agent definition"""
    name: str
    role: AgentRole
    version: str
    description: str
    capabilities: List[str]
    tools: List[dict]
    model_config: dict
    default_budget: dict
    system_prompt: str


@dataclass
class CreateAgentCommand:
    """Command to create agent instance"""
    definition_id: UUID
    custom_budget: Optional[dict] = None
    permissions: Optional[List[str]] = None


@dataclass
class AssignTaskToAgentCommand:
    """Command to assign task to agent"""
    task_id: UUID
    required_capabilities: List[str]
    prefer_role: Optional[str] = None


# ==================== Use Cases ====================

class RegisterAgentDefinitionUseCase:
    """
    Use Case: Register a new agent definition (blueprint).
    """

    def __init__(
        self,
        definition_repository: AgentDefinitionRepository,
        event_publisher: Optional[Any] = None
    ):
        self.definition_repository = definition_repository
        self.event_publisher = event_publisher

    async def execute(self, command: RegisterAgentDefinitionCommand) -> UUID:
        """Register new agent definition"""
        from datetime import datetime
        from core.domain.agent import Tool, ModelConfig, BudgetLimits, Capability

        # Create definition
        definition = AgentDefinition(
            id=uuid4(),
            name=command.name,
            role=command.role,
            version=command.version,
            description=command.description,
            default_capabilities=[
                Capability(name=cap) for cap in command.capabilities
            ],
            default_tools=[
                Tool(
                    name=t["name"],
                    description=t["description"],
                    parameters=t.get("parameters", {}),
                    required_permissions=t.get("required_permissions", [])
                ) for t in command.tools
            ],
            model_config=ModelConfig(**command.model_config),
            default_budget=BudgetLimits(**command.default_budget),
            system_prompt_template=command.system_prompt,
            created_at=datetime.utcnow()
        )

        # Persist
        await self.definition_repository.save(definition)

        return definition.id


class CreateAgentUseCase:
    """
    Use Case: Create a new agent instance from definition.
    """

    def __init__(
        self,
        agent_repository: AgentRepository,
        definition_repository: AgentDefinitionRepository,
        event_publisher: Optional[Any] = None
    ):
        self.agent_repository = agent_repository
        self.definition_repository = definition_repository
        self.event_publisher = event_publisher

    async def execute(self, command: CreateAgentCommand) -> AgentId:
        """Create agent instance"""
        from datetime import datetime

        # Load definition
        definition = await self.definition_repository.get_by_id(command.definition_id)
        if not definition:
            raise ValueError(f"Agent definition {command.definition_id} not found")

        # Parse custom budget if provided
        custom_budget = None
        if command.custom_budget:
            custom_budget = BudgetLimits(**command.custom_budget)

        # Instantiate agent
        agent_id = AgentId(uuid4())
        agent = definition.instantiate(
            agent_id=agent_id,
            custom_budget=custom_budget,
            custom_permissions=command.permissions or []
        )

        # Make idle
        agent.make_idle()

        # Persist
        await self.agent_repository.save(agent)

        # Publish event
        if self.event_publisher:
            event = AgentCreated(
                occurred_at=datetime.utcnow(),
                event_id=uuid4(),
                agent_id=agent.id,
                definition_id=command.definition_id,
                role=agent.role,
                capabilities=[cap.name for cap in agent.capabilities]
            )
            await self.event_publisher.publish(event)

        return agent.id


class AssignTaskToAgentUseCase:
    """
    Use Case: Assign a task to the best available agent.

    Orchestrates:
    1. Find idle agents
    2. Match to task requirements
    3. Assign task
    4. Publish event
    """

    def __init__(
        self,
        agent_repository: AgentRepository,
        event_publisher: Optional[Any] = None
    ):
        self.agent_repository = agent_repository
        self.event_publisher = event_publisher

    async def execute(self, command: AssignTaskToAgentCommand) -> Optional[AgentId]:
        """
        Assign task to best agent.

        Returns:
            AgentId if agent found and assigned, None otherwise
        """
        from datetime import datetime

        # Get idle agents
        idle_agents = await self.agent_repository.get_idle_agents(
            required_capabilities=command.required_capabilities
        )

        if not idle_agents:
            return None

        # Find best match
        best_agent = AgentMatcher.find_best_match(
            agents=idle_agents,
            required_capabilities=command.required_capabilities,
            prefer_role=command.prefer_role
        )

        if not best_agent:
            return None

        # Assign task
        best_agent.assign_task(command.task_id)

        # Persist
        await self.agent_repository.save(best_agent)

        # Publish event
        if self.event_publisher:
            event = AgentAssigned(
                occurred_at=datetime.utcnow(),
                event_id=uuid4(),
                agent_id=best_agent.id,
                task_id=command.task_id,
                required_capabilities=command.required_capabilities
            )
            await self.event_publisher.publish(event)

        return best_agent.id


class CompleteAgentTaskUseCase:
    """
    Use Case: Complete an agent's current task.
    """

    def __init__(
        self,
        agent_repository: AgentRepository,
        event_publisher: Optional[Any] = None
    ):
        self.agent_repository = agent_repository
        self.event_publisher = event_publisher

    async def execute(self, agent_id: AgentId) -> None:
        """Complete agent's current task"""
        from datetime import datetime

        # Load agent
        agent = await self.agent_repository.get_by_id(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        task_id = agent.current_task_id

        # Complete task
        agent.complete_task()

        # Make idle
        agent.make_idle()

        # Persist
        await self.agent_repository.save(agent)

        # Publish event
        if self.event_publisher and task_id:
            event = AgentCompletedTask(
                occurred_at=datetime.utcnow(),
                event_id=uuid4(),
                agent_id=agent.id,
                task_id=task_id,
                duration_seconds=0  # Would be calculated from task start
            )
            await self.event_publisher.publish(event)
