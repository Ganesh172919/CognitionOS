"""
Agent Orchestrator Service.

Manages agent lifecycle, task assignment, and execution coordination.
"""

import sys
import os

# Add shared libs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from datetime import datetime, timedelta
from typing import List, Dict, Optional
from uuid import UUID, uuid4
from enum import Enum

from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

from shared.libs.config import AgentOrchestratorConfig, load_config
from shared.libs.logger import setup_logger, get_contextual_logger
from shared.libs.models import (
    AgentDefinition, AgentInstance, AgentState, AgentRole,
    Task, TaskStatus, BudgetLimits, BudgetUsed,
    ToolDefinition, ModelConfig, ErrorResponse
)
from shared.libs.middleware import (
    TracingMiddleware,
    LoggingMiddleware,
    ErrorHandlingMiddleware
)
from shared.libs.utils import calculate_exponential_backoff


# Configuration
config = load_config(AgentOrchestratorConfig)
logger = setup_logger(__name__, level=config.log_level)

# FastAPI app
app = FastAPI(
    title="CognitionOS Agent Orchestrator",
    version=config.service_version,
    description="Agent lifecycle and execution management"
)

# Add middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(TracingMiddleware)


# ============================================================================
# Request/Response Models
# ============================================================================

class RegisterAgentRequest(BaseModel):
    """Request to register a new agent definition."""
    name: str
    role: AgentRole
    description: str
    capabilities: List[str]
    tools: List[ToolDefinition]
    model_config: ModelConfig
    default_budget: BudgetLimits
    system_prompt: str


class AssignTaskRequest(BaseModel):
    """Request to assign a task to an agent."""
    task_id: UUID
    user_id: UUID
    required_capabilities: List[str]
    task_data: Dict
    priority: int = 0


class AgentStatusResponse(BaseModel):
    """Agent status information."""
    instance_id: UUID
    definition_id: UUID
    role: AgentRole
    state: AgentState
    current_task_id: Optional[UUID]
    budget_used: BudgetUsed
    uptime_seconds: int


class TaskExecutionResult(BaseModel):
    """Result of task execution."""
    task_id: UUID
    agent_id: UUID
    success: bool
    output: Optional[Dict] = None
    error: Optional[str] = None
    duration_seconds: float
    cost: float


# ============================================================================
# In-Memory Storage
# ============================================================================

# In production, use database + Redis
agent_definitions: Dict[UUID, AgentDefinition] = {}
agent_instances: Dict[UUID, AgentInstance] = {}
tasks_queue: List[Task] = []
task_results: Dict[UUID, TaskExecutionResult] = {}


# ============================================================================
# Agent Registry
# ============================================================================

class AgentRegistry:
    """
    Registry of available agent definitions.

    Stores agent templates that can be instantiated.
    """

    def __init__(self):
        self.logger = get_contextual_logger(__name__, component="AgentRegistry")
        self._initialize_default_agents()

    def _initialize_default_agents(self):
        """Register default agent types."""

        # Planner Agent
        planner = AgentDefinition(
            name="Planner Agent",
            role=AgentRole.PLANNER,
            version="1.0.0",
            description="Decomposes goals and creates plans",
            capabilities=["planning", "reasoning", "analysis"],
            tools=[
                ToolDefinition(
                    name="create_plan",
                    description="Create execution plan",
                    parameters={"goal": "string"},
                    required_permissions=[]
                )
            ],
            model_config=ModelConfig(
                provider="openai",
                model_name="gpt-4",
                temperature=0.7
            ),
            default_budget=BudgetLimits(
                max_tokens=4000,
                max_cost_usd=0.5,
                max_time_seconds=120
            ),
            system_prompt="You are a planning agent. Break down complex goals into executable steps."
        )
        agent_definitions[planner.id] = planner

        # Executor Agent
        executor = AgentDefinition(
            name="Executor Agent",
            role=AgentRole.EXECUTOR,
            version="1.0.0",
            description="Executes concrete tasks and operations",
            capabilities=["execution", "tool_use", "code_generation"],
            tools=[
                ToolDefinition(
                    name="execute_code",
                    description="Execute Python code",
                    parameters={"code": "string"},
                    required_permissions=["code_execution"]
                ),
                ToolDefinition(
                    name="api_call",
                    description="Make HTTP API calls",
                    parameters={"url": "string", "method": "string"},
                    required_permissions=["network"]
                )
            ],
            model_config=ModelConfig(
                provider="openai",
                model_name="gpt-3.5-turbo",
                temperature=0.3
            ),
            default_budget=BudgetLimits(
                max_tokens=2000,
                max_cost_usd=0.1,
                max_time_seconds=300
            ),
            system_prompt="You are an executor agent. Run tools and execute tasks precisely."
        )
        agent_definitions[executor.id] = executor

        # Critic Agent
        critic = AgentDefinition(
            name="Critic Agent",
            role=AgentRole.CRITIC,
            version="1.0.0",
            description="Evaluates quality and correctness",
            capabilities=["validation", "quality_check", "testing"],
            tools=[
                ToolDefinition(
                    name="validate_output",
                    description="Validate task output",
                    parameters={"output": "object", "criteria": "object"},
                    required_permissions=[]
                )
            ],
            model_config=ModelConfig(
                provider="openai",
                model_name="gpt-4",
                temperature=0.2
            ),
            default_budget=BudgetLimits(
                max_tokens=2000,
                max_cost_usd=0.2,
                max_time_seconds=60
            ),
            system_prompt="You are a critic agent. Evaluate outputs for quality and correctness."
        )
        agent_definitions[critic.id] = critic

        self.logger.info(
            "Initialized default agents",
            extra={"count": len(agent_definitions)}
        )

    def register(self, agent_def: AgentDefinition) -> UUID:
        """Register a new agent definition."""
        agent_definitions[agent_def.id] = agent_def
        self.logger.info(
            "Agent registered",
            extra={"agent_id": str(agent_def.id), "role": agent_def.role}
        )
        return agent_def.id

    def find_by_capabilities(
        self,
        required_capabilities: List[str]
    ) -> List[AgentDefinition]:
        """
        Find agent definitions matching required capabilities.

        Args:
            required_capabilities: List of required capabilities

        Returns:
            Matching agent definitions
        """
        matches = []
        for agent_def in agent_definitions.values():
            if all(cap in agent_def.capabilities for cap in required_capabilities):
                matches.append(agent_def)
        return matches

    def get(self, agent_id: UUID) -> Optional[AgentDefinition]:
        """Get agent definition by ID."""
        return agent_definitions.get(agent_id)


# ============================================================================
# Agent Pool
# ============================================================================

class AgentPool:
    """
    Manages pool of agent instances.

    Spawns, reuses, and terminates agents.
    """

    def __init__(self, max_agents: int):
        self.max_agents = max_agents
        self.logger = get_contextual_logger(__name__, component="AgentPool")

    def spawn(
        self,
        definition_id: UUID,
        user_id: UUID
    ) -> AgentInstance:
        """
        Spawn a new agent instance.

        Args:
            definition_id: Agent definition to instantiate
            user_id: User owning this agent

        Returns:
            New agent instance
        """
        if len(agent_instances) >= self.max_agents:
            raise RuntimeError("Agent pool exhausted")

        agent_def = agent_definitions.get(definition_id)
        if not agent_def:
            raise ValueError(f"Agent definition not found: {definition_id}")

        instance = AgentInstance(
            definition_id=definition_id,
            user_id=user_id,
            state=AgentState.IDLE,
            budget_limits=agent_def.default_budget
        )

        agent_instances[instance.id] = instance

        self.logger.info(
            "Agent spawned",
            extra={
                "instance_id": str(instance.id),
                "role": agent_def.role,
                "user_id": str(user_id)
            }
        )

        return instance

    def get_idle_agent(
        self,
        agent_definitions_list: List[AgentDefinition]
    ) -> Optional[AgentInstance]:
        """
        Get an idle agent matching one of the definitions.

        Args:
            agent_definitions_list: Acceptable agent definitions

        Returns:
            Idle agent instance or None
        """
        definition_ids = {ad.id for ad in agent_definitions_list}

        for instance in agent_instances.values():
            if (instance.state == AgentState.IDLE and
                instance.definition_id in definition_ids):
                return instance

        return None

    def terminate(self, instance_id: UUID):
        """Terminate an agent instance."""
        if instance_id in agent_instances:
            del agent_instances[instance_id]
            self.logger.info(
                "Agent terminated",
                extra={"instance_id": str(instance_id)}
            )

    def get_status(self, instance_id: UUID) -> Optional[AgentStatusResponse]:
        """Get agent status."""
        instance = agent_instances.get(instance_id)
        if not instance:
            return None

        uptime = (datetime.utcnow() - instance.created_at).total_seconds()

        agent_def = agent_definitions.get(instance.definition_id)

        return AgentStatusResponse(
            instance_id=instance.id,
            definition_id=instance.definition_id,
            role=agent_def.role if agent_def else AgentRole.EXECUTOR,
            state=instance.state,
            current_task_id=instance.current_task_id,
            budget_used=instance.budget_used,
            uptime_seconds=int(uptime)
        )


# ============================================================================
# Task Assignment
# ============================================================================

class TaskAssigner:
    """
    Assigns tasks to agents.

    Matches task requirements with agent capabilities.
    """

    def __init__(self, registry: AgentRegistry, pool: AgentPool):
        self.registry = registry
        self.pool = pool
        self.logger = get_contextual_logger(__name__, component="TaskAssigner")

    def assign(
        self,
        task: Task
    ) -> AgentInstance:
        """
        Assign task to an agent.

        Args:
            task: Task to assign

        Returns:
            Agent instance assigned to task

        Raises:
            RuntimeError: If no suitable agent available
        """
        self.logger.info(
            "Assigning task",
            extra={"task_id": str(task.id), "capabilities": task.required_capabilities}
        )

        # Find agent definitions matching capabilities
        matching_defs = self.registry.find_by_capabilities(task.required_capabilities)

        if not matching_defs:
            raise RuntimeError(
                f"No agent capable of: {task.required_capabilities}"
            )

        # Try to reuse idle agent
        agent = self.pool.get_idle_agent(matching_defs)

        # Spawn new agent if none available
        if not agent:
            agent = self.pool.spawn(
                definition_id=matching_defs[0].id,
                user_id=task.user_id
            )

        # Assign task to agent
        agent.state = AgentState.ASSIGNED
        agent.current_task_id = task.id
        task.assigned_agent_id = agent.id
        task.status = TaskStatus.ASSIGNED

        self.logger.info(
            "Task assigned",
            extra={
                "task_id": str(task.id),
                "agent_id": str(agent.id)
            }
        )

        return agent


# ============================================================================
# Task Executor
# ============================================================================

class TaskExecutor:
    """
    Executes tasks using agents.

    Simulates agent execution (in production, integrates with AI Runtime).
    """

    def __init__(self):
        self.logger = get_contextual_logger(__name__, component="TaskExecutor")

    async def execute(
        self,
        task: Task,
        agent: AgentInstance
    ) -> TaskExecutionResult:
        """
        Execute a task with an agent.

        Args:
            task: Task to execute
            agent: Agent to use

        Returns:
            Execution result
        """
        self.logger.info(
            "Executing task",
            extra={
                "task_id": str(task.id),
                "agent_id": str(agent.id)
            }
        )

        start_time = datetime.utcnow()

        # Update states
        agent.state = AgentState.EXECUTING
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = start_time

        try:
            # In production:
            # 1. Load task context from memory
            # 2. Call AI Runtime with agent config
            # 3. Execute tools as needed
            # 4. Validate output with critic
            # 5. Store results in memory

            # For now, simulate execution
            import asyncio
            await asyncio.sleep(0.1)  # Simulate work

            # Simulate success
            output = {
                "result": f"Completed: {task.description}",
                "agent_role": agent_definitions[agent.definition_id].role.value
            }

            # Update task
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.output_data = output

            # Update agent
            agent.state = AgentState.COMPLETED
            agent.current_task_id = None

            duration = (datetime.utcnow() - start_time).total_seconds()
            cost = 0.01  # Simulate cost

            # Update budget
            agent.budget_used.tokens_used += 500
            agent.budget_used.cost_usd += cost
            agent.budget_used.time_seconds += duration

            result = TaskExecutionResult(
                task_id=task.id,
                agent_id=agent.id,
                success=True,
                output=output,
                duration_seconds=duration,
                cost=cost
            )

            # Return agent to pool
            agent.state = AgentState.IDLE

            self.logger.info(
                "Task completed",
                extra={
                    "task_id": str(task.id),
                    "duration": duration,
                    "cost": cost
                }
            )

            return result

        except Exception as e:
            # Handle failure
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            agent.state = AgentState.FAILED

            duration = (datetime.utcnow() - start_time).total_seconds()

            self.logger.error(
                "Task failed",
                extra={
                    "task_id": str(task.id),
                    "error": str(e)
                }
            )

            return TaskExecutionResult(
                task_id=task.id,
                agent_id=agent.id,
                success=False,
                error=str(e),
                duration_seconds=duration,
                cost=0.0
            )


# ============================================================================
# Orchestrator
# ============================================================================

class Orchestrator:
    """
    Main orchestration engine.

    Coordinates all agent operations.
    """

    def __init__(self):
        self.registry = AgentRegistry()
        self.pool = AgentPool(max_agents=config.max_agents)
        self.assigner = TaskAssigner(self.registry, self.pool)
        self.executor = TaskExecutor()
        self.logger = get_contextual_logger(__name__, component="Orchestrator")

    async def execute_task(self, task: Task) -> TaskExecutionResult:
        """
        Execute a task end-to-end.

        Args:
            task: Task to execute

        Returns:
            Execution result
        """
        # Assign to agent
        agent = self.assigner.assign(task)

        # Execute
        result = await self.executor.execute(task, agent)

        # Handle retry on failure
        if not result.success and task.retry_count < task.max_retries:
            task.retry_count += 1
            backoff = calculate_exponential_backoff(task.retry_count)

            self.logger.info(
                "Retrying task",
                extra={
                    "task_id": str(task.id),
                    "retry_count": task.retry_count,
                    "backoff_seconds": backoff
                }
            )

            # In production, schedule retry with backoff
            # For now, just return the failure
            pass

        # Store result
        task_results[task.id] = result

        return result


# ============================================================================
# API Endpoints
# ============================================================================

orchestrator = Orchestrator()


@app.post("/agents/register", response_model=AgentDefinition)
async def register_agent(request: RegisterAgentRequest):
    """Register a new agent definition."""
    agent_def = AgentDefinition(
        name=request.name,
        role=request.role,
        version="1.0.0",
        description=request.description,
        capabilities=request.capabilities,
        tools=request.tools,
        model_config=request.model_config,
        default_budget=request.default_budget,
        system_prompt=request.system_prompt
    )

    orchestrator.registry.register(agent_def)
    return agent_def


@app.get("/agents/definitions")
async def list_agent_definitions():
    """List all registered agent definitions."""
    return list(agent_definitions.values())


@app.get("/agents/instances")
async def list_agent_instances():
    """List all active agent instances."""
    return [
        orchestrator.pool.get_status(instance_id)
        for instance_id in agent_instances.keys()
    ]


@app.get("/agents/{instance_id}", response_model=AgentStatusResponse)
async def get_agent_status(instance_id: UUID):
    """Get status of a specific agent."""
    status_response = orchestrator.pool.get_status(instance_id)
    if not status_response:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found: {instance_id}"
        )
    return status_response


@app.post("/tasks/execute", response_model=TaskExecutionResult)
async def execute_task(
    request: AssignTaskRequest,
    background_tasks: BackgroundTasks
):
    """
    Execute a task.

    Assigns to an agent and executes.
    """
    task = Task(
        id=request.task_id,
        user_id=request.user_id,
        session_id=uuid4(),
        goal_id=uuid4(),
        name=f"Task {request.task_id}",
        description=str(request.task_data),
        required_capabilities=request.required_capabilities,
        input_data=request.task_data
    )

    result = await orchestrator.execute_task(task)
    return result


@app.get("/tasks/{task_id}/result", response_model=TaskExecutionResult)
async def get_task_result(task_id: UUID):
    """Get task execution result."""
    if task_id not in task_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task result not found: {task_id}"
        )
    return task_results[task_id]


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "agent-orchestrator",
        "version": config.service_version,
        "timestamp": datetime.utcnow().isoformat(),
        "active_agents": len(agent_instances),
        "max_agents": config.max_agents
    }


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize service on startup."""
    logger.info(
        "Agent Orchestrator starting",
        extra={
            "version": config.service_version,
            "max_agents": config.max_agents
        }
    )


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Agent Orchestrator shutting down")

    # Terminate all agents
    for instance_id in list(agent_instances.keys()):
        orchestrator.pool.terminate(instance_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower()
    )
