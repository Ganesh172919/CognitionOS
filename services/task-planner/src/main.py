"""
Task Planner Service.

Decomposes high-level goals into executable task DAGs.
"""

import sys
import os

# Add shared libs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple
from uuid import UUID, uuid4

import networkx as nx
from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, Field

from shared.libs.config import BaseConfig, load_config
from shared.libs.logger import setup_logger, get_contextual_logger
from shared.libs.models import (
    Task, TaskStatus, Goal, AgentRole,
    ErrorResponse, BaseIDModel
)
from shared.libs.middleware import (
    TracingMiddleware,
    LoggingMiddleware,
    ErrorHandlingMiddleware
)


# Configuration
config = load_config(BaseConfig)
config.service_name = "task-planner"
config.port = 8002
logger = setup_logger(__name__, level=config.log_level)

# FastAPI app
app = FastAPI(
    title="CognitionOS Task Planner",
    version=config.service_version,
    description="Goal decomposition and task planning service"
)

# Add middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(TracingMiddleware)


# ============================================================================
# Request/Response Models
# ============================================================================

class PlanRequest(BaseModel):
    """Request to create a plan from a goal."""
    user_id: UUID
    goal: str
    context: Optional[Dict[str, any]] = Field(default_factory=dict)
    max_depth: int = Field(default=10, ge=1, le=20)


class TaskNode(BaseModel):
    """Task node in the DAG."""
    task_id: UUID
    name: str
    description: str
    required_capabilities: List[str]
    dependencies: List[UUID]
    estimated_duration_seconds: int = 60
    priority: int = 0  # Higher = more important


class PlanResponse(BaseIDModel):
    """Response containing task plan."""
    goal_id: UUID
    user_id: UUID
    goal: str
    tasks: List[TaskNode]
    execution_order: List[List[UUID]]  # Stages for parallel execution
    total_estimated_duration_seconds: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ReplanRequest(BaseModel):
    """Request to replan after task failure."""
    goal_id: UUID
    failed_task_id: UUID
    error_message: str


# ============================================================================
# In-Memory Storage
# ============================================================================

plans_db: Dict[UUID, PlanResponse] = {}
goals_db: Dict[UUID, Goal] = {}


# ============================================================================
# Task Templates
# ============================================================================

# Predefined task templates for common goals
TASK_TEMPLATES = {
    "build_web_app": [
        {
            "name": "Design Database Schema",
            "capabilities": ["database_design"],
            "dependencies": []
        },
        {
            "name": "Implement Backend API",
            "capabilities": ["backend_development", "api_development"],
            "dependencies": ["Design Database Schema"]
        },
        {
            "name": "Build Frontend UI",
            "capabilities": ["frontend_development"],
            "dependencies": []
        },
        {
            "name": "Write Tests",
            "capabilities": ["testing"],
            "dependencies": ["Implement Backend API", "Build Frontend UI"]
        },
        {
            "name": "Deploy Application",
            "capabilities": ["deployment", "devops"],
            "dependencies": ["Write Tests"]
        }
    ],
    "data_analysis": [
        {
            "name": "Load and Clean Data",
            "capabilities": ["data_processing"],
            "dependencies": []
        },
        {
            "name": "Exploratory Analysis",
            "capabilities": ["data_analysis", "visualization"],
            "dependencies": ["Load and Clean Data"]
        },
        {
            "name": "Build Model",
            "capabilities": ["machine_learning"],
            "dependencies": ["Exploratory Analysis"]
        },
        {
            "name": "Evaluate Model",
            "capabilities": ["data_analysis"],
            "dependencies": ["Build Model"]
        },
        {
            "name": "Generate Report",
            "capabilities": ["documentation", "visualization"],
            "dependencies": ["Evaluate Model"]
        }
    ]
}


# ============================================================================
# Planning Logic
# ============================================================================

class TaskPlanner:
    """
    Task planning engine.

    Decomposes goals into DAGs of executable tasks.
    """

    def __init__(self):
        self.logger = get_contextual_logger(__name__, component="TaskPlanner")

    def plan(
        self,
        user_id: UUID,
        goal: str,
        context: Dict[str, any],
        max_depth: int
    ) -> PlanResponse:
        """
        Create a task plan from a goal.

        Args:
            user_id: User requesting the plan
            goal: High-level goal description
            context: Additional context
            max_depth: Maximum task decomposition depth

        Returns:
            Task plan with DAG
        """
        self.logger.info(
            "Creating plan",
            extra={"user_id": str(user_id), "goal": goal}
        )

        # Try template-based planning first
        plan = self._try_template_planning(user_id, goal)

        if plan:
            self.logger.info("Used template-based planning")
            return plan

        # Fall back to AI-powered planning
        self.logger.info("Using AI-powered planning")
        return self._ai_planning(user_id, goal, context, max_depth)

    def _try_template_planning(
        self,
        user_id: UUID,
        goal: str
    ) -> Optional[PlanResponse]:
        """
        Attempt template-based planning.

        Args:
            user_id: User ID
            goal: Goal description

        Returns:
            Plan if template matches, None otherwise
        """
        goal_lower = goal.lower()

        # Match goal to template
        template_key = None
        if any(keyword in goal_lower for keyword in ["web app", "website", "web application"]):
            template_key = "build_web_app"
        elif any(keyword in goal_lower for keyword in ["analyze", "analysis", "data"]):
            template_key = "data_analysis"

        if not template_key:
            return None

        # Build plan from template
        template = TASK_TEMPLATES[template_key]
        return self._build_plan_from_template(user_id, goal, template)

    def _build_plan_from_template(
        self,
        user_id: UUID,
        goal: str,
        template: List[Dict]
    ) -> PlanResponse:
        """
        Build a plan from a template.

        Args:
            user_id: User ID
            goal: Goal description
            template: Task template

        Returns:
            Complete plan
        """
        goal_id = uuid4()
        tasks = []
        task_map = {}  # name -> task_id

        # Create tasks
        for task_template in template:
            task_id = uuid4()
            task_map[task_template["name"]] = task_id

            tasks.append(TaskNode(
                task_id=task_id,
                name=task_template["name"],
                description=f"{task_template['name']} for: {goal}",
                required_capabilities=task_template["capabilities"],
                dependencies=[],  # Will be filled in next step
                estimated_duration_seconds=300  # 5 minutes default
            ))

        # Add dependencies
        for i, task_template in enumerate(template):
            dep_ids = [
                task_map[dep_name]
                for dep_name in task_template.get("dependencies", [])
            ]
            tasks[i].dependencies = dep_ids

        # Calculate execution order
        execution_order = self._calculate_execution_order(tasks)

        # Calculate total duration
        total_duration = self._calculate_total_duration(tasks, execution_order)

        return PlanResponse(
            goal_id=goal_id,
            user_id=user_id,
            goal=goal,
            tasks=tasks,
            execution_order=execution_order,
            total_estimated_duration_seconds=total_duration
        )

    def _ai_planning(
        self,
        user_id: UUID,
        goal: str,
        context: Dict,
        max_depth: int
    ) -> PlanResponse:
        """
        AI-powered planning using LLM.

        This would integrate with AI Runtime service in production.
        For now, create a simple default plan.

        Args:
            user_id: User ID
            goal: Goal description
            context: Additional context
            max_depth: Maximum decomposition depth

        Returns:
            AI-generated plan
        """
        # In production, call AI Runtime to:
        # 1. Understand the goal
        # 2. Identify required steps
        # 3. Determine dependencies
        # 4. Estimate complexity

        # For now, create a generic plan
        goal_id = uuid4()

        # Generic planning: break into analyze, plan, execute, verify
        analyze_id = uuid4()
        plan_id = uuid4()
        execute_id = uuid4()
        verify_id = uuid4()

        tasks = [
            TaskNode(
                task_id=analyze_id,
                name="Analyze Requirements",
                description=f"Analyze requirements for: {goal}",
                required_capabilities=["reasoning", "analysis"],
                dependencies=[],
                estimated_duration_seconds=120
            ),
            TaskNode(
                task_id=plan_id,
                name="Create Detailed Plan",
                description=f"Create detailed plan for: {goal}",
                required_capabilities=["planning", "reasoning"],
                dependencies=[analyze_id],
                estimated_duration_seconds=180
            ),
            TaskNode(
                task_id=execute_id,
                name="Execute Plan",
                description=f"Execute plan for: {goal}",
                required_capabilities=["execution", "tool_use"],
                dependencies=[plan_id],
                estimated_duration_seconds=600
            ),
            TaskNode(
                task_id=verify_id,
                name="Verify Results",
                description=f"Verify results for: {goal}",
                required_capabilities=["validation", "testing"],
                dependencies=[execute_id],
                estimated_duration_seconds=120
            )
        ]

        execution_order = self._calculate_execution_order(tasks)
        total_duration = self._calculate_total_duration(tasks, execution_order)

        return PlanResponse(
            goal_id=goal_id,
            user_id=user_id,
            goal=goal,
            tasks=tasks,
            execution_order=execution_order,
            total_estimated_duration_seconds=total_duration
        )

    def _calculate_execution_order(
        self,
        tasks: List[TaskNode]
    ) -> List[List[UUID]]:
        """
        Calculate execution order with parallelization.

        Uses topological sort to find execution stages where
        tasks with no dependencies can run in parallel.

        Args:
            tasks: List of tasks

        Returns:
            List of stages, each stage is a list of task IDs
        """
        # Build dependency graph
        graph = nx.DiGraph()

        for task in tasks:
            graph.add_node(task.task_id)
            for dep_id in task.dependencies:
                graph.add_edge(dep_id, task.task_id)

        # Check for cycles
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Circular dependency detected in task graph")

        # Topological generations = execution stages
        stages = []
        for generation in nx.topological_generations(graph):
            stages.append(list(generation))

        return stages

    def _calculate_total_duration(
        self,
        tasks: List[TaskNode],
        execution_order: List[List[UUID]]
    ) -> int:
        """
        Calculate total estimated duration.

        Assumes tasks in the same stage run in parallel,
        so total duration is the sum of the longest task in each stage.

        Args:
            tasks: List of tasks
            execution_order: Execution stages

        Returns:
            Total duration in seconds
        """
        task_map = {task.task_id: task for task in tasks}
        total = 0

        for stage in execution_order:
            # Max duration in this stage (parallel execution)
            stage_duration = max(
                task_map[task_id].estimated_duration_seconds
                for task_id in stage
            )
            total += stage_duration

        return total

    def replan(
        self,
        goal_id: UUID,
        failed_task_id: UUID,
        error_message: str
    ) -> PlanResponse:
        """
        Replan after task failure.

        Args:
            goal_id: Goal being replanned
            failed_task_id: Task that failed
            error_message: Error description

        Returns:
            Updated plan
        """
        self.logger.info(
            "Replanning after failure",
            extra={
                "goal_id": str(goal_id),
                "failed_task_id": str(failed_task_id)
            }
        )

        # Get original plan
        if goal_id not in plans_db:
            raise ValueError(f"Plan not found: {goal_id}")

        original_plan = plans_db[goal_id]

        # Simple replan strategy: retry the failed task
        # In production, could:
        # - Break failed task into smaller tasks
        # - Try alternative approach
        # - Adjust dependencies

        # For now, just mark for retry
        self.logger.info("Marking task for retry")

        return original_plan


# ============================================================================
# API Endpoints
# ============================================================================

planner = TaskPlanner()


@app.post("/plan", response_model=PlanResponse)
async def create_plan(request: PlanRequest):
    """
    Create a task plan from a goal.

    Decomposes the goal into executable tasks with dependencies.
    """
    log = get_contextual_logger(
        __name__,
        action="create_plan",
        user_id=str(request.user_id)
    )

    try:
        plan = planner.plan(
            user_id=request.user_id,
            goal=request.goal,
            context=request.context,
            max_depth=request.max_depth
        )

        # Store plan
        plans_db[plan.goal_id] = plan

        # Create goal record
        goal = Goal(
            user_id=request.user_id,
            description=request.goal,
            task_ids=[task.task_id for task in plan.tasks]
        )
        goals_db[plan.goal_id] = goal

        log.info(
            "Plan created successfully",
            extra={
                "goal_id": str(plan.goal_id),
                "task_count": len(plan.tasks)
            }
        )

        return plan

    except ValueError as e:
        log.error("Planning failed", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@app.get("/plan/{goal_id}", response_model=PlanResponse)
async def get_plan(goal_id: UUID):
    """Get a plan by goal ID."""
    if goal_id not in plans_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plan not found: {goal_id}"
        )

    return plans_db[goal_id]


@app.post("/replan", response_model=PlanResponse)
async def replan_after_failure(request: ReplanRequest):
    """
    Replan after task failure.

    Creates an updated plan that handles the failure.
    """
    log = get_contextual_logger(
        __name__,
        action="replan",
        goal_id=str(request.goal_id)
    )

    try:
        plan = planner.replan(
            goal_id=request.goal_id,
            failed_task_id=request.failed_task_id,
            error_message=request.error_message
        )

        log.info("Replan completed")
        return plan

    except ValueError as e:
        log.error("Replan failed", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "task-planner",
        "version": config.service_version,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize service on startup."""
    logger.info(
        "Task Planner starting",
        extra={
            "version": config.service_version,
            "environment": config.environment
        }
    )


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Task Planner shutting down")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower()
    )
