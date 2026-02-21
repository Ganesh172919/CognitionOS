"""
Agent Planner - Task Planning and Decomposition Engine

Intelligent task planning system that breaks down high-level requirements
into executable task graphs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Types of tasks the agent can execute"""
    CODE_GENERATION = "code_generation"
    CODE_REFACTORING = "code_refactoring"
    TEST_WRITING = "test_writing"
    DOCUMENTATION = "documentation"
    DEBUGGING = "debugging"
    OPTIMIZATION = "optimization"
    ARCHITECTURE_DESIGN = "architecture_design"
    CODE_REVIEW = "code_review"


class TaskPriority(str, Enum):
    """Task execution priority"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PlanningStrategy(str, Enum):
    """Planning strategies"""
    TOP_DOWN = "top_down"  # Start with high-level architecture
    BOTTOM_UP = "bottom_up"  # Start with smallest components
    ITERATIVE = "iterative"  # Build and refine incrementally
    DEPENDENCY_FIRST = "dependency_first"  # Resolve dependencies first


@dataclass
class TaskNode:
    """
    Represents a single task in the execution plan
    """
    id: str
    task_type: TaskType
    description: str
    priority: TaskPriority

    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)

    # Execution details
    estimated_complexity: int = 1  # 1-10 scale
    estimated_time_minutes: int = 30

    # Context
    input_context: Dict[str, Any] = field(default_factory=dict)
    output_requirements: List[str] = field(default_factory=list)

    # Constraints
    must_succeed: bool = True
    max_retries: int = 3
    timeout_seconds: int = 300

    # Status tracking
    status: str = "pending"  # pending, running, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None

    def can_execute(self, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are completed"""
        return all(dep in completed_tasks for dep in self.depends_on)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "task_type": self.task_type.value,
            "description": self.description,
            "priority": self.priority.value,
            "depends_on": self.depends_on,
            "estimated_complexity": self.estimated_complexity,
            "status": self.status,
            "input_context": self.input_context,
            "output_requirements": self.output_requirements
        }


@dataclass
class ExecutionPlan:
    """
    Complete execution plan with task graph
    """
    id: str
    objective: str
    strategy: PlanningStrategy
    tasks: List[TaskNode]
    created_at: datetime

    # Metadata
    total_estimated_time: int = 0
    total_complexity: int = 0
    critical_path: List[str] = field(default_factory=list)

    def get_executable_tasks(self, completed_tasks: Set[str]) -> List[TaskNode]:
        """Get tasks that can be executed now"""
        return [
            task for task in self.tasks
            if task.status == "pending" and task.can_execute(completed_tasks)
        ]

    def get_task(self, task_id: str) -> Optional[TaskNode]:
        """Get task by ID"""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def calculate_critical_path(self) -> List[str]:
        """Calculate critical path through task graph"""
        # Simple implementation - would use proper graph algorithms
        critical_tasks = [
            task for task in self.tasks
            if task.priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]
        ]
        return [task.id for task in sorted(critical_tasks, key=lambda t: len(t.depends_on))]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "objective": self.objective,
            "strategy": self.strategy.value,
            "tasks": [task.to_dict() for task in self.tasks],
            "total_estimated_time": self.total_estimated_time,
            "total_complexity": self.total_complexity,
            "critical_path": self.critical_path
        }


class AgentPlanner:
    """
    Intelligent agent planner that creates execution plans

    Uses LLM-powered analysis to break down complex requirements into
    executable task graphs.
    """

    def __init__(self, llm_provider: Optional[Any] = None):
        self.llm_provider = llm_provider
        self._task_counter = 0

    async def create_plan(
        self,
        objective: str,
        constraints: Optional[Dict[str, Any]] = None,
        strategy: PlanningStrategy = PlanningStrategy.TOP_DOWN,
        existing_codebase: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """
        Create execution plan from high-level objective

        Args:
            objective: High-level goal description
            constraints: Optional constraints (time, resources, etc.)
            strategy: Planning strategy to use
            existing_codebase: Context about existing code

        Returns:
            Complete execution plan
        """
        logger.info(f"Creating execution plan for: {objective}")

        # Analyze objective
        analysis = await self._analyze_objective(objective, existing_codebase)

        # Decompose into tasks
        tasks = await self._decompose_into_tasks(
            objective=objective,
            analysis=analysis,
            strategy=strategy,
            constraints=constraints
        )

        # Build dependency graph
        tasks = self._build_dependencies(tasks, analysis)

        # Optimize task order
        tasks = self._optimize_task_order(tasks, strategy)

        # Create execution plan
        plan = ExecutionPlan(
            id=f"plan_{datetime.utcnow().timestamp()}",
            objective=objective,
            strategy=strategy,
            tasks=tasks,
            created_at=datetime.utcnow()
        )

        # Calculate metrics
        plan.total_estimated_time = sum(t.estimated_time_minutes for t in tasks)
        plan.total_complexity = sum(t.estimated_complexity for t in tasks)
        plan.critical_path = plan.calculate_critical_path()

        logger.info(f"Created plan with {len(tasks)} tasks, "
                   f"estimated time: {plan.total_estimated_time}min")

        return plan

    async def _analyze_objective(
        self,
        objective: str,
        existing_codebase: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze objective to understand requirements"""

        if self.llm_provider:
            # Use LLM for intelligent analysis
            prompt = f"""Analyze the following software development objective:

Objective: {objective}

Existing Codebase: {json.dumps(existing_codebase, indent=2) if existing_codebase else 'New project'}

Provide analysis:
1. Key requirements
2. Technical complexity (1-10)
3. Required components
4. Dependencies between components
5. Risks and challenges
6. Suggested architecture approach

Return as JSON."""

            analysis = await self.llm_provider.generate(prompt)
            return json.loads(analysis)

        # Fallback: Simple keyword-based analysis
        analysis = {
            "complexity": 5,
            "components": [],
            "dependencies": {},
            "risks": [],
            "approach": "iterative"
        }

        # Detect component types from objective
        if "api" in objective.lower():
            analysis["components"].append("api_endpoint")
        if "database" in objective.lower():
            analysis["components"].append("database_schema")
        if "test" in objective.lower():
            analysis["components"].append("test_suite")
        if "frontend" in objective.lower() or "ui" in objective.lower():
            analysis["components"].append("ui_component")

        return analysis

    async def _decompose_into_tasks(
        self,
        objective: str,
        analysis: Dict[str, Any],
        strategy: PlanningStrategy,
        constraints: Optional[Dict[str, Any]]
    ) -> List[TaskNode]:
        """Decompose objective into executable tasks"""

        tasks = []

        # Architecture design (if new project or significant changes)
        if strategy == PlanningStrategy.TOP_DOWN:
            tasks.append(TaskNode(
                id=self._next_task_id(),
                task_type=TaskType.ARCHITECTURE_DESIGN,
                description="Design system architecture and component structure",
                priority=TaskPriority.CRITICAL,
                estimated_complexity=analysis.get("complexity", 5),
                estimated_time_minutes=60
            ))

        # Generate tasks for each component
        for component in analysis.get("components", []):
            # Code generation task
            code_task = TaskNode(
                id=self._next_task_id(),
                task_type=TaskType.CODE_GENERATION,
                description=f"Implement {component}",
                priority=TaskPriority.HIGH,
                estimated_complexity=5,
                estimated_time_minutes=45,
                input_context={"component": component}
            )
            tasks.append(code_task)

            # Test writing task
            test_task = TaskNode(
                id=self._next_task_id(),
                task_type=TaskType.TEST_WRITING,
                description=f"Write tests for {component}",
                priority=TaskPriority.MEDIUM,
                estimated_complexity=3,
                estimated_time_minutes=30,
                depends_on=[code_task.id],
                input_context={"component": component}
            )
            tasks.append(test_task)

        # Documentation task
        if not constraints or constraints.get("include_docs", True):
            doc_task = TaskNode(
                id=self._next_task_id(),
                task_type=TaskType.DOCUMENTATION,
                description="Generate documentation",
                priority=TaskPriority.LOW,
                estimated_complexity=2,
                estimated_time_minutes=20
            )
            tasks.append(doc_task)

        # Code review task
        review_task = TaskNode(
            id=self._next_task_id(),
            task_type=TaskType.CODE_REVIEW,
            description="Review generated code for quality and best practices",
            priority=TaskPriority.HIGH,
            estimated_complexity=4,
            estimated_time_minutes=30
        )
        tasks.append(review_task)

        # Optimization task
        if constraints and constraints.get("optimize_performance", False):
            opt_task = TaskNode(
                id=self._next_task_id(),
                task_type=TaskType.OPTIMIZATION,
                description="Optimize code performance",
                priority=TaskPriority.MEDIUM,
                estimated_complexity=6,
                estimated_time_minutes=45
            )
            tasks.append(opt_task)

        return tasks

    def _build_dependencies(
        self,
        tasks: List[TaskNode],
        analysis: Dict[str, Any]
    ) -> List[TaskNode]:
        """Build dependency graph between tasks"""

        # Simple rules-based dependency building
        architecture_tasks = [t for t in tasks if t.task_type == TaskType.ARCHITECTURE_DESIGN]
        code_tasks = [t for t in tasks if t.task_type == TaskType.CODE_GENERATION]
        test_tasks = [t for t in tasks if t.task_type == TaskType.TEST_WRITING]
        review_tasks = [t for t in tasks if t.task_type == TaskType.CODE_REVIEW]
        doc_tasks = [t for t in tasks if t.task_type == TaskType.DOCUMENTATION]

        # Code depends on architecture
        if architecture_tasks:
            arch_id = architecture_tasks[0].id
            for task in code_tasks:
                if arch_id not in task.depends_on:
                    task.depends_on.append(arch_id)

        # Review depends on code and tests
        for review_task in review_tasks:
            review_task.depends_on.extend([t.id for t in code_tasks])
            review_task.depends_on.extend([t.id for t in test_tasks])

        # Documentation depends on everything
        for doc_task in doc_tasks:
            doc_task.depends_on.extend([t.id for t in code_tasks])
            doc_task.depends_on.extend([t.id for t in test_tasks])

        return tasks

    def _optimize_task_order(
        self,
        tasks: List[TaskNode],
        strategy: PlanningStrategy
    ) -> List[TaskNode]:
        """Optimize task execution order"""

        if strategy == PlanningStrategy.DEPENDENCY_FIRST:
            # Sort by number of dependencies (fewer first)
            return sorted(tasks, key=lambda t: len(t.depends_on))
        elif strategy == PlanningStrategy.BOTTOM_UP:
            # Reverse topological sort
            return sorted(tasks, key=lambda t: len(t.depends_on), reverse=True)
        else:
            # Default: priority and dependencies
            return sorted(tasks, key=lambda t: (
                {"critical": 0, "high": 1, "medium": 2, "low": 3}[t.priority.value],
                len(t.depends_on)
            ))

    def _next_task_id(self) -> str:
        """Generate next task ID"""
        self._task_counter += 1
        return f"task_{self._task_counter}"

    async def replan(
        self,
        plan: ExecutionPlan,
        failed_task_id: str,
        error: str
    ) -> ExecutionPlan:
        """
        Create new plan after task failure

        Analyzes failure and creates recovery plan.
        """
        logger.info(f"Replanning after failure of task {failed_task_id}")

        failed_task = plan.get_task(failed_task_id)
        if not failed_task:
            raise ValueError(f"Task {failed_task_id} not found")

        # Analyze failure
        if self.llm_provider:
            prompt = f"""A task failed during execution:

Task: {failed_task.description}
Error: {error}

Suggest:
1. Root cause
2. Alternative approach
3. New tasks needed
4. Which existing tasks need modification

Return as JSON."""

            recovery_plan = await self.llm_provider.generate(prompt)
            recovery = json.loads(recovery_plan)
        else:
            recovery = {
                "approach": "retry",
                "new_tasks": []
            }

        # Create recovery tasks
        new_tasks = []
        if recovery.get("approach") == "retry":
            # Simply retry the failed task
            failed_task.status = "pending"
            failed_task.error = None
            failed_task.max_retries -= 1
        else:
            # Create alternative tasks
            for new_task_desc in recovery.get("new_tasks", []):
                new_task = TaskNode(
                    id=self._next_task_id(),
                    task_type=TaskType.CODE_GENERATION,
                    description=new_task_desc,
                    priority=TaskPriority.HIGH,
                    estimated_complexity=5,
                    estimated_time_minutes=30
                )
                new_tasks.append(new_task)

        # Update plan
        plan.tasks.extend(new_tasks)
        plan.total_estimated_time = sum(t.estimated_time_minutes for t in plan.tasks)

        return plan

    async def optimize_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """
        Optimize existing plan for efficiency

        Looks for parallelization opportunities and redundancies.
        """
        logger.info("Optimizing execution plan")

        # Find tasks that can run in parallel
        # (tasks with no dependencies or same dependencies)
        parallel_groups = []
        for task in plan.tasks:
            if task.status == "pending":
                # Find compatible tasks
                compatible = [
                    t for t in plan.tasks
                    if t.status == "pending" and
                    t.id != task.id and
                    set(t.depends_on) == set(task.depends_on)
                ]
                if compatible:
                    parallel_groups.append([task] + compatible)

        # Mark parallel execution capability
        for group in parallel_groups:
            for task in group:
                task.input_context["can_parallelize"] = True
                task.input_context["parallel_group"] = group[0].id

        # Remove redundant tasks
        unique_tasks = []
        seen_descriptions = set()
        for task in plan.tasks:
            if task.description not in seen_descriptions:
                unique_tasks.append(task)
                seen_descriptions.add(task.description)
            else:
                logger.info(f"Removed redundant task: {task.description}")

        plan.tasks = unique_tasks
        plan.total_estimated_time = sum(t.estimated_time_minutes for t in plan.tasks)

        return plan
