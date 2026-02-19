"""
Autonomous AI Agent Planning Engine
Decomposes high-level requirements into executable task graphs with intelligent planning.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskPriority(str, Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskType(str, Enum):
    """Types of tasks the agent can handle"""
    CODE_GENERATION = "code_generation"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"
    OPTIMIZATION = "optimization"
    BUG_FIX = "bug_fix"
    FEATURE = "feature"


@dataclass
class TaskNode:
    """Represents a single task in the execution graph"""
    task_id: str
    name: str
    description: str
    task_type: TaskType
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    estimated_complexity: int = 1  # 1-10 scale
    context: Dict[str, Any] = field(default_factory=dict)
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3


class RequirementAnalysis(BaseModel):
    """Analysis of user requirements"""
    primary_goal: str
    complexity_score: int = Field(ge=1, le=10)
    estimated_tasks: int
    required_capabilities: List[str]
    constraints: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    """Complete execution plan for a request"""
    plan_id: str
    requirement: str
    analysis: RequirementAnalysis
    tasks: List[TaskNode]
    execution_order: List[str]  # Task IDs in execution order
    estimated_duration_minutes: int
    confidence_score: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AutonomousPlanner:
    """
    Intelligent task planning engine that decomposes complex requirements
    into executable task graphs with dependency management.
    """

    def __init__(self):
        self.plans: Dict[str, ExecutionPlan] = {}
        self.task_templates = self._initialize_task_templates()

    def _initialize_task_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize common task patterns and templates"""
        return {
            "feature_implementation": {
                "phases": [
                    "requirement_analysis",
                    "architecture_design",
                    "implementation",
                    "testing",
                    "documentation",
                    "integration"
                ],
                "complexity_multiplier": 1.5
            },
            "bug_fix": {
                "phases": [
                    "bug_reproduction",
                    "root_cause_analysis",
                    "fix_implementation",
                    "testing",
                    "regression_check"
                ],
                "complexity_multiplier": 0.8
            },
            "refactoring": {
                "phases": [
                    "code_analysis",
                    "design_improvements",
                    "incremental_refactor",
                    "test_validation",
                    "performance_check"
                ],
                "complexity_multiplier": 1.2
            },
            "optimization": {
                "phases": [
                    "performance_profiling",
                    "bottleneck_identification",
                    "optimization_implementation",
                    "benchmark_validation",
                    "documentation"
                ],
                "complexity_multiplier": 1.3
            }
        }

    async def analyze_requirement(self, requirement: str) -> RequirementAnalysis:
        """
        Analyze user requirement to understand scope and complexity
        """
        # Intelligent requirement analysis
        analysis = RequirementAnalysis(
            primary_goal=requirement,
            complexity_score=self._estimate_complexity(requirement),
            estimated_tasks=self._estimate_task_count(requirement),
            required_capabilities=self._identify_required_capabilities(requirement),
            constraints=self._extract_constraints(requirement),
            risks=self._identify_risks(requirement),
            success_criteria=self._define_success_criteria(requirement)
        )

        return analysis

    def _estimate_complexity(self, requirement: str) -> int:
        """Estimate complexity score (1-10) based on requirement analysis"""
        complexity_indicators = {
            "new feature": 6,
            "refactor": 5,
            "optimize": 6,
            "fix bug": 3,
            "add test": 2,
            "update documentation": 1,
            "architecture": 8,
            "migration": 7,
            "integration": 7,
            "scalability": 8,
            "security": 7,
            "performance": 6
        }

        requirement_lower = requirement.lower()
        scores = []

        for indicator, score in complexity_indicators.items():
            if indicator in requirement_lower:
                scores.append(score)

        if not scores:
            # Default complexity based on length and keywords
            word_count = len(requirement.split())
            if word_count < 10:
                return 3
            elif word_count < 30:
                return 5
            else:
                return 7

        return min(10, max(scores) + len(scores) - 1)

    def _estimate_task_count(self, requirement: str) -> int:
        """Estimate number of tasks needed"""
        complexity = self._estimate_complexity(requirement)

        # Base task count on complexity
        if complexity <= 3:
            return 3  # Simple: design, implement, test
        elif complexity <= 6:
            return 5  # Medium: analysis, design, implement, test, integrate
        else:
            return 8  # Complex: full SDLC

    def _identify_required_capabilities(self, requirement: str) -> List[str]:
        """Identify what capabilities are needed"""
        capabilities = []
        requirement_lower = requirement.lower()

        capability_keywords = {
            "code_generation": ["implement", "create", "build", "develop", "code"],
            "testing": ["test", "validate", "verify", "check"],
            "documentation": ["document", "explain", "describe", "readme"],
            "architecture": ["design", "architect", "structure", "pattern"],
            "optimization": ["optimize", "improve", "enhance", "performance"],
            "refactoring": ["refactor", "restructure", "reorganize", "clean"],
            "analysis": ["analyze", "investigate", "research", "study"],
            "integration": ["integrate", "connect", "combine", "merge"],
            "security": ["secure", "protect", "auth", "encrypt"],
            "database": ["database", "sql", "query", "storage"]
        }

        for capability, keywords in capability_keywords.items():
            if any(keyword in requirement_lower for keyword in keywords):
                capabilities.append(capability)

        return capabilities if capabilities else ["general"]

    def _extract_constraints(self, requirement: str) -> List[str]:
        """Extract constraints from requirement"""
        constraints = []
        requirement_lower = requirement.lower()

        if "backward compatible" in requirement_lower:
            constraints.append("Must maintain backward compatibility")
        if "no breaking changes" in requirement_lower:
            constraints.append("No breaking changes allowed")
        if "performance" in requirement_lower:
            constraints.append("Performance optimization required")
        if "local" in requirement_lower or "localhost" in requirement_lower:
            constraints.append("Must work on local system")
        if "production" in requirement_lower:
            constraints.append("Production-grade quality required")

        return constraints

    def _identify_risks(self, requirement: str) -> List[str]:
        """Identify potential risks"""
        risks = []
        complexity = self._estimate_complexity(requirement)

        if complexity >= 7:
            risks.append("High complexity may require extended development time")

        requirement_lower = requirement.lower()

        if "database" in requirement_lower or "migration" in requirement_lower:
            risks.append("Database changes require careful migration strategy")
        if "security" in requirement_lower or "auth" in requirement_lower:
            risks.append("Security changes require thorough testing")
        if "api" in requirement_lower and "breaking" not in requirement_lower:
            risks.append("API changes may affect existing clients")

        return risks

    def _define_success_criteria(self, requirement: str) -> List[str]:
        """Define success criteria"""
        criteria = [
            "Implementation completes without errors",
            "All tests pass successfully",
            "Code follows project standards"
        ]

        requirement_lower = requirement.lower()

        if "test" in requirement_lower:
            criteria.append("Test coverage meets requirements")
        if "performance" in requirement_lower or "optimize" in requirement_lower:
            criteria.append("Performance metrics show improvement")
        if "production" in requirement_lower:
            criteria.append("Production readiness validated")

        return criteria

    async def create_execution_plan(
        self,
        requirement: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """
        Create a complete execution plan with task graph
        """
        # Analyze requirement
        analysis = await self.analyze_requirement(requirement)

        # Generate task graph
        tasks = self._generate_task_graph(analysis, context or {})

        # Calculate execution order (topological sort)
        execution_order = self._calculate_execution_order(tasks)

        # Estimate duration
        estimated_duration = self._estimate_duration(tasks, analysis.complexity_score)

        # Calculate confidence
        confidence = self._calculate_confidence(analysis, tasks)

        plan = ExecutionPlan(
            plan_id=str(uuid4()),
            requirement=requirement,
            analysis=analysis,
            tasks=[],  # Will be serialized differently
            execution_order=execution_order,
            estimated_duration_minutes=estimated_duration,
            confidence_score=confidence
        )

        # Store plan with tasks
        self.plans[plan.plan_id] = plan

        # Store tasks separately for mutation
        for task in tasks:
            task.context["plan_id"] = plan.plan_id

        return plan

    def _generate_task_graph(
        self,
        analysis: RequirementAnalysis,
        context: Dict[str, Any]
    ) -> List[TaskNode]:
        """Generate task graph based on analysis"""
        tasks = []

        # Determine task template based on requirements
        template_type = self._select_template(analysis.required_capabilities)
        template = self.task_templates.get(template_type, self.task_templates["feature_implementation"])

        # Create tasks for each phase
        previous_task_id = None

        for idx, phase in enumerate(template["phases"]):
            task = TaskNode(
                task_id=f"task_{uuid4().hex[:8]}",
                name=self._humanize_phase_name(phase),
                description=self._generate_task_description(phase, analysis),
                task_type=self._map_phase_to_task_type(phase),
                priority=self._calculate_task_priority(idx, len(template["phases"])),
                estimated_complexity=self._estimate_task_complexity(
                    analysis.complexity_score,
                    template["complexity_multiplier"]
                ),
                context={
                    "phase": phase,
                    "template": template_type,
                    **context
                }
            )

            # Add dependency on previous task (sequential by default)
            if previous_task_id:
                task.dependencies.add(previous_task_id)
                # Find and update previous task's dependents
                for t in tasks:
                    if t.task_id == previous_task_id:
                        t.dependents.add(task.task_id)
                        break

            tasks.append(task)
            previous_task_id = task.task_id

        return tasks

    def _select_template(self, capabilities: List[str]) -> str:
        """Select appropriate task template"""
        if "optimization" in capabilities:
            return "optimization"
        elif "refactoring" in capabilities:
            return "refactoring"
        elif "bug_fix" in [cap.replace("_", " ") for cap in capabilities]:
            return "bug_fix"
        else:
            return "feature_implementation"

    def _humanize_phase_name(self, phase: str) -> str:
        """Convert phase name to human-readable format"""
        return phase.replace("_", " ").title()

    def _generate_task_description(self, phase: str, analysis: RequirementAnalysis) -> str:
        """Generate detailed task description"""
        descriptions = {
            "requirement_analysis": f"Analyze requirements for: {analysis.primary_goal}",
            "architecture_design": "Design system architecture and component structure",
            "implementation": "Implement core functionality with production-grade code",
            "testing": "Create comprehensive test suite and validate functionality",
            "documentation": "Create documentation and usage examples",
            "integration": "Integrate with existing systems and validate",
            "bug_reproduction": "Reproduce and document the bug",
            "root_cause_analysis": "Identify root cause of the issue",
            "fix_implementation": "Implement fix with proper error handling",
            "regression_check": "Ensure fix doesn't introduce regressions",
            "code_analysis": "Analyze current code structure and identify improvements",
            "design_improvements": "Design better architecture and patterns",
            "incremental_refactor": "Refactor code incrementally with tests",
            "test_validation": "Validate all tests pass after refactoring",
            "performance_check": "Validate performance characteristics",
            "performance_profiling": "Profile current performance and identify bottlenecks",
            "bottleneck_identification": "Identify specific performance bottlenecks",
            "optimization_implementation": "Implement optimizations",
            "benchmark_validation": "Validate performance improvements with benchmarks"
        }

        return descriptions.get(phase, f"Execute {phase.replace('_', ' ')}")

    def _map_phase_to_task_type(self, phase: str) -> TaskType:
        """Map phase to task type"""
        phase_mapping = {
            "requirement_analysis": TaskType.ARCHITECTURE,
            "architecture_design": TaskType.ARCHITECTURE,
            "implementation": TaskType.CODE_GENERATION,
            "testing": TaskType.TESTING,
            "documentation": TaskType.DOCUMENTATION,
            "integration": TaskType.FEATURE,
            "bug_reproduction": TaskType.BUG_FIX,
            "root_cause_analysis": TaskType.BUG_FIX,
            "fix_implementation": TaskType.BUG_FIX,
            "regression_check": TaskType.TESTING,
            "code_analysis": TaskType.REFACTORING,
            "design_improvements": TaskType.ARCHITECTURE,
            "incremental_refactor": TaskType.REFACTORING,
            "test_validation": TaskType.TESTING,
            "performance_check": TaskType.OPTIMIZATION,
            "performance_profiling": TaskType.OPTIMIZATION,
            "bottleneck_identification": TaskType.OPTIMIZATION,
            "optimization_implementation": TaskType.OPTIMIZATION,
            "benchmark_validation": TaskType.TESTING
        }

        return phase_mapping.get(phase, TaskType.FEATURE)

    def _calculate_task_priority(self, task_index: int, total_tasks: int) -> TaskPriority:
        """Calculate task priority based on position"""
        if task_index == 0:
            return TaskPriority.CRITICAL  # First task is always critical
        elif task_index < total_tasks * 0.3:
            return TaskPriority.HIGH
        elif task_index < total_tasks * 0.7:
            return TaskPriority.MEDIUM
        else:
            return TaskPriority.LOW

    def _estimate_task_complexity(self, base_complexity: int, multiplier: float) -> int:
        """Estimate individual task complexity"""
        return min(10, max(1, int(base_complexity * multiplier / 2)))

    def _calculate_execution_order(self, tasks: List[TaskNode]) -> List[str]:
        """
        Calculate execution order using topological sort (Kahn's algorithm)
        """
        # Build adjacency list and in-degree count
        in_degree = {task.task_id: len(task.dependencies) for task in tasks}
        adjacency = {task.task_id: list(task.dependents) for task in tasks}

        # Find all nodes with no dependencies
        queue = [task.task_id for task in tasks if len(task.dependencies) == 0]
        execution_order = []

        while queue:
            # Sort by priority for deterministic order
            task_priorities = {
                task.task_id: (task.priority.value, task.task_id)
                for task in tasks if task.task_id in queue
            }
            queue.sort(key=lambda tid: task_priorities[tid])

            current = queue.pop(0)
            execution_order.append(current)

            # Reduce in-degree for dependent tasks
            for dependent in adjacency[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return execution_order

    def _estimate_duration(self, tasks: List[TaskNode], base_complexity: int) -> int:
        """Estimate total duration in minutes"""
        # Base time per complexity point
        base_minutes_per_point = 5

        total_complexity = sum(task.estimated_complexity for task in tasks)
        estimated_minutes = total_complexity * base_minutes_per_point

        # Add overhead for coordination
        coordination_overhead = len(tasks) * 2  # 2 minutes per task for setup

        return estimated_minutes + coordination_overhead

    def _calculate_confidence(
        self,
        analysis: RequirementAnalysis,
        tasks: List[TaskNode]
    ) -> float:
        """Calculate confidence score for the plan"""
        # Start with high confidence
        confidence = 0.9

        # Reduce based on complexity
        if analysis.complexity_score >= 8:
            confidence -= 0.1
        elif analysis.complexity_score >= 6:
            confidence -= 0.05

        # Reduce based on number of risks
        confidence -= len(analysis.risks) * 0.05

        # Reduce based on number of tasks
        if len(tasks) > 10:
            confidence -= 0.1
        elif len(tasks) > 6:
            confidence -= 0.05

        return max(0.5, min(1.0, confidence))

    async def get_next_executable_tasks(self, plan_id: str) -> List[TaskNode]:
        """Get all tasks that are ready to execute (dependencies satisfied)"""
        # In a real implementation, this would query stored tasks
        # For now, return empty list
        return []

    async def update_task_status(
        self,
        plan_id: str,
        task_id: str,
        status: TaskStatus,
        output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> bool:
        """Update task status and propagate changes"""
        # In a real implementation, this would update stored tasks
        return True

    def get_plan_progress(self, plan_id: str) -> Dict[str, Any]:
        """Get current progress of a plan"""
        if plan_id not in self.plans:
            return {"error": "Plan not found"}

        plan = self.plans[plan_id]

        # In a real implementation, would query actual task statuses
        return {
            "plan_id": plan_id,
            "requirement": plan.requirement,
            "estimated_duration_minutes": plan.estimated_duration_minutes,
            "confidence_score": plan.confidence_score,
            "created_at": plan.created_at.isoformat()
        }
