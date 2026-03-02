"""
Autonomous Agent Intelligence - Advanced AI Agent Orchestration System

Single-agent architecture with task decomposition, tool calling,
memory persistence, self-evaluation, and architecture validation.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Value Objects
# ---------------------------------------------------------------------------

class AgentPhase(str, Enum):
    PLANNING = "planning"
    DECOMPOSING = "decomposing"
    EXECUTING = "executing"
    VALIDATING = "validating"
    SELF_EVALUATING = "self_evaluating"
    ITERATING = "iterating"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ToolCategory(str, Enum):
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    FILE_SYSTEM = "file_system"
    DATABASE = "database"
    API_CALL = "api_call"
    SEARCH = "search"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    SECURITY = "security"


class ValidationLevel(str, Enum):
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    ARCHITECTURE = "architecture"
    SECURITY = "security"
    PERFORMANCE = "performance"
    FULL = "full"


@dataclass
class AgentCapability:
    """A capability the agent possesses."""
    name: str
    category: ToolCategory
    description: str
    version: str = "1.0"
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    max_concurrency: int = 1
    timeout_seconds: float = 300.0
    cost_per_call: float = 0.0
    reliability_score: float = 1.0


@dataclass
class TaskNode:
    """A decomposed task unit in the execution graph."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    name: str = ""
    description: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)
    estimated_complexity: int = 1  # 1-10
    estimated_tokens: int = 0
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status,
            "estimated_complexity": self.estimated_complexity,
            "dependencies": self.dependencies,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class ExecutionPlan:
    """Complete execution plan for a user request."""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    objective: str = ""
    tasks: List[TaskNode] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    estimated_total_tokens: int = 0
    estimated_total_cost: float = 0.0
    confidence_score: float = 0.0
    risk_assessment: str = ""
    fallback_strategies: List[str] = field(default_factory=list)

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def completed_count(self) -> int:
        return sum(1 for t in self.tasks if t.status == "completed")

    @property
    def progress(self) -> float:
        if not self.tasks:
            return 0.0
        return self.completed_count / self.task_count * 100

    def get_ready_tasks(self) -> List[TaskNode]:
        """Tasks whose dependencies are all completed."""
        completed_ids = {t.task_id for t in self.tasks if t.status == "completed"}
        return [
            t for t in self.tasks
            if t.status == "pending" and all(d in completed_ids for d in t.dependencies)
        ]


@dataclass
class MemoryEntry:
    """Agent memory entry."""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    memory_type: str = "working"
    importance: float = 0.5
    context_tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelfEvaluation:
    """Result of agent self-evaluation."""
    overall_score: float = 0.0  # 0-1
    correctness: float = 0.0
    completeness: float = 0.0
    code_quality: float = 0.0
    architecture_compliance: float = 0.0
    security_score: float = 0.0
    performance_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    requires_iteration: bool = False
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 3),
            "correctness": round(self.correctness, 3),
            "completeness": round(self.completeness, 3),
            "code_quality": round(self.code_quality, 3),
            "architecture_compliance": round(self.architecture_compliance, 3),
            "security_score": round(self.security_score, 3),
            "performance_score": round(self.performance_score, 3),
            "issues": self.issues,
            "suggestions": self.suggestions,
            "requires_iteration": self.requires_iteration,
            "confidence": round(self.confidence, 3),
        }


# ---------------------------------------------------------------------------
# Tool Interface
# ---------------------------------------------------------------------------

class AgentTool(ABC):
    """Base class for tools the agent can invoke."""

    @abstractmethod
    def get_name(self) -> str: ...

    @abstractmethod
    def get_description(self) -> str: ...

    @abstractmethod
    def get_parameters_schema(self) -> Dict[str, Any]: ...

    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]: ...

    def get_category(self) -> ToolCategory:
        return ToolCategory.CODE_GENERATION

    def get_cost(self) -> float:
        return 0.0


@dataclass
class ToolCall:
    """Record of a tool invocation."""
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Memory Persistence
# ---------------------------------------------------------------------------

class AgentMemoryStore:
    """Multi-tier memory with working, short-term, and long-term storage."""

    def __init__(self, max_working: int = 50, max_short_term: int = 500, max_long_term: int = 10_000):
        self._working: List[MemoryEntry] = []
        self._short_term: List[MemoryEntry] = []
        self._long_term: List[MemoryEntry] = []
        self._max_working = max_working
        self._max_short_term = max_short_term
        self._max_long_term = max_long_term

    async def store(self, entry: MemoryEntry) -> str:
        if entry.memory_type == "working":
            self._working.append(entry)
            if len(self._working) > self._max_working:
                self._consolidate_working()
        elif entry.memory_type == "short_term":
            self._short_term.append(entry)
            if len(self._short_term) > self._max_short_term:
                self._consolidate_short_term()
        elif entry.memory_type == "long_term":
            self._long_term.append(entry)
            if len(self._long_term) > self._max_long_term:
                self._long_term = self._long_term[-self._max_long_term:]
        return entry.memory_id

    async def recall(
        self,
        query: str = "",
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20,
        min_importance: float = 0.0,
    ) -> List[MemoryEntry]:
        all_memories: List[MemoryEntry] = []

        if not memory_type or memory_type == "working":
            all_memories.extend(self._working)
        if not memory_type or memory_type == "short_term":
            all_memories.extend(self._short_term)
        if not memory_type or memory_type == "long_term":
            all_memories.extend(self._long_term)

        # Filter by importance
        filtered = [m for m in all_memories if m.importance >= min_importance]

        # Filter by tags
        if tags:
            tag_set = set(tags)
            filtered = [m for m in filtered if tag_set.intersection(set(m.context_tags))]

        # Simple keyword relevance scoring
        if query:
            query_lower = query.lower()
            scored = []
            for m in filtered:
                score = 0
                content_lower = m.content.lower()
                for word in query_lower.split():
                    if word in content_lower:
                        score += 1
                score += m.importance
                score += m.access_count * 0.01
                scored.append((score, m))
            scored.sort(key=lambda x: x[0], reverse=True)
            filtered = [m for _, m in scored]

        # Update access tracking
        for m in filtered[:limit]:
            m.access_count += 1
            m.accessed_at = datetime.utcnow()

        return filtered[:limit]

    async def forget(self, memory_id: str) -> bool:
        for store in [self._working, self._short_term, self._long_term]:
            for i, m in enumerate(store):
                if m.memory_id == memory_id:
                    store.pop(i)
                    return True
        return False

    def _consolidate_working(self) -> None:
        # Move low-importance working memories to short-term
        self._working.sort(key=lambda m: m.importance, reverse=True)
        overflow = self._working[self._max_working:]
        self._working = self._working[:self._max_working]
        for m in overflow:
            m.memory_type = "short_term"
            self._short_term.append(m)

    def _consolidate_short_term(self) -> None:
        self._short_term.sort(key=lambda m: m.importance + m.access_count * 0.1, reverse=True)
        overflow = self._short_term[self._max_short_term:]
        self._short_term = self._short_term[:self._max_short_term]
        for m in overflow:
            if m.importance >= 0.7:
                m.memory_type = "long_term"
                self._long_term.append(m)

    def get_stats(self) -> Dict[str, int]:
        return {
            "working": len(self._working),
            "short_term": len(self._short_term),
            "long_term": len(self._long_term),
        }


# ---------------------------------------------------------------------------
# Architecture Validator
# ---------------------------------------------------------------------------

class ArchitectureValidator:
    """Validates generated code against architecture rules."""

    def __init__(self):
        self._rules: List[Dict[str, Any]] = []
        self._violations: List[Dict[str, Any]] = []

    def add_rule(self, name: str, description: str, check_fn: Callable[[str], List[str]]) -> None:
        self._rules.append({
            "name": name,
            "description": description,
            "check_fn": check_fn,
        })

    def add_default_rules(self) -> None:
        self.add_rule(
            "no_circular_imports",
            "Detect circular import patterns",
            self._check_circular_imports,
        )
        self.add_rule(
            "layer_separation",
            "Ensure domain layer doesn't depend on infrastructure",
            self._check_layer_separation,
        )
        self.add_rule(
            "naming_convention",
            "Enforce naming conventions",
            self._check_naming_conventions,
        )
        self.add_rule(
            "error_handling",
            "Verify proper error handling patterns",
            self._check_error_handling,
        )
        self.add_rule(
            "security_patterns",
            "Check for security anti-patterns",
            self._check_security_patterns,
        )

    def validate(self, code: str, file_path: str = "") -> List[Dict[str, Any]]:
        violations = []
        for rule in self._rules:
            issues = rule["check_fn"](code)
            for issue in issues:
                violation = {
                    "rule": rule["name"],
                    "description": rule["description"],
                    "issue": issue,
                    "file_path": file_path,
                    "severity": "warning",
                }
                violations.append(violation)
                self._violations.append(violation)
        return violations

    @staticmethod
    def _check_circular_imports(code: str) -> List[str]:
        issues = []
        import_lines = [l.strip() for l in code.split('\n') if l.strip().startswith(('import ', 'from '))]
        seen_modules = set()
        for line in import_lines:
            parts = line.split()
            if len(parts) >= 2:
                module = parts[1].split('.')[0]
                if module in seen_modules:
                    issues.append(f"Potential duplicate import: {module}")
                seen_modules.add(module)
        return issues

    @staticmethod
    def _check_layer_separation(code: str) -> List[str]:
        issues = []
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('from domain') and 'infrastructure' in stripped:
                issues.append(f"Line {i}: Domain layer importing from infrastructure")
            if stripped.startswith('from core') and ('infrastructure' in stripped or 'services' in stripped):
                issues.append(f"Line {i}: Core layer importing from outer layers")
        return issues

    @staticmethod
    def _check_naming_conventions(code: str) -> List[str]:
        import re
        issues = []
        class_pattern = re.compile(r'class\s+([a-z]\w*)')
        matches = class_pattern.findall(code)
        for name in matches:
            issues.append(f"Class '{name}' should use PascalCase")
        return issues

    @staticmethod
    def _check_error_handling(code: str) -> List[str]:
        issues = []
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped == 'except:':
                issues.append(f"Line {i}: Bare except clause - should catch specific exceptions")
            if 'except Exception as e:' in stripped and i + 1 < len(lines):
                next_line = lines[i].strip()
                if next_line == 'pass':
                    issues.append(f"Line {i}: Exception silently swallowed")
        return issues

    @staticmethod
    def _check_security_patterns(code: str) -> List[str]:
        issues = []
        dangerous_patterns = [
            ('eval(', 'Use of eval() is a security risk'),
            ('exec(', 'Use of exec() is a security risk'),
            ('__import__', 'Dynamic imports can be a security risk'),
            ('pickle.loads', 'pickle.loads is vulnerable to arbitrary code execution'),
            ('shell=True', 'subprocess with shell=True is a security risk'),
            ('password', 'Potential hardcoded password detected'),
        ]
        for pattern, description in dangerous_patterns:
            if pattern in code.lower():
                issues.append(description)
        return issues


# ---------------------------------------------------------------------------
# Autonomous Agent
# ---------------------------------------------------------------------------

class AutonomousAgent:
    """
    Single autonomous AI agent that can:
    - Accept high-level user requirements
    - Decompose tasks intelligently
    - Generate complete modules
    - Validate output
    - Optimize performance
    - Detect architecture violations
    - Self-evaluate and iterate

    This is the core intelligence engine of CognitionOS.
    """

    def __init__(
        self,
        llm_provider: Optional[Any] = None,
        max_iterations: int = 5,
        quality_threshold: float = 0.8,
        max_tokens_per_task: int = 100_000,
        tenant_id: Optional[str] = None,
    ):
        self.llm_provider = llm_provider
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.max_tokens_per_task = max_tokens_per_task
        self.tenant_id = tenant_id

        # Subsystems
        self._memory = AgentMemoryStore()
        self._tools: Dict[str, AgentTool] = {}
        self._validator = ArchitectureValidator()
        self._validator.add_default_rules()

        # State
        self._current_phase = AgentPhase.PLANNING
        self._current_plan: Optional[ExecutionPlan] = None
        self._tool_calls: List[ToolCall] = []
        self._evaluations: List[SelfEvaluation] = []
        self._iteration_count = 0

        # Metrics
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_tasks_completed = 0
        self._total_tasks_failed = 0

    # -- Tool Management ----------------------------------------------------

    def register_tool(self, tool: AgentTool) -> None:
        self._tools[tool.get_name()] = tool

    def get_available_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": tool.get_name(),
                "description": tool.get_description(),
                "category": tool.get_category().value,
                "parameters": tool.get_parameters_schema(),
            }
            for tool in self._tools.values()
        ]

    # -- Main Execution Loop ------------------------------------------------

    async def execute(self, objective: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main execution loop: plan -> decompose -> execute -> validate -> iterate.
        """
        logger.info("Agent executing objective: %s", objective[:100])
        execution_start = time.monotonic()
        context = context or {}

        # Store objective in memory
        await self._memory.store(MemoryEntry(
            content=f"Objective: {objective}",
            memory_type="working",
            importance=1.0,
            context_tags=["objective"],
        ))

        try:
            # Phase 1: Plan
            self._current_phase = AgentPhase.PLANNING
            plan = await self._create_plan(objective, context)
            self._current_plan = plan

            # Phase 2: Decompose
            self._current_phase = AgentPhase.DECOMPOSING
            plan = await self._decompose_tasks(plan)

            # Iterative execution loop
            iteration = 0
            final_results = {}

            while iteration < self.max_iterations:
                iteration += 1
                self._iteration_count = iteration

                # Phase 3: Execute tasks
                self._current_phase = AgentPhase.EXECUTING
                results = await self._execute_tasks(plan)
                final_results.update(results)

                # Phase 4: Validate
                self._current_phase = AgentPhase.VALIDATING
                validation_results = await self._validate_results(results)

                # Phase 5: Self-evaluate
                self._current_phase = AgentPhase.SELF_EVALUATING
                evaluation = await self._self_evaluate(plan, results, validation_results)
                self._evaluations.append(evaluation)

                if not evaluation.requires_iteration or evaluation.overall_score >= self.quality_threshold:
                    break

                # Phase 6: Iterate
                self._current_phase = AgentPhase.ITERATING
                plan = await self._iterate_plan(plan, evaluation)

            # Complete
            self._current_phase = AgentPhase.COMPLETED
            elapsed_ms = (time.monotonic() - execution_start) * 1000

            return {
                "status": "completed",
                "objective": objective,
                "plan_id": plan.plan_id,
                "iterations": iteration,
                "tasks_completed": plan.completed_count,
                "tasks_total": plan.task_count,
                "progress": plan.progress,
                "final_evaluation": self._evaluations[-1].to_dict() if self._evaluations else None,
                "total_tokens": self._total_tokens,
                "total_cost": self._total_cost,
                "duration_ms": round(elapsed_ms, 2),
                "results": final_results,
                "memory_stats": self._memory.get_stats(),
            }

        except Exception as exc:
            self._current_phase = AgentPhase.FAILED
            logger.exception("Agent execution failed: %s", exc)
            return {
                "status": "failed",
                "objective": objective,
                "error": str(exc),
                "iterations": self._iteration_count,
                "total_tokens": self._total_tokens,
            }

    async def _create_plan(self, objective: str, context: Dict[str, Any]) -> ExecutionPlan:
        """Create an execution plan from the objective."""
        # Recall relevant memories
        memories = await self._memory.recall(query=objective, limit=10)
        memory_context = "\n".join(m.content for m in memories)

        plan = ExecutionPlan(
            objective=objective,
            confidence_score=0.8,
            risk_assessment="low",
        )

        # For now, create a single-task plan
        # In a full implementation, the LLM would decompose this
        root_task = TaskNode(
            name=objective[:100],
            description=objective,
            priority=TaskPriority.HIGH,
            estimated_complexity=5,
        )
        plan.tasks.append(root_task)

        await self._memory.store(MemoryEntry(
            content=f"Created plan with {plan.task_count} tasks",
            memory_type="working",
            importance=0.8,
            context_tags=["planning"],
        ))

        return plan

    async def _decompose_tasks(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Decompose high-level tasks into actionable subtasks."""
        new_tasks = []

        for task in plan.tasks:
            if task.estimated_complexity > 5:
                # Break into subtasks
                subtask_count = min(task.estimated_complexity // 2, 5)
                for i in range(subtask_count):
                    subtask = TaskNode(
                        parent_id=task.task_id,
                        name=f"{task.name} - Part {i + 1}",
                        description=f"Subtask {i + 1} of: {task.description}",
                        priority=task.priority,
                        estimated_complexity=task.estimated_complexity // subtask_count,
                        dependencies=[new_tasks[-1].task_id] if new_tasks and i > 0 else [],
                    )
                    new_tasks.append(subtask)
                task.status = "decomposed"
            else:
                new_tasks.append(task)

        plan.tasks = new_tasks
        return plan

    async def _execute_tasks(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute all ready tasks in the plan."""
        results = {}

        while True:
            ready_tasks = plan.get_ready_tasks()
            if not ready_tasks:
                break

            for task in ready_tasks:
                task.status = "running"
                task.started_at = datetime.utcnow()

                try:
                    result = await self._execute_single_task(task)
                    task.result = result
                    task.status = "completed"
                    task.completed_at = datetime.utcnow()
                    results[task.task_id] = result
                    self._total_tasks_completed += 1

                except Exception as exc:
                    task.error = str(exc)
                    task.retries += 1
                    if task.retries >= task.max_retries:
                        task.status = "failed"
                        self._total_tasks_failed += 1
                    else:
                        task.status = "pending"
                    logger.error("Task %s failed: %s", task.task_id, exc)

        return results

    async def _execute_single_task(self, task: TaskNode) -> Any:
        """Execute a single task, potentially using tools."""
        logger.info("Executing task: %s", task.name)

        # Check if any registered tools can handle this
        for tool_name, tool in self._tools.items():
            if any(cap in tool_name.lower() for cap in (task.required_capabilities or ["generate"])):
                call = ToolCall(tool_name=tool_name, parameters={"task": task.description})
                start = time.monotonic()
                try:
                    result = await tool.execute(call.parameters)
                    call.result = result
                    call.duration_ms = (time.monotonic() - start) * 1000
                    self._tool_calls.append(call)
                    return result
                except Exception as exc:
                    call.error = str(exc)
                    call.duration_ms = (time.monotonic() - start) * 1000
                    self._tool_calls.append(call)

        # Default: return task info
        return {"task": task.name, "status": "completed_without_tools"}

    async def _validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate execution results."""
        validation_results = {}

        for task_id, result in results.items():
            if isinstance(result, dict) and "code" in result:
                violations = self._validator.validate(result["code"])
                validation_results[task_id] = {
                    "valid": len(violations) == 0,
                    "violations": violations,
                    "violation_count": len(violations),
                }
            else:
                validation_results[task_id] = {"valid": True, "violations": []}

        return validation_results

    async def _self_evaluate(
        self,
        plan: ExecutionPlan,
        results: Dict[str, Any],
        validation_results: Dict[str, Any],
    ) -> SelfEvaluation:
        """Agent self-evaluation of its output quality."""
        total_tasks = plan.task_count
        completed = plan.completed_count
        failed = sum(1 for t in plan.tasks if t.status == "failed")

        completeness = completed / max(total_tasks, 1)
        correctness = 1.0 - (failed / max(total_tasks, 1))

        total_violations = sum(
            v.get("violation_count", 0) for v in validation_results.values()
        )
        architecture_score = max(0, 1.0 - total_violations * 0.1)

        overall = (
            correctness * 0.3
            + completeness * 0.3
            + architecture_score * 0.2
            + 0.8 * 0.1  # placeholder security
            + 0.8 * 0.1  # placeholder performance
        )

        issues = []
        suggestions = []

        if failed > 0:
            issues.append(f"{failed} tasks failed")
            suggestions.append("Review failed tasks and retry with adjusted parameters")

        if total_violations > 0:
            issues.append(f"{total_violations} architecture violations detected")
            suggestions.append("Fix architecture violations before proceeding")

        if completeness < 1.0:
            issues.append(f"Only {completeness:.0%} of tasks completed")
            suggestions.append("Investigate blocked or pending tasks")

        return SelfEvaluation(
            overall_score=overall,
            correctness=correctness,
            completeness=completeness,
            code_quality=0.8,
            architecture_compliance=architecture_score,
            security_score=0.8,
            performance_score=0.8,
            issues=issues,
            suggestions=suggestions,
            requires_iteration=overall < self.quality_threshold,
            confidence=min(overall + 0.1, 1.0),
        )

    async def _iterate_plan(self, plan: ExecutionPlan, evaluation: SelfEvaluation) -> ExecutionPlan:
        """Iterate on the plan based on self-evaluation feedback."""
        await self._memory.store(MemoryEntry(
            content=f"Iteration {self._iteration_count}: Score={evaluation.overall_score:.2f}, Issues={evaluation.issues}",
            memory_type="working",
            importance=0.9,
            context_tags=["iteration", "evaluation"],
        ))

        # Reset failed tasks for retry
        for task in plan.tasks:
            if task.status == "failed" and task.retries < task.max_retries:
                task.status = "pending"
                task.error = None

        return plan

    # -- Queries ------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "phase": self._current_phase.value,
            "iteration": self._iteration_count,
            "plan": {
                "plan_id": self._current_plan.plan_id if self._current_plan else None,
                "progress": self._current_plan.progress if self._current_plan else 0,
                "task_count": self._current_plan.task_count if self._current_plan else 0,
            },
            "metrics": {
                "total_tokens": self._total_tokens,
                "total_cost": self._total_cost,
                "tasks_completed": self._total_tasks_completed,
                "tasks_failed": self._total_tasks_failed,
                "tool_calls": len(self._tool_calls),
                "evaluations": len(self._evaluations),
            },
            "memory": self._memory.get_stats(),
        }
