"""
Agent Execution Engine

Orchestrates multi-step autonomous agent execution:
- Sequential and parallel step execution
- Tool call management with retry logic
- Memory integration (stores observations and results)
- Budget and safety enforcement
- Streaming progress events
- Full audit trail
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

from .tool_registry import ToolRegistry, ToolExecutionResult
from .vector_memory import VectorMemoryStore, MemoryTier, MemoryType


class StepType(str, Enum):
    THINK = "think"           # Internal reasoning step (no tool)
    TOOL_CALL = "tool_call"   # Execute a registered tool
    LLM_CALL = "llm_call"     # Call an LLM provider
    PARALLEL = "parallel"     # Run multiple steps concurrently
    BRANCH = "branch"         # Conditional branching
    CHECKPOINT = "checkpoint" # Persist state


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class ExecutionStep:
    """Defines a single step in an agent's execution plan"""
    step_id: str
    name: str
    step_type: StepType
    payload: Dict[str, Any]          # Type-specific configuration
    depends_on: List[str] = field(default_factory=list)  # step_ids
    max_retries: int = 2
    retry_delay_s: float = 1.0
    timeout_s: float = 60.0
    on_failure: str = "abort"        # "abort" | "continue" | "skip"
    condition: Optional[str] = None  # Python expression evaluated against context


@dataclass
class StepResult:
    """Result of executing a single step"""
    step_id: str
    name: str
    status: StepStatus
    output: Any
    error: Optional[str] = None
    started_at: float = 0.0
    completed_at: float = 0.0
    attempt: int = 1
    tool_result: Optional[ToolExecutionResult] = None

    @property
    def duration_ms(self) -> float:
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at) * 1000
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "duration_ms": round(self.duration_ms, 2),
            "attempt": self.attempt,
        }


@dataclass
class ExecutionContext:
    """Mutable context shared across all steps in a single execution"""
    execution_id: str
    agent_id: str
    goal: str
    variables: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    started_at: float = field(default_factory=time.time)
    budget_usd: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.variables[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def add_cost(self, cost_usd: float, tokens: int = 0) -> None:
        self.total_cost_usd += cost_usd
        self.total_tokens += tokens

    def is_over_budget(self) -> bool:
        if self.budget_usd is None:
            return False
        return self.total_cost_usd >= self.budget_usd

    def evaluate_condition(self, expression: str) -> bool:
        """
        Safely evaluate a boolean expression in the context of variables.
        Uses ast.literal_eval for safe constant expressions and a restricted
        operator-only evaluator for comparisons to prevent code injection.
        Only supports: variable references, comparisons (==, !=, <, <=, >, >=),
        boolean operators (and, or, not), and literal values.
        """
        import ast
        import operator as op

        _SAFE_OPS = {
            ast.Eq: op.eq,
            ast.NotEq: op.ne,
            ast.Lt: op.lt,
            ast.LtE: op.le,
            ast.Gt: op.gt,
            ast.GtE: op.ge,
            ast.And: lambda a, b: a and b,
            ast.Or: lambda a, b: a or b,
            ast.Not: op.not_,
            ast.Is: op.is_,
            ast.IsNot: op.is_not,
            ast.In: lambda a, b: a in b,
            ast.NotIn: lambda a, b: a not in b,
        }

        def _eval(node: ast.AST) -> Any:
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name):
                if node.id == "True":
                    return True
                if node.id == "False":
                    return False
                if node.id == "None":
                    return None
                if node.id in self.variables:
                    return self.variables[node.id]
                raise ValueError(f"Unknown variable '{node.id}'")
            if isinstance(node, ast.BoolOp):
                values = [_eval(v) for v in node.values]
                combine = _SAFE_OPS[type(node.op)]
                result = values[0]
                for v in values[1:]:
                    result = combine(result, v)
                return result
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                return not _eval(node.operand)
            if isinstance(node, ast.Compare):
                left = _eval(node.left)
                for comparator_op, comparator in zip(node.ops, node.comparators):
                    right = _eval(comparator)
                    if type(comparator_op) not in _SAFE_OPS:
                        raise ValueError(f"Unsupported operator: {type(comparator_op).__name__}")
                    if not _SAFE_OPS[type(comparator_op)](left, right):
                        return False
                    left = right
                return True
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")

        try:
            tree = ast.parse(expression, mode="eval")
            return bool(_eval(tree.body))
        except Exception:  # noqa: BLE001
            return False

    def summary(self) -> Dict[str, Any]:
        completed = sum(
            1 for r in self.step_results.values() if r.status == StepStatus.SUCCESS
        )
        failed = sum(
            1 for r in self.step_results.values() if r.status == StepStatus.FAILED
        )
        return {
            "execution_id": self.execution_id,
            "agent_id": self.agent_id,
            "goal": self.goal,
            "total_steps": len(self.step_results),
            "completed_steps": completed,
            "failed_steps": failed,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_tokens": self.total_tokens,
            "duration_s": round(time.time() - self.started_at, 2),
        }


class ProgressEvent:
    """Streaming progress event from the execution engine"""
    __slots__ = ("event_type", "execution_id", "step_id", "data", "timestamp")

    def __init__(
        self,
        event_type: str,
        execution_id: str,
        data: Dict[str, Any],
        step_id: Optional[str] = None,
    ) -> None:
        self.event_type = event_type
        self.execution_id = execution_id
        self.step_id = step_id
        self.data = data
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "execution_id": self.execution_id,
            "step_id": self.step_id,
            "timestamp": self.timestamp,
            **self.data,
        }


class AgentExecutionEngine:
    """
    Executes a sequence of agent steps with full orchestration.

    Features:
    - Sequential + parallel step execution
    - Dependency resolution (topological sort)
    - Automatic retry with exponential backoff
    - Memory integration (episodic store)
    - Budget enforcement
    - Streaming progress events via async queue
    - Safety boundaries (blocked tool names, max iterations)
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        memory_store: Optional[VectorMemoryStore] = None,
        max_steps: int = 100,
        blocked_tools: Optional[List[str]] = None,
    ) -> None:
        self._tools = tool_registry or ToolRegistry()
        self._memory = memory_store or VectorMemoryStore()
        self._max_steps = max_steps
        self._blocked_tools = set(blocked_tools or [])
        self._event_queue: asyncio.Queue = asyncio.Queue()

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    async def execute(
        self,
        steps: List[ExecutionStep],
        goal: str,
        agent_id: str,
        variables: Optional[Dict[str, Any]] = None,
        budget_usd: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionContext:
        """
        Execute a plan of steps and return the final context.
        Steps are run in dependency order.
        """
        ctx = ExecutionContext(
            execution_id=str(uuid4()),
            agent_id=agent_id,
            goal=goal,
            variables=variables or {},
            budget_usd=budget_usd,
            metadata=metadata or {},
        )

        await self._emit(ProgressEvent(
            "execution_started",
            ctx.execution_id,
            {"goal": goal, "total_steps": len(steps)},
        ))

        self._memory.store(
            f"Starting execution: {goal}",
            memory_type=MemoryType.PLAN,
            tier=MemoryTier.WORKING,
            importance=0.8,
            agent_id=agent_id,
        )

        ordered = self._topological_sort(steps)

        if len(ordered) > self._max_steps:
            ordered = ordered[: self._max_steps]

        for step in ordered:
            if ctx.is_over_budget():
                await self._emit(ProgressEvent(
                    "budget_exceeded",
                    ctx.execution_id,
                    {"cost_usd": ctx.total_cost_usd, "budget_usd": budget_usd},
                    step_id=step.step_id,
                ))
                break

            if step.condition and not ctx.evaluate_condition(step.condition):
                result = StepResult(
                    step_id=step.step_id,
                    name=step.name,
                    status=StepStatus.SKIPPED,
                    output=None,
                    started_at=time.time(),
                    completed_at=time.time(),
                )
                ctx.step_results[step.step_id] = result
                continue

            if not self._dependencies_met(step, ctx):
                result = StepResult(
                    step_id=step.step_id,
                    name=step.name,
                    status=StepStatus.SKIPPED,
                    output=None,
                    error="Dependencies not met",
                    started_at=time.time(),
                    completed_at=time.time(),
                )
                ctx.step_results[step.step_id] = result
                continue

            result = await self._execute_step_with_retry(step, ctx)
            ctx.step_results[step.step_id] = result

            if result.status == StepStatus.SUCCESS and result.output is not None:
                ctx.set(f"step_{step.step_id}_output", result.output)
                self._memory.store(
                    f"Step '{step.name}' completed: {str(result.output)[:200]}",
                    memory_type=MemoryType.RESULT,
                    tier=MemoryTier.EPISODIC,
                    importance=0.6,
                    agent_id=agent_id,
                    session_id=ctx.execution_id,
                )

            if result.status == StepStatus.FAILED and step.on_failure == "abort":
                await self._emit(ProgressEvent(
                    "execution_aborted",
                    ctx.execution_id,
                    {"failed_step": step.step_id, "error": result.error},
                ))
                break

        await self._emit(ProgressEvent(
            "execution_completed",
            ctx.execution_id,
            ctx.summary(),
        ))
        return ctx

    async def stream_events(self) -> AsyncIterator[ProgressEvent]:
        """Async generator that yields progress events as they occur"""
        while True:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=30.0)
                yield event
                if event.event_type in ("execution_completed", "execution_aborted"):
                    break
            except asyncio.TimeoutError:
                break

    def get_memory_context(
        self,
        query: str,
        agent_id: str,
        max_tokens: int = 1000,
    ) -> str:
        """Build memory context string for LLM injection"""
        return self._memory.build_context(query, max_tokens=max_tokens, agent_id=agent_id)

    # ──────────────────────────────────────────────
    # Step execution
    # ──────────────────────────────────────────────

    async def _execute_step_with_retry(
        self,
        step: ExecutionStep,
        ctx: ExecutionContext,
    ) -> StepResult:
        attempt = 0
        delay = step.retry_delay_s

        while attempt <= step.max_retries:
            attempt += 1
            await self._emit(ProgressEvent(
                "step_started",
                ctx.execution_id,
                {"name": step.name, "type": step.step_type.value, "attempt": attempt},
                step_id=step.step_id,
            ))

            result = await self._execute_step(step, ctx, attempt)

            if result.status == StepStatus.SUCCESS:
                await self._emit(ProgressEvent(
                    "step_completed",
                    ctx.execution_id,
                    result.to_dict(),
                    step_id=step.step_id,
                ))
                return result

            if attempt <= step.max_retries:
                await self._emit(ProgressEvent(
                    "step_retrying",
                    ctx.execution_id,
                    {"attempt": attempt, "delay_s": delay, "error": result.error},
                    step_id=step.step_id,
                ))
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)  # exponential backoff, cap at 30s
            else:
                await self._emit(ProgressEvent(
                    "step_failed",
                    ctx.execution_id,
                    result.to_dict(),
                    step_id=step.step_id,
                ))

        return result  # type: ignore[return-value]

    async def _execute_step(
        self,
        step: ExecutionStep,
        ctx: ExecutionContext,
        attempt: int,
    ) -> StepResult:
        started_at = time.time()

        try:
            if step.step_type == StepType.TOOL_CALL:
                output = await self._run_tool_call(step, ctx)
            elif step.step_type == StepType.THINK:
                output = step.payload.get("reasoning", "")
            elif step.step_type == StepType.PARALLEL:
                output = await self._run_parallel(step, ctx)
            elif step.step_type == StepType.LLM_CALL:
                # Placeholder – real impl injects LLM router
                output = step.payload.get("static_response", "LLM call placeholder")
            elif step.step_type == StepType.CHECKPOINT:
                output = ctx.summary()
            else:
                output = step.payload

            return StepResult(
                step_id=step.step_id,
                name=step.name,
                status=StepStatus.SUCCESS,
                output=output,
                started_at=started_at,
                completed_at=time.time(),
                attempt=attempt,
            )
        except Exception as exc:  # noqa: BLE001
            return StepResult(
                step_id=step.step_id,
                name=step.name,
                status=StepStatus.FAILED,
                output=None,
                error=str(exc),
                started_at=started_at,
                completed_at=time.time(),
                attempt=attempt,
            )

    async def _run_tool_call(
        self,
        step: ExecutionStep,
        ctx: ExecutionContext,
    ) -> Any:
        tool_name = step.payload.get("tool_name", "")
        if tool_name in self._blocked_tools:
            raise ToolBlockedError(f"Tool '{tool_name}' is blocked by safety policy")

        inputs = step.payload.get("inputs", {})
        # Interpolate variables from context
        resolved_inputs = self._resolve_variables(inputs, ctx)

        result = await asyncio.wait_for(
            self._tools.execute(tool_name, resolved_inputs, context=ctx.variables),
            timeout=step.timeout_s,
        )
        if not result.success:
            raise ToolExecutionFailed(result.error or "Tool execution failed")
        return result.output

    async def _run_parallel(
        self,
        step: ExecutionStep,
        ctx: ExecutionContext,
    ) -> List[Any]:
        sub_steps_data = step.payload.get("steps", [])
        sub_steps = [
            ExecutionStep(
                step_id=s.get("step_id", str(uuid4())),
                name=s.get("name", ""),
                step_type=StepType(s.get("step_type", StepType.TOOL_CALL.value)),
                payload=s.get("payload", {}),
                max_retries=s.get("max_retries", 1),
                timeout_s=s.get("timeout_s", step.timeout_s),
            )
            for s in sub_steps_data
        ]
        tasks = [self._execute_step(s, ctx, 1) for s in sub_steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r.output if isinstance(r, StepResult) else str(r) for r in results]

    # ──────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────

    def _topological_sort(self, steps: List[ExecutionStep]) -> List[ExecutionStep]:
        """Return steps in dependency order using Kahn's algorithm"""
        step_map = {s.step_id: s for s in steps}
        in_degree: Dict[str, int] = {s.step_id: 0 for s in steps}

        for step in steps:
            for dep in step.depends_on:
                if dep in in_degree:
                    in_degree[step.step_id] += 1

        queue = [s for s in steps if in_degree[s.step_id] == 0]
        ordered: List[ExecutionStep] = []

        while queue:
            current = queue.pop(0)
            ordered.append(current)
            for step in steps:
                if current.step_id in step.depends_on:
                    in_degree[step.step_id] -= 1
                    if in_degree[step.step_id] == 0:
                        queue.append(step)

        # Append any remaining (handles cycles gracefully)
        remaining_ids = {s.step_id for s in ordered}
        for step in steps:
            if step.step_id not in remaining_ids:
                ordered.append(step)

        return ordered

    def _dependencies_met(self, step: ExecutionStep, ctx: ExecutionContext) -> bool:
        for dep_id in step.depends_on:
            result = ctx.step_results.get(dep_id)
            if result is None or result.status == StepStatus.FAILED:
                return False
        return True

    def _resolve_variables(
        self,
        inputs: Dict[str, Any],
        ctx: ExecutionContext,
    ) -> Dict[str, Any]:
        """Replace {{variable_name}} placeholders with context values"""
        resolved: Dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(value, str) and "{{" in value:
                for var_name, var_value in ctx.variables.items():
                    placeholder = f"{{{{{var_name}}}}}"
                    if placeholder in value:
                        value = value.replace(placeholder, str(var_value))
            resolved[key] = value
        return resolved

    async def _emit(self, event: ProgressEvent) -> None:
        await self._event_queue.put(event)


class ToolBlockedError(RuntimeError):
    """Raised when a tool is in the blocked list"""


class ToolExecutionFailed(RuntimeError):
    """Raised when a tool execution returns an error"""
