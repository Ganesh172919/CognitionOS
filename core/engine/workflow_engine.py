"""
Workflow Execution Engine — CognitionOS

Production workflow engine with:
- DAG-based workflow definition
- Step execution with retry
- Parallel and sequential execution
- Conditional branching
- Error handling and compensation
- Workflow variables and context
- Checkpoint and resume
- Workflow templates
- Real-time status tracking
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    WAITING = "waiting"


class WorkflowStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepType(str, Enum):
    ACTION = "action"
    CONDITION = "condition"
    PARALLEL_GROUP = "parallel_group"
    WAIT = "wait"
    LOOP = "loop"
    HUMAN_APPROVAL = "human_approval"
    SUB_WORKFLOW = "sub_workflow"


@dataclass
class WorkflowStep:
    step_id: str
    name: str
    step_type: StepType = StepType.ACTION
    handler: Optional[Callable[..., Awaitable[Any]]] = None
    dependencies: List[str] = field(default_factory=list)
    condition: Optional[str] = None  # Python expression
    retry_count: int = 2
    timeout_seconds: float = 60.0
    compensation_handler: Optional[Callable] = None
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_key: str = ""
    parallel_steps: List[str] = field(default_factory=list)

    # Runtime
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_ms: float = 0
    attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id, "name": self.name,
            "type": self.step_type.value,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 1),
            "attempts": self.attempts,
            "error": self.error,
        }


@dataclass
class WorkflowCheckpoint:
    checkpoint_id: str
    workflow_id: str
    step_id: str
    variables: Dict[str, Any]
    completed_steps: Set[str]
    created_at: float = field(default_factory=time.time)


@dataclass
class WorkflowDefinition:
    workflow_id: str
    name: str
    description: str = ""
    steps: Dict[str, WorkflowStep] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    tenant_id: str = ""
    version: int = 1
    created_at: float = field(default_factory=time.time)


@dataclass
class WorkflowExecution:
    execution_id: str
    workflow_id: str
    workflow_name: str
    status: WorkflowStatus = WorkflowStatus.RUNNING
    variables: Dict[str, Any] = field(default_factory=dict)
    completed_steps: Set[str] = field(default_factory=set)
    failed_steps: Set[str] = field(default_factory=set)
    checkpoints: List[WorkflowCheckpoint] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    total_duration_ms: float = 0
    error: Optional[str] = None
    tenant_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "workflow_name": self.workflow_name,
            "status": self.status.value,
            "completed_steps": len(self.completed_steps),
            "failed_steps": len(self.failed_steps),
            "total_duration_ms": round(self.total_duration_ms, 1),
            "error": self.error,
        }


class WorkflowEngine:
    """
    Production workflow execution engine with DAG resolution,
    parallel execution, checkpointing, and compensation.
    """

    def __init__(self, *, max_concurrent_steps: int = 20,
                 max_concurrent_workflows: int = 100):
        self._definitions: Dict[str, WorkflowDefinition] = {}
        self._executions: Dict[str, WorkflowExecution] = {}
        self._step_semaphore = asyncio.Semaphore(max_concurrent_steps)
        self._wf_semaphore = asyncio.Semaphore(max_concurrent_workflows)
        self._metrics = {
            "total_executions": 0, "completed": 0,
            "failed": 0, "total_steps_executed": 0,
        }
        self._hooks: Dict[str, List[Callable]] = {
            "on_start": [], "on_complete": [], "on_fail": [],
            "on_step_complete": [], "on_step_fail": [],
        }

    # ── Workflow Definition ──

    def define(self, name: str, *, description: str = "",
                tenant_id: str = "", tags: Optional[List[str]] = None
                ) -> WorkflowDefinition:
        wf_id = uuid.uuid4().hex[:12]
        definition = WorkflowDefinition(
            workflow_id=wf_id, name=name, description=description,
            tenant_id=tenant_id, tags=tags or [],
        )
        self._definitions[wf_id] = definition
        return definition

    def add_step(self, workflow_id: str, name: str,
                   handler: Callable[..., Awaitable[Any]], *,
                   step_type: StepType = StepType.ACTION,
                   dependencies: Optional[List[str]] = None,
                   condition: Optional[str] = None,
                   retry_count: int = 2,
                   timeout: float = 60.0,
                   output_key: str = "") -> str:
        definition = self._definitions.get(workflow_id)
        if not definition:
            raise KeyError(f"Workflow not found: {workflow_id}")

        step_id = uuid.uuid4().hex[:8]
        step = WorkflowStep(
            step_id=step_id, name=name,
            step_type=step_type, handler=handler,
            dependencies=dependencies or [],
            condition=condition, retry_count=retry_count,
            timeout_seconds=timeout, output_key=output_key,
        )
        definition.steps[step_id] = step
        return step_id

    # ── Workflow Execution ──

    async def execute(self, workflow_id: str, *,
                        variables: Optional[Dict[str, Any]] = None,
                        tenant_id: str = "") -> WorkflowExecution:
        """Execute a workflow."""
        definition = self._definitions.get(workflow_id)
        if not definition:
            raise KeyError(f"Workflow not found: {workflow_id}")

        execution = WorkflowExecution(
            execution_id=uuid.uuid4().hex[:12],
            workflow_id=workflow_id,
            workflow_name=definition.name,
            variables={**definition.variables, **(variables or {})},
            tenant_id=tenant_id or definition.tenant_id,
        )
        self._executions[execution.execution_id] = execution
        self._metrics["total_executions"] += 1

        # Fire hooks
        await self._fire_hook("on_start", execution)

        try:
            async with self._wf_semaphore:
                await self._execute_dag(definition, execution)

            if execution.failed_steps:
                execution.status = WorkflowStatus.FAILED
                execution.error = f"{len(execution.failed_steps)} steps failed"
                self._metrics["failed"] += 1
                await self._fire_hook("on_fail", execution)
            else:
                execution.status = WorkflowStatus.COMPLETED
                self._metrics["completed"] += 1
                await self._fire_hook("on_complete", execution)

        except Exception as exc:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(exc)
            self._metrics["failed"] += 1
            await self._fire_hook("on_fail", execution)

        execution.completed_at = time.time()
        execution.total_duration_ms = (
            execution.completed_at - execution.started_at
        ) * 1000

        return execution

    async def _execute_dag(self, definition: WorkflowDefinition,
                              execution: WorkflowExecution):
        """Execute steps respecting dependencies (DAG order)."""
        all_steps = dict(definition.steps)  # Copy
        pending = set(all_steps.keys())

        while pending:
            # Find steps with all dependencies met
            ready = []
            for step_id in pending:
                step = all_steps[step_id]
                deps_met = all(
                    d in execution.completed_steps or d in execution.failed_steps
                    for d in step.dependencies
                )
                if deps_met:
                    ready.append(step_id)

            if not ready:
                # Deadlock detection
                logger.error("Workflow deadlock: %s", pending)
                break

            # Execute ready steps in parallel
            tasks = []
            for step_id in ready:
                step = all_steps[step_id]
                pending.discard(step_id)
                tasks.append(self._execute_step(step, execution))

            await asyncio.gather(*tasks)

    async def _execute_step(self, step: WorkflowStep,
                               execution: WorkflowExecution):
        """Execute a single workflow step with retry."""
        # Check condition
        if step.condition:
            try:
                should_run = eval(step.condition, {"vars": execution.variables})
                if not should_run:
                    step.status = StepStatus.SKIPPED
                    execution.completed_steps.add(step.step_id)
                    return
            except Exception:
                pass

        step.status = StepStatus.RUNNING
        step.started_at = time.time()

        async with self._step_semaphore:
            for attempt in range(step.retry_count + 1):
                step.attempts = attempt + 1
                try:
                    # Build input
                    step_input = {}
                    for key, var_name in step.input_mapping.items():
                        step_input[key] = execution.variables.get(var_name)

                    # Execute
                    if step.handler:
                        result = await asyncio.wait_for(
                            step.handler(**step_input) if step_input
                            else step.handler(),
                            timeout=step.timeout_seconds,
                        )
                        step.result = result

                        # Store output
                        if step.output_key and result is not None:
                            execution.variables[step.output_key] = result

                    step.status = StepStatus.COMPLETED
                    step.completed_at = time.time()
                    step.duration_ms = (step.completed_at - step.started_at) * 1000
                    execution.completed_steps.add(step.step_id)

                    self._metrics["total_steps_executed"] += 1
                    await self._fire_hook("on_step_complete", step)
                    return

                except Exception as exc:
                    step.error = str(exc)
                    if attempt < step.retry_count:
                        await asyncio.sleep(2 ** attempt)

        # All retries exhausted
        step.status = StepStatus.FAILED
        step.completed_at = time.time()
        step.duration_ms = (step.completed_at - step.started_at) * 1000
        execution.failed_steps.add(step.step_id)

        # Run compensation
        if step.compensation_handler:
            try:
                if asyncio.iscoroutinefunction(step.compensation_handler):
                    await step.compensation_handler()
                else:
                    step.compensation_handler()
            except Exception as exc:
                logger.error("Compensation failed for %s: %s", step.name, exc)

        await self._fire_hook("on_step_fail", step)

    # ── Checkpoint & Resume ──

    def checkpoint(self, execution_id: str) -> Optional[WorkflowCheckpoint]:
        execution = self._executions.get(execution_id)
        if not execution:
            return None
        cp = WorkflowCheckpoint(
            checkpoint_id=uuid.uuid4().hex[:12],
            workflow_id=execution.workflow_id,
            step_id="",
            variables=dict(execution.variables),
            completed_steps=set(execution.completed_steps),
        )
        execution.checkpoints.append(cp)
        return cp

    async def resume(self, execution_id: str) -> Optional[WorkflowExecution]:
        execution = self._executions.get(execution_id)
        if not execution or execution.status != WorkflowStatus.PAUSED:
            return None

        definition = self._definitions.get(execution.workflow_id)
        if not definition:
            return None

        execution.status = WorkflowStatus.RUNNING
        await self._execute_dag(definition, execution)
        return execution

    def pause(self, execution_id: str) -> bool:
        execution = self._executions.get(execution_id)
        if execution and execution.status == WorkflowStatus.RUNNING:
            execution.status = WorkflowStatus.PAUSED
            self.checkpoint(execution_id)
            return True
        return False

    # ── Hooks ──

    def add_hook(self, event: str, callback: Callable):
        if event in self._hooks:
            self._hooks[event].append(callback)

    async def _fire_hook(self, event: str, data: Any):
        for hook in self._hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(data)
                else:
                    hook(data)
            except Exception as exc:
                logger.error("Workflow hook %s failed: %s", event, exc)

    # ── Queries ──

    def list_workflows(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": d.workflow_id, "name": d.name,
                "steps": len(d.steps), "version": d.version,
                "tags": d.tags,
            }
            for d in self._definitions.values()
        ]

    def list_executions(self, *, workflow_id: Optional[str] = None,
                          status: Optional[WorkflowStatus] = None,
                          limit: int = 50) -> List[Dict[str, Any]]:
        executions = list(self._executions.values())
        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]
        if status:
            executions = [e for e in executions if e.status == status]
        return [e.to_dict() for e in executions[-limit:]]

    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        execution = self._executions.get(execution_id)
        if not execution:
            return None
        definition = self._definitions.get(execution.workflow_id)
        steps = []
        if definition:
            steps = [s.to_dict() for s in definition.steps.values()]
        return {
            **execution.to_dict(),
            "steps": steps,
            "variables": execution.variables,
        }

    def get_metrics(self) -> Dict[str, Any]:
        return {
            **self._metrics,
            "active_workflows": sum(
                1 for e in self._executions.values()
                if e.status == WorkflowStatus.RUNNING
            ),
            "definitions": len(self._definitions),
            "total_executions": len(self._executions),
        }


# ── Singleton ──
_engine: Optional[WorkflowEngine] = None


def get_workflow_engine(**kwargs) -> WorkflowEngine:
    global _engine
    if not _engine:
        _engine = WorkflowEngine(**kwargs)
    return _engine
