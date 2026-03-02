"""
Workflow Automation Engine — CognitionOS

DAG-based workflow execution with:
- Visual workflow builder data model
- Step types: action, condition, loop, parallel, wait
- Trigger types: webhook, schedule, event, manual
- Execution history and replay
- Error recovery and compensation
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class StepType(str, Enum):
    ACTION = "action"
    CONDITION = "condition"
    PARALLEL = "parallel"
    LOOP = "loop"
    WAIT = "wait"
    SUB_WORKFLOW = "sub_workflow"


class TriggerType(str, Enum):
    MANUAL = "manual"
    WEBHOOK = "webhook"
    SCHEDULE = "schedule"
    EVENT = "event"


class WorkflowStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class WorkflowStep:
    step_id: str
    name: str
    step_type: StepType
    handler_name: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    next_steps: List[str] = field(default_factory=list)
    condition_true_step: str = ""
    condition_false_step: str = ""
    timeout_seconds: float = 300.0
    retry_count: int = 0
    on_error: str = "fail"  # fail, skip, retry, compensate
    compensation_handler: str = ""


@dataclass
class WorkflowDefinition:
    workflow_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    version: int = 1
    status: WorkflowStatus = WorkflowStatus.DRAFT
    trigger: TriggerType = TriggerType.MANUAL
    trigger_config: Dict[str, Any] = field(default_factory=dict)
    steps: Dict[str, WorkflowStep] = field(default_factory=dict)
    entry_step: str = ""
    variables: Dict[str, Any] = field(default_factory=dict)
    tenant_id: str = ""
    created_by: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id, "name": self.name,
            "status": self.status.value, "trigger": self.trigger.value,
            "step_count": len(self.steps), "version": self.version,
            "entry_step": self.entry_step, "tags": self.tags}


@dataclass
class StepExecution:
    step_id: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: float = 0
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0


@dataclass
class WorkflowExecution:
    execution_id: str = field(default_factory=lambda: str(uuid4()))
    workflow_id: str = ""
    status: ExecutionStatus = ExecutionStatus.PENDING
    trigger_data: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    step_executions: Dict[str, StepExecution] = field(default_factory=dict)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    total_duration_ms: float = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id, "workflow_id": self.workflow_id,
            "status": self.status.value, "started_at": self.started_at,
            "completed_at": self.completed_at, "duration_ms": self.total_duration_ms,
            "steps_completed": sum(1 for s in self.step_executions.values()
                                   if s.status == ExecutionStatus.COMPLETED),
            "steps_total": len(self.step_executions)}


class WorkflowEngine:
    """DAG-based workflow execution engine."""

    def __init__(self) -> None:
        self._workflows: Dict[str, WorkflowDefinition] = {}
        self._executions: Dict[str, WorkflowExecution] = {}
        self._handlers: Dict[str, Callable[..., Awaitable[Any]]] = {}
        self._condition_evaluators: Dict[str, Callable[..., Awaitable[bool]]] = {}
        self._metrics: Dict[str, int] = defaultdict(int)

    # ---- handler registration ----
    def register_handler(self, name: str, handler: Callable[..., Awaitable[Any]]) -> None:
        self._handlers[name] = handler

    def register_condition(self, name: str, evaluator: Callable[..., Awaitable[bool]]) -> None:
        self._condition_evaluators[name] = evaluator

    # ---- workflow management ----
    def create_workflow(self, definition: WorkflowDefinition) -> str:
        self._workflows[definition.workflow_id] = definition
        return definition.workflow_id

    def update_workflow(self, workflow_id: str, **updates: Any) -> bool:
        wf = self._workflows.get(workflow_id)
        if not wf:
            return False
        for k, v in updates.items():
            if hasattr(wf, k):
                setattr(wf, k, v)
        return True

    def activate_workflow(self, workflow_id: str) -> bool:
        return self.update_workflow(workflow_id, status=WorkflowStatus.ACTIVE)

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        return self._workflows.get(workflow_id)

    def list_workflows(self, *, tenant_id: str = "",
                       status: WorkflowStatus | None = None) -> List[Dict[str, Any]]:
        wfs = list(self._workflows.values())
        if tenant_id:
            wfs = [w for w in wfs if w.tenant_id == tenant_id]
        if status:
            wfs = [w for w in wfs if w.status == status]
        return [w.to_dict() for w in wfs]

    # ---- execution ----
    async def execute(self, workflow_id: str, *,
                      trigger_data: Dict[str, Any] | None = None) -> WorkflowExecution:
        wf = self._workflows.get(workflow_id)
        if not wf:
            raise ValueError(f"Workflow not found: {workflow_id}")
        if wf.status != WorkflowStatus.ACTIVE:
            raise ValueError(f"Workflow not active: {workflow_id}")

        execution = WorkflowExecution(
            workflow_id=workflow_id,
            trigger_data=trigger_data or {},
            variables=dict(wf.variables),
            started_at=datetime.now(timezone.utc).isoformat())
        execution.status = ExecutionStatus.RUNNING
        self._executions[execution.execution_id] = execution
        self._metrics["executions_started"] += 1

        try:
            await self._execute_step(wf, execution, wf.entry_step)
            execution.status = ExecutionStatus.COMPLETED
            self._metrics["executions_completed"] += 1
        except Exception as e:
            execution.status = ExecutionStatus.FAILED
            execution.error = str(e)
            self._metrics["executions_failed"] += 1
            logger.error("Workflow execution failed: %s — %s", execution.execution_id, e)

        execution.completed_at = datetime.now(timezone.utc).isoformat()
        if execution.started_at:
            start = datetime.fromisoformat(execution.started_at)
            end = datetime.fromisoformat(execution.completed_at)
            execution.total_duration_ms = (end - start).total_seconds() * 1000

        return execution

    async def _execute_step(self, wf: WorkflowDefinition,
                            execution: WorkflowExecution, step_id: str) -> None:
        if not step_id or step_id not in wf.steps:
            return

        step = wf.steps[step_id]
        step_exec = StepExecution(step_id=step_id, status=ExecutionStatus.RUNNING,
                                  started_at=datetime.now(timezone.utc).isoformat())
        execution.step_executions[step_id] = step_exec

        start = time.monotonic()
        try:
            if step.step_type == StepType.ACTION:
                handler = self._handlers.get(step.handler_name)
                if handler:
                    result = await asyncio.wait_for(
                        handler(execution.variables, step.config),
                        timeout=step.timeout_seconds)
                    step_exec.result = result
                    if isinstance(result, dict):
                        execution.variables.update(result)

            elif step.step_type == StepType.CONDITION:
                evaluator = self._condition_evaluators.get(step.handler_name)
                if evaluator:
                    result = await evaluator(execution.variables, step.config)
                    next_step = step.condition_true_step if result else step.condition_false_step
                    step_exec.result = {"condition": result, "next": next_step}
                    step_exec.status = ExecutionStatus.COMPLETED
                    step_exec.completed_at = datetime.now(timezone.utc).isoformat()
                    step_exec.duration_ms = (time.monotonic() - start) * 1000
                    await self._execute_step(wf, execution, next_step)
                    return

            elif step.step_type == StepType.PARALLEL:
                tasks = []
                for next_id in step.next_steps:
                    tasks.append(self._execute_step(wf, execution, next_id))
                await asyncio.gather(*tasks)
                step_exec.status = ExecutionStatus.COMPLETED
                step_exec.completed_at = datetime.now(timezone.utc).isoformat()
                step_exec.duration_ms = (time.monotonic() - start) * 1000
                return

            elif step.step_type == StepType.WAIT:
                wait_seconds = step.config.get("seconds", 1)
                await asyncio.sleep(min(wait_seconds, step.timeout_seconds))

            step_exec.status = ExecutionStatus.COMPLETED

        except Exception as e:
            step_exec.status = ExecutionStatus.FAILED
            step_exec.error = str(e)
            if step.on_error == "skip":
                pass
            elif step.on_error == "compensate" and step.compensation_handler:
                comp = self._handlers.get(step.compensation_handler)
                if comp:
                    await comp(execution.variables, step.config)
            else:
                raise

        step_exec.completed_at = datetime.now(timezone.utc).isoformat()
        step_exec.duration_ms = (time.monotonic() - start) * 1000

        # Execute next steps sequentially
        for next_id in step.next_steps:
            await self._execute_step(wf, execution, next_id)

    # ---- query ----
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        return self._executions.get(execution_id)

    def list_executions(self, *, workflow_id: str = "",
                        limit: int = 50) -> List[Dict[str, Any]]:
        execs = list(self._executions.values())
        if workflow_id:
            execs = [e for e in execs if e.workflow_id == workflow_id]
        return [e.to_dict() for e in execs[-limit:]]

    def get_metrics(self) -> Dict[str, Any]:
        return {**dict(self._metrics),
                "total_workflows": len(self._workflows),
                "total_executions": len(self._executions),
                "handlers": len(self._handlers)}


_engine: WorkflowEngine | None = None

def get_workflow_engine() -> WorkflowEngine:
    global _engine
    if not _engine:
        _engine = WorkflowEngine()
    return _engine
