"""
Enterprise Workflow Automation Engine — Visual workflow DSL, trigger system,
versioned workflows, step execution, conditional branching, and audit logging.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# ─────────────────────────────────── Enums ───────────────────────────────────


class WorkflowStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class StepType(str, Enum):
    ACTION = "action"
    CONDITION = "condition"
    PARALLEL = "parallel"
    LOOP = "loop"
    WAIT = "wait"
    TRANSFORM = "transform"
    HTTP_REQUEST = "http_request"
    NOTIFICATION = "notification"
    AI_INFERENCE = "ai_inference"
    DATA_QUERY = "data_query"
    SUB_WORKFLOW = "sub_workflow"


class TriggerType(str, Enum):
    MANUAL = "manual"
    SCHEDULE = "schedule"
    WEBHOOK = "webhook"
    EVENT = "event"
    API_CALL = "api_call"
    DATA_CHANGE = "data_change"
    THRESHOLD = "threshold"


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING = "waiting"
    RETRYING = "retrying"


class RetryStrategy(str, Enum):
    FIXED = "fixed"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR = "linear"
    NO_RETRY = "no_retry"


# ────────────────────────────── Data structures ──────────────────────────────


@dataclass
class WorkflowStep:
    step_id: str
    name: str
    step_type: StepType
    config: Dict[str, Any]
    next_steps: List[str]
    on_success: List[str]
    on_failure: List[str]
    retry_config: Dict[str, Any]
    timeout_s: Optional[float]
    input_mapping: Dict[str, str]
    output_mapping: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "step_type": self.step_type.value,
            "config": self.config,
            "next_steps": self.next_steps,
            "on_success": self.on_success,
            "on_failure": self.on_failure,
            "retry_config": self.retry_config,
            "timeout_s": self.timeout_s,
            "input_mapping": self.input_mapping,
            "output_mapping": self.output_mapping,
            "metadata": self.metadata,
        }


@dataclass
class WorkflowTrigger:
    trigger_id: str
    trigger_type: TriggerType
    config: Dict[str, Any]
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_id": self.trigger_id,
            "trigger_type": self.trigger_type.value,
            "config": self.config,
            "enabled": self.enabled,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
            "trigger_count": self.trigger_count,
        }


@dataclass
class WorkflowDefinition:
    workflow_id: str
    name: str
    description: str
    version: str
    status: WorkflowStatus
    steps: Dict[str, WorkflowStep]
    triggers: List[WorkflowTrigger]
    entry_step_id: str
    variables: Dict[str, Any]
    tags: List[str]
    owner_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "status": self.status.value,
            "step_count": len(self.steps),
            "steps": {sid: s.to_dict() for sid, s in self.steps.items()},
            "triggers": [t.to_dict() for t in self.triggers],
            "entry_step_id": self.entry_step_id,
            "variables": self.variables,
            "tags": self.tags,
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class StepExecution:
    exec_id: str
    step_id: str
    workflow_execution_id: str
    status: ExecutionStatus
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    error: Optional[str]
    attempt: int
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exec_id": self.exec_id,
            "step_id": self.step_id,
            "workflow_execution_id": self.workflow_execution_id,
            "status": self.status.value,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
            "attempt": self.attempt,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
        }


@dataclass
class WorkflowExecution:
    execution_id: str
    workflow_id: str
    workflow_version: str
    status: ExecutionStatus
    trigger_type: TriggerType
    input_data: Dict[str, Any]
    context: Dict[str, Any]
    step_executions: List[StepExecution]
    current_step_id: Optional[str]
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.completed_at:
            delta = self.completed_at - self.started_at
            return delta.total_seconds() * 1000
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "workflow_id": self.workflow_id,
            "workflow_version": self.workflow_version,
            "status": self.status.value,
            "trigger_type": self.trigger_type.value,
            "input_data": self.input_data,
            "context": self.context,
            "step_count": len(self.step_executions),
            "step_executions": [s.to_dict() for s in self.step_executions],
            "current_step_id": self.current_step_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "metadata": self.metadata,
        }


# ─────────────────────────── Step Executors ─────────────────────────────────


class StepExecutorRegistry:
    """
    Registry of step type executors. Custom executors can be registered
    for new step types.
    """

    def __init__(self):
        self._executors: Dict[StepType, Callable] = {}
        self._register_defaults()

    def register(self, step_type: StepType, executor: Callable) -> None:
        self._executors[step_type] = executor

    async def execute(
        self,
        step: WorkflowStep,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        executor = self._executors.get(step.step_type)
        if executor is None:
            raise ValueError(f"No executor registered for step type {step.step_type.value}")
        return await executor(step, context)

    def _register_defaults(self) -> None:
        self._executors[StepType.ACTION] = self._exec_action
        self._executors[StepType.CONDITION] = self._exec_condition
        self._executors[StepType.TRANSFORM] = self._exec_transform
        self._executors[StepType.WAIT] = self._exec_wait
        self._executors[StepType.NOTIFICATION] = self._exec_notification
        self._executors[StepType.HTTP_REQUEST] = self._exec_http_request
        self._executors[StepType.AI_INFERENCE] = self._exec_ai_inference
        self._executors[StepType.DATA_QUERY] = self._exec_data_query
        self._executors[StepType.PARALLEL] = self._exec_parallel
        self._executors[StepType.LOOP] = self._exec_loop

    @staticmethod
    async def _exec_action(step: WorkflowStep, ctx: Dict[str, Any]) -> Dict[str, Any]:
        action_name = step.config.get("action_name", "unknown")
        action_params = step.config.get("params", {})
        await asyncio.sleep(0)  # Yield control
        return {
            "action_executed": action_name,
            "params": action_params,
            "result": f"Action '{action_name}' completed successfully",
            "success": True,
        }

    @staticmethod
    async def _exec_condition(step: WorkflowStep, ctx: Dict[str, Any]) -> Dict[str, Any]:
        condition_expr = step.config.get("condition", "true")
        variable = step.config.get("variable", "")
        value = ctx.get(variable, ctx.get("input", {}).get(variable))
        expected = step.config.get("expected_value")
        operator = step.config.get("operator", "eq")
        result = False
        if operator == "eq":
            result = value == expected
        elif operator == "ne":
            result = value != expected
        elif operator == "gt":
            try:
                result = float(value or 0) > float(expected or 0)
            except (ValueError, TypeError):
                result = False
        elif operator == "lt":
            try:
                result = float(value or 0) < float(expected or 0)
            except (ValueError, TypeError):
                result = False
        elif operator == "contains":
            result = expected in str(value or "")
        elif operator == "truthy":
            result = bool(value)
        return {"condition_result": result, "branch": "true" if result else "false"}

    @staticmethod
    async def _exec_transform(step: WorkflowStep, ctx: Dict[str, Any]) -> Dict[str, Any]:
        transform_type = step.config.get("transform_type", "passthrough")
        input_key = step.config.get("input_key", "data")
        output_key = step.config.get("output_key", "transformed_data")
        data = ctx.get(input_key, ctx.get("input", {}))
        transformed: Any = data

        if transform_type == "to_uppercase" and isinstance(data, str):
            transformed = data.upper()
        elif transform_type == "to_lowercase" and isinstance(data, str):
            transformed = data.lower()
        elif transform_type == "flatten" and isinstance(data, dict):
            transformed = {
                f"{k}.{sk}": sv
                for k, v in data.items()
                if isinstance(v, dict)
                for sk, sv in v.items()
            }
        elif transform_type == "json_encode":
            transformed = json.dumps(data)
        elif transform_type == "extract_field":
            field_name = step.config.get("field_name", "")
            transformed = data.get(field_name) if isinstance(data, dict) else None
        elif transform_type == "filter_list":
            filter_key = step.config.get("filter_key", "")
            filter_val = step.config.get("filter_value")
            if isinstance(data, list):
                transformed = [
                    item for item in data
                    if isinstance(item, dict) and item.get(filter_key) == filter_val
                ]

        return {output_key: transformed, "transform_type": transform_type, "success": True}

    @staticmethod
    async def _exec_wait(step: WorkflowStep, ctx: Dict[str, Any]) -> Dict[str, Any]:
        wait_s = min(step.config.get("wait_seconds", 1), 5)  # Cap at 5s for safety
        await asyncio.sleep(wait_s)
        return {"waited_seconds": wait_s, "success": True}

    @staticmethod
    async def _exec_notification(step: WorkflowStep, ctx: Dict[str, Any]) -> Dict[str, Any]:
        channel = step.config.get("channel", "email")
        recipients = step.config.get("recipients", [])
        subject = step.config.get("subject", "Workflow Notification")
        body_template = step.config.get("body_template", "Workflow step executed")
        # Render template with context variables
        body = body_template
        for key, val in ctx.items():
            body = body.replace(f"{{{key}}}", str(val))
        return {
            "notification_sent": True,
            "channel": channel,
            "recipients": recipients,
            "subject": subject,
            "body_preview": body[:100],
        }

    @staticmethod
    async def _exec_http_request(step: WorkflowStep, ctx: Dict[str, Any]) -> Dict[str, Any]:
        method = step.config.get("method", "GET")
        url = step.config.get("url", "")
        headers = step.config.get("headers", {})
        # Replace context variables in URL
        for key, val in ctx.items():
            url = url.replace(f"{{{key}}}", str(val))
        # Simulated HTTP execution (actual HTTP would use aiohttp)
        return {
            "method": method,
            "url": url,
            "status_code": 200,
            "response": {"success": True, "message": "Simulated HTTP response"},
            "success": True,
        }

    @staticmethod
    async def _exec_ai_inference(step: WorkflowStep, ctx: Dict[str, Any]) -> Dict[str, Any]:
        model = step.config.get("model", "gpt-4")
        prompt_template = step.config.get("prompt_template", "Process: {input}")
        input_data = ctx.get("input", {})
        prompt = prompt_template
        for key, val in input_data.items() if isinstance(input_data, dict) else []:
            prompt = prompt.replace(f"{{{key}}}", str(val))
        return {
            "model": model,
            "prompt_length": len(prompt),
            "inference_result": f"AI inference completed for model {model}",
            "tokens_used": len(prompt.split()) * 2,
            "success": True,
        }

    @staticmethod
    async def _exec_data_query(step: WorkflowStep, ctx: Dict[str, Any]) -> Dict[str, Any]:
        query_type = step.config.get("query_type", "select")
        table = step.config.get("table", "")
        filters = step.config.get("filters", {})
        return {
            "query_type": query_type,
            "table": table,
            "filters": filters,
            "rows_returned": 0,
            "result": [],
            "success": True,
        }

    @staticmethod
    async def _exec_parallel(step: WorkflowStep, ctx: Dict[str, Any]) -> Dict[str, Any]:
        sub_steps = step.config.get("steps", [])
        return {
            "parallel_steps": len(sub_steps),
            "completed": len(sub_steps),
            "success": True,
        }

    @staticmethod
    async def _exec_loop(step: WorkflowStep, ctx: Dict[str, Any]) -> Dict[str, Any]:
        collection_key = step.config.get("collection_key", "items")
        collection = ctx.get(collection_key, [])
        if not isinstance(collection, list):
            collection = []
        max_iterations = min(step.config.get("max_iterations", 100), 1000)
        iterations = min(len(collection), max_iterations)
        return {
            "iterations_executed": iterations,
            "collection_size": len(collection),
            "success": True,
        }


# ─────────────────────── Workflow Version Manager ───────────────────────────


class WorkflowVersionManager:
    """
    Manages workflow versions with semantic versioning and diff tracking.
    """

    def __init__(self):
        self._versions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def save_version(
        self, workflow: WorkflowDefinition, change_note: str = ""
    ) -> str:
        version_entry = {
            "version": workflow.version,
            "snapshot": workflow.to_dict(),
            "change_note": change_note,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        self._versions[workflow.workflow_id].append(version_entry)
        return workflow.version

    def get_version(
        self, workflow_id: str, version: str
    ) -> Optional[Dict[str, Any]]:
        for v in self._versions.get(workflow_id, []):
            if v["version"] == version:
                return v
        return None

    def list_versions(self, workflow_id: str) -> List[Dict[str, Any]]:
        return [
            {"version": v["version"], "saved_at": v["saved_at"], "change_note": v["change_note"]}
            for v in self._versions.get(workflow_id, [])
        ]

    def rollback(
        self, workflow_id: str, target_version: str
    ) -> Optional[Dict[str, Any]]:
        return self.get_version(workflow_id, target_version)

    def bump_version(self, current_version: str, bump_type: str = "patch") -> str:
        parts = current_version.lstrip("v").split(".")
        try:
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        except (IndexError, ValueError):
            return "1.0.1"
        if bump_type == "major":
            return f"{major + 1}.0.0"
        if bump_type == "minor":
            return f"{major}.{minor + 1}.0"
        return f"{major}.{minor}.{patch + 1}"


# ─────────────────────── Workflow Engine ────────────────────────────────────


class WorkflowAutomationEngine:
    """
    Master workflow automation engine: DSL builder, execution, versioning,
    audit logging, and analytics.
    """

    def __init__(self):
        self._workflows: Dict[str, WorkflowDefinition] = {}
        self._executions: Dict[str, WorkflowExecution] = {}
        self.executor_registry = StepExecutorRegistry()
        self.version_mgr = WorkflowVersionManager()
        self._audit_log: List[Dict[str, Any]] = []
        self._max_audit_entries = 1000
        self._execution_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "total_executions": 0,
                "successful": 0,
                "failed": 0,
                "total_duration_ms": 0.0,
            }
        )

    def create_workflow(
        self,
        name: str,
        description: str,
        owner_id: str,
        steps: List[Dict[str, Any]],
        triggers: Optional[List[Dict[str, Any]]] = None,
        variables: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> WorkflowDefinition:
        workflow_id = str(uuid.uuid4())
        step_objects: Dict[str, WorkflowStep] = {}
        entry_step_id = ""

        for i, s in enumerate(steps):
            step_id = s.get("step_id") or str(uuid.uuid4())
            step_obj = WorkflowStep(
                step_id=step_id,
                name=s.get("name", f"Step {i + 1}"),
                step_type=StepType(s.get("step_type", "action")),
                config=s.get("config", {}),
                next_steps=s.get("next_steps", []),
                on_success=s.get("on_success", []),
                on_failure=s.get("on_failure", []),
                retry_config=s.get("retry_config", {
                    "strategy": RetryStrategy.EXPONENTIAL_BACKOFF.value,
                    "max_attempts": 3,
                    "initial_delay_s": 1.0,
                }),
                timeout_s=s.get("timeout_s"),
                input_mapping=s.get("input_mapping", {}),
                output_mapping=s.get("output_mapping", {}),
                metadata=s.get("metadata", {}),
            )
            step_objects[step_id] = step_obj
            if i == 0:
                entry_step_id = step_id

        trigger_objects = [
            WorkflowTrigger(
                trigger_id=str(uuid.uuid4()),
                trigger_type=TriggerType(t.get("trigger_type", "manual")),
                config=t.get("config", {}),
                enabled=t.get("enabled", True),
            )
            for t in (triggers or [])
        ]
        if not trigger_objects:
            trigger_objects = [
                WorkflowTrigger(
                    trigger_id=str(uuid.uuid4()),
                    trigger_type=TriggerType.MANUAL,
                    config={},
                )
            ]

        workflow = WorkflowDefinition(
            workflow_id=workflow_id,
            name=name,
            description=description,
            version="1.0.0",
            status=WorkflowStatus.DRAFT,
            steps=step_objects,
            triggers=trigger_objects,
            entry_step_id=entry_step_id,
            variables=variables or {},
            tags=tags or [],
            owner_id=owner_id,
        )
        self._workflows[workflow_id] = workflow
        self.version_mgr.save_version(workflow, "Initial version")
        self._audit("workflow_created", {"workflow_id": workflow_id, "name": name})
        return workflow

    def activate_workflow(self, workflow_id: str) -> bool:
        wf = self._workflows.get(workflow_id)
        if wf is None:
            return False
        wf.status = WorkflowStatus.ACTIVE
        wf.updated_at = datetime.now(timezone.utc)
        self._audit("workflow_activated", {"workflow_id": workflow_id})
        return True

    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Optional[Dict[str, Any]] = None,
        trigger_type: TriggerType = TriggerType.MANUAL,
        execution_id: Optional[str] = None,
    ) -> WorkflowExecution:
        wf = self._workflows.get(workflow_id)
        if wf is None:
            raise ValueError(f"Workflow {workflow_id} not found")
        if wf.status not in (WorkflowStatus.ACTIVE, WorkflowStatus.DRAFT):
            raise ValueError(f"Workflow is in status {wf.status.value}, cannot execute")

        exec_id = execution_id or str(uuid.uuid4())
        execution = WorkflowExecution(
            execution_id=exec_id,
            workflow_id=workflow_id,
            workflow_version=wf.version,
            status=ExecutionStatus.RUNNING,
            trigger_type=trigger_type,
            input_data=input_data or {},
            context={**wf.variables, "input": input_data or {}},
            step_executions=[],
            current_step_id=wf.entry_step_id,
        )
        self._executions[exec_id] = execution

        try:
            await self._execute_from_step(execution, wf, wf.entry_step_id)
            execution.status = ExecutionStatus.COMPLETED
        except Exception as exc:
            execution.status = ExecutionStatus.FAILED
            execution.error = str(exc)
        finally:
            execution.completed_at = datetime.now(timezone.utc)
            stats = self._execution_stats[workflow_id]
            stats["total_executions"] += 1
            if execution.status == ExecutionStatus.COMPLETED:
                stats["successful"] += 1
            else:
                stats["failed"] += 1
            if execution.duration_ms:
                stats["total_duration_ms"] += execution.duration_ms

        self._audit(
            "workflow_executed",
            {
                "workflow_id": workflow_id,
                "execution_id": exec_id,
                "status": execution.status.value,
                "duration_ms": execution.duration_ms,
            },
        )
        return execution

    async def _execute_from_step(
        self,
        execution: WorkflowExecution,
        workflow: WorkflowDefinition,
        step_id: str,
        max_steps: int = 100,
    ) -> None:
        visited_steps: List[str] = []
        current_step_id: Optional[str] = step_id

        for _ in range(max_steps):
            if current_step_id is None:
                break
            if current_step_id in visited_steps:
                break  # Cycle detection
            step = workflow.steps.get(current_step_id)
            if step is None:
                break
            execution.current_step_id = current_step_id
            visited_steps.append(current_step_id)

            # Execute step with retry
            step_exec = await self._execute_step_with_retry(
                step, execution
            )
            execution.step_executions.append(step_exec)

            if step_exec.status == ExecutionStatus.FAILED:
                if step.on_failure:
                    current_step_id = step.on_failure[0]
                else:
                    raise RuntimeError(
                        f"Step '{step.name}' failed: {step_exec.error}"
                    )
                continue

            # Determine next step
            if step.step_type == StepType.CONDITION:
                output = step_exec.output_data or {}
                branch = output.get("branch", "false")
                next_id = (
                    step.on_success[0]
                    if branch == "true" and step.on_success
                    else step.on_failure[0]
                    if branch == "false" and step.on_failure
                    else step.next_steps[0]
                    if step.next_steps
                    else None
                )
                current_step_id = next_id
            elif step.on_success:
                current_step_id = step.on_success[0]
            elif step.next_steps:
                current_step_id = step.next_steps[0]
            else:
                current_step_id = None

            # Merge step output into context
            if step_exec.output_data:
                for out_key, ctx_key in step.output_mapping.items():
                    val = step_exec.output_data.get(out_key)
                    if val is not None:
                        execution.context[ctx_key] = val

    async def _execute_step_with_retry(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> StepExecution:
        retry_config = step.retry_config
        max_attempts = retry_config.get("max_attempts", 1)
        strategy = RetryStrategy(
            retry_config.get("strategy", RetryStrategy.NO_RETRY.value)
        )
        initial_delay = retry_config.get("initial_delay_s", 1.0)

        for attempt in range(1, max_attempts + 1):
            step_exec = StepExecution(
                exec_id=str(uuid.uuid4()),
                step_id=step.step_id,
                workflow_execution_id=execution.execution_id,
                status=ExecutionStatus.RUNNING,
                input_data=self._resolve_inputs(step, execution.context),
                output_data=None,
                error=None,
                attempt=attempt,
            )
            try:
                start = time.monotonic()
                output = await self.executor_registry.execute(
                    step, execution.context
                )
                elapsed_ms = (time.monotonic() - start) * 1000
                step_exec.status = ExecutionStatus.COMPLETED
                step_exec.output_data = output
                step_exec.completed_at = datetime.now(timezone.utc)
                step_exec.duration_ms = elapsed_ms
                return step_exec
            except Exception as exc:
                step_exec.error = str(exc)
                step_exec.status = ExecutionStatus.FAILED
                step_exec.completed_at = datetime.now(timezone.utc)
                if attempt < max_attempts:
                    delay = self._compute_retry_delay(strategy, initial_delay, attempt)
                    await asyncio.sleep(min(delay, 5.0))  # Cap retry delay
                    continue
                return step_exec
        return step_exec  # type: ignore

    def _resolve_inputs(
        self, step: WorkflowStep, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        inputs = {}
        for local_key, ctx_key in step.input_mapping.items():
            inputs[local_key] = context.get(ctx_key)
        return inputs

    @staticmethod
    def _compute_retry_delay(
        strategy: RetryStrategy, initial_delay: float, attempt: int
    ) -> float:
        if strategy == RetryStrategy.FIXED:
            return initial_delay
        if strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return initial_delay * (2 ** (attempt - 1))
        if strategy == RetryStrategy.LINEAR:
            return initial_delay * attempt
        return 0.0

    def _audit(self, event: str, data: Dict[str, Any]) -> None:
        self._audit_log.append(
            {
                "event": event,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        if len(self._audit_log) > self._max_audit_entries:
            self._audit_log = self._audit_log[-self._max_audit_entries:]

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        return self._workflows.get(workflow_id)

    def list_workflows(
        self,
        owner_id: Optional[str] = None,
        status_filter: Optional[WorkflowStatus] = None,
        tag_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        workflows = list(self._workflows.values())
        if owner_id:
            workflows = [w for w in workflows if w.owner_id == owner_id]
        if status_filter:
            workflows = [w for w in workflows if w.status == status_filter]
        if tag_filter:
            workflows = [w for w in workflows if tag_filter in w.tags]
        return [w.to_dict() for w in workflows]

    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        return self._executions.get(execution_id)

    def list_executions(
        self,
        workflow_id: Optional[str] = None,
        status_filter: Optional[ExecutionStatus] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        execs = list(self._executions.values())
        if workflow_id:
            execs = [e for e in execs if e.workflow_id == workflow_id]
        if status_filter:
            execs = [e for e in execs if e.status == status_filter]
        execs = sorted(execs, key=lambda e: e.started_at, reverse=True)[:limit]
        return [e.to_dict() for e in execs]

    def get_workflow_analytics(self, workflow_id: str) -> Dict[str, Any]:
        stats = self._execution_stats.get(workflow_id, {})
        total = stats.get("total_executions", 0)
        successful = stats.get("successful", 0)
        return {
            "workflow_id": workflow_id,
            "total_executions": total,
            "successful": successful,
            "failed": stats.get("failed", 0),
            "success_rate": successful / max(total, 1),
            "avg_duration_ms": (
                stats.get("total_duration_ms", 0) / max(total, 1)
            ),
            "versions": self.version_mgr.list_versions(workflow_id),
        }

    def get_audit_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._audit_log[-limit:]

    def get_engine_stats(self) -> Dict[str, Any]:
        workflows = list(self._workflows.values())
        by_status: Dict[str, int] = defaultdict(int)
        for w in workflows:
            by_status[w.status.value] += 1
        total_execs = sum(
            s["total_executions"] for s in self._execution_stats.values()
        )
        return {
            "total_workflows": len(workflows),
            "by_status": dict(by_status),
            "total_executions": total_execs,
            "total_step_types": len(StepType),
            "total_trigger_types": len(TriggerType),
        }
