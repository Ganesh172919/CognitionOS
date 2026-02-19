"""
Advanced Workflow Orchestration Engine
Complex workflow execution with branching, parallel tasks, compensation, and state management.
Supports distributed workflows, retries, timeouts, and event-driven orchestration.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    COMPENSATED = "compensated"
    TIMEOUT = "timeout"


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Types of workflow tasks"""
    ACTION = "action"  # Execute action
    DECISION = "decision"  # Conditional branching
    PARALLEL = "parallel"  # Run tasks in parallel
    WAIT = "wait"  # Wait for event or time
    LOOP = "loop"  # Iterate over items
    HUMAN_TASK = "human_task"  # Manual approval
    SAGA = "saga"  # Distributed transaction with compensation


@dataclass
class TaskDefinition:
    """Definition of a workflow task"""
    task_id: str
    name: str
    task_type: TaskType
    action: Optional[str] = None  # Function/service to call
    parameters: Dict[str, Any] = field(default_factory=dict)
    retry_policy: Optional[Dict[str, Any]] = None
    timeout_seconds: Optional[int] = None
    compensation_action: Optional[str] = None  # For SAGA pattern
    depends_on: List[str] = field(default_factory=list)  # Task dependencies
    condition: Optional[str] = None  # For decision tasks
    branches: Dict[str, List[str]] = field(default_factory=dict)  # For decision branches
    parallel_tasks: List[str] = field(default_factory=list)  # For parallel execution


@dataclass
class TaskExecution:
    """Runtime task execution state"""
    task_id: str
    execution_id: str
    status: TaskStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    compensation_executed: bool = False


@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    workflow_id: str
    name: str
    description: str
    version: str
    tasks: List[TaskDefinition]
    initial_task_id: str
    variables: Dict[str, Any] = field(default_factory=dict)
    max_execution_time_seconds: Optional[int] = None
    enable_compensation: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class WorkflowExecution:
    """Runtime workflow execution state"""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    task_executions: Dict[str, TaskExecution]
    context: Dict[str, Any]  # Shared execution context
    started_at: datetime
    completed_at: Optional[datetime] = None
    current_tasks: Set[str] = field(default_factory=set)
    completed_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Set[str] = field(default_factory=set)
    error: Optional[str] = None
    compensation_tasks: List[str] = field(default_factory=list)


@dataclass
class WorkflowEvent:
    """Event for workflow orchestration"""
    event_id: str
    event_type: str
    execution_id: str
    task_id: Optional[str]
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class WorkflowOrchestrationEngine:
    """
    Advanced Workflow Orchestration Engine

    Features:
    - Complex workflow definitions with branching
    - Parallel task execution
    - Event-driven choreography
    - Automatic retry with backoff
    - Timeout handling
    - SAGA pattern with compensation
    - State machine execution
    - Human-in-the-loop tasks
    - Distributed transaction coordination
    - Workflow versioning
    """

    def __init__(self):
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.task_handlers: Dict[str, Callable] = {}
        self.event_listeners: Dict[str, List[Callable]] = {}
        self.compensation_handlers: Dict[str, Callable] = {}

    async def register_workflow(self, workflow: WorkflowDefinition):
        """Register workflow definition"""
        self.workflows[workflow.workflow_id] = workflow

    async def register_task_handler(
        self,
        action_name: str,
        handler: Callable
    ):
        """Register handler for task action"""
        self.task_handlers[action_name] = handler

    async def register_compensation_handler(
        self,
        action_name: str,
        handler: Callable
    ):
        """Register compensation handler for SAGA pattern"""
        self.compensation_handlers[action_name] = handler

    async def start_workflow(
        self,
        workflow_id: str,
        input_data: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecution:
        """
        Start workflow execution

        Args:
            workflow_id: ID of workflow to execute
            input_data: Initial input data

        Returns:
            Workflow execution instance
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow = self.workflows[workflow_id]
        execution_id = str(uuid4())

        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            task_executions={},
            context=input_data or {},
            started_at=datetime.utcnow(),
            current_tasks={workflow.initial_task_id}
        )

        self.executions[execution_id] = execution

        # Start execution in background
        asyncio.create_task(self._execute_workflow(execution_id))

        return execution

    async def _execute_workflow(self, execution_id: str):
        """Execute workflow"""
        execution = self.executions[execution_id]
        workflow = self.workflows[execution.workflow_id]

        try:
            # Check max execution time
            start_time = datetime.utcnow()
            max_time = workflow.max_execution_time_seconds

            while execution.current_tasks and execution.status == WorkflowStatus.RUNNING:
                # Check timeout
                if max_time and (datetime.utcnow() - start_time).total_seconds() > max_time:
                    execution.status = WorkflowStatus.FAILED
                    execution.error = "Workflow timeout"
                    break

                # Execute current tasks
                tasks_to_execute = list(execution.current_tasks)
                execution.current_tasks.clear()

                # Execute tasks (possibly in parallel)
                results = await asyncio.gather(
                    *[self._execute_task(execution_id, task_id) for task_id in tasks_to_execute],
                    return_exceptions=True
                )

                # Process results
                for task_id, result in zip(tasks_to_execute, results):
                    task_execution = execution.task_executions[task_id]

                    if isinstance(result, Exception):
                        task_execution.status = TaskStatus.FAILED
                        task_execution.error = str(result)
                        execution.failed_tasks.add(task_id)

                        # Trigger compensation if enabled
                        if workflow.enable_compensation:
                            execution.status = WorkflowStatus.COMPENSATING
                            await self._compensate_workflow(execution_id)
                            execution.status = WorkflowStatus.COMPENSATED
                            return
                    else:
                        task_execution.status = TaskStatus.COMPLETED
                        task_execution.result = result
                        execution.completed_tasks.add(task_id)

                        # Determine next tasks
                        next_tasks = await self._get_next_tasks(
                            workflow,
                            task_id,
                            task_execution,
                            execution
                        )
                        execution.current_tasks.update(next_tasks)

            # Workflow completed
            if not execution.current_tasks and not execution.failed_tasks:
                execution.status = WorkflowStatus.COMPLETED
                execution.completed_at = datetime.utcnow()

        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.utcnow()

    async def _execute_task(
        self,
        execution_id: str,
        task_id: str
    ) -> Any:
        """Execute single task"""
        execution = self.executions[execution_id]
        workflow = self.workflows[execution.workflow_id]

        # Find task definition
        task_def = next((t for t in workflow.tasks if t.task_id == task_id), None)
        if not task_def:
            raise ValueError(f"Task {task_id} not found")

        # Create task execution
        task_execution = TaskExecution(
            task_id=task_id,
            execution_id=execution_id,
            status=TaskStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        execution.task_executions[task_id] = task_execution

        try:
            # Execute based on task type
            if task_def.task_type == TaskType.ACTION:
                result = await self._execute_action_task(task_def, execution)
            elif task_def.task_type == TaskType.DECISION:
                result = await self._execute_decision_task(task_def, execution)
            elif task_def.task_type == TaskType.PARALLEL:
                result = await self._execute_parallel_task(task_def, execution)
            elif task_def.task_type == TaskType.WAIT:
                result = await self._execute_wait_task(task_def, execution)
            elif task_def.task_type == TaskType.LOOP:
                result = await self._execute_loop_task(task_def, execution)
            elif task_def.task_type == TaskType.HUMAN_TASK:
                result = await self._execute_human_task(task_def, execution)
            elif task_def.task_type == TaskType.SAGA:
                result = await self._execute_saga_task(task_def, execution)
            else:
                raise ValueError(f"Unknown task type: {task_def.task_type}")

            task_execution.completed_at = datetime.utcnow()
            return result

        except Exception as e:
            # Retry logic
            if task_def.retry_policy:
                max_retries = task_def.retry_policy.get("max_retries", 0)
                if task_execution.retry_count < max_retries:
                    task_execution.retry_count += 1
                    backoff = task_def.retry_policy.get("backoff_seconds", 1) * (2 ** task_execution.retry_count)
                    await asyncio.sleep(backoff)
                    return await self._execute_task(execution_id, task_id)

            raise

    async def _execute_action_task(
        self,
        task_def: TaskDefinition,
        execution: WorkflowExecution
    ) -> Any:
        """Execute action task"""
        if not task_def.action or task_def.action not in self.task_handlers:
            raise ValueError(f"Handler not found for action: {task_def.action}")

        handler = self.task_handlers[task_def.action]

        # Resolve parameters from context
        params = self._resolve_parameters(task_def.parameters, execution.context)

        # Execute with timeout
        if task_def.timeout_seconds:
            result = await asyncio.wait_for(
                handler(**params),
                timeout=task_def.timeout_seconds
            )
        else:
            result = await handler(**params)

        # Store result in context
        execution.context[f"{task_def.task_id}_result"] = result

        return result

    async def _execute_decision_task(
        self,
        task_def: TaskDefinition,
        execution: WorkflowExecution
    ) -> str:
        """Execute decision/branching task"""
        if not task_def.condition:
            return "default"

        # Evaluate condition against context
        condition_result = self._evaluate_condition(
            task_def.condition,
            execution.context
        )

        return "true" if condition_result else "false"

    async def _execute_parallel_task(
        self,
        task_def: TaskDefinition,
        execution: WorkflowExecution
    ) -> List[Any]:
        """Execute tasks in parallel"""
        if not task_def.parallel_tasks:
            return []

        results = await asyncio.gather(
            *[self._execute_task(execution.execution_id, t) for t in task_def.parallel_tasks],
            return_exceptions=True
        )

        return results

    async def _execute_wait_task(
        self,
        task_def: TaskDefinition,
        execution: WorkflowExecution
    ) -> None:
        """Wait for event or time"""
        wait_seconds = task_def.parameters.get("wait_seconds")
        if wait_seconds:
            await asyncio.sleep(wait_seconds)

        # Could also wait for specific event
        event_type = task_def.parameters.get("event_type")
        if event_type:
            # Wait for event (simplified)
            await asyncio.sleep(1)

    async def _execute_loop_task(
        self,
        task_def: TaskDefinition,
        execution: WorkflowExecution
    ) -> List[Any]:
        """Execute loop over items"""
        items = task_def.parameters.get("items", [])
        results = []

        for item in items:
            # Add item to context
            execution.context["loop_item"] = item

            # Execute loop body tasks
            for subtask_id in task_def.parallel_tasks:
                result = await self._execute_task(execution.execution_id, subtask_id)
                results.append(result)

        return results

    async def _execute_human_task(
        self,
        task_def: TaskDefinition,
        execution: WorkflowExecution
    ) -> None:
        """Execute human task (requires manual intervention)"""
        # Mark execution as paused waiting for human input
        execution.status = WorkflowStatus.PAUSED

        # In production: Send notification, wait for approval
        # For now, simulate
        await asyncio.sleep(1)

    async def _execute_saga_task(
        self,
        task_def: TaskDefinition,
        execution: WorkflowExecution
    ) -> Any:
        """Execute SAGA task with compensation"""
        result = await self._execute_action_task(task_def, execution)

        # Track for potential compensation
        if task_def.compensation_action:
            execution.compensation_tasks.append(task_def.task_id)

        return result

    def _resolve_parameters(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve parameter values from context"""
        resolved = {}

        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("$"):
                # Reference to context variable
                context_key = value[1:]
                resolved[key] = context.get(context_key)
            else:
                resolved[key] = value

        return resolved

    def _evaluate_condition(
        self,
        condition: str,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate condition expression"""
        # Simple condition evaluation
        # In production: Use expression parser or DSL
        try:
            return eval(condition, {"__builtins__": {}}, context)
        except Exception:
            return False

    async def _get_next_tasks(
        self,
        workflow: WorkflowDefinition,
        completed_task_id: str,
        task_execution: TaskExecution,
        execution: WorkflowExecution
    ) -> Set[str]:
        """Determine next tasks to execute"""
        next_tasks = set()

        # Find task definition
        task_def = next((t for t in workflow.tasks if t.task_id == completed_task_id), None)
        if not task_def:
            return next_tasks

        # For decision tasks, follow branch
        if task_def.task_type == TaskType.DECISION:
            branch = task_execution.result
            if branch in task_def.branches:
                next_tasks.update(task_def.branches[branch])

        # Find tasks that depend on this task
        for task in workflow.tasks:
            if completed_task_id in task.depends_on:
                # Check if all dependencies completed
                if all(dep in execution.completed_tasks for dep in task.depends_on):
                    next_tasks.add(task.task_id)

        return next_tasks

    async def _compensate_workflow(self, execution_id: str):
        """Execute compensation logic (SAGA pattern)"""
        execution = self.executions[execution_id]
        workflow = self.workflows[execution.workflow_id]

        # Execute compensation in reverse order
        for task_id in reversed(execution.compensation_tasks):
            task_def = next((t for t in workflow.tasks if t.task_id == task_id), None)
            if not task_def or not task_def.compensation_action:
                continue

            if task_def.compensation_action in self.compensation_handlers:
                handler = self.compensation_handlers[task_def.compensation_action]
                try:
                    await handler(execution.context)
                    execution.task_executions[task_id].compensation_executed = True
                except Exception as e:
                    # Log compensation failure
                    print(f"Compensation failed for {task_id}: {e}")

    async def pause_workflow(self, execution_id: str):
        """Pause workflow execution"""
        if execution_id in self.executions:
            self.executions[execution_id].status = WorkflowStatus.PAUSED

    async def resume_workflow(self, execution_id: str):
        """Resume paused workflow"""
        if execution_id in self.executions:
            execution = self.executions[execution_id]
            if execution.status == WorkflowStatus.PAUSED:
                execution.status = WorkflowStatus.RUNNING
                asyncio.create_task(self._execute_workflow(execution_id))

    async def cancel_workflow(self, execution_id: str):
        """Cancel workflow execution"""
        if execution_id in self.executions:
            execution = self.executions[execution_id]
            execution.status = WorkflowStatus.CANCELLED
            execution.completed_at = datetime.utcnow()

    async def get_workflow_status(
        self,
        execution_id: str
    ) -> Dict[str, Any]:
        """Get workflow execution status"""
        if execution_id not in self.executions:
            return {"error": "Execution not found"}

        execution = self.executions[execution_id]

        return {
            "execution_id": execution_id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "started_at": execution.started_at.isoformat(),
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "completed_tasks": len(execution.completed_tasks),
            "failed_tasks": len(execution.failed_tasks),
            "current_tasks": list(execution.current_tasks),
            "error": execution.error
        }

    async def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get workflow execution metrics"""
        total = len(self.executions)
        completed = sum(1 for e in self.executions.values() if e.status == WorkflowStatus.COMPLETED)
        failed = sum(1 for e in self.executions.values() if e.status == WorkflowStatus.FAILED)
        running = sum(1 for e in self.executions.values() if e.status == WorkflowStatus.RUNNING)

        avg_duration = 0.0
        durations = []
        for execution in self.executions.values():
            if execution.completed_at:
                duration = (execution.completed_at - execution.started_at).total_seconds()
                durations.append(duration)

        if durations:
            avg_duration = sum(durations) / len(durations)

        return {
            "total_executions": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "success_rate": completed / total if total > 0 else 0.0,
            "average_duration_seconds": avg_duration,
            "total_workflows": len(self.workflows)
        }
