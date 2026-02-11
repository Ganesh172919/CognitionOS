"""
Workflow Application - Use Cases

Application layer use cases for Workflow bounded context.
Orchestrates domain entities and coordinates with infrastructure.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from core.domain.workflow import (
    Workflow,
    WorkflowExecution,
    WorkflowStep,
    StepExecution,
    WorkflowId,
    Version,
    StepId,
    WorkflowStatus,
    ExecutionStatus,
    WorkflowRepository,
    WorkflowExecutionRepository,
    WorkflowValidator,
    WorkflowExecutionOrchestrator,
    WorkflowCreated,
    WorkflowExecutionStarted,
    WorkflowExecutionCompleted,
    WorkflowExecutionFailed,
    StepExecutionCompleted
)


# ==================== DTOs (Data Transfer Objects) ====================

@dataclass
class CreateWorkflowCommand:
    """Command to create a new workflow"""
    workflow_id: str
    version: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    schedule: Optional[str] = None
    tags: List[str] = None
    created_by: Optional[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class ExecuteWorkflowCommand:
    """Command to execute a workflow"""
    workflow_id: str
    workflow_version: str
    inputs: Dict[str, Any]
    user_id: Optional[UUID] = None


@dataclass
class WorkflowExecutionResult:
    """Result of workflow execution"""
    execution_id: UUID
    workflow_id: str
    workflow_version: str
    status: ExecutionStatus
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ==================== Use Cases ====================

class CreateWorkflowUseCase:
    """
    Use Case: Create a new workflow definition.

    Orchestrates:
    1. Parse workflow definition
    2. Validate DAG structure
    3. Persist workflow
    4. Publish domain event
    """

    def __init__(
        self,
        workflow_repository: WorkflowRepository,
        event_publisher: Optional[Any] = None  # EventBus interface
    ):
        self.workflow_repository = workflow_repository
        self.event_publisher = event_publisher

    async def execute(self, command: CreateWorkflowCommand) -> WorkflowId:
        """
        Create a new workflow.

        Args:
            command: Create workflow command

        Returns:
            WorkflowId of created workflow

        Raises:
            ValueError: If workflow validation fails
        """
        # Parse steps
        steps = []
        for step_data in command.steps:
            step = WorkflowStep(
                id=StepId(step_data["id"]),
                type=step_data["type"],
                name=step_data.get("name", step_data["id"]),
                params=step_data.get("params", {}),
                depends_on=[StepId(dep) for dep in step_data.get("depends_on", [])],
                timeout_seconds=self._parse_timeout(step_data.get("timeout", "300s")),
                retry_count=step_data.get("retry", 0),
                condition=step_data.get("condition"),
                agent_role=step_data.get("agent_role")
            )
            steps.append(step)

        # Create workflow entity
        workflow = Workflow(
            id=WorkflowId(command.workflow_id),
            version=Version.from_string(command.version),
            name=command.name,
            description=command.description,
            steps=steps,
            status=WorkflowStatus.DRAFT,
            schedule=command.schedule,
            tags=command.tags,
            created_by=command.created_by
        )

        # Validate
        is_valid, error = WorkflowValidator.validate_dag(workflow)
        if not is_valid:
            raise ValueError(f"Invalid workflow DAG: {error}")

        # Activate if valid
        workflow.activate()

        # Persist
        await self.workflow_repository.save(workflow)

        # Publish event
        if self.event_publisher:
            event = WorkflowCreated(
                occurred_at=workflow.created_at,
                event_id=uuid4(),
                workflow_id=workflow.id,
                version=workflow.version,
                name=workflow.name,
                created_by=workflow.created_by
            )
            await self.event_publisher.publish(event)

        return workflow.id

    def _parse_timeout(self, timeout_str: str) -> int:
        """Parse timeout string like '60s' or '5m' to seconds"""
        if timeout_str.endswith('s'):
            return int(timeout_str[:-1])
        elif timeout_str.endswith('m'):
            return int(timeout_str[:-1]) * 60
        elif timeout_str.endswith('h'):
            return int(timeout_str[:-1]) * 3600
        else:
            return int(timeout_str)


class ExecuteWorkflowUseCase:
    """
    Use Case: Execute a workflow.

    Orchestrates:
    1. Load workflow definition
    2. Create execution instance
    3. Start execution
    4. Publish domain event
    """

    def __init__(
        self,
        workflow_repository: WorkflowRepository,
        execution_repository: WorkflowExecutionRepository,
        event_publisher: Optional[Any] = None
    ):
        self.workflow_repository = workflow_repository
        self.execution_repository = execution_repository
        self.event_publisher = event_publisher

    async def execute(self, command: ExecuteWorkflowCommand) -> UUID:
        """
        Start workflow execution.

        Args:
            command: Execute workflow command

        Returns:
            Execution ID

        Raises:
            ValueError: If workflow cannot be executed
        """
        # Load workflow
        workflow_id = WorkflowId(command.workflow_id)
        version = Version.from_string(command.workflow_version)

        workflow = await self.workflow_repository.get_by_id(workflow_id, version)
        if not workflow:
            raise ValueError(f"Workflow {command.workflow_id} v{command.workflow_version} not found")

        # Check if can execute
        if not workflow.can_execute():
            raise ValueError(f"Workflow cannot be executed (status: {workflow.status})")

        # Create execution
        execution = WorkflowExecution(
            id=uuid4(),
            workflow_id=workflow.id,
            workflow_version=workflow.version,
            status=ExecutionStatus.PENDING,
            inputs=command.inputs,
            user_id=command.user_id
        )

        # Start execution
        execution.start()

        # Persist
        await self.execution_repository.save(execution)

        # Create step executions
        for step in workflow.steps:
            step_execution = StepExecution(
                id=uuid4(),
                execution_id=execution.id,
                step_id=step.id,
                step_type=step.type,
                status=ExecutionStatus.PENDING
            )
            await self.execution_repository.save_step_execution(step_execution)

        # Publish event
        if self.event_publisher:
            event = WorkflowExecutionStarted(
                occurred_at=execution.started_at,
                event_id=uuid4(),
                execution_id=execution.id,
                workflow_id=workflow.id,
                workflow_version=workflow.version,
                inputs=command.inputs,
                user_id=command.user_id
            )
            await self.event_publisher.publish(event)

        return execution.id


class GetWorkflowExecutionStatusUseCase:
    """
    Use Case: Get workflow execution status.

    Retrieves current status of a workflow execution.
    """

    def __init__(
        self,
        execution_repository: WorkflowExecutionRepository
    ):
        self.execution_repository = execution_repository

    async def execute(self, execution_id: UUID) -> WorkflowExecutionResult:
        """
        Get execution status.

        Args:
            execution_id: Execution ID

        Returns:
            Workflow execution result

        Raises:
            ValueError: If execution not found
        """
        execution = await self.execution_repository.get_by_id(execution_id)
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")

        return WorkflowExecutionResult(
            execution_id=execution.id,
            workflow_id=execution.workflow_id.value,
            workflow_version=str(execution.workflow_version),
            status=execution.status,
            outputs=execution.outputs,
            error=execution.error
        )


class ProcessWorkflowStepUseCase:
    """
    Use Case: Process next ready workflow steps.

    Orchestrates:
    1. Find ready steps (dependencies met)
    2. Execute steps
    3. Update execution status
    4. Check if workflow complete
    """

    def __init__(
        self,
        workflow_repository: WorkflowRepository,
        execution_repository: WorkflowExecutionRepository,
        step_executor: Any,  # StepExecutor interface
        event_publisher: Optional[Any] = None
    ):
        self.workflow_repository = workflow_repository
        self.execution_repository = execution_repository
        self.step_executor = step_executor
        self.event_publisher = event_publisher

    async def execute(self, execution_id: UUID) -> bool:
        """
        Process workflow steps.

        Args:
            execution_id: Workflow execution ID

        Returns:
            True if workflow complete, False if still running
        """
        # Load execution
        execution = await self.execution_repository.get_by_id(execution_id)
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")

        if execution.is_terminal():
            return True

        # Load workflow
        workflow = await self.workflow_repository.get_by_id(
            execution.workflow_id,
            execution.workflow_version
        )
        if not workflow:
            raise ValueError(f"Workflow not found")

        # Load step executions
        step_executions = await self.execution_repository.get_step_executions(execution_id)

        # Get ready steps
        ready_steps = WorkflowExecutionOrchestrator.get_ready_steps(
            workflow,
            execution,
            step_executions
        )

        # Execute ready steps
        for step in ready_steps:
            await self._execute_step(execution, step, step_executions)

        # Check if complete
        is_complete, is_successful, error = WorkflowExecutionOrchestrator.is_execution_complete(
            workflow,
            step_executions
        )

        if is_complete:
            if is_successful:
                execution.complete(self._gather_outputs(step_executions))
            else:
                execution.fail(error)

            await self.execution_repository.save(execution)

            # Publish event
            if self.event_publisher:
                if is_successful:
                    event = WorkflowExecutionCompleted(
                        occurred_at=execution.completed_at,
                        event_id=uuid4(),
                        execution_id=execution.id,
                        workflow_id=workflow.id,
                        workflow_version=workflow.version,
                        outputs=execution.outputs,
                        duration_seconds=execution.duration_seconds()
                    )
                else:
                    event = WorkflowExecutionFailed(
                        occurred_at=execution.completed_at,
                        event_id=uuid4(),
                        execution_id=execution.id,
                        workflow_id=workflow.id,
                        workflow_version=workflow.version,
                        error=error,
                        failed_step_id=None
                    )
                await self.event_publisher.publish(event)

            return True

        return False

    async def _execute_step(
        self,
        execution: WorkflowExecution,
        step: WorkflowStep,
        step_executions: List[StepExecution]
    ) -> None:
        """Execute a single step"""
        # Find step execution
        step_exec = next(
            (se for se in step_executions if se.step_id == step.id),
            None
        )

        if not step_exec:
            raise ValueError(f"Step execution not found for {step.id}")

        # Start step
        step_exec.start()
        await self.execution_repository.save_step_execution(step_exec)

        # Execute (delegate to step executor)
        try:
            result = await self.step_executor.execute(step, execution.inputs)
            step_exec.complete(result)
        except Exception as e:
            step_exec.fail(str(e))

        await self.execution_repository.save_step_execution(step_exec)

    def _gather_outputs(self, step_executions: List[StepExecution]) -> Dict[str, Any]:
        """Gather outputs from all steps"""
        outputs = {}
        for step_exec in step_executions:
            if step_exec.output:
                outputs[step_exec.step_id.value] = step_exec.output
        return outputs
