"""
Workflow Infrastructure - PostgreSQL Repository Implementation

Concrete implementation of WorkflowRepository using PostgreSQL.
"""

from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.domain.workflow import (
    Workflow,
    WorkflowExecution,
    StepExecution,
    WorkflowId,
    Version,
    StepId,
    WorkflowStatus,
    ExecutionStatus,
    WorkflowStep,
    WorkflowRepository,
    WorkflowExecutionRepository
)

from infrastructure.persistence.workflow_models import (
    WorkflowModel,
    WorkflowExecutionModel,
    StepExecutionModel,
    WorkflowStatusEnum,
    ExecutionStatusEnum
)


class PostgreSQLWorkflowRepository(WorkflowRepository):
    """
    PostgreSQL implementation of WorkflowRepository.

    Maps between domain entities and SQLAlchemy models.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, workflow: Workflow) -> None:
        """Persist workflow to database"""
        # Convert domain entity to ORM model
        model = self._to_model(workflow)

        # Merge (insert or update)
        self.session.add(model)
        await self.session.flush()

    async def get_by_id(self, workflow_id: WorkflowId, version: Version) -> Optional[Workflow]:
        """Retrieve workflow by ID and version"""
        stmt = select(WorkflowModel).where(
            and_(
                WorkflowModel.id == workflow_id.value,
                WorkflowModel.version_major == version.major,
                WorkflowModel.version_minor == version.minor,
                WorkflowModel.version_patch == version.patch
            )
        )

        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()

        if model is None:
            return None

        return self._to_entity(model)

    async def get_latest_version(self, workflow_id: WorkflowId) -> Optional[Workflow]:
        """Get latest version of workflow"""
        stmt = (
            select(WorkflowModel)
            .options(selectinload(WorkflowModel.executions))
            .where(WorkflowModel.id == workflow_id.value)
            .order_by(
                WorkflowModel.version_major.desc(),
                WorkflowModel.version_minor.desc(),
                WorkflowModel.version_patch.desc()
            )
            .limit(1)
        )

        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()

        if model is None:
            return None

        return self._to_entity(model)

    async def get_by_status(self, status: WorkflowStatus, limit: int = 100) -> List[Workflow]:
        """Get workflows by status"""
        db_status = WorkflowStatusEnum(status.value)

        stmt = (
            select(WorkflowModel)
            .where(WorkflowModel.status == db_status)
            .limit(limit)
        )

        result = await self.session.execute(stmt)
        models = result.scalars().all()

        return [self._to_entity(model) for model in models]

    async def get_scheduled_workflows(self) -> List[Workflow]:
        """Get all workflows with schedules"""
        stmt = select(WorkflowModel).where(WorkflowModel.schedule.isnot(None))

        result = await self.session.execute(stmt)
        models = result.scalars().all()

        return [self._to_entity(model) for model in models]

    async def delete(self, workflow_id: WorkflowId, version: Version) -> bool:
        """Delete workflow"""
        stmt = select(WorkflowModel).where(
            and_(
                WorkflowModel.id == workflow_id.value,
                WorkflowModel.version_major == version.major,
                WorkflowModel.version_minor == version.minor,
                WorkflowModel.version_patch == version.patch
            )
        )

        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()

        if model is None:
            return False

        await self.session.delete(model)
        await self.session.flush()
        return True

    async def exists(self, workflow_id: WorkflowId, version: Version) -> bool:
        """Check if workflow exists"""
        workflow = await self.get_by_id(workflow_id, version)
        return workflow is not None

    async def list_versions(self, workflow_id: WorkflowId) -> List[Version]:
        """List all versions of a workflow"""
        stmt = (
            select(
                WorkflowModel.version_major,
                WorkflowModel.version_minor,
                WorkflowModel.version_patch
            )
            .where(WorkflowModel.id == workflow_id.value)
            .order_by(
                WorkflowModel.version_major.desc(),
                WorkflowModel.version_minor.desc(),
                WorkflowModel.version_patch.desc()
            )
        )

        result = await self.session.execute(stmt)
        rows = result.all()

        return [
            Version(major=row[0], minor=row[1], patch=row[2])
            for row in rows
        ]

    def _to_model(self, workflow: Workflow) -> WorkflowModel:
        """Convert domain entity to ORM model"""
        # Serialize steps to JSON
        steps_json = [
            {
                "id": step.id.value,
                "type": step.type,
                "name": step.name,
                "params": step.params,
                "depends_on": [dep.value for dep in step.depends_on],
                "timeout_seconds": step.timeout_seconds,
                "retry_count": step.retry_count,
                "condition": step.condition,
                "agent_role": step.agent_role
            }
            for step in workflow.steps
        ]

        return WorkflowModel(
            id=workflow.id.value,
            version_major=workflow.version.major,
            version_minor=workflow.version.minor,
            version_patch=workflow.version.patch,
            name=workflow.name,
            description=workflow.description,
            status=WorkflowStatusEnum(workflow.status.value),
            schedule=workflow.schedule,
            tags=workflow.tags,
            steps=steps_json,
            created_by=workflow.created_by,
            created_at=workflow.created_at
        )

    def _to_entity(self, model: WorkflowModel) -> Workflow:
        """Convert ORM model to domain entity"""
        # Deserialize steps from JSON
        steps = [
            WorkflowStep(
                id=StepId(step_data["id"]),
                type=step_data["type"],
                name=step_data["name"],
                params=step_data.get("params", {}),
                depends_on=[StepId(dep) for dep in step_data.get("depends_on", [])],
                timeout_seconds=step_data.get("timeout_seconds", 300),
                retry_count=step_data.get("retry_count", 0),
                condition=step_data.get("condition"),
                agent_role=step_data.get("agent_role")
            )
            for step_data in model.steps
        ]

        return Workflow(
            id=WorkflowId(model.id),
            version=Version(
                major=model.version_major,
                minor=model.version_minor,
                patch=model.version_patch
            ),
            name=model.name,
            description=model.description,
            steps=steps,
            status=WorkflowStatus(model.status.value),
            schedule=model.schedule,
            tags=model.tags or [],
            created_at=model.created_at,
            created_by=model.created_by
        )


class PostgreSQLWorkflowExecutionRepository(WorkflowExecutionRepository):
    """
    PostgreSQL implementation of WorkflowExecutionRepository.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, execution: WorkflowExecution) -> None:
        """Persist workflow execution"""
        model = self._to_model(execution)
        self.session.add(model)
        await self.session.flush()

    async def get_by_id(self, execution_id: UUID) -> Optional[WorkflowExecution]:
        """Retrieve execution by ID"""
        stmt = (
            select(WorkflowExecutionModel)
            .options(selectinload(WorkflowExecutionModel.step_executions))
            .where(WorkflowExecutionModel.id == execution_id)
        )

        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()

        if model is None:
            return None

        return self._to_entity(model)

    async def get_by_workflow(
        self,
        workflow_id: WorkflowId,
        limit: int = 100,
        status: Optional[ExecutionStatus] = None
    ) -> List[WorkflowExecution]:
        """Get executions for a workflow"""
        stmt = (
            select(WorkflowExecutionModel)
            .options(selectinload(WorkflowExecutionModel.step_executions))
            .where(WorkflowExecutionModel.workflow_id == workflow_id.value)
        )

        if status:
            db_status = ExecutionStatusEnum(status.value)
            stmt = stmt.where(WorkflowExecutionModel.status == db_status)

        stmt = stmt.limit(limit).order_by(WorkflowExecutionModel.created_at.desc())

        result = await self.session.execute(stmt)
        models = result.scalars().all()

        return [self._to_entity(model) for model in models]

    async def get_active_executions(self) -> List[WorkflowExecution]:
        """Get all running executions"""
        stmt = select(WorkflowExecutionModel).where(
            WorkflowExecutionModel.status == ExecutionStatusEnum.RUNNING
        )

        result = await self.session.execute(stmt)
        models = result.scalars().all()

        return [self._to_entity(model) for model in models]

    async def save_step_execution(self, step_execution: StepExecution) -> None:
        """Persist step execution"""
        model = self._step_to_model(step_execution)
        self.session.add(model)
        await self.session.flush()

    async def get_step_execution(self, step_execution_id: UUID) -> Optional[StepExecution]:
        """Retrieve step execution by ID"""
        stmt = select(StepExecutionModel).where(StepExecutionModel.id == step_execution_id)

        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()

        if model is None:
            return None

        return self._step_to_entity(model)

    async def get_step_executions(self, execution_id: UUID) -> List[StepExecution]:
        """Get all step executions for a workflow execution"""
        stmt = (
            select(StepExecutionModel)
            .where(StepExecutionModel.execution_id == execution_id)
            .order_by(StepExecutionModel.started_at.asc())
        )

        result = await self.session.execute(stmt)
        models = result.scalars().all()

        return [self._step_to_entity(model) for model in models]

    async def get_pending_steps(self, execution_id: UUID) -> List[StepExecution]:
        """Get pending step executions"""
        stmt = select(StepExecutionModel).where(
            and_(
                StepExecutionModel.execution_id == execution_id,
                StepExecutionModel.status == ExecutionStatusEnum.PENDING
            )
        )

        result = await self.session.execute(stmt)
        models = result.scalars().all()

        return [self._step_to_entity(model) for model in models]

    async def get_failed_steps(self, execution_id: UUID) -> List[StepExecution]:
        """Get failed step executions"""
        stmt = select(StepExecutionModel).where(
            and_(
                StepExecutionModel.execution_id == execution_id,
                StepExecutionModel.status == ExecutionStatusEnum.FAILED
            )
        )

        result = await self.session.execute(stmt)
        models = result.scalars().all()

        return [self._step_to_entity(model) for model in models]

    async def delete_execution(self, execution_id: UUID) -> bool:
        """Delete workflow execution"""
        stmt = select(WorkflowExecutionModel).where(WorkflowExecutionModel.id == execution_id)

        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()

        if model is None:
            return False

        await self.session.delete(model)
        await self.session.flush()
        return True

    def _to_model(self, execution: WorkflowExecution) -> WorkflowExecutionModel:
        """Convert domain entity to ORM model"""
        return WorkflowExecutionModel(
            id=execution.id,
            workflow_id=execution.workflow_id.value,
            workflow_version_major=execution.workflow_version.major,
            workflow_version_minor=execution.workflow_version.minor,
            workflow_version_patch=execution.workflow_version.patch,
            status=ExecutionStatusEnum(execution.status.value),
            inputs=execution.inputs,
            outputs=execution.outputs,
            error=execution.error,
            user_id=execution.user_id,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            created_at=execution.created_at
        )

    def _to_entity(self, model: WorkflowExecutionModel) -> WorkflowExecution:
        """Convert ORM model to domain entity"""
        return WorkflowExecution(
            id=model.id,
            workflow_id=WorkflowId(model.workflow_id),
            workflow_version=Version(
                major=model.workflow_version_major,
                minor=model.workflow_version_minor,
                patch=model.workflow_version_patch
            ),
            status=ExecutionStatus(model.status.value),
            inputs=model.inputs or {},
            outputs=model.outputs,
            error=model.error,
            user_id=model.user_id,
            started_at=model.started_at,
            completed_at=model.completed_at,
            created_at=model.created_at
        )

    def _step_to_model(self, step_execution: StepExecution) -> StepExecutionModel:
        """Convert step execution to ORM model"""
        return StepExecutionModel(
            id=step_execution.id,
            execution_id=step_execution.execution_id,
            step_id=step_execution.step_id.value,
            step_type=step_execution.step_type,
            status=ExecutionStatusEnum(step_execution.status.value),
            output=step_execution.output,
            error=step_execution.error,
            agent_id=step_execution.agent_id,
            retry_count=step_execution.retry_count,
            started_at=step_execution.started_at,
            completed_at=step_execution.completed_at
        )

    def _step_to_entity(self, model: StepExecutionModel) -> StepExecution:
        """Convert step execution model to domain entity"""
        return StepExecution(
            id=model.id,
            execution_id=model.execution_id,
            step_id=StepId(model.step_id),
            step_type=model.step_type,
            status=ExecutionStatus(model.status.value),
            output=model.output,
            error=model.error,
            agent_id=model.agent_id,
            retry_count=model.retry_count,
            started_at=model.started_at,
            completed_at=model.completed_at
        )
