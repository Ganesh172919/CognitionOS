"""
Workflow Domain - Repository Interfaces

Pure interfaces defining how to persist and retrieve workflows.
NO implementation details - these are defined in infrastructure layer.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from .entities import (
    Workflow,
    WorkflowExecution,
    WorkflowId,
    StepExecution,
    Version,
    WorkflowStatus,
    ExecutionStatus
)


class WorkflowRepository(ABC):
    """
    Repository interface for Workflow aggregate.

    Implementations must be provided by infrastructure layer.
    Following Repository pattern from DDD.
    """

    @abstractmethod
    async def save(self, workflow: Workflow) -> None:
        """
        Persist workflow aggregate.

        Args:
            workflow: Workflow to save
        """
        pass

    @abstractmethod
    async def get_by_id(self, workflow_id: WorkflowId, version: Version) -> Optional[Workflow]:
        """
        Retrieve workflow by ID and version.

        Args:
            workflow_id: Workflow identifier
            version: Workflow version

        Returns:
            Workflow if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_latest_version(self, workflow_id: WorkflowId) -> Optional[Workflow]:
        """
        Get latest version of workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Latest workflow version if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_by_status(self, status: WorkflowStatus, limit: int = 100) -> List[Workflow]:
        """
        Get workflows by status.

        Args:
            status: Workflow status filter
            limit: Maximum number of workflows to return

        Returns:
            List of workflows matching status
        """
        pass

    @abstractmethod
    async def get_scheduled_workflows(self) -> List[Workflow]:
        """
        Get all workflows with cron schedules.

        Returns:
            List of scheduled workflows
        """
        pass

    @abstractmethod
    async def delete(self, workflow_id: WorkflowId, version: Version) -> bool:
        """
        Delete workflow.

        Args:
            workflow_id: Workflow identifier
            version: Workflow version

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def exists(self, workflow_id: WorkflowId, version: Version) -> bool:
        """
        Check if workflow exists.

        Args:
            workflow_id: Workflow identifier
            version: Workflow version

        Returns:
            True if exists, False otherwise
        """
        pass

    @abstractmethod
    async def list_versions(self, workflow_id: WorkflowId) -> List[Version]:
        """
        List all versions of a workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            List of versions sorted by version number
        """
        pass


class WorkflowExecutionRepository(ABC):
    """
    Repository interface for WorkflowExecution aggregate.

    Handles persistence of workflow executions and step executions.
    """

    @abstractmethod
    async def save(self, execution: WorkflowExecution) -> None:
        """
        Persist workflow execution.

        Args:
            execution: Workflow execution to save
        """
        pass

    @abstractmethod
    async def get_by_id(self, execution_id: UUID) -> Optional[WorkflowExecution]:
        """
        Retrieve execution by ID.

        Args:
            execution_id: Execution identifier

        Returns:
            Workflow execution if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_by_workflow(
        self,
        workflow_id: WorkflowId,
        limit: int = 100,
        status: Optional[ExecutionStatus] = None
    ) -> List[WorkflowExecution]:
        """
        Get executions for a workflow.

        Args:
            workflow_id: Workflow identifier
            limit: Maximum number of executions to return
            status: Optional status filter

        Returns:
            List of workflow executions
        """
        pass

    @abstractmethod
    async def get_active_executions(self) -> List[WorkflowExecution]:
        """
        Get all currently running executions.

        Returns:
            List of running workflow executions
        """
        pass

    @abstractmethod
    async def save_step_execution(self, step_execution: StepExecution) -> None:
        """
        Persist step execution.

        Args:
            step_execution: Step execution to save
        """
        pass

    @abstractmethod
    async def get_step_execution(self, step_execution_id: UUID) -> Optional[StepExecution]:
        """
        Retrieve step execution by ID.

        Args:
            step_execution_id: Step execution identifier

        Returns:
            Step execution if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_step_executions(
        self,
        execution_id: UUID
    ) -> List[StepExecution]:
        """
        Get all step executions for a workflow execution.

        Args:
            execution_id: Workflow execution identifier

        Returns:
            List of step executions ordered by start time
        """
        pass

    @abstractmethod
    async def get_pending_steps(self, execution_id: UUID) -> List[StepExecution]:
        """
        Get pending step executions.

        Args:
            execution_id: Workflow execution identifier

        Returns:
            List of pending step executions
        """
        pass

    @abstractmethod
    async def get_failed_steps(self, execution_id: UUID) -> List[StepExecution]:
        """
        Get failed step executions.

        Args:
            execution_id: Workflow execution identifier

        Returns:
            List of failed step executions
        """
        pass

    @abstractmethod
    async def delete_execution(self, execution_id: UUID) -> bool:
        """
        Delete workflow execution and all associated step executions.

        Args:
            execution_id: Execution identifier

        Returns:
            True if deleted, False if not found
        """
        pass
