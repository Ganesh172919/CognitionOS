"""
Workflow API Routes

Provides REST endpoints for workflow management and execution.
"""

import os
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.application.workflow.use_cases import (
    CreateWorkflowCommand,
    ExecuteWorkflowCommand,
)
from services.api.src.schemas.workflows import (
    CreateWorkflowRequest,
    ExecuteWorkflowRequest,
    WorkflowResponse,
    WorkflowExecutionResponse,
    WorkflowListResponse,
    SuccessResponse,
)
from services.api.src.dependencies.injection import (
    get_db_session,
    get_create_workflow_use_case,
    get_execute_workflow_use_case,
    get_workflow_status_use_case,
    get_workflow_repository,
    get_workflow_execution_repository,
)
from services.api.src.auth import CurrentUser, get_current_user_optional


router = APIRouter(prefix="/api/v3/workflows", tags=["Workflows"])


@router.post(
    "",
    response_model=WorkflowResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new workflow",
    description="Create a new workflow definition with steps and dependencies",
)
async def create_workflow(
    request: CreateWorkflowRequest,
    session: AsyncSession = Depends(get_db_session),
) -> WorkflowResponse:
    """Create a new workflow"""
    try:
        # Get use case with session-bound repository
        workflow_repo = await get_workflow_repository(session)
        use_case = get_create_workflow_use_case(workflow_repository=workflow_repo)
        
        # Convert request to command
        command = CreateWorkflowCommand(
            workflow_id=request.workflow_id,
            version=request.version,
            name=request.name,
            description=request.description,
            steps=[step.model_dump() for step in request.steps],
            schedule=request.schedule,
            tags=request.tags,
            created_by=request.created_by,
        )
        
        # Execute use case
        workflow_id = await use_case.execute(command)
        
        # Retrieve and return created workflow
        workflow = await workflow_repo.get_by_id(workflow_id)
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve created workflow"
            )
        
        # Convert domain entity to response
        return WorkflowResponse(
            workflow_id=workflow.workflow_id.value,
            version=str(workflow.version),
            name=workflow.name,
            description=workflow.description,
            status=workflow.status.value,
            steps=[
                {
                    "step_id": step.step_id.value,
                    "name": step.name,
                    "agent_capability": step.agent_capability,
                    "inputs": step.inputs,
                    "depends_on": [dep.value for dep in step.depends_on],
                    "timeout_seconds": step.timeout_seconds,
                    "retry_config": step.retry_config,
                }
                for step in workflow.steps
            ],
            schedule=workflow.schedule,
            tags=workflow.tags,
            created_by=workflow.created_by,
            created_at=workflow.created_at,
            updated_at=workflow.updated_at,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create workflow: {str(e)}"
        )


@router.get(
    "/{workflow_id}/{version}",
    response_model=WorkflowResponse,
    summary="Get a workflow",
    description="Retrieve a specific workflow by ID and version",
)
async def get_workflow(
    workflow_id: str,
    version: str,
    session: AsyncSession = Depends(get_db_session),
) -> WorkflowResponse:
    """Get a workflow by ID and version"""
    try:
        from core.domain.workflow import WorkflowId, Version
        
        workflow_repo = await get_workflow_repository(session)
        workflow = await workflow_repo.get_by_id(WorkflowId(workflow_id))
        
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow {workflow_id} version {version} not found"
            )
        
        return WorkflowResponse(
            workflow_id=workflow.workflow_id.value,
            version=str(workflow.version),
            name=workflow.name,
            description=workflow.description,
            status=workflow.status.value,
            steps=[
                {
                    "step_id": step.step_id.value,
                    "name": step.name,
                    "agent_capability": step.agent_capability,
                    "inputs": step.inputs,
                    "depends_on": [dep.value for dep in step.depends_on],
                    "timeout_seconds": step.timeout_seconds,
                    "retry_config": step.retry_config,
                }
                for step in workflow.steps
            ],
            schedule=workflow.schedule,
            tags=workflow.tags,
            created_by=workflow.created_by,
            created_at=workflow.created_at,
            updated_at=workflow.updated_at,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve workflow: {str(e)}"
        )


@router.post(
    "/execute",
    response_model=WorkflowExecutionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Execute a workflow",
    description="Start execution of a workflow with provided inputs",
)
async def execute_workflow(
    request: ExecuteWorkflowRequest,
    session: AsyncSession = Depends(get_db_session),
) -> WorkflowExecutionResponse:
    """Execute a workflow"""
    try:
        # Get use case with session-bound repositories
        workflow_repo = await get_workflow_repository(session)
        execution_repo = await get_workflow_execution_repository(session)
        use_case = get_execute_workflow_use_case(
            workflow_repository=workflow_repo,
            execution_repository=execution_repo,
        )
        
        # Convert request to command
        command = ExecuteWorkflowCommand(
            workflow_id=request.workflow_id,
            workflow_version=request.workflow_version,
            inputs=request.inputs,
            user_id=request.user_id,
        )
        
        # Execute use case
        result = await use_case.execute(command)
        
        return WorkflowExecutionResponse(
            execution_id=result.execution_id,
            workflow_id=result.workflow_id,
            workflow_version=result.workflow_version,
            status=result.status.value,
            inputs=request.inputs,
            outputs=result.outputs,
            error=result.error,
            started_at=datetime.utcnow(),
            completed_at=None,
            user_id=request.user_id,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute workflow: {str(e)}"
        )


@router.get(
    "/executions/{execution_id}",
    response_model=WorkflowExecutionResponse,
    summary="Get execution status",
    description="Retrieve the status and details of a workflow execution",
)
async def get_execution_status(
    execution_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> WorkflowExecutionResponse:
    """Get execution status"""
    try:
        execution_repo = await get_workflow_execution_repository(session)
        use_case = get_workflow_status_use_case(execution_repository=execution_repo)
        
        execution = await execution_repo.get_by_id(execution_id)
        
        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution {execution_id} not found"
            )
        
        return WorkflowExecutionResponse(
            execution_id=execution.execution_id,
            workflow_id=execution.workflow_id.value,
            workflow_version=str(execution.workflow_version),
            status=execution.status.value,
            inputs=execution.inputs,
            outputs=execution.outputs,
            error=execution.error_message,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            user_id=execution.user_id,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve execution status: {str(e)}"
        )


@router.get(
    "",
    response_model=WorkflowListResponse,
    summary="List workflows",
    description="List all workflows with pagination and filtering",
)
async def list_workflows(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    session: AsyncSession = Depends(get_db_session),
) -> WorkflowListResponse:
    """List workflows"""
    try:
        workflow_repo = await get_workflow_repository(session)
        workflows = await workflow_repo.list_all()
        
        # Apply filtering
        if status:
            workflows = [w for w in workflows if w.status.value == status]
        
        # Apply pagination
        start = (page - 1) * page_size
        end = start + page_size
        paginated_workflows = workflows[start:end]
        
        return WorkflowListResponse(
            workflows=[
                WorkflowResponse(
                    workflow_id=w.workflow_id.value,
                    version=str(w.version),
                    name=w.name,
                    description=w.description,
                    status=w.status.value,
                    steps=[
                        {
                            "step_id": step.step_id.value,
                            "name": step.name,
                            "agent_capability": step.agent_capability,
                            "inputs": step.inputs,
                            "depends_on": [dep.value for dep in step.depends_on],
                            "timeout_seconds": step.timeout_seconds,
                            "retry_config": step.retry_config,
                        }
                        for step in w.steps
                    ],
                    schedule=w.schedule,
                    tags=w.tags,
                    created_by=w.created_by,
                    created_at=w.created_at,
                    updated_at=w.updated_at,
                )
                for w in paginated_workflows
            ],
            total=len(workflows),
            page=page,
            page_size=page_size,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list workflows: {str(e)}"
        )
