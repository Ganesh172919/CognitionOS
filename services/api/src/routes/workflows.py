"""
Workflow API Routes

Provides REST endpoints for workflow management and execution.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status as http_status
from sqlalchemy.ext.asyncio import AsyncSession

from core.application.workflow.use_cases import (
    CreateWorkflowCommand,
    ExecuteWorkflowCommand,
)
from core.domain.billing.entities import SubscriptionTier
from core.domain.billing.services import EntitlementService, UsageMeteringService
from core.domain.workflow import WorkflowId, WorkflowStatus, Version
from infrastructure.middleware.tenant_context import get_current_tenant
from infrastructure.persistence.billing_repository import (
    PostgreSQLSubscriptionRepository,
    PostgreSQLUsageRecordRepository,
)
from services.api.src.schemas.workflows import (
    CreateWorkflowRequest,
    ExecuteWorkflowRequest,
    WorkflowResponse,
    WorkflowExecutionResponse,
    WorkflowListResponse,
)
from services.api.src.dependencies.injection import (
    get_db_session,
    get_create_workflow_use_case,
    get_execute_workflow_use_case,
    get_workflow_repository,
    get_workflow_execution_repository,
)
from services.api.src.auth import CurrentUser, get_current_user_optional


router = APIRouter(prefix="/api/v3/workflows", tags=["Workflows"])

logger = logging.getLogger(__name__)


def _workflow_to_response(workflow) -> WorkflowResponse:
    return WorkflowResponse(
        workflow_id=workflow.id.value,
        version=str(workflow.version),
        name=workflow.name,
        description=workflow.description,
        status=workflow.status.value,
        steps=[
            {
                "step_id": step.id.value,
                "name": step.name,
                "type": step.type,
                "params": step.params or {},
                "depends_on": [dep.value for dep in step.depends_on],
                "timeout_seconds": step.timeout_seconds,
                "retry_count": step.retry_count,
                "condition": step.condition,
                "agent_role": step.agent_role,
            }
            for step in workflow.steps
        ],
        schedule=workflow.schedule,
        tags=workflow.tags or [],
        created_by=workflow.created_by,
        created_at=workflow.created_at,
        updated_at=getattr(workflow, "updated_at", workflow.created_at),
    )


def _execution_to_response(execution) -> WorkflowExecutionResponse:
    return WorkflowExecutionResponse(
        execution_id=execution.id,
        workflow_id=execution.workflow_id.value,
        workflow_version=str(execution.workflow_version),
        status=execution.status.value,
        inputs=execution.inputs or {},
        outputs=execution.outputs or None,
        error=execution.error,
        started_at=execution.started_at or datetime.utcnow(),
        completed_at=execution.completed_at,
        user_id=execution.user_id,
    )


@router.post(
    "",
    response_model=WorkflowResponse,
    status_code=http_status.HTTP_201_CREATED,
    summary="Create a new workflow",
    description="Create a new workflow definition with steps and dependencies",
)
async def create_workflow(
    request: CreateWorkflowRequest,
    session: AsyncSession = Depends(get_db_session),
    current_user: Optional[CurrentUser] = Depends(get_current_user_optional),
) -> WorkflowResponse:
    """Create a new workflow"""
    try:
        # Get use case with session-bound repository
        workflow_repo = await get_workflow_repository(session)
        use_case = get_create_workflow_use_case(workflow_repository=workflow_repo)

        created_by = request.created_by or (current_user.user_id if current_user else None)

        steps = [
            {
                "id": step.step_id,
                "type": step.type,
                "name": step.name or step.step_id,
                "params": step.params or {},
                "depends_on": step.depends_on,
                "timeout": f"{step.timeout_seconds}s",
                "retry": step.retry_count,
                "condition": step.condition,
                "agent_role": step.agent_role,
            }
            for step in request.steps
        ]

        # Convert request to command (normalized to application-layer contract)
        command = CreateWorkflowCommand(
            workflow_id=request.workflow_id,
            version=request.version,
            name=request.name,
            description=request.description,
            steps=steps,
            schedule=request.schedule,
            tags=request.tags,
            created_by=created_by,
        )

        # Execute use case
        workflow_id = await use_case.execute(command)

        # Retrieve and return created workflow
        workflow = await workflow_repo.get_by_id(workflow_id, Version.from_string(request.version))
        if not workflow:
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve created workflow"
            )

        return _workflow_to_response(workflow)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create workflow: {str(e)}"
        )


@router.get(
    "/create",
    response_model=WorkflowResponse,
    status_code=http_status.HTTP_201_CREATED,
    summary="Create a new workflow (legacy)",
    description="Legacy alias for workflow creation. Prefer POST /api/v3/workflows.",
)
async def create_workflow_legacy(
    request: CreateWorkflowRequest,
    session: AsyncSession = Depends(get_db_session),
    current_user: Optional[CurrentUser] = Depends(get_current_user_optional),
) -> WorkflowResponse:
    return await create_workflow(request=request, session=session, current_user=current_user)


@router.post(
    "/execute",
    response_model=WorkflowExecutionResponse,
    status_code=http_status.HTTP_202_ACCEPTED,
    summary="Execute a workflow",
    description="Start execution of a workflow with provided inputs",
)
async def execute_workflow(
    request: ExecuteWorkflowRequest,
    session: AsyncSession = Depends(get_db_session),
    current_user: Optional[CurrentUser] = Depends(get_current_user_optional),
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

        workflow_version = request.workflow_version
        if not workflow_version:
            latest = await workflow_repo.get_latest_version(WorkflowId(request.workflow_id))
            if not latest:
                raise HTTPException(
                    status_code=http_status.HTTP_404_NOT_FOUND,
                    detail=f"Workflow {request.workflow_id} not found",
                )
            workflow_version = str(latest.version)

        user_id = request.user_id
        if user_id is None and current_user is not None:
            try:
                user_id = UUID(current_user.user_id)
            except Exception:
                user_id = None

        tenant = get_current_tenant()
        usage_repo: Optional[PostgreSQLUsageRecordRepository] = None

        # Enforce entitlement (only when tenant context is established).
        if tenant:
            usage_repo = PostgreSQLUsageRecordRepository(session)
            subscription_repo = PostgreSQLSubscriptionRepository(session)

            subscription = await subscription_repo.get_by_tenant(tenant.id)
            if subscription:
                entitlement_service = EntitlementService(
                    subscription_repository=subscription_repo,
                    usage_repository=usage_repo,
                )
                check = await entitlement_service.check_entitlement(
                    tenant_id=tenant.id,
                    resource_type="executions",
                    quantity=Decimal("1"),
                )
                if not check.allowed:
                    raise HTTPException(
                        status_code=http_status.HTTP_403_FORBIDDEN,
                        detail=check.reason or "Execution quota exceeded",
                    )
            else:
                # Fallback to tier limits derived from tenant.subscription_tier when no subscription exists.
                try:
                    tier = SubscriptionTier(tenant.subscription_tier)
                except Exception:
                    tier = SubscriptionTier.FREE

                limit = EntitlementService.TIER_LIMITS.get(tier, {}).get("executions")
                if limit is not None:
                    now = datetime.utcnow()
                    month_start = datetime(now.year, now.month, 1)
                    current_usage = await usage_repo.aggregate_usage(
                        tenant_id=tenant.id,
                        resource_type="executions",
                        start_time=month_start,
                        end_time=now,
                    )
                    if current_usage + Decimal("1") > limit:
                        raise HTTPException(
                            status_code=http_status.HTTP_403_FORBIDDEN,
                            detail="Usage limit exceeded for executions",
                        )

        # Convert request to command
        command = ExecuteWorkflowCommand(
            workflow_id=request.workflow_id,
            workflow_version=workflow_version,
            inputs=request.inputs or {},
            user_id=user_id,
        )

        # Execute use case
        execution_id = await use_case.execute(command)
        execution = await execution_repo.get_by_id(execution_id)
        if not execution:
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve created execution",
            )

        # Record metered usage after the execution is successfully created.
        if tenant and usage_repo:
            usage_service = UsageMeteringService(usage_repository=usage_repo)
            try:
                await usage_service.record_usage(
                    tenant_id=tenant.id,
                    resource_type="executions",
                    quantity=Decimal("1"),
                    unit="count",
                    metadata={
                        "workflow_id": request.workflow_id,
                        "workflow_version": workflow_version,
                        "execution_id": str(execution.id),
                        "user_id": str(user_id) if user_id else None,
                    },
                )
            except Exception as e:
                logger.warning(
                    "Failed to record execution usage",
                    extra={
                        "tenant_id": str(tenant.id),
                        "workflow_id": request.workflow_id,
                        "execution_id": str(execution.id),
                        "error": str(e),
                    },
                )

        return _execution_to_response(execution)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute workflow: {str(e)}"
        )


@router.get(
    "/executions/{execution_id}",
    response_model=WorkflowExecutionResponse,
    summary="Get execution status",
    description="Retrieve the status and details of a workflow execution",
)
async def get_execution_status(
    execution_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> WorkflowExecutionResponse:
    """Get execution status"""
    try:
        execution_repo = await get_workflow_execution_repository(session)

        try:
            execution_uuid = UUID(execution_id)
        except ValueError:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Execution {execution_id} not found",
            )

        execution = await execution_repo.get_by_id(execution_uuid)

        if not execution:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Execution {execution_id} not found"
            )

        return _execution_to_response(execution)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve execution status: {str(e)}"
        )


@router.get(
    "/status/{workflow_id}",
    summary="Get workflow status (legacy)",
    description="Legacy endpoint that returns the latest execution status for a workflow.",
)
async def get_workflow_status_legacy(
    workflow_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    try:
        workflow_repo = await get_workflow_repository(session)
        workflow = await workflow_repo.get_latest_version(WorkflowId(workflow_id))
        if not workflow:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Workflow {workflow_id} not found",
            )

        execution_repo = await get_workflow_execution_repository(session)
        executions = await execution_repo.get_by_workflow(WorkflowId(workflow_id), limit=1)
        latest = executions[0] if executions else None

        return {
            "workflow_id": workflow.id.value,
            "execution_id": str(latest.id) if latest else None,
            "status": latest.status.value if latest else "not_started",
        }
    except ValueError as e:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get(
    "/list",
    response_model=WorkflowListResponse,
    summary="List workflows (legacy)",
    description="Legacy alias for listing workflows. Prefer GET /api/v3/workflows.",
)
async def list_workflows_legacy(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    session: AsyncSession = Depends(get_db_session),
) -> WorkflowListResponse:
    return await list_workflows(page=page, page_size=page_size, status=status, session=session)


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

        fetch_limit = max(page * page_size, 1)

        if status:
            try:
                status_enum = WorkflowStatus(status.lower())
            except ValueError:
                raise HTTPException(
                    status_code=http_status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status '{status}'",
                )
            workflows = await workflow_repo.get_by_status(status_enum, limit=fetch_limit)
        else:
            workflows = []
            for status_enum in WorkflowStatus:
                workflows.extend(await workflow_repo.get_by_status(status_enum, limit=fetch_limit))

        workflows.sort(key=lambda w: w.created_at, reverse=True)

        # Apply pagination
        start = (page - 1) * page_size
        end = start + page_size
        paginated_workflows = workflows[start:end]

        return WorkflowListResponse(
            workflows=[
                _workflow_to_response(w)
                for w in paginated_workflows
            ],
            total=len(workflows),
            page=page,
            page_size=page_size,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list workflows: {str(e)}"
        )


@router.get(
    "/{workflow_id}/{version}",
    response_model=WorkflowResponse,
    summary="Get a workflow",
    description="Retrieve a specific workflow by ID and version",
)
async def get_workflow_by_version(
    workflow_id: str,
    version: str,
    session: AsyncSession = Depends(get_db_session),
) -> WorkflowResponse:
    try:
        workflow_repo = await get_workflow_repository(session)
        workflow = await workflow_repo.get_by_id(WorkflowId(workflow_id), Version.from_string(version))

        if not workflow:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Workflow {workflow_id} version {version} not found",
            )

        return _workflow_to_response(workflow)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve workflow: {str(e)}",
        )


@router.get(
    "/{workflow_id}",
    response_model=WorkflowResponse,
    summary="Get workflow (latest)",
    description="Retrieve the latest version of a workflow by ID.",
)
async def get_workflow_latest(
    workflow_id: str,
    session: AsyncSession = Depends(get_db_session),
) -> WorkflowResponse:
    try:
        workflow_repo = await get_workflow_repository(session)
        workflow = await workflow_repo.get_latest_version(WorkflowId(workflow_id))
        if not workflow:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND,
                detail=f"Workflow {workflow_id} not found",
            )
        return _workflow_to_response(workflow)
    except ValueError as e:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=str(e))
