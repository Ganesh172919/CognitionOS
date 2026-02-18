"""
Cost Governance API Routes

Provides REST endpoints for budget management and cost tracking.
"""

import os
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.application.cost_governance.use_cases import (
    CreateBudgetCommand,
    RecordCostCommand,
    CostSummaryQuery,
    EnforceBudgetLimitsCommand,
    CreateWorkflowBudgetUseCase,
    RecordCostUseCase,
    GetCostSummaryUseCase,
    EnforceBudgetLimitsUseCase,
)
from services.api.src.schemas.phase3 import (
    CreateBudgetRequest,
    RecordCostRequest,
    BudgetResponse,
    CostEntryResponse,
    CostSummaryResponse,
    EnforceBudgetRequest,
    EnforceBudgetResponse,
)
from services.api.src.dependencies.injection import (
    get_db_session,
    get_cost_governance_service,
    get_event_bus,
)


router = APIRouter(prefix="/api/v3/cost", tags=["Cost Governance"])


@router.post(
    "/budgets",
    response_model=BudgetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create workflow budget",
    description="Create a budget for a workflow execution with warning and critical thresholds",
)
async def create_budget(
    request: CreateBudgetRequest,
    session: AsyncSession = Depends(get_db_session),
) -> BudgetResponse:
    """Create workflow budget"""
    try:
        # Get cost governance service and use case
        cost_governance_service = await get_cost_governance_service(session)
        event_bus = get_event_bus()
        use_case = CreateWorkflowBudgetUseCase(
            cost_governance_service=cost_governance_service,
            event_publisher=event_bus,
        )
        
        # Convert request to command
        command = CreateBudgetCommand(
            workflow_execution_id=request.workflow_execution_id,
            allocated_budget=request.allocated_budget,
            currency=request.currency,
            warning_threshold=request.warning_threshold,
            critical_threshold=request.critical_threshold,
        )
        
        # Execute use case
        result = await use_case.execute(command)
        
        return BudgetResponse(
            budget_id=result.budget_id,
            workflow_execution_id=result.workflow_execution_id,
            allocated_budget=result.allocated_budget,
            consumed_budget=result.consumed_budget,
            remaining_budget=result.remaining_budget,
            currency=result.currency,
            status=result.status.value,
            usage_percentage=result.usage_percentage,
            warning_threshold=result.warning_threshold,
            critical_threshold=result.critical_threshold,
            is_exhausted=result.is_exhausted,
            is_suspended=result.is_suspended,
            created_at=result.created_at,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create budget: {str(e)}"
        )


@router.post(
    "/track",
    response_model=CostEntryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record cost entry",
    description="Record a cost entry for a workflow execution with automatic threshold checking",
)
async def record_cost(
    request: RecordCostRequest,
    session: AsyncSession = Depends(get_db_session),
) -> CostEntryResponse:
    """Record cost entry"""
    try:
        # Get cost governance service and use case
        cost_governance_service = await get_cost_governance_service(session)
        event_bus = get_event_bus()
        use_case = RecordCostUseCase(
            cost_governance_service=cost_governance_service,
            event_publisher=event_bus,
        )
        
        # Convert request to command
        command = RecordCostCommand(
            workflow_execution_id=request.workflow_execution_id,
            operation_type=request.operation_type,
            provider=request.provider,
            cost=request.cost,
            agent_id=request.agent_id,
            model=request.model,
            tokens_used=request.tokens_used,
            execution_time_ms=request.execution_time_ms,
            memory_bytes=request.memory_bytes,
            task_id=request.task_id,
            step_name=request.step_name,
            currency=request.currency,
        )
        
        # Execute use case
        result = await use_case.execute(command)
        
        return CostEntryResponse(
            cost_entry_id=result.cost_entry_id,
            workflow_execution_id=result.workflow_execution_id,
            operation_type=result.operation_type.value,
            provider=result.provider,
            cost=result.cost,
            agent_id=result.agent_id,
            model=result.model,
            tokens_used=result.tokens_used,
            consumed_budget=result.consumed_budget,
            remaining_budget=result.remaining_budget,
            usage_percentage=result.usage_percentage,
            created_at=result.created_at,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record cost: {str(e)}"
        )


async def _get_workflow_cost_summary(
    workflow_execution_id: UUID,
    session: AsyncSession,
) -> CostSummaryResponse:
    """Helper function to get cost summary for a workflow"""
    cost_governance_service = await get_cost_governance_service(session)
    use_case = GetCostSummaryUseCase(cost_governance_service=cost_governance_service)
    
    query = CostSummaryQuery(workflow_execution_id=workflow_execution_id)
    result = await use_case.execute(query)
    
    return CostSummaryResponse(
        workflow_execution_id=result.workflow_execution_id,
        total_cost=result.total_cost,
        budget_exists=result.budget_exists,
        allocated_budget=result.allocated_budget,
        consumed_budget=result.consumed_budget,
        remaining_budget=result.remaining_budget,
        usage_percentage=result.usage_percentage,
        status=result.status,
        currency=result.currency,
        is_exhausted=result.is_exhausted,
        is_suspended=result.is_suspended,
        cost_by_operation_type=result.cost_by_operation_type,
        cost_by_agent=result.cost_by_agent,
    )


@router.get(
    "/workflow/{workflow_execution_id}",
    response_model=CostSummaryResponse,
    summary="Get cost summary",
    description="Retrieve comprehensive cost summary for a workflow execution",
)
async def get_cost_summary(
    workflow_execution_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> CostSummaryResponse:
    """Get cost summary for workflow"""
    try:
        return await _get_workflow_cost_summary(workflow_execution_id, session)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve cost summary: {str(e)}"
        )


@router.post(
    "/enforce",
    response_model=EnforceBudgetResponse,
    status_code=status.HTTP_200_OK,
    summary="Enforce budget limits",
    description="Enforce budget limits and suspend workflow if budget is exhausted",
)
async def enforce_budget_limits(
    request: EnforceBudgetRequest,
    session: AsyncSession = Depends(get_db_session),
) -> EnforceBudgetResponse:
    """Enforce budget limits"""
    try:
        # Get cost governance service and use case
        cost_governance_service = await get_cost_governance_service(session)
        event_bus = get_event_bus()
        use_case = EnforceBudgetLimitsUseCase(
            cost_governance_service=cost_governance_service,
            event_publisher=event_bus,
        )
        
        # Convert request to command
        command = EnforceBudgetLimitsCommand(workflow_execution_id=request.workflow_execution_id)
        
        # Execute use case
        result = await use_case.execute(command)
        
        return EnforceBudgetResponse(
            status=result["status"],
            events_raised=result["events_raised"],
            suspended=result["suspended"],
            should_halt=result["should_halt"],
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enforce budget limits: {str(e)}"
        )


@router.get(
    "/workflow/{workflow_execution_id}/breakdown",
    response_model=CostSummaryResponse,
    summary="Get cost breakdown",
    description="Retrieve detailed cost breakdown by operation type and agent (alias for cost summary endpoint)",
)
async def get_cost_breakdown(
    workflow_execution_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> CostSummaryResponse:
    """Get cost breakdown for workflow"""
    try:
        return await _get_workflow_cost_summary(workflow_execution_id, session)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve cost breakdown: {str(e)}"
        )
