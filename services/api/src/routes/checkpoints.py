"""
Checkpoint API Routes

Provides REST endpoints for checkpoint management.
"""

import os
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.application.checkpoint.use_cases import (
    CreateCheckpointCommand,
    RestoreCheckpointCommand,
    ListCheckpointsQuery,
    CreateCheckpointUseCase,
    RestoreCheckpointUseCase,
    ListCheckpointsUseCase,
)
from services.api.src.schemas.phase3 import (
    CreateCheckpointRequest,
    RestoreCheckpointRequest,
    CheckpointResponse,
    CheckpointListResponse,
)
from services.api.src.dependencies.injection import (
    get_db_session,
    get_checkpoint_service,
    get_event_bus,
)


router = APIRouter(prefix="/api/v3/checkpoints", tags=["Checkpoints"])


@router.post(
    "",
    response_model=CheckpointResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a checkpoint",
    description="Create a checkpoint from workflow execution state",
)
async def create_checkpoint(
    request: CreateCheckpointRequest,
    session: AsyncSession = Depends(get_db_session),
) -> CheckpointResponse:
    """Create a new checkpoint"""
    try:
        # Get checkpoint service and use case
        checkpoint_service = await get_checkpoint_service(session)
        event_bus = get_event_bus()
        use_case = CreateCheckpointUseCase(
            checkpoint_service=checkpoint_service,
            event_publisher=event_bus,
        )
        
        # Convert request to command
        command = CreateCheckpointCommand(
            workflow_execution_id=request.workflow_execution_id,
            execution_variables=request.execution_variables,
            execution_context=request.execution_context,
            current_step_id=request.current_step_id,
            completed_steps=request.completed_steps,
            pending_steps=request.pending_steps,
            failed_steps=request.failed_steps,
            skipped_steps=request.skipped_steps,
            total_steps=request.total_steps,
            allocated_budget=request.allocated_budget,
            consumed_budget=request.consumed_budget,
            memory_snapshot_ref=request.memory_snapshot_ref,
            active_tasks=request.active_tasks,
            compression_enabled=request.compression_enabled,
            error_state=request.error_state,
        )
        
        # Execute use case
        result = await use_case.execute(command)
        
        return CheckpointResponse(
            checkpoint_id=result.checkpoint_id,
            workflow_execution_id=result.workflow_execution_id,
            checkpoint_number=result.checkpoint_number,
            status=result.status.value,
            completion_percentage=result.completion_percentage,
            budget_consumed=result.budget_consumed,
            checkpoint_size_bytes=result.checkpoint_size_bytes,
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
            detail=f"Failed to create checkpoint: {str(e)}"
        )


@router.post(
    "/{checkpoint_id}/restore",
    response_model=CheckpointResponse,
    status_code=status.HTTP_200_OK,
    summary="Restore a checkpoint",
    description="Restore workflow execution from a checkpoint",
)
async def restore_checkpoint(
    checkpoint_id: UUID,
    request: RestoreCheckpointRequest,
    session: AsyncSession = Depends(get_db_session),
) -> CheckpointResponse:
    """Restore a checkpoint"""
    try:
        # Get checkpoint service and use case
        checkpoint_service = await get_checkpoint_service(session)
        event_bus = get_event_bus()
        use_case = RestoreCheckpointUseCase(
            checkpoint_service=checkpoint_service,
            event_publisher=event_bus,
        )
        
        # Convert request to command
        command = RestoreCheckpointCommand(
            checkpoint_id=checkpoint_id,
            recovery_reason=request.recovery_reason,
        )
        
        # Execute use case
        result = await use_case.execute(command)
        
        return CheckpointResponse(
            checkpoint_id=result.checkpoint_id,
            workflow_execution_id=result.workflow_execution_id,
            checkpoint_number=result.checkpoint_number,
            status=result.status.value,
            completion_percentage=result.completion_percentage,
            budget_consumed=result.budget_consumed,
            checkpoint_size_bytes=result.checkpoint_size_bytes,
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
            detail=f"Failed to restore checkpoint: {str(e)}"
        )


@router.get(
    "/workflow/{workflow_execution_id}",
    response_model=CheckpointListResponse,
    summary="List checkpoints for workflow",
    description="Retrieve checkpoint history for a workflow execution",
)
async def list_checkpoints(
    workflow_execution_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> CheckpointListResponse:
    """List checkpoints for a workflow execution"""
    try:
        # Get checkpoint service and use case
        checkpoint_service = await get_checkpoint_service(session)
        use_case = ListCheckpointsUseCase(checkpoint_service=checkpoint_service)
        
        # Create query
        query = ListCheckpointsQuery(workflow_execution_id=workflow_execution_id)
        
        # Execute use case
        results = await use_case.execute(query)
        
        return CheckpointListResponse(
            checkpoints=[
                CheckpointResponse(
                    checkpoint_id=r.checkpoint_id,
                    workflow_execution_id=r.workflow_execution_id,
                    checkpoint_number=r.checkpoint_number,
                    status=r.status.value,
                    completion_percentage=r.completion_percentage,
                    budget_consumed=r.budget_consumed,
                    checkpoint_size_bytes=r.checkpoint_size_bytes,
                    created_at=r.created_at,
                )
                for r in results
            ],
            total=len(results),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list checkpoints: {str(e)}"
        )


@router.delete(
    "/{checkpoint_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a checkpoint",
    description="Delete a specific checkpoint by ID",
)
async def delete_checkpoint(
    checkpoint_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> None:
    """Delete a checkpoint"""
    try:
        # Get checkpoint service
        checkpoint_service = await get_checkpoint_service(session)
        
        # Delete checkpoint via service
        await checkpoint_service.delete_checkpoint(checkpoint_id)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete checkpoint: {str(e)}"
        )


@router.get(
    "/{checkpoint_id}",
    response_model=CheckpointResponse,
    summary="Get checkpoint by ID",
    description="Retrieve a specific checkpoint by its ID",
)
async def get_checkpoint(
    checkpoint_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> CheckpointResponse:
    """Get a checkpoint by ID"""
    try:
        # Get checkpoint service
        checkpoint_service = await get_checkpoint_service(session)
        
        # Get checkpoint
        checkpoint = await checkpoint_service.get_checkpoint(checkpoint_id)
        
        if not checkpoint:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Checkpoint {checkpoint_id} not found"
            )
        
        return CheckpointResponse(
            checkpoint_id=checkpoint.id.value,
            workflow_execution_id=checkpoint.workflow_execution_id,
            checkpoint_number=checkpoint.checkpoint_number,
            status=checkpoint.status.value,
            completion_percentage=checkpoint.get_completion_percentage(),
            budget_consumed=checkpoint.budget_state.consumed,
            checkpoint_size_bytes=checkpoint.checkpoint_size_bytes,
            created_at=checkpoint.created_at.isoformat(),
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve checkpoint: {str(e)}"
        )
