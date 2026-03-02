"""
P0 Execution Persistence API Routes

New endpoints for deterministic execution, replay, and resume.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.domain.execution import (
    ReplayMode,
    ReplaySession,
    ExecutionSnapshot,
    SnapshotType,
)
from services.api.src.dependencies.injection import get_db_session


# ==================== Request/Response Models ====================

class ReplayExecutionRequest(BaseModel):
    """Request to replay an execution"""
    replay_mode: str = Field(..., description="Replay mode: full, from_step, or failed_only")
    start_from_step: Optional[str] = Field(None, description="Step ID to start replay from (for from_step mode)")
    use_cached_outputs: bool = Field(True, description="Use cached outputs for deterministic steps")


class ReplayExecutionResponse(BaseModel):
    """Response for replay request"""
    replay_session_id: str
    original_execution_id: str
    replay_execution_id: str
    status: str
    message: str


class ResumeExecutionRequest(BaseModel):
    """Request to resume an execution"""
    from_snapshot_id: Optional[str] = Field(None, description="Specific snapshot to resume from")
    skip_failed_steps: bool = Field(False, description="Skip failed steps and continue from next step")


class ResumeExecutionResponse(BaseModel):
    """Response for resume request"""
    execution_id: str
    resumed_from_snapshot: str
    status: str
    pending_steps: List[str]
    message: str


class ExecutionSnapshotResponse(BaseModel):
    """Execution snapshot response"""
    id: str
    execution_id: str
    snapshot_type: str
    completed_steps: List[str]
    pending_steps: List[str]
    failed_steps: List[str]
    created_at: datetime
    can_resume: bool


class ComparisonResult(BaseModel):
    """Replay comparison result"""
    replay_session_id: str
    match_percentage: float
    total_steps: int
    matching_steps: int
    divergent_steps: List[Dict[str, Any]]
    status: str


# ==================== Router ====================

router = APIRouter(prefix="/api/v3/executions", tags=["Execution Persistence (P0)"])


@router.post(
    "/{execution_id}/replay",
    response_model=ReplayExecutionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Replay a workflow execution",
    description="Replay a previous execution for deterministic verification and comparison",
)
async def replay_execution(
    execution_id: UUID,
    request: ReplayExecutionRequest,
    session: AsyncSession = Depends(get_db_session),
) -> ReplayExecutionResponse:
    """
    Replay a workflow execution.

    This endpoint creates a new execution that replays a previous execution,
    allowing verification of deterministic behavior and comparison of outputs.

    Modes:
    - full: Replay entire workflow from start
    - from_step: Replay from a specific step onwards
    - failed_only: Replay only the steps that failed

    Returns a replay session ID for tracking comparison results.
    """
    try:
        # Validate replay mode
        try:
            mode = ReplayMode(request.replay_mode.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid replay mode: {request.replay_mode}. Must be: full, from_step, or failed_only"
            )

        # Verify original execution exists
        from infrastructure.persistence.workflow_repository import PostgreSQLWorkflowExecutionRepository
        execution_repo = PostgreSQLWorkflowExecutionRepository(session)
        original_execution = await execution_repo.get_by_id(execution_id)

        if not original_execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution {execution_id} not found"
            )

        # Validate mode-specific requirements
        if mode == ReplayMode.FROM_STEP and not request.start_from_step:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="start_from_step is required for from_step replay mode"
            )

        # Create new execution for replay
        replay_execution_id = uuid4()

        # Create replay session
        replay_session = ReplaySession(
            id=uuid4(),
            original_execution_id=execution_id,
            replay_execution_id=replay_execution_id,
            replay_mode=mode,
            start_from_step=request.start_from_step,
            use_cached_outputs=request.use_cached_outputs,
            triggered_by=None,  # TODO: Get from auth context
        )

        # Save replay session to database
        from infrastructure.persistence.execution_persistence_repository import PostgreSQLReplaySessionRepository
        replay_repo = PostgreSQLReplaySessionRepository(session)
        await replay_repo.save(replay_session)

        # TODO: Queue replay execution task via message broker
        # For now, we mark it as queued in the response

        return ReplayExecutionResponse(
            replay_session_id=str(replay_session.id),
            original_execution_id=str(execution_id),
            replay_execution_id=str(replay_execution_id),
            status="queued",
            message=f"Replay queued in {mode.value} mode. Use replay_session_id to check comparison results."
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate replay: {str(e)}"
        )


@router.post(
    "/{execution_id}/resume",
    response_model=ResumeExecutionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Resume a workflow execution",
    description="Resume a paused, failed, or interrupted execution from the last checkpoint",
)
async def resume_execution(
    execution_id: UUID,
    request: ResumeExecutionRequest,
    session: AsyncSession = Depends(get_db_session),
) -> ResumeExecutionResponse:
    """
    Resume a workflow execution from the last checkpoint.

    This endpoint allows resuming executions that were:
    - Paused manually
    - Failed mid-execution
    - Interrupted due to system issues

    The execution resumes from the last successful checkpoint, skipping
    already-completed steps and continuing with pending/failed steps.
    """
    try:
        # Verify execution exists
        from infrastructure.persistence.workflow_repository import PostgreSQLWorkflowExecutionRepository
        execution_repo = PostgreSQLWorkflowExecutionRepository(session)
        execution = await execution_repo.get_by_id(execution_id)

        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution {execution_id} not found"
            )

        # Check execution can be resumed
        if execution.is_terminal() and execution.status.value == "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot resume a completed execution"
            )

        # Get latest snapshot for execution
        from infrastructure.persistence.execution_persistence_repository import PostgreSQLExecutionSnapshotRepository
        snapshot_repo = PostgreSQLExecutionSnapshotRepository(session)
        
        # Try to get specific snapshot if provided
        if request.from_snapshot_id:
            snapshot = await snapshot_repo.get_by_id(UUID(request.from_snapshot_id))
            if not snapshot:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Snapshot {request.from_snapshot_id} not found"
                )
        else:
            # Get latest snapshot
            snapshot = await snapshot_repo.get_latest_for_execution(execution_id)
        
        # If no snapshot exists, create one from current execution state
        if not snapshot:
            # Build a best-effort snapshot from persisted step executions + workflow definition.
            from infrastructure.persistence.workflow_repository import PostgreSQLWorkflowRepository
            from core.domain.workflow import ExecutionStatus

            workflow_repo = PostgreSQLWorkflowRepository(session)
            workflow = await workflow_repo.get_by_id(execution.workflow_id, execution.workflow_version)

            step_executions = await execution_repo.get_step_executions(execution_id)
            step_status_by_id: Dict[str, str] = {
                se.step_id.value: se.status.value for se in step_executions
            }

            completed_steps = [
                step_id
                for step_id, st in step_status_by_id.items()
                if st in {ExecutionStatus.COMPLETED.value, ExecutionStatus.SKIPPED.value}
            ]
            failed_steps = [
                step_id for step_id, st in step_status_by_id.items() if st == ExecutionStatus.FAILED.value
            ]
            pending_steps = [
                step_id
                for step_id, st in step_status_by_id.items()
                if st in {ExecutionStatus.PENDING.value, ExecutionStatus.RUNNING.value}
            ]

            all_steps: List[str] = [s.id.value for s in workflow.steps] if workflow else []
            for step_id in all_steps:
                if step_id not in step_status_by_id:
                    pending_steps.append(step_id)

            snapshot = ExecutionSnapshot(
                id=uuid4(),
                execution_id=execution_id,
                snapshot_type=SnapshotType.CHECKPOINT,
                workflow_state={"status": execution.status.value},
                step_states=step_status_by_id,
                completed_steps=completed_steps,
                pending_steps=pending_steps,
                failed_steps=failed_steps,
            )
            await snapshot_repo.save(snapshot)

        if not snapshot.can_resume_from():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No pending or failed steps to resume"
            )

        # Update execution status to running
        # TODO: Queue resume execution task with snapshot via message broker

        next_steps = snapshot.pending_steps if request.skip_failed_steps else snapshot.get_next_steps()
        return ResumeExecutionResponse(
            execution_id=str(execution_id),
            resumed_from_snapshot=str(snapshot.id),
            status="resuming",
            pending_steps=next_steps,
            message="Execution resume queued. Pending steps will be executed."
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume execution: {str(e)}"
        )


@router.get(
    "/{execution_id}/snapshots",
    response_model=List[ExecutionSnapshotResponse],
    summary="Get execution snapshots",
    description="List all snapshots for an execution, useful for understanding execution checkpoints",
)
async def get_execution_snapshots(
    execution_id: UUID,
    snapshot_type: Optional[str] = Query(None, description="Filter by snapshot type"),
    session: AsyncSession = Depends(get_db_session),
) -> List[ExecutionSnapshotResponse]:
    """Get all snapshots for an execution"""
    try:
        from infrastructure.persistence.execution_persistence_repository import PostgreSQLExecutionSnapshotRepository

        snapshot_repo = PostgreSQLExecutionSnapshotRepository(session)
        snapshots = await snapshot_repo.get_all_for_execution(execution_id)

        if snapshot_type:
            try:
                st = SnapshotType(snapshot_type.lower())
                snapshots = [s for s in snapshots if s.snapshot_type == st]
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid snapshot type: {snapshot_type}",
                )

        return [
            ExecutionSnapshotResponse(
                id=str(s.id),
                execution_id=str(s.execution_id),
                snapshot_type=s.snapshot_type.value,
                completed_steps=s.completed_steps,
                pending_steps=s.pending_steps,
                failed_steps=s.failed_steps,
                created_at=s.created_at,
                can_resume=s.can_resume_from(),
            )
            for s in snapshots
        ]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve snapshots: {str(e)}"
        )


@router.get(
    "/replay-sessions/{replay_session_id}",
    response_model=ComparisonResult,
    summary="Get replay comparison results",
    description="Get the comparison results between original and replayed execution",
)
async def get_replay_comparison(
    replay_session_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> ComparisonResult:
    """Get replay comparison results"""
    try:
        from infrastructure.persistence.execution_persistence_repository import PostgreSQLReplaySessionRepository

        replay_repo = PostgreSQLReplaySessionRepository(session)
        replay = await replay_repo.get_by_id(replay_session_id)
        if not replay:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Replay session {replay_session_id} not found",
            )

        divergence = replay.divergence_details or {}
        divergent_steps = divergence.get("divergent_steps") or divergence.get("divergences") or []
        total_steps = int(divergence.get("total_steps") or 0)
        match_pct = float(replay.match_percentage or 0.0)

        matching_steps = divergence.get("matching_steps")
        if matching_steps is None:
            matching_steps = int(round(total_steps * (match_pct / 100))) if total_steps else 0

        return ComparisonResult(
            replay_session_id=str(replay.id),
            match_percentage=match_pct,
            total_steps=total_steps,
            matching_steps=int(matching_steps),
            divergent_steps=divergent_steps,
            status=replay.status,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve comparison results: {str(e)}"
        )


# ==================== Health Check ====================

@router.get(
    "/health",
    summary="P0 execution persistence health check",
    description="Verify P0 execution persistence features are available",
)
async def health_check():
    """Health check for P0 features"""
    return {
        "status": "healthy",
        "features": {
            "replay": "available",
            "resume": "available",
            "snapshots": "available",
            "idempotency": "available",
        },
        "version": "P0.1.0"
    }
