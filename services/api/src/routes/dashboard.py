"""
Local Dashboard & Task Introspection Routes

These endpoints exist to support the bundled Next.js dashboard in `frontend/`
without requiring additional microservices for localhost development.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.src.dependencies.injection import (
    get_db_session,
    check_database_health,
    check_redis_health,
    check_rabbitmq_health,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3", tags=["Dashboard"])


class TimelineSummary(BaseModel):
    events: List[Dict[str, Any]] = Field(default_factory=list)


class DashboardTimelineSummary(BaseModel):
    total_tokens: int = 0
    total_cost_usd: float = 0.0


class DashboardResponse(BaseModel):
    system_health: str
    service_metrics: Dict[str, Any] = Field(default_factory=dict)
    error_rates: Dict[str, float] = Field(default_factory=dict)
    latency_percentiles: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    active_alerts: List[Dict[str, Any]] = Field(default_factory=list)
    recent_failures: List[Dict[str, Any]] = Field(default_factory=list)
    timeline_summary: DashboardTimelineSummary = Field(default_factory=DashboardTimelineSummary)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ActiveTask(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    status: str


class ExplainRequest(BaseModel):
    task_id: UUID
    level: str = Field(default="standard")
    include_timeline: bool = Field(default=True)
    include_reasoning: bool = Field(default=True)
    include_confidence: bool = Field(default=True)


class ReasoningDecision(BaseModel):
    step: int
    type: str
    description: str
    confidence: Optional[float] = None
    alternatives: Optional[List[Any]] = None
    rationale: Optional[str] = None


class ReasoningSummary(BaseModel):
    total_steps: int
    reasoning_summary: str
    key_decisions: List[ReasoningDecision] = Field(default_factory=list)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    overall_quality: float


class ConfidenceAnalysis(BaseModel):
    average_confidence: float
    recommendation: str


class ExplainResponse(BaseModel):
    task_id: UUID
    summary: str
    reasoning_summary: Optional[ReasoningSummary] = None
    timeline_summary: Optional[TimelineSummary] = None
    confidence_analysis: Optional[ConfidenceAnalysis] = None


@router.get(
    "/dashboard",
    response_model=DashboardResponse,
    summary="Dashboard data for localhost UI",
    description="Aggregated dashboard data used by the bundled Next.js frontend.",
)
async def get_dashboard() -> DashboardResponse:
    db_healthy = await check_database_health()
    redis_healthy = await check_redis_health()
    rabbitmq_healthy = await check_rabbitmq_health()

    if not db_healthy:
        system_health = "degraded"
    elif not (redis_healthy and rabbitmq_healthy):
        system_health = "warning"
    else:
        system_health = "healthy"

    return DashboardResponse(
        system_health=system_health,
        # Keep the shape expected by the UI; leave detailed metrics to Prometheus (/metrics).
        service_metrics={},
        error_rates={},
        latency_percentiles={},
        active_alerts=[],
        recent_failures=[],
    )


@router.get(
    "/tasks/active",
    response_model=List[ActiveTask],
    summary="List active tasks",
    description="Returns running workflow executions as 'tasks' for the localhost UI.",
)
async def list_active_tasks(
    session: AsyncSession = Depends(get_db_session),
) -> List[ActiveTask]:
    try:
        from infrastructure.persistence.workflow_repository import PostgreSQLWorkflowExecutionRepository

        repo = PostgreSQLWorkflowExecutionRepository(session)
        executions = await repo.get_active_executions()

        tasks: List[ActiveTask] = []
        for execution in executions:
            tasks.append(
                ActiveTask(
                    id=str(execution.id),
                    title=f"Workflow {execution.workflow_id.value}",
                    description=f"Execution {execution.id} ({execution.status.value})",
                    status=execution.status.value,
                )
            )

        return tasks
    except Exception:
        logger.exception("Failed to list active tasks; returning empty list")
        return []


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Explain a task/execution",
    description="Generates a lightweight explanation of a workflow execution for the localhost UI.",
)
async def explain_task(
    request: ExplainRequest,
    session: AsyncSession = Depends(get_db_session),
) -> ExplainResponse:
    try:
        from infrastructure.persistence.workflow_repository import PostgreSQLWorkflowExecutionRepository

        repo = PostgreSQLWorkflowExecutionRepository(session)
        execution = await repo.get_by_id(request.task_id)
        if not execution:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task/execution not found: {request.task_id}",
            )

        steps = await repo.get_step_executions(request.task_id)

        summary = (
            f"Workflow execution {execution.id} for workflow {execution.workflow_id.value} "
            f"(v{execution.workflow_version}) is {execution.status.value}."
        )

        reasoning_summary: Optional[ReasoningSummary] = None
        if request.include_reasoning:
            key_decisions: List[ReasoningDecision] = []
            confidence_by_type: Dict[str, List[float]] = {}

            for idx, step in enumerate(steps, start=1):
                step_status = step.status.value
                if step_status == "completed":
                    confidence = 0.9
                elif step_status == "running":
                    confidence = 0.65
                elif step_status == "failed":
                    confidence = 0.35
                else:
                    confidence = 0.5

                key_decisions.append(
                    ReasoningDecision(
                        step=idx,
                        type=step.step_type or "step",
                        description=f"Step {step.step_id.value} is {step_status}.",
                        confidence=confidence,
                        rationale="Derived from recorded step execution status.",
                    )
                )
                confidence_by_type.setdefault(step.step_type or "step", []).append(confidence)

            confidence_scores = {
                step_type: (sum(scores) / len(scores)) for step_type, scores in confidence_by_type.items()
            }
            overall_quality = (
                (sum(confidence_scores.values()) / len(confidence_scores)) if confidence_scores else 0.75
            )

            reasoning_summary = ReasoningSummary(
                total_steps=len(steps),
                reasoning_summary=(
                    f"Explanation derived from execution metadata for {len(steps)} step(s). "
                    "For richer traces, enable instrumentation and persist reasoning events."
                ),
                key_decisions=key_decisions,
                confidence_scores=confidence_scores,
                overall_quality=max(0.0, min(1.0, overall_quality)),
            )

        timeline_summary: Optional[TimelineSummary] = None
        if request.include_timeline:
            events: List[Dict[str, Any]] = []
            for step in steps:
                duration_ms = 0
                if step.started_at and step.completed_at:
                    duration_ms = int((step.completed_at - step.started_at).total_seconds() * 1000)

                events.append(
                    {
                        "name": f"{step.step_id.value} ({step.status.value})",
                        "duration_ms": duration_ms,
                    }
                )
            timeline_summary = TimelineSummary(events=events)

        confidence_analysis: Optional[ConfidenceAnalysis] = None
        if request.include_confidence:
            average_confidence = reasoning_summary.overall_quality if reasoning_summary else 0.75
            if average_confidence >= 0.8:
                recommendation = "Execution looks healthy; proceed or start the next workflow."
            elif average_confidence >= 0.6:
                recommendation = "Monitor execution; investigate slow or pending steps if it stalls."
            else:
                recommendation = "Review failed steps and retry the execution after addressing errors."

            confidence_analysis = ConfidenceAnalysis(
                average_confidence=max(0.0, min(1.0, float(average_confidence))),
                recommendation=recommendation,
            )

        return ExplainResponse(
            task_id=request.task_id,
            summary=summary,
            reasoning_summary=reasoning_summary,
            timeline_summary=timeline_summary,
            confidence_analysis=confidence_analysis,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to explain task")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to explain task: {e}",
        )

