"""
Agent Runs v4 API.

Stable, tenant-scoped endpoints for persisted single-agent runtime executions.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.domain.agent_run.entities import (
    AgentRun,
    AgentRunArtifact,
    AgentRunEvaluation,
    AgentRunStep,
    ArtifactKind,
    RunStatus,
    StepStatus,
    StepType,
)
from infrastructure.persistence.agent_run_repository import PostgreSQLAgentRunRepository
from infrastructure.tasks.celery_config import celery_app
from services.api.src.dependencies.injection import get_db_session

from core.domain.billing.entities import SubscriptionTier
from core.domain.billing.services import EntitlementService
from infrastructure.billing.feature_gate_service import require_pro_tier
from infrastructure.persistence.billing_repository import (
    PostgreSQLSubscriptionRepository,
    PostgreSQLUsageRecordRepository,
)


router = APIRouter(prefix="/api/v4/agent-runs", tags=["Agent Runs (v4)"])


def _require_tenant_id(request: Request) -> UUID:
    tenant_id = getattr(request.state, "tenant_id", None)
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context required (provide X-Tenant-Slug or authenticate via API key).",
        )
    return tenant_id


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    return str(value)


async def _get_repo(session: AsyncSession = Depends(get_db_session)) -> PostgreSQLAgentRunRepository:
    return PostgreSQLAgentRunRepository(session)


class CreateAgentRunRequest(BaseModel):
    requirement: str = Field(..., min_length=1, max_length=20000)
    budgets: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentRunResponse(BaseModel):
    id: UUID
    tenant_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    requirement: str
    status: RunStatus
    budgets: Dict[str, Any]
    usage: Dict[str, Any]
    error: Optional[str] = None
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @staticmethod
    def from_entity(run: AgentRun) -> "AgentRunResponse":
        return AgentRunResponse(
            id=run.id,
            tenant_id=run.tenant_id,
            user_id=run.user_id,
            requirement=run.requirement,
            status=run.status,
            budgets=run.budgets,
            usage=run.usage,
            error=run.error,
            metadata=run.metadata,
            created_at=run.created_at,
            updated_at=run.updated_at,
            started_at=run.started_at,
            completed_at=run.completed_at,
        )


class AgentRunStepResponse(BaseModel):
    id: UUID
    run_id: UUID
    step_index: int
    step_type: StepType
    status: StepStatus
    input: Dict[str, Any]
    output: Optional[Dict[str, Any]] = None
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    tokens_used: int = 0
    cost_usd: Decimal = Decimal("0")
    duration_ms: Optional[int] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime

    @staticmethod
    def from_entity(step: AgentRunStep) -> "AgentRunStepResponse":
        return AgentRunStepResponse(
            id=step.id,
            run_id=step.run_id,
            step_index=step.step_index,
            step_type=step.step_type,
            status=step.status,
            input=step.input,
            output=step.output,
            tool_calls=step.tool_calls,
            tokens_used=step.tokens_used,
            cost_usd=step.cost_usd,
            duration_ms=step.duration_ms,
            error=step.error,
            started_at=step.started_at,
            completed_at=step.completed_at,
            created_at=step.created_at,
        )


class AgentRunArtifactResponse(BaseModel):
    id: UUID
    run_id: UUID
    kind: ArtifactKind
    name: str
    content_type: str
    step_id: Optional[UUID] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    storage_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    @staticmethod
    def from_entity(artifact: AgentRunArtifact) -> "AgentRunArtifactResponse":
        return AgentRunArtifactResponse(
            id=artifact.id,
            run_id=artifact.run_id,
            kind=artifact.kind,
            name=artifact.name,
            content_type=artifact.content_type,
            step_id=artifact.step_id,
            sha256=artifact.sha256,
            size_bytes=artifact.size_bytes,
            storage_url=artifact.storage_url,
            metadata=artifact.metadata,
            created_at=artifact.created_at,
        )


class AgentRunEvaluationResponse(BaseModel):
    id: UUID
    run_id: UUID
    success: bool
    confidence: float
    quality_scores: Dict[str, Any] = Field(default_factory=dict)
    policy_violations: List[Dict[str, Any]] = Field(default_factory=list)
    retry_plan: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    @staticmethod
    def from_entity(evaluation: AgentRunEvaluation) -> "AgentRunEvaluationResponse":
        return AgentRunEvaluationResponse(
            id=evaluation.id,
            run_id=evaluation.run_id,
            success=evaluation.success,
            confidence=evaluation.confidence,
            quality_scores=evaluation.quality_scores,
            policy_violations=evaluation.policy_violations,
            retry_plan=evaluation.retry_plan,
            created_at=evaluation.created_at,
        )


class StartAgentRunResponse(BaseModel):
    run_id: UUID
    status: RunStatus
    task_id: Optional[str] = None
    queued_at: datetime


@router.post("", response_model=AgentRunResponse, status_code=status.HTTP_201_CREATED)
async def create_agent_run(
    request: Request,
    payload: CreateAgentRunRequest,
    repo: PostgreSQLAgentRunRepository = Depends(_get_repo),
) -> AgentRunResponse:
    tenant_id = _require_tenant_id(request)
    run = AgentRun.create(
        requirement=payload.requirement,
        tenant_id=tenant_id,
        budgets=payload.budgets,
        metadata=payload.metadata,
    )
    created = await repo.create_run(run)
    return AgentRunResponse.from_entity(created)


@router.get("", response_model=List[AgentRunResponse])
async def list_agent_runs(
    request: Request,
    limit: int = 50,
    repo: PostgreSQLAgentRunRepository = Depends(_get_repo),
) -> List[AgentRunResponse]:
    tenant_id = _require_tenant_id(request)
    runs = await repo.list_runs(tenant_id=tenant_id, limit=max(1, min(200, limit)))
    return [AgentRunResponse.from_entity(r) for r in runs]


@router.get("/{run_id}", response_model=AgentRunResponse)
async def get_agent_run(
    request: Request,
    run_id: UUID,
    repo: PostgreSQLAgentRunRepository = Depends(_get_repo),
) -> AgentRunResponse:
    tenant_id = _require_tenant_id(request)
    run = await repo.get_run(run_id=run_id, tenant_id=tenant_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent run not found.")
    return AgentRunResponse.from_entity(run)


@router.post("/{run_id}/start", response_model=StartAgentRunResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_agent_run(
    request: Request,
    run_id: UUID,
    session: AsyncSession = Depends(get_db_session),
    repo: PostgreSQLAgentRunRepository = Depends(_get_repo),
) -> StartAgentRunResponse:
    tenant_id = _require_tenant_id(request)
    run = await repo.get_run(run_id=run_id, tenant_id=tenant_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent run not found.")

    if run.is_terminal():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent run is already terminal ({run.status.value}).",
        )

    if run.status in {RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.VALIDATING}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent run is already started ({run.status.value}).",
        )

    # Monetization & abuse protection: enforce concurrency and monthly execution quotas.
    active_runs = 0
    try:
        active_runs = int(await repo.count_active_runs(tenant_id))
    except Exception:
        active_runs = 0

    subscription_repo = PostgreSQLSubscriptionRepository(session)
    usage_repo = PostgreSQLUsageRecordRepository(session)
    subscription = await subscription_repo.get_by_tenant(tenant_id)

    # Determine tier (subscription if present, else tenant metadata fallback).
    if subscription:
        tier = subscription.tier
    else:
        tier_value = getattr(getattr(request.state, "tenant", None), "subscription_tier", None) or "free"
        try:
            tier = SubscriptionTier(str(tier_value))
        except Exception:
            tier = SubscriptionTier.FREE

    concurrency_limits = {
        SubscriptionTier.FREE: 1,
        SubscriptionTier.PRO: 5,
        SubscriptionTier.TEAM: 20,
        SubscriptionTier.ENTERPRISE: None,  # Unlimited
    }
    max_active = concurrency_limits.get(tier, 1)
    if max_active is not None and active_runs >= int(max_active):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Concurrency limit exceeded for tier {tier.value} (active_runs={active_runs}, limit={max_active}).",
        )

    # Quota check uses reserved executions = (active_runs + 1) to prevent unbounded queueing.
    if subscription:
        entitlement_service = EntitlementService(
            subscription_repository=subscription_repo,
            usage_repository=usage_repo,
        )
        check = await entitlement_service.check_entitlement(
            tenant_id=tenant_id,
            resource_type="executions",
            quantity=Decimal(str(active_runs + 1)),
        )
        if not check.allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=check.reason or "Execution quota exceeded",
            )
    else:
        limit = EntitlementService.TIER_LIMITS.get(tier, {}).get("executions")
        if limit is not None:
            now = datetime.utcnow()
            month_start = datetime(now.year, now.month, 1)
            current_usage = await usage_repo.aggregate_usage(
                tenant_id=tenant_id,
                resource_type="executions",
                start_time=month_start,
                end_time=now,
            )
            if current_usage + Decimal(str(active_runs + 1)) > limit:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Usage limit exceeded for executions",
                )

    run.mark_queued()
    await repo.update_run(run)

    try:
        async_result = celery_app.send_task(
            "infrastructure.tasks.agent_tasks.execute_agent_run",
            kwargs={"run_id": str(run.id), "tenant_id": str(tenant_id)},
        )
    except Exception as exc:  # noqa: BLE001
        run.fail(error=f"Failed to enqueue agent run: {exc}")
        await repo.update_run(run)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to enqueue agent run. Worker queue unavailable.",
        ) from exc
    return StartAgentRunResponse(
        run_id=run.id,
        status=RunStatus.QUEUED,
        task_id=async_result.id if async_result else None,
        queued_at=datetime.utcnow(),
    )


@router.post("/{run_id}:start", response_model=StartAgentRunResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_agent_run_action(
    request: Request,
    run_id: UUID,
    session: AsyncSession = Depends(get_db_session),
    repo: PostgreSQLAgentRunRepository = Depends(_get_repo),
) -> StartAgentRunResponse:
    """Action-style alias for starting a run (`{id}:start`)."""
    return await start_agent_run(request=request, run_id=run_id, session=session, repo=repo)


@router.post("/{run_id}/cancel", response_model=AgentRunResponse)
async def cancel_agent_run(
    request: Request,
    run_id: UUID,
    reason: str = "cancelled",
    repo: PostgreSQLAgentRunRepository = Depends(_get_repo),
) -> AgentRunResponse:
    tenant_id = _require_tenant_id(request)
    run = await repo.get_run(run_id=run_id, tenant_id=tenant_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent run not found.")

    if run.is_terminal():
        return AgentRunResponse.from_entity(run)

    run.cancel(reason=reason)
    updated = await repo.update_run(run)
    return AgentRunResponse.from_entity(updated)


@router.post("/{run_id}:cancel", response_model=AgentRunResponse)
async def cancel_agent_run_action(
    request: Request,
    run_id: UUID,
    reason: str = "cancelled",
    repo: PostgreSQLAgentRunRepository = Depends(_get_repo),
) -> AgentRunResponse:
    """Action-style alias for cancelling a run (`{id}:cancel`)."""
    return await cancel_agent_run(request=request, run_id=run_id, reason=reason, repo=repo)


class RerunAgentRunRequest(BaseModel):
    budgets: Optional[Dict[str, Any]] = Field(default=None, description="Optional budgets override/patch.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata patch.")


@router.post("/{run_id}:rerun", response_model=AgentRunResponse, status_code=status.HTTP_201_CREATED)
async def rerun_agent_run_action(
    request: Request,
    run_id: UUID,
    payload: RerunAgentRunRequest,
    repo: PostgreSQLAgentRunRepository = Depends(_get_repo),
) -> AgentRunResponse:
    """Create a new AgentRun using the previous run as context (`{id}:rerun`)."""
    tenant_id = _require_tenant_id(request)
    prior = await repo.get_run(run_id=run_id, tenant_id=tenant_id)
    if not prior:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent run not found.")

    budgets = dict(prior.budgets or {})
    if payload.budgets:
        budgets.update(payload.budgets)

    metadata = dict(prior.metadata or {})
    metadata.update({"rerun_of": str(prior.id), "rerun_created_at": datetime.utcnow().isoformat()})
    if payload.metadata:
        metadata.update(payload.metadata)

    new_run = AgentRun.create(
        requirement=prior.requirement,
        tenant_id=tenant_id,
        user_id=prior.user_id,
        budgets=budgets,
        metadata=metadata,
    )
    created = await repo.create_run(new_run)
    return AgentRunResponse.from_entity(created)


@router.get("/{run_id}/steps", response_model=List[AgentRunStepResponse])
async def list_agent_run_steps(
    request: Request,
    run_id: UUID,
    repo: PostgreSQLAgentRunRepository = Depends(_get_repo),
) -> List[AgentRunStepResponse]:
    tenant_id = _require_tenant_id(request)
    run = await repo.get_run(run_id=run_id, tenant_id=tenant_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent run not found.")

    steps = await repo.list_steps(run_id=run_id)
    return [AgentRunStepResponse.from_entity(s) for s in steps]


@router.get("/{run_id}/artifacts", response_model=List[AgentRunArtifactResponse])
async def list_agent_run_artifacts(
    request: Request,
    run_id: UUID,
    repo: PostgreSQLAgentRunRepository = Depends(_get_repo),
) -> List[AgentRunArtifactResponse]:
    tenant_id = _require_tenant_id(request)
    run = await repo.get_run(run_id=run_id, tenant_id=tenant_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent run not found.")

    artifacts = await repo.list_artifacts(run_id=run_id)
    return [AgentRunArtifactResponse.from_entity(a) for a in artifacts]


@router.get("/{run_id}/evaluation", response_model=Optional[AgentRunEvaluationResponse])
async def get_latest_evaluation(
    request: Request,
    run_id: UUID,
    repo: PostgreSQLAgentRunRepository = Depends(_get_repo),
) -> Optional[AgentRunEvaluationResponse]:
    tenant_id = _require_tenant_id(request)
    run = await repo.get_run(run_id=run_id, tenant_id=tenant_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent run not found.")

    evaluation = await repo.get_latest_evaluation(run_id=run_id)
    return AgentRunEvaluationResponse.from_entity(evaluation) if evaluation else None


@router.get("/{run_id}/events")
async def stream_agent_run_events(
    request: Request,
    run_id: UUID,
    _: None = Depends(require_pro_tier()),
):
    """Stream agent run events via SSE. Requires Pro tier or higher."""
    tenant_id = _require_tenant_id(request)

    async def event_stream():
        # Streaming responses outlive request-scoped dependencies; create short-lived DB sessions per poll.
        from services.api.src.dependencies.injection import async_session_factory, get_engine

        get_engine()
        last_status: Optional[str] = None
        last_step_count = -1
        last_artifact_count = -1
        poll_s = 1.0

        while True:
            async with async_session_factory() as session:
                repo = PostgreSQLAgentRunRepository(session)
                run = await repo.get_run(run_id=run_id, tenant_id=tenant_id)
                if not run:
                    yield "event: error\ndata: {\"error\": \"not_found\"}\n\n"
                    return

                steps = await repo.list_steps(run_id=run_id)
                artifacts = await repo.list_artifacts(run_id=run_id)

            changed = (
                run.status.value != last_status
                or len(steps) != last_step_count
                or len(artifacts) != last_artifact_count
            )
            if changed:
                payload = {
                    "run": AgentRunResponse.from_entity(run).model_dump(),
                    "steps": [AgentRunStepResponse.from_entity(s).model_dump() for s in steps],
                    "artifacts": [AgentRunArtifactResponse.from_entity(a).model_dump() for a in artifacts],
                }
                yield f"event: update\ndata: {json.dumps(payload, default=_json_default)}\n\n"
                last_status = run.status.value
                last_step_count = len(steps)
                last_artifact_count = len(artifacts)

            if run.is_terminal():
                yield "event: end\ndata: {}\n\n"
                return

            await asyncio.sleep(poll_s)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/{run_id}/stream")
async def stream_agent_run_stream(
    request: Request,
    run_id: UUID,
    _: None = Depends(require_pro_tier()),
):
    """Alias for SSE streaming (`/{id}/stream`). Requires Pro tier or higher."""
    return await stream_agent_run_events(request=request, run_id=run_id)
