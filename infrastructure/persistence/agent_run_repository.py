"""
Agent Run Repository Implementation

PostgreSQL implementation of the AgentRunRepository interface.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import List, Optional
from uuid import UUID

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.domain.agent_run.entities import (
    AgentRun,
    AgentRunArtifact,
    AgentRunEvaluation,
    AgentRunMemoryLink,
    AgentRunStep,
    ArtifactKind,
    RunStatus,
    StepStatus,
    StepType,
)
from core.domain.agent_run.repositories import AgentRunRepository
from infrastructure.persistence.agent_run_models import (
    AgentRunArtifactModel,
    AgentRunEvaluationModel,
    AgentRunMemoryLinkModel,
    AgentRunModel,
    AgentRunStepModel,
)


logger = logging.getLogger(__name__)


class PostgreSQLAgentRunRepository(AgentRunRepository):
    """PostgreSQL implementation of AgentRunRepository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_run(self, run: AgentRun) -> AgentRun:
        model = self._run_to_model(run)
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        return self._run_to_entity(model)

    async def update_run(self, run: AgentRun) -> AgentRun:
        stmt = select(AgentRunModel).where(AgentRunModel.id == run.id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()

        if model is None:
            raise ValueError(f"Agent run not found: {run.id}")

        model.requirement = run.requirement
        model.status = run.status.value
        model.budgets = run.budgets
        model.usage = run.usage
        model.error = run.error
        model.run_metadata = run.metadata
        model.updated_at = run.updated_at
        model.started_at = run.started_at
        model.completed_at = run.completed_at

        await self.session.flush()
        await self.session.refresh(model)
        return self._run_to_entity(model)

    async def get_run(self, run_id: UUID, tenant_id: Optional[UUID]) -> Optional[AgentRun]:
        stmt = select(AgentRunModel).where(AgentRunModel.id == run_id)
        if tenant_id is not None:
            stmt = stmt.where(AgentRunModel.tenant_id == tenant_id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._run_to_entity(model) if model else None

    async def list_runs(self, tenant_id: Optional[UUID], limit: int = 50) -> List[AgentRun]:
        stmt = select(AgentRunModel)
        if tenant_id is not None:
            stmt = stmt.where(AgentRunModel.tenant_id == tenant_id)
        stmt = stmt.order_by(desc(AgentRunModel.created_at)).limit(limit)
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._run_to_entity(m) for m in models]

    async def create_step(self, step: AgentRunStep) -> AgentRunStep:
        model = self._step_to_model(step)
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        return self._step_to_entity(model)

    async def update_step(self, step: AgentRunStep) -> AgentRunStep:
        stmt = select(AgentRunStepModel).where(AgentRunStepModel.id == step.id)
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        if model is None:
            raise ValueError(f"Agent run step not found: {step.id}")

        model.status = step.status.value
        model.input = step.input
        model.output = step.output
        model.tool_calls = step.tool_calls
        model.tokens_used = step.tokens_used
        model.cost_usd = step.cost_usd
        model.duration_ms = step.duration_ms
        model.error = step.error
        model.started_at = step.started_at
        model.completed_at = step.completed_at

        await self.session.flush()
        await self.session.refresh(model)
        return self._step_to_entity(model)

    async def list_steps(self, run_id: UUID) -> List[AgentRunStep]:
        stmt = select(AgentRunStepModel).where(AgentRunStepModel.run_id == run_id).order_by(
            AgentRunStepModel.step_index.asc()
        )
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._step_to_entity(m) for m in models]

    async def create_artifact(self, artifact: AgentRunArtifact) -> AgentRunArtifact:
        model = self._artifact_to_model(artifact)
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        return self._artifact_to_entity(model)

    async def list_artifacts(self, run_id: UUID) -> List[AgentRunArtifact]:
        stmt = (
            select(AgentRunArtifactModel)
            .where(AgentRunArtifactModel.run_id == run_id)
            .order_by(desc(AgentRunArtifactModel.created_at))
        )
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        return [self._artifact_to_entity(m) for m in models]

    async def create_evaluation(self, evaluation: AgentRunEvaluation) -> AgentRunEvaluation:
        model = self._evaluation_to_model(evaluation)
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        return self._evaluation_to_entity(model)

    async def get_latest_evaluation(self, run_id: UUID) -> Optional[AgentRunEvaluation]:
        stmt = (
            select(AgentRunEvaluationModel)
            .where(AgentRunEvaluationModel.run_id == run_id)
            .order_by(desc(AgentRunEvaluationModel.created_at))
            .limit(1)
        )
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._evaluation_to_entity(model) if model else None

    async def link_memory(self, link: AgentRunMemoryLink) -> AgentRunMemoryLink:
        model = self._memory_link_to_model(link)
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        return self._memory_link_to_entity(model)

    async def count_active_runs(self, tenant_id: UUID) -> int:
        stmt = (
            select(AgentRunModel.id)
            .where(AgentRunModel.tenant_id == tenant_id)
            .where(AgentRunModel.status.in_([RunStatus.QUEUED.value, RunStatus.RUNNING.value, RunStatus.VALIDATING.value]))
        )
        result = await self.session.execute(stmt)
        return len(result.scalars().all())

    def _run_to_model(self, entity: AgentRun) -> AgentRunModel:
        return AgentRunModel(
            id=entity.id,
            tenant_id=entity.tenant_id,
            user_id=entity.user_id,
            requirement=entity.requirement,
            status=entity.status.value,
            budgets=entity.budgets,
            usage=entity.usage,
            error=entity.error,
            run_metadata=entity.metadata,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            started_at=entity.started_at,
            completed_at=entity.completed_at,
        )

    def _run_to_entity(self, model: AgentRunModel) -> AgentRun:
        return AgentRun(
            id=model.id,
            requirement=model.requirement,
            status=RunStatus(model.status),
            tenant_id=model.tenant_id,
            user_id=model.user_id,
            budgets=model.budgets or {},
            usage=model.usage or {},
            error=model.error,
            metadata=model.run_metadata or {},
            created_at=model.created_at,
            updated_at=model.updated_at,
            started_at=model.started_at,
            completed_at=model.completed_at,
        )

    def _step_to_model(self, entity: AgentRunStep) -> AgentRunStepModel:
        return AgentRunStepModel(
            id=entity.id,
            run_id=entity.run_id,
            step_index=entity.step_index,
            step_type=entity.step_type.value,
            status=entity.status.value,
            input=entity.input,
            output=entity.output,
            tool_calls=entity.tool_calls,
            tokens_used=entity.tokens_used,
            cost_usd=entity.cost_usd,
            duration_ms=entity.duration_ms,
            error=entity.error,
            started_at=entity.started_at,
            completed_at=entity.completed_at,
            created_at=entity.created_at,
        )

    def _step_to_entity(self, model: AgentRunStepModel) -> AgentRunStep:
        return AgentRunStep(
            id=model.id,
            run_id=model.run_id,
            step_index=model.step_index,
            step_type=StepType(model.step_type),
            status=StepStatus(model.status),
            input=model.input or {},
            output=model.output,
            tool_calls=model.tool_calls or [],
            tokens_used=model.tokens_used or 0,
            cost_usd=Decimal(str(model.cost_usd)) if model.cost_usd is not None else Decimal("0"),
            duration_ms=model.duration_ms,
            error=model.error,
            started_at=model.started_at,
            completed_at=model.completed_at,
            created_at=model.created_at,
        )

    def _artifact_to_model(self, entity: AgentRunArtifact) -> AgentRunArtifactModel:
        return AgentRunArtifactModel(
            id=entity.id,
            run_id=entity.run_id,
            step_id=entity.step_id,
            kind=entity.kind.value,
            name=entity.name,
            content_type=entity.content_type,
            content_text=entity.content_text,
            content_bytes=entity.content_bytes,
            sha256=entity.sha256,
            size_bytes=entity.size_bytes,
            storage_url=entity.storage_url,
            artifact_metadata=entity.metadata,
            created_at=entity.created_at,
        )

    def _artifact_to_entity(self, model: AgentRunArtifactModel) -> AgentRunArtifact:
        return AgentRunArtifact(
            id=model.id,
            run_id=model.run_id,
            step_id=model.step_id,
            kind=ArtifactKind(model.kind),
            name=model.name,
            content_type=model.content_type,
            content_text=model.content_text,
            content_bytes=model.content_bytes,
            sha256=model.sha256,
            size_bytes=model.size_bytes,
            storage_url=model.storage_url,
            metadata=model.artifact_metadata or {},
            created_at=model.created_at,
        )

    def _evaluation_to_model(self, entity: AgentRunEvaluation) -> AgentRunEvaluationModel:
        return AgentRunEvaluationModel(
            id=entity.id,
            run_id=entity.run_id,
            success=entity.success,
            confidence=entity.confidence,
            quality_scores=entity.quality_scores,
            policy_violations=entity.policy_violations,
            retry_plan=entity.retry_plan,
            created_at=entity.created_at,
        )

    def _evaluation_to_entity(self, model: AgentRunEvaluationModel) -> AgentRunEvaluation:
        return AgentRunEvaluation(
            id=model.id,
            run_id=model.run_id,
            success=model.success,
            confidence=float(model.confidence) if model.confidence is not None else 0.0,
            quality_scores=model.quality_scores or {},
            policy_violations=model.policy_violations or [],
            retry_plan=model.retry_plan or {},
            created_at=model.created_at,
        )

    def _memory_link_to_model(self, entity: AgentRunMemoryLink) -> AgentRunMemoryLinkModel:
        return AgentRunMemoryLinkModel(
            id=entity.id,
            run_id=entity.run_id,
            memory_id=entity.memory_id,
            memory_tier=entity.memory_tier,
            relation=entity.relation,
            created_at=entity.created_at,
        )

    def _memory_link_to_entity(self, model: AgentRunMemoryLinkModel) -> AgentRunMemoryLink:
        return AgentRunMemoryLink(
            id=model.id,
            run_id=model.run_id,
            memory_id=model.memory_id,
            memory_tier=model.memory_tier,
            relation=model.relation,
            created_at=model.created_at,
        )

