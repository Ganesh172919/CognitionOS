"""
Single-Agent Runtime Kernel.

Executes a persisted AgentRun with planning, codegen, validation, evaluation, and artifacts.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_config
from core.domain.agent_run.entities import (
    AgentRunArtifact,
    AgentRunEvaluation,
    AgentRunStep,
    ArtifactKind,
    RunStatus,
    StepType,
)
from infrastructure.llm.provider import LLMRouter, create_llm_router
from infrastructure.persistence.agent_run_repository import PostgreSQLAgentRunRepository
from infrastructure.persistence.billing_repository import PostgreSQLUsageRecordRepository

from cognitionos_platform.agent.codegen import deterministic_codegen, llm_codegen
from cognitionos_platform.agent.planner import deterministic_plan, llm_plan
from cognitionos_platform.agent.validator import repair_with_llm, validate
from cognitionos_platform.runtime.settings import get_platform_runtime_settings
from cognitionos_platform.sandbox.runner_client import ToolRunnerClient


logger = logging.getLogger(__name__)


def _as_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, indent=2, default=str)


def _get_llm() -> Optional[tuple[LLMRouter, str]]:
    cfg = get_config()
    if not (cfg.llm.openai_api_key or cfg.llm.anthropic_api_key):
        return None
    try:
        llm = create_llm_router(
            openai_api_key=cfg.llm.openai_api_key,
            anthropic_api_key=cfg.llm.anthropic_api_key,
            max_retries=cfg.llm.max_retries,
            timeout=cfg.llm.timeout,
        )
        return llm, cfg.llm.default_model
    except Exception:  # noqa: BLE001
        logger.exception("Failed to initialize LLM router")
        return None


def _get_tool_runner() -> Optional[ToolRunnerClient]:
    settings = get_platform_runtime_settings()
    if not settings.tool_runner_url:
        return None
    return ToolRunnerClient(base_url=settings.tool_runner_url)


async def execute_agent_run(session: AsyncSession, *, run_id: UUID, tenant_id: UUID) -> None:
    repo = PostgreSQLAgentRunRepository(session)
    usage_repo = PostgreSQLUsageRecordRepository(session)

    run = await repo.get_run(run_id=run_id, tenant_id=tenant_id)
    if not run:
        logger.warning("Agent run not found", extra={"run_id": str(run_id), "tenant_id": str(tenant_id)})
        return

    if run.is_terminal():
        return

    if run.status == RunStatus.CANCELLED:
        return

    try:
        run.start()
        await repo.update_run(run)

        llm_info = _get_llm()
        llm = llm_info[0] if llm_info else None
        model = llm_info[1] if llm_info else ""

        sandbox = _get_tool_runner()
        agent_id = str(run.id)
        user_id = str(run.user_id or uuid4())

        total_tokens = 0
        total_cost = Decimal("0")
        policy_mode = str(run.budgets.get("policy_mode", "warn")).strip().lower()
        if policy_mode not in {"warn", "block"}:
            policy_mode = "warn"

        # Step 0: planning
        plan_step = AgentRunStep.create(run_id=run.id, step_index=0, step_type=StepType.PLANNING, input={"requirement": run.requirement})
        plan_step.start()
        plan_step = await repo.create_step(plan_step)

        if llm:
            plan, usage = await llm_plan(run.requirement, llm=llm, model=model)
        else:
            plan, usage = deterministic_plan(run.requirement), {"tokens": {"total_tokens": 0}, "cost_usd": 0.0, "model": None, "provider": None}

        step_tokens = int((usage.get("tokens") or {}).get("total_tokens") or 0)
        step_cost = Decimal(str(usage.get("cost_usd") or 0))
        total_tokens += step_tokens
        total_cost += step_cost

        plan_step.complete(output=plan, tokens_used=step_tokens, cost_usd=step_cost, tool_calls=[{"type": "llm_plan", "usage": usage}])
        plan_step = await repo.update_step(plan_step)

        await repo.create_artifact(
            AgentRunArtifact.create_text(
                run_id=run.id,
                kind=ArtifactKind.REPORT,
                name="plan.json",
                content_text=_as_json(plan),
                step_id=plan_step.id,
                content_type="application/json",
            )
        )

        # Step 1: codegen
        code_step = AgentRunStep.create(run_id=run.id, step_index=1, step_type=StepType.CODEGEN, input={"plan_version": plan.get("version")})
        code_step.start()
        code_step = await repo.create_step(code_step)

        if llm:
            codegen, usage = await llm_codegen(run.requirement, plan, llm=llm, model=model)
        else:
            codegen, usage = deterministic_codegen(run.requirement)

        step_tokens = int((usage.get("tokens") or {}).get("total_tokens") or 0)
        step_cost = Decimal(str(usage.get("cost_usd") or 0))
        total_tokens += step_tokens
        total_cost += step_cost

        code_step.complete(output={"codegen": {k: v for k, v in codegen.items() if k != "files"}}, tokens_used=step_tokens, cost_usd=step_cost, tool_calls=[{"type": "llm_codegen", "usage": usage}])
        code_step = await repo.update_step(code_step)

        files = codegen.get("files", [])
        await repo.create_artifact(
            AgentRunArtifact.create_text(
                run_id=run.id,
                kind=ArtifactKind.CODE,
                name="generated_files.json",
                content_text=_as_json({"files": files, "notes": codegen.get("notes")}),
                step_id=code_step.id,
                content_type="application/json",
            )
        )

        # Step 2: validation (+ repair)
        validation_step = AgentRunStep.create(
            run_id=run.id,
            step_index=2,
            step_type=StepType.VALIDATION,
            input={"files": [f.get("path") for f in files]},
        )
        validation_step.start()
        validation_step = await repo.create_step(validation_step)

        validation = await validate(
            files,
            sandbox=sandbox,
            user_id=user_id,
            agent_id=agent_id,
            policy_mode=policy_mode,
        )
        repair_attempts = int(run.budgets.get("max_repair_attempts", 2))
        attempts_used = 0
        while (not validation.get("ok")) and llm and attempts_used < repair_attempts and files:
            attempts_used += 1
            repaired, usage = await repair_with_llm(
                requirement=run.requirement,
                plan=plan,
                previous=codegen,
                validation=validation,
                llm=llm,
                model=model,
            )
            step_tokens = int((usage.get("tokens") or {}).get("total_tokens") or 0)
            step_cost = Decimal(str(usage.get("cost_usd") or 0))
            total_tokens += step_tokens
            total_cost += step_cost

            codegen = repaired
            files = codegen.get("files", files)
            validation = await validate(
                files,
                sandbox=sandbox,
                user_id=user_id,
                agent_id=agent_id,
                policy_mode=policy_mode,
            )

        validation_step.complete(
            output={"validation": validation, "repair_attempts_used": attempts_used},
            tokens_used=0,
            cost_usd=Decimal("0"),
            tool_calls=[{"type": "validation", "sandbox": bool(sandbox)}],
        )
        validation_step = await repo.update_step(validation_step)

        await repo.create_artifact(
            AgentRunArtifact.create_text(
                run_id=run.id,
                kind=ArtifactKind.REPORT,
                name="validation_report.json",
                content_text=_as_json(validation),
                step_id=validation_step.id,
                content_type="application/json",
            )
        )

        # Step 3: evaluation
        success = bool(validation.get("ok"))
        confidence = 0.9 if success else 0.25
        evaluation = AgentRunEvaluation.create(
            run_id=run.id,
            success=success,
            confidence=confidence,
            quality_scores={
                "files": len(files),
                "tokens": total_tokens,
                "cost_usd": str(total_cost),
            },
            policy_violations=list(validation.get("policy_violations") or []),
            retry_plan={"repair_attempts_used": attempts_used} if not success else {},
        )
        await repo.create_evaluation(evaluation)

        # Usage accounting via UsageMeterService
        if run.tenant_id:
            from infrastructure.billing.usage_meter_service import UsageMeterService

            meter = UsageMeterService(usage_record_repository=usage_repo)
            await meter.record_agent_run_usage(
                tenant_id=run.tenant_id,
                run_id=run.id,
                total_tokens=total_tokens,
                cost_usd=total_cost,
                model=model or None,
            )

        run.usage = {"tokens": total_tokens, "cost_usd": str(total_cost)}
        if success:
            run.complete()
        else:
            run.fail("Validation failed for generated artifacts")
        await repo.update_run(run)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent run execution failed", extra={"run_id": str(run_id), "tenant_id": str(tenant_id)})
        run = await repo.get_run(run_id=run_id, tenant_id=tenant_id)
        if run and not run.is_terminal():
            run.fail(str(exc))
            await repo.update_run(run)
        raise
