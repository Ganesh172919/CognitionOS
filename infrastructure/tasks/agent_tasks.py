"""
Agent Run Async Tasks

Celery tasks for executing persisted agent runs.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from infrastructure.tasks.celery_config import celery_app


logger = logging.getLogger(__name__)


async def _execute_agent_run_async(*, run_id: UUID, tenant_id: UUID) -> None:
    from services.api.src.dependencies.injection import async_session_factory, get_engine

    # Ensure engine + session factory are initialized in this worker process.
    get_engine()

    async with async_session_factory() as session:
        try:
            from cognitionos_platform.agent.kernel import execute_agent_run

            await execute_agent_run(session, run_id=run_id, tenant_id=tenant_id)
            await session.commit()
        except Exception:  # noqa: BLE001
            await session.rollback()
            raise


@celery_app.task(name="infrastructure.tasks.agent_tasks.execute_agent_run", bind=True)
def execute_agent_run(self, run_id: str, tenant_id: str) -> None:
    """Execute an AgentRun by ID (tenant-scoped)."""
    try:
        asyncio.run(_execute_agent_run_async(run_id=UUID(run_id), tenant_id=UUID(tenant_id)))
    except Exception as exc:  # noqa: BLE001
        logger.error("Agent run task failed", extra={"run_id": run_id, "tenant_id": tenant_id, "error": str(exc)})
        raise

