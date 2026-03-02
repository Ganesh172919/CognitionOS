from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI, Request
from httpx import AsyncClient

from core.domain.agent_run.entities import AgentRun, AgentRunArtifact, AgentRunEvaluation, AgentRunMemoryLink, AgentRunStep
from cognitionos_platform.api.v4.routers.agent_runs import router as agent_runs_router, _get_repo


class FakeAgentRunRepo:
    def __init__(self):
        self.runs: Dict[UUID, AgentRun] = {}

    async def create_run(self, run: AgentRun) -> AgentRun:
        self.runs[run.id] = run
        return run

    async def update_run(self, run: AgentRun) -> AgentRun:
        self.runs[run.id] = run
        return run

    async def get_run(self, run_id: UUID, tenant_id: Optional[UUID]) -> Optional[AgentRun]:
        run = self.runs.get(run_id)
        if not run:
            return None
        if tenant_id is not None and run.tenant_id != tenant_id:
            return None
        return run

    async def list_runs(self, tenant_id: Optional[UUID], limit: int = 50) -> List[AgentRun]:
        runs = list(self.runs.values())
        if tenant_id is not None:
            runs = [r for r in runs if r.tenant_id == tenant_id]
        return sorted(runs, key=lambda r: r.created_at, reverse=True)[:limit]

    async def create_step(self, step: AgentRunStep) -> AgentRunStep:
        return step

    async def update_step(self, step: AgentRunStep) -> AgentRunStep:
        return step

    async def list_steps(self, run_id: UUID) -> List[AgentRunStep]:
        return []

    async def create_artifact(self, artifact: AgentRunArtifact) -> AgentRunArtifact:
        return artifact

    async def list_artifacts(self, run_id: UUID) -> List[AgentRunArtifact]:
        return []

    async def create_evaluation(self, evaluation: AgentRunEvaluation) -> AgentRunEvaluation:
        return evaluation

    async def get_latest_evaluation(self, run_id: UUID) -> Optional[AgentRunEvaluation]:
        return None

    async def link_memory(self, link: AgentRunMemoryLink) -> AgentRunMemoryLink:
        return link


@pytest.mark.asyncio
async def test_v4_agent_runs_requires_tenant():
    app = FastAPI()
    app.include_router(agent_runs_router)

    async def override_repo():
        return FakeAgentRunRepo()

    app.dependency_overrides[_get_repo] = override_repo

    async with AsyncClient(app=app, base_url="http://test") as client:
        resp = await client.post("/api/v4/agent-runs", json={"requirement": "do x"})
        assert resp.status_code == 400
        assert "Tenant context required" in resp.text


@pytest.mark.asyncio
async def test_v4_agent_runs_create_list_get_cancel():
    tenant_id = uuid4()
    fake_repo = FakeAgentRunRepo()

    app = FastAPI()
    app.include_router(agent_runs_router)

    @app.middleware("http")
    async def inject_tenant(request: Request, call_next):
        request.state.tenant_id = tenant_id
        return await call_next(request)

    async def override_repo():
        return fake_repo

    app.dependency_overrides[_get_repo] = override_repo

    async with AsyncClient(app=app, base_url="http://test") as client:
        create = await client.post("/api/v4/agent-runs", json={"requirement": "generate python code"})
        assert create.status_code == 201
        created = create.json()
        run_id = UUID(created["id"])
        assert created["tenant_id"] == str(tenant_id)
        assert created["status"] == "created"

        lst = await client.get("/api/v4/agent-runs")
        assert lst.status_code == 200
        assert any(r["id"] == str(run_id) for r in lst.json())

        get = await client.get(f"/api/v4/agent-runs/{run_id}")
        assert get.status_code == 200
        assert get.json()["id"] == str(run_id)

        cancel = await client.post(f"/api/v4/agent-runs/{run_id}/cancel", params={"reason": "user_cancel"})
        assert cancel.status_code == 200
        assert cancel.json()["status"] == "cancelled"


def test_stdlib_platform_not_shadowed():
    import platform as std_platform

    assert callable(std_platform.system)

