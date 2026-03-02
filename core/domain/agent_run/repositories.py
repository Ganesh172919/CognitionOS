"""Agent Run domain repositories (interfaces)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from core.domain.agent_run.entities import (
    AgentRun,
    AgentRunArtifact,
    AgentRunEvaluation,
    AgentRunMemoryLink,
    AgentRunStep,
)


class AgentRunRepository(ABC):
    @abstractmethod
    async def create_run(self, run: AgentRun) -> AgentRun: ...

    @abstractmethod
    async def update_run(self, run: AgentRun) -> AgentRun: ...

    @abstractmethod
    async def get_run(self, run_id: UUID, tenant_id: Optional[UUID]) -> Optional[AgentRun]: ...

    @abstractmethod
    async def list_runs(self, tenant_id: Optional[UUID], limit: int = 50) -> List[AgentRun]: ...

    @abstractmethod
    async def create_step(self, step: AgentRunStep) -> AgentRunStep: ...

    @abstractmethod
    async def update_step(self, step: AgentRunStep) -> AgentRunStep: ...

    @abstractmethod
    async def list_steps(self, run_id: UUID) -> List[AgentRunStep]: ...

    @abstractmethod
    async def create_artifact(self, artifact: AgentRunArtifact) -> AgentRunArtifact: ...

    @abstractmethod
    async def list_artifacts(self, run_id: UUID) -> List[AgentRunArtifact]: ...

    @abstractmethod
    async def create_evaluation(self, evaluation: AgentRunEvaluation) -> AgentRunEvaluation: ...

    @abstractmethod
    async def get_latest_evaluation(self, run_id: UUID) -> Optional[AgentRunEvaluation]: ...

    @abstractmethod
    async def link_memory(self, link: AgentRunMemoryLink) -> AgentRunMemoryLink: ...

    # Convenience for gating
    async def count_active_runs(self, tenant_id: UUID) -> int:  # pragma: no cover
        """Optional: count active (queued/running/validating) runs for tenant."""
        raise NotImplementedError
