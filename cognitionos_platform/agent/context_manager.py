"""
Agent Context Manager - Session context and token budget.

Tracks context and token usage for a single agent run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TokenBudget:
    """Token budget for an agent run."""

    max_tokens: int
    used_tokens: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    def consume(self, tokens: int) -> bool:
        """Consume tokens. Returns False if budget exceeded."""
        if self.used_tokens + tokens > self.max_tokens:
            return False
        self.used_tokens += tokens
        return True


@dataclass
class SessionContext:
    """Session context for an agent run."""

    run_id: str
    requirement: str
    token_budget: TokenBudget
    plan: Optional[Dict[str, Any]] = None
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_artifact(self, kind: str, name: str, content: Any) -> None:
        self.artifacts.append({"kind": kind, "name": name, "content": content})


class AgentContextManager:
    """
    Manages session context and token budget for agent runs.
    """

    def __init__(
        self,
        run_id: str,
        requirement: str,
        max_tokens: int = 16000,
    ):
        self.run_id = run_id
        self.requirement = requirement
        self._context = SessionContext(
            run_id=run_id,
            requirement=requirement,
            token_budget=TokenBudget(max_tokens=max_tokens),
        )

    @property
    def context(self) -> SessionContext:
        return self._context

    def consume_tokens(self, tokens: int) -> bool:
        """Consume tokens. Returns False if budget exceeded."""
        return self._context.token_budget.consume(tokens)

    def set_plan(self, plan: Dict[str, Any]) -> None:
        self._context.plan = plan
