"""
Agent Planner (single-agent runtime).

Produces a machine-readable task plan from a high-level requirement.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Tuple

from infrastructure.llm.provider import LLMRequest, LLMRouter


logger = logging.getLogger(__name__)


def deterministic_plan(requirement: str) -> Dict[str, Any]:
    return {
        "version": "v1",
        "strategy": "deterministic",
        "requirement": requirement,
        "steps": [
            {
                "id": "plan",
                "type": "planning",
                "title": "Derive implementation plan",
                "description": "Convert requirement into an ordered set of actions and acceptance checks.",
            },
            {
                "id": "codegen",
                "type": "codegen",
                "title": "Generate code artifacts",
                "description": "Generate code outputs as artifacts with clear file boundaries.",
            },
            {
                "id": "validate",
                "type": "validation",
                "title": "Validate generated code",
                "description": "Syntax/structure validation and repair loop when needed.",
            },
            {
                "id": "evaluate",
                "type": "evaluation",
                "title": "Self-evaluate",
                "description": "Assess quality and produce retry plan if needed.",
            },
        ],
        "acceptance_checks": [
            "All generated Python files compile (syntax check).",
            "Validation report contains no blocking errors.",
        ],
    }


async def llm_plan(
    requirement: str,
    *,
    llm: LLMRouter,
    model: str,
    max_tokens: int = 900,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    system = (
        "You are an expert software planner. Output ONLY valid JSON. "
        "Schema: {\"version\":\"v1\",\"strategy\":\"llm\",\"steps\":[{\"id\":str,\"type\":str,"
        "\"title\":str,\"description\":str}],\"acceptance_checks\":[str]}."
    )
    req = LLMRequest(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": requirement},
        ],
        model=model,
        temperature=0.2,
        max_tokens=max_tokens,
        metadata={"purpose": "agent_planning"},
    )
    resp = await llm.generate(req)
    usage = {"tokens": resp.usage, "cost_usd": resp.cost_usd, "model": resp.model, "provider": resp.provider.value}
    try:
        data = json.loads(resp.content)
        if not isinstance(data, dict) or "steps" not in data:
            raise ValueError("Planner JSON missing required fields")
        return data, usage
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM planner output invalid JSON; falling back to deterministic", exc_info=True)
        fallback = deterministic_plan(requirement)
        fallback["llm_raw"] = resp.content
        fallback["llm_error"] = str(exc)
        return fallback, usage

