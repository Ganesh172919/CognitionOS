"""
Tool Runner client.

Canonical boundary for sandboxed tool execution.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx


logger = logging.getLogger(__name__)


class ToolRunnerClient:
    def __init__(self, *, base_url: str, timeout_seconds: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    async def execute_tool(
        self,
        *,
        tool_name: str,
        parameters: Dict[str, Any],
        permissions: List[str],
        user_id: str,
        agent_id: str,
        timeout_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload = {
            "tool_name": tool_name,
            "parameters": parameters,
            "user_id": user_id,
            "agent_id": agent_id,
            "permissions": permissions,
            "timeout_seconds": timeout_seconds,
        }
        url = f"{self.base_url}/execute"
        timeout = httpx.Timeout(timeout_seconds or self._timeout_seconds)

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code >= 400:
                logger.warning("Tool runner error", extra={"status": resp.status_code, "body": resp.text})
                resp.raise_for_status()
            return resp.json()

