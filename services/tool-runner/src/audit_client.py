"""
Audit Client for Tool Runner.

Sends audit logs to the audit-log service for all tool executions.
"""

import httpx
import os
from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime


class AuditClient:
    """
    Client to send audit logs to audit-log service.
    """

    def __init__(self, audit_service_url: Optional[str] = None):
        """
        Initialize audit client.

        Args:
            audit_service_url: URL of audit-log service
        """
        self.audit_service_url = audit_service_url or os.getenv(
            "AUDIT_SERVICE_URL",
            "http://audit-log:8007"
        )
        self.timeout = httpx.Timeout(5.0, connect=2.0)

    async def log_tool_execution(
        self,
        tool_name: str,
        user_id: UUID,
        agent_id: UUID,
        parameters: Dict[str, Any],
        permissions: list[str],
        outcome: str,
        success: bool,
        duration_ms: int,
        error: Optional[str] = None,
        output: Optional[Any] = None
    ):
        """
        Log a tool execution to audit service.

        Args:
            tool_name: Name of tool executed
            user_id: User ID
            agent_id: Agent ID
            parameters: Tool parameters
            permissions: Permissions granted
            outcome: Outcome (success, failure, denied)
            success: Whether execution succeeded
            duration_ms: Execution duration in milliseconds
            error: Error message if failed
            output: Tool output if succeeded
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                await client.post(
                    f"{self.audit_service_url}/audit",
                    json={
                        "event_type": f"tool_execution_{tool_name}",
                        "event_category": "tool_execution",
                        "severity": "error" if not success else "info",
                        "actor_type": "agent",
                        "actor_id": str(agent_id),
                        "user_id": str(user_id),
                        "agent_id": str(agent_id),
                        "action": f"execute_{tool_name}",
                        "resource_type": "tool",
                        "resource_id": tool_name,
                        "outcome": outcome,
                        "details": {
                            "tool": tool_name,
                            "permissions_used": permissions,
                            "success": success
                        },
                        "request_data": {
                            "parameters": parameters
                        },
                        "response_data": {
                            "output": str(output)[:500] if output else None  # Truncate large outputs
                        } if success else None,
                        "error_message": error,
                        "permission_required": ",".join(permissions),
                        "permission_granted": True,  # If we got here, permission was granted
                        "duration_ms": duration_ms
                    }
                )
            except httpx.ConnectError:
                # Audit service unavailable - log locally but don't fail the operation
                pass
            except Exception:
                # Don't fail tool execution due to audit logging issues
                pass

    async def log_permission_denied(
        self,
        tool_name: str,
        user_id: UUID,
        agent_id: UUID,
        required_permissions: list[str],
        granted_permissions: list[str]
    ):
        """
        Log a permission denied event.

        Args:
            tool_name: Tool that was denied
            user_id: User ID
            agent_id: Agent ID
            required_permissions: Permissions that were required
            granted_permissions: Permissions that were granted
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                await client.post(
                    f"{self.audit_service_url}/audit",
                    json={
                        "event_type": "tool_permission_denied",
                        "event_category": "security",
                        "severity": "warning",
                        "actor_type": "agent",
                        "actor_id": str(agent_id),
                        "user_id": str(user_id),
                        "agent_id": str(agent_id),
                        "action": f"attempt_execute_{tool_name}",
                        "resource_type": "tool",
                        "resource_id": tool_name,
                        "outcome": "denied",
                        "details": {
                            "tool": tool_name,
                            "required_permissions": required_permissions,
                            "granted_permissions": granted_permissions,
                            "missing_permissions": list(set(required_permissions) - set(granted_permissions))
                        },
                        "permission_required": ",".join(required_permissions),
                        "permission_granted": False,
                        "duration_ms": 0
                    }
                )
            except Exception:
                pass
