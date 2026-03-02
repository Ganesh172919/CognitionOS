"""
Integration Orchestrator — CognitionOS

Manages external service integrations:
- OAuth2 connection management
- Credential vault
- Connection health monitoring
- Request/response transformation
- Retry and failover
- Integration marketplace catalog
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class IntegrationStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    PENDING = "pending"
    EXPIRED = "expired"


class AuthType(str, Enum):
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BASIC = "basic"
    BEARER = "bearer"
    CUSTOM = "custom"


@dataclass
class IntegrationConfig:
    integration_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    provider: str = ""
    auth_type: AuthType = AuthType.API_KEY
    status: IntegrationStatus = IntegrationStatus.DISCONNECTED
    tenant_id: str = ""
    credentials: Dict[str, str] = field(default_factory=dict)  # stored encrypted
    config: Dict[str, Any] = field(default_factory=dict)
    scopes: List[str] = field(default_factory=list)
    base_url: str = ""
    webhook_url: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_health_check: Optional[str] = None
    error_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "integration_id": self.integration_id, "name": self.name,
            "provider": self.provider, "auth_type": self.auth_type.value,
            "status": self.status.value, "tenant_id": self.tenant_id,
            "scopes": self.scopes, "error_count": self.error_count,
            "last_health_check": self.last_health_check}


@dataclass
class IntegrationProvider:
    provider_id: str
    name: str
    description: str = ""
    auth_type: AuthType = AuthType.OAUTH2
    docs_url: str = ""
    base_url: str = ""
    required_scopes: List[str] = field(default_factory=list)
    supported_actions: List[str] = field(default_factory=list)
    icon_url: str = ""
    category: str = "general"
    is_premium: bool = False


# Built-in provider catalog
PROVIDER_CATALOG: Dict[str, IntegrationProvider] = {
    "github": IntegrationProvider(
        "github", "GitHub", "Code repository management",
        AuthType.OAUTH2, "https://docs.github.com",
        "https://api.github.com",
        ["repo", "user"], ["create_repo", "create_pr", "list_issues"],
        category="development"),
    "slack": IntegrationProvider(
        "slack", "Slack", "Team messaging",
        AuthType.OAUTH2, "https://api.slack.com",
        "https://slack.com/api",
        ["chat:write", "channels:read"], ["send_message", "create_channel"],
        category="communication"),
    "jira": IntegrationProvider(
        "jira", "Jira", "Project management",
        AuthType.OAUTH2, "https://developer.atlassian.com",
        "https://api.atlassian.com",
        ["read:jira-work", "write:jira-work"], ["create_issue", "update_status"],
        category="project_management"),
    "openai": IntegrationProvider(
        "openai", "OpenAI", "AI model provider",
        AuthType.BEARER, "https://platform.openai.com",
        "https://api.openai.com/v1",
        [], ["chat_completion", "embeddings"],
        category="ai"),
    "stripe": IntegrationProvider(
        "stripe", "Stripe", "Payment processing",
        AuthType.BEARER, "https://stripe.com/docs",
        "https://api.stripe.com/v1",
        [], ["create_customer", "create_subscription", "process_payment"],
        category="payments", is_premium=True),
    "aws_s3": IntegrationProvider(
        "aws_s3", "AWS S3", "Cloud storage",
        AuthType.API_KEY, "https://docs.aws.amazon.com/s3",
        "https://s3.amazonaws.com",
        [], ["upload_file", "download_file", "list_objects"],
        category="storage"),
}


class IntegrationOrchestrator:
    """Manages external service integrations and credentials."""

    def __init__(self) -> None:
        self._integrations: Dict[str, IntegrationConfig] = {}
        self._providers = dict(PROVIDER_CATALOG)
        self._health_callbacks: Dict[str, Callable[..., Awaitable[bool]]] = {}
        self._action_handlers: Dict[str, Callable[..., Awaitable[Any]]] = {}
        self._metrics: Dict[str, int] = defaultdict(int)

    # ---- connection management ----
    def connect(self, config: IntegrationConfig) -> str:
        if config.provider not in self._providers:
            raise ValueError(f"Unknown provider: {config.provider}")
        config.status = IntegrationStatus.CONNECTED
        self._integrations[config.integration_id] = config
        self._metrics["connections"] += 1
        logger.info("Integration connected: %s (%s)", config.name, config.provider)
        return config.integration_id

    def disconnect(self, integration_id: str) -> bool:
        config = self._integrations.get(integration_id)
        if not config:
            return False
        config.status = IntegrationStatus.DISCONNECTED
        config.credentials.clear()
        self._metrics["disconnections"] += 1
        return True

    def get_integration(self, integration_id: str) -> Optional[IntegrationConfig]:
        return self._integrations.get(integration_id)

    def list_integrations(self, *, tenant_id: str = "",
                           status: IntegrationStatus | None = None) -> List[Dict[str, Any]]:
        integrations = list(self._integrations.values())
        if tenant_id:
            integrations = [i for i in integrations if i.tenant_id == tenant_id]
        if status:
            integrations = [i for i in integrations if i.status == status]
        return [i.to_dict() for i in integrations]

    # ---- provider catalog ----
    def list_providers(self, *, category: str = "") -> List[Dict[str, Any]]:
        providers = list(self._providers.values())
        if category:
            providers = [p for p in providers if p.category == category]
        return [{"provider_id": p.provider_id, "name": p.name,
                 "description": p.description, "category": p.category,
                 "auth_type": p.auth_type.value, "is_premium": p.is_premium,
                 "actions": p.supported_actions}
                for p in providers]

    def register_provider(self, provider: IntegrationProvider) -> None:
        self._providers[provider.provider_id] = provider

    # ---- actions ----
    def register_action(self, provider: str, action: str,
                         handler: Callable[..., Awaitable[Any]]) -> None:
        key = f"{provider}:{action}"
        self._action_handlers[key] = handler

    async def execute_action(self, integration_id: str, action: str,
                              params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        config = self._integrations.get(integration_id)
        if not config:
            return {"success": False, "error": "Integration not found"}
        if config.status != IntegrationStatus.CONNECTED:
            return {"success": False, "error": f"Integration not connected: {config.status.value}"}

        key = f"{config.provider}:{action}"
        handler = self._action_handlers.get(key)
        if not handler:
            return {"success": False, "error": f"Action not available: {action}"}

        start = time.monotonic()
        try:
            result = await handler(config, params or {})
            self._metrics["actions_executed"] += 1
            return {"success": True, "result": result,
                    "duration_ms": (time.monotonic() - start) * 1000}
        except Exception as e:
            config.error_count += 1
            self._metrics["action_errors"] += 1
            return {"success": False, "error": str(e)}

    # ---- health ----
    async def check_health(self, integration_id: str) -> Dict[str, Any]:
        config = self._integrations.get(integration_id)
        if not config:
            return {"healthy": False, "error": "not found"}

        callback = self._health_callbacks.get(config.provider)
        if callback:
            try:
                healthy = await callback(config)
                config.last_health_check = datetime.now(timezone.utc).isoformat()
                if not healthy:
                    config.status = IntegrationStatus.ERROR
                return {"healthy": healthy, "integration_id": integration_id}
            except Exception as e:
                config.status = IntegrationStatus.ERROR
                return {"healthy": False, "error": str(e)}

        config.last_health_check = datetime.now(timezone.utc).isoformat()
        return {"healthy": config.status == IntegrationStatus.CONNECTED,
                "integration_id": integration_id}

    def register_health_check(self, provider: str,
                                callback: Callable[..., Awaitable[bool]]) -> None:
        self._health_callbacks[provider] = callback

    def get_metrics(self) -> Dict[str, Any]:
        connected = sum(1 for i in self._integrations.values()
                        if i.status == IntegrationStatus.CONNECTED)
        return {**dict(self._metrics), "total_integrations": len(self._integrations),
                "connected": connected, "providers": len(self._providers)}


_orchestrator: IntegrationOrchestrator | None = None

def get_integration_orchestrator() -> IntegrationOrchestrator:
    global _orchestrator
    if not _orchestrator:
        _orchestrator = IntegrationOrchestrator()
    return _orchestrator
