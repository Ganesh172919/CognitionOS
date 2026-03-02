"""
Developer SDK — CognitionOS Platform SDK

Public-facing SDK for third-party integrations:
- API client with authentication and retry logic
- Webhook registration and management
- Plugin development helpers
- Event subscription client
- Rate-limit aware request handling
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class AuthMethod(str, Enum):
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    OAUTH2 = "oauth2"


class HTTPMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


@dataclass
class SDKConfig:
    base_url: str = "https://api.cognitionos.dev/v1"
    api_key: str = ""
    bearer_token: str = ""
    auth_method: AuthMethod = AuthMethod.API_KEY
    timeout: float = 30.0
    max_retries: int = 3
    retry_backoff: float = 1.0
    rate_limit_retry: bool = True
    webhook_secret: str = ""
    user_agent: str = "CognitionOS-SDK/1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "auth_method": self.auth_method.value,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }


@dataclass
class APIResponse:
    status_code: int
    data: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    request_id: str = ""
    rate_limit_remaining: int = -1
    rate_limit_reset: float = 0

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


@dataclass
class WebhookRegistration:
    webhook_id: str
    url: str
    events: List[str]
    secret: str
    active: bool = True
    created_at: float = field(default_factory=time.time)


class CognitionSDK:
    """
    Official CognitionOS Developer SDK.

    Provides authenticated access to all platform APIs with
    automatic retry logic, rate limit handling, and webhook support.
    """

    def __init__(self, config: Optional[SDKConfig] = None, **kwargs):
        self._config = config or SDKConfig(**kwargs)
        self._request_count = 0
        self._error_count = 0
        self._webhooks: Dict[str, WebhookRegistration] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        logger.info("CognitionOS SDK initialized (base=%s)", self._config.base_url)

    # ── Authentication ──

    def _build_auth_headers(self) -> Dict[str, str]:
        headers = {"User-Agent": self._config.user_agent}
        if self._config.auth_method == AuthMethod.API_KEY:
            headers["X-API-Key"] = self._config.api_key
        elif self._config.auth_method == AuthMethod.BEARER_TOKEN:
            headers["Authorization"] = f"Bearer {self._config.bearer_token}"
        return headers

    # ── HTTP Client ──

    async def request(self, method: HTTPMethod, path: str, *,
                      data: Optional[Dict] = None, params: Optional[Dict] = None,
                      headers: Optional[Dict] = None) -> APIResponse:
        """Make an authenticated API request with retry logic."""
        url = urljoin(self._config.base_url, path)
        req_headers = self._build_auth_headers()
        if headers:
            req_headers.update(headers)
        req_headers["Content-Type"] = "application/json"

        last_error = None
        for attempt in range(self._config.max_retries + 1):
            try:
                self._request_count += 1
                # Simulated HTTP request (in production, use httpx/aiohttp)
                response = APIResponse(
                    status_code=200,
                    data={"status": "ok", "path": path, "method": method.value},
                    request_id=hashlib.md5(f"{url}{time.time()}".encode()).hexdigest()[:12],
                )

                # Handle rate limiting
                if response.status_code == 429 and self._config.rate_limit_retry:
                    wait = min(self._config.retry_backoff * (2 ** attempt), 60)
                    logger.warning("Rate limited, waiting %.1fs (attempt %d)", wait, attempt + 1)
                    await asyncio.sleep(wait)
                    continue

                return response

            except Exception as e:
                last_error = str(e)
                self._error_count += 1
                if attempt < self._config.max_retries:
                    wait = self._config.retry_backoff * (2 ** attempt)
                    await asyncio.sleep(wait)

        return APIResponse(status_code=0, error=last_error or "Request failed after retries")

    # ── Convenience Methods ──

    async def get(self, path: str, **kwargs) -> APIResponse:
        return await self.request(HTTPMethod.GET, path, **kwargs)

    async def post(self, path: str, **kwargs) -> APIResponse:
        return await self.request(HTTPMethod.POST, path, **kwargs)

    async def put(self, path: str, **kwargs) -> APIResponse:
        return await self.request(HTTPMethod.PUT, path, **kwargs)

    async def delete(self, path: str, **kwargs) -> APIResponse:
        return await self.request(HTTPMethod.DELETE, path, **kwargs)

    # ── Agent API ──

    async def run_agent(self, prompt: str, *, model: str = "auto",
                        max_tokens: int = 4096) -> APIResponse:
        """Execute an AI agent task."""
        return await self.post("/agent/run", data={
            "prompt": prompt, "model": model, "max_tokens": max_tokens,
        })

    async def get_agent_status(self, task_id: str) -> APIResponse:
        return await self.get(f"/agent/tasks/{task_id}")

    async def list_agent_tasks(self, *, limit: int = 20,
                               offset: int = 0) -> APIResponse:
        return await self.get("/agent/tasks", params={"limit": limit, "offset": offset})

    # ── Workflow API ──

    async def create_workflow(self, definition: Dict[str, Any]) -> APIResponse:
        return await self.post("/workflows", data=definition)

    async def execute_workflow(self, workflow_id: str,
                               inputs: Optional[Dict] = None) -> APIResponse:
        return await self.post(f"/workflows/{workflow_id}/execute", data=inputs or {})

    async def get_workflow(self, workflow_id: str) -> APIResponse:
        return await self.get(f"/workflows/{workflow_id}")

    # ── Plugin API ──

    async def register_plugin(self, manifest: Dict[str, Any]) -> APIResponse:
        return await self.post("/plugins", data=manifest)

    async def list_plugins(self, *, category: str = "") -> APIResponse:
        params = {"category": category} if category else None
        return await self.get("/plugins", params=params)

    # ── Webhook Management ──

    def register_webhook(self, url: str, events: List[str],
                         secret: str = "") -> WebhookRegistration:
        wh_id = hashlib.md5(f"{url}{time.time()}".encode()).hexdigest()[:12]
        wh_secret = secret or hashlib.sha256(f"wh_{wh_id}".encode()).hexdigest()[:32]
        reg = WebhookRegistration(
            webhook_id=wh_id, url=url, events=events, secret=wh_secret,
        )
        self._webhooks[wh_id] = reg
        logger.info("Webhook registered: %s for events %s", wh_id, events)
        return reg

    def verify_webhook_signature(self, payload: bytes, signature: str,
                                 secret: str) -> bool:
        expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)

    def list_webhooks(self) -> List[Dict[str, Any]]:
        return [
            {"id": wh.webhook_id, "url": wh.url, "events": wh.events, "active": wh.active}
            for wh in self._webhooks.values()
        ]

    # ── Event Subscriptions ──

    def on_event(self, event_type: str, handler: Callable):
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    async def _dispatch_event(self, event_type: str, data: Dict[str, Any]):
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                result = handler(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("Event handler error for '%s': %s", event_type, e)

    # ── Analytics API ──

    async def get_usage(self, *, period: str = "7d") -> APIResponse:
        return await self.get("/analytics/usage", params={"period": period})

    async def get_billing(self) -> APIResponse:
        return await self.get("/billing/current")

    # ── Stats ──

    def get_stats(self) -> Dict[str, Any]:
        return {
            "config": self._config.to_dict(),
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "error_rate": round(self._error_count / max(self._request_count, 1), 4),
            "registered_webhooks": len(self._webhooks),
            "event_subscriptions": sum(len(h) for h in self._event_handlers.values()),
        }
