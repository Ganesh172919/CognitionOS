"""
API Gateway Middleware Stack — CognitionOS

Production middleware for FastAPI:
- Request/response logging
- CORS configuration
- Request ID injection
- Compression
- Security headers
- Request timing
- Tenant context injection
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class MiddlewareStack:
    """Composable middleware stack for the API gateway."""

    def __init__(self) -> None:
        self._pre_handlers: List[Callable] = []
        self._post_handlers: List[Callable] = []
        self._metrics: Dict[str, Any] = {
            "total_requests": 0, "total_errors": 0,
            "avg_response_ms": 0, "total_response_ms": 0}

    def add_pre(self, handler: Callable) -> None:
        self._pre_handlers.append(handler)

    def add_post(self, handler: Callable) -> None:
        self._post_handlers.append(handler)

    def process_request(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        """Run all pre-handlers on the request context."""
        request_context.setdefault("request_id", str(uuid.uuid4()))
        request_context.setdefault("start_time", time.monotonic())
        request_context.setdefault("headers", {})

        for handler in self._pre_handlers:
            try:
                result = handler(request_context)
                if result is not None:
                    request_context.update(result)
            except Exception as e:
                logger.error("Pre-handler error: %s", e)

        self._metrics["total_requests"] += 1
        return request_context

    def process_response(self, request_context: Dict[str, Any],
                          response: Dict[str, Any]) -> Dict[str, Any]:
        """Run all post-handlers on the response."""
        start = request_context.get("start_time", time.monotonic())
        duration_ms = (time.monotonic() - start) * 1000
        response["headers"] = response.get("headers", {})
        response["headers"]["X-Request-ID"] = request_context.get("request_id", "")
        response["headers"]["X-Response-Time"] = f"{duration_ms:.2f}ms"

        for handler in self._post_handlers:
            try:
                result = handler(request_context, response)
                if result is not None:
                    response.update(result)
            except Exception as e:
                logger.error("Post-handler error: %s", e)

        self._metrics["total_response_ms"] += duration_ms
        self._metrics["avg_response_ms"] = (
            self._metrics["total_response_ms"] / self._metrics["total_requests"])
        return response

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self._metrics)


# ---- Built-in middleware functions ----

def security_headers_middleware(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Inject security headers."""
    return {"security_headers": {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        "Content-Security-Policy": "default-src 'self'"}}


def tenant_context_middleware(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Extract tenant from request headers."""
    headers = ctx.get("headers", {})
    tenant_id = headers.get("X-Tenant-ID", headers.get("x-tenant-id", ""))
    return {"tenant_id": tenant_id}


def request_logging_middleware(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Log incoming request details."""
    logger.info("Request [%s] %s %s tenant=%s",
                ctx.get("request_id", "?"),
                ctx.get("method", "?"),
                ctx.get("path", "?"),
                ctx.get("tenant_id", "?"))
    return {}


def response_logging_middleware(ctx: Dict[str, Any], resp: Dict[str, Any]) -> Dict[str, Any]:
    """Log response details."""
    duration = resp.get("headers", {}).get("X-Response-Time", "?")
    logger.info("Response [%s] status=%s time=%s",
                ctx.get("request_id", "?"),
                resp.get("status_code", "?"),
                duration)
    return {}


def cors_middleware(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Inject CORS configuration."""
    return {"cors": {
        "allowed_origins": ["*"],
        "allowed_methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        "allowed_headers": ["Content-Type", "Authorization", "X-Tenant-ID",
                             "X-Request-ID", "X-API-Key"],
        "max_age": 86400}}


def create_default_stack() -> MiddlewareStack:
    """Create a production-ready middleware stack."""
    stack = MiddlewareStack()
    stack.add_pre(security_headers_middleware)
    stack.add_pre(tenant_context_middleware)
    stack.add_pre(cors_middleware)
    stack.add_pre(request_logging_middleware)
    stack.add_post(response_logging_middleware)
    return stack


_stack: MiddlewareStack | None = None

def get_middleware_stack() -> MiddlewareStack:
    global _stack
    if not _stack:
        _stack = create_default_stack()
    return _stack
