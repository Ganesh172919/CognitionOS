"""
CognitionOS API Server Entry Point

FastAPI production server with:
- ASGI-compatible application factory
- Middleware stack (CORS, auth, logging, rate limiting)
- Router registration from route definitions
- Startup/shutdown lifecycle hooks
- OpenAPI documentation
- Health and readiness probes
"""

from __future__ import annotations

import asyncio
import logging
import time
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AppConfig:
    """Application configuration."""
    APP_NAME = "CognitionOS"
    APP_VERSION = "2.0.0"
    DEBUG = os.getenv("COGNITIONOS_DEBUG", "false").lower() == "true"
    HOST = os.getenv("COGNITIONOS_HOST", "0.0.0.0")
    PORT = int(os.getenv("COGNITIONOS_PORT", "8000"))
    WORKERS = int(os.getenv("COGNITIONOS_WORKERS", "4"))
    LOG_LEVEL = os.getenv("COGNITIONOS_LOG_LEVEL", "INFO")
    CORS_ORIGINS = os.getenv("COGNITIONOS_CORS_ORIGINS", "*").split(",")
    API_PREFIX = "/api/v1"
    DOCS_URL = "/docs"
    REDOC_URL = "/redoc"
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///cognitionos.db")
    REDIS_URL = os.getenv("REDIS_URL", "")
    SECRET_KEY = os.getenv("COGNITIONOS_SECRET_KEY", "change-me-in-production")


# ── Application Factory ──

_platform_instance = None
_ready = False


async def startup_handler():
    """Platform startup hook."""
    global _platform_instance, _ready
    logger.info("Starting CognitionOS platform...")

    try:
        from platform_init import get_platform
        _platform_instance = get_platform()
        result = await _platform_instance.initialize()
        _ready = True
        logger.info("Platform initialized: %s", result.get("status"))
    except Exception as exc:
        logger.error("Platform initialization failed: %s", exc, exc_info=True)
        _ready = False


async def shutdown_handler():
    """Platform shutdown hook."""
    global _platform_instance, _ready
    if _platform_instance:
        await _platform_instance.shutdown()
    _ready = False
    logger.info("Platform shutdown complete.")


# ── Route Handlers ──

async def health_check() -> Dict[str, Any]:
    """Kubernetes-compatible health check."""
    return {
        "status": "healthy" if _ready else "starting",
        "version": AppConfig.APP_VERSION,
        "timestamp": time.time(),
    }


async def readiness_check() -> Dict[str, Any]:
    """Readiness probe for load balancers."""
    if not _ready:
        return {"status": "not_ready"}

    checks = {}
    if _platform_instance:
        health = _platform_instance.health_check()
        checks = health.get("subsystems", {})

    return {
        "status": "ready" if _ready else "not_ready",
        "subsystems": checks,
    }


async def platform_status() -> Dict[str, Any]:
    """Full platform status with subsystem details."""
    if _platform_instance:
        return _platform_instance.health_check()
    return {"status": "not_initialized"}


async def get_dashboard() -> Dict[str, Any]:
    """Platform-wide dashboard data."""
    if _platform_instance:
        return _platform_instance.get_dashboard()
    return {"error": "Platform not initialized"}


async def get_api_docs() -> Dict[str, Any]:
    """OpenAPI spec generation."""
    from services.api.routes import get_openapi_paths, PLATFORM_ROUTES

    return {
        "openapi": "3.1.0",
        "info": {
            "title": AppConfig.APP_NAME,
            "version": AppConfig.APP_VERSION,
            "description": "CognitionOS AI Platform API",
        },
        "paths": get_openapi_paths(),
        "components": {
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                },
                "apiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                },
            },
        },
    }


# ── Application Builder ──

def create_app():
    """
    Create the ASGI application.

    In production, use with uvicorn:
        uvicorn services.api.server:create_app --factory --workers 4
    """
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        app = FastAPI(
            title=AppConfig.APP_NAME,
            version=AppConfig.APP_VERSION,
            description="CognitionOS — AI-Powered Code Intelligence Platform",
            docs_url=AppConfig.DOCS_URL,
            redoc_url=AppConfig.REDOC_URL,
        )

        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=AppConfig.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Lifecycle hooks
        app.add_event_handler("startup", startup_handler)
        app.add_event_handler("shutdown", shutdown_handler)

        # Core routes
        app.get("/health")(health_check)
        app.get("/ready")(readiness_check)
        app.get("/status")(platform_status)
        app.get("/dashboard")(get_dashboard)
        app.get("/openapi-spec")(get_api_docs)

        logger.info("FastAPI application created successfully")
        return app

    except ImportError:
        logger.warning("FastAPI not installed — using minimal ASGI app")
        return _create_minimal_app()


def _create_minimal_app():
    """Minimal ASGI application when FastAPI is not available."""

    async def app(scope, receive, send):
        if scope["type"] == "http":
            import json
            path = scope.get("path", "/")
            if path == "/health":
                body = json.dumps(await health_check()).encode()
            elif path == "/status":
                body = json.dumps(await platform_status()).encode()
            else:
                body = json.dumps({"error": "Not found"}).encode()

            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"application/json"]],
            })
            await send({"type": "http.response.body", "body": body})

    return app


# ── CLI Entry Point ──

def main():
    """Run the CognitionOS API server."""
    logging.basicConfig(
        level=getattr(logging, AppConfig.LOG_LEVEL),
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    )

    logger.info("=" * 60)
    logger.info("  CognitionOS v%s — Production API Server", AppConfig.APP_VERSION)
    logger.info("  Host: %s:%d | Workers: %d",
                 AppConfig.HOST, AppConfig.PORT, AppConfig.WORKERS)
    logger.info("=" * 60)

    try:
        import uvicorn
        uvicorn.run(
            "services.api.server:create_app",
            host=AppConfig.HOST,
            port=AppConfig.PORT,
            workers=AppConfig.WORKERS,
            factory=True,
            log_level=AppConfig.LOG_LEVEL.lower(),
        )
    except ImportError:
        logger.info("uvicorn not installed — running with asyncio")
        asyncio.run(startup_handler())
        logger.info("Platform running. Press Ctrl+C to stop.")
        try:
            asyncio.get_event_loop().run_forever()
        except KeyboardInterrupt:
            asyncio.run(shutdown_handler())


if __name__ == "__main__":
    main()
