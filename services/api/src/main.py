"""
CognitionOS V3 API Service - Main Application

FastAPI application providing REST APIs for the V3 clean architecture.
"""

import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)
from typing import Any, Dict
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn

from core.config import get_config
from services.api.src.routes import workflows
from services.api.src.schemas.workflows import HealthCheckResponse, ErrorResponse
from services.api.src.dependencies.injection import (
    check_database_health,
    check_redis_health,
    check_rabbitmq_health,
)
from services.api.src.error_handlers import register_error_handlers
from infrastructure.observability import (
    setup_tracing,
    instrument_fastapi,
    PrometheusMiddleware,
    create_metrics_app,
)
from infrastructure.middleware.api_key_auth import APIKeyAuthMiddleware
from infrastructure.middleware.tenant_context import TenantContextMiddleware, TenantIsolationMiddleware
from services.api.src.middleware.saas_context import TenantRepositoryProxy, APIKeyRepositoryProxy


# Configuration
config = get_config()


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting %s v%s", config.service_name, config.service_version)
    logger.info("Environment: %s", config.environment)
    logger.info("API URL: http://%s:%s", config.api.host, config.api.port)

    # Initialize tracing
    if config.observability.enable_tracing:
        setup_tracing(service_name=config.service_name)
        instrument_fastapi(app)
        logger.info(
            "OpenTelemetry tracing enabled (Jaeger: %s:%s)",
            config.observability.jaeger_host,
            config.observability.jaeger_port,
        )

    # Initialize metrics
    if config.observability.enable_metrics:
        logger.info("Prometheus metrics enabled on /metrics")

    # Initialize Redis connection pool
    try:
        from infrastructure.persistence.redis_pool import RedisPoolManager

        redis_pool = await RedisPoolManager.get_instance()
        logger.info("Redis connection pool initialized")
    except Exception as e:
        logger.warning("Failed to initialize Redis pool: %s", e)
    
    yield

    # Shutdown - graceful cleanup
    logger.info("Shutting down gracefully...")

    import asyncio

    shutdown_timeout = config.api.shutdown_timeout_seconds
    logger.info("Waiting for in-flight requests to complete (%ss)...", shutdown_timeout)
    await asyncio.sleep(shutdown_timeout)

    # Close database connections
    try:
        from services.api.src.dependencies.injection import close_db

        await close_db()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error("Error closing database: %s", e)

    # Close Redis connection pool
    try:
        from infrastructure.persistence.redis_pool import RedisPoolManager

        await RedisPoolManager.reset()
        logger.info("Redis connection pool closed")
    except Exception as e:
        logger.error("Error closing Redis pool: %s", e)

    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="CognitionOS V3 API",
    version=config.service_version,
    description="""
    CognitionOS V3 Clean Architecture API
    
    Provides REST endpoints for:
    - Workflow management and execution
    - Agent orchestration
    - Memory operations
    - Task planning
    - **Phase 3: Extended Agent Operation**
      - Checkpoint/resume for 24+ hour workflows
      - Agent health monitoring with heartbeat tracking
      - Cost governance and budget enforcement
    
    Built with Domain-Driven Design and Clean Architecture principles.
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ==================== Middleware ====================

# Request ID middleware (should be first)
from services.api.src.middleware.request_id import RequestIDMiddleware
app.add_middleware(RequestIDMiddleware)

# Multi-tenancy & API key context (optional; resolved when headers are present)
_tenant_repo_proxy = TenantRepositoryProxy()
_api_key_repo_proxy = APIKeyRepositoryProxy()

# Order matters: last-added middleware runs first in Starlette.
app.add_middleware(TenantIsolationMiddleware)
app.add_middleware(
    TenantContextMiddleware,
    tenant_repository=_tenant_repo_proxy,
    require_tenant=False,
)
app.add_middleware(
    APIKeyAuthMiddleware,
    api_key_repository=_api_key_repo_proxy,
    tenant_repository=_tenant_repo_proxy,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Prometheus metrics middleware
if config.observability.enable_metrics:
    app.add_middleware(PrometheusMiddleware)


# ==================== Metrics Endpoint ====================

# Mount metrics endpoint
metrics_app = create_metrics_app()
app.mount("/metrics", metrics_app)


# ==================== Error Handlers ====================

# Register centralized error handlers
register_error_handlers(app)


# ==================== Health Check ====================

@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["Health"],
    summary="Health check",
    description="Check the health status of the API and its dependencies",
)
async def health_check() -> HealthCheckResponse:
    """Health check endpoint"""
    db_status = "healthy" if await check_database_health() else "unhealthy"
    redis_status = "healthy" if await check_redis_health() else "unhealthy"
    rabbitmq_status = "healthy" if await check_rabbitmq_health() else "unhealthy"
    
    return HealthCheckResponse(
        status="healthy" if db_status == "healthy" else "degraded",
        version=config.service_version,
        timestamp=datetime.utcnow(),
        database=db_status,
        redis=redis_status,
        rabbitmq=rabbitmq_status,
    )


@app.get(
    "/health/live",
    tags=["Health"],
    summary="Liveness probe",
    description="Kubernetes liveness probe - checks if the application is alive",
)
async def liveness_probe() -> Dict[str, Any]:
    """
    Liveness probe for Kubernetes.
    Returns 200 if the application is running.
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get(
    "/health/ready",
    tags=["Health"],
    summary="Readiness probe",
    description="Kubernetes readiness probe - checks if the application can serve requests",
)
async def readiness_probe() -> Dict[str, Any]:
    """
    Readiness probe for Kubernetes.
    Returns 200 only if all critical dependencies are healthy.
    """
    db_healthy = await check_database_health()
    redis_healthy = await check_redis_health()
    
    # Service is ready if database is healthy (Redis is optional)
    is_ready = db_healthy
    
    if not is_ready:
        from fastapi import status
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "timestamp": datetime.utcnow().isoformat(),
                "dependencies": {
                    "database": "healthy" if db_healthy else "unhealthy",
                    "redis": "healthy" if redis_healthy else "unhealthy",
                }
            }
        )
    
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
        "dependencies": {
            "database": "healthy",
            "redis": "healthy" if redis_healthy else "degraded",
        }
    }


@app.get(
    "/",
    tags=["Root"],
    summary="API information",
    description="Get API information and available endpoints",
)
async def root():
    """Root endpoint"""
    return {
        "service": config.service_name,
        "version": config.service_version,
        "environment": config.environment,
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
        "health": "/health",
    }


# ==================== Include Routers ====================

app.include_router(workflows.router)

# Import and include auth routes
from services.api.src.routes import auth
app.include_router(auth.router)

# Import and include Phase 3 routes
from services.api.src.routes import checkpoints, health, cost, memory
app.include_router(checkpoints.router)
app.include_router(health.router)
app.include_router(cost.router)
app.include_router(memory.router)

# Import and include P0 execution persistence routes
from services.api.src.routes import execution_persistence
app.include_router(execution_persistence.router)

# Import and include WebSocket routes
from services.api.src.routes import websocket as ws_router
app.include_router(ws_router.router)

# Import and include multi-tenancy and marketplace routes
from services.api.src.routes import tenants, subscriptions, plugins
app.include_router(tenants.router)
app.include_router(subscriptions.router)
app.include_router(plugins.router)

# Stripe webhooks (billing lifecycle)
from services.api.src.routes import webhooks
app.include_router(webhooks.router)

# Import and include advanced analytics routes
from services.api.src.routes import analytics_advanced, engagement, marketplace
app.include_router(analytics_advanced.router)
app.include_router(engagement.router)
app.include_router(marketplace.router)

# Import and include advanced systems routes
from services.api.src.routes import advanced_systems
app.include_router(advanced_systems.router)

# Import and include developer tools routes
from services.api.src.routes import developer_tools
app.include_router(developer_tools.router)

# Import and include reliability & workflow routes
from services.api.src.routes import reliability_workflows
app.include_router(reliability_workflows.router)

# Import and include intelligence routes (LLM router, tools, memory, telemetry, billing)
from services.api.src.routes import intelligence
app.include_router(intelligence.router)

# Import and include platform routes (RBAC, config, onboarding, DSL compiler, coordination)
from services.api.src.routes import platform as platform_router
app.include_router(platform_router.router)

# Import and include SaaS platform routes (subscriptions, tenants, usage, rate limits)
from services.api.src.routes import saas_platform
app.include_router(saas_platform.router)

# Import and include Enterprise Transformation routes (gateway, AI orchestration, code gen, analytics, feature flags)
from services.api.src.routes import enterprise_transformation
app.include_router(enterprise_transformation.router)

# Import and include Enterprise Extended routes (webhooks, streams, code review, cost optimization, marketplace)
from services.api.src.routes import enterprise_extended
app.include_router(enterprise_extended.router)

# Import and include Revenue Intelligence routes (dynamic pricing, LTV, payments, AI code generation)
from services.api.src.routes import revenue_intelligence
app.include_router(revenue_intelligence.router)

# Import and include Advanced Intelligence routes (ML pipeline, knowledge graph, self-healing, GraphQL, lakehouse)
from services.api.src.routes import advanced_intelligence
app.include_router(advanced_intelligence.router)

# Import and include Phase 5 Intelligence routes (federated learning, vector search, prompt engineering, tracing, zero-trust, load testing, SDK portal)
from services.api.src.routes import phase5_intelligence
app.include_router(phase5_intelligence.router)

# Import and include Phase 6 systems routes (cognitive AI, multi-agent, collaboration, data mesh, security, workflows, profiling)
from services.api.src.routes import phase6_systems
app.include_router(phase6_systems.router)

# ==================== Admin Panel API ====================

try:
    from services.admin_panel_api.src.routers.admin_dashboard import router as admin_dashboard_router

    app.include_router(admin_dashboard_router)
    logger.info("Mounted admin dashboard API under /api/admin")
except Exception as e:
    logger.warning("Failed to mount admin dashboard API: %s", e)

# ==================== v4 Stable Platform APIs ====================

# v4 is additive and intentionally does not remove or change existing v3 routes.
try:
    from cognitionos_platform.api.v4.routers import api_router as platform_v4_router

    app.include_router(platform_v4_router)
    logger.info("Mounted platform v4 API routes under /api/v4")
except Exception as e:
    logger.warning("Failed to mount platform v4 API routes: %s", e)


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        log_level=config.api.log_level,
        access_log=True,
    )
