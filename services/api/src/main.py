"""
CognitionOS V3 API Service - Main Application

FastAPI application providing REST APIs for the V3 clean architecture.
"""

import os
from datetime import datetime
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


# Configuration
config = get_config()


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print(f"Starting {config.service_name} v{config.service_version}")
    print(f"Environment: {config.environment}")
    print(f"API URL: http://{config.api.host}:{config.api.port}")
    
    # Initialize tracing
    if config.observability.enable_tracing:
        setup_tracing(service_name=config.service_name)
        instrument_fastapi(app)
        print(f"OpenTelemetry tracing enabled (Jaeger: {config.observability.jaeger_host}:{config.observability.jaeger_port})")
    
    # Initialize metrics
    if config.observability.enable_metrics:
        print(f"Prometheus metrics enabled on /metrics")
    
    # Initialize Redis connection pool
    try:
        from infrastructure.persistence.redis_pool import RedisPoolManager
        redis_pool = await RedisPoolManager.get_instance()
        print("Redis connection pool initialized")
    except Exception as e:
        print(f"Warning: Failed to initialize Redis pool: {e}")
    
    yield
    
    # Shutdown - graceful cleanup
    print("Shutting down gracefully...")
    
    # Give existing requests time to complete (configurable timeout)
    import asyncio
    shutdown_timeout = config.api.shutdown_timeout_seconds
    print(f"Waiting for in-flight requests to complete ({shutdown_timeout}s)...")
    await asyncio.sleep(shutdown_timeout)
    
    # Close database connections
    try:
        from services.api.src.dependencies.injection import close_db
        await close_db()
        print("Database connections closed")
    except Exception as e:
        print(f"Error closing database: {e}")
    
    # Close Redis connection pool
    try:
        from infrastructure.persistence.redis_pool import RedisPoolManager
        await RedisPoolManager.reset()
        print("Redis connection pool closed")
    except Exception as e:
        print(f"Error closing Redis pool: {e}")
    
    print("Shutdown complete")


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
