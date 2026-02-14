"""
CognitionOS V3 API Service - Main Application

FastAPI application providing REST APIs for the V3 clean architecture.
"""

import sys
import os
from datetime import datetime
from contextlib import asynccontextmanager

# Add core modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

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
    
    yield
    
    # Shutdown
    print("Shutting down...")


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
    
    Built with Domain-Driven Design and Clean Architecture principles.
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ==================== Middleware ====================

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


# ==================== Error Handlers ====================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message=str(exc),
            trace_id=getattr(request.state, "trace_id", None),
        ).model_dump(),
    )


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
