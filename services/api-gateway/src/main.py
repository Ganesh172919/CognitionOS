"""
API Gateway main application.

Routes requests to appropriate microservices with rate limiting,
authentication, and circuit breaker protection.
"""

import os

# Add shared libs to path

from datetime import datetime
from typing import Optional
from uuid import UUID

import httpx
from fastapi import FastAPI, Request, WebSocket, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse

from shared.libs.config import APIGatewayConfig, load_config
from shared.libs.logger import setup_logger, get_contextual_logger, set_trace_id
from shared.libs.models import ErrorResponse
from shared.libs.middleware import (
    TracingMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
    ErrorHandlingMiddleware,
    CORSMiddleware
)
from shared.libs.utils import CircuitBreaker


# Configuration
config = load_config(APIGatewayConfig)
logger = setup_logger(__name__, level=config.log_level)

# FastAPI app
app = FastAPI(
    title="CognitionOS API Gateway",
    version=config.service_version,
    description="API Gateway and request router"
)

# Add middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(TracingMiddleware)
app.add_middleware(CORSMiddleware, allowed_origins=config.cors_origins)
app.add_middleware(
    RateLimitMiddleware,
    capacity=config.rate_limit_per_minute,
    refill_rate=config.rate_limit_per_minute / 60.0
)

# HTTP client for proxying requests
http_client: Optional[httpx.AsyncClient] = None

# Circuit breakers for each service
circuit_breakers = {}


# ============================================================================
# Service Registry
# ============================================================================

SERVICE_ROUTES = {
    "/auth": config.auth_service_url,
    "/tasks": config.task_service_url if hasattr(config, 'task_service_url') else "http://localhost:8002",
    "/agents": getattr(config, 'agent_service_url', "http://localhost:8003"),
    "/memory": getattr(config, 'memory_service_url', "http://localhost:8004"),
}


def get_service_url(path: str) -> Optional[str]:
    """
    Get service URL for a given path.

    Args:
        path: Request path

    Returns:
        Service URL or None if no match
    """
    for route_prefix, service_url in SERVICE_ROUTES.items():
        if path.startswith(route_prefix):
            return service_url
    return None


def get_circuit_breaker(service_url: str) -> CircuitBreaker:
    """Get or create circuit breaker for service."""
    if service_url not in circuit_breakers:
        circuit_breakers[service_url] = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            timeout_seconds=config.circuit_breaker_timeout
        )
    return circuit_breakers[service_url]


# ============================================================================
# Request Proxying
# ============================================================================

async def proxy_request(
    request: Request,
    service_url: str,
    path: str
) -> JSONResponse:
    """
    Proxy HTTP request to downstream service.

    Args:
        request: Incoming request
        service_url: Target service URL
        path: Request path

    Returns:
        Response from downstream service
    """
    log = get_contextual_logger(
        __name__,
        action="proxy_request",
        service=service_url,
        path=path
    )

    # Build target URL
    # Remove the service prefix from path
    for route_prefix in SERVICE_ROUTES.keys():
        if path.startswith(route_prefix):
            path = path[len(route_prefix):]
            break

    target_url = f"{service_url}{path}"

    # Copy headers (except Host)
    headers = dict(request.headers)
    headers.pop("host", None)

    # Add trace ID
    if hasattr(request.state, "trace_id"):
        headers["X-Trace-ID"] = str(request.state.trace_id)

    # Get circuit breaker
    circuit_breaker = get_circuit_breaker(service_url)

    try:
        # Proxy request through circuit breaker
        async def make_request():
            if request.method == "GET":
                return await http_client.get(
                    target_url,
                    params=request.query_params,
                    headers=headers,
                    timeout=config.request_timeout
                )
            elif request.method == "POST":
                body = await request.body()
                return await http_client.post(
                    target_url,
                    params=request.query_params,
                    headers=headers,
                    content=body,
                    timeout=config.request_timeout
                )
            elif request.method == "PUT":
                body = await request.body()
                return await http_client.put(
                    target_url,
                    params=request.query_params,
                    headers=headers,
                    content=body,
                    timeout=config.request_timeout
                )
            elif request.method == "DELETE":
                return await http_client.delete(
                    target_url,
                    params=request.query_params,
                    headers=headers,
                    timeout=config.request_timeout
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
                    detail=f"Method {request.method} not allowed"
                )

        response = circuit_breaker.call(make_request)
        if isinstance(response, tuple):  # asyncio coroutine
            response = await response[0]
        else:
            response = await response

        log.info(
            "Request proxied successfully",
            extra={"status_code": response.status_code}
        )

        # Return response
        return JSONResponse(
            status_code=response.status_code,
            content=response.json() if response.content else {},
            headers=dict(response.headers)
        )

    except httpx.TimeoutException:
        log.error("Request timeout", extra={"timeout": config.request_timeout})
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timeout"
        )
    except httpx.ConnectError as e:
        log.error("Connection error", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable"
        )
    except Exception as e:
        if "Circuit breaker is open" in str(e):
            log.error("Circuit breaker open")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service temporarily unavailable"
            )
        log.error("Proxy error", extra={"error": str(e)}, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# ============================================================================
# Routes
# ============================================================================

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def route_request(request: Request, path: str):
    """
    Route request to appropriate service.

    Handles all HTTP methods and proxies to downstream services.
    """
    full_path = f"/{path}"

    # Get service URL
    service_url = get_service_url(full_path)

    if not service_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No service found for path: {full_path}"
        )

    # Proxy request
    return await proxy_request(request, service_url, full_path)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.

    Clients connect here to receive live updates about task execution,
    agent status, and system events.
    """
    await websocket.accept()

    log = get_contextual_logger(__name__, action="websocket_connect")
    log.info("WebSocket connection established")

    try:
        # Authentication (get token from query params or first message)
        auth_message = await websocket.receive_json()
        access_token = auth_message.get("access_token")

        if not access_token:
            await websocket.send_json({
                "error": "authentication_required",
                "message": "Please provide access_token"
            })
            await websocket.close(code=1008)
            return

        # Validate token with auth service
        # In production, validate JWT here
        # For now, just accept any token

        await websocket.send_json({
            "type": "connected",
            "message": "WebSocket connected successfully"
        })

        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_json()

            # Handle different message types
            message_type = data.get("type")

            if message_type == "ping":
                await websocket.send_json({"type": "pong"})
            elif message_type == "subscribe":
                # Subscribe to specific events
                topic = data.get("topic")
                await websocket.send_json({
                    "type": "subscribed",
                    "topic": topic
                })
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })

    except Exception as e:
        log.error("WebSocket error", extra={"error": str(e)})
        try:
            await websocket.close(code=1011)
        except:
            pass


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "api-gateway",
        "version": config.service_version,
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            name: circuit_breakers.get(url, CircuitBreaker()).state
            for name, url in SERVICE_ROUTES.items()
        }
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "CognitionOS API Gateway",
        "version": config.service_version,
        "documentation": "/docs",
        "health": "/health",
        "routes": list(SERVICE_ROUTES.keys())
    }


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize service on startup."""
    global http_client

    logger.info(
        "API Gateway starting",
        extra={
            "version": config.service_version,
            "environment": config.environment
        }
    )

    # Initialize HTTP client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(config.request_timeout),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
    )

    # Initialize circuit breakers
    for service_url in SERVICE_ROUTES.values():
        get_circuit_breaker(service_url)

    logger.info("API Gateway ready")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global http_client

    logger.info("API Gateway shutting down")

    # Close HTTP client
    if http_client:
        await http_client.aclose()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower()
    )
