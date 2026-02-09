"""
Reusable middleware for web services.
"""

import time
from datetime import datetime
from functools import wraps
from typing import Callable, Optional
from uuid import UUID

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from shared.libs.logger import get_logger, set_trace_id, get_trace_id
from shared.libs.models import ErrorResponse
from shared.libs.utils import RateLimiter


logger = get_logger(__name__)


class TracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add trace IDs to all requests.
    """

    async def dispatch(self, request: Request, call_next):
        """Process request with tracing."""
        # Get or generate trace ID
        trace_id = request.headers.get("X-Trace-ID")
        if trace_id:
            try:
                trace_id = UUID(trace_id)
            except ValueError:
                trace_id = None

        if not trace_id:
            trace_id = set_trace_id()
        else:
            set_trace_id(trace_id)

        # Add trace ID to request state
        request.state.trace_id = trace_id

        # Process request
        response = await call_next(request)

        # Add trace ID to response headers
        response.headers["X-Trace-ID"] = str(trace_id)

        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all requests and responses.
    """

    async def dispatch(self, request: Request, call_next):
        """Log request and response."""
        start_time = time.time()

        # Log request
        logger.info(
            "Request started",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None,
            }
        )

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)

        # Log response
        logger.info(
            "Request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            }
        )

        # Add duration to response headers
        response.headers["X-Response-Time"] = str(duration_ms)

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to rate limit requests.
    """

    def __init__(self, app, capacity: int = 60, refill_rate: float = 1.0):
        """
        Initialize rate limiter.

        Args:
            app: FastAPI application
            capacity: Maximum requests in bucket
            refill_rate: Requests per second refill rate
        """
        super().__init__(app)
        self.limiters = {}  # IP -> RateLimiter
        self.capacity = capacity
        self.refill_rate = refill_rate

    async def dispatch(self, request: Request, call_next):
        """Check rate limit before processing."""
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Get or create rate limiter for this IP
        if client_ip not in self.limiters:
            self.limiters[client_ip] = RateLimiter(
                capacity=self.capacity,
                refill_rate=self.refill_rate
            )

        limiter = self.limiters[client_ip]

        # Check if allowed
        if not limiter.allow():
            logger.warning(
                "Rate limit exceeded",
                extra={"client_ip": client_ip}
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=ErrorResponse(
                    error="rate_limit_exceeded",
                    message="Too many requests. Please try again later."
                ).dict()
            )

        return await call_next(request)


class CORSMiddleware(BaseHTTPMiddleware):
    """
    CORS middleware with configurable origins.
    """

    def __init__(self, app, allowed_origins: list):
        """
        Initialize CORS middleware.

        Args:
            app: FastAPI application
            allowed_origins: List of allowed origins or ["*"] for all
        """
        super().__init__(app)
        self.allowed_origins = allowed_origins

    async def dispatch(self, request: Request, call_next):
        """Add CORS headers to response."""
        response = await call_next(request)

        origin = request.headers.get("origin")

        if "*" in self.allowed_origins or origin in self.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Trace-ID"
            response.headers["Access-Control-Max-Age"] = "3600"

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to catch and format exceptions.
    """

    async def dispatch(self, request: Request, call_next):
        """Catch exceptions and return formatted error."""
        try:
            return await call_next(request)
        except Exception as e:
            logger.error(
                "Unhandled exception",
                extra={"error": str(e), "type": type(e).__name__},
                exc_info=True
            )

            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="internal_server_error",
                    message="An unexpected error occurred.",
                    details={"trace_id": str(get_trace_id())} if get_trace_id() else None
                ).dict()
            )


def require_auth(required_roles: Optional[list] = None):
    """
    Decorator to require authentication and optionally specific roles.

    Args:
        required_roles: List of required roles, or None for any authenticated user

    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Check if user is authenticated
            if not hasattr(request.state, "user"):
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content=ErrorResponse(
                        error="unauthorized",
                        message="Authentication required."
                    ).dict()
                )

            user = request.state.user

            # Check roles if required
            if required_roles and user.role not in required_roles:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content=ErrorResponse(
                        error="forbidden",
                        message="Insufficient permissions."
                    ).dict()
                )

            return await func(request, *args, **kwargs)

        return wrapper
    return decorator


def validate_user_isolation(user_id_param: str = "user_id"):
    """
    Decorator to ensure user can only access their own resources.

    Args:
        user_id_param: Name of the user_id parameter

    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Get authenticated user
            if not hasattr(request.state, "user"):
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content=ErrorResponse(
                        error="unauthorized",
                        message="Authentication required."
                    ).dict()
                )

            auth_user = request.state.user

            # Get requested user_id
            requested_user_id = kwargs.get(user_id_param)

            # Admin can access any resource
            if auth_user.role == "admin":
                return await func(request, *args, **kwargs)

            # Otherwise, user can only access their own resources
            if str(auth_user.id) != str(requested_user_id):
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content=ErrorResponse(
                        error="forbidden",
                        message="You can only access your own resources."
                    ).dict()
                )

            return await func(request, *args, **kwargs)

        return wrapper
    return decorator
