"""
CognitionOS Error Handlers

Centralized error handling for FastAPI application.
Provides consistent error responses and proper logging.
"""

import logging
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.exceptions import (
    CognitionOSException,
    DatabaseError,
    WorkflowError,
    AgentError,
    AuthenticationError,
    AuthorizationError,
    TenantError,
    BillingError,
    PaymentRequiredError,
    ValidationError,
    RateLimitError,
    ResourceError,
)


logger = logging.getLogger(__name__)


def generate_error_id() -> str:
    """Generate unique error ID for tracking"""
    return f"err_{uuid.uuid4().hex[:12]}"


def create_error_response(
    error_id: str,
    status_code: int,
    error_type: str,
    message: str,
    details: Dict[str, Any] = None,
    path: str = None,
) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "error": {
            "id": error_id,
            "type": error_type,
            "message": message,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat(),
            "path": path,
        },
        "status": status_code,
    }


async def cognitionos_exception_handler(
    request: Request,
    exc: CognitionOSException,
) -> JSONResponse:
    """Handle CognitionOS custom exceptions"""
    error_id = generate_error_id()
    
    # Map exception types to HTTP status codes
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    if isinstance(exc, ValidationError):
        status_code = status.HTTP_400_BAD_REQUEST
    elif isinstance(exc, (AuthenticationError, AuthorizationError)):
        status_code = status.HTTP_401_UNAUTHORIZED
    elif isinstance(exc, (WorkflowError, AgentError, ResourceError)):
        if "NotFound" in exc.__class__.__name__:
            status_code = status.HTTP_404_NOT_FOUND
        else:
            status_code = status.HTTP_400_BAD_REQUEST
    elif isinstance(exc, (TenantError, BillingError)):
        # Check exception class name for payment-related errors
        if isinstance(exc, PaymentRequiredError):
            status_code = status.HTTP_402_PAYMENT_REQUIRED
        else:
            status_code = status.HTTP_403_FORBIDDEN
    elif isinstance(exc, RateLimitError):
        status_code = status.HTTP_429_TOO_MANY_REQUESTS
    elif isinstance(exc, DatabaseError):
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    # Log error with details
    logger.error(
        f"Error {error_id}: {exc.__class__.__name__} - {exc.message}",
        extra={
            "error_id": error_id,
            "exception_type": exc.__class__.__name__,
            "error_code": exc.error_code,
            "details": exc.details,
            "path": str(request.url),
            "method": request.method,
        },
        exc_info=True,
    )
    
    # Create response
    response_data = create_error_response(
        error_id=error_id,
        status_code=status_code,
        error_type=exc.error_code,
        message=exc.message,
        details=exc.details,
        path=str(request.url.path),
    )
    
    return JSONResponse(
        status_code=status_code,
        content=response_data,
    )


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Handle standard HTTP exceptions"""
    error_id = generate_error_id()
    
    # Log error
    logger.warning(
        f"HTTP Error {error_id}: {exc.status_code} - {exc.detail}",
        extra={
            "error_id": error_id,
            "status_code": exc.status_code,
            "path": str(request.url),
            "method": request.method,
        },
    )
    
    # Create response
    response_data = create_error_response(
        error_id=error_id,
        status_code=exc.status_code,
        error_type="HTTPError",
        message=str(exc.detail),
        path=str(request.url.path),
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle request validation errors"""
    error_id = generate_error_id()
    
    # Extract validation errors
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })
    
    # Log validation error
    logger.warning(
        f"Validation Error {error_id}: {len(errors)} field(s) failed",
        extra={
            "error_id": error_id,
            "validation_errors": errors,
            "path": str(request.url),
            "method": request.method,
        },
    )
    
    # Create response
    response_data = create_error_response(
        error_id=error_id,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_type="ValidationError",
        message="Request validation failed",
        details={"errors": errors},
        path=str(request.url.path),
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response_data,
    )


async def general_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected exceptions"""
    error_id = generate_error_id()
    
    # Log full traceback for debugging
    logger.error(
        f"Unexpected Error {error_id}: {exc.__class__.__name__}",
        extra={
            "error_id": error_id,
            "exception_type": exc.__class__.__name__,
            "path": str(request.url),
            "method": request.method,
            "traceback": traceback.format_exc(),
        },
        exc_info=True,
    )
    
    # In production, don't expose internal error details
    import os
    is_production = os.getenv("ENVIRONMENT", "development") == "production"
    
    if is_production:
        message = "An internal server error occurred. Please contact support with error ID."
        details = {}
    else:
        message = f"Internal server error: {str(exc)}"
        details = {
            "exception_type": exc.__class__.__name__,
            "traceback": traceback.format_exc().split("\n")[-10:],  # Last 10 lines
        }
    
    # Create response
    response_data = create_error_response(
        error_id=error_id,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_type="InternalServerError",
        message=message,
        details=details,
        path=str(request.url.path),
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_data,
    )


def register_error_handlers(app):
    """Register all error handlers with FastAPI app"""
    app.add_exception_handler(CognitionOSException, cognitionos_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
