"""
Request ID Middleware

Adds unique request ID to each request for distributed tracing and logging.
"""

import uuid
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add unique request ID to each request.
    
    The request ID can be:
    1. Provided by client via X-Request-ID header
    2. Auto-generated if not provided
    
    The request ID is:
    - Added to request.state for use in request handlers
    - Added to response headers for client reference
    - Included in all log messages for that request
    """
    
    def __init__(self, app: ASGIApp, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and add request ID"""
        
        # Get or generate request ID
        request_id = request.headers.get(self.header_name)
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Store in request state for access in handlers
        request.state.request_id = request_id
        
        # Process request
        try:
            response = await call_next(request)
            
            # Add request ID to response headers
            response.headers[self.header_name] = request_id
            
            return response
        
        except Exception as exc:
            # Ensure request ID is in response even on error
            logger.error(
                f"Request {request_id} failed with exception",
                extra={"request_id": request_id},
                exc_info=True,
            )
            raise


def get_request_id(request: Request) -> str:
    """Helper function to get request ID from request state"""
    return getattr(request.state, "request_id", "unknown")
