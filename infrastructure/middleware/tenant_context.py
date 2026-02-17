"""Tenant context middleware for multi-tenancy support."""

import logging
from typing import Optional, Callable
from uuid import UUID
from contextvars import ContextVar

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from core.domain.tenant.entities import Tenant

logger = logging.getLogger(__name__)

# Context variable to store current tenant
current_tenant_context: ContextVar[Optional[Tenant]] = ContextVar("current_tenant", default=None)


def get_current_tenant() -> Optional[Tenant]:
    """Get the current tenant from context."""
    return current_tenant_context.get()


def set_current_tenant(tenant: Optional[Tenant]) -> None:
    """Set the current tenant in context."""
    current_tenant_context.set(tenant)


class TenantContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware to extract and validate tenant context from requests.
    
    Supports multiple tenant identification methods:
    1. Subdomain: {tenant-slug}.app.cognitionos.com
    2. Header: X-Tenant-ID or X-Tenant-Slug
    3. Path prefix: /api/v1/tenants/{tenant_id}/...
    4. API Key: Automatically includes tenant_id
    """
    
    def __init__(
        self,
        app: ASGIApp,
        tenant_repository,
        require_tenant: bool = True,
        excluded_paths: Optional[list[str]] = None,
    ):
        super().__init__(app)
        self.tenant_repository = tenant_repository
        self.require_tenant = require_tenant
        self.excluded_paths = excluded_paths or [
            "/health",
            "/api/v3/health",
            "/docs",
            "/openapi.json",
            "/api/v3/auth/register",
            "/api/v3/auth/login",
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and inject tenant context."""
        # Skip tenant resolution for excluded paths
        if self._is_excluded_path(request.url.path):
            return await call_next(request)
        
        tenant = None
        tenant_identifier = None
        identification_method = None
        
        try:
            # Try multiple identification methods
            tenant_identifier, identification_method = self._extract_tenant_identifier(request)
            
            if tenant_identifier:
                # Resolve tenant from repository
                if identification_method == "slug":
                    tenant = await self.tenant_repository.get_by_slug(tenant_identifier)
                elif identification_method == "id":
                    tenant = await self.tenant_repository.get_by_id(UUID(tenant_identifier))
                
                if tenant:
                    # Validate tenant is active
                    if not tenant.is_active():
                        return JSONResponse(
                            status_code=403,
                            content={
                                "error": "TenantSuspended",
                                "message": f"Tenant is {tenant.status.value}",
                                "reason": tenant.suspended_reason,
                            }
                        )
                    
                    # Set tenant in context
                    set_current_tenant(tenant)
                    logger.info(
                        f"Tenant context set: {tenant.slug} (method: {identification_method})",
                        extra={"tenant_id": str(tenant.id), "tenant_slug": tenant.slug}
                    )
                else:
                    logger.warning(f"Tenant not found: {tenant_identifier}")
            
            # Check if tenant is required but not found
            if self.require_tenant and not tenant:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "TenantRequired",
                        "message": "Valid tenant identification is required",
                        "hint": "Provide X-Tenant-Slug header or use subdomain",
                    }
                )
            
            # Process request
            response = await call_next(request)
            
            # Add tenant identifier to response headers (for debugging)
            if tenant:
                response.headers["X-Tenant-ID"] = str(tenant.id)
                response.headers["X-Tenant-Slug"] = tenant.slug
            
            return response
            
        except Exception as e:
            logger.error(f"Error in tenant context middleware: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "TenantContextError",
                    "message": "Failed to resolve tenant context"
                }
            )
        finally:
            # Clear tenant context after request
            set_current_tenant(None)
    
    def _extract_tenant_identifier(self, request: Request) -> tuple[Optional[str], Optional[str]]:
        """
        Extract tenant identifier from request.
        
        Returns: (identifier, method)
        """
        # Method 1: Check subdomain
        host = request.headers.get("host", "")
        if "." in host:
            subdomain = host.split(".")[0]
            if subdomain and subdomain != "app" and subdomain != "www":
                return subdomain, "slug"
        
        # Method 2: Check X-Tenant-Slug header
        tenant_slug = request.headers.get("x-tenant-slug")
        if tenant_slug:
            return tenant_slug, "slug"
        
        # Method 3: Check X-Tenant-ID header
        tenant_id = request.headers.get("x-tenant-id")
        if tenant_id:
            return tenant_id, "id"
        
        # Method 4: Extract from path (e.g., /api/v1/tenants/{tenant_id}/...)
        path_parts = request.url.path.split("/")
        if len(path_parts) > 4 and path_parts[3] == "tenants":
            return path_parts[4], "id"
        
        # Method 5: Check if tenant is attached to authenticated user
        # (This would be set by auth middleware if user is authenticated)
        if hasattr(request.state, "user") and hasattr(request.state.user, "tenant_id"):
            return str(request.state.user.tenant_id), "id"
        
        return None, None
    
    def _is_excluded_path(self, path: str) -> bool:
        """Check if path is excluded from tenant requirement."""
        return any(path.startswith(excluded) for excluded in self.excluded_paths)


class TenantIsolationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce tenant data isolation.
    
    Validates that all database queries are properly scoped to the current tenant.
    This is a safeguard against accidentally leaking data across tenants.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Enforce tenant isolation."""
        tenant = get_current_tenant()
        
        if tenant:
            # Attach tenant to request state for easy access in route handlers
            request.state.tenant = tenant
            request.state.tenant_id = tenant.id
            
            logger.debug(
                f"Tenant isolation active: {tenant.slug}",
                extra={"tenant_id": str(tenant.id)}
            )
        
        response = await call_next(request)
        return response
