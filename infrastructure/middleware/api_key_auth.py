"""API key authentication middleware."""

import hashlib
import logging
from typing import Callable, Optional
from uuid import UUID

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from infrastructure.middleware.tenant_context import set_current_tenant

logger = logging.getLogger(__name__)


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    API key authentication middleware.
    
    Supports programmatic access via API keys for machine-to-machine
    communication. API keys are prefixed with 'cog_' for identification.
    
    Header format: Authorization: Bearer cog_xxxxxxxxxxxxxxxxxxxx
    """
    
    def __init__(
        self,
        app: ASGIApp,
        api_key_repository,
        tenant_repository,
        excluded_paths: Optional[list[str]] = None,
    ):
        super().__init__(app)
        self.api_key_repository = api_key_repository
        self.tenant_repository = tenant_repository
        self.excluded_paths = excluded_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/",
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Authenticate request via API key if present."""
        # Skip for excluded paths
        if self._is_excluded_path(request.url.path):
            return await call_next(request)
        
        # Check for API key in Authorization header
        auth_header = request.headers.get("authorization", "")
        
        if auth_header.startswith("Bearer cog_"):
            # Extract API key
            api_key = auth_header.replace("Bearer ", "")
            
            # Validate and authenticate
            try:
                authenticated, tenant, api_key_obj, error_msg = await self._authenticate_api_key(api_key)
                
                if not authenticated:
                    return JSONResponse(
                        status_code=401,
                        content={
                            "error": "InvalidAPIKey",
                            "message": error_msg or "Invalid or expired API key",
                        },
                        headers={"WWW-Authenticate": "Bearer"}
                    )
                
                # Set tenant context
                set_current_tenant(tenant)
                
                # Attach API key to request state
                request.state.api_key = api_key_obj
                request.state.authenticated_via = "api_key"
                
                logger.info(
                    f"API key authentication successful: {api_key_obj['name']}",
                    extra={
                        "tenant_id": str(tenant.id),
                        "api_key_id": str(api_key_obj['id']),
                    }
                )
                
                # Update last_used_at
                await self._update_last_used(api_key_obj['id'])
                
            except Exception as e:
                logger.error(f"API key authentication error: {e}", exc_info=True)
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": "AuthenticationError",
                        "message": "Failed to authenticate API key",
                    }
                )
        
        # Process request
        response = await call_next(request)
        return response
    
    async def _authenticate_api_key(
        self,
        api_key: str,
    ) -> tuple[bool, Optional[any], Optional[dict], Optional[str]]:
        """
        Authenticate API key and return tenant.
        
        Returns: (authenticated, tenant, api_key_obj, error_message)
        """
        # Validate key format
        if not api_key.startswith("cog_"):
            return False, None, None, "Invalid API key format"
        
        # Hash the key for lookup
        key_hash = self._hash_api_key(api_key)
        
        # Look up API key in database
        api_key_model = await self.api_key_repository.get_by_hash(key_hash)
        
        if not api_key_model:
            return False, None, None, "API key not found or revoked"
        
        # Check if active
        if not api_key_model.is_active:
            return False, None, None, "API key is inactive"
        
        # Check expiration
        from datetime import datetime
        if api_key_model.expires_at and datetime.utcnow() > api_key_model.expires_at:
            return False, None, None, "API key has expired"
        
        # Get tenant
        tenant = await self.tenant_repository.get_by_id(api_key_model.tenant_id)
        if not tenant:
            return False, None, None, "Tenant not found"
        
        # Check tenant status
        if not tenant.is_active():
            return False, None, None, f"Tenant is {tenant.status.value}"
        
        # Convert to dict for consistency with existing code
        api_key_obj = {
            'id': api_key_model.id,
            'name': api_key_model.name,
            'tenant_id': api_key_model.tenant_id,
            'scopes': api_key_model.scopes,
            'rate_limit_per_minute': api_key_model.rate_limit_per_minute,
            'is_active': api_key_model.is_active,
        }
        
        return True, tenant, api_key_obj, None
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for storage/lookup."""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    async def _update_last_used(self, api_key_id: UUID):
        """Update last_used_at timestamp for API key."""
        try:
            await self.api_key_repository.update_last_used(api_key_id)
        except Exception as e:
            logger.warning(f"Failed to update API key last_used_at: {e}")
    
    def _is_excluded_path(self, path: str) -> bool:
        """Check if path is excluded from API key authentication."""
        return any(path.startswith(excluded) for excluded in self.excluded_paths)


def generate_api_key() -> str:
    """
    Generate a secure API key.
    
    Format: cog_{32_random_hex_chars}
    Example: cog_1234567890abcdef1234567890abcdef
    """
    import secrets
    random_part = secrets.token_hex(16)  # 32 hex chars
    return f"cog_{random_part}"


def validate_api_key_format(api_key: str) -> bool:
    """Validate API key format."""
    if not api_key.startswith("cog_"):
        return False
    
    if len(api_key) != 36:  # cog_ (4) + 32 hex chars
        return False
    
    # Check that the part after prefix is valid hex
    try:
        int(api_key[4:], 16)
        return True
    except ValueError:
        return False
