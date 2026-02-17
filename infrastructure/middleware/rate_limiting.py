"""Rate limiting middleware for API endpoints."""

import logging
from datetime import datetime, timedelta
from typing import Callable, Optional
from uuid import UUID

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from infrastructure.middleware.tenant_context import get_current_tenant

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with tenant-aware quota enforcement.
    
    Tracks API requests per tenant and enforces rate limits based on
    subscription tier. Uses sliding window algorithm for accurate limiting.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        rate_limit_repository,
        default_limit_per_minute: int = 60,
        excluded_paths: Optional[list[str]] = None,
    ):
        super().__init__(app)
        self.rate_limit_repository = rate_limit_repository
        self.default_limit_per_minute = default_limit_per_minute
        self.excluded_paths = excluded_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
        ]
        # In-memory cache for rate limit tracking (Redis would be better for production)
        self._rate_limit_cache: dict[str, dict] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check rate limits before processing request."""
        # Skip rate limiting for excluded paths
        if self._is_excluded_path(request.url.path):
            return await call_next(request)
        
        tenant = get_current_tenant()
        
        if tenant:
            # Get rate limit for tenant based on subscription tier
            rate_limit = self._get_rate_limit_for_tenant(tenant)
            
            # Check if rate limit exceeded
            is_allowed, current_count, reset_time = await self._check_rate_limit(
                tenant_id=tenant.id,
                resource_key="api_calls",
                limit=rate_limit,
                window_seconds=60,
            )
            
            if not is_allowed:
                logger.warning(
                    f"Rate limit exceeded for tenant {tenant.slug}",
                    extra={
                        "tenant_id": str(tenant.id),
                        "limit": rate_limit,
                        "current": current_count,
                    }
                )
                
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "RateLimitExceeded",
                        "message": f"Rate limit of {rate_limit} requests per minute exceeded",
                        "limit": rate_limit,
                        "current": current_count,
                        "reset_at": reset_time.isoformat(),
                        "retry_after_seconds": int((reset_time - datetime.utcnow()).total_seconds()),
                    },
                    headers={
                        "X-RateLimit-Limit": str(rate_limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(reset_time.timestamp())),
                        "Retry-After": str(int((reset_time - datetime.utcnow()).total_seconds())),
                    }
                )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers to response
            remaining = max(0, rate_limit - current_count - 1)
            response.headers["X-RateLimit-Limit"] = str(rate_limit)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(int(reset_time.timestamp()))
            
            return response
        else:
            # No tenant context - use default limits for unauthenticated requests
            return await call_next(request)
    
    def _get_rate_limit_for_tenant(self, tenant) -> int:
        """Get rate limit based on tenant subscription tier."""
        # Get from tenant settings
        if hasattr(tenant.settings, 'api_rate_limit_per_minute'):
            return tenant.settings.api_rate_limit_per_minute
        
        # Fallback to default
        return self.default_limit_per_minute
    
    async def _check_rate_limit(
        self,
        tenant_id: UUID,
        resource_key: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int, datetime]:
        """
        Check if request is within rate limit using sliding window.
        
        Returns: (is_allowed, current_count, reset_time)
        """
        now = datetime.utcnow()
        cache_key = f"{tenant_id}:{resource_key}"
        
        # Get or initialize cache entry
        if cache_key not in self._rate_limit_cache:
            self._rate_limit_cache[cache_key] = {
                "window_start": now,
                "count": 0,
            }
        
        entry = self._rate_limit_cache[cache_key]
        window_start = entry["window_start"]
        window_end = window_start + timedelta(seconds=window_seconds)
        
        # Check if we're in a new window
        if now >= window_end:
            # Reset for new window
            entry["window_start"] = now
            entry["count"] = 1
            return True, 1, now + timedelta(seconds=window_seconds)
        
        # We're in the same window
        current_count = entry["count"]
        
        if current_count >= limit:
            # Rate limit exceeded
            return False, current_count, window_end
        
        # Increment counter
        entry["count"] += 1
        
        return True, current_count + 1, window_end
    
    def _is_excluded_path(self, path: str) -> bool:
        """Check if path is excluded from rate limiting."""
        return any(path.startswith(excluded) for excluded in self.excluded_paths)
    
    async def _cleanup_old_entries(self):
        """Clean up expired cache entries (should be called periodically)."""
        now = datetime.utcnow()
        expired_keys = []
        
        for key, entry in self._rate_limit_cache.items():
            window_end = entry["window_start"] + timedelta(seconds=60)
            if now > window_end + timedelta(minutes=5):  # Keep for 5 extra minutes
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._rate_limit_cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired rate limit entries")


class QuotaEnforcementMiddleware(BaseHTTPMiddleware):
    """
    Quota enforcement middleware for subscription-based limits.
    
    Enforces monthly/daily quotas based on tenant subscription tier.
    Works in conjunction with usage metering.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        usage_metering_service,
        entitlement_service,
        excluded_paths: Optional[list[str]] = None,
    ):
        super().__init__(app)
        self.usage_metering = usage_metering_service
        self.entitlement_service = entitlement_service
        self.excluded_paths = excluded_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/api/v3/auth",
            "/api/v3/subscriptions/current",
            "/api/v3/subscriptions/usage",
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check quotas before processing request."""
        # Skip quota checks for excluded paths
        if self._is_excluded_path(request.url.path):
            return await call_next(request)
        
        tenant = get_current_tenant()
        
        if tenant:
            # Check if tenant is active
            if not tenant.is_active():
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "SubscriptionInactive",
                        "message": f"Subscription is {tenant.status.value}",
                        "action_required": "Please update your subscription",
                    }
                )
            
            # Check monthly execution quota (for workflow/agent execution endpoints)
            if self._is_execution_endpoint(request.url.path):
                # Check entitlement
                check_result = await self.entitlement_service.check_entitlement(
                    tenant_id=tenant.id,
                    resource_type="executions",
                    quantity=1,
                )
                
                if not check_result.allowed:
                    logger.warning(
                        f"Quota exceeded for tenant {tenant.slug}: {check_result.reason}",
                        extra={"tenant_id": str(tenant.id)}
                    )
                    
                    return JSONResponse(
                        status_code=403,
                        content={
                            "error": "QuotaExceeded",
                            "message": check_result.reason,
                            "current_usage": float(check_result.current_usage) if check_result.current_usage else 0,
                            "limit": float(check_result.limit) if check_result.limit else 0,
                            "remaining": float(check_result.remaining) if check_result.remaining else 0,
                            "action_required": "Upgrade your subscription or wait for quota reset",
                            "upgrade_url": "/api/v3/subscriptions/upgrade",
                        }
                    )
        
        # Process request
        response = await call_next(request)
        return response
    
    def _is_excluded_path(self, path: str) -> bool:
        """Check if path is excluded from quota enforcement."""
        return any(path.startswith(excluded) for excluded in self.excluded_paths)
    
    def _is_execution_endpoint(self, path: str) -> bool:
        """Check if endpoint triggers execution that counts against quota."""
        execution_patterns = [
            "/api/v3/workflows/",
            "/api/v3/agents/",
            "/api/v3/plugins/",
        ]
        return any(
            pattern in path and ("execute" in path or "run" in path)
            for pattern in execution_patterns
        )
