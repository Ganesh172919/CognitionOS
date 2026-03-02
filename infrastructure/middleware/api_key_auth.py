"""API key authentication middleware."""

import asyncio
import hashlib
import logging
import time
from typing import Any, Callable, Optional
from uuid import UUID

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from core.config import get_config
from infrastructure.middleware.tenant_context import set_current_tenant

logger = logging.getLogger(__name__)

_in_memory_rate_limits: dict[str, tuple[int, int]] = {}
_in_memory_rate_limit_lock = asyncio.Lock()


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
        # Backoff to avoid hammering Redis when it's unavailable.
        self._redis_rate_limit_disabled_until: float = 0.0
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
        try:
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
                            headers={"WWW-Authenticate": "Bearer"},
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
                        },
                    )

                    # Apply per-key rate limit (best-effort: Redis-backed when available; in-memory fallback otherwise).
                    allowed, headers, denial = await self._enforce_api_key_rate_limit(request, api_key_obj)
                    if not allowed:
                        return denial

                    # Enforce monthly API call quotas (best-effort: Redis-backed).
                    allowed, denial = await self._enforce_api_call_quota(tenant, api_key_obj)
                    if not allowed:
                        return denial

                    # Update last_used_at (after passing rate limits)
                    await self._update_last_used(api_key_obj["id"])

                    # Usage metering (best-effort): track tenant-scoped API call usage in Redis for later aggregation.
                    try:
                        await self._track_api_call_usage(api_key_obj)
                    except Exception:  # noqa: BLE001
                        logger.debug("Failed to track api_call usage", exc_info=True)

                except Exception as e:
                    logger.error(f"API key authentication error: {e}", exc_info=True)
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": "AuthenticationError",
                            "message": "Failed to authenticate API key",
                        },
                    )

            # Process request
            response = await call_next(request)

            # Propagate rate-limit headers when present.
            headers = getattr(request.state, "rate_limit_headers", None)
            if isinstance(headers, dict):
                for k, v in headers.items():
                    response.headers[k] = str(v)

            return response
        finally:
            # Always clear tenant context to avoid cross-request leakage.
            set_current_tenant(None)

    async def _enforce_api_key_rate_limit(self, request: Request, api_key_obj: dict) -> tuple[bool, dict, Optional[JSONResponse]]:
        """
        Enforce API key rate limits.

        Returns: (allowed, headers, denial_response)
        """
        cfg = get_config()
        raw_limit = api_key_obj.get("rate_limit_per_minute")
        if raw_limit is None:
            raw_limit = cfg.api.rate_limit_per_minute
        try:
            limit = int(raw_limit or 0)
        except Exception:
            limit = 0

        # Treat non-positive limits as unlimited to preserve backwards compatibility unless explicitly set on key.
        if limit <= 0:
            request.state.rate_limit_headers = {}
            return True, {}, None

        api_key_id = str(api_key_obj.get("id") or "")
        now = int(time.time())
        window_start = now - (now % 60)
        reset = window_start + 60

        count = None
        try:
            if time.time() < self._redis_rate_limit_disabled_until:
                raise RuntimeError("redis_rate_limit_temporarily_disabled")

            from infrastructure.persistence.redis_pool import get_redis_client

            redis = await get_redis_client()
            key = f"ratelimit:api_key:{api_key_id}:{window_start}"
            count = int(await redis.incr(key))
            if count == 1:
                await redis.expire(key, 60)
        except Exception:
            # Back off briefly after any Redis failure.
            self._redis_rate_limit_disabled_until = time.time() + 10.0
            # Redis unavailable: fallback to in-memory token counter for this process.
            async with _in_memory_rate_limit_lock:
                prior_window, prior_count = _in_memory_rate_limits.get(api_key_id, (0, 0))
                if prior_window != window_start:
                    prior_window, prior_count = window_start, 0
                prior_count += 1
                _in_memory_rate_limits[api_key_id] = (prior_window, prior_count)
                count = prior_count

        remaining = max(0, limit - int(count or 0))
        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset),
        }
        request.state.rate_limit_headers = headers

        if int(count or 0) > limit:
            retry_after = max(0, reset - now)
            headers["Retry-After"] = str(retry_after)
            logger.warning(
                "API key rate limit exceeded",
                extra={
                    "api_key_id": api_key_id,
                    "tenant_id": str(api_key_obj.get("tenant_id") or ""),
                    "limit_per_minute": limit,
                    "count": int(count or 0),
                    "path": request.url.path,
                },
            )
            denial = JSONResponse(
                status_code=429,
                content={
                    "error": "RateLimitExceeded",
                    "message": "API key rate limit exceeded",
                    "limit_per_minute": limit,
                    "reset_epoch_seconds": reset,
                },
                headers=headers,
            )
            return False, headers, denial

        return True, headers, None

    async def _enforce_api_call_quota(self, tenant: Any, api_key_obj: dict) -> tuple[bool, Optional[JSONResponse]]:
        """
        Enforce a tier-based monthly API call quota.

        This is designed to be low-latency and avoids per-request database reads by using:
        - tenant.subscription_tier as the source of truth for tier
        - Redis counters for in-period usage
        """
        tier_value = str(getattr(tenant, "subscription_tier", "") or "free").strip().lower()
        try:
            from core.domain.billing.entities import SubscriptionTier

            tier = SubscriptionTier(tier_value)
        except Exception:
            from core.domain.billing.entities import SubscriptionTier

            tier = SubscriptionTier.FREE

        from core.domain.billing.services import EntitlementService

        limit = EntitlementService.TIER_LIMITS.get(tier, {}).get("api_calls")
        if limit is None:
            return True, None

        # Redis-backed monthly counter
        from infrastructure.persistence.redis_pool import get_redis_client

        redis = await get_redis_client()
        tenant_id = str(getattr(tenant, "id", "") or api_key_obj.get("tenant_id") or "").strip()
        if not tenant_id:
            return True, None

        now = time.gmtime()
        month_key = f"{now.tm_year:04d}{now.tm_mon:02d}"
        key = f"quota:api_calls:tenant:{tenant_id}:{month_key}"

        limit_int = int(limit)
        ttl_seconds = 40 * 24 * 60 * 60

        try:
            # Atomic check-and-increment to avoid counting denied requests.
            script = """
            local k = KEYS[1]
            local lim = tonumber(ARGV[1])
            local ttl = tonumber(ARGV[2])
            local current = tonumber(redis.call('GET', k) or '0')
            if current >= lim then
              return {0, current}
            end
            current = redis.call('INCR', k)
            if current == 1 then
              redis.call('EXPIRE', k, ttl)
            end
            return {1, current}
            """
            allowed_count = await redis.eval(script, 1, key, limit_int, ttl_seconds)
            allowed_flag = int(allowed_count[0]) if isinstance(allowed_count, (list, tuple)) else 1
            count = int(allowed_count[1]) if isinstance(allowed_count, (list, tuple)) else int(allowed_count)
        except Exception:
            # Redis is unavailable; allow to preserve availability.
            return True, None

        if allowed_flag == 0:
            denial = JSONResponse(
                status_code=403,
                content={
                    "error": "UsageQuotaExceeded",
                    "message": "Monthly API call quota exceeded for this tenant",
                    "resource_type": "api_calls",
                    "tier": tier.value,
                    "limit_per_month": limit_int,
                    "current_usage": count,
                },
            )
            return False, denial

        return True, None

    async def _track_api_call_usage(self, api_key_obj: dict) -> None:
        """
        Track API calls for usage-based billing and quota enforcement.

        This intentionally avoids per-request database writes by buffering counts in Redis.
        A background job can periodically flush these aggregates into `usage_records`.
        """
        tenant_id = str(api_key_obj.get("tenant_id") or "").strip()
        if not tenant_id:
            return

        now = int(time.time())
        window_start = now - (now % 60)

        from infrastructure.persistence.redis_pool import get_redis_client

        redis = await get_redis_client()
        key = f"usage:api_calls:tenant:{tenant_id}:{window_start}"
        count = int(await redis.incr(key))
        if count == 1:
            # Keep a small buffer to ensure the flusher can pick up late-running windows.
            await redis.expire(key, 5 * 60)
    
    async def _authenticate_api_key(
        self,
        api_key: str,
    ) -> tuple[bool, Optional[Any], Optional[dict], Optional[str]]:
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
        from datetime import datetime, timezone
        if api_key_model.expires_at and datetime.now(timezone.utc) > api_key_model.expires_at:
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
