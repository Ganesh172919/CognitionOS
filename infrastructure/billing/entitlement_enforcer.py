"""
Entitlement Enforcer Infrastructure

Runtime entitlement enforcement with decorator and FastAPI dependency injection.
"""

import logging
from decimal import Decimal
from functools import wraps
from typing import Optional, Callable, Any
from uuid import UUID

from fastapi import Depends, HTTPException, status

from core.domain.billing.entities import EntitlementCheck
from core.domain.billing.services import EntitlementService, UsageMeteringService
from core.domain.billing.repositories import (
    SubscriptionRepository,
    UsageRecordRepository,
)
from infrastructure.middleware.tenant_context import get_current_tenant

logger = logging.getLogger(__name__)


class EntitlementEnforcementError(Exception):
    """Exception raised when entitlement check fails."""
    pass


class EntitlementEnforcer:
    """
    Entitlement enforcer that wraps EntitlementService with automatic usage recording.
    
    Provides runtime enforcement of subscription limits and automatic usage tracking.
    """
    
    def __init__(
        self,
        entitlement_service: EntitlementService,
        usage_service: UsageMeteringService,
    ):
        """
        Initialize entitlement enforcer.
        
        Args:
            entitlement_service: Service for checking entitlements
            usage_service: Service for recording usage
        """
        self.entitlement_service = entitlement_service
        self.usage_service = usage_service
        logger.info("EntitlementEnforcer initialized")
    
    async def check_and_record(
        self,
        tenant_id: UUID,
        resource_type: str,
        quantity: Decimal = Decimal("1"),
        unit: str = "count",
        metadata: Optional[dict] = None,
    ) -> EntitlementCheck:
        """
        Check entitlement and record usage if allowed.
        
        Args:
            tenant_id: Tenant identifier
            resource_type: Type of resource being consumed
            quantity: Quantity of resource
            unit: Unit of measurement
            metadata: Optional usage metadata
            
        Returns:
            EntitlementCheck result
            
        Raises:
            EntitlementEnforcementError: If entitlement check fails
        """
        check_result = await self.entitlement_service.check_entitlement(
            tenant_id=tenant_id,
            resource_type=resource_type,
            quantity=quantity,
        )
        
        if not check_result.allowed:
            logger.warning(
                f"Entitlement check failed for tenant {tenant_id}: {check_result.reason}",
                extra={
                    "tenant_id": str(tenant_id),
                    "resource_type": resource_type,
                    "quantity": str(quantity),
                    "reason": check_result.reason,
                }
            )
            raise EntitlementEnforcementError(check_result.reason)
        
        try:
            await self.usage_service.record_usage(
                tenant_id=tenant_id,
                resource_type=resource_type,
                quantity=quantity,
                unit=unit,
                metadata=metadata,
            )
            
            logger.debug(
                f"Recorded usage for tenant {tenant_id}: {quantity} {resource_type}",
                extra={
                    "tenant_id": str(tenant_id),
                    "resource_type": resource_type,
                    "quantity": str(quantity),
                }
            )
        except Exception as e:
            logger.error(
                f"Failed to record usage for tenant {tenant_id}: {e}",
                extra={
                    "tenant_id": str(tenant_id),
                    "resource_type": resource_type,
                    "error": str(e),
                }
            )
        
        return check_result
    
    async def check_only(
        self,
        tenant_id: UUID,
        resource_type: str,
        quantity: Decimal = Decimal("1"),
    ) -> EntitlementCheck:
        """
        Check entitlement without recording usage.
        
        Args:
            tenant_id: Tenant identifier
            resource_type: Type of resource
            quantity: Quantity of resource
            
        Returns:
            EntitlementCheck result
        """
        return await self.entitlement_service.check_entitlement(
            tenant_id=tenant_id,
            resource_type=resource_type,
            quantity=quantity,
        )
    
    async def record_usage_only(
        self,
        tenant_id: UUID,
        resource_type: str,
        quantity: Decimal = Decimal("1"),
        unit: str = "count",
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Record usage without checking entitlement.
        
        Use this when entitlement was already checked separately.
        
        Args:
            tenant_id: Tenant identifier
            resource_type: Type of resource
            quantity: Quantity of resource
            unit: Unit of measurement
            metadata: Optional usage metadata
        """
        try:
            await self.usage_service.record_usage(
                tenant_id=tenant_id,
                resource_type=resource_type,
                quantity=quantity,
                unit=unit,
                metadata=metadata,
            )
            
            logger.debug(
                f"Recorded usage for tenant {tenant_id}: {quantity} {resource_type}",
                extra={
                    "tenant_id": str(tenant_id),
                    "resource_type": resource_type,
                    "quantity": str(quantity),
                }
            )
        except Exception as e:
            logger.error(
                f"Failed to record usage for tenant {tenant_id}: {e}",
                extra={
                    "tenant_id": str(tenant_id),
                    "resource_type": resource_type,
                    "error": str(e),
                }
            )
            raise


def require_entitlement(
    resource_type: str,
    quantity: Decimal = Decimal("1"),
    unit: str = "count",
):
    """
    Decorator to enforce entitlement checks on functions.
    
    Automatically checks entitlement and records usage when the decorated
    function is called. The function must have a tenant_id parameter or
    access to current tenant context.
    
    Args:
        resource_type: Type of resource being consumed
        quantity: Quantity of resource (default: 1)
        unit: Unit of measurement (default: "count")
        
    Returns:
        Decorated function
        
    Example:
        @require_entitlement("executions", quantity=Decimal("1"))
        async def execute_workflow(tenant_id: UUID, workflow_id: UUID):
            # This will only run if tenant has available executions
            pass
            
        @require_entitlement("tokens", quantity=Decimal("1000"))
        async def process_with_llm(tenant_id: UUID, prompt: str):
            # This will only run if tenant has 1000+ tokens available
            pass
    """
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            enforcer = kwargs.get("enforcer")
            tenant_id = kwargs.get("tenant_id")
            
            if not enforcer:
                logger.warning(
                    f"No enforcer provided to {func.__name__}, skipping entitlement check"
                )
                return await func(*args, **kwargs)
            
            if not tenant_id:
                tenant = get_current_tenant()
                if tenant:
                    tenant_id = tenant.id
                else:
                    logger.warning(
                        f"No tenant_id available for {func.__name__}, skipping entitlement check"
                    )
                    return await func(*args, **kwargs)
            
            metadata = {
                "function": func.__name__,
                "module": func.__module__,
            }
            
            await enforcer.check_and_record(
                tenant_id=tenant_id,
                resource_type=resource_type,
                quantity=quantity,
                unit=unit,
                metadata=metadata,
            )
            
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger.warning(
                f"Synchronous function {func.__name__} decorated with require_entitlement. "
                "Entitlement checks work best with async functions."
            )
            return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


async def get_entitlement_enforcer(
    subscription_repo: SubscriptionRepository = Depends(),
    usage_repo: UsageRecordRepository = Depends(),
) -> EntitlementEnforcer:
    """
    FastAPI dependency for getting EntitlementEnforcer.
    
    This should be configured in your FastAPI app's dependency injection.
    
    Args:
        subscription_repo: Subscription repository (injected)
        usage_repo: Usage record repository (injected)
        
    Returns:
        Configured EntitlementEnforcer instance
        
    Example:
        from fastapi import FastAPI, Depends
        
        app = FastAPI()
        
        @app.post("/workflows/{workflow_id}/execute")
        async def execute_workflow(
            workflow_id: UUID,
            tenant_id: UUID,
            enforcer: EntitlementEnforcer = Depends(get_entitlement_enforcer),
        ):
            await enforcer.check_and_record(
                tenant_id=tenant_id,
                resource_type="executions",
                quantity=Decimal("1"),
            )
            # Execute workflow...
    """
    entitlement_service = EntitlementService(
        subscription_repository=subscription_repo,
        usage_repository=usage_repo,
    )
    
    usage_service = UsageMeteringService(
        usage_repository=usage_repo,
    )
    
    return EntitlementEnforcer(
        entitlement_service=entitlement_service,
        usage_service=usage_service,
    )


class EntitlementMiddleware:
    """
    Optional middleware for automatic entitlement enforcement.
    
    Can be used to enforce entitlements based on request path patterns.
    """
    
    def __init__(
        self,
        enforcer: EntitlementEnforcer,
        path_mappings: Optional[dict[str, tuple[str, Decimal]]] = None,
    ):
        """
        Initialize entitlement middleware.
        
        Args:
            enforcer: EntitlementEnforcer instance
            path_mappings: Optional mapping of path patterns to (resource_type, quantity)
        """
        self.enforcer = enforcer
        self.path_mappings = path_mappings or {}
        logger.info("EntitlementMiddleware initialized")
    
    async def check_request_entitlement(
        self,
        path: str,
        tenant_id: UUID,
    ) -> Optional[EntitlementCheck]:
        """
        Check entitlement for a request path.
        
        Args:
            path: Request path
            tenant_id: Tenant identifier
            
        Returns:
            EntitlementCheck if path matched, None otherwise
        """
        for pattern, (resource_type, quantity) in self.path_mappings.items():
            if pattern in path:
                return await self.enforcer.check_and_record(
                    tenant_id=tenant_id,
                    resource_type=resource_type,
                    quantity=quantity,
                )
        
        return None


def check_entitlement_or_raise(
    check_result: EntitlementCheck,
    status_code: int = status.HTTP_402_PAYMENT_REQUIRED,
) -> None:
    """
    Raise HTTPException if entitlement check failed.
    
    Helper function for FastAPI endpoints.
    
    Args:
        check_result: Result of entitlement check
        status_code: HTTP status code to return (default: 402)
        
    Raises:
        HTTPException: If entitlement check failed
        
    Example:
        check = await enforcer.check_only(tenant_id, "executions")
        check_entitlement_or_raise(check)
        # Continue processing...
    """
    if not check_result.allowed:
        detail = {
            "error": "entitlement_exceeded",
            "message": check_result.reason,
            "resource_limit": str(check_result.limit) if check_result.limit else None,
            "current_usage": str(check_result.current_usage) if check_result.current_usage else None,
            "remaining": str(check_result.remaining) if check_result.remaining else None,
        }
        
        raise HTTPException(
            status_code=status_code,
            detail=detail,
        )
