"""
Feature Gate Service - Wiring DynamicFeatureGate to API routes.

Provides FastAPI dependencies and decorators for tier-based feature access.
"""

from functools import wraps
from typing import Callable, Optional
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status

from core.config.tier_config import SubscriptionTier, TierConfig, get_tier_config
from core.domain.billing.entities import SubscriptionTier as DomainSubscriptionTier


def _map_domain_tier_to_config(tier: Optional[str]) -> SubscriptionTier:
    """Map domain subscription tier string to config tier enum."""
    if not tier:
        return SubscriptionTier.FREE
    t = str(tier).strip().lower()
    mapping = {
        "free": SubscriptionTier.FREE,
        "pro": SubscriptionTier.PRO,
        "team": SubscriptionTier.TEAM,
        "enterprise": SubscriptionTier.ENTERPRISE,
    }
    return mapping.get(t, SubscriptionTier.FREE)


class FeatureGateService:
    """
    Service for checking feature access based on subscription tier.
    Wires TierConfig to API routes via FastAPI dependencies.
    """

    def __init__(self, tier_config=None):
        self._tier_config = tier_config or get_tier_config()

    def has_feature(self, tier: SubscriptionTier, feature: str) -> bool:
        """Check if tier has access to feature."""
        return self._tier_config.has_feature(tier, feature)

    def get_min_tier_for_feature(self, feature: str) -> SubscriptionTier:
        """Get minimum tier required for feature."""
        for tier in [
            SubscriptionTier.FREE,
            SubscriptionTier.PRO,
            SubscriptionTier.TEAM,
            SubscriptionTier.ENTERPRISE,
        ]:
            if self._tier_config.has_feature(tier, feature):
                return tier
        return SubscriptionTier.ENTERPRISE


_feature_gate_service: Optional[FeatureGateService] = None


def get_feature_gate_service() -> FeatureGateService:
    """Get feature gate service singleton."""
    global _feature_gate_service
    if _feature_gate_service is None:
        _feature_gate_service = FeatureGateService()
    return _feature_gate_service


async def get_tenant_tier_from_request(request: Request) -> SubscriptionTier:
    """
    Extract tenant subscription tier from request context.
    Uses get_current_tenant() from TenantContextMiddleware or API key auth.
    """
    from infrastructure.middleware.tenant_context import get_current_tenant

    tenant = get_current_tenant()
    if tenant and tenant.subscription_tier:
        return _map_domain_tier_to_config(tenant.subscription_tier)
    tier_str = getattr(request.state, "subscription_tier", None)
    if tier_str is not None:
        return _map_domain_tier_to_config(tier_str)
    return SubscriptionTier.FREE


def require_feature(feature: str):
    """
    FastAPI dependency that requires a feature for the current tenant.
    Raises 403 if tenant's tier does not have access.
    """

    async def _dependency(request: Request):
        gate = get_feature_gate_service()
        tier = await get_tenant_tier_from_request(request)
        if not gate.has_feature(tier, feature):
            min_tier = gate.get_min_tier_for_feature(feature)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "feature_not_available",
                    "feature": feature,
                    "upgrade_required": min_tier.value,
                    "message": f"Feature '{feature}' requires {min_tier.value} tier or higher.",
                },
            )
        return tier

    return _dependency


def require_tier(min_tier: SubscriptionTier):
    """
    FastAPI dependency that requires minimum subscription tier.
    Raises 403 if tenant's tier is below minimum.
    """

    tier_order = {
        SubscriptionTier.FREE: 0,
        SubscriptionTier.PRO: 1,
        SubscriptionTier.TEAM: 2,
        SubscriptionTier.ENTERPRISE: 3,
    }

    async def _dependency(request: Request):
        tier = await get_tenant_tier_from_request(request)
        if tier_order.get(tier, 0) < tier_order.get(min_tier, 0):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "tier_too_low",
                    "required_tier": min_tier.value,
                    "current_tier": tier.value,
                    "message": f"This action requires {min_tier.value} tier or higher.",
                },
            )
        return tier

    return _dependency


def require_pro_tier():
    """Shorthand for require_tier(SubscriptionTier.PRO)."""
    return require_tier(SubscriptionTier.PRO)


def require_team_tier():
    """Shorthand for require_tier(SubscriptionTier.TEAM)."""
    return require_tier(SubscriptionTier.TEAM)


def require_enterprise_tier():
    """Shorthand for require_tier(SubscriptionTier.ENTERPRISE)."""
    return require_tier(SubscriptionTier.ENTERPRISE)
