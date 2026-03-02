"""
SaaS tier configuration for CognitionOS.

Defines limits and features per subscription tier.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set


class SubscriptionTier(str, Enum):
    """Subscription tier levels."""

    FREE = "free"
    PRO = "pro"
    TEAM = "team"
    ENTERPRISE = "enterprise"


@dataclass(frozen=True)
class SubscriptionTierLimits:
    """Limits for a subscription tier."""

    tier: SubscriptionTier
    executions_per_month: int
    tokens_per_month: int
    api_keys_max: int
    rate_limit_per_minute: int
    concurrent_runs: int
    streaming_enabled: bool
    plugin_install_enabled: bool
    sso_enabled: bool
    rbac_enabled: bool
    features: Set[str] = field(default_factory=set)


class TierConfig:
    """Tier configuration provider."""

    _limits: Dict[SubscriptionTier, SubscriptionTierLimits] = {}
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        if cls._initialized:
            return
        cls._limits = {
            SubscriptionTier.FREE: SubscriptionTierLimits(
                tier=SubscriptionTier.FREE,
                executions_per_month=100,
                tokens_per_month=10_000,
                api_keys_max=1,
                rate_limit_per_minute=60,
                concurrent_runs=1,
                streaming_enabled=False,
                plugin_install_enabled=False,
                sso_enabled=False,
                rbac_enabled=False,
                features={"basic_workflows", "api_access"},
            ),
            SubscriptionTier.PRO: SubscriptionTierLimits(
                tier=SubscriptionTier.PRO,
                executions_per_month=1_000,
                tokens_per_month=100_000,
                api_keys_max=5,
                rate_limit_per_minute=300,
                concurrent_runs=3,
                streaming_enabled=True,
                plugin_install_enabled=True,
                sso_enabled=False,
                rbac_enabled=False,
                features={
                    "basic_workflows",
                    "api_access",
                    "advanced_workflows",
                    "streaming",
                    "plugin_install",
                },
            ),
            SubscriptionTier.TEAM: SubscriptionTierLimits(
                tier=SubscriptionTier.TEAM,
                executions_per_month=10_000,
                tokens_per_month=1_000_000,
                api_keys_max=20,
                rate_limit_per_minute=1000,
                concurrent_runs=10,
                streaming_enabled=True,
                plugin_install_enabled=True,
                sso_enabled=False,
                rbac_enabled=True,
                features={
                    "basic_workflows",
                    "api_access",
                    "advanced_workflows",
                    "streaming",
                    "plugin_install",
                    "rbac",
                    "team_collaboration",
                },
            ),
            SubscriptionTier.ENTERPRISE: SubscriptionTierLimits(
                tier=SubscriptionTier.ENTERPRISE,
                executions_per_month=-1,  # Unlimited
                tokens_per_month=-1,
                api_keys_max=-1,
                rate_limit_per_minute=10000,
                concurrent_runs=100,
                streaming_enabled=True,
                plugin_install_enabled=True,
                sso_enabled=True,
                rbac_enabled=True,
                features={
                    "basic_workflows",
                    "api_access",
                    "advanced_workflows",
                    "streaming",
                    "plugin_install",
                    "rbac",
                    "team_collaboration",
                    "sso",
                    "dedicated_support",
                    "sla",
                },
            ),
        }
        cls._initialized = True

    @classmethod
    def get_limits(cls, tier: SubscriptionTier) -> SubscriptionTierLimits:
        """Get limits for a tier."""
        cls._ensure_initialized()
        return cls._limits.get(tier, cls._limits[SubscriptionTier.FREE])

    @classmethod
    def has_feature(cls, tier: SubscriptionTier, feature: str) -> bool:
        """Check if tier has access to feature."""
        limits = cls.get_limits(tier)
        return feature in limits.features

    @classmethod
    def get_rate_limit(cls, tier: SubscriptionTier) -> int:
        """Get rate limit per minute for tier."""
        return cls.get_limits(tier).rate_limit_per_minute


def get_tier_config() -> TierConfig:
    """Get tier config singleton."""
    return TierConfig
