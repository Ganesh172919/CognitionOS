"""
Subscription Tier Management System

Handles subscription tiers, feature gating, and entitlements for SaaS platform.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)


class TierLevel(str, Enum):
    """Subscription tier levels"""
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class FeatureType(str, Enum):
    """Types of features that can be gated"""
    API_ACCESS = "api_access"
    WORKFLOW_EXECUTION = "workflow_execution"
    AGENT_USAGE = "agent_usage"
    MEMORY_STORAGE = "memory_storage"
    PLUGIN_MARKETPLACE = "plugin_marketplace"
    CUSTOM_INTEGRATION = "custom_integration"
    PRIORITY_SUPPORT = "priority_support"
    SSO = "sso"
    ADVANCED_ANALYTICS = "advanced_analytics"
    CUSTOM_BRANDING = "custom_branding"
    API_RATE_LIMIT = "api_rate_limit"
    CONCURRENT_WORKFLOWS = "concurrent_workflows"
    TEAM_COLLABORATION = "team_collaboration"
    AUDIT_LOGS = "audit_logs"
    SLA_GUARANTEE = "sla_guarantee"


@dataclass
class TierFeature:
    """Feature definition with limits and permissions"""
    feature_type: FeatureType
    enabled: bool
    limit: Optional[int] = None  # None means unlimited
    metadata: Dict[str, Any] = field(default_factory=dict)

    def check_usage(self, current_usage: int) -> bool:
        """Check if usage is within limits"""
        if not self.enabled:
            return False
        if self.limit is None:
            return True
        return current_usage < self.limit

    def remaining(self, current_usage: int) -> Optional[int]:
        """Calculate remaining quota"""
        if not self.enabled:
            return 0
        if self.limit is None:
            return None  # Unlimited
        return max(0, self.limit - current_usage)


@dataclass
class SubscriptionTier:
    """Complete subscription tier definition"""
    level: TierLevel
    name: str
    description: str
    price_monthly: float
    price_yearly: float
    features: Dict[FeatureType, TierFeature]

    # Resource limits
    api_calls_per_month: Optional[int] = None
    workflows_per_month: Optional[int] = None
    agent_compute_hours: Optional[int] = None
    storage_gb: Optional[int] = None
    team_members: Optional[int] = None
    concurrent_executions: int = 1

    # Support and SLA
    support_level: str = "community"
    response_time_hours: Optional[int] = None
    uptime_sla_percent: Optional[float] = None

    # Advanced features
    custom_domain: bool = False
    white_label: bool = False
    dedicated_instance: bool = False
    priority_queue: bool = False

    # Billing settings
    trial_days: int = 0
    setup_fee: float = 0.0
    overage_pricing: Dict[str, float] = field(default_factory=dict)

    def has_feature(self, feature_type: FeatureType) -> bool:
        """Check if tier has a specific feature enabled"""
        feature = self.features.get(feature_type)
        return feature is not None and feature.enabled

    def get_feature_limit(self, feature_type: FeatureType) -> Optional[int]:
        """Get limit for a specific feature"""
        feature = self.features.get(feature_type)
        if feature and feature.enabled:
            return feature.limit
        return 0

    def calculate_overage_cost(self, resource: str, overage_amount: int) -> float:
        """Calculate cost for exceeding limits"""
        unit_price = self.overage_pricing.get(resource, 0.0)
        return unit_price * overage_amount

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "level": self.level.value,
            "name": self.name,
            "description": self.description,
            "price_monthly": self.price_monthly,
            "price_yearly": self.price_yearly,
            "features": {
                ft.value: {
                    "enabled": f.enabled,
                    "limit": f.limit,
                    "metadata": f.metadata
                }
                for ft, f in self.features.items()
            },
            "limits": {
                "api_calls_per_month": self.api_calls_per_month,
                "workflows_per_month": self.workflows_per_month,
                "agent_compute_hours": self.agent_compute_hours,
                "storage_gb": self.storage_gb,
                "team_members": self.team_members,
                "concurrent_executions": self.concurrent_executions
            },
            "support": {
                "level": self.support_level,
                "response_time_hours": self.response_time_hours,
                "uptime_sla_percent": self.uptime_sla_percent
            },
            "advanced": {
                "custom_domain": self.custom_domain,
                "white_label": self.white_label,
                "dedicated_instance": self.dedicated_instance,
                "priority_queue": self.priority_queue
            },
            "billing": {
                "trial_days": self.trial_days,
                "setup_fee": self.setup_fee,
                "overage_pricing": self.overage_pricing
            }
        }


class FeatureGate:
    """Feature gating system for controlling access"""

    def __init__(self, subscription_tier: SubscriptionTier):
        self.tier = subscription_tier
        self._cache: Dict[str, Any] = {}

    def check_access(
        self,
        feature_type: FeatureType,
        current_usage: Optional[int] = None
    ) -> bool:
        """
        Check if user has access to a feature

        Args:
            feature_type: Type of feature to check
            current_usage: Current usage count for limited features

        Returns:
            True if access granted, False otherwise
        """
        feature = self.tier.features.get(feature_type)
        if not feature or not feature.enabled:
            return False

        if current_usage is not None and feature.limit is not None:
            return current_usage < feature.limit

        return True

    def get_remaining_quota(
        self,
        feature_type: FeatureType,
        current_usage: int
    ) -> Optional[int]:
        """Get remaining quota for a feature"""
        feature = self.tier.features.get(feature_type)
        if not feature or not feature.enabled:
            return 0
        return feature.remaining(current_usage)

    def require_feature(
        self,
        feature_type: FeatureType,
        error_message: Optional[str] = None
    ):
        """Decorator to gate endpoints by feature"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.check_access(feature_type):
                    msg = error_message or f"Feature {feature_type.value} not available in your plan"
                    raise FeatureNotAvailableError(msg, feature_type)
                return func(*args, **kwargs)
            return wrapper
        return decorator


class FeatureNotAvailableError(Exception):
    """Exception raised when feature is not available in current tier"""

    def __init__(self, message: str, feature_type: FeatureType):
        self.message = message
        self.feature_type = feature_type
        super().__init__(self.message)


class SubscriptionTierManager:
    """Manages subscription tiers and tier-related operations"""

    def __init__(self):
        self.tiers: Dict[TierLevel, SubscriptionTier] = {}
        self._initialize_default_tiers()

    def _initialize_default_tiers(self):
        """Initialize default subscription tiers"""

        # FREE TIER
        free_features = {
            FeatureType.API_ACCESS: TierFeature(FeatureType.API_ACCESS, True, 1000),
            FeatureType.WORKFLOW_EXECUTION: TierFeature(FeatureType.WORKFLOW_EXECUTION, True, 10),
            FeatureType.AGENT_USAGE: TierFeature(FeatureType.AGENT_USAGE, True, 5),
            FeatureType.MEMORY_STORAGE: TierFeature(FeatureType.MEMORY_STORAGE, True, 100),
            FeatureType.PLUGIN_MARKETPLACE: TierFeature(FeatureType.PLUGIN_MARKETPLACE, True, 3),
            FeatureType.API_RATE_LIMIT: TierFeature(FeatureType.API_RATE_LIMIT, True, 10),
            FeatureType.CONCURRENT_WORKFLOWS: TierFeature(FeatureType.CONCURRENT_WORKFLOWS, True, 1),
        }

        self.tiers[TierLevel.FREE] = SubscriptionTier(
            level=TierLevel.FREE,
            name="Free",
            description="Perfect for trying out CognitionOS",
            price_monthly=0.0,
            price_yearly=0.0,
            features=free_features,
            api_calls_per_month=1000,
            workflows_per_month=10,
            agent_compute_hours=5,
            storage_gb=1,
            team_members=1,
            concurrent_executions=1,
            support_level="community",
            trial_days=0
        )

        # STARTER TIER
        starter_features = {
            FeatureType.API_ACCESS: TierFeature(FeatureType.API_ACCESS, True, 10000),
            FeatureType.WORKFLOW_EXECUTION: TierFeature(FeatureType.WORKFLOW_EXECUTION, True, 100),
            FeatureType.AGENT_USAGE: TierFeature(FeatureType.AGENT_USAGE, True, 50),
            FeatureType.MEMORY_STORAGE: TierFeature(FeatureType.MEMORY_STORAGE, True, 1000),
            FeatureType.PLUGIN_MARKETPLACE: TierFeature(FeatureType.PLUGIN_MARKETPLACE, True, 10),
            FeatureType.API_RATE_LIMIT: TierFeature(FeatureType.API_RATE_LIMIT, True, 100),
            FeatureType.CONCURRENT_WORKFLOWS: TierFeature(FeatureType.CONCURRENT_WORKFLOWS, True, 3),
            FeatureType.TEAM_COLLABORATION: TierFeature(FeatureType.TEAM_COLLABORATION, True, 3),
        }

        self.tiers[TierLevel.STARTER] = SubscriptionTier(
            level=TierLevel.STARTER,
            name="Starter",
            description="For small teams and growing projects",
            price_monthly=29.0,
            price_yearly=290.0,
            features=starter_features,
            api_calls_per_month=10000,
            workflows_per_month=100,
            agent_compute_hours=50,
            storage_gb=10,
            team_members=3,
            concurrent_executions=3,
            support_level="email",
            response_time_hours=48,
            trial_days=14,
            overage_pricing={
                "api_calls": 0.001,
                "workflows": 0.50,
                "compute_hours": 1.00
            }
        )

        # PRO TIER
        pro_features = {
            FeatureType.API_ACCESS: TierFeature(FeatureType.API_ACCESS, True, 100000),
            FeatureType.WORKFLOW_EXECUTION: TierFeature(FeatureType.WORKFLOW_EXECUTION, True, 1000),
            FeatureType.AGENT_USAGE: TierFeature(FeatureType.AGENT_USAGE, True, 500),
            FeatureType.MEMORY_STORAGE: TierFeature(FeatureType.MEMORY_STORAGE, True, None),
            FeatureType.PLUGIN_MARKETPLACE: TierFeature(FeatureType.PLUGIN_MARKETPLACE, True, None),
            FeatureType.CUSTOM_INTEGRATION: TierFeature(FeatureType.CUSTOM_INTEGRATION, True),
            FeatureType.PRIORITY_SUPPORT: TierFeature(FeatureType.PRIORITY_SUPPORT, True),
            FeatureType.ADVANCED_ANALYTICS: TierFeature(FeatureType.ADVANCED_ANALYTICS, True),
            FeatureType.API_RATE_LIMIT: TierFeature(FeatureType.API_RATE_LIMIT, True, 1000),
            FeatureType.CONCURRENT_WORKFLOWS: TierFeature(FeatureType.CONCURRENT_WORKFLOWS, True, 10),
            FeatureType.TEAM_COLLABORATION: TierFeature(FeatureType.TEAM_COLLABORATION, True, 10),
            FeatureType.AUDIT_LOGS: TierFeature(FeatureType.AUDIT_LOGS, True),
        }

        self.tiers[TierLevel.PRO] = SubscriptionTier(
            level=TierLevel.PRO,
            name="Pro",
            description="For professional teams and businesses",
            price_monthly=99.0,
            price_yearly=990.0,
            features=pro_features,
            api_calls_per_month=100000,
            workflows_per_month=1000,
            agent_compute_hours=500,
            storage_gb=100,
            team_members=10,
            concurrent_executions=10,
            support_level="priority",
            response_time_hours=12,
            uptime_sla_percent=99.5,
            priority_queue=True,
            trial_days=14,
            overage_pricing={
                "api_calls": 0.0008,
                "workflows": 0.30,
                "compute_hours": 0.75
            }
        )

        # ENTERPRISE TIER
        enterprise_features = {
            FeatureType.API_ACCESS: TierFeature(FeatureType.API_ACCESS, True, None),
            FeatureType.WORKFLOW_EXECUTION: TierFeature(FeatureType.WORKFLOW_EXECUTION, True, None),
            FeatureType.AGENT_USAGE: TierFeature(FeatureType.AGENT_USAGE, True, None),
            FeatureType.MEMORY_STORAGE: TierFeature(FeatureType.MEMORY_STORAGE, True, None),
            FeatureType.PLUGIN_MARKETPLACE: TierFeature(FeatureType.PLUGIN_MARKETPLACE, True, None),
            FeatureType.CUSTOM_INTEGRATION: TierFeature(FeatureType.CUSTOM_INTEGRATION, True),
            FeatureType.PRIORITY_SUPPORT: TierFeature(FeatureType.PRIORITY_SUPPORT, True),
            FeatureType.SSO: TierFeature(FeatureType.SSO, True),
            FeatureType.ADVANCED_ANALYTICS: TierFeature(FeatureType.ADVANCED_ANALYTICS, True),
            FeatureType.CUSTOM_BRANDING: TierFeature(FeatureType.CUSTOM_BRANDING, True),
            FeatureType.API_RATE_LIMIT: TierFeature(FeatureType.API_RATE_LIMIT, True, None),
            FeatureType.CONCURRENT_WORKFLOWS: TierFeature(FeatureType.CONCURRENT_WORKFLOWS, True, None),
            FeatureType.TEAM_COLLABORATION: TierFeature(FeatureType.TEAM_COLLABORATION, True, None),
            FeatureType.AUDIT_LOGS: TierFeature(FeatureType.AUDIT_LOGS, True),
            FeatureType.SLA_GUARANTEE: TierFeature(FeatureType.SLA_GUARANTEE, True),
        }

        self.tiers[TierLevel.ENTERPRISE] = SubscriptionTier(
            level=TierLevel.ENTERPRISE,
            name="Enterprise",
            description="For large organizations with custom needs",
            price_monthly=499.0,
            price_yearly=4990.0,
            features=enterprise_features,
            api_calls_per_month=None,
            workflows_per_month=None,
            agent_compute_hours=None,
            storage_gb=None,
            team_members=None,
            concurrent_executions=50,
            support_level="dedicated",
            response_time_hours=1,
            uptime_sla_percent=99.99,
            custom_domain=True,
            white_label=True,
            dedicated_instance=True,
            priority_queue=True,
            setup_fee=1000.0,
            trial_days=30
        )

    def get_tier(self, level: TierLevel) -> Optional[SubscriptionTier]:
        """Get tier by level"""
        return self.tiers.get(level)

    def list_tiers(self) -> List[SubscriptionTier]:
        """List all available tiers"""
        return list(self.tiers.values())

    def compare_tiers(self, tier1: TierLevel, tier2: TierLevel) -> Dict[str, Any]:
        """Compare two tiers"""
        t1 = self.get_tier(tier1)
        t2 = self.get_tier(tier2)

        if not t1 or not t2:
            return {}

        comparison = {
            "tier1": t1.to_dict(),
            "tier2": t2.to_dict(),
            "differences": {
                "price_monthly": t2.price_monthly - t1.price_monthly,
                "price_yearly": t2.price_yearly - t1.price_yearly,
                "features": []
            }
        }

        # Find feature differences
        all_features = set(t1.features.keys()) | set(t2.features.keys())
        for feature_type in all_features:
            f1 = t1.features.get(feature_type)
            f2 = t2.features.get(feature_type)

            if (not f1 or not f1.enabled) and (f2 and f2.enabled):
                comparison["differences"]["features"].append({
                    "feature": feature_type.value,
                    "status": "new_in_tier2"
                })
            elif (f1 and f1.enabled) and (not f2 or not f2.enabled):
                comparison["differences"]["features"].append({
                    "feature": feature_type.value,
                    "status": "removed_in_tier2"
                })
            elif f1 and f2 and f1.limit != f2.limit:
                comparison["differences"]["features"].append({
                    "feature": feature_type.value,
                    "tier1_limit": f1.limit,
                    "tier2_limit": f2.limit
                })

        return comparison

    def recommend_tier(
        self,
        monthly_api_calls: int,
        monthly_workflows: int,
        team_size: int,
        needs_sso: bool = False,
        needs_custom_branding: bool = False
    ) -> TierLevel:
        """Recommend appropriate tier based on usage patterns"""

        # Enterprise requirements
        if needs_sso or needs_custom_branding or team_size > 10:
            return TierLevel.ENTERPRISE

        # Pro requirements
        if (monthly_api_calls > 10000 or
            monthly_workflows > 100 or
            team_size > 3):
            return TierLevel.PRO

        # Starter requirements
        if monthly_api_calls > 1000 or monthly_workflows > 10:
            return TierLevel.STARTER

        # Default to free
        return TierLevel.FREE

    def create_feature_gate(self, tier_level: TierLevel) -> FeatureGate:
        """Create a feature gate for a tier"""
        tier = self.get_tier(tier_level)
        if not tier:
            raise ValueError(f"Invalid tier level: {tier_level}")
        return FeatureGate(tier)
