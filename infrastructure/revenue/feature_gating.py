"""
Dynamic Feature Gating System
Controls feature access based on subscription tiers with real-time enforcement.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field


class SubscriptionTier(str, Enum):
    """Subscription tier levels"""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class FeatureCategory(str, Enum):
    """Feature categories"""
    CORE = "core"
    ADVANCED = "advanced"
    PREMIUM = "premium"
    ENTERPRISE_ONLY = "enterprise_only"


class QuotaType(str, Enum):
    """Types of quotas"""
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_DAY = "requests_per_day"
    STORAGE_GB = "storage_gb"
    TEAM_MEMBERS = "team_members"
    WORKFLOWS = "workflows"
    API_CALLS = "api_calls"
    LLM_TOKENS_PER_MONTH = "llm_tokens_per_month"
    CONCURRENT_EXECUTIONS = "concurrent_executions"


class Feature(BaseModel):
    """Feature definition"""
    feature_id: str
    name: str
    description: str
    category: FeatureCategory
    required_tier: SubscriptionTier
    enabled_by_default: bool = True
    beta: bool = False
    deprecated: bool = False


class Quota(BaseModel):
    """Quota limit"""
    quota_type: QuotaType
    limit: int
    current_usage: int = 0
    reset_period: str = "monthly"  # daily, monthly, never


class TierConfiguration(BaseModel):
    """Configuration for a subscription tier"""
    tier: SubscriptionTier
    name: str
    description: str
    monthly_price: float
    annual_price: float
    features: List[str]  # Feature IDs
    quotas: Dict[QuotaType, Quota]
    support_level: str  # email, priority, dedicated
    sla_uptime: float  # 99.0, 99.9, 99.99
    custom_branding: bool = False
    api_rate_limit: int = 1000  # requests per minute


class FeatureGateResult(BaseModel):
    """Result of feature gate check"""
    allowed: bool
    reason: Optional[str] = None
    upgrade_required: Optional[SubscriptionTier] = None
    quota_exceeded: Optional[QuotaType] = None


class TenantSubscription(BaseModel):
    """Tenant subscription status"""
    tenant_id: str
    tier: SubscriptionTier
    custom_features: Set[str] = Field(default_factory=set)
    blocked_features: Set[str] = Field(default_factory=set)
    quotas: Dict[QuotaType, Quota]
    trial_end: Optional[datetime] = None
    is_trial: bool = False
    payment_status: str = "active"  # active, past_due, canceled
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DynamicFeatureGate:
    """
    Dynamic feature gating system with real-time tier enforcement.
    """

    def __init__(self):
        self.features: Dict[str, Feature] = {}
        self.tier_configs: Dict[SubscriptionTier, TierConfiguration] = {}
        self.tenant_subscriptions: Dict[str, TenantSubscription] = {}
        self._initialize_features()
        self._initialize_tier_configs()

    def _initialize_features(self):
        """Initialize feature definitions"""

        # Core features (Free tier)
        self.features["basic_workflows"] = Feature(
            feature_id="basic_workflows",
            name="Basic Workflows",
            description="Create and execute basic workflows",
            category=FeatureCategory.CORE,
            required_tier=SubscriptionTier.FREE
        )

        self.features["api_access"] = Feature(
            feature_id="api_access",
            name="API Access",
            description="Access to REST API",
            category=FeatureCategory.CORE,
            required_tier=SubscriptionTier.FREE
        )

        # Advanced features (Starter tier)
        self.features["advanced_workflows"] = Feature(
            feature_id="advanced_workflows",
            name="Advanced Workflows",
            description="Complex workflows with conditional logic",
            category=FeatureCategory.ADVANCED,
            required_tier=SubscriptionTier.STARTER
        )

        self.features["team_collaboration"] = Feature(
            feature_id="team_collaboration",
            name="Team Collaboration",
            description="Collaborate with team members",
            category=FeatureCategory.ADVANCED,
            required_tier=SubscriptionTier.STARTER
        )

        self.features["webhook_integration"] = Feature(
            feature_id="webhook_integration",
            name="Webhook Integration",
            description="Integrate with webhooks",
            category=FeatureCategory.ADVANCED,
            required_tier=SubscriptionTier.STARTER
        )

        # Premium features (Professional tier)
        self.features["custom_agents"] = Feature(
            feature_id="custom_agents",
            name="Custom AI Agents",
            description="Create custom AI agents",
            category=FeatureCategory.PREMIUM,
            required_tier=SubscriptionTier.PROFESSIONAL
        )

        self.features["priority_execution"] = Feature(
            feature_id="priority_execution",
            name="Priority Execution",
            description="Execute workflows with priority",
            category=FeatureCategory.PREMIUM,
            required_tier=SubscriptionTier.PROFESSIONAL
        )

        self.features["advanced_analytics"] = Feature(
            feature_id="advanced_analytics",
            name="Advanced Analytics",
            description="Detailed analytics and insights",
            category=FeatureCategory.PREMIUM,
            required_tier=SubscriptionTier.PROFESSIONAL
        )

        self.features["audit_logs"] = Feature(
            feature_id="audit_logs",
            name="Audit Logs",
            description="Comprehensive audit logging",
            category=FeatureCategory.PREMIUM,
            required_tier=SubscriptionTier.PROFESSIONAL
        )

        # Enterprise features
        self.features["sso_integration"] = Feature(
            feature_id="sso_integration",
            name="SSO Integration",
            description="Single Sign-On integration",
            category=FeatureCategory.ENTERPRISE_ONLY,
            required_tier=SubscriptionTier.ENTERPRISE
        )

        self.features["dedicated_infrastructure"] = Feature(
            feature_id="dedicated_infrastructure",
            name="Dedicated Infrastructure",
            description="Dedicated compute resources",
            category=FeatureCategory.ENTERPRISE_ONLY,
            required_tier=SubscriptionTier.ENTERPRISE
        )

        self.features["custom_sla"] = Feature(
            feature_id="custom_sla",
            name="Custom SLA",
            description="Custom Service Level Agreement",
            category=FeatureCategory.ENTERPRISE_ONLY,
            required_tier=SubscriptionTier.ENTERPRISE
        )

        self.features["on_premise_deployment"] = Feature(
            feature_id="on_premise_deployment",
            name="On-Premise Deployment",
            description="Deploy on your infrastructure",
            category=FeatureCategory.ENTERPRISE_ONLY,
            required_tier=SubscriptionTier.ENTERPRISE
        )

    def _initialize_tier_configs(self):
        """Initialize tier configurations"""

        # Free Tier
        self.tier_configs[SubscriptionTier.FREE] = TierConfiguration(
            tier=SubscriptionTier.FREE,
            name="Free",
            description="Get started with basic features",
            monthly_price=0.0,
            annual_price=0.0,
            features=[
                "basic_workflows",
                "api_access"
            ],
            quotas={
                QuotaType.WORKFLOWS: Quota(quota_type=QuotaType.WORKFLOWS, limit=5),
                QuotaType.API_CALLS: Quota(quota_type=QuotaType.API_CALLS, limit=1000),
                QuotaType.LLM_TOKENS_PER_MONTH: Quota(
                    quota_type=QuotaType.LLM_TOKENS_PER_MONTH,
                    limit=100000
                ),
                QuotaType.STORAGE_GB: Quota(quota_type=QuotaType.STORAGE_GB, limit=1),
                QuotaType.TEAM_MEMBERS: Quota(quota_type=QuotaType.TEAM_MEMBERS, limit=1),
                QuotaType.CONCURRENT_EXECUTIONS: Quota(
                    quota_type=QuotaType.CONCURRENT_EXECUTIONS,
                    limit=1
                )
            },
            support_level="community",
            sla_uptime=99.0,
            api_rate_limit=10
        )

        # Starter Tier
        self.tier_configs[SubscriptionTier.STARTER] = TierConfiguration(
            tier=SubscriptionTier.STARTER,
            name="Starter",
            description="For growing teams",
            monthly_price=29.0,
            annual_price=290.0,  # 2 months free
            features=[
                "basic_workflows",
                "api_access",
                "advanced_workflows",
                "team_collaboration",
                "webhook_integration"
            ],
            quotas={
                QuotaType.WORKFLOWS: Quota(quota_type=QuotaType.WORKFLOWS, limit=50),
                QuotaType.API_CALLS: Quota(quota_type=QuotaType.API_CALLS, limit=50000),
                QuotaType.LLM_TOKENS_PER_MONTH: Quota(
                    quota_type=QuotaType.LLM_TOKENS_PER_MONTH,
                    limit=1000000
                ),
                QuotaType.STORAGE_GB: Quota(quota_type=QuotaType.STORAGE_GB, limit=10),
                QuotaType.TEAM_MEMBERS: Quota(quota_type=QuotaType.TEAM_MEMBERS, limit=5),
                QuotaType.CONCURRENT_EXECUTIONS: Quota(
                    quota_type=QuotaType.CONCURRENT_EXECUTIONS,
                    limit=5
                )
            },
            support_level="email",
            sla_uptime=99.5,
            api_rate_limit=100
        )

        # Professional Tier
        self.tier_configs[SubscriptionTier.PROFESSIONAL] = TierConfiguration(
            tier=SubscriptionTier.PROFESSIONAL,
            name="Professional",
            description="For professional teams",
            monthly_price=99.0,
            annual_price=990.0,  # 2 months free
            features=[
                "basic_workflows",
                "api_access",
                "advanced_workflows",
                "team_collaboration",
                "webhook_integration",
                "custom_agents",
                "priority_execution",
                "advanced_analytics",
                "audit_logs"
            ],
            quotas={
                QuotaType.WORKFLOWS: Quota(quota_type=QuotaType.WORKFLOWS, limit=500),
                QuotaType.API_CALLS: Quota(quota_type=QuotaType.API_CALLS, limit=500000),
                QuotaType.LLM_TOKENS_PER_MONTH: Quota(
                    quota_type=QuotaType.LLM_TOKENS_PER_MONTH,
                    limit=10000000
                ),
                QuotaType.STORAGE_GB: Quota(quota_type=QuotaType.STORAGE_GB, limit=100),
                QuotaType.TEAM_MEMBERS: Quota(quota_type=QuotaType.TEAM_MEMBERS, limit=25),
                QuotaType.CONCURRENT_EXECUTIONS: Quota(
                    quota_type=QuotaType.CONCURRENT_EXECUTIONS,
                    limit=20
                )
            },
            support_level="priority",
            sla_uptime=99.9,
            api_rate_limit=1000,
            custom_branding=True
        )

        # Enterprise Tier
        self.tier_configs[SubscriptionTier.ENTERPRISE] = TierConfiguration(
            tier=SubscriptionTier.ENTERPRISE,
            name="Enterprise",
            description="For large organizations",
            monthly_price=499.0,
            annual_price=4990.0,  # 2 months free
            features=list(self.features.keys()),  # All features
            quotas={
                QuotaType.WORKFLOWS: Quota(quota_type=QuotaType.WORKFLOWS, limit=999999),
                QuotaType.API_CALLS: Quota(quota_type=QuotaType.API_CALLS, limit=9999999),
                QuotaType.LLM_TOKENS_PER_MONTH: Quota(
                    quota_type=QuotaType.LLM_TOKENS_PER_MONTH,
                    limit=100000000
                ),
                QuotaType.STORAGE_GB: Quota(quota_type=QuotaType.STORAGE_GB, limit=1000),
                QuotaType.TEAM_MEMBERS: Quota(quota_type=QuotaType.TEAM_MEMBERS, limit=999999),
                QuotaType.CONCURRENT_EXECUTIONS: Quota(
                    quota_type=QuotaType.CONCURRENT_EXECUTIONS,
                    limit=100
                )
            },
            support_level="dedicated",
            sla_uptime=99.99,
            api_rate_limit=10000,
            custom_branding=True
        )

    async def check_feature_access(
        self,
        tenant_id: str,
        feature_id: str
    ) -> FeatureGateResult:
        """
        Check if tenant has access to a feature
        """
        # Get tenant subscription
        subscription = self.tenant_subscriptions.get(tenant_id)

        if not subscription:
            return FeatureGateResult(
                allowed=False,
                reason="No active subscription found",
                upgrade_required=SubscriptionTier.STARTER
            )

        # Check payment status
        if subscription.payment_status != "active":
            return FeatureGateResult(
                allowed=False,
                reason=f"Payment status: {subscription.payment_status}"
            )

        # Check if feature exists
        if feature_id not in self.features:
            return FeatureGateResult(
                allowed=False,
                reason="Feature not found"
            )

        feature = self.features[feature_id]

        # Check if feature is blocked for this tenant
        if feature_id in subscription.blocked_features:
            return FeatureGateResult(
                allowed=False,
                reason="Feature is blocked for this tenant"
            )

        # Check if feature is explicitly enabled for this tenant
        if feature_id in subscription.custom_features:
            return FeatureGateResult(allowed=True)

        # Check tier requirement
        tier_order = [
            SubscriptionTier.FREE,
            SubscriptionTier.STARTER,
            SubscriptionTier.PROFESSIONAL,
            SubscriptionTier.ENTERPRISE
        ]

        current_tier_index = tier_order.index(subscription.tier)
        required_tier_index = tier_order.index(feature.required_tier)

        if current_tier_index < required_tier_index:
            return FeatureGateResult(
                allowed=False,
                reason=f"Feature requires {feature.required_tier.value} tier or higher",
                upgrade_required=feature.required_tier
            )

        return FeatureGateResult(allowed=True)

    async def check_quota(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        requested_amount: int = 1
    ) -> FeatureGateResult:
        """
        Check if tenant has quota available
        """
        subscription = self.tenant_subscriptions.get(tenant_id)

        if not subscription:
            return FeatureGateResult(
                allowed=False,
                reason="No active subscription"
            )

        if quota_type not in subscription.quotas:
            return FeatureGateResult(allowed=True)  # No limit set

        quota = subscription.quotas[quota_type]

        if quota.current_usage + requested_amount > quota.limit:
            return FeatureGateResult(
                allowed=False,
                reason=f"Quota exceeded: {quota.current_usage}/{quota.limit}",
                quota_exceeded=quota_type
            )

        return FeatureGateResult(allowed=True)

    async def consume_quota(
        self,
        tenant_id: str,
        quota_type: QuotaType,
        amount: int = 1
    ) -> bool:
        """
        Consume quota for a tenant
        """
        subscription = self.tenant_subscriptions.get(tenant_id)

        if not subscription or quota_type not in subscription.quotas:
            return False

        quota = subscription.quotas[quota_type]
        quota.current_usage += amount

        return True

    async def reset_quota(
        self,
        tenant_id: str,
        quota_type: QuotaType
    ) -> bool:
        """
        Reset quota (called periodically)
        """
        subscription = self.tenant_subscriptions.get(tenant_id)

        if not subscription or quota_type not in subscription.quotas:
            return False

        subscription.quotas[quota_type].current_usage = 0
        return True

    def create_subscription(
        self,
        tenant_id: str,
        tier: SubscriptionTier,
        is_trial: bool = False,
        trial_days: int = 14
    ) -> TenantSubscription:
        """
        Create a new subscription for tenant
        """
        tier_config = self.tier_configs[tier]

        # Copy quotas
        quotas = {
            qt: Quota(
                quota_type=qt,
                limit=q.limit,
                reset_period=q.reset_period
            )
            for qt, q in tier_config.quotas.items()
        }

        subscription = TenantSubscription(
            tenant_id=tenant_id,
            tier=tier,
            quotas=quotas,
            is_trial=is_trial
        )

        if is_trial:
            from datetime import timedelta
            subscription.trial_end = datetime.utcnow() + timedelta(days=trial_days)

        self.tenant_subscriptions[tenant_id] = subscription

        return subscription

    def upgrade_subscription(
        self,
        tenant_id: str,
        new_tier: SubscriptionTier
    ) -> Optional[TenantSubscription]:
        """
        Upgrade subscription to new tier
        """
        subscription = self.tenant_subscriptions.get(tenant_id)

        if not subscription:
            return None

        tier_order = [
            SubscriptionTier.FREE,
            SubscriptionTier.STARTER,
            SubscriptionTier.PROFESSIONAL,
            SubscriptionTier.ENTERPRISE
        ]

        current_index = tier_order.index(subscription.tier)
        new_index = tier_order.index(new_tier)

        if new_index <= current_index:
            return None  # Not an upgrade

        # Update tier
        subscription.tier = new_tier

        # Update quotas
        new_tier_config = self.tier_configs[new_tier]
        subscription.quotas = {
            qt: Quota(
                quota_type=qt,
                limit=q.limit,
                reset_period=q.reset_period
            )
            for qt, q in new_tier_config.quotas.items()
        }

        subscription.updated_at = datetime.utcnow()

        return subscription

    def get_available_features(
        self,
        tenant_id: str
    ) -> List[Feature]:
        """
        Get list of features available to tenant
        """
        subscription = self.tenant_subscriptions.get(tenant_id)

        if not subscription:
            return []

        tier_config = self.tier_configs[subscription.tier]

        available = []
        for feature_id in tier_config.features:
            if feature_id in self.features:
                available.append(self.features[feature_id])

        # Add custom enabled features
        for feature_id in subscription.custom_features:
            if feature_id in self.features and feature_id not in tier_config.features:
                available.append(self.features[feature_id])

        return available

    def get_tier_comparison(self) -> Dict[str, Any]:
        """
        Get comparison of all tiers for marketing
        """
        comparison = {}

        for tier, config in self.tier_configs.items():
            comparison[tier.value] = {
                "name": config.name,
                "description": config.description,
                "monthly_price": config.monthly_price,
                "annual_price": config.annual_price,
                "features_count": len(config.features),
                "quotas": {
                    qt.value: q.limit
                    for qt, q in config.quotas.items()
                },
                "support_level": config.support_level,
                "sla_uptime": config.sla_uptime,
                "custom_branding": config.custom_branding
            }

        return comparison
