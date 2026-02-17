"""Tenant domain entities for multi-tenancy."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from uuid import UUID, uuid4


class TenantStatus(str, Enum):
    """Tenant status enumeration."""
    
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    CHURNED = "churned"
    PENDING = "pending"


@dataclass
class TenantSettings:
    """Tenant-specific configuration settings."""
    
    max_users: int = 5
    max_agents: int = 10
    max_workflows: int = 50
    max_executions_per_month: int = 1000
    max_storage_gb: int = 10
    api_rate_limit_per_minute: int = 60
    enable_plugins: bool = False
    enable_custom_models: bool = False
    enable_priority_execution: bool = False
    custom_domain: Optional[str] = None
    webhook_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tenant:
    """
    Tenant entity representing an isolated customer environment.
    
    This is the root entity for multi-tenancy, ensuring complete data isolation
    across different customers. All domain entities should reference tenant_id.
    """
    
    id: UUID
    name: str
    slug: str  # URL-friendly unique identifier
    status: TenantStatus
    settings: TenantSettings
    subscription_tier: str  # 'free', 'pro', 'team', 'enterprise'
    created_at: datetime
    updated_at: datetime
    trial_ends_at: Optional[datetime] = None
    suspended_at: Optional[datetime] = None
    suspended_reason: Optional[str] = None
    owner_user_id: Optional[UUID] = None
    billing_email: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        name: str,
        slug: str,
        owner_user_id: UUID,
        billing_email: str,
        subscription_tier: str = "free",
    ) -> "Tenant":
        """Create a new tenant with default settings."""
        now = datetime.utcnow()
        
        # Default settings based on tier
        settings = cls._default_settings_for_tier(subscription_tier)
        
        return cls(
            id=uuid4(),
            name=name,
            slug=slug,
            status=TenantStatus.ACTIVE,
            settings=settings,
            subscription_tier=subscription_tier,
            created_at=now,
            updated_at=now,
            owner_user_id=owner_user_id,
            billing_email=billing_email,
        )
    
    @staticmethod
    def _default_settings_for_tier(tier: str) -> TenantSettings:
        """Get default settings based on subscription tier."""
        settings_map = {
            "free": TenantSettings(
                max_users=5,
                max_agents=10,
                max_workflows=50,
                max_executions_per_month=1000,
                max_storage_gb=1,
                api_rate_limit_per_minute=60,
                enable_plugins=False,
                enable_custom_models=False,
                enable_priority_execution=False,
            ),
            "pro": TenantSettings(
                max_users=10,
                max_agents=50,
                max_workflows=500,
                max_executions_per_month=50000,
                max_storage_gb=50,
                api_rate_limit_per_minute=300,
                enable_plugins=True,
                enable_custom_models=False,
                enable_priority_execution=True,
            ),
            "team": TenantSettings(
                max_users=50,
                max_agents=200,
                max_workflows=2000,
                max_executions_per_month=500000,
                max_storage_gb=200,
                api_rate_limit_per_minute=1000,
                enable_plugins=True,
                enable_custom_models=True,
                enable_priority_execution=True,
            ),
            "enterprise": TenantSettings(
                max_users=999999,
                max_agents=999999,
                max_workflows=999999,
                max_executions_per_month=999999999,
                max_storage_gb=999999,
                api_rate_limit_per_minute=10000,
                enable_plugins=True,
                enable_custom_models=True,
                enable_priority_execution=True,
            ),
        }
        return settings_map.get(tier, settings_map["free"])
    
    def suspend(self, reason: str) -> None:
        """Suspend the tenant."""
        self.status = TenantStatus.SUSPENDED
        self.suspended_at = datetime.utcnow()
        self.suspended_reason = reason
        self.updated_at = datetime.utcnow()
    
    def reactivate(self) -> None:
        """Reactivate a suspended tenant."""
        self.status = TenantStatus.ACTIVE
        self.suspended_at = None
        self.suspended_reason = None
        self.updated_at = datetime.utcnow()
    
    def upgrade_tier(self, new_tier: str) -> None:
        """Upgrade subscription tier and update settings."""
        self.subscription_tier = new_tier
        self.settings = self._default_settings_for_tier(new_tier)
        self.updated_at = datetime.utcnow()
    
    def downgrade_tier(self, new_tier: str) -> None:
        """Downgrade subscription tier and update settings."""
        self.subscription_tier = new_tier
        self.settings = self._default_settings_for_tier(new_tier)
        self.updated_at = datetime.utcnow()
    
    def update_settings(self, **kwargs) -> None:
        """Update tenant settings."""
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        self.updated_at = datetime.utcnow()
    
    def is_active(self) -> bool:
        """Check if tenant is active."""
        return self.status == TenantStatus.ACTIVE
    
    def is_trial(self) -> bool:
        """Check if tenant is in trial."""
        return self.status == TenantStatus.TRIAL
    
    def is_suspended(self) -> bool:
        """Check if tenant is suspended."""
        return self.status == TenantStatus.SUSPENDED
    
    def trial_expired(self) -> bool:
        """Check if trial has expired."""
        if self.trial_ends_at is None:
            return False
        return datetime.utcnow() > self.trial_ends_at
