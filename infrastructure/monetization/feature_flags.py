"""
Feature Flag System with Tier-Based Access Control

Dynamic feature flags for A/B testing, gradual rollout, and tier-based access.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import logging
import hashlib

logger = logging.getLogger(__name__)


class FlagType(str, Enum):
    """Types of feature flags"""
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    TIER_BASED = "tier_based"
    USER_SEGMENT = "user_segment"


class RolloutStrategy(str, Enum):
    """Feature rollout strategies"""
    ALL_AT_ONCE = "all_at_once"
    GRADUAL = "gradual"
    CANARY = "canary"
    AB_TEST = "ab_test"


@dataclass
class FeatureFlag:
    """Feature flag configuration"""
    key: str
    name: str
    description: str
    flag_type: FlagType
    enabled: bool = False

    # Rollout configuration
    rollout_percentage: int = 0  # 0-100
    rollout_strategy: RolloutStrategy = RolloutStrategy.ALL_AT_ONCE

    # Tier-based access
    enabled_tiers: Set[str] = field(default_factory=set)
    disabled_tiers: Set[str] = field(default_factory=set)

    # User segment targeting
    enabled_users: Set[str] = field(default_factory=set)
    enabled_segments: Set[str] = field(default_factory=set)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None

    # Analytics
    evaluation_count: int = 0
    true_count: int = 0

    def evaluate(
        self,
        user_id: Optional[str] = None,
        tier: Optional[str] = None,
        segment: Optional[str] = None
    ) -> bool:
        """
        Evaluate feature flag

        Args:
            user_id: User identifier
            tier: Subscription tier
            segment: User segment

        Returns:
            Whether feature is enabled
        """
        self.evaluation_count += 1

        if not self.enabled:
            return False

        # Check tier-based access
        if self.flag_type == FlagType.TIER_BASED:
            if tier:
                if tier in self.disabled_tiers:
                    return False
                if self.enabled_tiers and tier not in self.enabled_tiers:
                    return False

        # Check user targeting
        if user_id:
            if user_id in self.enabled_users:
                self.true_count += 1
                return True

        # Check segment targeting
        if segment and segment in self.enabled_segments:
            self.true_count += 1
            return True

        # Percentage rollout
        if self.flag_type == FlagType.PERCENTAGE:
            if user_id:
                # Consistent hashing for stable rollout
                hash_value = int(hashlib.md5(f"{self.key}:{user_id}".encode()).hexdigest(), 16)
                percentage = hash_value % 100
                if percentage < self.rollout_percentage:
                    self.true_count += 1
                    return True
            return False

        # Boolean flag
        if self.flag_type == FlagType.BOOLEAN:
            if self.enabled:
                self.true_count += 1
                return True

        return False


class TierBasedFlags:
    """Pre-configured tier-based feature flags"""

    @staticmethod
    def create_standard_flags() -> List[FeatureFlag]:
        """Create standard tier-based flags"""
        return [
            FeatureFlag(
                key="advanced_analytics",
                name="Advanced Analytics",
                description="Access to advanced analytics dashboard",
                flag_type=FlagType.TIER_BASED,
                enabled=True,
                enabled_tiers={"pro", "enterprise"}
            ),
            FeatureFlag(
                key="custom_integrations",
                name="Custom Integrations",
                description="Build custom integrations",
                flag_type=FlagType.TIER_BASED,
                enabled=True,
                enabled_tiers={"pro", "enterprise"}
            ),
            FeatureFlag(
                key="sso",
                name="Single Sign-On",
                description="SSO authentication",
                flag_type=FlagType.TIER_BASED,
                enabled=True,
                enabled_tiers={"enterprise"}
            ),
            FeatureFlag(
                key="white_label",
                name="White Label",
                description="Custom branding and white labeling",
                flag_type=FlagType.TIER_BASED,
                enabled=True,
                enabled_tiers={"enterprise"}
            ),
            FeatureFlag(
                key="priority_support",
                name="Priority Support",
                description="24/7 priority support",
                flag_type=FlagType.TIER_BASED,
                enabled=True,
                enabled_tiers={"pro", "enterprise"}
            ),
            FeatureFlag(
                key="api_v2",
                name="API v2 Access",
                description="Access to new API version",
                flag_type=FlagType.TIER_BASED,
                enabled=True,
                enabled_tiers={"starter", "pro", "enterprise"}
            ),
            FeatureFlag(
                key="batch_operations",
                name="Batch Operations",
                description="Bulk API operations",
                flag_type=FlagType.TIER_BASED,
                enabled=True,
                enabled_tiers={"pro", "enterprise"}
            ),
            FeatureFlag(
                key="export_data",
                name="Data Export",
                description="Export all data in multiple formats",
                flag_type=FlagType.TIER_BASED,
                enabled=True,
                enabled_tiers={"starter", "pro", "enterprise"}
            ),
            FeatureFlag(
                key="audit_logs",
                name="Audit Logs",
                description="Detailed audit logging",
                flag_type=FlagType.TIER_BASED,
                enabled=True,
                enabled_tiers={"pro", "enterprise"}
            ),
            FeatureFlag(
                key="dedicated_instance",
                name="Dedicated Instance",
                description="Private dedicated infrastructure",
                flag_type=FlagType.TIER_BASED,
                enabled=True,
                enabled_tiers={"enterprise"}
            )
        ]


class FeatureFlagManager:
    """
    Feature flag management system

    Manages feature flags with tier-based access, A/B testing,
    and gradual rollout capabilities.
    """

    def __init__(self, storage_backend: Optional[Any] = None):
        self.storage = storage_backend
        self._flags: Dict[str, FeatureFlag] = {}
        self._initialize_standard_flags()

    def _initialize_standard_flags(self):
        """Initialize standard tier-based flags"""
        for flag in TierBasedFlags.create_standard_flags():
            self._flags[flag.key] = flag

    async def create_flag(
        self,
        key: str,
        name: str,
        description: str,
        flag_type: FlagType = FlagType.BOOLEAN,
        created_by: Optional[str] = None
    ) -> FeatureFlag:
        """Create new feature flag"""
        flag = FeatureFlag(
            key=key,
            name=name,
            description=description,
            flag_type=flag_type,
            created_by=created_by
        )

        self._flags[key] = flag

        if self.storage:
            await self.storage.save_flag(flag)

        logger.info(f"Created feature flag: {key}")
        return flag

    async def update_flag(
        self,
        key: str,
        **kwargs
    ) -> Optional[FeatureFlag]:
        """Update feature flag"""
        flag = self._flags.get(key)
        if not flag:
            return None

        for attr, value in kwargs.items():
            if hasattr(flag, attr):
                setattr(flag, attr, value)

        flag.updated_at = datetime.utcnow()

        if self.storage:
            await self.storage.update_flag(flag)

        return flag

    def is_enabled(
        self,
        key: str,
        user_id: Optional[str] = None,
        tier: Optional[str] = None,
        segment: Optional[str] = None
    ) -> bool:
        """
        Check if feature is enabled

        Args:
            key: Feature flag key
            user_id: User identifier
            tier: Subscription tier
            segment: User segment

        Returns:
            Whether feature is enabled
        """
        flag = self._flags.get(key)
        if not flag:
            logger.warning(f"Feature flag not found: {key}")
            return False

        return flag.evaluate(user_id=user_id, tier=tier, segment=segment)

    async def enable_flag(self, key: str):
        """Enable feature flag"""
        await self.update_flag(key, enabled=True)

    async def disable_flag(self, key: str):
        """Disable feature flag"""
        await self.update_flag(key, enabled=False)

    async def set_rollout_percentage(self, key: str, percentage: int):
        """Set rollout percentage"""
        if not 0 <= percentage <= 100:
            raise ValueError("Percentage must be between 0 and 100")

        await self.update_flag(
            key,
            rollout_percentage=percentage,
            flag_type=FlagType.PERCENTAGE
        )

    async def add_tier_access(self, key: str, tier: str):
        """Add tier to enabled tiers"""
        flag = self._flags.get(key)
        if flag:
            flag.enabled_tiers.add(tier)
            flag.updated_at = datetime.utcnow()

            if self.storage:
                await self.storage.update_flag(flag)

    async def remove_tier_access(self, key: str, tier: str):
        """Remove tier from enabled tiers"""
        flag = self._flags.get(key)
        if flag:
            flag.enabled_tiers.discard(tier)
            flag.disabled_tiers.add(tier)
            flag.updated_at = datetime.utcnow()

            if self.storage:
                await self.storage.update_flag(flag)

    async def enable_for_user(self, key: str, user_id: str):
        """Enable feature for specific user"""
        flag = self._flags.get(key)
        if flag:
            flag.enabled_users.add(user_id)
            flag.updated_at = datetime.utcnow()

            if self.storage:
                await self.storage.update_flag(flag)

    async def enable_for_segment(self, key: str, segment: str):
        """Enable feature for user segment"""
        flag = self._flags.get(key)
        if flag:
            flag.enabled_segments.add(segment)
            flag.updated_at = datetime.utcnow()

            if self.storage:
                await self.storage.update_flag(flag)

    def get_flag(self, key: str) -> Optional[FeatureFlag]:
        """Get feature flag"""
        return self._flags.get(key)

    def list_flags(
        self,
        tier: Optional[str] = None,
        enabled_only: bool = False
    ) -> List[FeatureFlag]:
        """List feature flags"""
        flags = list(self._flags.values())

        if tier:
            flags = [
                f for f in flags
                if not f.enabled_tiers or tier in f.enabled_tiers
            ]

        if enabled_only:
            flags = [f for f in flags if f.enabled]

        return flags

    def get_analytics(self, key: str) -> Dict[str, Any]:
        """Get feature flag analytics"""
        flag = self._flags.get(key)
        if not flag:
            return {}

        return {
            "key": flag.key,
            "name": flag.name,
            "evaluation_count": flag.evaluation_count,
            "true_count": flag.true_count,
            "false_count": flag.evaluation_count - flag.true_count,
            "true_rate": flag.true_count / flag.evaluation_count if flag.evaluation_count > 0 else 0,
            "enabled_tiers": list(flag.enabled_tiers),
            "rollout_percentage": flag.rollout_percentage
        }

    async def gradual_rollout(
        self,
        key: str,
        target_percentage: int,
        step_percentage: int = 10,
        step_duration_hours: int = 24
    ):
        """
        Implement gradual rollout

        Args:
            key: Feature flag key
            target_percentage: Target rollout percentage
            step_percentage: Percentage increase per step
            step_duration_hours: Hours between steps
        """
        flag = self._flags.get(key)
        if not flag:
            return

        current = flag.rollout_percentage

        logger.info(f"Starting gradual rollout for {key}: {current}% -> {target_percentage}%")

        # This would be implemented with a scheduler
        # For now, just set the target
        await self.set_rollout_percentage(key, target_percentage)

    def evaluate_multiple(
        self,
        keys: List[str],
        user_id: Optional[str] = None,
        tier: Optional[str] = None,
        segment: Optional[str] = None
    ) -> Dict[str, bool]:
        """Evaluate multiple feature flags at once"""
        return {
            key: self.is_enabled(key, user_id, tier, segment)
            for key in keys
        }
