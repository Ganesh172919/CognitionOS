"""
Subscription Lifecycle Management

Complete upgrade/downgrade workflows with prorated billing,
migration handling, and customer success tracking.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SubscriptionAction(str, Enum):
    """Subscription lifecycle actions"""
    UPGRADE = "upgrade"
    DOWNGRADE = "downgrade"
    CANCEL = "cancel"
    REACTIVATE = "reactivate"
    PAUSE = "pause"
    RESUME = "resume"


class MigrationStrategy(str, Enum):
    """Data migration strategies for plan changes"""
    IMMEDIATE = "immediate"
    END_OF_PERIOD = "end_of_period"
    SCHEDULED = "scheduled"
    GRADUAL = "gradual"


@dataclass
class SubscriptionChange:
    """Record of subscription change"""
    change_id: str
    tenant_id: str
    action: SubscriptionAction
    from_tier: str
    to_tier: str
    effective_date: datetime

    # Financial
    prorated_credit: float = 0.0
    prorated_charge: float = 0.0
    net_change: float = 0.0

    # Migration
    migration_strategy: MigrationStrategy = MigrationStrategy.IMMEDIATE
    data_migration_required: bool = False
    migration_completed: bool = False

    # Tracking
    initiated_by: Optional[str] = None
    reason: Optional[str] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class UpgradeWorkflow:
    """
    Handles subscription upgrades with prorated billing
    """

    def __init__(self, billing_service: Optional[Any] = None):
        self.billing_service = billing_service

    async def execute_upgrade(
        self,
        tenant_id: str,
        from_tier: str,
        to_tier: str,
        immediate: bool = True,
        initiated_by: Optional[str] = None
    ) -> SubscriptionChange:
        """
        Execute subscription upgrade

        Args:
            tenant_id: Tenant identifier
            from_tier: Current tier
            to_tier: Target tier
            immediate: Apply immediately or at period end
            initiated_by: User who initiated upgrade

        Returns:
            SubscriptionChange record
        """
        logger.info(f"Upgrading {tenant_id} from {from_tier} to {to_tier}")

        # Calculate prorated charges
        proration = await self._calculate_proration(
            tenant_id, from_tier, to_tier, immediate
        )

        change = SubscriptionChange(
            change_id=f"upgrade_{tenant_id}_{int(datetime.utcnow().timestamp())}",
            tenant_id=tenant_id,
            action=SubscriptionAction.UPGRADE,
            from_tier=from_tier,
            to_tier=to_tier,
            effective_date=datetime.utcnow() if immediate else self._get_period_end(tenant_id),
            prorated_credit=proration["credit"],
            prorated_charge=proration["charge"],
            net_change=proration["net"],
            migration_strategy=MigrationStrategy.IMMEDIATE if immediate else MigrationStrategy.END_OF_PERIOD,
            initiated_by=initiated_by
        )

        # Execute upgrade
        if immediate:
            await self._apply_upgrade(change)
        else:
            await self._schedule_upgrade(change)

        # Send notifications
        await self._notify_upgrade(change)

        return change

    async def _calculate_proration(
        self,
        tenant_id: str,
        from_tier: str,
        to_tier: str,
        immediate: bool
    ) -> Dict[str, float]:
        """Calculate prorated amounts"""

        # Get current subscription details
        current_period_start = datetime.utcnow() - timedelta(days=15)  # Mock
        current_period_end = datetime.utcnow() + timedelta(days=15)    # Mock

        # Calculate remaining days
        days_remaining = (current_period_end - datetime.utcnow()).days
        days_total = (current_period_end - current_period_start).days

        # Tier pricing (mock - would come from subscription service)
        tier_pricing = {
            "free": 0,
            "starter": 29,
            "pro": 99,
            "enterprise": 499
        }

        from_price = tier_pricing.get(from_tier.lower(), 0)
        to_price = tier_pricing.get(to_tier.lower(), 0)

        if immediate:
            # Credit unused portion of current plan
            credit = (from_price * days_remaining) / days_total

            # Charge for new plan (prorated)
            charge = (to_price * days_remaining) / days_total

            net = charge - credit
        else:
            # No proration if waiting until period end
            credit = 0
            charge = to_price
            net = to_price

        return {
            "credit": round(credit, 2),
            "charge": round(charge, 2),
            "net": round(net, 2)
        }

    async def _apply_upgrade(self, change: SubscriptionChange):
        """Apply upgrade immediately"""
        logger.info(f"Applying immediate upgrade: {change.change_id}")

        # Update subscription in billing system
        if self.billing_service:
            await self.billing_service.update_subscription(
                tenant_id=change.tenant_id,
                new_tier=change.to_tier
            )

        # Update feature access
        await self._update_feature_access(change.tenant_id, change.to_tier)

        # Process payment for net charge
        if change.net_change > 0:
            await self._process_payment(change.tenant_id, change.net_change)

        change.completed_at = datetime.utcnow()

    async def _schedule_upgrade(self, change: SubscriptionChange):
        """Schedule upgrade for period end"""
        logger.info(f"Scheduling upgrade: {change.change_id}")
        # Would store in scheduler for execution at period end

    async def _update_feature_access(self, tenant_id: str, new_tier: str):
        """Update feature access for tenant"""
        logger.info(f"Updating feature access for {tenant_id} to {new_tier}")

    async def _process_payment(self, tenant_id: str, amount: float):
        """Process payment"""
        logger.info(f"Processing payment of ${amount} for {tenant_id}")

    def _get_period_end(self, tenant_id: str) -> datetime:
        """Get current billing period end"""
        return datetime.utcnow() + timedelta(days=15)  # Mock

    async def _notify_upgrade(self, change: SubscriptionChange):
        """Send upgrade notification"""
        logger.info(f"Sending upgrade notification for {change.tenant_id}")


class DowngradeWorkflow:
    """
    Handles subscription downgrades with data migration
    """

    def __init__(self, billing_service: Optional[Any] = None):
        self.billing_service = billing_service

    async def execute_downgrade(
        self,
        tenant_id: str,
        from_tier: str,
        to_tier: str,
        reason: Optional[str] = None,
        initiated_by: Optional[str] = None
    ) -> SubscriptionChange:
        """
        Execute subscription downgrade

        Args:
            tenant_id: Tenant identifier
            from_tier: Current tier
            to_tier: Target tier
            reason: Reason for downgrade
            initiated_by: User who initiated

        Returns:
            SubscriptionChange record
        """
        logger.info(f"Downgrading {tenant_id} from {from_tier} to {to_tier}")

        # Check data migration requirements
        migration_required = await self._check_migration_requirements(
            tenant_id, from_tier, to_tier
        )

        change = SubscriptionChange(
            change_id=f"downgrade_{tenant_id}_{int(datetime.utcnow().timestamp())}",
            tenant_id=tenant_id,
            action=SubscriptionAction.DOWNGRADE,
            from_tier=from_tier,
            to_tier=to_tier,
            effective_date=self._get_period_end(tenant_id),  # Always end of period
            migration_strategy=MigrationStrategy.END_OF_PERIOD,
            data_migration_required=migration_required,
            reason=reason,
            initiated_by=initiated_by
        )

        # Schedule downgrade
        await self._schedule_downgrade(change)

        # Send notifications with migration warnings if needed
        await self._notify_downgrade(change)

        return change

    async def _check_migration_requirements(
        self,
        tenant_id: str,
        from_tier: str,
        to_tier: str
    ) -> bool:
        """Check if data migration is required"""

        # Check current usage against new limits
        # Would query actual usage from database
        current_usage = {
            "api_calls": 50000,
            "workflows": 500,
            "storage_gb": 50,
            "team_members": 8
        }

        # New tier limits
        new_limits = {
            "free": {"api_calls": 1000, "workflows": 10, "storage_gb": 1, "team_members": 1},
            "starter": {"api_calls": 10000, "workflows": 100, "storage_gb": 10, "team_members": 3},
            "pro": {"api_calls": 100000, "workflows": 1000, "storage_gb": 100, "team_members": 10}
        }

        limits = new_limits.get(to_tier.lower(), {})

        # Check if any usage exceeds new limits
        for metric, value in current_usage.items():
            if metric in limits and value > limits[metric]:
                logger.warning(f"Migration required: {metric} usage {value} exceeds new limit {limits[metric]}")
                return True

        return False

    async def _schedule_downgrade(self, change: SubscriptionChange):
        """Schedule downgrade for period end"""
        logger.info(f"Scheduling downgrade: {change.change_id}")

        # Store scheduled change
        # Would use scheduler service

        # If migration required, start preparing
        if change.data_migration_required:
            await self._prepare_migration(change)

    async def _prepare_migration(self, change: SubscriptionChange):
        """Prepare data migration plan"""
        logger.info(f"Preparing data migration for {change.tenant_id}")

        # Identify data that exceeds new limits
        # Create migration plan
        # Notify customer of actions needed

    def _get_period_end(self, tenant_id: str) -> datetime:
        """Get current billing period end"""
        return datetime.utcnow() + timedelta(days=15)  # Mock

    async def _notify_downgrade(self, change: SubscriptionChange):
        """Send downgrade notification with warnings"""
        logger.info(f"Sending downgrade notification for {change.tenant_id}")

        if change.data_migration_required:
            logger.warning(f"Data migration required for {change.tenant_id}")


class SubscriptionLifecycleManager:
    """
    Complete subscription lifecycle management

    Handles all subscription changes, tracking, and customer success.
    """

    def __init__(
        self,
        billing_service: Optional[Any] = None,
        notification_service: Optional[Any] = None
    ):
        self.billing_service = billing_service
        self.notification_service = notification_service
        self.upgrade_workflow = UpgradeWorkflow(billing_service)
        self.downgrade_workflow = DowngradeWorkflow(billing_service)
        self._changes: List[SubscriptionChange] = []

    async def upgrade_subscription(
        self,
        tenant_id: str,
        from_tier: str,
        to_tier: str,
        immediate: bool = True,
        initiated_by: Optional[str] = None
    ) -> SubscriptionChange:
        """Upgrade subscription"""
        change = await self.upgrade_workflow.execute_upgrade(
            tenant_id, from_tier, to_tier, immediate, initiated_by
        )
        self._changes.append(change)
        return change

    async def downgrade_subscription(
        self,
        tenant_id: str,
        from_tier: str,
        to_tier: str,
        reason: Optional[str] = None,
        initiated_by: Optional[str] = None
    ) -> SubscriptionChange:
        """Downgrade subscription"""
        change = await self.downgrade_workflow.execute_downgrade(
            tenant_id, from_tier, to_tier, reason, initiated_by
        )
        self._changes.append(change)
        return change

    async def cancel_subscription(
        self,
        tenant_id: str,
        reason: Optional[str] = None,
        immediate: bool = False
    ) -> SubscriptionChange:
        """Cancel subscription"""
        logger.info(f"Canceling subscription for {tenant_id}")

        change = SubscriptionChange(
            change_id=f"cancel_{tenant_id}_{int(datetime.utcnow().timestamp())}",
            tenant_id=tenant_id,
            action=SubscriptionAction.CANCEL,
            from_tier="current",
            to_tier="canceled",
            effective_date=datetime.utcnow() if immediate else self._get_period_end(tenant_id),
            reason=reason
        )

        # Execute cancellation
        if immediate:
            await self._execute_immediate_cancellation(change)
        else:
            await self._schedule_cancellation(change)

        self._changes.append(change)
        return change

    async def pause_subscription(
        self,
        tenant_id: str,
        duration_days: Optional[int] = None
    ) -> SubscriptionChange:
        """Pause subscription temporarily"""
        logger.info(f"Pausing subscription for {tenant_id}")

        resume_date = datetime.utcnow() + timedelta(days=duration_days) if duration_days else None

        change = SubscriptionChange(
            change_id=f"pause_{tenant_id}_{int(datetime.utcnow().timestamp())}",
            tenant_id=tenant_id,
            action=SubscriptionAction.PAUSE,
            from_tier="current",
            to_tier="paused",
            effective_date=datetime.utcnow(),
            metadata={"resume_date": resume_date.isoformat() if resume_date else None}
        )

        await self._execute_pause(change)
        self._changes.append(change)
        return change

    async def _execute_immediate_cancellation(self, change: SubscriptionChange):
        """Execute immediate cancellation"""
        logger.info(f"Executing immediate cancellation for {change.tenant_id}")

        # Revoke access
        # Cancel billing
        # Archive data

        change.completed_at = datetime.utcnow()

    async def _schedule_cancellation(self, change: SubscriptionChange):
        """Schedule cancellation for period end"""
        logger.info(f"Scheduling cancellation for {change.tenant_id}")

    async def _execute_pause(self, change: SubscriptionChange):
        """Pause subscription"""
        logger.info(f"Pausing subscription for {change.tenant_id}")

        # Suspend billing
        # Preserve data
        # Limit access

        change.completed_at = datetime.utcnow()

    def _get_period_end(self, tenant_id: str) -> datetime:
        """Get current billing period end"""
        return datetime.utcnow() + timedelta(days=15)

    def get_change_history(
        self,
        tenant_id: str,
        limit: int = 10
    ) -> List[SubscriptionChange]:
        """Get subscription change history"""
        changes = [c for c in self._changes if c.tenant_id == tenant_id]
        return sorted(changes, key=lambda c: c.effective_date, reverse=True)[:limit]

    def get_lifecycle_metrics(self) -> Dict[str, Any]:
        """Get lifecycle metrics"""
        total_changes = len(self._changes)

        action_counts = {}
        for change in self._changes:
            action_counts[change.action.value] = action_counts.get(change.action.value, 0) + 1

        return {
            "total_changes": total_changes,
            "by_action": action_counts,
            "upgrade_rate": action_counts.get("upgrade", 0) / max(total_changes, 1),
            "downgrade_rate": action_counts.get("downgrade", 0) / max(total_changes, 1),
            "cancel_rate": action_counts.get("cancel", 0) / max(total_changes, 1)
        }
