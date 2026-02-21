"""
Viral Growth Engine with Advanced Referral System
Network effects, viral loops, and reward mechanisms to drive user acquisition
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
import hashlib


class ReferralRewardType(Enum):
    """Type of referral reward"""
    CREDIT = "credit"
    DISCOUNT = "discount"
    FREE_TIER_UPGRADE = "free_tier_upgrade"
    FEATURE_UNLOCK = "feature_unlock"
    COMPUTE_CREDITS = "compute_credits"


class ViralMechanism(Enum):
    """Viral growth mechanism"""
    REFERRAL_LINK = "referral_link"
    INVITE_CODE = "invite_code"
    SOCIAL_SHARE = "social_share"
    TEAM_INVITE = "team_invite"
    API_INTEGRATION = "api_integration"
    MARKETPLACE_PLUGIN = "marketplace_plugin"


@dataclass
class ReferralProgram:
    """Referral program configuration"""
    program_id: str
    name: str
    referrer_reward_type: ReferralRewardType
    referrer_reward_amount: Decimal
    referee_reward_type: ReferralRewardType
    referee_reward_amount: Decimal
    min_referee_spend: Decimal = Decimal("0")  # Min spend to trigger reward
    max_referrals_per_user: int = 999
    expiry_days: int = 365
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Referral:
    """Individual referral record"""
    referral_id: str
    program_id: str
    referrer_user_id: str
    referrer_tenant_id: str
    referee_user_id: Optional[str] = None
    referee_tenant_id: Optional[str] = None
    referral_code: str = ""
    referral_link: str = ""
    status: str = "pending"  # pending, converted, rewarded, expired
    created_at: datetime = field(default_factory=datetime.utcnow)
    converted_at: Optional[datetime] = None
    rewarded_at: Optional[datetime] = None
    referee_spend_total: Decimal = Decimal("0")
    referrer_reward_issued: bool = False
    referee_reward_issued: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ViralLoop:
    """Viral loop configuration"""
    loop_id: str
    name: str
    mechanism: ViralMechanism
    trigger_event: str  # User action that triggers sharing
    incentive_message: str
    share_url_template: str
    conversion_goal: str  # What counts as conversion
    k_factor_target: float = 1.5  # Target viral coefficient
    current_k_factor: float = 0.0
    enabled: bool = True


@dataclass
class NetworkEffect:
    """Network effect measurement"""
    tenant_id: str
    user_count: int
    active_user_count: int
    collaboration_events: int
    shared_resources: int
    network_value_score: float  # Metcalfe's law calculation
    growth_rate: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GrowthMetrics:
    """Growth metrics and analytics"""
    period_start: datetime
    period_end: datetime
    new_users: int
    activated_users: int
    retained_users: int
    churned_users: int
    referral_signups: int
    organic_signups: int
    activation_rate: float
    retention_rate: float
    viral_coefficient: float
    customer_acquisition_cost: Decimal
    lifetime_value: Decimal
    payback_period_days: int


class ReferralEngine:
    """Manages referral program and viral growth"""

    def __init__(self):
        self.programs: Dict[str, ReferralProgram] = {}
        self.referrals: Dict[str, Referral] = {}
        self.viral_loops: Dict[str, ViralLoop] = {}
        self.user_referrals: Dict[str, List[str]] = {}  # user_id -> referral_ids

    async def create_referral_program(
        self,
        name: str,
        referrer_reward_type: ReferralRewardType,
        referrer_reward_amount: Decimal,
        referee_reward_type: ReferralRewardType,
        referee_reward_amount: Decimal,
        min_referee_spend: Decimal = Decimal("0"),
        max_referrals_per_user: int = 999
    ) -> ReferralProgram:
        """Create new referral program"""
        import uuid

        program = ReferralProgram(
            program_id=str(uuid.uuid4()),
            name=name,
            referrer_reward_type=referrer_reward_type,
            referrer_reward_amount=referrer_reward_amount,
            referee_reward_type=referee_reward_type,
            referee_reward_amount=referee_reward_amount,
            min_referee_spend=min_referee_spend,
            max_referrals_per_user=max_referrals_per_user
        )

        self.programs[program.program_id] = program
        return program

    async def generate_referral_link(
        self,
        program_id: str,
        referrer_user_id: str,
        referrer_tenant_id: str
    ) -> Referral:
        """Generate referral link for user"""
        import uuid

        program = self.programs.get(program_id)
        if not program:
            raise ValueError("Program not found")

        # Check max referrals
        user_referrals = self.user_referrals.get(referrer_user_id, [])
        if len(user_referrals) >= program.max_referrals_per_user:
            raise ValueError("Max referrals limit reached")

        # Generate referral code
        referral_id = str(uuid.uuid4())
        referral_code = hashlib.md5(
            f"{referrer_user_id}:{referral_id}".encode()
        ).hexdigest()[:8].upper()

        # Generate referral link
        referral_link = f"https://platform.cognitionos.ai/signup?ref={referral_code}"

        referral = Referral(
            referral_id=referral_id,
            program_id=program_id,
            referrer_user_id=referrer_user_id,
            referrer_tenant_id=referrer_tenant_id,
            referral_code=referral_code,
            referral_link=referral_link
        )

        self.referrals[referral_id] = referral

        if referrer_user_id not in self.user_referrals:
            self.user_referrals[referrer_user_id] = []
        self.user_referrals[referrer_user_id].append(referral_id)

        return referral

    async def convert_referral(
        self,
        referral_code: str,
        referee_user_id: str,
        referee_tenant_id: str
    ) -> Referral:
        """Mark referral as converted"""
        # Find referral by code
        referral = None
        for ref in self.referrals.values():
            if ref.referral_code == referral_code:
                referral = ref
                break

        if not referral:
            raise ValueError("Invalid referral code")

        if referral.status != "pending":
            raise ValueError("Referral already converted")

        referral.referee_user_id = referee_user_id
        referral.referee_tenant_id = referee_tenant_id
        referral.status = "converted"
        referral.converted_at = datetime.utcnow()

        # Issue referee reward immediately
        await self._issue_referee_reward(referral)

        return referral

    async def _issue_referee_reward(self, referral: Referral):
        """Issue reward to referee"""
        program = self.programs.get(referral.program_id)
        if not program:
            return

        # In real implementation, would apply the reward
        # For now, just mark as issued
        referral.referee_reward_issued = True

    async def track_referee_spend(
        self,
        referee_user_id: str,
        amount: Decimal
    ):
        """Track referee spending to trigger referrer rewards"""
        # Find referrals where this user is referee
        for referral in self.referrals.values():
            if referral.referee_user_id == referee_user_id:
                referral.referee_spend_total += amount

                program = self.programs.get(referral.program_id)
                if not program:
                    continue

                # Check if min spend reached and reward not yet issued
                if (referral.referee_spend_total >= program.min_referee_spend and
                    not referral.referrer_reward_issued):
                    await self._issue_referrer_reward(referral)

    async def _issue_referrer_reward(self, referral: Referral):
        """Issue reward to referrer"""
        program = self.programs.get(referral.program_id)
        if not program:
            return

        # In real implementation, would apply the reward
        referral.referrer_reward_issued = True
        referral.rewarded_at = datetime.utcnow()
        referral.status = "rewarded"

    async def get_referral_stats(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Get referral statistics for user"""
        referral_ids = self.user_referrals.get(user_id, [])
        referrals = [self.referrals[rid] for rid in referral_ids if rid in self.referrals]

        total_referrals = len(referrals)
        pending = len([r for r in referrals if r.status == "pending"])
        converted = len([r for r in referrals if r.status in ["converted", "rewarded"]])
        rewarded = len([r for r in referrals if r.status == "rewarded"])

        total_rewards = Decimal("0")
        for referral in referrals:
            if referral.referrer_reward_issued:
                program = self.programs.get(referral.program_id)
                if program:
                    total_rewards += program.referrer_reward_amount

        conversion_rate = (converted / total_referrals * 100) if total_referrals > 0 else 0

        return {
            "user_id": user_id,
            "total_referrals": total_referrals,
            "pending": pending,
            "converted": converted,
            "rewarded": rewarded,
            "conversion_rate": round(conversion_rate, 2),
            "total_rewards_earned": float(total_rewards)
        }


class ViralLoopEngine:
    """Manages viral loops and network effects"""

    def __init__(self):
        self.viral_loops: Dict[str, ViralLoop] = {}
        self.network_effects: Dict[str, List[NetworkEffect]] = {}
        self.loop_events: Dict[str, List[Dict[str, Any]]] = {}

    async def create_viral_loop(
        self,
        name: str,
        mechanism: ViralMechanism,
        trigger_event: str,
        incentive_message: str,
        share_url_template: str,
        conversion_goal: str,
        k_factor_target: float = 1.5
    ) -> ViralLoop:
        """Create viral loop"""
        import uuid

        loop = ViralLoop(
            loop_id=str(uuid.uuid4()),
            name=name,
            mechanism=mechanism,
            trigger_event=trigger_event,
            incentive_message=incentive_message,
            share_url_template=share_url_template,
            conversion_goal=conversion_goal,
            k_factor_target=k_factor_target
        )

        self.viral_loops[loop.loop_id] = loop
        return loop

    async def trigger_viral_loop(
        self,
        loop_id: str,
        user_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Trigger viral loop for user"""
        loop = self.viral_loops.get(loop_id)
        if not loop or not loop.enabled:
            return {"triggered": False}

        # Generate personalized share content
        share_url = loop.share_url_template.format(**context, user_id=user_id)

        # Record event
        event = {
            "timestamp": datetime.utcnow(),
            "user_id": user_id,
            "loop_id": loop_id,
            "share_url": share_url,
            "context": context
        }

        if loop_id not in self.loop_events:
            self.loop_events[loop_id] = []
        self.loop_events[loop_id].append(event)

        return {
            "triggered": True,
            "loop_id": loop_id,
            "share_url": share_url,
            "incentive_message": loop.incentive_message
        }

    async def calculate_viral_coefficient(
        self,
        loop_id: str,
        time_period_days: int = 30
    ) -> float:
        """Calculate viral coefficient (k-factor) for loop"""
        cutoff = datetime.utcnow() - timedelta(days=time_period_days)

        events = [
            e for e in self.loop_events.get(loop_id, [])
            if e["timestamp"] >= cutoff
        ]

        if not events:
            return 0.0

        # Count unique users who triggered the loop
        triggers = len(set(e["user_id"] for e in events))

        # Count conversions (simplified - would track actual conversions)
        # k = (invites sent per user) * (conversion rate)
        # Simplified: assume 20% conversion rate
        invites_per_user = len(events) / triggers if triggers > 0 else 0
        conversion_rate = 0.20  # Simplified

        k_factor = invites_per_user * conversion_rate

        # Update loop
        loop = self.viral_loops.get(loop_id)
        if loop:
            loop.current_k_factor = k_factor

        return k_factor

    async def measure_network_effects(
        self,
        tenant_id: str,
        user_count: int,
        active_user_count: int,
        collaboration_events: int,
        shared_resources: int
    ) -> NetworkEffect:
        """Measure network effects for tenant"""
        # Calculate network value using Metcalfe's law: Value ~ n^2
        # Adjusted for active users
        network_value_score = (active_user_count ** 2) * 0.001

        # Calculate growth rate
        historical = self.network_effects.get(tenant_id, [])
        if historical:
            prev = historical[-1]
            days_diff = (datetime.utcnow() - prev.timestamp).days
            if days_diff > 0:
                growth_rate = (
                    (active_user_count - prev.active_user_count) / prev.active_user_count * 100
                ) / days_diff
            else:
                growth_rate = 0.0
        else:
            growth_rate = 0.0

        effect = NetworkEffect(
            tenant_id=tenant_id,
            user_count=user_count,
            active_user_count=active_user_count,
            collaboration_events=collaboration_events,
            shared_resources=shared_resources,
            network_value_score=network_value_score,
            growth_rate=growth_rate
        )

        if tenant_id not in self.network_effects:
            self.network_effects[tenant_id] = []
        self.network_effects[tenant_id].append(effect)

        return effect


class GrowthAnalyticsEngine:
    """Analyzes growth metrics and provides insights"""

    def __init__(self):
        self.user_events: Dict[str, List[Dict[str, Any]]] = {}

    async def calculate_growth_metrics(
        self,
        period_start: datetime,
        period_end: datetime,
        new_users: int,
        activated_users: int,
        retained_users: int,
        churned_users: int,
        referral_signups: int,
        organic_signups: int,
        total_acquisition_cost: Decimal,
        total_revenue: Decimal
    ) -> GrowthMetrics:
        """Calculate comprehensive growth metrics"""
        # Activation rate
        activation_rate = (
            (activated_users / new_users * 100)
            if new_users > 0 else 0
        )

        # Retention rate
        total_active_start = activated_users + retained_users
        retention_rate = (
            (retained_users / total_active_start * 100)
            if total_active_start > 0 else 0
        )

        # Viral coefficient
        viral_coefficient = (
            (referral_signups / new_users)
            if new_users > 0 else 0
        )

        # Customer acquisition cost
        cac = (
            (total_acquisition_cost / new_users)
            if new_users > 0 else Decimal("0")
        )

        # Lifetime value (simplified - 24 month projection)
        avg_revenue_per_user = (
            (total_revenue / activated_users)
            if activated_users > 0 else Decimal("0")
        )
        ltv = avg_revenue_per_user * Decimal("24")  # 24 months

        # Payback period (CAC / monthly revenue per user)
        monthly_revenue = avg_revenue_per_user
        payback_period_days = (
            int((cac / monthly_revenue) * 30)
            if monthly_revenue > 0 else 999
        )

        return GrowthMetrics(
            period_start=period_start,
            period_end=period_end,
            new_users=new_users,
            activated_users=activated_users,
            retained_users=retained_users,
            churned_users=churned_users,
            referral_signups=referral_signups,
            organic_signups=organic_signups,
            activation_rate=activation_rate,
            retention_rate=retention_rate,
            viral_coefficient=viral_coefficient,
            customer_acquisition_cost=cac,
            lifetime_value=ltv,
            payback_period_days=payback_period_days
        )

    async def get_growth_insights(
        self,
        metrics: GrowthMetrics
    ) -> List[Dict[str, str]]:
        """Get actionable growth insights"""
        insights = []

        # Analyze viral coefficient
        if metrics.viral_coefficient < 0.5:
            insights.append({
                "type": "viral_coefficient",
                "severity": "high",
                "message": f"Viral coefficient is low ({metrics.viral_coefficient:.2f}). Implement stronger referral incentives.",
                "recommendation": "Increase referral rewards and make sharing easier"
            })
        elif metrics.viral_coefficient > 1.0:
            insights.append({
                "type": "viral_coefficient",
                "severity": "positive",
                "message": f"Excellent viral coefficient ({metrics.viral_coefficient:.2f})! Self-sustaining growth achieved.",
                "recommendation": "Maintain current growth strategies and scale up"
            })

        # Analyze activation rate
        if metrics.activation_rate < 40:
            insights.append({
                "type": "activation",
                "severity": "high",
                "message": f"Low activation rate ({metrics.activation_rate:.1f}%). Users not finding value quickly.",
                "recommendation": "Improve onboarding flow and time-to-value"
            })

        # Analyze retention
        if metrics.retention_rate < 60:
            insights.append({
                "type": "retention",
                "severity": "medium",
                "message": f"Retention rate needs improvement ({metrics.retention_rate:.1f}%).",
                "recommendation": "Focus on engagement features and user success"
            })

        # Analyze unit economics
        ltv_cac_ratio = float(metrics.lifetime_value / metrics.customer_acquisition_cost) if metrics.customer_acquisition_cost > 0 else 0
        if ltv_cac_ratio < 3:
            insights.append({
                "type": "unit_economics",
                "severity": "high",
                "message": f"LTV:CAC ratio is low ({ltv_cac_ratio:.1f}). Not sustainable.",
                "recommendation": "Reduce acquisition costs or increase customer lifetime value"
            })

        # Analyze payback period
        if metrics.payback_period_days > 180:
            insights.append({
                "type": "payback_period",
                "severity": "medium",
                "message": f"Long payback period ({metrics.payback_period_days} days).",
                "recommendation": "Optimize pricing or reduce CAC"
            })

        return insights
