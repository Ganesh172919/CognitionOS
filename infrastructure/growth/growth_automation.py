"""
Growth Automation Engine — CognitionOS

Automated growth and user lifecycle system with:
- Drip campaign orchestration
- Onboarding journey tracking
- Churn prediction and prevention
- User engagement scoring
- Referral system management
- Activation metrics
- Growth experiment framework
- Automated communication triggers
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class LifecycleStage(str, Enum):
    SIGNED_UP = "signed_up"
    ONBOARDING = "onboarding"
    ACTIVATED = "activated"
    ENGAGED = "engaged"
    POWER_USER = "power_user"
    AT_RISK = "at_risk"
    CHURNED = "churned"
    REACTIVATED = "reactivated"


class CampaignType(str, Enum):
    ONBOARDING = "onboarding"
    ACTIVATION = "activation"
    ENGAGEMENT = "engagement"
    RETENTION = "retention"
    UPGRADE = "upgrade"
    WINBACK = "winback"
    REFERRAL = "referral"


class CommunicationType(str, Enum):
    EMAIL = "email"
    IN_APP = "in_app"
    PUSH = "push"
    SMS = "sms"
    WEBHOOK = "webhook"


@dataclass
class UserLifecycle:
    user_id: str
    tenant_id: str
    stage: LifecycleStage = LifecycleStage.SIGNED_UP
    signed_up_at: float = field(default_factory=time.time)
    activated_at: Optional[float] = None
    last_active: float = field(default_factory=time.time)
    session_count: int = 0
    total_actions: int = 0
    key_actions_completed: Set[str] = field(default_factory=set)
    engagement_score: float = 0.0
    churn_risk: float = 0.0
    tier: str = "free"
    referral_code: str = ""
    referred_by: str = ""
    referral_count: int = 0
    ltv: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id, "tenant_id": self.tenant_id,
            "stage": self.stage.value,
            "engagement_score": round(self.engagement_score, 2),
            "churn_risk": round(self.churn_risk, 2),
            "session_count": self.session_count,
            "total_actions": self.total_actions,
            "tier": self.tier,
            "referral_count": self.referral_count,
            "ltv": round(self.ltv, 2),
        }


@dataclass
class CampaignStep:
    step_id: str
    name: str
    delay_hours: float = 0
    communication_type: CommunicationType = CommunicationType.EMAIL
    template: str = ""
    condition: Optional[Dict[str, Any]] = None
    completed_users: Set[str] = field(default_factory=set)


@dataclass
class Campaign:
    campaign_id: str
    name: str
    campaign_type: CampaignType
    steps: List[CampaignStep] = field(default_factory=list)
    target_segment: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    enrolled_users: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "name": self.name,
            "type": self.campaign_type.value,
            "steps": len(self.steps),
            "enrolled_users": len(self.enrolled_users),
            "active": self.active,
        }


@dataclass
class GrowthExperiment:
    experiment_id: str
    name: str
    hypothesis: str
    metric: str
    variants: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    user_assignments: Dict[str, str] = field(default_factory=dict)  # user_id -> variant
    results: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    active: bool = True
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name, "hypothesis": self.hypothesis,
            "metric": self.metric,
            "variants": list(self.variants.keys()),
            "participants": len(self.user_assignments),
            "active": self.active,
        }


# ── Engagement Scoring ──

ACTIVATION_MILESTONES = {
    "first_agent_created": 25,
    "first_task_completed": 20,
    "api_key_generated": 10,
    "workflow_created": 15,
    "team_member_invited": 15,
    "plugin_installed": 10,
    "billing_setup": 5,
}


class EngagementScorer:
    """Calculate user engagement scores."""

    def calculate(self, lifecycle: UserLifecycle) -> float:
        score = 0.0

        # Activation milestones (0-100 points)
        for milestone, points in ACTIVATION_MILESTONES.items():
            if milestone in lifecycle.key_actions_completed:
                score += points

        # Recency (0-20 points)
        days_since_active = (time.time() - lifecycle.last_active) / 86400
        if days_since_active < 1:
            score += 20
        elif days_since_active < 3:
            score += 15
        elif days_since_active < 7:
            score += 10
        elif days_since_active < 14:
            score += 5

        # Frequency (0-20 points)
        days_since_signup = max((time.time() - lifecycle.signed_up_at) / 86400, 1)
        sessions_per_day = lifecycle.session_count / days_since_signup
        if sessions_per_day >= 2:
            score += 20
        elif sessions_per_day >= 1:
            score += 15
        elif sessions_per_day >= 0.5:
            score += 10
        elif sessions_per_day >= 0.1:
            score += 5

        # Volume (0-10 points)
        if lifecycle.total_actions > 100:
            score += 10
        elif lifecycle.total_actions > 50:
            score += 7
        elif lifecycle.total_actions > 20:
            score += 4

        return min(score, 150)  # Cap at 150


class ChurnPredictor:
    """Predict churn risk based on user behavior."""

    def predict(self, lifecycle: UserLifecycle) -> float:
        risk = 0.0

        # Inactivity
        days_inactive = (time.time() - lifecycle.last_active) / 86400
        if days_inactive > 30:
            risk += 0.4
        elif days_inactive > 14:
            risk += 0.25
        elif days_inactive > 7:
            risk += 0.15
        elif days_inactive > 3:
            risk += 0.05

        # Low engagement
        if lifecycle.engagement_score < 20:
            risk += 0.2
        elif lifecycle.engagement_score < 40:
            risk += 0.1

        # Not activated
        if not lifecycle.activated_at:
            days_since_signup = (time.time() - lifecycle.signed_up_at) / 86400
            if days_since_signup > 7:
                risk += 0.2
            elif days_since_signup > 3:
                risk += 0.1

        # Declining usage
        if lifecycle.session_count > 5:
            # Simple check: if sessions per day trending down
            days = max((time.time() - lifecycle.signed_up_at) / 86400, 1)
            rate = lifecycle.session_count / days
            if rate < 0.1:
                risk += 0.15

        return min(risk, 1.0)


class GrowthAutomationEngine:
    """
    Master growth automation engine combining campaigns, lifecycle tracking,
    engagement scoring, churn prediction, and experiments.
    """

    def __init__(self):
        self._users: Dict[str, UserLifecycle] = {}
        self._campaigns: Dict[str, Campaign] = {}
        self._experiments: Dict[str, GrowthExperiment] = {}
        self._scorer = EngagementScorer()
        self._churn_predictor = ChurnPredictor()
        self._referral_codes: Dict[str, str] = {}  # code -> user_id
        self._communications: List[Dict[str, Any]] = []

    # ── User Lifecycle ──

    def register_user(self, user_id: str, tenant_id: str, *,
                        referred_by: str = "") -> UserLifecycle:
        referral_code = hashlib.sha256(
            f"{user_id}:{time.time()}".encode()
        ).hexdigest()[:8]

        lifecycle = UserLifecycle(
            user_id=user_id, tenant_id=tenant_id,
            referral_code=referral_code,
            referred_by=referred_by,
        )
        self._users[user_id] = lifecycle
        self._referral_codes[referral_code] = user_id

        # Credit referrer
        if referred_by and referred_by in self._referral_codes:
            referrer_id = self._referral_codes[referred_by]
            if referrer_id in self._users:
                self._users[referrer_id].referral_count += 1

        # Auto-enroll in onboarding campaign
        self._auto_enroll(lifecycle)
        return lifecycle

    def record_action(self, user_id: str, action: str):
        user = self._users.get(user_id)
        if not user:
            return

        user.total_actions += 1
        user.last_active = time.time()

        if action in ACTIVATION_MILESTONES:
            user.key_actions_completed.add(action)

        # Update stage
        self._update_stage(user)
        user.engagement_score = self._scorer.calculate(user)
        user.churn_risk = self._churn_predictor.predict(user)

    def record_session(self, user_id: str):
        user = self._users.get(user_id)
        if user:
            user.session_count += 1
            user.last_active = time.time()

    def _update_stage(self, user: UserLifecycle):
        milestones_pct = len(user.key_actions_completed) / max(len(ACTIVATION_MILESTONES), 1)

        if user.churn_risk > 0.6:
            user.stage = LifecycleStage.AT_RISK
        elif milestones_pct >= 0.8 and user.engagement_score > 100:
            user.stage = LifecycleStage.POWER_USER
        elif milestones_pct >= 0.5:
            user.stage = LifecycleStage.ENGAGED
            if not user.activated_at:
                user.activated_at = time.time()
        elif milestones_pct >= 0.2:
            user.stage = LifecycleStage.ACTIVATED
            if not user.activated_at:
                user.activated_at = time.time()
        elif user.session_count > 1:
            user.stage = LifecycleStage.ONBOARDING

    # ── Campaigns ──

    def create_campaign(self, name: str, campaign_type: CampaignType,
                          steps: Optional[List[Dict[str, Any]]] = None,
                          target_segment: Optional[Dict] = None) -> Campaign:
        campaign = Campaign(
            campaign_id=uuid.uuid4().hex[:12],
            name=name, campaign_type=campaign_type,
            target_segment=target_segment or {},
        )

        if steps:
            for s in steps:
                campaign.steps.append(CampaignStep(
                    step_id=uuid.uuid4().hex[:8],
                    name=s.get("name", ""),
                    delay_hours=s.get("delay_hours", 0),
                    communication_type=CommunicationType(
                        s.get("type", "email")
                    ),
                    template=s.get("template", ""),
                ))

        self._campaigns[campaign.campaign_id] = campaign
        return campaign

    def _auto_enroll(self, user: UserLifecycle):
        for campaign in self._campaigns.values():
            if not campaign.active:
                continue
            if campaign.campaign_type == CampaignType.ONBOARDING:
                campaign.enrolled_users.add(user.user_id)

    def enroll_user(self, campaign_id: str, user_id: str):
        campaign = self._campaigns.get(campaign_id)
        if campaign:
            campaign.enrolled_users.add(user_id)

    # ── Experiments ──

    def create_experiment(self, name: str, hypothesis: str,
                            metric: str,
                            variants: Dict[str, Dict[str, Any]]) -> GrowthExperiment:
        experiment = GrowthExperiment(
            experiment_id=uuid.uuid4().hex[:12],
            name=name, hypothesis=hypothesis,
            metric=metric, variants=variants,
        )
        self._experiments[experiment.experiment_id] = experiment
        return experiment

    def assign_experiment(self, experiment_id: str,
                            user_id: str) -> Optional[str]:
        exp = self._experiments.get(experiment_id)
        if not exp or not exp.active:
            return None

        if user_id in exp.user_assignments:
            return exp.user_assignments[user_id]

        # Deterministic assignment
        variant_names = list(exp.variants.keys())
        hash_val = int(hashlib.md5(
            f"{experiment_id}:{user_id}".encode()
        ).hexdigest(), 16) % len(variant_names)
        variant = variant_names[hash_val]
        exp.user_assignments[user_id] = variant
        return variant

    def record_experiment_metric(self, experiment_id: str,
                                    user_id: str, value: float):
        exp = self._experiments.get(experiment_id)
        if exp and user_id in exp.user_assignments:
            variant = exp.user_assignments[user_id]
            exp.results[variant].append(value)

    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        exp = self._experiments.get(experiment_id)
        if not exp:
            return {}

        results = {}
        for variant, values in exp.results.items():
            if values:
                import statistics
                results[variant] = {
                    "participants": len([
                        u for u, v in exp.user_assignments.items() if v == variant
                    ]),
                    "observations": len(values),
                    "mean": round(statistics.mean(values), 3),
                    "median": round(statistics.median(values), 3),
                }

        return {
            "experiment_id": experiment_id,
            "name": exp.name,
            "variants": results,
        }

    # ── Churn Prevention ──

    def get_at_risk_users(self, *, threshold: float = 0.5) -> List[Dict[str, Any]]:
        at_risk = []
        for user in self._users.values():
            user.churn_risk = self._churn_predictor.predict(user)
            if user.churn_risk >= threshold:
                at_risk.append({
                    "user_id": user.user_id,
                    "tenant_id": user.tenant_id,
                    "churn_risk": round(user.churn_risk, 2),
                    "days_inactive": round(
                        (time.time() - user.last_active) / 86400, 1
                    ),
                    "engagement_score": round(user.engagement_score, 1),
                    "stage": user.stage.value,
                })
        return sorted(at_risk, key=lambda x: -x["churn_risk"])

    # ── Queries ──

    def get_user_lifecycle(self, user_id: str) -> Optional[Dict[str, Any]]:
        user = self._users.get(user_id)
        return user.to_dict() if user else None

    def get_growth_metrics(self) -> Dict[str, Any]:
        total = len(self._users)
        by_stage = defaultdict(int)
        activated = 0
        at_risk = 0

        for user in self._users.values():
            by_stage[user.stage.value] += 1
            if user.activated_at:
                activated += 1
            if user.churn_risk > 0.5:
                at_risk += 1

        return {
            "total_users": total,
            "activation_rate_pct": round(activated / max(total, 1) * 100, 1),
            "at_risk_count": at_risk,
            "by_stage": dict(by_stage),
            "active_campaigns": sum(
                1 for c in self._campaigns.values() if c.active
            ),
            "active_experiments": sum(
                1 for e in self._experiments.values() if e.active
            ),
            "total_referrals": sum(u.referral_count for u in self._users.values()),
        }


# ── Singleton ──
_engine: Optional[GrowthAutomationEngine] = None


def get_growth_engine() -> GrowthAutomationEngine:
    global _engine
    if not _engine:
        _engine = GrowthAutomationEngine()
    return _engine
