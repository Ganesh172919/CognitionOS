"""
AI-Powered Onboarding Service — CognitionOS

Intelligent user onboarding with:
- Multi-step onboarding flows
- Progress tracking
- Personalized recommendations
- Checklist management
- Goal setting and tracking
- In-product tours
- Adoption scoring
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class OnboardingStage(str, Enum):
    SIGNUP = "signup"
    PROFILE_SETUP = "profile_setup"
    FIRST_AGENT = "first_agent"
    FIRST_TASK = "first_task"
    INTEGRATION = "integration"
    TEAM_INVITE = "team_invite"
    BILLING_SETUP = "billing_setup"
    COMPLETED = "completed"


@dataclass
class OnboardingStep:
    step_id: str
    name: str
    description: str
    stage: OnboardingStage
    is_required: bool = True
    order: int = 0
    action_url: str = ""
    estimated_minutes: int = 2
    tips: List[str] = field(default_factory=list)


@dataclass
class UserOnboarding:
    user_id: str
    tenant_id: str = ""
    current_stage: OnboardingStage = OnboardingStage.SIGNUP
    completed_steps: Set[str] = field(default_factory=set)
    skipped_steps: Set[str] = field(default_factory=set)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    progress_pct: float = 0.0
    goals: List[str] = field(default_factory=list)
    user_type: str = "individual"  # individual, team_lead, developer, enterprise
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    engagement_touchpoints: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id, "current_stage": self.current_stage.value,
            "progress_pct": round(self.progress_pct, 1),
            "completed_steps": len(self.completed_steps),
            "started_at": self.started_at, "completed_at": self.completed_at,
            "user_type": self.user_type, "goals": self.goals}


# Default onboarding flow
DEFAULT_STEPS = [
    OnboardingStep("create_account", "Create Account", "Sign up and verify email",
                   OnboardingStage.SIGNUP, order=1, tips=["Use a work email for team features"]),
    OnboardingStep("setup_profile", "Set Up Profile", "Configure your workspace",
                   OnboardingStage.PROFILE_SETUP, order=2),
    OnboardingStep("create_agent", "Create Your First Agent", "Build an AI agent",
                   OnboardingStage.FIRST_AGENT, order=3, estimated_minutes=5,
                   tips=["Start with a simple code generation task"]),
    OnboardingStep("run_task", "Run Your First Task", "Submit a task to your agent",
                   OnboardingStage.FIRST_TASK, order=4, estimated_minutes=3),
    OnboardingStep("connect_api", "Connect via API", "Generate an API key",
                   OnboardingStage.INTEGRATION, order=5, is_required=False),
    OnboardingStep("invite_team", "Invite Team Members", "Collaborate with your team",
                   OnboardingStage.TEAM_INVITE, order=6, is_required=False),
    OnboardingStep("setup_billing", "Set Up Billing", "Choose a plan",
                   OnboardingStage.BILLING_SETUP, order=7, is_required=False),
]


class OnboardingService:
    """Manages user onboarding flows with personalization and progress tracking."""

    def __init__(self, *, steps: List[OnboardingStep] | None = None) -> None:
        self._steps = {s.step_id: s for s in (steps or DEFAULT_STEPS)}
        self._ordered = sorted(self._steps.values(), key=lambda s: s.order)
        self._users: Dict[str, UserOnboarding] = {}
        self._metrics: Dict[str, int] = defaultdict(int)

    # ---- user management ----
    def start_onboarding(self, user_id: str, *, tenant_id: str = "",
                          user_type: str = "individual",
                          goals: List[str] | None = None) -> UserOnboarding:
        onboarding = UserOnboarding(
            user_id=user_id, tenant_id=tenant_id,
            user_type=user_type, goals=goals or [])
        onboarding.recommendations = self._generate_recommendations(user_type)
        self._users[user_id] = onboarding
        self._metrics["onboarding_started"] += 1
        return onboarding

    def complete_step(self, user_id: str, step_id: str) -> Optional[UserOnboarding]:
        onboarding = self._users.get(user_id)
        if not onboarding:
            return None
        if step_id not in self._steps:
            return None

        onboarding.completed_steps.add(step_id)
        onboarding.engagement_touchpoints += 1
        self._update_progress(onboarding)
        self._update_stage(onboarding)
        self._metrics["steps_completed"] += 1

        if onboarding.progress_pct >= 100:
            onboarding.completed_at = datetime.now(timezone.utc).isoformat()
            self._metrics["onboarding_completed"] += 1

        return onboarding

    def skip_step(self, user_id: str, step_id: str) -> Optional[UserOnboarding]:
        onboarding = self._users.get(user_id)
        if not onboarding:
            return None
        step = self._steps.get(step_id)
        if not step or step.is_required:
            return None
        onboarding.skipped_steps.add(step_id)
        self._update_progress(onboarding)
        return onboarding

    # ---- progress ----
    def _update_progress(self, onboarding: UserOnboarding) -> None:
        required = [s for s in self._ordered if s.is_required]
        optional = [s for s in self._ordered if not s.is_required]
        completed_required = sum(1 for s in required if s.step_id in onboarding.completed_steps)
        completed_optional = sum(1 for s in optional
                                 if s.step_id in onboarding.completed_steps
                                 or s.step_id in onboarding.skipped_steps)
        total_required = len(required)
        total_optional = len(optional)
        total = total_required + total_optional
        done = completed_required + completed_optional
        onboarding.progress_pct = (done / total * 100) if total > 0 else 0

    def _update_stage(self, onboarding: UserOnboarding) -> None:
        for step in self._ordered:
            if step.step_id not in onboarding.completed_steps and step.step_id not in onboarding.skipped_steps:
                onboarding.current_stage = step.stage
                return
        onboarding.current_stage = OnboardingStage.COMPLETED

    # ---- recommendations ----
    def _generate_recommendations(self, user_type: str) -> List[Dict[str, Any]]:
        recs = [
            {"title": "Quick Start Guide", "type": "doc", "priority": 1,
             "description": "Learn the basics in 5 minutes"},
            {"title": "Template Gallery", "type": "feature", "priority": 2,
             "description": "Browse pre-built agent templates"},
        ]
        if user_type == "developer":
            recs.append({"title": "API Documentation", "type": "doc", "priority": 1,
                         "description": "Integrate CognitionOS into your workflow"})
        elif user_type == "team_lead":
            recs.append({"title": "Team Management", "type": "feature", "priority": 1,
                         "description": "Set up roles and permissions"})
        elif user_type == "enterprise":
            recs.append({"title": "SSO Setup", "type": "feature", "priority": 1,
                         "description": "Configure single sign-on"})
        return recs

    def get_next_steps(self, user_id: str) -> List[Dict[str, Any]]:
        onboarding = self._users.get(user_id)
        if not onboarding:
            return []
        steps = []
        for step in self._ordered:
            if step.step_id in onboarding.completed_steps:
                continue
            if step.step_id in onboarding.skipped_steps:
                continue
            steps.append({
                "step_id": step.step_id, "name": step.name,
                "description": step.description, "required": step.is_required,
                "estimated_minutes": step.estimated_minutes, "tips": step.tips})
            if len(steps) >= 3:
                break
        return steps

    # ---- query ----
    def get_onboarding(self, user_id: str) -> Optional[Dict[str, Any]]:
        o = self._users.get(user_id)
        if not o:
            return None
        result = o.to_dict()
        result["next_steps"] = self.get_next_steps(user_id)
        result["recommendations"] = o.recommendations
        return result

    # ---- adoption scoring ----
    def get_adoption_score(self, user_id: str) -> Dict[str, Any]:
        o = self._users.get(user_id)
        if not o:
            return {"user_id": user_id, "score": 0}
        completed = len(o.completed_steps)
        total = len(self._steps)
        score = (completed / total * 100) if total > 0 else 0
        return {"user_id": user_id, "score": round(score, 1),
                "completed_steps": completed, "total_steps": total,
                "engagement_touchpoints": o.engagement_touchpoints}

    def get_metrics(self) -> Dict[str, Any]:
        total = len(self._users)
        completed = sum(1 for u in self._users.values() if u.completed_at)
        avg_progress = (sum(u.progress_pct for u in self._users.values()) / total) if total else 0
        return {**dict(self._metrics), "total_users": total,
                "completed": completed, "avg_progress_pct": round(avg_progress, 1),
                "completion_rate_pct": round(completed / total * 100, 1) if total else 0}


_service: OnboardingService | None = None

def get_onboarding_service() -> OnboardingService:
    global _service
    if not _service:
        _service = OnboardingService()
    return _service
