"""
User Onboarding Engine

Progressive onboarding system that guides users through feature discovery:
- Milestone-based onboarding with smart sequencing
- Feature spotlight triggers (show feature when user is ready)
- In-product checklist with progress tracking
- Contextual tooltips/hints based on user behavior
- Personalized onboarding paths (developer, analyst, admin)
- Completion reward tracking for engagement
- A/B testable onboarding flows
- Analytics hooks for conversion funnel measurement
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4


class OnboardingPersona(str, Enum):
    """User persona determines which onboarding path to follow"""
    DEVELOPER = "developer"
    ANALYST = "analyst"
    ADMIN = "admin"
    POWER_USER = "power_user"
    GENERAL = "general"


class MilestoneStatus(str, Enum):
    LOCKED = "locked"       # Prerequisites not met
    AVAILABLE = "available" # Ready to start
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class TriggerType(str, Enum):
    PAGE_VISIT = "page_visit"
    FEATURE_USE = "feature_use"
    TIME_ELAPSED = "time_elapsed"
    MILESTONE_COMPLETE = "milestone_complete"
    MANUAL = "manual"


@dataclass
class OnboardingMilestone:
    """A single onboarding milestone (step/task)"""
    milestone_id: str
    title: str
    description: str
    persona: Optional[OnboardingPersona] = None  # None = all personas
    prerequisites: List[str] = field(default_factory=list)  # milestone_ids
    trigger: TriggerType = TriggerType.MANUAL
    trigger_value: Optional[str] = None    # e.g. page name, feature name
    reward_points: int = 10
    is_optional: bool = False
    cta_text: str = "Get Started"
    cta_action: Optional[str] = None      # e.g. "/workflows/new"
    completion_video_url: Optional[str] = None
    estimated_minutes: int = 5
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "milestone_id": self.milestone_id,
            "title": self.title,
            "description": self.description,
            "trigger": self.trigger.value,
            "reward_points": self.reward_points,
            "is_optional": self.is_optional,
            "cta_text": self.cta_text,
            "cta_action": self.cta_action,
            "estimated_minutes": self.estimated_minutes,
        }


@dataclass
class UserOnboardingState:
    """Tracks a single user's onboarding progress"""
    user_id: str
    tenant_id: str
    persona: OnboardingPersona
    completed_milestones: Set[str] = field(default_factory=set)
    skipped_milestones: Set[str] = field(default_factory=set)
    in_progress_milestones: Set[str] = field(default_factory=set)
    total_points: int = 0
    started_at: float = field(default_factory=time.time)
    last_activity_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def completion_rate(self) -> float:
        """Percentage of non-optional milestones completed"""
        return (
            len(self.completed_milestones)
            / max(1, len(self.completed_milestones) + len(self.in_progress_milestones))
        )

    def is_complete(self, milestone_id: str) -> bool:
        return milestone_id in self.completed_milestones

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "persona": self.persona.value,
            "completed_count": len(self.completed_milestones),
            "total_points": self.total_points,
            "completion_rate": round(self.completion_rate * 100, 1),
            "started_at": self.started_at,
            "last_activity_at": self.last_activity_at,
        }


@dataclass
class OnboardingChecklist:
    """A grouped view of onboarding milestones for display"""
    checklist_id: str
    title: str
    items: List[Dict[str, Any]]
    completed_count: int
    total_count: int
    next_milestone: Optional[Dict[str, Any]]
    progress_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checklist_id": self.checklist_id,
            "title": self.title,
            "items": self.items,
            "completed_count": self.completed_count,
            "total_count": self.total_count,
            "next_milestone": self.next_milestone,
            "progress_pct": round(self.progress_pct, 1),
        }


@dataclass
class FeatureSpotlight:
    """A contextual feature spotlight to show to a user"""
    spotlight_id: str
    feature_name: str
    headline: str
    body: str
    target_selector: Optional[str] = None  # CSS selector or element ID
    position: str = "bottom"               # top/bottom/left/right
    cta_text: str = "Try it"
    cta_url: Optional[str] = None
    dismiss_text: str = "Maybe later"
    show_after_milestone: Optional[str] = None
    persona: Optional[OnboardingPersona] = None
    max_shows: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spotlight_id": self.spotlight_id,
            "feature_name": self.feature_name,
            "headline": self.headline,
            "body": self.body,
            "target_selector": self.target_selector,
            "position": self.position,
            "cta_text": self.cta_text,
            "cta_url": self.cta_url,
            "dismiss_text": self.dismiss_text,
        }


class OnboardingEngine:
    """
    Manages user onboarding journeys with milestone tracking and feature discovery.

    Usage::

        engine = OnboardingEngine()
        engine.register_milestone(OnboardingMilestone(
            milestone_id="create_first_workflow",
            title="Create Your First Workflow",
            description="...",
        ))
        engine.start_onboarding("user-1", "tenant-a", OnboardingPersona.DEVELOPER)
        engine.complete_milestone("user-1", "create_first_workflow")
        checklist = engine.get_checklist("user-1")
    """

    def __init__(self) -> None:
        self._milestones: Dict[str, OnboardingMilestone] = {}
        self._spotlights: Dict[str, FeatureSpotlight] = {}
        self._user_states: Dict[str, UserOnboardingState] = {}
        self._spotlight_show_counts: Dict[str, Dict[str, int]] = {}
        self._analytics: List[Dict[str, Any]] = []
        self._setup_default_milestones()
        self._setup_default_spotlights()

    # ──────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────

    def register_milestone(self, milestone: OnboardingMilestone) -> None:
        self._milestones[milestone.milestone_id] = milestone

    def register_spotlight(self, spotlight: FeatureSpotlight) -> None:
        self._spotlights[spotlight.spotlight_id] = spotlight

    # ──────────────────────────────────────────────
    # User Lifecycle
    # ──────────────────────────────────────────────

    def start_onboarding(
        self,
        user_id: str,
        tenant_id: str,
        persona: OnboardingPersona = OnboardingPersona.GENERAL,
    ) -> UserOnboardingState:
        if user_id in self._user_states:
            return self._user_states[user_id]
        state = UserOnboardingState(
            user_id=user_id,
            tenant_id=tenant_id,
            persona=persona,
        )
        self._user_states[user_id] = state
        self._track_event(user_id, "onboarding_started", {"persona": persona.value})
        return state

    def get_state(self, user_id: str) -> Optional[UserOnboardingState]:
        return self._user_states.get(user_id)

    def complete_milestone(
        self,
        user_id: str,
        milestone_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Mark a milestone as completed, returning earned rewards"""
        state = self._user_states.get(user_id)
        if not state:
            return {"error": "User onboarding not started"}

        milestone = self._milestones.get(milestone_id)
        if not milestone:
            return {"error": f"Unknown milestone: {milestone_id}"}

        if milestone_id in state.completed_milestones:
            return {"already_completed": True, "points": 0}

        # Check prerequisites
        missing_prereqs = [
            p for p in milestone.prerequisites
            if p not in state.completed_milestones
        ]
        if missing_prereqs:
            return {"error": f"Prerequisites not met: {missing_prereqs}"}

        state.completed_milestones.add(milestone_id)
        state.in_progress_milestones.discard(milestone_id)
        state.total_points += milestone.reward_points
        state.last_activity_at = time.time()

        self._track_event(user_id, "milestone_completed", {
            "milestone_id": milestone_id,
            "points_earned": milestone.reward_points,
            **(metadata or {}),
        })

        # Check what unlocks
        newly_available = self._get_newly_available(state)

        return {
            "completed": True,
            "points_earned": milestone.reward_points,
            "total_points": state.total_points,
            "newly_available": [m.to_dict() for m in newly_available],
        }

    def skip_milestone(self, user_id: str, milestone_id: str) -> bool:
        state = self._user_states.get(user_id)
        if not state:
            return False
        milestone = self._milestones.get(milestone_id)
        if not milestone or not milestone.is_optional:
            return False
        state.skipped_milestones.add(milestone_id)
        state.last_activity_at = time.time()
        return True

    def trigger_event(
        self,
        user_id: str,
        trigger_type: TriggerType,
        trigger_value: Optional[str] = None,
    ) -> List[OnboardingMilestone]:
        """
        Fire an event that may trigger auto-completion of milestones.
        Returns list of milestones that were auto-completed.
        """
        state = self._user_states.get(user_id)
        if not state:
            return []

        triggered: List[OnboardingMilestone] = []
        for milestone in self._milestones.values():
            if milestone.milestone_id in state.completed_milestones:
                continue
            if milestone.trigger != trigger_type:
                continue
            if trigger_value and milestone.trigger_value and milestone.trigger_value != trigger_value:
                continue
            # Check persona
            if milestone.persona and milestone.persona != state.persona:
                continue
            result = self.complete_milestone(user_id, milestone.milestone_id)
            if result.get("completed"):
                triggered.append(milestone)

        return triggered

    # ──────────────────────────────────────────────
    # Checklist & Progress
    # ──────────────────────────────────────────────

    def get_checklist(self, user_id: str) -> Optional[OnboardingChecklist]:
        state = self._user_states.get(user_id)
        if not state:
            return None

        relevant = [
            m for m in self._milestones.values()
            if m.persona is None or m.persona == state.persona
        ]

        items = []
        for m in relevant:
            if m.milestone_id in state.completed_milestones:
                milestone_status = MilestoneStatus.COMPLETED
            elif m.milestone_id in state.skipped_milestones:
                milestone_status = MilestoneStatus.SKIPPED
            elif all(p in state.completed_milestones for p in m.prerequisites):
                milestone_status = MilestoneStatus.AVAILABLE
            else:
                milestone_status = MilestoneStatus.LOCKED

            items.append({
                **m.to_dict(),
                "status": milestone_status.value,
            })

        completed_count = len(state.completed_milestones)
        required_total = sum(1 for m in relevant if not m.is_optional)
        progress_pct = (completed_count / max(1, len(relevant))) * 100

        next_available = next(
            (item for item in items if item["status"] == MilestoneStatus.AVAILABLE.value),
            None,
        )

        return OnboardingChecklist(
            checklist_id=str(uuid4()),
            title="Getting Started",
            items=items,
            completed_count=completed_count,
            total_count=len(relevant),
            next_milestone=next_available,
            progress_pct=progress_pct,
        )

    def get_spotlights_for_user(self, user_id: str) -> List[FeatureSpotlight]:
        """Return spotlights that should be shown to this user right now"""
        state = self._user_states.get(user_id)
        if not state:
            return []

        show_counts = self._spotlight_show_counts.setdefault(user_id, {})
        to_show: List[FeatureSpotlight] = []

        for spotlight in self._spotlights.values():
            # Check persona
            if spotlight.persona and spotlight.persona != state.persona:
                continue
            # Check show count
            shown = show_counts.get(spotlight.spotlight_id, 0)
            if shown >= spotlight.max_shows:
                continue
            # Check prerequisite milestone
            if (
                spotlight.show_after_milestone
                and spotlight.show_after_milestone not in state.completed_milestones
            ):
                continue
            to_show.append(spotlight)

        return to_show

    def record_spotlight_shown(self, user_id: str, spotlight_id: str) -> None:
        counts = self._spotlight_show_counts.setdefault(user_id, {})
        counts[spotlight_id] = counts.get(spotlight_id, 0) + 1

    # ──────────────────────────────────────────────
    # Analytics
    # ──────────────────────────────────────────────

    def get_funnel_analytics(self) -> Dict[str, Any]:
        """Aggregate onboarding funnel statistics"""
        if not self._user_states:
            return {"total_users": 0}

        total = len(self._user_states)
        completed_any = sum(1 for s in self._user_states.values() if s.completed_milestones)
        avg_points = sum(s.total_points for s in self._user_states.values()) / total
        persona_dist: Dict[str, int] = {}
        for state in self._user_states.values():
            persona_dist[state.persona.value] = persona_dist.get(state.persona.value, 0) + 1

        milestone_completion: Dict[str, int] = {}
        for state in self._user_states.values():
            for mid in state.completed_milestones:
                milestone_completion[mid] = milestone_completion.get(mid, 0) + 1

        return {
            "total_users": total,
            "started_any_milestone": completed_any,
            "activation_rate_pct": round((completed_any / total) * 100, 1),
            "avg_points_per_user": round(avg_points, 1),
            "persona_distribution": persona_dist,
            "milestone_completion_counts": milestone_completion,
        }

    def _track_event(self, user_id: str, event: str, properties: Dict[str, Any]) -> None:
        self._analytics.append({
            "user_id": user_id,
            "event": event,
            "properties": properties,
            "timestamp": time.time(),
        })
        if len(self._analytics) > 10000:
            self._analytics = self._analytics[-10000:]

    def _get_newly_available(self, state: UserOnboardingState) -> List[OnboardingMilestone]:
        """Find milestones whose prerequisites were just satisfied"""
        return [
            m for m in self._milestones.values()
            if m.milestone_id not in state.completed_milestones
            and m.milestone_id not in state.skipped_milestones
            and (m.persona is None or m.persona == state.persona)
            and all(p in state.completed_milestones for p in m.prerequisites)
        ]

    # ──────────────────────────────────────────────
    # Default Content
    # ──────────────────────────────────────────────

    def _setup_default_milestones(self) -> None:
        defaults = [
            OnboardingMilestone(
                milestone_id="complete_profile",
                title="Complete Your Profile",
                description="Add your name and role to personalize your experience",
                trigger=TriggerType.MANUAL,
                reward_points=10,
                estimated_minutes=2,
            ),
            OnboardingMilestone(
                milestone_id="create_api_key",
                title="Generate Your API Key",
                description="Create an API key to start integrating CognitionOS",
                prerequisites=["complete_profile"],
                trigger=TriggerType.FEATURE_USE,
                trigger_value="api_keys",
                reward_points=20,
                cta_action="/api-keys/new",
                estimated_minutes=1,
            ),
            OnboardingMilestone(
                milestone_id="create_first_workflow",
                title="Create Your First Workflow",
                description="Build an automated workflow using the visual editor or DSL",
                prerequisites=["create_api_key"],
                trigger=TriggerType.FEATURE_USE,
                trigger_value="workflows",
                reward_points=30,
                cta_action="/workflows/new",
                estimated_minutes=5,
                persona=OnboardingPersona.DEVELOPER,
            ),
            OnboardingMilestone(
                milestone_id="run_first_agent",
                title="Run Your First AI Agent",
                description="Execute an autonomous agent to complete a real task",
                prerequisites=["create_first_workflow"],
                trigger=TriggerType.FEATURE_USE,
                trigger_value="agents",
                reward_points=50,
                cta_action="/agents/run",
                estimated_minutes=3,
            ),
            OnboardingMilestone(
                milestone_id="install_plugin",
                title="Install a Plugin",
                description="Extend CognitionOS capabilities with a marketplace plugin",
                prerequisites=["create_api_key"],
                trigger=TriggerType.FEATURE_USE,
                trigger_value="marketplace",
                reward_points=20,
                cta_action="/marketplace",
                is_optional=True,
                estimated_minutes=2,
            ),
            OnboardingMilestone(
                milestone_id="invite_teammate",
                title="Invite a Teammate",
                description="Collaborate with your team on CognitionOS",
                prerequisites=["complete_profile"],
                trigger=TriggerType.MANUAL,
                reward_points=25,
                cta_action="/settings/team",
                is_optional=True,
                estimated_minutes=1,
            ),
        ]
        for m in defaults:
            self._milestones[m.milestone_id] = m

    def _setup_default_spotlights(self) -> None:
        defaults = [
            FeatureSpotlight(
                spotlight_id="workflow_builder_intro",
                feature_name="workflow_builder",
                headline="Build Workflows Visually",
                body="Drag-and-drop steps, set conditions, and automate complex tasks without writing code.",
                target_selector="#workflow-builder-btn",
                position="bottom",
                cta_text="Open Builder",
                cta_url="/workflows/new",
                show_after_milestone="create_api_key",
                max_shows=2,
            ),
            FeatureSpotlight(
                spotlight_id="agent_run_intro",
                feature_name="autonomous_agent",
                headline="Autonomous AI Agents",
                body="Let AI agents break down complex tasks, use tools, and deliver results automatically.",
                target_selector="#agent-run-btn",
                position="right",
                cta_text="Try an Agent",
                cta_url="/agents/run",
                show_after_milestone="create_first_workflow",
                max_shows=3,
            ),
            FeatureSpotlight(
                spotlight_id="analytics_intro",
                feature_name="analytics",
                headline="Track Usage & Performance",
                body="Monitor agent executions, workflow runs, and cost in real-time with our analytics dashboard.",
                target_selector="#analytics-link",
                position="left",
                cta_text="View Analytics",
                cta_url="/analytics",
                show_after_milestone="run_first_agent",
                max_shows=2,
            ),
        ]
        for s in defaults:
            self._spotlights[s.spotlight_id] = s


# Module-level singleton
_onboarding_engine: Optional[OnboardingEngine] = None


def get_onboarding_engine() -> OnboardingEngine:
    global _onboarding_engine
    if _onboarding_engine is None:
        _onboarding_engine = OnboardingEngine()
    return _onboarding_engine
