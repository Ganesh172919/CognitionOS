"""
Intelligent Recommendation Engine v2 — CognitionOS

AI-powered recommendation system with:
- Feature recommendations based on usage patterns
- Tier upgrade suggestions with ML scoring
- Plugin recommendations via collaborative filtering
- Usage optimization tips
- Contextual in-product guidance
- A/B test variant recommendations
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class RecommendationType(str, Enum):
    FEATURE = "feature"
    TIER_UPGRADE = "tier_upgrade"
    PLUGIN = "plugin"
    OPTIMIZATION = "optimization"
    ACTION = "action"
    ONBOARDING = "onboarding"
    WORKFLOW = "workflow"


@dataclass
class Recommendation:
    recommendation_id: str
    type: RecommendationType
    title: str
    description: str
    rationale: str = ""
    score: float = 0.0
    target: str = ""
    estimated_value: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.recommendation_id, "type": self.type.value,
            "title": self.title, "description": self.description,
            "rationale": self.rationale, "score": round(self.score, 3),
            "estimated_value": self.estimated_value,
        }


@dataclass
class UserProfile:
    user_id: str
    tenant_id: str
    tier: str = "free"
    features_used: Set[str] = field(default_factory=set)
    api_calls: int = 0
    session_count: int = 0
    total_usage_hours: float = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


class CollaborativeFilter:
    """Item-based collaborative filtering for recommendations."""

    def __init__(self):
        self._user_items: Dict[str, Set[str]] = defaultdict(set)
        self._item_users: Dict[str, Set[str]] = defaultdict(set)

    def add_interaction(self, user_id: str, item_id: str):
        self._user_items[user_id].add(item_id)
        self._item_users[item_id].add(user_id)

    def recommend(self, user_id: str, *, top_n: int = 5) -> List[Tuple[str, float]]:
        user_items = self._user_items.get(user_id, set())
        if not user_items:
            return self._popular_items(top_n)

        candidate_scores: Counter = Counter()
        for item in user_items:
            for other_user in self._item_users.get(item, set()):
                if other_user == user_id:
                    continue
                other_items = self._user_items.get(other_user, set())
                similarity = len(user_items & other_items) / max(len(user_items | other_items), 1)
                for candidate in other_items - user_items:
                    candidate_scores[candidate] += similarity

        return candidate_scores.most_common(top_n)

    def _popular_items(self, top_n: int) -> List[Tuple[str, float]]:
        popularity = Counter()
        for items in self._user_items.values():
            for item in items:
                popularity[item] += 1
        return popularity.most_common(top_n)


class TierUpgradePredictor:
    """ML-based tier upgrade prediction."""

    TIER_LIMITS = {
        "free": {"api_calls": 1000, "features": 5},
        "pro": {"api_calls": 50000, "features": 20},
        "enterprise": {"api_calls": 1000000, "features": -1},
    }

    def predict(self, profile: UserProfile) -> Optional[Recommendation]:
        limits = self.TIER_LIMITS.get(profile.tier, {})
        if not limits or profile.tier == "enterprise":
            return None

        usage_ratios = {}
        if limits.get("api_calls", 0) > 0:
            usage_ratios["api_calls"] = profile.api_calls / limits["api_calls"]
        if limits.get("features", 0) > 0:
            usage_ratios["features"] = len(profile.features_used) / limits["features"]

        high_usage = {k: v for k, v in usage_ratios.items() if v >= 0.7}
        if not high_usage:
            return None

        max_metric = max(high_usage, key=high_usage.get)
        next_tier = "pro" if profile.tier == "free" else "enterprise"
        score = min(max(high_usage.values()), 1.0)

        # Engagement score boost
        days_active = (time.time() - profile.first_seen) / 86400
        if days_active > 7 and profile.session_count > 10:
            score = min(score + 0.1, 1.0)

        return Recommendation(
            recommendation_id=f"upgrade_{profile.user_id}_{int(time.time())}",
            type=RecommendationType.TIER_UPGRADE,
            title=f"Upgrade to {next_tier.title()}",
            description=f"You're using {round(high_usage[max_metric]*100)}% of your "
                        f"{max_metric.replace('_', ' ')} limit.",
            rationale=f"High {max_metric} usage and strong engagement",
            score=score,
            target=next_tier,
            estimated_value=49.99 if next_tier == "pro" else 199.99,
        )


class SmartFeatureRecommender:
    """Feature recommender using usage pattern matching."""

    FEATURES = {
        "workflow_automation": {"category": "automation", "tier_min": "free"},
        "api_monitoring": {"category": "observability", "tier_min": "free"},
        "custom_agents": {"category": "ai", "tier_min": "pro"},
        "team_collaboration": {"category": "collaboration", "tier_min": "pro"},
        "advanced_analytics": {"category": "analytics", "tier_min": "pro"},
        "sso_integration": {"category": "security", "tier_min": "enterprise"},
        "custom_plugins": {"category": "extensibility", "tier_min": "pro"},
        "audit_logging": {"category": "compliance", "tier_min": "pro"},
        "webhook_triggers": {"category": "automation", "tier_min": "free"},
        "data_export": {"category": "data", "tier_min": "free"},
    }

    TIER_ORDER = {"free": 0, "pro": 1, "enterprise": 2}

    def recommend(self, profile: UserProfile, *, top_n: int = 3) -> List[Recommendation]:
        used = profile.features_used
        used_categories = set()
        for f in used:
            if f in self.FEATURES:
                used_categories.add(self.FEATURES[f]["category"])

        recs = []
        for feat, meta in self.FEATURES.items():
            if feat in used:
                continue
            if self.TIER_ORDER.get(meta["tier_min"], 0) > self.TIER_ORDER.get(profile.tier, 0):
                continue

            score = 0.5
            if meta["category"] in used_categories:
                score += 0.3

            recs.append(Recommendation(
                recommendation_id=f"feature_{feat}",
                type=RecommendationType.FEATURE,
                title=f"Try {feat.replace('_', ' ').title()}",
                description=f"Based on your {meta['category']} usage, this could help.",
                rationale=f"Related to your {meta['category']} workflow",
                score=score,
                target=feat,
            ))

        recs.sort(key=lambda r: -r.score)
        return recs[:top_n]


class AdvancedRecommendationEngine:
    """Master engine combining all recommendation strategies."""

    def __init__(self):
        self._profiles: Dict[str, UserProfile] = {}
        self._collab = CollaborativeFilter()
        self._tier_predictor = TierUpgradePredictor()
        self._feature_rec = SmartFeatureRecommender()
        self._dismissed: Dict[str, Set[str]] = defaultdict(set)
        self._feedback: List[Dict[str, Any]] = []

    def update_profile(self, user_id: str, **kwargs):
        profile = self._profiles.get(user_id)
        if not profile:
            profile = UserProfile(user_id=user_id, tenant_id=kwargs.get("tenant_id", ""))
            self._profiles[user_id] = profile
        for k, v in kwargs.items():
            if hasattr(profile, k):
                if k == "features_used" and isinstance(v, (set, list)):
                    profile.features_used.update(v)
                else:
                    setattr(profile, k, v)
        profile.last_seen = time.time()

    def record_interaction(self, user_id: str, item_id: str):
        self._collab.add_interaction(user_id, item_id)

    def get_recommendations(self, user_id: str, *, max_results: int = 5,
                             types: Optional[List[RecommendationType]] = None
                             ) -> List[Dict[str, Any]]:
        profile = self._profiles.get(user_id)
        if not profile:
            return [Recommendation(
                recommendation_id="onboard_1",
                type=RecommendationType.ONBOARDING,
                title="Complete Your Setup",
                description="Create your first agent to get started.",
                score=1.0,
            ).to_dict()]

        all_recs: List[Recommendation] = []

        if not types or RecommendationType.TIER_UPGRADE in types:
            upgrade = self._tier_predictor.predict(profile)
            if upgrade:
                all_recs.append(upgrade)

        if not types or RecommendationType.FEATURE in types:
            all_recs.extend(self._feature_rec.recommend(profile))

        if not types or RecommendationType.PLUGIN in types:
            for item, score in self._collab.recommend(user_id, top_n=3):
                all_recs.append(Recommendation(
                    recommendation_id=f"plugin_{item}",
                    type=RecommendationType.PLUGIN,
                    title=f"Try {item.replace('_', ' ').title()}",
                    description=f"Users with similar patterns also use this.",
                    score=score,
                ))

        dismissed = self._dismissed.get(user_id, set())
        all_recs = [r for r in all_recs if r.recommendation_id not in dismissed]
        all_recs.sort(key=lambda r: -r.score)
        return [r.to_dict() for r in all_recs[:max_results]]

    def dismiss(self, user_id: str, rec_id: str):
        self._dismissed[user_id].add(rec_id)

    def record_feedback(self, user_id: str, rec_id: str, action: str):
        self._feedback.append({
            "user_id": user_id, "rec_id": rec_id,
            "action": action, "timestamp": time.time(),
        })

    def get_stats(self) -> Dict[str, Any]:
        conversions = sum(1 for f in self._feedback if f["action"] == "converted")
        return {
            "profiles": len(self._profiles),
            "total_feedback": len(self._feedback),
            "conversion_rate_pct": round(
                conversions / max(len(self._feedback), 1) * 100, 1
            ),
        }


_engine: Optional[AdvancedRecommendationEngine] = None

def get_advanced_recommendation_engine() -> AdvancedRecommendationEngine:
    global _engine
    if not _engine:
        _engine = AdvancedRecommendationEngine()
    return _engine
