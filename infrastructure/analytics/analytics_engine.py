"""
Data Analytics Module — CognitionOS

User behavior and system analytics:
- Event tracking with properties
- User journey mapping
- Funnel analysis
- Engagement scoring
- Feature usage tracking
- Retention metrics
- A/B test result aggregation
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsEvent:
    event_name: str
    user_id: str
    tenant_id: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: str = ""
    source: str = ""


@dataclass
class UserProfile:
    user_id: str
    tenant_id: str = ""
    first_seen: str = ""
    last_seen: str = ""
    event_count: int = 0
    features_used: Set[str] = field(default_factory=set)
    engagement_score: float = 0.0
    tier: str = "free"
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunnelStep:
    name: str
    event_name: str
    count: int = 0
    conversion_rate: float = 0.0


@dataclass
class FunnelResult:
    funnel_name: str
    steps: List[FunnelStep]
    total_entered: int = 0
    total_completed: int = 0
    overall_conversion: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "funnel_name": self.funnel_name,
            "total_entered": self.total_entered,
            "total_completed": self.total_completed,
            "overall_conversion_pct": round(self.overall_conversion, 2),
            "steps": [{"name": s.name, "event": s.event_name,
                       "count": s.count, "conversion_pct": round(s.conversion_rate, 2)}
                      for s in self.steps]}


class AnalyticsEngine:
    """User behavior analytics with funnel analysis and engagement scoring."""

    def __init__(self) -> None:
        self._events: List[AnalyticsEvent] = []
        self._profiles: Dict[str, UserProfile] = {}
        self._feature_usage: Dict[str, int] = defaultdict(int)
        self._funnels: Dict[str, List[str]] = {}  # funnel_name -> [event_names]
        self._ab_results: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list))

    # ---- tracking ----
    def track(self, event: AnalyticsEvent) -> None:
        self._events.append(event)
        self._update_profile(event)
        if "feature" in event.properties:
            self._feature_usage[event.properties["feature"]] += 1

    def _update_profile(self, event: AnalyticsEvent) -> None:
        uid = event.user_id
        if uid not in self._profiles:
            self._profiles[uid] = UserProfile(
                user_id=uid, tenant_id=event.tenant_id,
                first_seen=event.timestamp)
        profile = self._profiles[uid]
        profile.last_seen = event.timestamp
        profile.event_count += 1
        if "feature" in event.properties:
            profile.features_used.add(event.properties["feature"])

    # ---- engagement scoring ----
    def calculate_engagement(self, user_id: str, *, days: int = 30) -> float:
        profile = self._profiles.get(user_id)
        if not profile:
            return 0.0
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        recent_events = [e for e in self._events
                         if e.user_id == user_id and e.timestamp >= cutoff]

        # Score based on: frequency, feature breadth, recency
        frequency_score = min(1.0, len(recent_events) / (days * 3))
        breadth_score = min(1.0, len(profile.features_used) / 10)
        recency_days = 0
        if profile.last_seen:
            last = datetime.fromisoformat(profile.last_seen)
            recency_days = (datetime.now(timezone.utc) - last).days
        recency_score = max(0, 1.0 - recency_days / days)

        score = frequency_score * 0.4 + breadth_score * 0.3 + recency_score * 0.3
        profile.engagement_score = round(score, 3)
        return score

    # ---- funnel analysis ----
    def define_funnel(self, name: str, event_names: List[str]) -> None:
        self._funnels[name] = event_names

    def analyze_funnel(self, name: str, *, days: int = 30) -> FunnelResult:
        event_names = self._funnels.get(name, [])
        if not event_names:
            return FunnelResult(funnel_name=name, steps=[])

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        recent = [e for e in self._events if e.timestamp >= cutoff]

        # Track users through each step
        user_events: Dict[str, Set[str]] = defaultdict(set)
        for e in recent:
            if e.event_name in event_names:
                user_events[e.user_id].add(e.event_name)

        steps: List[FunnelStep] = []
        prev_count = 0
        for i, event_name in enumerate(event_names):
            count = sum(1 for uid, events in user_events.items()
                        if event_name in events and
                        all(event_names[j] in events for j in range(i)))
            rate = (count / prev_count * 100) if prev_count > 0 else 100
            steps.append(FunnelStep(name=event_name, event_name=event_name,
                                     count=count, conversion_rate=rate))
            if i == 0:
                prev_count = count
            else:
                prev_count = count

        total_entered = steps[0].count if steps else 0
        total_completed = steps[-1].count if steps else 0
        overall = (total_completed / total_entered * 100) if total_entered > 0 else 0

        return FunnelResult(funnel_name=name, steps=steps,
                            total_entered=total_entered,
                            total_completed=total_completed,
                            overall_conversion=overall)

    # ---- feature usage ----
    def get_feature_usage(self, *, top_n: int = 20) -> List[Dict[str, Any]]:
        sorted_features = sorted(self._feature_usage.items(), key=lambda x: -x[1])
        return [{"feature": f, "usage_count": c} for f, c in sorted_features[:top_n]]

    # ---- retention ----
    def calculate_retention(self, *, cohort_days: int = 7, periods: int = 8) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        cohorts: Dict[int, Dict[str, Any]] = {}

        for period in range(periods):
            start = now - timedelta(days=(period + 1) * cohort_days)
            end = now - timedelta(days=period * cohort_days)
            start_iso, end_iso = start.isoformat(), end.isoformat()

            cohort_users = set()
            for e in self._events:
                if start_iso <= e.timestamp <= end_iso:
                    cohort_users.add(e.user_id)

            if not cohort_users:
                continue

            retained = set()
            for uid in cohort_users:
                for e in self._events:
                    if e.user_id == uid and e.timestamp > end_iso:
                        retained.add(uid)
                        break

            cohorts[period] = {
                "period": f"Week {period + 1}",
                "cohort_size": len(cohort_users),
                "retained": len(retained),
                "retention_pct": round(len(retained) / len(cohort_users) * 100, 1)}

        return cohorts

    # ---- A/B testing ----
    def record_ab_result(self, test_name: str, variant: str, metric_value: float) -> None:
        self._ab_results[test_name][variant].append(metric_value)

    def get_ab_results(self, test_name: str) -> Dict[str, Any]:
        variants = self._ab_results.get(test_name, {})
        results = {}
        for variant, values in variants.items():
            n = len(values)
            avg = sum(values) / n if n else 0
            results[variant] = {"count": n, "avg": round(avg, 4),
                                "min": round(min(values), 4) if values else 0,
                                "max": round(max(values), 4) if values else 0}
        return {"test_name": test_name, "variants": results}

    # ---- dashboard ----
    def get_dashboard(self, *, days: int = 30) -> Dict[str, Any]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        recent = [e for e in self._events if e.timestamp >= cutoff]
        active_users = len(set(e.user_id for e in recent))
        return {
            "total_events": len(self._events),
            "recent_events": len(recent),
            "total_users": len(self._profiles),
            "active_users": active_users,
            "top_features": self.get_feature_usage(top_n=10),
            "total_funnels": len(self._funnels),
            "ab_tests": len(self._ab_results)}


_engine: AnalyticsEngine | None = None

def get_analytics_engine() -> AnalyticsEngine:
    global _engine
    if not _engine:
        _engine = AnalyticsEngine()
    return _engine
