"""
Data Analytics & Usage Intelligence Engine — CognitionOS

Real-time analytics platform providing:
- Event tracking and funnel analysis
- User behavior analytics
- Feature usage heatmaps
- Cohort analysis
- Revenue analytics integration
- Real-time dashboards
- Anomaly detection in usage patterns
- Predictive usage forecasting
- API usage analytics
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import statistics
import time
import uuid
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Awaitable, Dict, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)


class EventCategory(str, Enum):
    USER_ACTION = "user_action"
    SYSTEM = "system"
    API_CALL = "api_call"
    FEATURE_USAGE = "feature_usage"
    ERROR = "error"
    BILLING = "billing"
    AGENT = "agent"
    PERFORMANCE = "performance"


@dataclass
class AnalyticsEvent:
    event_id: str
    event_name: str
    category: EventCategory
    timestamp: float = field(default_factory=time.time)
    user_id: str = ""
    tenant_id: str = ""
    session_id: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    value: Optional[float] = None

    @staticmethod
    def create(name: str, category: EventCategory, **kwargs) -> "AnalyticsEvent":
        return AnalyticsEvent(
            event_id=uuid.uuid4().hex[:16], event_name=name,
            category=category, **kwargs,
        )


@dataclass
class FunnelStep:
    name: str
    event_name: str
    count: int = 0
    unique_users: int = 0
    conversion_rate: float = 0.0
    avg_time_to_next_ms: float = 0.0


@dataclass
class CohortMetrics:
    cohort_id: str
    period: str  # "2024-W01", "2024-01", etc.
    size: int = 0
    retention_rates: Dict[int, float] = field(default_factory=dict)  # period -> rate
    avg_revenue: float = 0.0
    active_users: int = 0
    churn_rate: float = 0.0


@dataclass
class UsageHeatmapEntry:
    feature: str
    hour: int
    day_of_week: int
    count: int = 0
    unique_users: int = 0


class TimeSeriesAggregator:
    """Aggregate events into time-series data."""

    def __init__(self):
        self._buckets: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def add(self, metric_name: str, value: float, timestamp: float):
        bucket_key = self._bucket_key(timestamp)
        self._buckets[metric_name][bucket_key] += value

    def get_series(self, metric_name: str, *,
                    last_hours: int = 24,
                    granularity: str = "hour") -> List[Dict[str, Any]]:
        series = self._buckets.get(metric_name, {})
        now = time.time()
        cutoff = now - (last_hours * 3600)

        filtered = []
        for key, value in sorted(series.items()):
            ts = self._key_to_timestamp(key)
            if ts >= cutoff:
                filtered.append({
                    "timestamp": key,
                    "value": round(value, 3),
                })
        return filtered

    def _bucket_key(self, ts: float, granularity: str = "hour") -> str:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        if granularity == "minute":
            return dt.strftime("%Y-%m-%d %H:%M")
        elif granularity == "hour":
            return dt.strftime("%Y-%m-%d %H:00")
        elif granularity == "day":
            return dt.strftime("%Y-%m-%d")
        return dt.strftime("%Y-%m-%d %H:00")

    def _key_to_timestamp(self, key: str) -> float:
        try:
            dt = datetime.strptime(key, "%Y-%m-%d %H:%M")
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            try:
                dt = datetime.strptime(key, "%Y-%m-%d %H:00")
                return dt.replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                return 0


class AnomalyDetector:
    """Detect anomalies in usage patterns using z-score analysis."""

    def __init__(self, *, window_size: int = 168,  # 1 week of hourly data
                 z_threshold: float = 3.0):
        self._window_size = window_size
        self._z_threshold = z_threshold
        self._history: Dict[str, List[float]] = defaultdict(list)

    def add_observation(self, metric_name: str, value: float):
        history = self._history[metric_name]
        history.append(value)
        if len(history) > self._window_size * 2:
            self._history[metric_name] = history[-self._window_size:]

    def check(self, metric_name: str, value: float) -> Optional[Dict[str, Any]]:
        """Check if value is anomalous. Returns anomaly info or None."""
        history = self._history.get(metric_name, [])
        if len(history) < 10:
            return None

        mean = statistics.mean(history)
        std = statistics.stdev(history) if len(history) > 1 else 0
        if std == 0:
            return None

        z_score = (value - mean) / std

        if abs(z_score) > self._z_threshold:
            return {
                "metric": metric_name,
                "value": value,
                "mean": round(mean, 2),
                "std_dev": round(std, 2),
                "z_score": round(z_score, 2),
                "direction": "high" if z_score > 0 else "low",
                "severity": "critical" if abs(z_score) > 5 else "warning",
            }
        return None

    def get_all_anomalies(self) -> List[Dict[str, Any]]:
        anomalies = []
        for name, history in self._history.items():
            if history:
                result = self.check(name, history[-1])
                if result:
                    anomalies.append(result)
        return anomalies


class UsageAnalyticsEngine:
    """
    Core analytics engine for tracking, analyzing, and visualizing
    platform usage patterns.
    """

    def __init__(self, *, max_events: int = 1_000_000,
                 anomaly_detection: bool = True):
        self._events: List[AnalyticsEvent] = []
        self._max_events = max_events
        self._time_series = TimeSeriesAggregator()
        self._anomaly_detector = AnomalyDetector() if anomaly_detection else None
        self._counters: Dict[str, int] = defaultdict(int)
        self._user_sessions: Dict[str, List[str]] = defaultdict(list)
        self._feature_usage: Dict[str, Counter] = defaultdict(Counter)
        self._api_usage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._revenue_events: List[Dict[str, Any]] = []

    # ── Event Tracking ──

    async def track(self, event: AnalyticsEvent):
        """Track an analytics event."""
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events // 2:]

        # Update counters
        self._counters[event.event_name] += 1
        self._counters[f"category:{event.category.value}"] += 1

        # Time series
        self._time_series.add(event.event_name, 1, event.timestamp)
        if event.value:
            self._time_series.add(f"{event.event_name}_value", event.value, event.timestamp)

        # User sessions
        if event.user_id and event.session_id:
            if event.session_id not in self._user_sessions[event.user_id]:
                self._user_sessions[event.user_id].append(event.session_id)

        # Feature usage
        if event.category == EventCategory.FEATURE_USAGE:
            self._feature_usage[event.tenant_id][event.event_name] += 1

        # API usage
        if event.category == EventCategory.API_CALL:
            endpoint = event.properties.get("endpoint", "unknown")
            self._api_usage[event.tenant_id][endpoint] += 1

        # Revenue tracking
        if event.category == EventCategory.BILLING and event.value:
            self._revenue_events.append({
                "timestamp": event.timestamp,
                "tenant_id": event.tenant_id,
                "amount": event.value,
                "type": event.event_name,
            })

        # Anomaly detection
        if self._anomaly_detector:
            self._anomaly_detector.add_observation(event.event_name, 1)

    async def track_batch(self, events: List[AnalyticsEvent]):
        for event in events:
            await self.track(event)

    # ── Funnel Analysis ──

    def analyze_funnel(self, steps: List[str], *,
                        time_window_hours: int = 24,
                        tenant_id: Optional[str] = None) -> List[FunnelStep]:
        """Analyze conversion funnel."""
        cutoff = time.time() - (time_window_hours * 3600)
        relevant_events = [
            e for e in self._events
            if e.timestamp >= cutoff
            and (not tenant_id or e.tenant_id == tenant_id)
        ]

        # Build user journeys
        user_events: Dict[str, List[AnalyticsEvent]] = defaultdict(list)
        for event in relevant_events:
            if event.user_id:
                user_events[event.user_id].append(event)

        funnel_results = []
        users_at_step: Set[str] = set()
        first_step = True

        for step_name in steps:
            step_users: Set[str] = set()
            for user_id, events in user_events.items():
                if first_step or user_id in users_at_step:
                    if any(e.event_name == step_name for e in events):
                        step_users.add(user_id)

            total_count = sum(
                1 for e in relevant_events if e.event_name == step_name
            )
            prev_count = len(users_at_step) if not first_step else len(step_users)
            conversion = (len(step_users) / max(prev_count, 1)) * 100

            funnel_results.append(FunnelStep(
                name=step_name,
                event_name=step_name,
                count=total_count,
                unique_users=len(step_users),
                conversion_rate=round(conversion, 2),
            ))

            users_at_step = step_users
            first_step = False

        return funnel_results

    # ── Feature Usage ──

    def get_feature_usage(self, tenant_id: str, *,
                           top_n: int = 20) -> List[Dict[str, Any]]:
        usage = self._feature_usage.get(tenant_id, Counter())
        return [
            {"feature": name, "count": count}
            for name, count in usage.most_common(top_n)
        ]

    def get_feature_adoption(self, feature_name: str) -> Dict[str, Any]:
        """Get adoption metrics for a specific feature."""
        total_tenants = len(self._feature_usage)
        using_tenants = sum(
            1 for usage in self._feature_usage.values()
            if feature_name in usage
        )
        return {
            "feature": feature_name,
            "total_tenants": total_tenants,
            "using_tenants": using_tenants,
            "adoption_rate_pct": round(
                using_tenants / max(total_tenants, 1) * 100, 1
            ),
        }

    # ── API Usage ──

    def get_api_usage(self, tenant_id: str, *,
                       top_n: int = 20) -> List[Dict[str, Any]]:
        usage = self._api_usage.get(tenant_id, {})
        sorted_usage = sorted(usage.items(), key=lambda x: -x[1])
        return [
            {"endpoint": endpoint, "calls": count}
            for endpoint, count in sorted_usage[:top_n]
        ]

    def get_api_usage_summary(self) -> Dict[str, Any]:
        total_calls = sum(
            sum(v.values()) for v in self._api_usage.values()
        )
        top_endpoints: Counter = Counter()
        for usage in self._api_usage.values():
            top_endpoints.update(usage)

        return {
            "total_api_calls": total_calls,
            "total_tenants": len(self._api_usage),
            "top_endpoints": [
                {"endpoint": ep, "calls": count}
                for ep, count in top_endpoints.most_common(10)
            ],
        }

    # ── Revenue Analytics ──

    def get_revenue_metrics(self, *, days: int = 30) -> Dict[str, Any]:
        cutoff = time.time() - (days * 86400)
        recent = [e for e in self._revenue_events if e["timestamp"] >= cutoff]

        total_revenue = sum(e["amount"] for e in recent)
        by_tenant: Dict[str, float] = defaultdict(float)
        for e in recent:
            by_tenant[e["tenant_id"]] += e["amount"]

        daily_revenue: Dict[str, float] = defaultdict(float)
        for e in recent:
            day = datetime.fromtimestamp(e["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d")
            daily_revenue[day] += e["amount"]

        return {
            "total_revenue": round(total_revenue, 2),
            "period_days": days,
            "avg_daily_revenue": round(total_revenue / max(days, 1), 2),
            "paying_tenants": len(by_tenant),
            "arpu": round(total_revenue / max(len(by_tenant), 1), 2),
            "top_tenants": sorted(
                [{"tenant_id": tid, "revenue": round(rev, 2)}
                 for tid, rev in by_tenant.items()],
                key=lambda x: -x["revenue"]
            )[:10],
            "daily_trend": [
                {"date": date, "revenue": round(rev, 2)}
                for date, rev in sorted(daily_revenue.items())
            ],
        }

    # ── User Analytics ──

    def get_user_engagement(self, *, days: int = 7) -> Dict[str, Any]:
        cutoff = time.time() - (days * 86400)
        recent = [e for e in self._events if e.timestamp >= cutoff and e.user_id]

        unique_users: Set[str] = set()
        daily_active: Dict[str, Set[str]] = defaultdict(set)
        actions_per_user: Dict[str, int] = defaultdict(int)

        for e in recent:
            unique_users.add(e.user_id)
            day = datetime.fromtimestamp(e.timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
            daily_active[day].add(e.user_id)
            actions_per_user[e.user_id] += 1

        avg_actions = (
            sum(actions_per_user.values()) / max(len(actions_per_user), 1)
        )

        return {
            "period_days": days,
            "unique_users": len(unique_users),
            "avg_daily_active": round(
                sum(len(users) for users in daily_active.values()) /
                max(len(daily_active), 1), 1
            ),
            "avg_actions_per_user": round(avg_actions, 1),
            "daily_active_users": [
                {"date": date, "users": len(users)}
                for date, users in sorted(daily_active.items())
            ],
        }

    # ── Time Series ──

    def get_time_series(self, metric: str, **kwargs) -> List[Dict[str, Any]]:
        return self._time_series.get_series(metric, **kwargs)

    # ── Anomalies ──

    def get_anomalies(self) -> List[Dict[str, Any]]:
        if self._anomaly_detector:
            return self._anomaly_detector.get_all_anomalies()
        return []

    # ── Dashboard ──

    def get_dashboard_data(self) -> Dict[str, Any]:
        return {
            "total_events": len(self._events),
            "event_counts": dict(sorted(
                self._counters.items(), key=lambda x: -x[1]
            )[:20]),
            "user_engagement": self.get_user_engagement(days=7),
            "revenue": self.get_revenue_metrics(days=30),
            "api_usage": self.get_api_usage_summary(),
            "anomalies": self.get_anomalies(),
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_events": len(self._events),
            "unique_event_types": len(set(e.event_name for e in self._events)),
            "tenants_tracked": len(self._feature_usage),
            "users_tracked": len(self._user_sessions),
            "revenue_events": len(self._revenue_events),
        }


# ── Singleton ──
_analytics: Optional[UsageAnalyticsEngine] = None


def get_analytics_engine() -> UsageAnalyticsEngine:
    global _analytics
    if not _analytics:
        _analytics = UsageAnalyticsEngine()
    return _analytics
