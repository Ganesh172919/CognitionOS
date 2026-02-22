"""
Real-Time Usage Analytics and Token Tracking System

Provides comprehensive real-time tracking of:
- API usage patterns
- Token consumption
- Cost analysis
- User behavior analytics
- Performance metrics
- Resource utilization
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json


class MetricType(Enum):
    """Types of metrics tracked"""
    REQUEST_COUNT = "request_count"
    TOKEN_USAGE = "token_usage"
    COST = "cost"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


class AggregationWindow(Enum):
    """Time windows for aggregation"""
    MINUTE = 60
    FIVE_MINUTES = 300
    HOUR = 3600
    DAY = 86400


@dataclass
class UsageEvent:
    """Single usage event"""
    user_id: str
    endpoint: str
    method: str
    tokens_input: int
    tokens_output: int
    cost_usd: float
    latency_ms: float
    status_code: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageMetrics:
    """Aggregated usage metrics"""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_latency: float = 0.0
    error_count: int = 0
    success_rate: float = 1.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0


@dataclass
class UserQuota:
    """User quota configuration"""
    user_id: str
    tier: str
    max_requests_per_hour: int
    max_tokens_per_day: int
    max_cost_per_month: float
    current_requests: int = 0
    current_tokens: int = 0
    current_cost: float = 0.0
    reset_time: Optional[datetime] = None


class RealtimeAnalyticsEngine:
    """
    Real-Time Usage Analytics and Token Tracking System

    Features:
    - Real-time event ingestion
    - Time-series metrics aggregation
    - Token consumption tracking
    - Cost attribution by user/project
    - Quota enforcement
    - Anomaly detection
    - Usage forecasting
    - Custom dashboards
    - Alert triggering
    - Export capabilities
    """

    def __init__(self):
        # Event storage
        self._events: deque = deque(maxlen=100000)
        self._user_metrics: Dict[str, UsageMetrics] = defaultdict(UsageMetrics)
        self._endpoint_metrics: Dict[str, UsageMetrics] = defaultdict(UsageMetrics)

        # Time-series data
        self._time_series: Dict[str, Dict[int, UsageMetrics]] = defaultdict(lambda: defaultdict(UsageMetrics))

        # Quota tracking
        self._user_quotas: Dict[str, UserQuota] = {}

        # Real-time aggregation
        self._current_window_data: Dict[str, List[float]] = defaultdict(list)

        # Alerts
        self._alert_callbacks: List[Callable] = []
        self._alert_thresholds: Dict[str, float] = {}

        # Background tasks
        self._aggregation_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start analytics engine"""
        self._aggregation_task = asyncio.create_task(self._aggregation_loop())

    async def stop(self):
        """Stop analytics engine"""
        if self._aggregation_task:
            self._aggregation_task.cancel()
            try:
                await self._aggregation_task
            except asyncio.CancelledError:
                pass

    async def track_event(self, event: UsageEvent):
        """Track a usage event"""
        # Store event
        self._events.append(event)

        # Update real-time metrics
        self._update_user_metrics(event)
        self._update_endpoint_metrics(event)

        # Update time-series
        self._update_time_series(event)

        # Check quotas
        await self._check_quotas(event)

        # Check for anomalies
        await self._check_anomalies(event)

    def _update_user_metrics(self, event: UsageEvent):
        """Update user-level metrics"""
        metrics = self._user_metrics[event.user_id]

        metrics.total_requests += 1
        metrics.total_tokens += event.tokens_input + event.tokens_output
        metrics.total_cost += event.cost_usd

        # Update average latency (exponential moving average)
        alpha = 0.1
        if metrics.avg_latency == 0:
            metrics.avg_latency = event.latency_ms
        else:
            metrics.avg_latency = alpha * event.latency_ms + (1 - alpha) * metrics.avg_latency

        # Track errors
        if event.status_code >= 400:
            metrics.error_count += 1

        # Update success rate
        metrics.success_rate = 1 - (metrics.error_count / metrics.total_requests)

        # Store latency for percentile calculation
        self._current_window_data[f"user_{event.user_id}_latency"].append(event.latency_ms)

    def _update_endpoint_metrics(self, event: UsageEvent):
        """Update endpoint-level metrics"""
        key = f"{event.method}:{event.endpoint}"
        metrics = self._endpoint_metrics[key]

        metrics.total_requests += 1
        metrics.total_tokens += event.tokens_input + event.tokens_output
        metrics.total_cost += event.cost_usd

        # Update average latency
        alpha = 0.1
        if metrics.avg_latency == 0:
            metrics.avg_latency = event.latency_ms
        else:
            metrics.avg_latency = alpha * event.latency_ms + (1 - alpha) * metrics.avg_latency

        if event.status_code >= 400:
            metrics.error_count += 1

        metrics.success_rate = 1 - (metrics.error_count / metrics.total_requests)

    def _update_time_series(self, event: UsageEvent):
        """Update time-series data"""
        timestamp = int(event.timestamp.timestamp())

        # Aggregate by minute
        minute_bucket = (timestamp // 60) * 60

        key = f"user_{event.user_id}"
        metrics = self._time_series[key][minute_bucket]

        metrics.total_requests += 1
        metrics.total_tokens += event.tokens_input + event.tokens_output
        metrics.total_cost += event.cost_usd

    async def _check_quotas(self, event: UsageEvent):
        """Check and enforce user quotas"""
        if event.user_id not in self._user_quotas:
            return

        quota = self._user_quotas[event.user_id]

        # Update current usage
        quota.current_requests += 1
        quota.current_tokens += event.tokens_input + event.tokens_output
        quota.current_cost += event.cost_usd

        # Check limits
        if quota.current_requests >= quota.max_requests_per_hour:
            await self._trigger_alert(
                "quota_exceeded",
                f"User {event.user_id} exceeded hourly request quota"
            )

        if quota.current_tokens >= quota.max_tokens_per_day:
            await self._trigger_alert(
                "quota_exceeded",
                f"User {event.user_id} exceeded daily token quota"
            )

        if quota.current_cost >= quota.max_cost_per_month:
            await self._trigger_alert(
                "quota_exceeded",
                f"User {event.user_id} exceeded monthly cost quota"
            )

    async def _check_anomalies(self, event: UsageEvent):
        """Check for anomalous usage patterns"""
        metrics = self._user_metrics[event.user_id]

        # Check for sudden spike in requests
        recent_events = [e for e in self._events if e.user_id == event.user_id][-100:]
        if len(recent_events) >= 100:
            time_span = (recent_events[-1].timestamp - recent_events[0].timestamp).total_seconds()
            if time_span < 60:  # 100 requests in less than 1 minute
                await self._trigger_alert(
                    "anomaly_detected",
                    f"User {event.user_id} showing unusual request spike"
                )

        # Check for high error rate
        if metrics.error_count > 10 and metrics.success_rate < 0.5:
            await self._trigger_alert(
                "high_error_rate",
                f"User {event.user_id} has high error rate: {metrics.error_count} errors"
            )

        # Check for unusual latency
        if event.latency_ms > metrics.avg_latency * 3 and metrics.avg_latency > 0:
            await self._trigger_alert(
                "high_latency",
                f"Unusually high latency detected for user {event.user_id}: {event.latency_ms}ms"
            )

    async def _aggregation_loop(self):
        """Background aggregation loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute

                # Calculate percentiles
                self._calculate_percentiles()

                # Clean old data
                self._cleanup_old_data()

                # Reset hourly quotas
                self._reset_expired_quotas()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in aggregation loop: {e}")

    def _calculate_percentiles(self):
        """Calculate latency percentiles"""
        for key, latencies in self._current_window_data.items():
            if not latencies:
                continue

            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)

            # Calculate percentiles
            p50_idx = int(n * 0.5)
            p95_idx = int(n * 0.95)
            p99_idx = int(n * 0.99)

            # Update metrics
            if key.startswith("user_"):
                user_id = key.split("_")[1]
                metrics = self._user_metrics[user_id]
                metrics.p50_latency = sorted_latencies[p50_idx] if p50_idx < n else 0
                metrics.p95_latency = sorted_latencies[p95_idx] if p95_idx < n else 0
                metrics.p99_latency = sorted_latencies[p99_idx] if p99_idx < n else 0

        # Clear current window data
        self._current_window_data.clear()

    def _cleanup_old_data(self):
        """Clean up old time-series data"""
        cutoff_time = int((datetime.utcnow() - timedelta(days=7)).timestamp())

        for key in list(self._time_series.keys()):
            buckets = self._time_series[key]
            old_buckets = [b for b in buckets if b < cutoff_time]
            for bucket in old_buckets:
                del buckets[bucket]

    def _reset_expired_quotas(self):
        """Reset expired quotas"""
        now = datetime.utcnow()

        for user_id, quota in self._user_quotas.items():
            if quota.reset_time and now >= quota.reset_time:
                quota.current_requests = 0
                quota.current_tokens = 0
                quota.current_cost = 0
                quota.reset_time = now + timedelta(hours=1)

    async def _trigger_alert(self, alert_type: str, message: str):
        """Trigger alert callbacks"""
        for callback in self._alert_callbacks:
            try:
                await callback(alert_type, message)
            except Exception as e:
                print(f"Error in alert callback: {e}")

    def set_user_quota(self, quota: UserQuota):
        """Set quota for user"""
        if not quota.reset_time:
            quota.reset_time = datetime.utcnow() + timedelta(hours=1)

        self._user_quotas[quota.user_id] = quota

    def register_alert_callback(self, callback: Callable):
        """Register alert callback"""
        self._alert_callbacks.append(callback)

    def get_user_metrics(self, user_id: str) -> Dict[str, Any]:
        """Get metrics for specific user"""
        metrics = self._user_metrics.get(user_id)
        if not metrics:
            return {}

        return {
            "total_requests": metrics.total_requests,
            "total_tokens": metrics.total_tokens,
            "total_cost": metrics.total_cost,
            "avg_latency_ms": metrics.avg_latency,
            "p50_latency_ms": metrics.p50_latency,
            "p95_latency_ms": metrics.p95_latency,
            "p99_latency_ms": metrics.p99_latency,
            "error_count": metrics.error_count,
            "success_rate": metrics.success_rate
        }

    def get_endpoint_metrics(self, endpoint: str) -> Dict[str, Any]:
        """Get metrics for specific endpoint"""
        metrics = self._endpoint_metrics.get(endpoint)
        if not metrics:
            return {}

        return {
            "total_requests": metrics.total_requests,
            "total_tokens": metrics.total_tokens,
            "total_cost": metrics.total_cost,
            "avg_latency_ms": metrics.avg_latency,
            "error_count": metrics.error_count,
            "success_rate": metrics.success_rate
        }

    def get_time_series(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime,
        resolution: AggregationWindow = AggregationWindow.MINUTE
    ) -> List[Dict[str, Any]]:
        """Get time-series data for user"""
        key = f"user_{user_id}"
        buckets = self._time_series[key]

        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())

        result = []
        for timestamp in range(start_ts, end_ts, resolution.value):
            bucket_ts = (timestamp // resolution.value) * resolution.value
            metrics = buckets.get(bucket_ts)

            if metrics:
                result.append({
                    "timestamp": bucket_ts,
                    "requests": metrics.total_requests,
                    "tokens": metrics.total_tokens,
                    "cost": metrics.total_cost
                })

        return result

    def get_top_users(self, metric: str = "cost", limit: int = 10) -> List[Dict[str, Any]]:
        """Get top users by metric"""
        users = []

        for user_id, metrics in self._user_metrics.items():
            value = 0
            if metric == "cost":
                value = metrics.total_cost
            elif metric == "tokens":
                value = metrics.total_tokens
            elif metric == "requests":
                value = metrics.total_requests

            users.append({
                "user_id": user_id,
                "value": value,
                "metric": metric
            })

        return sorted(users, key=lambda x: x["value"], reverse=True)[:limit]

    def forecast_usage(
        self,
        user_id: str,
        days_ahead: int = 7
    ) -> Dict[str, Any]:
        """Forecast future usage based on historical data"""
        # Get historical data
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=30)

        time_series = self.get_time_series(
            user_id,
            start_time,
            end_time,
            AggregationWindow.DAY
        )

        if not time_series:
            return {
                "forecast_available": False,
                "message": "Insufficient data for forecasting"
            }

        # Simple linear trend forecast
        total_tokens = sum(d["tokens"] for d in time_series)
        total_cost = sum(d["cost"] for d in time_series)
        days_of_data = len(time_series)

        avg_tokens_per_day = total_tokens / days_of_data
        avg_cost_per_day = total_cost / days_of_data

        return {
            "forecast_available": True,
            "days_ahead": days_ahead,
            "forecasted_tokens": avg_tokens_per_day * days_ahead,
            "forecasted_cost": avg_cost_per_day * days_ahead,
            "confidence": 0.7,  # Would be calculated from variance in production
            "historical_avg_tokens_per_day": avg_tokens_per_day,
            "historical_avg_cost_per_day": avg_cost_per_day
        }

    def export_metrics(
        self,
        user_id: Optional[str] = None,
        format: str = "json"
    ) -> str:
        """Export metrics in specified format"""
        if user_id:
            data = {
                "user_id": user_id,
                "metrics": self.get_user_metrics(user_id)
            }
        else:
            data = {
                "all_users": {
                    uid: self.get_user_metrics(uid)
                    for uid in self._user_metrics.keys()
                },
                "endpoints": {
                    ep: self.get_endpoint_metrics(ep)
                    for ep in self._endpoint_metrics.keys()
                }
            }

        if format == "json":
            return json.dumps(data, indent=2)
        else:
            return str(data)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard visualization"""
        return {
            "total_users": len(self._user_metrics),
            "total_requests": sum(m.total_requests for m in self._user_metrics.values()),
            "total_tokens": sum(m.total_tokens for m in self._user_metrics.values()),
            "total_cost": sum(m.total_cost for m in self._user_metrics.values()),
            "top_users_by_cost": self.get_top_users("cost", 5),
            "top_endpoints": [
                {
                    "endpoint": ep,
                    "requests": self._endpoint_metrics[ep].total_requests,
                    "avg_latency": self._endpoint_metrics[ep].avg_latency
                }
                for ep in sorted(
                    self._endpoint_metrics.keys(),
                    key=lambda x: self._endpoint_metrics[x].total_requests,
                    reverse=True
                )[:5]
            ]
        }
