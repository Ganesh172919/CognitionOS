"""
Analytics Aggregation Engine — CognitionOS Data Layer

Real-time analytics aggregation for platform observability:
- Time-series metric collection
- Sliding window aggregations
- Dashboard data feeds
- Revenue and usage analytics
- Agent performance tracking
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    RATE = "rate"


class AggregationWindow(str, Enum):
    MINUTE = "1m"
    FIVE_MINUTES = "5m"
    HOUR = "1h"
    DAY = "1d"

    @property
    def seconds(self) -> float:
        return {"1m": 60, "5m": 300, "1h": 3600, "1d": 86400}[self.value]


@dataclass
class MetricPoint:
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class AggregatedMetric:
    name: str
    window: AggregationWindow
    count: int = 0
    total: float = 0
    min_val: float = float("inf")
    max_val: float = float("-inf")
    last_val: float = 0
    window_start: float = 0
    window_end: float = 0
    tags: Dict[str, str] = field(default_factory=dict)

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "window": self.window.value,
            "count": self.count, "avg": round(self.avg, 3),
            "min": round(self.min_val, 3) if self.min_val != float("inf") else 0,
            "max": round(self.max_val, 3) if self.max_val != float("-inf") else 0,
            "total": round(self.total, 3), "last": round(self.last_val, 3),
        }


class MetricBuffer:
    """Fixed-size circular buffer for metric points."""

    def __init__(self, max_size: int = 10000):
        self._buffer: Deque[MetricPoint] = deque(maxlen=max_size)

    def add(self, point: MetricPoint):
        self._buffer.append(point)

    def get_window(self, start: float, end: float) -> List[MetricPoint]:
        return [p for p in self._buffer if start <= p.timestamp <= end]

    def __len__(self) -> int:
        return len(self._buffer)


class AnalyticsAggregator:
    """
    Real-time analytics aggregation engine.

    Collects metrics from across the platform and provides
    aggregated time-series data for dashboards and monitoring.
    """

    def __init__(self, *, buffer_size: int = 50000, flush_interval: float = 60.0):
        self._buffers: Dict[str, MetricBuffer] = defaultdict(lambda: MetricBuffer(buffer_size))
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._flush_interval = flush_interval
        self._total_points = 0
        self._dashboards: Dict[str, Dict[str, Any]] = {}
        logger.info("AnalyticsAggregator initialized (buffer=%d)", buffer_size)

    # ── Metric Recording ──

    def record(self, name: str, value: float, *,
               metric_type: MetricType = MetricType.GAUGE,
               tags: Optional[Dict[str, str]] = None):
        """Record a single metric data point."""
        point = MetricPoint(name=name, value=value,
                            metric_type=metric_type, tags=tags or {})
        self._buffers[name].add(point)
        self._total_points += 1

        if metric_type == MetricType.COUNTER:
            self._counters[name] += value
        elif metric_type == MetricType.GAUGE:
            self._gauges[name] = value

    def increment(self, name: str, delta: float = 1.0, **tags):
        self.record(name, delta, metric_type=MetricType.COUNTER, tags=tags)

    def gauge(self, name: str, value: float, **tags):
        self.record(name, value, metric_type=MetricType.GAUGE, tags=tags)

    def histogram(self, name: str, value: float, **tags):
        self.record(name, value, metric_type=MetricType.HISTOGRAM, tags=tags)

    # ── Aggregation ──

    def aggregate(self, name: str, window: AggregationWindow) -> AggregatedMetric:
        """Compute aggregated metrics over a time window."""
        now = time.time()
        start = now - window.seconds
        buffer = self._buffers.get(name)
        if not buffer:
            return AggregatedMetric(name=name, window=window)

        points = buffer.get_window(start, now)
        agg = AggregatedMetric(name=name, window=window,
                               window_start=start, window_end=now)
        for p in points:
            agg.count += 1
            agg.total += p.value
            agg.min_val = min(agg.min_val, p.value)
            agg.max_val = max(agg.max_val, p.value)
            agg.last_val = p.value

        return agg

    def get_timeseries(self, name: str, window: AggregationWindow,
                       buckets: int = 30) -> List[Dict[str, Any]]:
        """Get time-series data broken into equal buckets."""
        now = time.time()
        start = now - window.seconds
        bucket_size = window.seconds / buckets
        buffer = self._buffers.get(name)
        if not buffer:
            return []

        points = buffer.get_window(start, now)
        series = []
        for i in range(buckets):
            bucket_start = start + i * bucket_size
            bucket_end = bucket_start + bucket_size
            bucket_points = [p.value for p in points
                             if bucket_start <= p.timestamp < bucket_end]
            series.append({
                "time": round(bucket_start, 1),
                "count": len(bucket_points),
                "avg": round(sum(bucket_points) / len(bucket_points), 3) if bucket_points else 0,
                "max": round(max(bucket_points), 3) if bucket_points else 0,
            })
        return series

    # ── Dashboard Data ──

    def register_dashboard(self, dashboard_id: str, metrics: List[str],
                           window: AggregationWindow = AggregationWindow.HOUR):
        """Register a dashboard configuration."""
        self._dashboards[dashboard_id] = {
            "metrics": metrics, "window": window,
        }

    def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get aggregated data for a registered dashboard."""
        config = self._dashboards.get(dashboard_id)
        if not config:
            return {"error": f"Dashboard '{dashboard_id}' not found"}

        data = {}
        window = config["window"]
        for metric_name in config["metrics"]:
            data[metric_name] = self.aggregate(metric_name, window).to_dict()

        return {"dashboard_id": dashboard_id, "window": window.value, "metrics": data}

    # ── Revenue Analytics ──

    def get_revenue_summary(self, window: AggregationWindow = AggregationWindow.DAY) -> Dict:
        return {
            "mrr": self.aggregate("revenue.mrr", window).to_dict(),
            "new_subscriptions": self.aggregate("revenue.new_subs", window).to_dict(),
            "churn": self.aggregate("revenue.churn", window).to_dict(),
            "token_revenue": self.aggregate("revenue.token_usage", window).to_dict(),
        }

    def get_agent_performance(self, window: AggregationWindow = AggregationWindow.HOUR) -> Dict:
        return {
            "tasks_completed": self.aggregate("agent.tasks_completed", window).to_dict(),
            "avg_latency_ms": self.aggregate("agent.latency_ms", window).to_dict(),
            "quality_score": self.aggregate("agent.quality_score", window).to_dict(),
            "tokens_used": self.aggregate("agent.tokens_used", window).to_dict(),
        }

    # ── Stats ──

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_points_recorded": self._total_points,
            "tracked_metrics": len(self._buffers),
            "active_counters": len(self._counters),
            "active_gauges": len(self._gauges),
            "registered_dashboards": len(self._dashboards),
        }
