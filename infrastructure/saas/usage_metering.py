"""
Usage Metering and Tracking System

Comprehensive usage tracking for API calls, workflows, compute time, and tokens.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics to track"""
    API_CALL = "api_call"
    WORKFLOW_EXECUTION = "workflow_execution"
    AGENT_INVOCATION = "agent_invocation"
    TOKEN_USAGE = "token_usage"
    COMPUTE_TIME = "compute_time"
    STORAGE_USAGE = "storage_usage"
    BANDWIDTH = "bandwidth"
    PLUGIN_EXECUTION = "plugin_execution"


class AggregationPeriod(str, Enum):
    """Time periods for aggregation"""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


@dataclass
class UsageMetric:
    """Individual usage metric record"""
    metric_type: MetricType
    tenant_id: str
    user_id: Optional[str]
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Cost tracking
    cost_usd: Optional[float] = None

    # Context
    resource_id: Optional[str] = None
    endpoint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "metric_type": self.metric_type.value,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "cost_usd": self.cost_usd,
            "resource_id": self.resource_id,
            "endpoint": self.endpoint,
            "metadata": self.metadata
        }


@dataclass
class UsageSummary:
    """Aggregated usage summary"""
    tenant_id: str
    period_start: datetime
    period_end: datetime
    metrics: Dict[MetricType, float]
    total_cost_usd: float
    breakdown: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tenant_id": self.tenant_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "metrics": {k.value: v for k, v in self.metrics.items()},
            "total_cost_usd": self.total_cost_usd,
            "breakdown": self.breakdown
        }


class UsageTracker:
    """
    Track and record usage metrics

    Handles real-time tracking of all usage metrics for billing and analytics.
    """

    def __init__(self, storage_backend: Optional[Any] = None):
        self.storage = storage_backend
        self._buffer: List[UsageMetric] = []
        self._buffer_size = 100
        self._flush_interval = timedelta(seconds=60)
        self._last_flush = datetime.utcnow()

    def record(
        self,
        metric_type: MetricType,
        tenant_id: str,
        value: float,
        user_id: Optional[str] = None,
        cost_usd: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UsageMetric:
        """
        Record a usage metric

        Args:
            metric_type: Type of metric
            tenant_id: Tenant identifier
            value: Metric value
            user_id: Optional user identifier
            cost_usd: Optional cost in USD
            metadata: Optional additional metadata

        Returns:
            Created usage metric
        """
        metric = UsageMetric(
            metric_type=metric_type,
            tenant_id=tenant_id,
            user_id=user_id,
            value=value,
            timestamp=datetime.utcnow(),
            cost_usd=cost_usd,
            metadata=metadata or {}
        )

        self._buffer.append(metric)

        # Auto-flush if buffer is full or time elapsed
        if (len(self._buffer) >= self._buffer_size or
            datetime.utcnow() - self._last_flush >= self._flush_interval):
            self._flush()

        return metric

    def _flush(self):
        """Flush buffered metrics to storage"""
        if not self._buffer:
            return

        if self.storage:
            try:
                self.storage.bulk_insert(self._buffer)
                logger.info(f"Flushed {len(self._buffer)} metrics to storage")
            except Exception as e:
                logger.error(f"Failed to flush metrics: {e}")

        self._buffer.clear()
        self._last_flush = datetime.utcnow()

    def get_usage(
        self,
        tenant_id: str,
        start_time: datetime,
        end_time: datetime,
        metric_types: Optional[List[MetricType]] = None
    ) -> List[UsageMetric]:
        """
        Retrieve usage metrics for a time period

        Args:
            tenant_id: Tenant identifier
            start_time: Start of period
            end_time: End of period
            metric_types: Optional filter by metric types

        Returns:
            List of usage metrics
        """
        if not self.storage:
            # Return from buffer if no storage
            return [
                m for m in self._buffer
                if (m.tenant_id == tenant_id and
                    start_time <= m.timestamp <= end_time and
                    (not metric_types or m.metric_type in metric_types))
            ]

        return self.storage.query(
            tenant_id=tenant_id,
            start_time=start_time,
            end_time=end_time,
            metric_types=metric_types
        )


class TokenTracker:
    """
    Specialized tracker for LLM token usage

    Tracks input/output tokens, model usage, and associated costs.
    """

    # Pricing per 1K tokens (example rates)
    TOKEN_PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125}
    }

    def __init__(self, usage_tracker: UsageTracker):
        self.usage_tracker = usage_tracker

    def record_completion(
        self,
        tenant_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Record LLM completion usage

        Args:
            tenant_id: Tenant identifier
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            user_id: Optional user identifier
            metadata: Optional metadata

        Returns:
            Cost in USD
        """
        total_tokens = input_tokens + output_tokens

        # Calculate cost
        pricing = self.TOKEN_PRICING.get(model, {"input": 0.001, "output": 0.002})
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        total_cost = input_cost + output_cost

        # Record metric
        metric_metadata = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            **(metadata or {})
        }

        self.usage_tracker.record(
            metric_type=MetricType.TOKEN_USAGE,
            tenant_id=tenant_id,
            value=total_tokens,
            user_id=user_id,
            cost_usd=total_cost,
            metadata=metric_metadata
        )

        return total_cost

    def get_token_summary(
        self,
        tenant_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get token usage summary"""
        metrics = self.usage_tracker.get_usage(
            tenant_id=tenant_id,
            start_time=start_time,
            end_time=end_time,
            metric_types=[MetricType.TOKEN_USAGE]
        )

        summary = {
            "total_tokens": 0,
            "total_cost": 0.0,
            "by_model": defaultdict(lambda: {
                "tokens": 0,
                "cost": 0.0,
                "calls": 0
            })
        }

        for metric in metrics:
            summary["total_tokens"] += metric.value
            summary["total_cost"] += metric.cost_usd or 0.0

            model = metric.metadata.get("model", "unknown")
            summary["by_model"][model]["tokens"] += metric.value
            summary["by_model"][model]["cost"] += metric.cost_usd or 0.0
            summary["by_model"][model]["calls"] += 1

        summary["by_model"] = dict(summary["by_model"])
        return summary


class UsageAggregator:
    """
    Aggregate usage metrics for reporting and billing

    Provides various aggregations over time periods.
    """

    def __init__(self, usage_tracker: UsageTracker):
        self.usage_tracker = usage_tracker

    def aggregate(
        self,
        tenant_id: str,
        start_time: datetime,
        end_time: datetime,
        period: AggregationPeriod = AggregationPeriod.DAY
    ) -> List[UsageSummary]:
        """
        Aggregate usage over time periods

        Args:
            tenant_id: Tenant identifier
            start_time: Start of aggregation period
            end_time: End of aggregation period
            period: Aggregation granularity

        Returns:
            List of usage summaries per period
        """
        metrics = self.usage_tracker.get_usage(
            tenant_id=tenant_id,
            start_time=start_time,
            end_time=end_time
        )

        # Group by period
        period_buckets = defaultdict(lambda: {
            "metrics": defaultdict(float),
            "costs": 0.0,
            "breakdown": defaultdict(float)
        })

        for metric in metrics:
            bucket_key = self._get_bucket_key(metric.timestamp, period)
            period_buckets[bucket_key]["metrics"][metric.metric_type] += metric.value
            period_buckets[bucket_key]["costs"] += metric.cost_usd or 0.0

            # Track breakdown by endpoint
            if metric.endpoint:
                period_buckets[bucket_key]["breakdown"][metric.endpoint] += metric.value

        # Convert to summaries
        summaries = []
        for bucket_key, data in sorted(period_buckets.items()):
            period_start, period_end = self._get_period_bounds(bucket_key, period)
            summaries.append(UsageSummary(
                tenant_id=tenant_id,
                period_start=period_start,
                period_end=period_end,
                metrics=dict(data["metrics"]),
                total_cost_usd=data["costs"],
                breakdown=dict(data["breakdown"])
            ))

        return summaries

    def _get_bucket_key(self, timestamp: datetime, period: AggregationPeriod) -> str:
        """Get bucket key for timestamp and period"""
        if period == AggregationPeriod.MINUTE:
            return timestamp.strftime("%Y-%m-%d %H:%M")
        elif period == AggregationPeriod.HOUR:
            return timestamp.strftime("%Y-%m-%d %H:00")
        elif period == AggregationPeriod.DAY:
            return timestamp.strftime("%Y-%m-%d")
        elif period == AggregationPeriod.WEEK:
            return timestamp.strftime("%Y-W%W")
        elif period == AggregationPeriod.MONTH:
            return timestamp.strftime("%Y-%m")
        elif period == AggregationPeriod.YEAR:
            return timestamp.strftime("%Y")
        return timestamp.isoformat()

    def _get_period_bounds(
        self,
        bucket_key: str,
        period: AggregationPeriod
    ) -> tuple[datetime, datetime]:
        """Get start and end times for a bucket"""
        # Simplified implementation - would need proper parsing
        if period == AggregationPeriod.DAY:
            start = datetime.strptime(bucket_key, "%Y-%m-%d")
            end = start + timedelta(days=1)
        elif period == AggregationPeriod.MONTH:
            start = datetime.strptime(bucket_key + "-01", "%Y-%m-%d")
            # Approximate - would need calendar logic
            end = start + timedelta(days=30)
        else:
            start = datetime.utcnow()
            end = start

        return start, end

    def get_monthly_summary(
        self,
        tenant_id: str,
        month: datetime
    ) -> UsageSummary:
        """Get monthly usage summary"""
        start = month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Get last day of month
        if month.month == 12:
            end = start.replace(year=start.year + 1, month=1)
        else:
            end = start.replace(month=start.month + 1)

        summaries = self.aggregate(
            tenant_id=tenant_id,
            start_time=start,
            end_time=end,
            period=AggregationPeriod.MONTH
        )

        return summaries[0] if summaries else UsageSummary(
            tenant_id=tenant_id,
            period_start=start,
            period_end=end,
            metrics={},
            total_cost_usd=0.0
        )

    def detect_anomalies(
        self,
        tenant_id: str,
        lookback_days: int = 30,
        threshold_multiplier: float = 3.0
    ) -> List[Dict[str, Any]]:
        """
        Detect usage anomalies

        Args:
            tenant_id: Tenant to check
            lookback_days: Days to analyze
            threshold_multiplier: Threshold for anomaly detection

        Returns:
            List of detected anomalies
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=lookback_days)

        summaries = self.aggregate(
            tenant_id=tenant_id,
            start_time=start_time,
            end_time=end_time,
            period=AggregationPeriod.DAY
        )

        anomalies = []

        # Calculate baselines for each metric
        for metric_type in MetricType:
            values = [
                s.metrics.get(metric_type, 0.0)
                for s in summaries
                if metric_type in s.metrics
            ]

            if not values:
                continue

            avg = sum(values) / len(values)

            # Simple threshold-based detection
            threshold = avg * threshold_multiplier

            for summary in summaries:
                value = summary.metrics.get(metric_type, 0.0)
                if value > threshold:
                    anomalies.append({
                        "date": summary.period_start.strftime("%Y-%m-%d"),
                        "metric_type": metric_type.value,
                        "value": value,
                        "average": avg,
                        "threshold": threshold,
                        "severity": "high" if value > threshold * 1.5 else "medium"
                    })

        return anomalies

    def forecast_usage(
        self,
        tenant_id: str,
        forecast_days: int = 30
    ) -> Dict[str, Any]:
        """
        Forecast future usage based on historical trends

        Simple linear extrapolation - could be replaced with ML models.
        """
        # Get last 30 days
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=30)

        summaries = self.aggregate(
            tenant_id=tenant_id,
            start_time=start_time,
            end_time=end_time,
            period=AggregationPeriod.DAY
        )

        if not summaries:
            return {"error": "Insufficient data for forecasting"}

        # Calculate daily averages
        forecast = {
            "forecast_period_days": forecast_days,
            "metrics": {},
            "projected_cost": 0.0
        }

        for metric_type in MetricType:
            values = [
                s.metrics.get(metric_type, 0.0)
                for s in summaries
                if metric_type in s.metrics
            ]

            if values:
                daily_avg = sum(values) / len(values)
                projected = daily_avg * forecast_days
                forecast["metrics"][metric_type.value] = {
                    "daily_average": daily_avg,
                    "projected_total": projected
                }

        # Project costs
        total_cost = sum(s.total_cost_usd for s in summaries)
        daily_avg_cost = total_cost / len(summaries)
        forecast["projected_cost"] = daily_avg_cost * forecast_days

        return forecast
