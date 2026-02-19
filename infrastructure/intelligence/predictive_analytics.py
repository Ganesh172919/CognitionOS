"""
Predictive Analytics and Anomaly Detection Engine
ML-powered predictive analytics for resource usage, performance, and anomaly detection.
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import statistics
from pydantic import BaseModel, Field


class MetricType(str, Enum):
    """Types of metrics to analyze"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_TRAFFIC = "network_traffic"
    API_REQUESTS = "api_requests"
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    DATABASE_CONNECTIONS = "database_connections"
    QUEUE_SIZE = "queue_size"
    COST = "cost"


class AnomalyType(str, Enum):
    """Types of anomalies"""
    SPIKE = "spike"
    DROP = "drop"
    TREND_CHANGE = "trend_change"
    SEASONAL_DEVIATION = "seasonal_deviation"
    PATTERN_BREAK = "pattern_break"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class MetricDataPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Anomaly:
    """Detected anomaly"""
    anomaly_id: str
    metric_type: MetricType
    anomaly_type: AnomalyType
    severity: AlertSeverity
    detected_at: datetime
    value: float
    expected_range: Tuple[float, float]
    deviation_percentage: float
    confidence: float  # 0.0 to 1.0
    description: str
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class Prediction:
    """Future metric prediction"""
    prediction_id: str
    metric_type: MetricType
    predicted_for: datetime
    predicted_value: float
    confidence_interval: Tuple[float, float]  # (lower, upper)
    confidence: float
    method: str  # linear_regression, moving_average, etc.


class ForecastModel(BaseModel):
    """Forecast configuration"""
    model_id: str = Field(default_factory=lambda: str(uuid4()))
    metric_type: MetricType
    lookback_hours: int = 168  # 7 days
    forecast_hours: int = 24  # 1 day
    confidence_level: float = 0.95
    seasonal_period: Optional[int] = None  # hours (e.g., 24 for daily)


class PredictiveAnalyticsEngine:
    """
    Advanced predictive analytics and anomaly detection system.
    Uses statistical methods and ML techniques for forecasting and anomaly detection.
    """

    def __init__(self):
        # Store historical metrics
        self.metrics: Dict[MetricType, deque] = defaultdict(lambda: deque(maxlen=10000))

        # Store detected anomalies
        self.anomalies: List[Anomaly] = []

        # Store predictions
        self.predictions: Dict[str, Prediction] = {}

        # Baseline statistics for each metric
        self.baselines: Dict[MetricType, Dict[str, float]] = {}

        # Alert thresholds
        self.thresholds = self._initialize_thresholds()

    def _initialize_thresholds(self) -> Dict[MetricType, Dict[str, Any]]:
        """Initialize default alert thresholds"""
        return {
            MetricType.CPU_USAGE: {
                "warning": 70.0,
                "critical": 90.0,
                "spike_threshold": 2.5  # 2.5x std dev
            },
            MetricType.MEMORY_USAGE: {
                "warning": 75.0,
                "critical": 90.0,
                "spike_threshold": 2.5
            },
            MetricType.ERROR_RATE: {
                "warning": 1.0,  # 1%
                "critical": 5.0,  # 5%
                "spike_threshold": 3.0
            },
            MetricType.RESPONSE_TIME: {
                "warning": 1000.0,  # ms
                "critical": 3000.0,
                "spike_threshold": 2.0
            },
            MetricType.API_REQUESTS: {
                "spike_threshold": 3.0,
                "drop_threshold": 0.3  # 70% drop
            },
            MetricType.COST: {
                "warning": 1000.0,  # $1000/day
                "critical": 5000.0,
                "spike_threshold": 2.0
            }
        }

    async def ingest_metric(
        self,
        metric_type: MetricType,
        value: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Ingest a metric data point
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        data_point = MetricDataPoint(
            timestamp=timestamp,
            value=value,
            metadata=metadata or {}
        )

        self.metrics[metric_type].append(data_point)

        # Update baseline statistics
        await self._update_baseline(metric_type)

        # Check for anomalies
        anomaly = await self._detect_anomaly(metric_type, data_point)
        if anomaly:
            self.anomalies.append(anomaly)

    async def _update_baseline(self, metric_type: MetricType) -> None:
        """Update baseline statistics for a metric"""
        data_points = list(self.metrics[metric_type])

        if len(data_points) < 10:
            return  # Need minimum data points

        values = [dp.value for dp in data_points]

        self.baselines[metric_type] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "p95": self._percentile(values, 0.95),
            "p99": self._percentile(values, 0.99)
        }

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile"""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]

    async def _detect_anomaly(
        self,
        metric_type: MetricType,
        data_point: MetricDataPoint
    ) -> Optional[Anomaly]:
        """
        Detect anomalies using statistical methods
        """
        if metric_type not in self.baselines:
            return None

        baseline = self.baselines[metric_type]
        thresholds = self.thresholds.get(metric_type, {})

        value = data_point.value
        mean = baseline["mean"]
        stdev = baseline["stdev"]

        # Calculate z-score
        if stdev > 0:
            z_score = abs((value - mean) / stdev)
        else:
            z_score = 0

        # Spike detection
        spike_threshold = thresholds.get("spike_threshold", 3.0)
        if z_score > spike_threshold and value > mean:
            deviation_pct = ((value - mean) / mean) * 100

            return Anomaly(
                anomaly_id=str(uuid4()),
                metric_type=metric_type,
                anomaly_type=AnomalyType.SPIKE,
                severity=self._determine_severity(metric_type, value, thresholds),
                detected_at=data_point.timestamp,
                value=value,
                expected_range=(mean - 2*stdev, mean + 2*stdev),
                deviation_percentage=deviation_pct,
                confidence=min(z_score / spike_threshold, 1.0),
                description=f"{metric_type.value} spike detected: {value:.2f} (expected: {mean:.2f})",
                recommended_actions=self._get_spike_actions(metric_type)
            )

        # Drop detection
        drop_threshold = thresholds.get("drop_threshold", 0.5)
        if value < mean * drop_threshold:
            deviation_pct = ((mean - value) / mean) * 100

            return Anomaly(
                anomaly_id=str(uuid4()),
                metric_type=metric_type,
                anomaly_type=AnomalyType.DROP,
                severity=AlertSeverity.WARNING,
                detected_at=data_point.timestamp,
                value=value,
                expected_range=(mean - 2*stdev, mean + 2*stdev),
                deviation_percentage=deviation_pct,
                confidence=0.8,
                description=f"{metric_type.value} drop detected: {value:.2f} (expected: {mean:.2f})",
                recommended_actions=self._get_drop_actions(metric_type)
            )

        # Threshold-based alerts
        if "critical" in thresholds and value >= thresholds["critical"]:
            return Anomaly(
                anomaly_id=str(uuid4()),
                metric_type=metric_type,
                anomaly_type=AnomalyType.SPIKE,
                severity=AlertSeverity.CRITICAL,
                detected_at=data_point.timestamp,
                value=value,
                expected_range=(0, thresholds["critical"]),
                deviation_percentage=((value - thresholds["critical"]) / thresholds["critical"]) * 100,
                confidence=1.0,
                description=f"{metric_type.value} exceeded critical threshold: {value:.2f} >= {thresholds['critical']}",
                recommended_actions=self._get_critical_actions(metric_type)
            )

        elif "warning" in thresholds and value >= thresholds["warning"]:
            return Anomaly(
                anomaly_id=str(uuid4()),
                metric_type=metric_type,
                anomaly_type=AnomalyType.SPIKE,
                severity=AlertSeverity.WARNING,
                detected_at=data_point.timestamp,
                value=value,
                expected_range=(0, thresholds["warning"]),
                deviation_percentage=((value - thresholds["warning"]) / thresholds["warning"]) * 100,
                confidence=0.9,
                description=f"{metric_type.value} exceeded warning threshold: {value:.2f} >= {thresholds['warning']}",
                recommended_actions=self._get_warning_actions(metric_type)
            )

        return None

    def _determine_severity(
        self,
        metric_type: MetricType,
        value: float,
        thresholds: Dict[str, Any]
    ) -> AlertSeverity:
        """Determine severity based on value and thresholds"""
        if "critical" in thresholds and value >= thresholds["critical"]:
            return AlertSeverity.CRITICAL
        elif "warning" in thresholds and value >= thresholds["warning"]:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO

    def _get_spike_actions(self, metric_type: MetricType) -> List[str]:
        """Get recommended actions for spike anomalies"""
        actions = {
            MetricType.CPU_USAGE: [
                "Check for runaway processes",
                "Review recent code deployments",
                "Consider horizontal scaling",
                "Analyze application profiling data"
            ],
            MetricType.MEMORY_USAGE: [
                "Check for memory leaks",
                "Review caching strategies",
                "Analyze heap dumps",
                "Consider increasing instance size"
            ],
            MetricType.ERROR_RATE: [
                "Review recent deployments",
                "Check application logs",
                "Investigate error patterns",
                "Consider rollback if needed"
            ],
            MetricType.API_REQUESTS: [
                "Check for traffic spikes or attacks",
                "Review rate limiting",
                "Analyze request patterns",
                "Scale infrastructure if needed"
            ],
            MetricType.COST: [
                "Review resource usage",
                "Check for cost anomalies",
                "Analyze expensive operations",
                "Optimize resource allocation"
            ]
        }
        return actions.get(metric_type, ["Investigate the issue", "Review system metrics"])

    def _get_drop_actions(self, metric_type: MetricType) -> List[str]:
        """Get recommended actions for drop anomalies"""
        return [
            "Check system health",
            "Verify connectivity",
            "Review recent changes",
            "Investigate potential outages"
        ]

    def _get_critical_actions(self, metric_type: MetricType) -> List[str]:
        """Get actions for critical alerts"""
        return [
            "IMMEDIATE ACTION REQUIRED",
            "Page on-call engineer",
            "Check system availability",
            "Consider emergency scaling"
        ]

    def _get_warning_actions(self, metric_type: MetricType) -> List[str]:
        """Get actions for warning alerts"""
        return [
            "Monitor closely",
            "Review trends",
            "Prepare mitigation plan",
            "Alert team if escalates"
        ]

    async def forecast_metric(
        self,
        metric_type: MetricType,
        forecast_config: ForecastModel
    ) -> List[Prediction]:
        """
        Forecast future metric values
        """
        data_points = list(self.metrics[metric_type])

        if len(data_points) < 24:
            return []  # Need minimum historical data

        # Get recent data for forecasting
        lookback_cutoff = datetime.utcnow() - timedelta(hours=forecast_config.lookback_hours)
        recent_data = [
            dp for dp in data_points
            if dp.timestamp >= lookback_cutoff
        ]

        if not recent_data:
            return []

        predictions = []

        # Simple moving average forecast (production would use more sophisticated methods)
        window_size = min(24, len(recent_data))
        recent_values = [dp.value for dp in recent_data[-window_size:]]
        ma_value = statistics.mean(recent_values)
        ma_stdev = statistics.stdev(recent_values) if len(recent_values) > 1 else 0

        # Generate predictions for future time periods
        last_timestamp = recent_data[-1].timestamp

        for hour_ahead in range(1, forecast_config.forecast_hours + 1):
            predicted_timestamp = last_timestamp + timedelta(hours=hour_ahead)

            # Simple forecast with confidence interval
            prediction = Prediction(
                prediction_id=str(uuid4()),
                metric_type=metric_type,
                predicted_for=predicted_timestamp,
                predicted_value=ma_value,
                confidence_interval=(
                    ma_value - 1.96 * ma_stdev,  # 95% CI lower
                    ma_value + 1.96 * ma_stdev   # 95% CI upper
                ),
                confidence=0.85,
                method="moving_average"
            )

            predictions.append(prediction)
            self.predictions[prediction.prediction_id] = prediction

        return predictions

    async def detect_trends(
        self,
        metric_type: MetricType,
        window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Detect trends in metric data
        """
        data_points = list(self.metrics[metric_type])

        if len(data_points) < 10:
            return {"trend": "insufficient_data"}

        # Get recent data
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        recent_data = [dp for dp in data_points if dp.timestamp >= cutoff]

        if len(recent_data) < 10:
            return {"trend": "insufficient_data"}

        values = [dp.value for dp in recent_data]

        # Simple linear regression to detect trend
        n = len(values)
        x = list(range(n))

        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator

        # Classify trend
        if abs(slope) < 0.01:
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"

        # Calculate rate of change
        first_value = values[0]
        last_value = values[-1]
        rate_of_change = ((last_value - first_value) / first_value * 100) if first_value != 0 else 0

        return {
            "trend": trend,
            "slope": slope,
            "rate_of_change_percent": rate_of_change,
            "current_value": last_value,
            "window_hours": window_hours,
            "data_points": len(recent_data)
        }

    def get_recent_anomalies(
        self,
        hours: int = 24,
        severity: Optional[AlertSeverity] = None
    ) -> List[Anomaly]:
        """Get recent anomalies"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        anomalies = [
            a for a in self.anomalies
            if a.detected_at >= cutoff
        ]

        if severity:
            anomalies = [a for a in anomalies if a.severity == severity]

        # Sort by detection time, most recent first
        anomalies.sort(key=lambda x: x.detected_at, reverse=True)

        return anomalies

    def get_analytics_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive analytics dashboard data"""
        dashboard = {
            "summary": {
                "metrics_tracked": len(self.metrics),
                "total_data_points": sum(len(dq) for dq in self.metrics.values()),
                "anomalies_detected_24h": len(self.get_recent_anomalies(hours=24)),
                "critical_anomalies": len(self.get_recent_anomalies(hours=24, severity=AlertSeverity.CRITICAL))
            },
            "metrics": {}
        }

        # Add metrics data
        for metric_type, data_points in self.metrics.items():
            if not data_points:
                continue

            recent_data = [dp for dp in data_points if dp.timestamp >= datetime.utcnow() - timedelta(hours=1)]

            if recent_data:
                current_value = recent_data[-1].value
                avg_value = statistics.mean([dp.value for dp in recent_data])
            else:
                current_value = 0
                avg_value = 0

            baseline = self.baselines.get(metric_type, {})

            dashboard["metrics"][metric_type.value] = {
                "current_value": current_value,
                "avg_1h": avg_value,
                "baseline_mean": baseline.get("mean", 0),
                "baseline_p95": baseline.get("p95", 0),
                "total_data_points": len(data_points)
            }

        return dashboard

    def clear_old_data(self, days: int = 30) -> int:
        """Clear data older than specified days"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        cleared_count = 0

        for metric_type in self.metrics:
            original_size = len(self.metrics[metric_type])

            # Keep only recent data
            recent_data = [
                dp for dp in self.metrics[metric_type]
                if dp.timestamp >= cutoff
            ]

            self.metrics[metric_type].clear()
            self.metrics[metric_type].extend(recent_data)

            cleared_count += original_size - len(recent_data)

        return cleared_count
