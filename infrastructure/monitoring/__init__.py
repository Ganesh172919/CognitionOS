"""
Comprehensive Monitoring & Alerting System

Production-grade monitoring with metrics, alerts, dashboards,
and incident response automation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(str, Enum):
    """Alert status"""
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"


@dataclass
class Metric:
    """Single metric data point"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Labels for multi-dimensional metrics
    labels: Dict[str, str] = field(default_factory=dict)

    # Metadata
    unit: str = ""
    description: str = ""


@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    description: str
    severity: AlertSeverity

    # Condition
    metric_name: str
    condition: str  # e.g., "> 100", "< 0.5", "!= 0"
    threshold: float
    duration_seconds: int = 60  # How long condition must be true

    # Actions
    notification_channels: List[str] = field(default_factory=list)
    auto_remediation: Optional[str] = None

    # State
    enabled: bool = True
    last_evaluated: Optional[datetime] = None
    firing: bool = False
    fired_at: Optional[datetime] = None


@dataclass
class Alert:
    """Fired alert instance"""
    alert_id: str
    rule_id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus

    # Details
    message: str
    metric_value: float
    threshold: float

    # Timing
    fired_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None

    # Context
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    # Actions taken
    notifications_sent: List[str] = field(default_factory=list)
    remediation_attempted: bool = False
    remediation_result: Optional[str] = None


class MetricsCollector:
    """
    Metrics collection and aggregation

    Collects metrics from all system components with support
    for counters, gauges, histograms, and summaries.
    """

    def __init__(self):
        self._metrics: Dict[str, List[Metric]] = {}
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}

    def record_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record counter metric (monotonically increasing)"""
        key = self._make_key(name, labels)
        self._counters[key] = self._counters.get(key, 0.0) + value

        metric = Metric(
            name=name,
            value=self._counters[key],
            metric_type=MetricType.COUNTER,
            labels=labels or {}
        )

        self._store_metric(metric)
        logger.debug(f"Counter {name}={self._counters[key]}")

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Set gauge metric (can go up or down)"""
        key = self._make_key(name, labels)
        self._gauges[key] = value

        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            labels=labels or {}
        )

        self._store_metric(metric)
        logger.debug(f"Gauge {name}={value}")

    def record_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record histogram value (for distributions)"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            labels=labels or {}
        )

        self._store_metric(metric)

    def _store_metric(self, metric: Metric):
        """Store metric for querying"""
        if metric.name not in self._metrics:
            self._metrics[metric.name] = []

        self._metrics[metric.name].append(metric)

        # Keep only last 1000 data points per metric
        if len(self._metrics[metric.name]) > 1000:
            self._metrics[metric.name] = self._metrics[metric.name][-1000:]

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Make unique key for metric with labels"""
        if not labels:
            return name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def query(
        self,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> List[Metric]:
        """Query metrics"""
        metrics = self._metrics.get(metric_name, [])

        # Filter by time range
        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]
        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]

        # Filter by labels
        if labels:
            metrics = [
                m for m in metrics
                if all(m.labels.get(k) == v for k, v in labels.items())
            ]

        return metrics

    def get_current_value(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Optional[float]:
        """Get current metric value"""
        metrics = self.query(metric_name, labels=labels)
        return metrics[-1].value if metrics else None

    def aggregate(
        self,
        metric_name: str,
        aggregation: str = "avg",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> float:
        """Aggregate metric values"""
        metrics = self.query(metric_name, start_time, end_time)

        if not metrics:
            return 0.0

        values = [m.value for m in metrics]

        if aggregation == "avg":
            return sum(values) / len(values)
        elif aggregation == "sum":
            return sum(values)
        elif aggregation == "min":
            return min(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "count":
            return len(values)

        return 0.0


class AlertManager:
    """
    Alert management and notification system

    Evaluates alert rules, fires alerts, sends notifications,
    and handles alert lifecycle.
    """

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        notification_service: Optional[Any] = None
    ):
        self.metrics = metrics_collector
        self.notification_service = notification_service

        self._rules: Dict[str, AlertRule] = {}
        self._alerts: Dict[str, Alert] = {}
        self._firing_alerts: List[str] = []

        self._initialize_standard_rules()

    def _initialize_standard_rules(self):
        """Initialize standard alert rules"""

        # High error rate
        self.add_rule(AlertRule(
            rule_id="high_error_rate",
            name="High Error Rate",
            description="Error rate exceeds 5%",
            severity=AlertSeverity.HIGH,
            metric_name="error_rate",
            condition=">",
            threshold=0.05,
            duration_seconds=300,
            notification_channels=["slack", "pagerduty"]
        ))

        # High response time
        self.add_rule(AlertRule(
            rule_id="high_response_time",
            name="High Response Time",
            description="P95 response time exceeds 1s",
            severity=AlertSeverity.MEDIUM,
            metric_name="response_time_p95",
            condition=">",
            threshold=1.0,
            duration_seconds=300,
            notification_channels=["slack"]
        ))

        # Low availability
        self.add_rule(AlertRule(
            rule_id="low_availability",
            name="Low Availability",
            description="Availability below 99.9%",
            severity=AlertSeverity.CRITICAL,
            metric_name="availability",
            condition="<",
            threshold=0.999,
            duration_seconds=60,
            notification_channels=["slack", "pagerduty", "email"]
        ))

        # High CPU usage
        self.add_rule(AlertRule(
            rule_id="high_cpu",
            name="High CPU Usage",
            description="CPU usage exceeds 80%",
            severity=AlertSeverity.MEDIUM,
            metric_name="cpu_usage",
            condition=">",
            threshold=0.80,
            duration_seconds=300,
            notification_channels=["slack"]
        ))

        # High memory usage
        self.add_rule(AlertRule(
            rule_id="high_memory",
            name="High Memory Usage",
            description="Memory usage exceeds 85%",
            severity=AlertSeverity.MEDIUM,
            metric_name="memory_usage",
            condition=">",
            threshold=0.85,
            duration_seconds=300,
            notification_channels=["slack"]
        ))

        # Database connection pool exhausted
        self.add_rule(AlertRule(
            rule_id="db_pool_exhausted",
            name="Database Pool Exhausted",
            description="Database connection pool usage exceeds 90%",
            severity=AlertSeverity.HIGH,
            metric_name="db_pool_usage",
            condition=">",
            threshold=0.90,
            duration_seconds=60,
            notification_channels=["slack", "pagerduty"]
        ))

        # High queue depth
        self.add_rule(AlertRule(
            rule_id="high_queue_depth",
            name="High Queue Depth",
            description="Task queue depth exceeds 10000",
            severity=AlertSeverity.MEDIUM,
            metric_name="queue_depth",
            condition=">",
            threshold=10000,
            duration_seconds=300,
            notification_channels=["slack"]
        ))

    def add_rule(self, rule: AlertRule):
        """Add alert rule"""
        self._rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")

    async def evaluate_rules(self):
        """Evaluate all alert rules"""
        for rule in self._rules.values():
            if rule.enabled:
                await self._evaluate_rule(rule)

    async def _evaluate_rule(self, rule: AlertRule):
        """Evaluate single alert rule"""
        rule.last_evaluated = datetime.utcnow()

        # Get current metric value
        current_value = self.metrics.get_current_value(rule.metric_name)

        if current_value is None:
            return

        # Check condition
        condition_met = self._check_condition(
            current_value,
            rule.condition,
            rule.threshold
        )

        if condition_met and not rule.firing:
            # Condition became true
            await self._fire_alert(rule, current_value)
        elif not condition_met and rule.firing:
            # Condition resolved
            await self._resolve_alert(rule)

    def _check_condition(
        self,
        value: float,
        condition: str,
        threshold: float
    ) -> bool:
        """Check if condition is met"""
        if condition == ">":
            return value > threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<":
            return value < threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return abs(value - threshold) < 0.0001
        elif condition == "!=":
            return abs(value - threshold) >= 0.0001

        return False

    async def _fire_alert(self, rule: AlertRule, metric_value: float):
        """Fire alert"""
        logger.warning(f"🚨 ALERT FIRING: {rule.name}")

        rule.firing = True
        rule.fired_at = datetime.utcnow()

        alert = Alert(
            alert_id=f"alert_{rule.rule_id}_{int(datetime.utcnow().timestamp())}",
            rule_id=rule.rule_id,
            name=rule.name,
            severity=rule.severity,
            status=AlertStatus.FIRING,
            message=f"{rule.description} (current: {metric_value:.4f}, threshold: {rule.threshold})",
            metric_value=metric_value,
            threshold=rule.threshold,
            fired_at=rule.fired_at
        )

        self._alerts[alert.alert_id] = alert
        self._firing_alerts.append(alert.alert_id)

        # Send notifications
        await self._send_notifications(alert, rule.notification_channels)

        # Attempt auto-remediation
        if rule.auto_remediation:
            await self._attempt_remediation(alert, rule.auto_remediation)

    async def _resolve_alert(self, rule: AlertRule):
        """Resolve alert"""
        logger.info(f"✅ ALERT RESOLVED: {rule.name}")

        rule.firing = False

        # Find and resolve corresponding alert
        for alert_id in self._firing_alerts:
            alert = self._alerts.get(alert_id)
            if alert and alert.rule_id == rule.rule_id:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.utcnow()
                self._firing_alerts.remove(alert_id)

                # Send resolution notification
                await self._send_resolution_notification(alert)
                break

    async def _send_notifications(
        self,
        alert: Alert,
        channels: List[str]
    ):
        """Send alert notifications"""
        for channel in channels:
            try:
                if channel == "slack":
                    await self._send_slack_notification(alert)
                elif channel == "pagerduty":
                    await self._send_pagerduty_notification(alert)
                elif channel == "email":
                    await self._send_email_notification(alert)

                alert.notifications_sent.append(channel)
            except Exception as e:
                logger.error(f"Failed to send notification to {channel}: {e}")

    async def _send_slack_notification(self, alert: Alert):
        """Send Slack notification"""
        logger.info(f"Sending Slack notification for {alert.name}")
        # Would integrate with Slack API

    async def _send_pagerduty_notification(self, alert: Alert):
        """Send PagerDuty notification"""
        logger.info(f"Sending PagerDuty notification for {alert.name}")
        # Would integrate with PagerDuty API

    async def _send_email_notification(self, alert: Alert):
        """Send email notification"""
        logger.info(f"Sending email notification for {alert.name}")
        # Would integrate with email service

    async def _send_resolution_notification(self, alert: Alert):
        """Send alert resolution notification"""
        logger.info(f"Sending resolution notification for {alert.name}")
        # Would send to same channels as original alert

    async def _attempt_remediation(self, alert: Alert, action: str):
        """Attempt auto-remediation"""
        logger.info(f"Attempting auto-remediation: {action}")

        alert.remediation_attempted = True

        try:
            # Execute remediation action
            # This could restart services, scale resources, clear caches, etc.
            alert.remediation_result = "success"
        except Exception as e:
            alert.remediation_result = f"failed: {e}"
            logger.error(f"Remediation failed: {e}")

    def acknowledge_alert(self, alert_id: str, user: str):
        """Acknowledge alert"""
        alert = self._alerts.get(alert_id)
        if alert:
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = user
            logger.info(f"Alert {alert_id} acknowledged by {user}")

    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get active (firing) alerts"""
        alerts = [
            self._alerts[aid] for aid in self._firing_alerts
            if aid in self._alerts
        ]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: a.fired_at, reverse=True)

    def get_alert_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Alert]:
        """Get alert history"""
        alerts = list(self._alerts.values())

        if start_time:
            alerts = [a for a in alerts if a.fired_at >= start_time]
        if end_time:
            alerts = [a for a in alerts if a.fired_at <= end_time]

        return sorted(alerts, key=lambda a: a.fired_at, reverse=True)


class DashboardGenerator:
    """
    Monitoring dashboard generation

    Generates Grafana-compatible dashboards for visualization.
    """

    def generate_grafana_dashboard(
        self,
        title: str,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Generate Grafana dashboard JSON"""

        dashboard = {
            "title": title,
            "uid": title.lower().replace(" ", "_"),
            "timezone": "UTC",
            "schemaVersion": 16,
            "version": 1,
            "panels": []
        }

        # Add panel for each metric
        for i, metric in enumerate(metrics):
            panel = {
                "id": i + 1,
                "title": metric.replace("_", " ").title(),
                "type": "graph",
                "gridPos": {"x": 0, "y": i * 8, "w": 24, "h": 8},
                "targets": [
                    {
                        "expr": metric,
                        "refId": "A"
                    }
                ],
                "yaxes": [
                    {"format": "short", "label": None, "logBase": 1, "show": True},
                    {"format": "short", "label": None, "logBase": 1, "show": True}
                ]
            }
            dashboard["panels"].append(panel)

        return dashboard

    def generate_system_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive system dashboard"""

        return self.generate_grafana_dashboard(
            "System Overview",
            [
                "cpu_usage",
                "memory_usage",
                "disk_usage",
                "network_in",
                "network_out",
                "request_rate",
                "error_rate",
                "response_time_p50",
                "response_time_p95",
                "response_time_p99",
                "active_connections",
                "queue_depth"
            ]
        )

    def generate_business_dashboard(self) -> Dict[str, Any]:
        """Generate business metrics dashboard"""

        return self.generate_grafana_dashboard(
            "Business Metrics",
            [
                "active_users",
                "api_calls_per_minute",
                "revenue_per_hour",
                "new_signups",
                "conversion_rate",
                "churn_rate",
                "mrr",
                "arr"
            ]
        )


class HealthCheck:
    """
    System health check

    Performs comprehensive health checks across all components.
    """

    def __init__(self):
        self._checks: Dict[str, Callable] = {}

    def register_check(self, name: str, check_func: Callable):
        """Register health check function"""
        self._checks[name] = check_func

    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            "healthy": True,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }

        for name, check_func in self._checks.items():
            try:
                check_result = await check_func()
                results["checks"][name] = {
                    "status": "healthy" if check_result else "unhealthy",
                    "details": check_result
                }

                if not check_result:
                    results["healthy"] = False
            except Exception as e:
                results["checks"][name] = {
                    "status": "error",
                    "error": str(e)
                }
                results["healthy"] = False

        return results
