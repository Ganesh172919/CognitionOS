"""
Alert Management System

Production-grade alert management with rule engine, severity levels,
and intelligent alert aggregation.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"  # Urgent attention needed
    MEDIUM = "medium"  # Should be addressed soon
    LOW = "low"  # Informational
    INFO = "info"  # FYI only


class AlertState(str, Enum):
    """Alert states."""
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"


@dataclass
class Alert:
    """An active alert."""
    alert_id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    labels: Dict[str, str]
    annotations: Dict[str, str]
    state: AlertState
    starts_at: datetime
    ends_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    silence_until: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "rule_name": self.rule_name,
            "severity": self.severity,
            "message": self.message,
            "labels": self.labels,
            "annotations": self.annotations,
            "state": self.state,
            "starts_at": self.starts_at.isoformat(),
            "ends_at": self.ends_at.isoformat() if self.ends_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
        }


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    description: str
    severity: AlertSeverity
    condition: Callable[[Dict[str, Any]], bool]  # Function that returns True if alert should fire
    threshold: Optional[float] = None
    duration: int = 60  # Seconds the condition must be true
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    
    # State tracking
    _first_triggered: Optional[datetime] = None
    _last_checked: Optional[datetime] = None
    
    def check_condition(self, metrics: Dict[str, Any]) -> bool:
        """Check if alert condition is met."""
        try:
            self._last_checked = datetime.utcnow()
            result = self.condition(metrics)
            
            if result:
                if self._first_triggered is None:
                    self._first_triggered = datetime.utcnow()
                
                # Check if duration threshold met
                if self.duration > 0:
                    elapsed = (datetime.utcnow() - self._first_triggered).total_seconds()
                    return elapsed >= self.duration
                return True
            else:
                self._first_triggered = None
                return False
                
        except Exception as e:
            logger.error(f"Error checking alert condition for {self.name}: {e}")
            return False


class AlertManager:
    """
    Production alert management system.
    
    Features:
    - Rule-based alerting with severity levels
    - Alert aggregation and deduplication
    - Alert lifecycle management
    - Alert routing to multiple channels
    - Alert silencing and acknowledgment
    - Alert history tracking
    """
    
    def __init__(
        self,
        check_interval: int = 30,  # Check rules every 30 seconds
        retention_days: int = 30,  # Keep resolved alerts for 30 days
    ):
        """
        Initialize alert manager.
        
        Args:
            check_interval: Interval in seconds between rule checks
            retention_days: Days to retain resolved alerts
        """
        self.check_interval = check_interval
        self.retention_days = retention_days
        
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Alert routing
        self.routers: List[Callable] = []
        
        # Alert aggregation
        self.alert_groups: Dict[str, List[str]] = defaultdict(list)
        
        # Background task
        self._check_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("Alert manager initialized")
    
    def add_rule(self, rule: AlertRule):
        """
        Add an alert rule.
        
        Args:
            rule: Alert rule to add
        """
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name} (severity={rule.severity})")
    
    def remove_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    def add_router(self, router: Callable):
        """Add an alert router for notification delivery."""
        self.routers.append(router)
    
    async def start(self):
        """Start alert manager background checking."""
        if self._running:
            logger.warning("Alert manager already running")
            return
        
        self._running = True
        self._check_task = asyncio.create_task(self._check_rules_loop())
        logger.info("Alert manager started")
    
    async def stop(self):
        """Stop alert manager."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("Alert manager stopped")
    
    async def check_rules(self, metrics: Dict[str, Any]):
        """
        Check all rules against current metrics.
        
        Args:
            metrics: Current system metrics
        """
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            should_fire = rule.check_condition(metrics)
            
            if should_fire:
                await self._fire_alert(rule, metrics)
            else:
                await self._resolve_alert(rule.name)
    
    async def _fire_alert(self, rule: AlertRule, metrics: Dict[str, Any]):
        """Fire an alert."""
        alert_id = self._generate_alert_id(rule.name, rule.labels)
        
        # Check if alert already firing
        if alert_id in self.active_alerts:
            # Update existing alert
            alert = self.active_alerts[alert_id]
            logger.debug(f"Alert still firing: {alert_id}")
            return
        
        # Create new alert
        alert = Alert(
            alert_id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            message=self._format_message(rule, metrics),
            labels=rule.labels,
            annotations=rule.annotations,
            state=AlertState.FIRING,
            starts_at=datetime.utcnow(),
        )
        
        self.active_alerts[alert_id] = alert
        
        # Add to alert groups
        group_key = self._get_group_key(alert)
        self.alert_groups[group_key].append(alert_id)
        
        logger.warning(f"ALERT FIRING: {rule.name} ({rule.severity}) - {alert.message}")
        
        # Route alert to notification channels
        await self._route_alert(alert)
    
    async def _resolve_alert(self, rule_name: str):
        """Resolve an alert if it exists."""
        # Find alert by rule name
        alert_ids_to_resolve = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.rule_name == rule_name and alert.state == AlertState.FIRING
        ]
        
        for alert_id in alert_ids_to_resolve:
            alert = self.active_alerts[alert_id]
            alert.state = AlertState.RESOLVED
            alert.ends_at = datetime.utcnow()
            
            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]
            
            # Remove from group
            group_key = self._get_group_key(alert)
            if alert_id in self.alert_groups[group_key]:
                self.alert_groups[group_key].remove(alert_id)
            
            logger.info(f"Alert resolved: {alert_id}")
            
            # Notify resolution
            await self._route_alert(alert)
    
    async def _route_alert(self, alert: Alert):
        """Route alert to all registered routers."""
        for router in self.routers:
            try:
                await router(alert)
            except Exception as e:
                logger.error(f"Error routing alert to {router}: {e}")
    
    async def acknowledge_alert(self, alert_id: str, user: str):
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID to acknowledge
            user: User acknowledging the alert
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.state = AlertState.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = user
            
            logger.info(f"Alert acknowledged: {alert_id} by {user}")
    
    async def silence_alert(self, alert_id: str, duration_minutes: int):
        """
        Silence an alert for a duration.
        
        Args:
            alert_id: Alert ID to silence
            duration_minutes: Minutes to silence
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.state = AlertState.SILENCED
            alert.silence_until = datetime.utcnow() + timedelta(minutes=duration_minutes)
            
            logger.info(f"Alert silenced: {alert_id} for {duration_minutes} minutes")
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        state: Optional[AlertState] = None,
    ) -> List[Alert]:
        """
        Get active alerts with optional filtering.
        
        Args:
            severity: Filter by severity
            state: Filter by state
            
        Returns:
            List of matching alerts
        """
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if state:
            alerts = [a for a in alerts if a.state == state]
        
        # Sort by severity (critical first) then by start time
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 3,
            AlertSeverity.INFO: 4,
        }
        alerts.sort(key=lambda a: (severity_order[a.severity], a.starts_at))
        
        return alerts
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        active_count = len(self.active_alerts)
        by_severity = defaultdict(int)
        by_state = defaultdict(int)
        
        for alert in self.active_alerts.values():
            by_severity[alert.severity.value] += 1
            by_state[alert.state.value] += 1
        
        return {
            "total_active": active_count,
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for r in self.rules.values() if r.enabled),
            "by_severity": dict(by_severity),
            "by_state": dict(by_state),
            "alert_groups": len(self.alert_groups),
        }
    
    async def _check_rules_loop(self):
        """Background loop to check rules periodically."""
        while self._running:
            try:
                # Get current metrics (would integrate with monitoring system)
                metrics = await self._get_current_metrics()
                
                # Check all rules
                await self.check_rules(metrics)
                
                # Clean up old alerts
                await self._cleanup_old_alerts()
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert check loop: {e}", exc_info=True)
                await asyncio.sleep(self.check_interval)
    
    async def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics for rule evaluation."""
        # This would integrate with actual monitoring system
        # For now, return empty dict as placeholder
        return {}
    
    async def _cleanup_old_alerts(self):
        """Clean up old resolved alerts."""
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.ends_at and alert.ends_at > cutoff
        ]
    
    def _generate_alert_id(self, rule_name: str, labels: Dict[str, str]) -> str:
        """Generate unique alert ID from rule name and labels."""
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        id_str = f"{rule_name}:{label_str}"
        return hashlib.md5(id_str.encode()).hexdigest()[:16]
    
    def _get_group_key(self, alert: Alert) -> str:
        """Get grouping key for alert."""
        # Group by severity and common labels
        key_labels = ["service", "environment", "team"]
        label_parts = [alert.labels.get(k, "") for k in key_labels]
        return f"{alert.severity}:{':'.join(label_parts)}"
    
    def _format_message(self, rule: AlertRule, metrics: Dict[str, Any]) -> str:
        """Format alert message with metric values."""
        message = rule.description
        
        # Add threshold info if available
        if rule.threshold and "value" in metrics:
            message += f" (value={metrics['value']}, threshold={rule.threshold})"
        
        return message


# Pre-configured alert rules for common scenarios
def create_latency_alert(threshold_ms: float = 1000) -> AlertRule:
    """Create alert for high latency."""
    return AlertRule(
        name="high_latency",
        description=f"Request latency exceeds {threshold_ms}ms",
        severity=AlertSeverity.HIGH,
        condition=lambda m: m.get("latency_ms", 0) > threshold_ms,
        threshold=threshold_ms,
        duration=60,  # Must be high for 1 minute
        labels={"category": "performance"},
        annotations={"runbook": "https://docs.example.com/runbooks/high-latency"},
    )


def create_error_rate_alert(threshold_percent: float = 5.0) -> AlertRule:
    """Create alert for high error rate."""
    return AlertRule(
        name="high_error_rate",
        description=f"Error rate exceeds {threshold_percent}%",
        severity=AlertSeverity.CRITICAL,
        condition=lambda m: m.get("error_rate", 0) > threshold_percent,
        threshold=threshold_percent,
        duration=120,  # Must be high for 2 minutes
        labels={"category": "reliability"},
        annotations={"runbook": "https://docs.example.com/runbooks/high-error-rate"},
    )


def create_cost_alert(threshold_usd: float = 100.0) -> AlertRule:
    """Create alert for high cost."""
    return AlertRule(
        name="high_cost",
        description=f"Hourly cost exceeds ${threshold_usd}",
        severity=AlertSeverity.MEDIUM,
        condition=lambda m: m.get("cost_usd_per_hour", 0) > threshold_usd,
        threshold=threshold_usd,
        duration=300,  # Must be high for 5 minutes
        labels={"category": "cost"},
        annotations={"runbook": "https://docs.example.com/runbooks/high-cost"},
    )
