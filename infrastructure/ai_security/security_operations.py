"""
AI-Powered Security Operations Center — SIEM, threat intelligence,
behavioral anomaly detection, incident management, and automated response.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Set, Tuple


# ─────────────────────────────────── Enums ───────────────────────────────────


class ThreatSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class EventCategory(str, Enum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    NETWORK = "network"
    MALWARE = "malware"
    INJECTION = "injection"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    EXFILTRATION = "exfiltration"
    DENIAL_OF_SERVICE = "denial_of_service"
    CONFIGURATION_CHANGE = "configuration_change"
    ANOMALY = "anomaly"


class IncidentStatus(str, Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    REMEDIATED = "remediated"
    CLOSED = "closed"
    FALSE_POSITIVE = "false_positive"


class ResponseAction(str, Enum):
    ALERT = "alert"
    BLOCK_IP = "block_ip"
    REVOKE_SESSION = "revoke_session"
    RATE_LIMIT = "rate_limit"
    QUARANTINE = "quarantine"
    ESCALATE = "escalate"
    LOG_ONLY = "log_only"
    DISABLE_ACCOUNT = "disable_account"
    ROTATE_CREDENTIALS = "rotate_credentials"


class ThreatIndicatorType(str, Enum):
    IP_ADDRESS = "ip_address"
    DOMAIN = "domain"
    URL = "url"
    FILE_HASH = "file_hash"
    EMAIL = "email"
    USER_AGENT = "user_agent"
    SIGNATURE = "signature"


# ────────────────────────────── Data structures ──────────────────────────────


@dataclass
class SecurityEvent:
    event_id: str
    event_category: EventCategory
    severity: ThreatSeverity
    source_ip: Optional[str]
    destination_ip: Optional[str]
    user_id: Optional[str]
    resource: Optional[str]
    action: str
    outcome: str
    raw_data: Dict[str, Any]
    risk_score: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_category": self.event_category.value,
            "severity": self.severity.value,
            "source_ip": self.source_ip,
            "destination_ip": self.destination_ip,
            "user_id": self.user_id,
            "resource": self.resource,
            "action": self.action,
            "outcome": self.outcome,
            "risk_score": self.risk_score,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
        }


@dataclass
class ThreatIndicator:
    indicator_id: str
    indicator_type: ThreatIndicatorType
    value: str
    threat_name: str
    severity: ThreatSeverity
    confidence: float
    source: str
    first_seen: datetime
    last_seen: datetime
    hit_count: int = 0
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "indicator_id": self.indicator_id,
            "indicator_type": self.indicator_type.value,
            "value": self.value,
            "threat_name": self.threat_name,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "source": self.source,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "hit_count": self.hit_count,
            "active": self.active,
        }


@dataclass
class SecurityIncident:
    incident_id: str
    title: str
    description: str
    severity: ThreatSeverity
    status: IncidentStatus
    related_events: List[str]
    affected_users: List[str]
    affected_resources: List[str]
    assigned_to: Optional[str]
    timeline: List[Dict[str, Any]]
    response_actions_taken: List[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "status": self.status.value,
            "related_events": self.related_events,
            "affected_users": self.affected_users,
            "affected_resources": self.affected_resources,
            "assigned_to": self.assigned_to,
            "timeline": self.timeline,
            "response_actions_taken": self.response_actions_taken,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata,
        }


@dataclass
class DetectionRule:
    rule_id: str
    name: str
    description: str
    category: EventCategory
    severity: ThreatSeverity
    conditions: List[Dict[str, Any]]
    response_actions: List[ResponseAction]
    enabled: bool = True
    false_positive_rate: float = 0.02
    hit_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "severity": self.severity.value,
            "conditions": self.conditions,
            "response_actions": [a.value for a in self.response_actions],
            "enabled": self.enabled,
            "false_positive_rate": self.false_positive_rate,
            "hit_count": self.hit_count,
            "created_at": self.created_at.isoformat(),
        }


# ─────────────────────────── Threat Intelligence ────────────────────────────


class ThreatIntelligenceEngine:
    """
    Manages threat indicators, IP reputation, malicious pattern matching,
    and threat feed integration.
    """

    # Known malicious patterns (regex)
    INJECTION_PATTERNS = [
        r"(\b(union|select|insert|drop|update|delete|exec|xp_)\b)",
        r"(<script[^>]*>)",
        r"(javascript:)",
        r"(\.\./\.\./)",
        r"(\%00|\x00)",
        r"(eval\s*\()",
    ]

    def __init__(self):
        self._indicators: Dict[str, ThreatIndicator] = {}
        self._ip_index: Dict[str, str] = {}
        self._domain_index: Dict[str, str] = {}
        self._hash_index: Dict[str, str] = {}
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]
        # Seed with some known bad CIDRs (private ranges used for test)
        self._blocked_cidrs: List[ipaddress.IPv4Network] = []

    def add_indicator(self, indicator: ThreatIndicator) -> None:
        self._indicators[indicator.indicator_id] = indicator
        if indicator.indicator_type == ThreatIndicatorType.IP_ADDRESS:
            self._ip_index[indicator.value] = indicator.indicator_id
        elif indicator.indicator_type == ThreatIndicatorType.DOMAIN:
            self._domain_index[indicator.value.lower()] = indicator.indicator_id
        elif indicator.indicator_type == ThreatIndicatorType.FILE_HASH:
            self._hash_index[indicator.value.lower()] = indicator.indicator_id

    def check_ip(self, ip: str) -> Optional[ThreatIndicator]:
        indicator_id = self._ip_index.get(ip)
        if indicator_id:
            ind = self._indicators.get(indicator_id)
            if ind and ind.active:
                ind.hit_count += 1
                ind.last_seen = datetime.now(timezone.utc)
                return ind
        return None

    def check_domain(self, domain: str) -> Optional[ThreatIndicator]:
        indicator_id = self._domain_index.get(domain.lower())
        if indicator_id:
            ind = self._indicators.get(indicator_id)
            if ind and ind.active:
                ind.hit_count += 1
                return ind
        return None

    def check_content(self, content: str) -> List[Dict[str, Any]]:
        matches = []
        for pattern in self._compiled_patterns:
            match = pattern.search(content)
            if match:
                matches.append(
                    {
                        "pattern": pattern.pattern,
                        "matched_text": match.group(0)[:50],
                        "category": "injection_attempt",
                        "severity": ThreatSeverity.HIGH.value,
                    }
                )
        return matches

    def compute_risk_score(
        self,
        source_ip: Optional[str],
        user_id: Optional[str],
        action: str,
        content: Optional[str],
        context: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        score = 0.0
        risk_factors = []

        if source_ip:
            indicator = self.check_ip(source_ip)
            if indicator:
                severity_weights = {
                    ThreatSeverity.CRITICAL: 0.9,
                    ThreatSeverity.HIGH: 0.7,
                    ThreatSeverity.MEDIUM: 0.5,
                    ThreatSeverity.LOW: 0.3,
                    ThreatSeverity.INFORMATIONAL: 0.1,
                }
                score += severity_weights.get(indicator.severity, 0.5)
                risk_factors.append(f"Known malicious IP: {source_ip}")

        if content:
            content_matches = self.check_content(content)
            if content_matches:
                score += 0.4 * min(len(content_matches), 3)
                risk_factors.append(f"{len(content_matches)} injection patterns detected")

        failed_auth = context.get("failed_auth_count", 0)
        if failed_auth > 5:
            score += min(0.3, failed_auth * 0.03)
            risk_factors.append(f"High failed auth count: {failed_auth}")

        unusual_time = context.get("unusual_time", False)
        if unusual_time:
            score += 0.15
            risk_factors.append("Activity outside normal hours")

        return min(1.0, round(score, 4)), risk_factors

    def get_intel_stats(self) -> Dict[str, Any]:
        indicators = list(self._indicators.values())
        by_type: Dict[str, int] = defaultdict(int)
        by_severity: Dict[str, int] = defaultdict(int)
        for ind in indicators:
            by_type[ind.indicator_type.value] += 1
            by_severity[ind.severity.value] += 1
        return {
            "total_indicators": len(indicators),
            "active_indicators": sum(1 for i in indicators if i.active),
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "total_hits": sum(i.hit_count for i in indicators),
        }


# ─────────────────────── Behavioral Anomaly Detector ────────────────────────


class BehavioralAnomalyDetector:
    """
    Detects anomalous user/system behavior using statistical baselines,
    velocity checks, and pattern deviation analysis.
    """

    def __init__(self, window_minutes: int = 60):
        self.window_minutes = window_minutes
        self._user_baselines: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "avg_requests_per_hour": 50.0,
                "avg_bytes_transferred": 1024 * 100,
                "normal_hours": list(range(8, 20)),
                "normal_ips": set(),
                "normal_resources": set(),
                "action_counts": defaultdict(int),
            }
        )
        self._user_events: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=1000))
        self._ip_event_counts: Dict[str, Deque] = defaultdict(
            lambda: deque(maxlen=500)
        )

    def record_event(self, event: SecurityEvent) -> None:
        now_ts = time.monotonic()
        if event.user_id:
            self._user_events[event.user_id].append(
                {
                    "ts": now_ts,
                    "action": event.action,
                    "resource": event.resource,
                    "ip": event.source_ip,
                    "category": event.event_category.value,
                }
            )
        if event.source_ip:
            self._ip_event_counts[event.source_ip].append(now_ts)

    def analyze_user_behavior(
        self, user_id: str, event: SecurityEvent
    ) -> Dict[str, Any]:
        anomalies = []
        risk_boost = 0.0
        baseline = self._user_baselines[user_id]

        # Velocity check
        recent_events = self._get_recent_user_events(user_id, minutes=60)
        hourly_rate = len(recent_events)
        if hourly_rate > baseline["avg_requests_per_hour"] * 3:
            anomalies.append(
                {
                    "type": "high_velocity",
                    "detail": f"Request rate {hourly_rate}/hr vs baseline {baseline['avg_requests_per_hour']}/hr",
                    "severity": "high",
                }
            )
            risk_boost += 0.3

        # New IP detection
        if event.source_ip:
            normal_ips: Set[str] = baseline["normal_ips"]
            if len(normal_ips) > 0 and event.source_ip not in normal_ips:
                anomalies.append(
                    {
                        "type": "new_ip",
                        "detail": f"First seen IP: {event.source_ip}",
                        "severity": "medium",
                    }
                )
                risk_boost += 0.15
            # Add to normal IPs after first 5 different IPs
            if len(normal_ips) < 50:
                normal_ips.add(event.source_ip)

        # Off-hours access
        current_hour = datetime.now(timezone.utc).hour
        if current_hour not in baseline["normal_hours"]:
            anomalies.append(
                {
                    "type": "off_hours_access",
                    "detail": f"Activity at hour {current_hour} UTC",
                    "severity": "low",
                }
            )
            risk_boost += 0.1

        # Update baseline action counts
        baseline["action_counts"][event.action] += 1

        return {
            "user_id": user_id,
            "anomalies_detected": len(anomalies),
            "anomalies": anomalies,
            "risk_boost": round(risk_boost, 4),
            "hourly_rate": hourly_rate,
        }

    def analyze_ip_behavior(self, ip: str, window_minutes: int = 5) -> Dict[str, Any]:
        cutoff_ts = time.monotonic() - (window_minutes * 60)
        recent = [ts for ts in self._ip_event_counts.get(ip, []) if ts > cutoff_ts]
        rate = len(recent)
        anomalies = []
        risk_boost = 0.0
        if rate > 100:
            anomalies.append(
                {
                    "type": "high_request_rate",
                    "detail": f"{rate} requests in {window_minutes} minutes",
                    "severity": "high" if rate > 200 else "medium",
                }
            )
            risk_boost += min(0.5, rate / 500)
        return {
            "ip": ip,
            "request_rate": rate,
            "window_minutes": window_minutes,
            "anomalies": anomalies,
            "risk_boost": round(risk_boost, 4),
        }

    def _get_recent_user_events(
        self, user_id: str, minutes: int = 60
    ) -> List[Dict[str, Any]]:
        cutoff_ts = time.monotonic() - (minutes * 60)
        return [
            e
            for e in self._user_events.get(user_id, [])
            if e.get("ts", 0) > cutoff_ts
        ]


# ─────────────────────── Detection Rule Engine ──────────────────────────────


class DetectionRuleEngine:
    """
    Evaluates security events against detection rules with fast path matching
    and severity-based prioritization.
    """

    DEFAULT_RULES = [
        {
            "name": "Multiple Failed Logins",
            "category": EventCategory.AUTHENTICATION,
            "severity": ThreatSeverity.HIGH,
            "conditions": [
                {"field": "action", "op": "eq", "value": "login_failed"},
                {"field": "count_window_5min", "op": "gt", "value": 5},
            ],
            "response_actions": [ResponseAction.RATE_LIMIT, ResponseAction.ALERT],
        },
        {
            "name": "SQL Injection Attempt",
            "category": EventCategory.INJECTION,
            "severity": ThreatSeverity.CRITICAL,
            "conditions": [
                {"field": "risk_score", "op": "gt", "value": 0.7},
                {"field": "event_category", "op": "eq", "value": "injection"},
            ],
            "response_actions": [ResponseAction.BLOCK_IP, ResponseAction.ALERT],
        },
        {
            "name": "Unusual Data Export",
            "category": EventCategory.EXFILTRATION,
            "severity": ThreatSeverity.HIGH,
            "conditions": [
                {"field": "action", "op": "eq", "value": "bulk_export"},
                {"field": "risk_score", "op": "gt", "value": 0.5},
            ],
            "response_actions": [ResponseAction.ALERT, ResponseAction.LOG_ONLY],
        },
        {
            "name": "Privilege Escalation",
            "category": EventCategory.PRIVILEGE_ESCALATION,
            "severity": ThreatSeverity.CRITICAL,
            "conditions": [
                {"field": "action", "op": "contains", "value": "privilege"},
            ],
            "response_actions": [ResponseAction.ALERT, ResponseAction.ESCALATE],
        },
        {
            "name": "Admin Account Anomaly",
            "category": EventCategory.AUTHORIZATION,
            "severity": ThreatSeverity.HIGH,
            "conditions": [
                {"field": "resource", "op": "contains", "value": "admin"},
                {"field": "risk_score", "op": "gt", "value": 0.4},
            ],
            "response_actions": [ResponseAction.ALERT],
        },
    ]

    def __init__(self):
        self._rules: Dict[str, DetectionRule] = {}
        self._load_default_rules()

    def add_rule(self, rule: DetectionRule) -> None:
        self._rules[rule.rule_id] = rule

    def evaluate(self, event: SecurityEvent) -> List[Tuple[DetectionRule, List[ResponseAction]]]:
        triggered: List[Tuple[DetectionRule, List[ResponseAction]]] = []
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            if self._matches(event, rule):
                rule.hit_count += 1
                triggered.append((rule, rule.response_actions))
        return triggered

    def _matches(self, event: SecurityEvent, rule: DetectionRule) -> bool:
        for condition in rule.conditions:
            field = condition.get("field", "")
            op = condition.get("op", "eq")
            expected = condition.get("value")
            actual = self._get_field(event, field)
            if not self._evaluate_condition(actual, op, expected):
                return False
        return True

    def _get_field(self, event: SecurityEvent, field: str) -> Any:
        field_map = {
            "action": event.action,
            "event_category": event.event_category.value,
            "severity": event.severity.value,
            "risk_score": event.risk_score,
            "source_ip": event.source_ip,
            "user_id": event.user_id,
            "resource": event.resource,
            "outcome": event.outcome,
        }
        return field_map.get(field, event.raw_data.get(field))

    def _evaluate_condition(self, actual: Any, op: str, expected: Any) -> bool:
        if actual is None:
            return False
        if op == "eq":
            return actual == expected
        if op == "ne":
            return actual != expected
        if op == "gt":
            try:
                return float(actual) > float(expected)
            except (ValueError, TypeError):
                return False
        if op == "lt":
            try:
                return float(actual) < float(expected)
            except (ValueError, TypeError):
                return False
        if op == "contains":
            return expected is not None and str(expected).lower() in str(actual).lower()
        if op == "startswith":
            return str(actual).startswith(str(expected))
        return False

    def _load_default_rules(self) -> None:
        for r in self.DEFAULT_RULES:
            rule = DetectionRule(
                rule_id=str(uuid.uuid4()),
                name=r["name"],
                description=f"Default rule: {r['name']}",
                category=r["category"],
                severity=r["severity"],
                conditions=r["conditions"],
                response_actions=r["response_actions"],
            )
            self._rules[rule.rule_id] = rule

    def get_rule_stats(self) -> Dict[str, Any]:
        rules = list(self._rules.values())
        return {
            "total_rules": len(rules),
            "enabled_rules": sum(1 for r in rules if r.enabled),
            "total_hits": sum(r.hit_count for r in rules),
            "top_triggered": sorted(
                [{"rule": r.name, "hits": r.hit_count} for r in rules],
                key=lambda x: x["hits"],
                reverse=True,
            )[:5],
        }


# ─────────────────────── Incident Manager ───────────────────────────────────


class IncidentManager:
    """
    Manages security incident lifecycle: creation, investigation,
    escalation, response, and post-mortem.
    """

    def __init__(self):
        self._incidents: Dict[str, SecurityIncident] = {}
        self._sla_hours: Dict[ThreatSeverity, int] = {
            ThreatSeverity.CRITICAL: 1,
            ThreatSeverity.HIGH: 4,
            ThreatSeverity.MEDIUM: 24,
            ThreatSeverity.LOW: 72,
            ThreatSeverity.INFORMATIONAL: 168,
        }

    def create_incident(
        self,
        title: str,
        description: str,
        severity: ThreatSeverity,
        related_events: List[str],
        affected_users: Optional[List[str]] = None,
        affected_resources: Optional[List[str]] = None,
    ) -> SecurityIncident:
        incident = SecurityIncident(
            incident_id=str(uuid.uuid4()),
            title=title,
            description=description,
            severity=severity,
            status=IncidentStatus.OPEN,
            related_events=related_events,
            affected_users=affected_users or [],
            affected_resources=affected_resources or [],
            assigned_to=None,
            timeline=[
                {
                    "event": "incident_created",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "details": f"Incident created with severity {severity.value}",
                }
            ],
            response_actions_taken=[],
        )
        self._incidents[incident.incident_id] = incident
        return incident

    def update_status(
        self,
        incident_id: str,
        new_status: IncidentStatus,
        analyst_notes: str = "",
    ) -> bool:
        incident = self._incidents.get(incident_id)
        if incident is None:
            return False
        incident.status = new_status
        incident.updated_at = datetime.now(timezone.utc)
        if new_status in (IncidentStatus.REMEDIATED, IncidentStatus.CLOSED, IncidentStatus.FALSE_POSITIVE):
            incident.resolved_at = datetime.now(timezone.utc)
        incident.timeline.append(
            {
                "event": f"status_changed_to_{new_status.value}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": analyst_notes,
            }
        )
        return True

    def add_response_action(
        self,
        incident_id: str,
        action: ResponseAction,
        details: str = "",
    ) -> bool:
        incident = self._incidents.get(incident_id)
        if incident is None:
            return False
        incident.response_actions_taken.append(action.value)
        incident.timeline.append(
            {
                "event": "response_action",
                "action": action.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": details,
            }
        )
        incident.updated_at = datetime.now(timezone.utc)
        return True

    def assign_incident(self, incident_id: str, analyst_id: str) -> bool:
        incident = self._incidents.get(incident_id)
        if incident is None:
            return False
        incident.assigned_to = analyst_id
        incident.status = IncidentStatus.INVESTIGATING
        incident.timeline.append(
            {
                "event": "assigned",
                "analyst": analyst_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        return True

    def get_sla_status(self, incident_id: str) -> Dict[str, Any]:
        incident = self._incidents.get(incident_id)
        if incident is None:
            return {"error": "Incident not found"}
        sla_hours = self._sla_hours.get(incident.severity, 24)
        elapsed = datetime.now(timezone.utc) - incident.created_at
        elapsed_hours = elapsed.total_seconds() / 3600
        breached = elapsed_hours > sla_hours and incident.status not in (
            IncidentStatus.REMEDIATED, IncidentStatus.CLOSED
        )
        return {
            "incident_id": incident_id,
            "sla_hours": sla_hours,
            "elapsed_hours": round(elapsed_hours, 2),
            "remaining_hours": max(0.0, round(sla_hours - elapsed_hours, 2)),
            "sla_breached": breached,
        }

    def list_incidents(
        self,
        status_filter: Optional[IncidentStatus] = None,
        severity_filter: Optional[ThreatSeverity] = None,
    ) -> List[SecurityIncident]:
        incidents = list(self._incidents.values())
        if status_filter:
            incidents = [i for i in incidents if i.status == status_filter]
        if severity_filter:
            incidents = [i for i in incidents if i.severity == severity_filter]
        return sorted(incidents, key=lambda i: i.created_at, reverse=True)

    def get_incident_metrics(self) -> Dict[str, Any]:
        incidents = list(self._incidents.values())
        by_severity: Dict[str, int] = defaultdict(int)
        by_status: Dict[str, int] = defaultdict(int)
        resolved = [
            i for i in incidents if i.resolved_at is not None and i.created_at
        ]
        mttr_hours = 0.0
        if resolved:
            mttr_hours = (
                sum(
                    (i.resolved_at - i.created_at).total_seconds() / 3600
                    for i in resolved
                    if i.resolved_at
                )
                / len(resolved)
            )
        for i in incidents:
            by_severity[i.severity.value] += 1
            by_status[i.status.value] += 1
        return {
            "total_incidents": len(incidents),
            "by_severity": dict(by_severity),
            "by_status": dict(by_status),
            "mean_time_to_resolve_hours": round(mttr_hours, 2),
            "open_incidents": by_status.get("open", 0) + by_status.get("investigating", 0),
        }


# ─────────────────────── SIEM Event Correlator ──────────────────────────────


class SIEMCorrelator:
    """
    Correlates security events using temporal and entity-based grouping
    to detect multi-step attack chains.
    """

    def __init__(self, correlation_window_s: int = 300):
        self.correlation_window_s = correlation_window_s
        self._event_buffer: Deque[SecurityEvent] = deque(maxlen=5000)
        self._user_event_chains: Dict[str, List[SecurityEvent]] = defaultdict(list)
        self._ip_event_chains: Dict[str, List[SecurityEvent]] = defaultdict(list)
        self._correlation_id_counter = 0

    def ingest_event(self, event: SecurityEvent) -> None:
        self._event_buffer.append(event)
        if event.user_id:
            self._user_event_chains[event.user_id].append(event)
            # Keep only recent events
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.correlation_window_s)
            self._user_event_chains[event.user_id] = [
                e for e in self._user_event_chains[event.user_id]
                if e.timestamp > cutoff
            ]
        if event.source_ip:
            self._ip_event_chains[event.source_ip].append(event)
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.correlation_window_s)
            self._ip_event_chains[event.source_ip] = [
                e for e in self._ip_event_chains[event.source_ip]
                if e.timestamp > cutoff
            ]

    def detect_attack_chains(self) -> List[Dict[str, Any]]:
        chains = []

        # Pattern: Recon → Auth failure → Escalation
        for user_id, events in self._user_event_chains.items():
            if len(events) < 3:
                continue
            categories = [e.event_category for e in events]
            if (
                EventCategory.AUTHENTICATION in categories
                and EventCategory.PRIVILEGE_ESCALATION in categories
            ):
                chains.append(
                    {
                        "correlation_id": self._next_correlation_id(),
                        "pattern": "auth_to_escalation",
                        "entity": user_id,
                        "entity_type": "user",
                        "event_count": len(events),
                        "severity": ThreatSeverity.CRITICAL.value,
                        "events": [e.event_id for e in events[:10]],
                    }
                )

        # Pattern: High velocity from single IP
        for ip, events in self._ip_event_chains.items():
            if len(events) > 50:
                chains.append(
                    {
                        "correlation_id": self._next_correlation_id(),
                        "pattern": "high_velocity_ip",
                        "entity": ip,
                        "entity_type": "ip",
                        "event_count": len(events),
                        "severity": ThreatSeverity.HIGH.value,
                        "events": [e.event_id for e in events[:10]],
                    }
                )

        return chains

    def _next_correlation_id(self) -> str:
        self._correlation_id_counter += 1
        return f"CORR-{self._correlation_id_counter:06d}"

    def get_event_summary(self) -> Dict[str, Any]:
        events = list(self._event_buffer)
        by_category: Dict[str, int] = defaultdict(int)
        by_severity: Dict[str, int] = defaultdict(int)
        for e in events:
            by_category[e.event_category.value] += 1
            by_severity[e.severity.value] += 1
        return {
            "total_events_buffered": len(events),
            "by_category": dict(by_category),
            "by_severity": dict(by_severity),
            "unique_users": len(self._user_event_chains),
            "unique_ips": len(self._ip_event_chains),
        }


# ────────────────────── Security Operations Center ──────────────────────────


class SecurityOperationsCenter:
    """
    Master SOC engine integrating threat intel, behavioral analysis,
    detection rules, incident management, and SIEM correlation.
    """

    def __init__(self):
        self.threat_intel = ThreatIntelligenceEngine()
        self.anomaly_detector = BehavioralAnomalyDetector()
        self.rule_engine = DetectionRuleEngine()
        self.incident_mgr = IncidentManager()
        self.siem = SIEMCorrelator()
        self._blocked_ips: Set[str] = set()
        self._rate_limited_users: Dict[str, datetime] = {}
        self._auto_response_enabled = True

    def ingest_event(
        self,
        event_category: EventCategory,
        action: str,
        outcome: str,
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        raw_data: Optional[Dict[str, Any]] = None,
        content: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Check if IP is blocked
        if source_ip and source_ip in self._blocked_ips:
            return {
                "ingested": False,
                "reason": "IP blocked",
                "source_ip": source_ip,
            }

        # Compute risk score
        context: Dict[str, Any] = {}
        if user_id and user_id in self._rate_limited_users:
            context["rate_limited"] = True
        risk_score, risk_factors = self.threat_intel.compute_risk_score(
            source_ip, user_id, action, content, context
        )

        # Map risk score to severity
        severity = (
            ThreatSeverity.CRITICAL
            if risk_score >= 0.8
            else ThreatSeverity.HIGH
            if risk_score >= 0.6
            else ThreatSeverity.MEDIUM
            if risk_score >= 0.4
            else ThreatSeverity.LOW
            if risk_score >= 0.2
            else ThreatSeverity.INFORMATIONAL
        )

        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            event_category=event_category,
            severity=severity,
            source_ip=source_ip,
            destination_ip=None,
            user_id=user_id,
            resource=resource,
            action=action,
            outcome=outcome,
            raw_data=raw_data or {},
            risk_score=risk_score,
            tags=risk_factors,
        )

        # Record in SIEM and anomaly detector
        self.siem.ingest_event(event)
        self.anomaly_detector.record_event(event)

        # Behavioral anomaly analysis
        behavioral = {}
        if user_id:
            behavioral = self.anomaly_detector.analyze_user_behavior(user_id, event)
            event.risk_score = min(
                1.0, event.risk_score + behavioral.get("risk_boost", 0.0)
            )

        # Evaluate detection rules
        triggered_rules = self.rule_engine.evaluate(event)
        response_actions = set()
        for rule, actions in triggered_rules:
            for act in actions:
                response_actions.add(act)

        # Execute auto-responses
        auto_responses_executed = []
        if self._auto_response_enabled:
            auto_responses_executed = self._execute_responses(event, response_actions)

        # Auto-create incident for high-severity events
        incident_id = None
        if event.severity in (ThreatSeverity.CRITICAL, ThreatSeverity.HIGH) and triggered_rules:
            incident = self.incident_mgr.create_incident(
                title=f"{event.event_category.value.title()} Alert: {action}",
                description=(
                    f"Auto-created incident for {event.severity.value} event. "
                    f"Risk score: {event.risk_score:.2f}. Factors: {', '.join(risk_factors[:3])}"
                ),
                severity=event.severity,
                related_events=[event.event_id],
                affected_users=[user_id] if user_id else [],
                affected_resources=[resource] if resource else [],
            )
            incident_id = incident.incident_id

        return {
            "event_id": event.event_id,
            "risk_score": event.risk_score,
            "severity": event.severity.value,
            "risk_factors": risk_factors,
            "triggered_rules": len(triggered_rules),
            "response_actions": [a.value for a in response_actions],
            "auto_responses_executed": auto_responses_executed,
            "incident_id": incident_id,
            "behavioral_anomalies": behavioral.get("anomalies", []),
        }

    def _execute_responses(
        self,
        event: SecurityEvent,
        actions: Set[ResponseAction],
    ) -> List[str]:
        executed = []
        if ResponseAction.BLOCK_IP in actions and event.source_ip:
            self._blocked_ips.add(event.source_ip)
            executed.append(f"blocked_ip:{event.source_ip}")
        if ResponseAction.RATE_LIMIT in actions and event.user_id:
            self._rate_limited_users[event.user_id] = datetime.now(timezone.utc)
            executed.append(f"rate_limited_user:{event.user_id}")
        if ResponseAction.ALERT in actions:
            executed.append("alert_triggered")
        return executed

    def get_soc_dashboard(self) -> Dict[str, Any]:
        event_summary = self.siem.get_event_summary()
        incident_metrics = self.incident_mgr.get_incident_metrics()
        intel_stats = self.threat_intel.get_intel_stats()
        rule_stats = self.rule_engine.get_rule_stats()
        attack_chains = self.siem.detect_attack_chains()
        return {
            "event_summary": event_summary,
            "incident_metrics": incident_metrics,
            "threat_intel": intel_stats,
            "detection_rules": rule_stats,
            "active_attack_chains": len(attack_chains),
            "blocked_ips": len(self._blocked_ips),
            "rate_limited_users": len(self._rate_limited_users),
            "auto_response_enabled": self._auto_response_enabled,
        }

    def add_threat_indicator(
        self,
        indicator_type: ThreatIndicatorType,
        value: str,
        threat_name: str,
        severity: ThreatSeverity,
        confidence: float = 0.9,
        source: str = "manual",
    ) -> ThreatIndicator:
        indicator = ThreatIndicator(
            indicator_id=str(uuid.uuid4()),
            indicator_type=indicator_type,
            value=value,
            threat_name=threat_name,
            severity=severity,
            confidence=confidence,
            source=source,
            first_seen=datetime.now(timezone.utc),
            last_seen=datetime.now(timezone.utc),
        )
        self.threat_intel.add_indicator(indicator)
        return indicator

    def unblock_ip(self, ip: str) -> bool:
        self._blocked_ips.discard(ip)
        return True

    def get_blocked_ips(self) -> List[str]:
        return list(self._blocked_ips)
