"""
Abuse Detection and Prevention System

Real-time detection of abusive patterns, automated responses,
and security threat mitigation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class AbuseType(str, Enum):
    """Types of abuse"""
    RATE_LIMIT_VIOLATION = "rate_limit_violation"
    QUOTA_MANIPULATION = "quota_manipulation"
    API_SCRAPING = "api_scraping"
    CREDENTIAL_STUFFING = "credential_stuffing"
    FAKE_ACCOUNT = "fake_account"
    PAYMENT_FRAUD = "payment_fraud"
    RESOURCE_HOGGING = "resource_hogging"
    SPAM = "spam"
    DATA_EXFILTRATION = "data_exfiltration"


class AbuseAction(str, Enum):
    """Actions to take on abuse"""
    LOG = "log"
    WARN = "warn"
    THROTTLE = "throttle"
    TEMPORARY_BAN = "temporary_ban"
    PERMANENT_BAN = "permanent_ban"
    REQUIRE_VERIFICATION = "require_verification"
    ESCALATE = "escalate"


class Severity(str, Enum):
    """Abuse severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AbusePattern:
    """Detected abuse pattern"""
    pattern_id: str
    abuse_type: AbuseType
    severity: Severity
    description: str

    # Detection criteria
    threshold: float
    time_window_seconds: int

    # Response
    action: AbuseAction
    ban_duration_hours: Optional[int] = None

    # Tracking
    detection_count: int = 0
    last_detected: Optional[datetime] = None


@dataclass
class AbuseIncident:
    """Single abuse incident"""
    incident_id: str
    tenant_id: str
    user_id: Optional[str]
    abuse_type: AbuseType
    severity: Severity
    detected_at: datetime

    # Details
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    pattern_id: Optional[str] = None

    # Response
    action_taken: AbuseAction = AbuseAction.LOG
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    notes: str = ""


class AbuseDetector:
    """
    Real-time abuse detection system

    Monitors user behavior for abusive patterns and takes
    automated action to protect the platform.
    """

    def __init__(self, storage_backend: Optional[Any] = None):
        self.storage = storage_backend
        self._patterns: Dict[str, AbusePattern] = {}
        self._incidents: List[AbuseIncident] = []
        self._tenant_violations: Dict[str, List[datetime]] = defaultdict(list)
        self._banned_tenants: Dict[str, datetime] = {}
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize abuse detection patterns"""

        # Rate limit violations
        self._patterns["rate_spike"] = AbusePattern(
            pattern_id="rate_spike",
            abuse_type=AbuseType.RATE_LIMIT_VIOLATION,
            severity=Severity.MEDIUM,
            description="Sudden spike in API requests",
            threshold=1000.0,  # 1000 req/min
            time_window_seconds=60,
            action=AbuseAction.THROTTLE
        )

        # Quota manipulation
        self._patterns["quota_reset_abuse"] = AbusePattern(
            pattern_id="quota_reset_abuse",
            abuse_type=AbuseType.QUOTA_MANIPULATION,
            severity=Severity.HIGH,
            description="Multiple account creations to reset quotas",
            threshold=5.0,  # 5 accounts from same IP
            time_window_seconds=3600,
            action=AbuseAction.TEMPORARY_BAN,
            ban_duration_hours=24
        )

        # API scraping
        self._patterns["systematic_scraping"] = AbusePattern(
            pattern_id="systematic_scraping",
            abuse_type=AbuseType.API_SCRAPING,
            severity=Severity.HIGH,
            description="Systematic data extraction patterns",
            threshold=10000.0,  # 10K requests/hour
            time_window_seconds=3600,
            action=AbuseAction.THROTTLE
        )

        # Credential stuffing
        self._patterns["credential_stuffing"] = AbusePattern(
            pattern_id="credential_stuffing",
            abuse_type=AbuseType.CREDENTIAL_STUFFING,
            severity=Severity.CRITICAL,
            description="Multiple failed login attempts",
            threshold=10.0,  # 10 failed logins
            time_window_seconds=300,
            action=AbuseAction.TEMPORARY_BAN,
            ban_duration_hours=1
        )

        # Payment fraud
        self._patterns["payment_fraud"] = AbusePattern(
            pattern_id="payment_fraud",
            abuse_type=AbuseType.PAYMENT_FRAUD,
            severity=Severity.CRITICAL,
            description="Multiple failed payment attempts",
            threshold=3.0,
            time_window_seconds=3600,
            action=AbuseAction.REQUIRE_VERIFICATION
        )

        # Resource hogging
        self._patterns["resource_hogging"] = AbusePattern(
            pattern_id="resource_hogging",
            abuse_type=AbuseType.RESOURCE_HOGGING,
            severity=Severity.HIGH,
            description="Excessive compute resource usage",
            threshold=100.0,  # 100 compute hours/day
            time_window_seconds=86400,
            action=AbuseAction.THROTTLE
        )

        # Data exfiltration
        self._patterns["data_exfiltration"] = AbusePattern(
            pattern_id="data_exfiltration",
            abuse_type=AbuseType.DATA_EXFILTRATION,
            severity=Severity.CRITICAL,
            description="Unusual data export patterns",
            threshold=1000.0,  # 1GB/hour
            time_window_seconds=3600,
            action=AbuseAction.ESCALATE
        )

    async def check_abuse(
        self,
        tenant_id: str,
        activity_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[AbuseIncident]:
        """
        Check for abusive behavior

        Args:
            tenant_id: Tenant to check
            activity_type: Type of activity
            metadata: Additional context

        Returns:
            AbuseIncident if abuse detected, None otherwise
        """

        # Check if tenant is banned
        if await self._is_banned(tenant_id):
            return self._create_incident(
                tenant_id,
                AbuseType.RATE_LIMIT_VIOLATION,
                Severity.HIGH,
                "Attempt to use service while banned",
                metadata or {}
            )

        # Check relevant patterns based on activity
        for pattern in self._get_relevant_patterns(activity_type):
            if await self._check_pattern(tenant_id, pattern, metadata):
                incident = self._create_incident(
                    tenant_id,
                    pattern.abuse_type,
                    pattern.severity,
                    pattern.description,
                    metadata or {},
                    pattern.pattern_id
                )

                # Take action
                await self._take_action(incident, pattern)

                return incident

        return None

    async def _is_banned(self, tenant_id: str) -> bool:
        """Check if tenant is currently banned"""
        if tenant_id in self._banned_tenants:
            ban_until = self._banned_tenants[tenant_id]
            if datetime.utcnow() < ban_until:
                return True
            else:
                # Ban expired, remove
                del self._banned_tenants[tenant_id]
        return False

    def _get_relevant_patterns(self, activity_type: str) -> List[AbusePattern]:
        """Get patterns relevant to activity type"""
        # Simple mapping - would be more sophisticated in production
        pattern_map = {
            "api_request": ["rate_spike", "systematic_scraping"],
            "login": ["credential_stuffing"],
            "payment": ["payment_fraud"],
            "compute": ["resource_hogging"],
            "export": ["data_exfiltration"]
        }

        pattern_ids = pattern_map.get(activity_type, [])
        return [self._patterns[pid] for pid in pattern_ids if pid in self._patterns]

    async def _check_pattern(
        self,
        tenant_id: str,
        pattern: AbusePattern,
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if pattern matches"""

        # Get recent activity for tenant
        violations = self._tenant_violations.get(tenant_id, [])

        # Filter to time window
        cutoff = datetime.utcnow() - timedelta(seconds=pattern.time_window_seconds)
        recent_violations = [v for v in violations if v > cutoff]

        # Check if threshold exceeded
        if len(recent_violations) >= pattern.threshold:
            pattern.detection_count += 1
            pattern.last_detected = datetime.utcnow()
            logger.warning(
                f"Pattern {pattern.pattern_id} detected for {tenant_id}: "
                f"{len(recent_violations)} violations in {pattern.time_window_seconds}s"
            )
            return True

        # Record this activity
        violations.append(datetime.utcnow())
        self._tenant_violations[tenant_id] = violations

        return False

    def _create_incident(
        self,
        tenant_id: str,
        abuse_type: AbuseType,
        severity: Severity,
        description: str,
        evidence: Dict[str, Any],
        pattern_id: Optional[str] = None
    ) -> AbuseIncident:
        """Create abuse incident"""

        incident = AbuseIncident(
            incident_id=f"incident_{len(self._incidents)}_{int(datetime.utcnow().timestamp())}",
            tenant_id=tenant_id,
            user_id=evidence.get("user_id"),
            abuse_type=abuse_type,
            severity=severity,
            detected_at=datetime.utcnow(),
            description=description,
            evidence=evidence,
            pattern_id=pattern_id
        )

        self._incidents.append(incident)

        if self.storage:
            # Would persist to database
            pass

        return incident

    async def _take_action(self, incident: AbuseIncident, pattern: AbusePattern):
        """Take action on abuse incident"""

        action = pattern.action
        incident.action_taken = action

        logger.warning(
            f"Taking action {action.value} on {incident.tenant_id} for {incident.abuse_type.value}"
        )

        if action == AbuseAction.LOG:
            # Already logged
            pass

        elif action == AbuseAction.WARN:
            await self._send_warning(incident)

        elif action == AbuseAction.THROTTLE:
            await self._apply_throttle(incident)

        elif action == AbuseAction.TEMPORARY_BAN:
            await self._apply_ban(incident, pattern.ban_duration_hours or 24)

        elif action == AbuseAction.PERMANENT_BAN:
            await self._apply_ban(incident, hours=None)

        elif action == AbuseAction.REQUIRE_VERIFICATION:
            await self._require_verification(incident)

        elif action == AbuseAction.ESCALATE:
            await self._escalate_to_security_team(incident)

    async def _send_warning(self, incident: AbuseIncident):
        """Send warning to tenant"""
        logger.info(f"Sending warning to {incident.tenant_id}")

    async def _apply_throttle(self, incident: AbuseIncident):
        """Apply rate throttling"""
        logger.info(f"Applying throttle to {incident.tenant_id}")
        # Would integrate with rate limiter to reduce limits

    async def _apply_ban(self, incident: AbuseIncident, hours: Optional[int]):
        """Apply temporary or permanent ban"""
        if hours:
            ban_until = datetime.utcnow() + timedelta(hours=hours)
            self._banned_tenants[incident.tenant_id] = ban_until
            logger.warning(f"Banned {incident.tenant_id} until {ban_until}")
        else:
            # Permanent ban
            ban_until = datetime.utcnow() + timedelta(days=3650)  # 10 years
            self._banned_tenants[incident.tenant_id] = ban_until
            logger.critical(f"PERMANENTLY banned {incident.tenant_id}")

    async def _require_verification(self, incident: AbuseIncident):
        """Require additional verification"""
        logger.info(f"Requiring verification for {incident.tenant_id}")
        # Would trigger verification flow

    async def _escalate_to_security_team(self, incident: AbuseIncident):
        """Escalate to security team"""
        logger.critical(f"ESCALATING incident {incident.incident_id} to security team")
        # Would create ticket, send alerts

    def get_incidents(
        self,
        tenant_id: Optional[str] = None,
        severity: Optional[Severity] = None,
        resolved: Optional[bool] = None,
        limit: int = 100
    ) -> List[AbuseIncident]:
        """Get abuse incidents"""

        incidents = self._incidents

        if tenant_id:
            incidents = [i for i in incidents if i.tenant_id == tenant_id]

        if severity:
            incidents = [i for i in incidents if i.severity == severity]

        if resolved is not None:
            incidents = [i for i in incidents if i.resolved == resolved]

        return sorted(incidents, key=lambda i: i.detected_at, reverse=True)[:limit]

    def get_abuse_statistics(self) -> Dict[str, Any]:
        """Get abuse statistics"""

        total_incidents = len(self._incidents)
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        by_action = defaultdict(int)

        for incident in self._incidents:
            by_type[incident.abuse_type.value] += 1
            by_severity[incident.severity.value] += 1
            by_action[incident.action_taken.value] += 1

        return {
            "total_incidents": total_incidents,
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "by_action": dict(by_action),
            "active_bans": len(self._banned_tenants),
            "patterns": {
                pid: {
                    "detection_count": p.detection_count,
                    "last_detected": p.last_detected.isoformat() if p.last_detected else None
                }
                for pid, p in self._patterns.items()
            }
        }

    async def resolve_incident(
        self,
        incident_id: str,
        notes: Optional[str] = None
    ) -> bool:
        """Mark incident as resolved"""

        incident = next((i for i in self._incidents if i.incident_id == incident_id), None)
        if not incident:
            return False

        incident.resolved = True
        incident.resolved_at = datetime.utcnow()
        if notes:
            incident.notes = notes

        logger.info(f"Resolved incident {incident_id}")
        return True

    async def whitelist_tenant(self, tenant_id: str):
        """Whitelist tenant (bypass abuse detection)"""
        # Would maintain whitelist
        logger.info(f"Whitelisted {tenant_id}")

    async def unban_tenant(self, tenant_id: str):
        """Remove ban from tenant"""
        if tenant_id in self._banned_tenants:
            del self._banned_tenants[tenant_id]
            logger.info(f"Unbanned {tenant_id}")
            return True
        return False
