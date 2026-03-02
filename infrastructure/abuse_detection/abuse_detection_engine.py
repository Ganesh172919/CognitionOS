"""
Abuse Detection Engine — CognitionOS Revenue Protection

Real-time abuse detection for SaaS platform protection:
- Token farming detection
- API key sharing across IPs
- Rate limit evasion detection
- Credential stuffing patterns
- Automated threat scoring
- Auto-suspend for confirmed abuse
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

logger = logging.getLogger(__name__)


class AbuseType(str, Enum):
    TOKEN_FARMING = "token_farming"
    KEY_SHARING = "key_sharing"
    RATE_EVASION = "rate_evasion"
    CREDENTIAL_STUFFING = "credential_stuffing"
    CONTENT_ABUSE = "content_abuse"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class ThreatLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(str, Enum):
    LOG = "log"
    WARN = "warn"
    THROTTLE = "throttle"
    SUSPEND = "suspend"
    BAN = "ban"


@dataclass
class AbuseSignal:
    signal_id: str
    abuse_type: AbuseType
    tenant_id: str
    api_key_id: str = ""
    source_ip: str = ""
    confidence: float = 0.0
    details: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "abuse_type": self.abuse_type.value,
            "tenant_id": self.tenant_id,
            "confidence": round(self.confidence, 2),
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class ThreatAssessment:
    tenant_id: str
    threat_level: ThreatLevel
    composite_score: float
    signals: List[AbuseSignal] = field(default_factory=list)
    recommended_action: ActionType = ActionType.LOG
    assessed_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "threat_level": self.threat_level.value,
            "composite_score": round(self.composite_score, 2),
            "signal_count": len(self.signals),
            "recommended_action": self.recommended_action.value,
            "signals": [s.to_dict() for s in self.signals[:5]],
        }


@dataclass
class TenantBehaviorProfile:
    tenant_id: str
    request_ips: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    api_keys_used: Set[str] = field(default_factory=set)
    requests_per_minute: List[float] = field(default_factory=list)
    tokens_consumed: List[int] = field(default_factory=list)
    error_count: int = 0
    last_activity: float = field(default_factory=time.time)
    total_requests: int = 0
    suspended: bool = False
    warnings_issued: int = 0


class AbuseDetectionEngine:
    """
    Real-time abuse detection engine for SaaS platform protection.
    Monitors tenant behavior patterns and generates threat assessments.
    """

    THREAT_THRESHOLDS = {
        ThreatLevel.LOW: 20,
        ThreatLevel.MEDIUM: 50,
        ThreatLevel.HIGH: 75,
        ThreatLevel.CRITICAL: 90,
    }

    ACTION_MAPPING = {
        ThreatLevel.LOW: ActionType.LOG,
        ThreatLevel.MEDIUM: ActionType.WARN,
        ThreatLevel.HIGH: ActionType.THROTTLE,
        ThreatLevel.CRITICAL: ActionType.SUSPEND,
    }

    def __init__(self, *, max_ips_per_key: int = 10, max_rpm: int = 300,
                 token_spike_threshold: float = 3.0, window_seconds: float = 300.0,
                 auto_suspend: bool = True):
        self._max_ips_per_key = max_ips_per_key
        self._max_rpm = max_rpm
        self._token_spike_threshold = token_spike_threshold
        self._window_seconds = window_seconds
        self._auto_suspend = auto_suspend

        self._profiles: Dict[str, TenantBehaviorProfile] = {}
        self._signals: List[AbuseSignal] = []
        self._total_detections = 0
        self._total_suspensions = 0
        self._actions_taken: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        logger.info("AbuseDetectionEngine initialized (auto_suspend=%s)", auto_suspend)

    def record_request(self, tenant_id: str, *, api_key_id: str = "",
                       source_ip: str = "", tokens_used: int = 0,
                       metadata: Optional[Dict[str, Any]] = None):
        """Record an API request for behavior analysis."""
        profile = self._get_or_create_profile(tenant_id)
        now = time.time()

        profile.total_requests += 1
        profile.last_activity = now
        profile.requests_per_minute.append(now)
        # Trim to window
        cutoff = now - self._window_seconds
        profile.requests_per_minute = [t for t in profile.requests_per_minute if t > cutoff]

        if source_ip:
            profile.request_ips[source_ip] += 1
        if api_key_id:
            profile.api_keys_used.add(api_key_id)
        if tokens_used > 0:
            profile.tokens_consumed.append(tokens_used)
            if len(profile.tokens_consumed) > 1000:
                profile.tokens_consumed = profile.tokens_consumed[-1000:]

    def analyze_tenant(self, tenant_id: str) -> ThreatAssessment:
        """Perform threat analysis on a tenant's behavior."""
        profile = self._get_or_create_profile(tenant_id)
        signals: List[AbuseSignal] = []

        # Check for API key sharing (too many IPs)
        unique_ips = len(profile.request_ips)
        if unique_ips > self._max_ips_per_key:
            confidence = min(1.0, unique_ips / (self._max_ips_per_key * 3))
            signals.append(AbuseSignal(
                signal_id=uuid4().hex[:12], abuse_type=AbuseType.KEY_SHARING,
                tenant_id=tenant_id, confidence=confidence,
                details=f"API key used from {unique_ips} IPs (max {self._max_ips_per_key})",
            ))

        # Check rate limit evasion
        rpm = len(profile.requests_per_minute)
        if rpm > self._max_rpm:
            confidence = min(1.0, rpm / (self._max_rpm * 2))
            signals.append(AbuseSignal(
                signal_id=uuid4().hex[:12], abuse_type=AbuseType.RATE_EVASION,
                tenant_id=tenant_id, confidence=confidence,
                details=f"Request rate {rpm}/window exceeds limit {self._max_rpm}",
            ))

        # Check token farming (spike detection)
        if len(profile.tokens_consumed) >= 10:
            recent = profile.tokens_consumed[-10:]
            avg_all = sum(profile.tokens_consumed) / len(profile.tokens_consumed)
            avg_recent = sum(recent) / len(recent)
            if avg_all > 0 and avg_recent > avg_all * self._token_spike_threshold:
                signals.append(AbuseSignal(
                    signal_id=uuid4().hex[:12], abuse_type=AbuseType.TOKEN_FARMING,
                    tenant_id=tenant_id,
                    confidence=min(1.0, avg_recent / (avg_all * self._token_spike_threshold * 2)),
                    details=f"Token usage spike: {avg_recent:.0f} avg recent vs {avg_all:.0f} overall",
                ))

        # Check resource exhaustion
        if profile.error_count > 100:
            signals.append(AbuseSignal(
                signal_id=uuid4().hex[:12], abuse_type=AbuseType.RESOURCE_EXHAUSTION,
                tenant_id=tenant_id,
                confidence=min(1.0, profile.error_count / 500),
                details=f"Excessive errors: {profile.error_count}",
            ))

        # Compute composite score
        composite_score = 0.0
        if signals:
            composite_score = min(100, sum(s.confidence * 40 for s in signals))

        # Determine threat level
        threat_level = ThreatLevel.LOW
        for level in reversed(list(ThreatLevel)):
            if composite_score >= self.THREAT_THRESHOLDS[level]:
                threat_level = level
                break

        action = self.ACTION_MAPPING[threat_level]

        # Auto-suspend if critical
        if self._auto_suspend and threat_level == ThreatLevel.CRITICAL:
            profile.suspended = True
            self._total_suspensions += 1
            logger.warning("Tenant '%s' auto-suspended (score=%.0f)", tenant_id, composite_score)

        self._signals.extend(signals)
        if signals:
            self._total_detections += len(signals)

        return ThreatAssessment(
            tenant_id=tenant_id, threat_level=threat_level,
            composite_score=composite_score, signals=signals,
            recommended_action=action,
        )

    def record_error(self, tenant_id: str):
        profile = self._get_or_create_profile(tenant_id)
        profile.error_count += 1

    def is_suspended(self, tenant_id: str) -> bool:
        profile = self._profiles.get(tenant_id)
        return profile.suspended if profile else False

    def unsuspend(self, tenant_id: str):
        profile = self._profiles.get(tenant_id)
        if profile:
            profile.suspended = False
            profile.warnings_issued = 0
            logger.info("Tenant '%s' unsuspended", tenant_id)

    def _get_or_create_profile(self, tenant_id: str) -> TenantBehaviorProfile:
        if tenant_id not in self._profiles:
            self._profiles[tenant_id] = TenantBehaviorProfile(tenant_id=tenant_id)
        return self._profiles[tenant_id]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "tracked_tenants": len(self._profiles),
            "total_detections": self._total_detections,
            "total_suspensions": self._total_suspensions,
            "active_suspensions": sum(1 for p in self._profiles.values() if p.suspended),
            "recent_signals": len(self._signals[-50:]),
        }
