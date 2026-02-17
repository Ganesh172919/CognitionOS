"""
Plugin Trust Scoring - Innovation Feature

Runtime plugin risk scoring based on multiple factors including code analysis,
execution history, community ratings, and behavioral patterns. Implements
policy-based execution gating and tenant-level overrides for flexible control.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4


class TrustFactorType(str, Enum):
    """Types of trust factors"""
    CODE_ANALYSIS = "code_analysis"
    EXECUTION_HISTORY = "execution_history"
    COMMUNITY_RATING = "community_rating"
    AUTHOR_REPUTATION = "author_reputation"
    PERMISSION_SCOPE = "permission_scope"
    SANDBOX_COMPLIANCE = "sandbox_compliance"
    UPDATE_FREQUENCY = "update_frequency"
    VULNERABILITY_SCAN = "vulnerability_scan"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"


class RiskLevel(str, Enum):
    """Risk levels for plugins"""
    MINIMAL = "minimal"        # 0-20: Very low risk
    LOW = "low"                # 21-40: Low risk
    MODERATE = "moderate"      # 41-60: Moderate risk
    HIGH = "high"              # 61-80: High risk
    CRITICAL = "critical"      # 81-100: Critical risk


class ExecutionPolicy(str, Enum):
    """Execution policies based on trust"""
    ALWAYS_ALLOW = "always_allow"        # No restrictions
    TRUSTED_ONLY = "trusted_only"        # Minimum trust threshold
    REVIEW_REQUIRED = "review_required"  # Manual review needed
    SANDBOXED_ONLY = "sandboxed_only"    # Must run in sandbox
    BLOCKED = "blocked"                  # Execution blocked


class BehaviorPattern(str, Enum):
    """Behavioral patterns for analysis"""
    NORMAL = "normal"
    SUSPICIOUS = "suspicious"
    ANOMALOUS = "anomalous"
    MALICIOUS = "malicious"


# ==================== Value Objects ====================

@dataclass(frozen=True)
class TrustFactor:
    """Individual trust factor contribution"""
    factor_type: TrustFactorType
    name: str
    score: float  # 0.0 - 1.0 (1.0 = most trustworthy)
    weight: float  # 0.0 - 1.0 (importance of this factor)
    confidence: float  # 0.0 - 1.0 (confidence in score)
    details: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for value, name in [
            (self.score, "score"),
            (self.weight, "weight"),
            (self.confidence, "confidence")
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")

    @property
    def weighted_score(self) -> float:
        """Calculate weighted contribution"""
        return self.score * self.weight * self.confidence

    @property
    def is_positive(self) -> bool:
        """Check if factor is positive (score > 0.5)"""
        return self.score > 0.5

    @property
    def is_confident(self) -> bool:
        """Check if factor has high confidence"""
        return self.confidence >= 0.7


@dataclass
class ExecutionConstraints:
    """Constraints for plugin execution"""
    max_memory_mb: int = 512
    max_cpu_percent: int = 50
    max_duration_seconds: int = 300
    network_access: bool = False
    filesystem_access: bool = False
    require_sandbox: bool = True
    allowed_apis: Set[str] = field(default_factory=set)
    blocked_apis: Set[str] = field(default_factory=set)

    def __post_init__(self):
        if self.max_memory_mb <= 0:
            raise ValueError("Max memory must be positive")
        if not 0 < self.max_cpu_percent <= 100:
            raise ValueError("Max CPU percent must be between 1 and 100")
        if self.max_duration_seconds <= 0:
            raise ValueError("Max duration must be positive")

    @staticmethod
    def for_risk_level(risk_level: RiskLevel) -> "ExecutionConstraints":
        """Create constraints based on risk level"""
        if risk_level == RiskLevel.MINIMAL:
            return ExecutionConstraints(
                max_memory_mb=1024,
                max_cpu_percent=80,
                max_duration_seconds=600,
                network_access=True,
                filesystem_access=True,
                require_sandbox=False
            )
        elif risk_level == RiskLevel.LOW:
            return ExecutionConstraints(
                max_memory_mb=512,
                max_cpu_percent=60,
                max_duration_seconds=300,
                network_access=True,
                filesystem_access=False,
                require_sandbox=True
            )
        elif risk_level == RiskLevel.MODERATE:
            return ExecutionConstraints(
                max_memory_mb=256,
                max_cpu_percent=40,
                max_duration_seconds=180,
                network_access=False,
                filesystem_access=False,
                require_sandbox=True
            )
        else:  # HIGH or CRITICAL
            return ExecutionConstraints(
                max_memory_mb=128,
                max_cpu_percent=25,
                max_duration_seconds=60,
                network_access=False,
                filesystem_access=False,
                require_sandbox=True,
                blocked_apis={"exec", "eval", "subprocess", "os.system"}
            )


# ==================== Entities ====================

@dataclass
class PluginTrustScore:
    """
    Comprehensive trust score for a plugin.
    
    Aggregates multiple trust factors into overall score and risk level.
    """
    id: UUID
    plugin_id: UUID
    score: int  # 0-100 (100 = most trustworthy, 0 = least)
    risk_score: int  # 0-100 (100 = highest risk, 0 = lowest)
    risk_level: RiskLevel
    factors: List[TrustFactor]
    execution_policy: ExecutionPolicy
    recommended_constraints: ExecutionConstraints
    calculation_metadata: Dict[str, Any] = field(default_factory=dict)
    calculated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=24))

    def __post_init__(self):
        if not 0 <= self.score <= 100:
            raise ValueError("Trust score must be between 0 and 100")
        if not 0 <= self.risk_score <= 100:
            raise ValueError("Risk score must be between 0 and 100")

    @staticmethod
    def calculate(
        plugin_id: UUID,
        factors: List[TrustFactor],
        baseline_policy: ExecutionPolicy = ExecutionPolicy.TRUSTED_ONLY
    ) -> "PluginTrustScore":
        """Calculate trust score from factors"""
        if not factors:
            raise ValueError("At least one trust factor required")
        
        # Calculate weighted average
        total_weighted_score = sum(f.weighted_score for f in factors)
        total_weight = sum(f.weight * f.confidence for f in factors)
        
        normalized_score = (total_weighted_score / total_weight) if total_weight > 0 else 0.0
        trust_score = int(normalized_score * 100)
        
        # Risk score is inverse of trust
        risk_score = 100 - trust_score
        
        # Determine risk level
        risk_level = PluginTrustScore._calculate_risk_level(risk_score)
        
        # Determine execution policy
        execution_policy = PluginTrustScore._determine_policy(trust_score, risk_level, baseline_policy)
        
        # Create recommended constraints
        recommended_constraints = ExecutionConstraints.for_risk_level(risk_level)
        
        # Gather metadata
        metadata = {
            "factor_count": len(factors),
            "positive_factors": sum(1 for f in factors if f.is_positive),
            "negative_factors": sum(1 for f in factors if not f.is_positive),
            "high_confidence_factors": sum(1 for f in factors if f.is_confident),
            "average_confidence": sum(f.confidence for f in factors) / len(factors)
        }
        
        return PluginTrustScore(
            id=uuid4(),
            plugin_id=plugin_id,
            score=trust_score,
            risk_score=risk_score,
            risk_level=risk_level,
            factors=factors,
            execution_policy=execution_policy,
            recommended_constraints=recommended_constraints,
            calculation_metadata=metadata
        )

    @staticmethod
    def _calculate_risk_level(risk_score: int) -> RiskLevel:
        """Calculate risk level from score"""
        if risk_score <= 20:
            return RiskLevel.MINIMAL
        elif risk_score <= 40:
            return RiskLevel.LOW
        elif risk_score <= 60:
            return RiskLevel.MODERATE
        elif risk_score <= 80:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    @staticmethod
    def _determine_policy(
        trust_score: int,
        risk_level: RiskLevel,
        baseline_policy: ExecutionPolicy
    ) -> ExecutionPolicy:
        """Determine execution policy"""
        if risk_level == RiskLevel.CRITICAL:
            return ExecutionPolicy.BLOCKED
        elif risk_level == RiskLevel.HIGH:
            return ExecutionPolicy.REVIEW_REQUIRED
        elif risk_level == RiskLevel.MODERATE:
            return ExecutionPolicy.SANDBOXED_ONLY
        elif trust_score >= 70:
            return ExecutionPolicy.ALWAYS_ALLOW
        else:
            return baseline_policy

    @property
    def is_trusted(self) -> bool:
        """Check if plugin is trusted (score >= 70)"""
        return self.score >= 70

    @property
    def is_expired(self) -> bool:
        """Check if score has expired"""
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def can_execute(self) -> bool:
        """Check if plugin can execute"""
        return self.execution_policy not in [
            ExecutionPolicy.BLOCKED,
            ExecutionPolicy.REVIEW_REQUIRED
        ]

    def get_factor(self, factor_type: TrustFactorType) -> Optional[TrustFactor]:
        """Get specific trust factor"""
        return next((f for f in self.factors if f.factor_type == factor_type), None)

    def get_critical_factors(self) -> List[TrustFactor]:
        """Get factors that significantly impact score"""
        return [
            f for f in self.factors
            if f.weight >= 0.7 or (not f.is_positive and f.weight >= 0.5)
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence"""
        return {
            "id": str(self.id),
            "plugin_id": str(self.plugin_id),
            "score": self.score,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level.value,
            "execution_policy": self.execution_policy.value,
            "is_trusted": self.is_trusted,
            "can_execute": self.can_execute,
            "factor_summary": {
                "total": len(self.factors),
                "positive": self.calculation_metadata.get("positive_factors", 0),
                "negative": self.calculation_metadata.get("negative_factors", 0)
            },
            "calculated_at": self.calculated_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "metadata": self.calculation_metadata
        }


@dataclass
class TenantPolicyOverride:
    """
    Tenant-specific policy override for plugin execution.
    
    Allows tenants to customize trust requirements.
    """
    id: UUID
    tenant_id: UUID
    plugin_id: Optional[UUID]  # None for global tenant policy
    override_policy: ExecutionPolicy
    min_trust_score: int
    custom_constraints: Optional[ExecutionConstraints]
    reason: str
    approved_by: UUID
    approved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0 <= self.min_trust_score <= 100:
            raise ValueError("Min trust score must be between 0 and 100")

    @staticmethod
    def create(
        tenant_id: UUID,
        override_policy: ExecutionPolicy,
        min_trust_score: int,
        reason: str,
        approved_by: UUID,
        plugin_id: Optional[UUID] = None,
        custom_constraints: Optional[ExecutionConstraints] = None,
        expires_at: Optional[datetime] = None
    ) -> "TenantPolicyOverride":
        """Create policy override"""
        return TenantPolicyOverride(
            id=uuid4(),
            tenant_id=tenant_id,
            plugin_id=plugin_id,
            override_policy=override_policy,
            min_trust_score=min_trust_score,
            custom_constraints=custom_constraints,
            reason=reason,
            approved_by=approved_by,
            expires_at=expires_at
        )

    @property
    def is_expired(self) -> bool:
        """Check if override has expired"""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_active(self) -> bool:
        """Check if override is currently active"""
        return not self.is_expired

    @property
    def applies_to_plugin(self) -> bool:
        """Check if override applies to specific plugin"""
        return self.plugin_id is not None

    def applies_to(self, plugin_id: UUID) -> bool:
        """Check if override applies to given plugin"""
        if self.plugin_id is None:
            return True  # Global override
        return self.plugin_id == plugin_id


@dataclass
class BehavioralAnomalyAlert:
    """
    Alert for detected behavioral anomaly.
    
    Tracks suspicious or anomalous plugin behavior at runtime.
    """
    id: UUID
    plugin_id: UUID
    tenant_id: UUID
    execution_id: UUID
    pattern: BehaviorPattern
    severity: int  # 0-100
    description: str
    evidence: Dict[str, Any]
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None

    def __post_init__(self):
        if not 0 <= self.severity <= 100:
            raise ValueError("Severity must be between 0 and 100")

    @staticmethod
    def create(
        plugin_id: UUID,
        tenant_id: UUID,
        execution_id: UUID,
        pattern: BehaviorPattern,
        severity: int,
        description: str,
        evidence: Dict[str, Any]
    ) -> "BehavioralAnomalyAlert":
        """Create behavioral anomaly alert"""
        return BehavioralAnomalyAlert(
            id=uuid4(),
            plugin_id=plugin_id,
            tenant_id=tenant_id,
            execution_id=execution_id,
            pattern=pattern,
            severity=severity,
            description=description,
            evidence=evidence
        )

    @property
    def is_critical(self) -> bool:
        """Check if alert is critical"""
        return self.severity >= 80 or self.pattern == BehaviorPattern.MALICIOUS

    def resolve(self, notes: str) -> None:
        """Mark alert as resolved"""
        self.resolved = True
        self.resolved_at = datetime.now(timezone.utc)
        self.resolution_notes = notes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence"""
        return {
            "id": str(self.id),
            "plugin_id": str(self.plugin_id),
            "tenant_id": str(self.tenant_id),
            "execution_id": str(self.execution_id),
            "pattern": self.pattern.value,
            "severity": self.severity,
            "description": self.description,
            "is_critical": self.is_critical,
            "resolved": self.resolved,
            "detected_at": self.detected_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


# ==================== Service ====================

class PluginTrustScoringService:
    """
    Plugin trust scoring service.
    
    Calculates trust scores and enforces execution policies.
    """

    def __init__(self):
        """Initialize plugin trust scoring service"""
        self._score_cache: Dict[UUID, PluginTrustScore] = {}
        self._tenant_overrides: Dict[UUID, List[TenantPolicyOverride]] = {}
        self._anomaly_alerts: List[BehavioralAnomalyAlert] = []

    async def calculate_trust_score(
        self,
        plugin_id: UUID,
        code_content: Optional[str] = None,
        execution_history: Optional[Dict[str, Any]] = None,
        community_data: Optional[Dict[str, Any]] = None
    ) -> PluginTrustScore:
        """
        Calculate comprehensive trust score for plugin.
        
        Args:
            plugin_id: Plugin identifier
            code_content: Plugin source code for analysis
            execution_history: Historical execution data
            community_data: Community ratings and feedback
            
        Returns:
            Plugin trust score
        """
        factors = []
        
        # Code analysis factor
        if code_content:
            code_factor = await self._analyze_code(code_content)
            factors.append(code_factor)
        
        # Execution history factor
        if execution_history:
            history_factor = await self._analyze_execution_history(execution_history)
            factors.append(history_factor)
        
        # Community rating factor
        if community_data:
            community_factor = await self._analyze_community_data(community_data)
            factors.append(community_factor)
        
        # Permission scope factor
        permission_factor = await self._analyze_permissions(plugin_id)
        factors.append(permission_factor)
        
        # If no factors available, use conservative defaults
        if not factors:
            factors = [TrustFactor(
                factor_type=TrustFactorType.CODE_ANALYSIS,
                name="Default Score",
                score=0.3,
                weight=1.0,
                confidence=0.5,
                details="No analysis data available - using conservative default"
            )]
        
        # Calculate score
        trust_score = PluginTrustScore.calculate(plugin_id, factors)
        
        # Cache score
        self._score_cache[plugin_id] = trust_score
        
        return trust_score

    async def evaluate_execution_request(
        self,
        plugin_id: UUID,
        tenant_id: UUID,
        trust_score: Optional[PluginTrustScore] = None
    ) -> Tuple[bool, ExecutionConstraints, str]:
        """
        Evaluate if plugin execution should be allowed.
        
        Args:
            plugin_id: Plugin identifier
            tenant_id: Tenant identifier
            trust_score: Pre-calculated trust score (optional)
            
        Returns:
            Tuple of (allowed, constraints, reason)
        """
        # Get or calculate trust score
        if not trust_score:
            trust_score = self._score_cache.get(plugin_id)
            if not trust_score or trust_score.is_expired:
                trust_score = await self.calculate_trust_score(plugin_id)
        
        # Check for tenant-specific overrides
        override = self._get_active_override(tenant_id, plugin_id)
        if override:
            if override.override_policy == ExecutionPolicy.BLOCKED:
                return False, trust_score.recommended_constraints, f"Blocked by tenant policy: {override.reason}"
            
            if trust_score.score < override.min_trust_score:
                return False, trust_score.recommended_constraints, f"Trust score {trust_score.score} below tenant minimum {override.min_trust_score}"
            
            constraints = override.custom_constraints or trust_score.recommended_constraints
            return True, constraints, f"Allowed by tenant override (score: {trust_score.score})"
        
        # Apply default policy
        if not trust_score.can_execute:
            reason = f"Execution policy: {trust_score.execution_policy.value}"
            if trust_score.execution_policy == ExecutionPolicy.BLOCKED:
                reason += f" - Risk level: {trust_score.risk_level.value}"
            return False, trust_score.recommended_constraints, reason
        
        return True, trust_score.recommended_constraints, f"Trusted (score: {trust_score.score})"

    async def report_behavioral_anomaly(
        self,
        plugin_id: UUID,
        tenant_id: UUID,
        execution_id: UUID,
        anomaly_type: BehaviorPattern,
        description: str,
        evidence: Dict[str, Any]
    ) -> BehavioralAnomalyAlert:
        """
        Report behavioral anomaly detected during execution.
        
        Args:
            plugin_id: Plugin identifier
            tenant_id: Tenant identifier
            execution_id: Execution identifier
            anomaly_type: Type of anomaly
            description: Anomaly description
            evidence: Evidence data
            
        Returns:
            Anomaly alert
        """
        # Calculate severity based on pattern
        severity_map = {
            BehaviorPattern.NORMAL: 0,
            BehaviorPattern.SUSPICIOUS: 40,
            BehaviorPattern.ANOMALOUS: 70,
            BehaviorPattern.MALICIOUS: 95
        }
        severity = severity_map[anomaly_type]
        
        # Create alert
        alert = BehavioralAnomalyAlert.create(
            plugin_id=plugin_id,
            tenant_id=tenant_id,
            execution_id=execution_id,
            pattern=anomaly_type,
            severity=severity,
            description=description,
            evidence=evidence
        )
        
        self._anomaly_alerts.append(alert)
        
        # If critical, invalidate trust score cache
        if alert.is_critical:
            if plugin_id in self._score_cache:
                del self._score_cache[plugin_id]
        
        return alert

    async def add_tenant_override(
        self,
        override: TenantPolicyOverride
    ) -> None:
        """Add tenant policy override"""
        tenant_id = override.tenant_id
        if tenant_id not in self._tenant_overrides:
            self._tenant_overrides[tenant_id] = []
        self._tenant_overrides[tenant_id].append(override)

    async def get_plugin_risk_summary(
        self,
        plugin_id: UUID
    ) -> Dict[str, Any]:
        """
        Get comprehensive risk summary for plugin.
        
        Args:
            plugin_id: Plugin identifier
            
        Returns:
            Risk summary dictionary
        """
        trust_score = self._score_cache.get(plugin_id)
        
        # Get anomaly alerts for plugin
        alerts = [a for a in self._anomaly_alerts if a.plugin_id == plugin_id]
        unresolved_alerts = [a for a in alerts if not a.resolved]
        critical_alerts = [a for a in unresolved_alerts if a.is_critical]
        
        summary = {
            "plugin_id": str(plugin_id),
            "has_trust_score": trust_score is not None,
            "total_anomaly_alerts": len(alerts),
            "unresolved_alerts": len(unresolved_alerts),
            "critical_alerts": len(critical_alerts),
            "recommendation": "Review required" if critical_alerts else "Monitoring"
        }
        
        if trust_score:
            summary.update({
                "trust_score": trust_score.score,
                "risk_score": trust_score.risk_score,
                "risk_level": trust_score.risk_level.value,
                "execution_policy": trust_score.execution_policy.value,
                "is_trusted": trust_score.is_trusted,
                "can_execute": trust_score.can_execute,
                "score_age_hours": (datetime.now(timezone.utc) - trust_score.calculated_at).total_seconds() / 3600
            })
        
        return summary

    # Private helper methods

    async def _analyze_code(self, code_content: str) -> TrustFactor:
        """Analyze code for trust factor"""
        # Simplified code analysis
        risk_indicators = [
            "eval(", "exec(", "os.system", "__import__",
            "subprocess", "pickle.loads", "yaml.load"
        ]
        
        indicator_count = sum(1 for indicator in risk_indicators if indicator in code_content)
        
        # Score decreases with risk indicators
        score = max(0.0, 1.0 - (indicator_count * 0.15))
        
        details = f"Code analysis: {indicator_count} risk indicators found"
        if indicator_count > 0:
            details += f" - Review required"
        
        return TrustFactor(
            factor_type=TrustFactorType.CODE_ANALYSIS,
            name="Static Code Analysis",
            score=score,
            weight=0.8,
            confidence=0.9,
            details=details,
            evidence={"risk_indicator_count": indicator_count}
        )

    async def _analyze_execution_history(self, history: Dict[str, Any]) -> TrustFactor:
        """Analyze execution history for trust factor"""
        total_executions = history.get("total_executions", 0)
        successful_executions = history.get("successful_executions", 0)
        failed_executions = history.get("failed_executions", 0)
        
        if total_executions == 0:
            score = 0.5
            confidence = 0.3
            details = "No execution history available"
        else:
            success_rate = successful_executions / total_executions
            score = success_rate
            confidence = min(total_executions / 100, 1.0)  # More executions = higher confidence
            details = f"Success rate: {success_rate:.1%} ({successful_executions}/{total_executions})"
        
        return TrustFactor(
            factor_type=TrustFactorType.EXECUTION_HISTORY,
            name="Execution History",
            score=score,
            weight=0.7,
            confidence=confidence,
            details=details,
            evidence=history
        )

    async def _analyze_community_data(self, community_data: Dict[str, Any]) -> TrustFactor:
        """Analyze community data for trust factor"""
        rating = community_data.get("average_rating", 0.0)  # 0-5 scale
        review_count = community_data.get("review_count", 0)
        install_count = community_data.get("install_count", 0)
        
        # Normalize rating to 0-1
        normalized_rating = rating / 5.0 if rating > 0 else 0.5
        
        # Confidence increases with reviews
        confidence = min(review_count / 50, 1.0)
        
        details = f"Community rating: {rating:.1f}/5.0 ({review_count} reviews, {install_count} installs)"
        
        return TrustFactor(
            factor_type=TrustFactorType.COMMUNITY_RATING,
            name="Community Trust",
            score=normalized_rating,
            weight=0.6,
            confidence=confidence,
            details=details,
            evidence=community_data
        )

    async def _analyze_permissions(self, plugin_id: UUID) -> TrustFactor:
        """Analyze permission scope for trust factor"""
        # Simplified permission analysis
        # Would check actual plugin manifest
        
        # Assume moderate permissions for now
        score = 0.6
        details = "Permission scope: Moderate"
        
        return TrustFactor(
            factor_type=TrustFactorType.PERMISSION_SCOPE,
            name="Permission Scope",
            score=score,
            weight=0.7,
            confidence=0.8,
            details=details
        )

    def _get_active_override(
        self,
        tenant_id: UUID,
        plugin_id: UUID
    ) -> Optional[TenantPolicyOverride]:
        """Get active policy override for tenant and plugin"""
        overrides = self._tenant_overrides.get(tenant_id, [])
        
        # First check plugin-specific override
        plugin_override = next(
            (o for o in overrides if o.applies_to(plugin_id) and o.is_active and o.applies_to_plugin),
            None
        )
        if plugin_override:
            return plugin_override
        
        # Then check global tenant override
        global_override = next(
            (o for o in overrides if o.applies_to(plugin_id) and o.is_active and not o.applies_to_plugin),
            None
        )
        return global_override
