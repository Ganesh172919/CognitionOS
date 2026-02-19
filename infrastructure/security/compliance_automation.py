"""
Compliance Automation Engine
Automated compliance checking and reporting for SOC2, GDPR, HIPAA, and other standards.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field


class ComplianceStandard(str, Enum):
    """Compliance standards supported"""
    SOC2 = "soc2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    CCPA = "ccpa"


class ControlCategory(str, Enum):
    """Security control categories"""
    ACCESS_CONTROL = "access_control"
    ENCRYPTION = "encryption"
    AUDIT_LOGGING = "audit_logging"
    DATA_PROTECTION = "data_protection"
    INCIDENT_RESPONSE = "incident_response"
    NETWORK_SECURITY = "network_security"
    PHYSICAL_SECURITY = "physical_security"
    RISK_MANAGEMENT = "risk_management"


class ComplianceStatus(str, Enum):
    """Compliance check status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_APPLICABLE = "not_applicable"
    PENDING_REVIEW = "pending_review"


@dataclass
class ComplianceControl:
    """Individual compliance control"""
    control_id: str
    standard: ComplianceStandard
    category: ControlCategory
    title: str
    description: str
    requirement: str
    automated_check: bool = True
    criticality: str = "medium"  # low, medium, high, critical
    evidence_required: List[str] = field(default_factory=list)


@dataclass
class ComplianceCheckResult:
    """Result of a compliance check"""
    check_id: str
    control_id: str
    status: ComplianceStatus
    timestamp: datetime
    evidence: Dict[str, Any] = field(default_factory=dict)
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    remediation_steps: List[str] = field(default_factory=list)


class ComplianceReport(BaseModel):
    """Comprehensive compliance report"""
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    standard: ComplianceStandard
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    period_start: datetime
    period_end: datetime
    overall_status: ComplianceStatus
    total_controls: int
    compliant_controls: int
    non_compliant_controls: int
    compliance_score: float  # 0.0 to 1.0
    check_results: List[ComplianceCheckResult] = Field(default_factory=list)
    critical_findings: List[str] = Field(default_factory=list)
    summary: str = ""


class ComplianceAutomationEngine:
    """
    Automated compliance checking and reporting system.
    Supports multiple compliance standards with automated controls.
    """

    def __init__(self):
        self.controls: Dict[str, ComplianceControl] = {}
        self.check_history: List[ComplianceCheckResult] = []
        self.reports: Dict[str, ComplianceReport] = {}
        self._initialize_controls()

    def _initialize_controls(self):
        """Initialize compliance controls for various standards"""

        # SOC2 Controls
        soc2_controls = [
            ComplianceControl(
                control_id="SOC2-CC6.1",
                standard=ComplianceStandard.SOC2,
                category=ControlCategory.ACCESS_CONTROL,
                title="Logical Access Controls",
                description="System access is restricted to authorized users",
                requirement="Implement authentication and authorization mechanisms",
                automated_check=True,
                criticality="high",
                evidence_required=["access_logs", "user_list", "permission_matrix"]
            ),
            ComplianceControl(
                control_id="SOC2-CC6.6",
                standard=ComplianceStandard.SOC2,
                category=ControlCategory.ENCRYPTION,
                title="Data Encryption",
                description="Data is encrypted at rest and in transit",
                requirement="Implement TLS 1.2+ and AES-256 encryption",
                automated_check=True,
                criticality="critical",
                evidence_required=["encryption_config", "tls_certificate"]
            ),
            ComplianceControl(
                control_id="SOC2-CC7.2",
                standard=ComplianceStandard.SOC2,
                category=ControlCategory.AUDIT_LOGGING,
                title="System Activity Monitoring",
                description="System activities are logged and monitored",
                requirement="Comprehensive audit logging with retention",
                automated_check=True,
                criticality="high",
                evidence_required=["audit_logs", "log_retention_policy"]
            ),
        ]

        # GDPR Controls
        gdpr_controls = [
            ComplianceControl(
                control_id="GDPR-Art32",
                standard=ComplianceStandard.GDPR,
                category=ControlCategory.DATA_PROTECTION,
                title="Security of Processing",
                description="Appropriate security measures for personal data",
                requirement="Implement encryption, pseudonymization, resilience",
                automated_check=True,
                criticality="critical",
                evidence_required=["security_measures", "encryption_status"]
            ),
            ComplianceControl(
                control_id="GDPR-Art33",
                standard=ComplianceStandard.GDPR,
                category=ControlCategory.INCIDENT_RESPONSE,
                title="Breach Notification",
                description="Data breach notification within 72 hours",
                requirement="Incident response and notification procedures",
                automated_check=False,
                criticality="critical",
                evidence_required=["incident_response_plan", "notification_templates"]
            ),
            ComplianceControl(
                control_id="GDPR-Art25",
                standard=ComplianceStandard.GDPR,
                category=ControlCategory.DATA_PROTECTION,
                title="Data Protection by Design",
                description="Privacy by design and default",
                requirement="Minimize data collection, maximize protection",
                automated_check=True,
                criticality="high",
                evidence_required=["data_minimization_policy", "privacy_settings"]
            ),
        ]

        # HIPAA Controls
        hipaa_controls = [
            ComplianceControl(
                control_id="HIPAA-164.308",
                standard=ComplianceStandard.HIPAA,
                category=ControlCategory.ACCESS_CONTROL,
                title="Administrative Safeguards",
                description="Access control and workforce training",
                requirement="Implement role-based access control",
                automated_check=True,
                criticality="critical",
                evidence_required=["access_control_policy", "training_records"]
            ),
            ComplianceControl(
                control_id="HIPAA-164.312",
                standard=ComplianceStandard.HIPAA,
                category=ControlCategory.ENCRYPTION,
                title="Technical Safeguards",
                description="Encryption and integrity controls",
                requirement="Encrypt ePHI at rest and in transit",
                automated_check=True,
                criticality="critical",
                evidence_required=["encryption_implementation", "integrity_controls"]
            ),
        ]

        # PCI DSS Controls
        pci_controls = [
            ComplianceControl(
                control_id="PCI-3.4",
                standard=ComplianceStandard.PCI_DSS,
                category=ControlCategory.ENCRYPTION,
                title="Cardholder Data Protection",
                description="Render PAN unreadable",
                requirement="Strong cryptography for cardholder data",
                automated_check=True,
                criticality="critical",
                evidence_required=["encryption_config", "key_management"]
            ),
            ComplianceControl(
                control_id="PCI-10.1",
                standard=ComplianceStandard.PCI_DSS,
                category=ControlCategory.AUDIT_LOGGING,
                title="Audit Trail",
                description="Log all access to cardholder data",
                requirement="Comprehensive audit logging mechanism",
                automated_check=True,
                criticality="critical",
                evidence_required=["audit_logs", "log_review_records"]
            ),
        ]

        # Register all controls
        all_controls = soc2_controls + gdpr_controls + hipaa_controls + pci_controls
        for control in all_controls:
            self.controls[control.control_id] = control

    async def run_compliance_check(
        self,
        control_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ComplianceCheckResult:
        """
        Run automated compliance check for a specific control
        """
        if control_id not in self.controls:
            raise ValueError(f"Control {control_id} not found")

        control = self.controls[control_id]
        check_id = str(uuid4())
        timestamp = datetime.utcnow()

        # Run automated checks based on control type
        if control.automated_check:
            status, evidence, findings, recommendations = await self._automated_check(
                control, context or {}
            )
        else:
            status = ComplianceStatus.PENDING_REVIEW
            evidence = {}
            findings = ["Manual review required"]
            recommendations = ["Schedule compliance review"]

        # Generate remediation steps if non-compliant
        remediation_steps = []
        if status == ComplianceStatus.NON_COMPLIANT:
            remediation_steps = self._generate_remediation_steps(control, findings)

        result = ComplianceCheckResult(
            check_id=check_id,
            control_id=control_id,
            status=status,
            timestamp=timestamp,
            evidence=evidence,
            findings=findings,
            recommendations=recommendations,
            remediation_steps=remediation_steps
        )

        self.check_history.append(result)
        return result

    async def _automated_check(
        self,
        control: ComplianceControl,
        context: Dict[str, Any]
    ) -> tuple:
        """
        Perform automated compliance check
        Returns: (status, evidence, findings, recommendations)
        """
        findings = []
        recommendations = []
        evidence = {}

        # Access Control Checks
        if control.category == ControlCategory.ACCESS_CONTROL:
            # Check authentication mechanisms
            has_mfa = context.get("mfa_enabled", False)
            has_rbac = context.get("rbac_enabled", False)
            password_policy = context.get("password_policy_strength", "weak")

            evidence["mfa_enabled"] = has_mfa
            evidence["rbac_enabled"] = has_rbac
            evidence["password_policy"] = password_policy

            if not has_mfa:
                findings.append("Multi-factor authentication not enabled")
                recommendations.append("Enable MFA for all user accounts")

            if not has_rbac:
                findings.append("Role-based access control not implemented")
                recommendations.append("Implement RBAC with least privilege")

            if password_policy != "strong":
                findings.append("Weak password policy detected")
                recommendations.append("Enforce strong password requirements")

            status = ComplianceStatus.COMPLIANT if not findings else ComplianceStatus.NON_COMPLIANT

        # Encryption Checks
        elif control.category == ControlCategory.ENCRYPTION:
            encryption_at_rest = context.get("encryption_at_rest", False)
            encryption_in_transit = context.get("encryption_in_transit", False)
            tls_version = context.get("tls_version", "1.0")
            encryption_algorithm = context.get("encryption_algorithm", "AES-128")

            evidence["encryption_at_rest"] = encryption_at_rest
            evidence["encryption_in_transit"] = encryption_in_transit
            evidence["tls_version"] = tls_version
            evidence["encryption_algorithm"] = encryption_algorithm

            if not encryption_at_rest:
                findings.append("Data not encrypted at rest")
                recommendations.append("Enable AES-256 encryption for data at rest")

            if not encryption_in_transit:
                findings.append("Data not encrypted in transit")
                recommendations.append("Enable TLS 1.2+ for all communications")

            if float(tls_version) < 1.2:
                findings.append(f"TLS version {tls_version} is outdated")
                recommendations.append("Upgrade to TLS 1.2 or higher")

            if "128" in encryption_algorithm:
                findings.append("Encryption algorithm strength insufficient")
                recommendations.append("Use AES-256 or equivalent")

            status = ComplianceStatus.COMPLIANT if not findings else ComplianceStatus.NON_COMPLIANT

        # Audit Logging Checks
        elif control.category == ControlCategory.AUDIT_LOGGING:
            logging_enabled = context.get("audit_logging_enabled", False)
            log_retention_days = context.get("log_retention_days", 0)
            log_integrity_protection = context.get("log_integrity_protection", False)

            evidence["logging_enabled"] = logging_enabled
            evidence["log_retention_days"] = log_retention_days
            evidence["log_integrity_protection"] = log_integrity_protection

            if not logging_enabled:
                findings.append("Audit logging not enabled")
                recommendations.append("Enable comprehensive audit logging")

            if log_retention_days < 90:
                findings.append(f"Log retention period too short: {log_retention_days} days")
                recommendations.append("Set log retention to at least 90 days")

            if not log_integrity_protection:
                findings.append("Log integrity protection not implemented")
                recommendations.append("Implement tamper-proof logging mechanism")

            status = ComplianceStatus.COMPLIANT if not findings else ComplianceStatus.NON_COMPLIANT

        # Data Protection Checks
        elif control.category == ControlCategory.DATA_PROTECTION:
            data_classification = context.get("data_classification_implemented", False)
            data_minimization = context.get("data_minimization_policy", False)
            anonymization = context.get("anonymization_capability", False)

            evidence["data_classification"] = data_classification
            evidence["data_minimization"] = data_minimization
            evidence["anonymization"] = anonymization

            if not data_classification:
                findings.append("Data classification not implemented")
                recommendations.append("Implement data classification scheme")

            if not data_minimization:
                findings.append("Data minimization policy not enforced")
                recommendations.append("Implement data minimization procedures")

            if not anonymization:
                findings.append("Data anonymization capability missing")
                recommendations.append("Implement data anonymization for sensitive data")

            status = ComplianceStatus.COMPLIANT if not findings else ComplianceStatus.NON_COMPLIANT

        else:
            # Default for other categories
            status = ComplianceStatus.PENDING_REVIEW
            findings.append("Automated check not available for this control")
            recommendations.append("Manual review required")

        return status, evidence, findings, recommendations

    def _generate_remediation_steps(
        self,
        control: ComplianceControl,
        findings: List[str]
    ) -> List[str]:
        """Generate remediation steps based on findings"""
        steps = []

        for finding in findings:
            if "mfa" in finding.lower() or "multi-factor" in finding.lower():
                steps.extend([
                    "1. Select MFA provider (e.g., Duo, Okta, Auth0)",
                    "2. Configure MFA in authentication system",
                    "3. Enforce MFA for all user roles",
                    "4. Provide user MFA setup documentation"
                ])

            elif "encryption" in finding.lower():
                steps.extend([
                    "1. Implement AES-256 encryption for data at rest",
                    "2. Configure TLS 1.2+ for data in transit",
                    "3. Implement key management system",
                    "4. Rotate encryption keys regularly"
                ])

            elif "logging" in finding.lower():
                steps.extend([
                    "1. Enable comprehensive audit logging",
                    "2. Configure log retention policy (90+ days)",
                    "3. Implement log integrity protection",
                    "4. Set up log monitoring and alerting"
                ])

            elif "rbac" in finding.lower():
                steps.extend([
                    "1. Define roles and permissions matrix",
                    "2. Implement role-based access control",
                    "3. Apply principle of least privilege",
                    "4. Regular access reviews"
                ])

        return list(set(steps))  # Remove duplicates

    async def generate_compliance_report(
        self,
        standard: ComplianceStandard,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ComplianceReport:
        """
        Generate comprehensive compliance report for a standard
        """
        if period_end is None:
            period_end = datetime.utcnow()
        if period_start is None:
            period_start = period_end - timedelta(days=30)

        # Get controls for this standard
        standard_controls = [
            c for c in self.controls.values()
            if c.standard == standard
        ]

        # Run checks for all controls
        check_results = []
        for control in standard_controls:
            result = await self.run_compliance_check(
                control.control_id,
                context
            )
            check_results.append(result)

        # Calculate compliance metrics
        total_controls = len(check_results)
        compliant_controls = len([
            r for r in check_results
            if r.status == ComplianceStatus.COMPLIANT
        ])
        non_compliant_controls = len([
            r for r in check_results
            if r.status == ComplianceStatus.NON_COMPLIANT
        ])

        compliance_score = compliant_controls / total_controls if total_controls > 0 else 0.0

        # Determine overall status
        if compliance_score >= 0.95:
            overall_status = ComplianceStatus.COMPLIANT
        elif compliance_score >= 0.70:
            overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            overall_status = ComplianceStatus.NON_COMPLIANT

        # Collect critical findings
        critical_findings = []
        for result in check_results:
            control = self.controls[result.control_id]
            if control.criticality == "critical" and result.status == ComplianceStatus.NON_COMPLIANT:
                critical_findings.extend(result.findings)

        # Generate summary
        summary = self._generate_report_summary(
            standard, compliance_score, critical_findings
        )

        report = ComplianceReport(
            standard=standard,
            period_start=period_start,
            period_end=period_end,
            overall_status=overall_status,
            total_controls=total_controls,
            compliant_controls=compliant_controls,
            non_compliant_controls=non_compliant_controls,
            compliance_score=compliance_score,
            check_results=check_results,
            critical_findings=critical_findings,
            summary=summary
        )

        self.reports[report.report_id] = report
        return report

    def _generate_report_summary(
        self,
        standard: ComplianceStandard,
        compliance_score: float,
        critical_findings: List[str]
    ) -> str:
        """Generate executive summary for compliance report"""
        score_pct = compliance_score * 100

        summary = f"""
Compliance Report Summary - {standard.value.upper()}

Overall Compliance Score: {score_pct:.1f}%

Status: {"COMPLIANT" if compliance_score >= 0.95 else "PARTIALLY COMPLIANT" if compliance_score >= 0.70 else "NON-COMPLIANT"}

Critical Issues: {len(critical_findings)}

The organization {"meets" if compliance_score >= 0.95 else "partially meets" if compliance_score >= 0.70 else "does not meet"} \
the requirements for {standard.value.upper()} compliance.

{"Immediate attention required for critical findings." if critical_findings else "No critical issues identified."}

Recommended Actions:
- Address all critical findings within 30 days
- Implement remediation steps for non-compliant controls
- Schedule follow-up assessment in 90 days
- Maintain continuous compliance monitoring
        """.strip()

        return summary

    async def continuous_monitoring(
        self,
        standard: ComplianceStandard,
        interval_hours: int = 24
    ) -> None:
        """
        Continuous compliance monitoring (simplified - in production would run as background task)
        """
        # This would be implemented as a background task in production
        # For now, just log the intent
        pass

    def get_control_details(self, control_id: str) -> Optional[ComplianceControl]:
        """Get details of a specific control"""
        return self.controls.get(control_id)

    def get_controls_by_standard(
        self,
        standard: ComplianceStandard
    ) -> List[ComplianceControl]:
        """Get all controls for a compliance standard"""
        return [
            c for c in self.controls.values()
            if c.standard == standard
        ]

    def get_recent_checks(
        self,
        limit: int = 10
    ) -> List[ComplianceCheckResult]:
        """Get recent compliance check results"""
        return sorted(
            self.check_history,
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]

    def get_compliance_dashboard_data(self) -> Dict[str, Any]:
        """Get data for compliance dashboard"""
        # Calculate metrics by standard
        by_standard = {}
        for standard in ComplianceStandard:
            controls = self.get_controls_by_standard(standard)
            recent_checks = [
                c for c in self.check_history
                if c.control_id in [ctrl.control_id for ctrl in controls]
            ]

            if recent_checks:
                compliant = len([
                    c for c in recent_checks
                    if c.status == ComplianceStatus.COMPLIANT
                ])
                score = compliant / len(recent_checks) if recent_checks else 0

                by_standard[standard.value] = {
                    "total_controls": len(controls),
                    "recent_checks": len(recent_checks),
                    "compliance_score": round(score * 100, 1),
                    "status": "compliant" if score >= 0.95 else "partial" if score >= 0.70 else "non_compliant"
                }

        return {
            "standards": by_standard,
            "total_checks_run": len(self.check_history),
            "total_controls": len(self.controls),
            "last_check": self.check_history[-1].timestamp.isoformat() if self.check_history else None
        }
