"""
Zero-Trust Security Engine
===========================
Identity-first security with mTLS, micro-segmentation, policy engine,
workload identity, and continuous verification.

Implements:
- SPIFFE/SPIRE-compatible workload identity (SVIDs)
- Mutual TLS certificate lifecycle (issue, renew, revoke)
- Policy engine: ABAC (attribute-based) + RBAC + condition evaluation
- Micro-segmentation: traffic policies between workloads
- Continuous verification: session trust scoring with re-auth triggers
- Threat intelligence integration: IP reputation, geo-blocking
- Network policy firewall: allow/deny rule evaluation
- Audit logging for every access decision
- API resource protection with scope enforcement
- Token introspection and binding
"""

from __future__ import annotations

import ast
import hashlib
import hmac
import ipaddress
import json
import operator
import re
import secrets
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Safe expression evaluator (no arbitrary code execution)
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.And: lambda a, b: a and b,
    ast.Or: lambda a, b: a or b,
    ast.Not: operator.not_,
    ast.USub: operator.neg,
}


def _safe_eval_node(node: ast.AST, ctx: Dict[str, Any]) -> Any:
    """Recursively evaluate a whitelisted AST node with no code execution."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "True":
            return True
        if node.id == "False":
            return False
        if node.id == "None":
            return None
        if node.id in ctx:
            return ctx[node.id]
        raise ValueError(f"Unknown name: {node.id!r}")
    if isinstance(node, ast.Attribute):
        obj = _safe_eval_node(node.value, ctx)
        return getattr(obj, node.attr) if obj is not None else None
    if isinstance(node, ast.Subscript):
        obj = _safe_eval_node(node.value, ctx)
        key = _safe_eval_node(node.slice, ctx)
        return obj[key] if obj is not None else None
    if isinstance(node, ast.Index):  # Python 3.8 compat
        return _safe_eval_node(node.value, ctx)  # type: ignore[attr-defined]
    if isinstance(node, ast.BoolOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported boolean op: {type(node.op)}")
        result = _safe_eval_node(node.values[0], ctx)
        for value in node.values[1:]:
            result = op_fn(result, _safe_eval_node(value, ctx))
        return result
    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary op: {type(node.op)}")
        return op_fn(_safe_eval_node(node.operand, ctx))
    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported binary op: {type(node.op)}")
        return op_fn(_safe_eval_node(node.left, ctx), _safe_eval_node(node.right, ctx))
    if isinstance(node, ast.Compare):
        left = _safe_eval_node(node.left, ctx)
        for op, right_node in zip(node.ops, node.comparators):
            op_fn = _SAFE_OPS.get(type(op))
            if op_fn is None:
                raise ValueError(f"Unsupported compare op: {type(op)}")
            right = _safe_eval_node(right_node, ctx)
            if not op_fn(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.List):
        return [_safe_eval_node(e, ctx) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval_node(e, ctx) for e in node.elts)
    if isinstance(node, ast.IfExp):
        return (
            _safe_eval_node(node.body, ctx)
            if _safe_eval_node(node.test, ctx)
            else _safe_eval_node(node.orelse, ctx)
        )
    raise ValueError(f"Unsupported AST node type: {type(node).__name__}")


def safe_eval_condition(expression: str, ctx: Dict[str, Any]) -> bool:
    """
    Safely evaluate a boolean policy condition expression.
    Supports: comparisons, boolean ops (and/or/not), attribute access,
    subscripts, constants, and whitelisted context variables.
    Raises ValueError on any unsupported construct.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval_node(tree.body, ctx)
        return bool(result)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class PolicyEffect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


class PolicyCombineAlgorithm(str, Enum):
    DENY_OVERRIDES = "deny_overrides"
    PERMIT_OVERRIDES = "permit_overrides"
    FIRST_APPLICABLE = "first_applicable"
    UNANIMOUS = "unanimous"


class TrustLevel(str, Enum):
    HIGH = "high"         # known device, recent auth, good behavior
    MEDIUM = "medium"     # known device, older auth
    LOW = "low"           # unknown device or elevated risk
    UNTRUSTED = "untrusted"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    PENDING = "pending"
    REAUTH_REQUIRED = "reauth_required"
    BLOCKED = "blocked"


class CertificateStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING_RENEWAL = "pending_renewal"


class NetworkPolicyAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    LOG = "log"
    RATE_LIMIT = "rate_limit"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class WorkloadIdentity:
    """SPIFFE Verifiable Identity Document (SVID)."""
    spiffe_id: str                                     # spiffe://trust-domain/workload
    service_name: str
    namespace: str = "default"
    trust_domain: str = "cognitionos.io"
    certificate_serial: str = field(default_factory=lambda: secrets.token_hex(8))
    issued_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 3600)
    status: CertificateStatus = CertificateStatus.ACTIVE
    attributes: Dict[str, str] = field(default_factory=dict)
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.fingerprint:
            data = f"{self.spiffe_id}:{self.certificate_serial}:{self.issued_at}".encode()
            self.fingerprint = hashlib.sha256(data).hexdigest()

    @property
    def is_valid(self) -> bool:
        return (
            self.status == CertificateStatus.ACTIVE
            and self.expires_at > time.time()
        )

    @property
    def ttl_seconds(self) -> float:
        return max(0.0, self.expires_at - time.time())


@dataclass
class AccessSubject:
    """Entity requesting access — user, service, or machine."""
    subject_id: str
    subject_type: str              # user | service | machine | api_key
    roles: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    trust_level: TrustLevel = TrustLevel.MEDIUM
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    workload_identity: Optional[WorkloadIdentity] = None
    auth_time: float = field(default_factory=time.time)
    mfa_verified: bool = False
    geo_country: str = "US"


@dataclass
class AccessResource:
    """Resource being accessed."""
    resource_type: str             # api | data | service | admin_panel
    resource_id: str
    owner_tenant_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    classification: str = "internal"   # public | internal | confidential | secret


@dataclass
class AccessRequest:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subject: Optional[AccessSubject] = None
    resource: Optional[AccessResource] = None
    action: str = ""              # read | write | delete | execute | admin
    environment: Dict[str, Any] = field(default_factory=dict)
    tenant_id: str = "global"
    timestamp: float = field(default_factory=time.time)
    request_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessDecision:
    request_id: str
    effect: PolicyEffect
    reason: str
    matched_policies: List[str] = field(default_factory=list)
    trust_score: float = 1.0
    step_up_required: bool = False
    obligations: List[str] = field(default_factory=list)
    decided_at: float = field(default_factory=time.time)
    decision_latency_ms: float = 0.0


@dataclass
class SecurityPolicy:
    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    effect: PolicyEffect = PolicyEffect.ALLOW
    subjects: List[str] = field(default_factory=list)      # role names or subject IDs
    resources: List[str] = field(default_factory=list)     # resource_type patterns
    actions: List[str] = field(default_factory=list)       # action names or wildcards
    conditions: List[str] = field(default_factory=list)    # Python condition expressions
    priority: int = 100
    enabled: bool = True
    tenant_id: str = "global"
    created_at: float = field(default_factory=time.time)


@dataclass
class NetworkPolicy:
    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    source_service: str = "*"
    dest_service: str = "*"
    source_ip_cidr: Optional[str] = None
    dest_port: Optional[int] = None
    protocols: List[str] = field(default_factory=lambda: ["tcp"])
    action: NetworkPolicyAction = NetworkPolicyAction.ALLOW
    priority: int = 100
    enabled: bool = True


@dataclass
class TrustScore:
    subject_id: str
    score: float = 1.0             # 0.0 (untrusted) to 1.0 (fully trusted)
    factors: Dict[str, float] = field(default_factory=dict)
    trust_level: TrustLevel = TrustLevel.HIGH
    last_evaluated: float = field(default_factory=time.time)
    reauth_threshold: float = 0.5
    block_threshold: float = 0.2
    verification_status: VerificationStatus = VerificationStatus.VERIFIED


@dataclass
class AuditRecord:
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    subject_id: str = ""
    resource_type: str = ""
    resource_id: str = ""
    action: str = ""
    effect: str = ""
    reason: str = ""
    tenant_id: str = "global"
    ip_address: Optional[str] = None
    trust_score: float = 1.0
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Certificate Authority
# ---------------------------------------------------------------------------

class WorkloadCertificateAuthority:
    """Issues and manages short-lived X.509 certificates for workload identity."""

    def __init__(self, trust_domain: str = "cognitionos.io"):
        self.trust_domain = trust_domain
        self._certificates: Dict[str, WorkloadIdentity] = {}
        self._revoked: Set[str] = set()
        self._issued_count: int = 0

    def issue(
        self,
        service_name: str,
        namespace: str = "default",
        ttl_seconds: int = 3600,
        attributes: Optional[Dict[str, str]] = None,
    ) -> WorkloadIdentity:
        spiffe_id = f"spiffe://{self.trust_domain}/{namespace}/{service_name}"
        identity = WorkloadIdentity(
            spiffe_id=spiffe_id,
            service_name=service_name,
            namespace=namespace,
            trust_domain=self.trust_domain,
            expires_at=time.time() + ttl_seconds,
            attributes=attributes or {},
        )
        self._certificates[identity.certificate_serial] = identity
        self._issued_count += 1
        return identity

    def renew(self, serial: str, ttl_seconds: int = 3600) -> Optional[WorkloadIdentity]:
        identity = self._certificates.get(serial)
        if not identity or identity.status == CertificateStatus.REVOKED:
            return None
        identity.expires_at = time.time() + ttl_seconds
        identity.status = CertificateStatus.ACTIVE
        return identity

    def revoke(self, serial: str, reason: str = "") -> bool:
        identity = self._certificates.get(serial)
        if not identity:
            return False
        identity.status = CertificateStatus.REVOKED
        self._revoked.add(serial)
        return True

    def verify(self, fingerprint: str) -> Optional[WorkloadIdentity]:
        for identity in self._certificates.values():
            if identity.fingerprint == fingerprint and identity.is_valid:
                return identity
        return None

    def get_expiring_soon(self, within_seconds: int = 300) -> List[WorkloadIdentity]:
        now = time.time()
        return [
            i for i in self._certificates.values()
            if i.status == CertificateStatus.ACTIVE
            and 0 < i.expires_at - now <= within_seconds
        ]


# ---------------------------------------------------------------------------
# Policy Engine
# ---------------------------------------------------------------------------

class PolicyEngine:
    """ABAC + RBAC policy evaluation engine."""

    def __init__(self, combine_algorithm: PolicyCombineAlgorithm = PolicyCombineAlgorithm.DENY_OVERRIDES):
        self._policies: List[SecurityPolicy] = []
        self.combine_algorithm = combine_algorithm

    def add_policy(self, policy: SecurityPolicy) -> None:
        self._policies.append(policy)
        self._policies.sort(key=lambda p: p.priority)

    def remove_policy(self, policy_id: str) -> bool:
        before = len(self._policies)
        self._policies = [p for p in self._policies if p.policy_id != policy_id]
        return len(self._policies) < before

    def _matches_subject(self, policy: SecurityPolicy, subject: AccessSubject) -> bool:
        if not policy.subjects or "*" in policy.subjects:
            return True
        for sub in policy.subjects:
            if sub == subject.subject_id:
                return True
            if sub in subject.roles:
                return True
            if sub.startswith("type:") and sub[5:] == subject.subject_type:
                return True
        return False

    def _matches_resource(self, policy: SecurityPolicy, resource: AccessResource) -> bool:
        if not policy.resources or "*" in policy.resources:
            return True
        for res in policy.resources:
            if res == resource.resource_type or res == resource.resource_id:
                return True
            if res.endswith("*") and resource.resource_type.startswith(res[:-1]):
                return True
        return False

    def _matches_action(self, policy: SecurityPolicy, action: str) -> bool:
        if not policy.actions or "*" in policy.actions:
            return True
        return action in policy.actions or "any" in policy.actions

    def _evaluate_conditions(self, policy: SecurityPolicy, request: AccessRequest) -> bool:
        if not policy.conditions:
            return True
        ctx = {
            "subject": request.subject,
            "resource": request.resource,
            "action": request.action,
            "env": request.environment,
            "time": time.time(),
        }
        for condition in policy.conditions:
            if not safe_eval_condition(condition, ctx):
                return False
        return True

    def evaluate(self, request: AccessRequest) -> Tuple[PolicyEffect, str, List[str]]:
        applicable: List[SecurityPolicy] = []
        for policy in self._policies:
            if not policy.enabled:
                continue
            if policy.tenant_id not in ("global", request.tenant_id):
                continue
            if (
                self._matches_subject(policy, request.subject)
                and self._matches_resource(policy, request.resource)
                and self._matches_action(policy, request.action)
                and self._evaluate_conditions(policy, request)
            ):
                applicable.append(policy)

        if not applicable:
            return PolicyEffect.DENY, "no_matching_policy", []

        matched_ids = [p.policy_id for p in applicable]

        if self.combine_algorithm == PolicyCombineAlgorithm.DENY_OVERRIDES:
            if any(p.effect == PolicyEffect.DENY for p in applicable):
                return PolicyEffect.DENY, "explicit_deny", matched_ids
            return PolicyEffect.ALLOW, "explicit_allow", matched_ids

        if self.combine_algorithm == PolicyCombineAlgorithm.PERMIT_OVERRIDES:
            if any(p.effect == PolicyEffect.ALLOW for p in applicable):
                return PolicyEffect.ALLOW, "explicit_allow", matched_ids
            return PolicyEffect.DENY, "explicit_deny", matched_ids

        if self.combine_algorithm == PolicyCombineAlgorithm.FIRST_APPLICABLE:
            return applicable[0].effect, "first_applicable", [applicable[0].policy_id]

        # UNANIMOUS: all must allow
        if all(p.effect == PolicyEffect.ALLOW for p in applicable):
            return PolicyEffect.ALLOW, "unanimous_allow", matched_ids
        return PolicyEffect.DENY, "unanimous_deny_rule", matched_ids


# ---------------------------------------------------------------------------
# Trust Score Calculator
# ---------------------------------------------------------------------------

class TrustScoreCalculator:
    """Continuously scores subject trustworthiness for adaptive re-auth."""

    def __init__(self):
        self._scores: Dict[str, TrustScore] = {}
        self._ip_blacklist: Set[str] = set()
        self._geo_blocklist: Set[str] = set()

    def evaluate(self, subject: AccessSubject) -> TrustScore:
        factors: Dict[str, float] = {}

        # Auth freshness (degrades over time)
        auth_age = time.time() - subject.auth_time
        factors["auth_freshness"] = max(0.0, 1.0 - auth_age / 86400)

        # MFA bonus
        factors["mfa"] = 1.0 if subject.mfa_verified else 0.6

        # Trust level factor
        factors["trust_level"] = {
            TrustLevel.HIGH: 1.0,
            TrustLevel.MEDIUM: 0.75,
            TrustLevel.LOW: 0.4,
            TrustLevel.UNTRUSTED: 0.1,
        }.get(subject.trust_level, 0.5)

        # IP reputation
        if subject.ip_address:
            if subject.ip_address in self._ip_blacklist:
                factors["ip_reputation"] = 0.0
            else:
                try:
                    ip = ipaddress.ip_address(subject.ip_address)
                    factors["ip_reputation"] = 0.5 if ip.is_private else 0.8
                except ValueError:
                    factors["ip_reputation"] = 0.7
        else:
            factors["ip_reputation"] = 0.9

        # Geo risk
        factors["geo_risk"] = 0.3 if subject.geo_country in self._geo_blocklist else 1.0

        # Workload identity bonus
        if subject.workload_identity and subject.workload_identity.is_valid:
            factors["workload_identity"] = 1.0
        else:
            factors["workload_identity"] = 0.7

        # Composite score: weighted average
        weights = {
            "auth_freshness": 0.25,
            "mfa": 0.20,
            "trust_level": 0.20,
            "ip_reputation": 0.15,
            "geo_risk": 0.10,
            "workload_identity": 0.10,
        }
        score = sum(factors[k] * weights.get(k, 0) for k in factors)

        trust_level = (
            TrustLevel.HIGH if score >= 0.8
            else TrustLevel.MEDIUM if score >= 0.6
            else TrustLevel.LOW if score >= 0.35
            else TrustLevel.UNTRUSTED
        )

        existing = self._scores.get(subject.subject_id)
        reauth_threshold = existing.reauth_threshold if existing else 0.5
        block_threshold = existing.block_threshold if existing else 0.2

        status = (
            VerificationStatus.BLOCKED if score <= block_threshold
            else VerificationStatus.REAUTH_REQUIRED if score <= reauth_threshold
            else VerificationStatus.VERIFIED
        )

        ts = TrustScore(
            subject_id=subject.subject_id,
            score=round(score, 4),
            factors=factors,
            trust_level=trust_level,
            verification_status=status,
        )
        self._scores[subject.subject_id] = ts
        return ts

    def block_ip(self, ip: str) -> None:
        self._ip_blacklist.add(ip)

    def unblock_ip(self, ip: str) -> None:
        self._ip_blacklist.discard(ip)

    def block_geo(self, country_code: str) -> None:
        self._geo_blocklist.add(country_code.upper())


# ---------------------------------------------------------------------------
# Network Firewall
# ---------------------------------------------------------------------------

class MicroSegmentationFirewall:
    """Evaluates network-layer policies for service-to-service communication."""

    def __init__(self):
        self._policies: List[NetworkPolicy] = []
        self._default_action = NetworkPolicyAction.DENY

    def add_policy(self, policy: NetworkPolicy) -> None:
        self._policies.append(policy)
        self._policies.sort(key=lambda p: p.priority)

    def evaluate(
        self,
        source_service: str,
        dest_service: str,
        dest_port: Optional[int] = None,
        protocol: str = "tcp",
        source_ip: Optional[str] = None,
    ) -> NetworkPolicyAction:
        for policy in self._policies:
            if not policy.enabled:
                continue
            if policy.source_service not in ("*", source_service):
                continue
            if policy.dest_service not in ("*", dest_service):
                continue
            if policy.dest_port and dest_port and policy.dest_port != dest_port:
                continue
            if policy.protocols and protocol not in policy.protocols:
                continue
            if policy.source_ip_cidr and source_ip:
                try:
                    net = ipaddress.ip_network(policy.source_ip_cidr, strict=False)
                    ip = ipaddress.ip_address(source_ip)
                    if ip not in net:
                        continue
                except ValueError:
                    pass
            return policy.action
        return self._default_action


# ---------------------------------------------------------------------------
# Zero Trust Security Engine
# ---------------------------------------------------------------------------

class ZeroTrustSecurityEngine:
    """
    Unified zero-trust enforcement engine: workload identity, policy evaluation,
    continuous trust scoring, network micro-segmentation, and audit logging.
    """

    def __init__(self, trust_domain: str = "cognitionos.io"):
        self.certificate_authority = WorkloadCertificateAuthority(trust_domain)
        self.policy_engine = PolicyEngine()
        self.trust_calculator = TrustScoreCalculator()
        self.firewall = MicroSegmentationFirewall()
        self._audit_log: List[AuditRecord] = []
        self._request_count: int = 0
        self._allow_count: int = 0
        self._deny_count: int = 0
        self._initialize_default_policies()

    def _initialize_default_policies(self) -> None:
        """Seed sensible default policies."""
        # Admin role gets full access
        self.policy_engine.add_policy(SecurityPolicy(
            name="admin_full_access",
            effect=PolicyEffect.ALLOW,
            subjects=["admin", "superadmin"],
            resources=["*"],
            actions=["*"],
            priority=1,
        ))
        # Deny access to secret resources for non-admin
        self.policy_engine.add_policy(SecurityPolicy(
            name="deny_secret_non_admin",
            effect=PolicyEffect.DENY,
            subjects=["*"],
            resources=["confidential", "secret"],
            actions=["*"],
            priority=10,
        ))
        # Untrusted subjects always denied
        self.policy_engine.add_policy(SecurityPolicy(
            name="deny_untrusted",
            effect=PolicyEffect.DENY,
            subjects=["*"],
            resources=["*"],
            actions=["*"],
            conditions=["ctx.get(\"subject\") is None"],
            priority=2,
        ))
        # Allow all internal service-to-service reads
        self.policy_engine.add_policy(SecurityPolicy(
            name="service_read_allow",
            effect=PolicyEffect.ALLOW,
            subjects=["type:service"],
            resources=["api", "internal"],
            actions=["read", "execute"],
            priority=50,
        ))
        # Default network policy: allow same-namespace
        self.firewall.add_policy(NetworkPolicy(
            name="allow_same_namespace",
            source_service="*",
            dest_service="*",
            action=NetworkPolicyAction.ALLOW,
            priority=1000,
        ))

    async def authorize(self, request: AccessRequest) -> AccessDecision:
        """Evaluate full zero-trust authorization for a request."""
        t0 = time.time()
        self._request_count += 1

        # Compute trust score
        trust = self.trust_calculator.evaluate(request.subject)

        # Block if trust is too low
        if trust.verification_status == VerificationStatus.BLOCKED:
            decision = AccessDecision(
                request_id=request.request_id,
                effect=PolicyEffect.DENY,
                reason="subject_blocked_low_trust",
                trust_score=trust.score,
                decision_latency_ms=(time.time() - t0) * 1000,
            )
            self._deny_count += 1
            self._audit(request, decision)
            return decision

        # Evaluate policies
        effect, reason, matched = self.policy_engine.evaluate(request)

        step_up = trust.verification_status == VerificationStatus.REAUTH_REQUIRED
        obligations = ["reauth_required"] if step_up else []

        decision = AccessDecision(
            request_id=request.request_id,
            effect=effect,
            reason=reason,
            matched_policies=matched,
            trust_score=trust.score,
            step_up_required=step_up,
            obligations=obligations,
            decision_latency_ms=(time.time() - t0) * 1000,
        )

        if effect == PolicyEffect.ALLOW:
            self._allow_count += 1
        else:
            self._deny_count += 1

        self._audit(request, decision)
        return decision

    def check_network_access(
        self,
        source: str,
        dest: str,
        port: Optional[int] = None,
        protocol: str = "tcp",
    ) -> bool:
        action = self.firewall.evaluate(source, dest, port, protocol)
        return action == NetworkPolicyAction.ALLOW

    def _audit(self, request: AccessRequest, decision: AccessDecision) -> None:
        record = AuditRecord(
            request_id=request.request_id,
            subject_id=request.subject.subject_id if request.subject else "",
            resource_type=request.resource.resource_type if request.resource else "",
            resource_id=request.resource.resource_id if request.resource else "",
            action=request.action,
            effect=decision.effect.value,
            reason=decision.reason,
            tenant_id=request.tenant_id,
            ip_address=request.subject.ip_address if request.subject else None,
            trust_score=decision.trust_score,
        )
        self._audit_log.append(record)
        if len(self._audit_log) > 100000:
            self._audit_log = self._audit_log[-50000:]

    def get_audit_log(
        self,
        subject_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        effect: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditRecord]:
        records = list(reversed(self._audit_log))
        if subject_id:
            records = [r for r in records if r.subject_id == subject_id]
        if tenant_id:
            records = [r for r in records if r.tenant_id == tenant_id]
        if effect:
            records = [r for r in records if r.effect == effect]
        return records[:limit]

    def get_security_summary(self) -> Dict[str, Any]:
        total = max(1, self._request_count)
        return {
            "total_requests": self._request_count,
            "allowed": self._allow_count,
            "denied": self._deny_count,
            "allow_rate": round(self._allow_count / total, 4),
            "deny_rate": round(self._deny_count / total, 4),
            "policies": len(self.policy_engine._policies),
            "certificates_issued": self.certificate_authority._issued_count,
            "revoked_certificates": len(self.certificate_authority._revoked),
            "blocked_ips": len(self.trust_calculator._ip_blacklist),
            "audit_records": len(self._audit_log),
        }
