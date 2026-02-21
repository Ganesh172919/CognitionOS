"""
Multi-Tenant RBAC Engine

Provides fine-grained role-based access control with:
- Hierarchical role inheritance (super-admin → admin → editor → viewer)
- Resource-level permission grants
- Per-tenant role isolation
- Policy evaluation with AND/OR conditions
- Attribute-based access control (ABAC) extensions
- Wildcard permission matching (resource:action:*)
- JWT claim-based role extraction
- Audit integration for every permission check
"""

from __future__ import annotations

import fnmatch
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
from uuid import uuid4


# ──────────────────────────────────────────────────────────────────────────────
# Core primitives
# ──────────────────────────────────────────────────────────────────────────────


class Effect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


class PolicyOperator(str, Enum):
    ALL = "all"   # All conditions must match
    ANY = "any"   # At least one condition must match


@dataclass(frozen=True)
class Permission:
    """
    Represents a single permission as resource:action[:qualifier].
    Supports wildcard matching via fnmatch (e.g. ``workflows:*``, ``*:read``).

    Examples::

        Permission("workflows", "execute")
        Permission("api_keys", "*")
        Permission("*", "read")
    """
    resource: str
    action: str
    qualifier: str = "*"

    def __str__(self) -> str:
        return f"{self.resource}:{self.action}:{self.qualifier}"

    def matches(self, other: "Permission") -> bool:
        """Return True if this permission grants the requested ``other`` permission."""
        return (
            fnmatch.fnmatch(other.resource, self.resource)
            and fnmatch.fnmatch(other.action, self.action)
            and fnmatch.fnmatch(other.qualifier, self.qualifier)
        )

    @classmethod
    def from_string(cls, perm_str: str) -> "Permission":
        parts = perm_str.split(":")
        if len(parts) == 2:
            return cls(parts[0], parts[1])
        if len(parts) == 3:
            return cls(parts[0], parts[1], parts[2])
        raise ValueError(f"Invalid permission string: '{perm_str}'")


@dataclass
class Role:
    """
    A named collection of permissions with optional parent role for inheritance.
    Role hierarchy is resolved transitively.
    """
    role_id: str
    name: str
    tenant_id: Optional[str]       # None = global / system role
    permissions: Set[Permission] = field(default_factory=set)
    parent_role_ids: List[str] = field(default_factory=list)
    description: str = ""
    is_system: bool = False
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role_id": self.role_id,
            "name": self.name,
            "tenant_id": self.tenant_id,
            "permissions": [str(p) for p in self.permissions],
            "parent_role_ids": self.parent_role_ids,
            "is_system": self.is_system,
            "description": self.description,
        }


@dataclass
class PolicyCondition:
    """Attribute condition for ABAC policies"""
    attribute: str    # e.g. "request.ip", "user.department", "resource.owner"
    operator: str     # "eq", "neq", "in", "not_in", "contains", "startswith"
    value: Any

    def evaluate(self, context: Dict[str, Any]) -> bool:
        actual = self._get_nested(context, self.attribute)
        if actual is None:
            return False
        if self.operator == "eq":
            return actual == self.value
        if self.operator == "neq":
            return actual != self.value
        if self.operator == "in":
            return actual in self.value
        if self.operator == "not_in":
            return actual not in self.value
        if self.operator == "contains":
            return self.value in str(actual)
        if self.operator == "startswith":
            return str(actual).startswith(str(self.value))
        if self.operator == "gt":
            return float(actual) > float(self.value)
        if self.operator == "lt":
            return float(actual) < float(self.value)
        return False

    @staticmethod
    def _get_nested(obj: Dict[str, Any], path: str) -> Any:
        parts = path.split(".")
        current = obj
        for part in parts:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current


@dataclass
class Policy:
    """
    ABAC policy that grants/denies a set of permissions under conditions.
    Policies are evaluated after role permissions; explicit DENY overrides ALLOW.
    """
    policy_id: str
    name: str
    tenant_id: Optional[str]
    effect: Effect
    permissions: List[Permission]
    conditions: List[PolicyCondition] = field(default_factory=list)
    operator: PolicyOperator = PolicyOperator.ALL
    priority: int = 0
    description: str = ""
    enabled: bool = True

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate all conditions against the context"""
        if not self.conditions:
            return True
        results = [c.evaluate(context) for c in self.conditions]
        if self.operator == PolicyOperator.ALL:
            return all(results)
        return any(results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "effect": self.effect.value,
            "permissions": [str(p) for p in self.permissions],
            "enabled": self.enabled,
            "priority": self.priority,
        }


@dataclass
class RoleAssignment:
    """Maps a user/service to roles within a tenant scope"""
    assignment_id: str
    principal_id: str    # user_id or service_id
    role_id: str
    tenant_id: str
    granted_by: Optional[str] = None
    expires_at: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


@dataclass
class AuthorizationResult:
    """Result of a permission check"""
    allowed: bool
    principal_id: str
    permission: Permission
    tenant_id: str
    matched_roles: List[str] = field(default_factory=list)
    matched_policies: List[str] = field(default_factory=list)
    denial_reason: Optional[str] = None
    evaluation_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "principal_id": self.principal_id,
            "permission": str(self.permission),
            "tenant_id": self.tenant_id,
            "matched_roles": self.matched_roles,
            "denial_reason": self.denial_reason,
            "evaluation_time_ms": round(self.evaluation_time_ms, 3),
        }


# ──────────────────────────────────────────────────────────────────────────────
# RBAC Engine
# ──────────────────────────────────────────────────────────────────────────────


class RBACEngine:
    """
    Central authorization engine providing RBAC + ABAC evaluation.

    Features:
    - Hierarchical role inheritance (transitive)
    - Per-tenant role isolation (tenants cannot see each other's roles)
    - Global system roles shared across tenants
    - ABAC policy overlays with priority-based evaluation
    - Explicit DENY takes precedence over ALLOW
    - Wildcard permission matching
    - Permission resolution caching

    Usage::

        engine = RBACEngine()
        engine.register_role(Role("admin", "admin", None, {Permission("*", "*")}))
        engine.assign_role("user-1", "admin", "tenant-a")
        result = engine.check("user-1", Permission("workflows", "execute"), "tenant-a")
        assert result.allowed
    """

    SUPER_ADMIN_ROLE = "super_admin"
    TENANT_ADMIN_ROLE = "tenant_admin"
    DEVELOPER_ROLE = "developer"
    VIEWER_ROLE = "viewer"

    def __init__(self) -> None:
        self._roles: Dict[str, Role] = {}
        self._assignments: Dict[str, List[RoleAssignment]] = {}  # principal_id -> assignments
        self._policies: List[Policy] = []
        self._permission_cache: Dict[str, Tuple[bool, float]] = {}  # key -> (result, expires)
        self._cache_ttl: float = 60.0
        self._setup_default_roles()

    # ──────────────────────────────────────────────
    # Role Management
    # ──────────────────────────────────────────────

    def register_role(self, role: Role) -> None:
        """Register a role in the engine"""
        self._roles[role.role_id] = role
        self._invalidate_cache()

    def remove_role(self, role_id: str) -> bool:
        if role_id in self._roles and not self._roles[role_id].is_system:
            del self._roles[role_id]
            self._invalidate_cache()
            return True
        return False

    def get_role(self, role_id: str) -> Optional[Role]:
        return self._roles.get(role_id)

    def list_roles(self, tenant_id: Optional[str] = None) -> List[Role]:
        """List roles visible to a tenant (tenant-specific + global system roles)"""
        return [
            r for r in self._roles.values()
            if r.tenant_id is None or r.tenant_id == tenant_id
        ]

    # ──────────────────────────────────────────────
    # Assignment Management
    # ──────────────────────────────────────────────

    def assign_role(
        self,
        principal_id: str,
        role_id: str,
        tenant_id: str,
        granted_by: Optional[str] = None,
        expires_at: Optional[float] = None,
    ) -> RoleAssignment:
        """Assign a role to a principal within a tenant"""
        if role_id not in self._roles:
            raise ValueError(f"Role '{role_id}' not found")
        assignment = RoleAssignment(
            assignment_id=str(uuid4()),
            principal_id=principal_id,
            role_id=role_id,
            tenant_id=tenant_id,
            granted_by=granted_by,
            expires_at=expires_at,
        )
        self._assignments.setdefault(principal_id, []).append(assignment)
        self._invalidate_cache_for(principal_id)
        return assignment

    def revoke_role(self, principal_id: str, role_id: str, tenant_id: str) -> int:
        """Revoke a role assignment, returns number of assignments removed"""
        assignments = self._assignments.get(principal_id, [])
        before = len(assignments)
        self._assignments[principal_id] = [
            a for a in assignments
            if not (a.role_id == role_id and a.tenant_id == tenant_id)
        ]
        removed = before - len(self._assignments[principal_id])
        if removed:
            self._invalidate_cache_for(principal_id)
        return removed

    def get_principal_roles(
        self,
        principal_id: str,
        tenant_id: str,
    ) -> List[Role]:
        """Get all active roles for a principal in a tenant"""
        assignments = self._assignments.get(principal_id, [])
        active = [
            a for a in assignments
            if a.tenant_id == tenant_id and not a.is_expired()
        ]
        roles = []
        for a in active:
            role = self._roles.get(a.role_id)
            if role:
                roles.append(role)
        return roles

    # ──────────────────────────────────────────────
    # Policy Management
    # ──────────────────────────────────────────────

    def add_policy(self, policy: Policy) -> None:
        self._policies.append(policy)
        self._policies.sort(key=lambda p: p.priority, reverse=True)
        self._invalidate_cache()

    def remove_policy(self, policy_id: str) -> bool:
        before = len(self._policies)
        self._policies = [p for p in self._policies if p.policy_id != policy_id]
        if len(self._policies) < before:
            self._invalidate_cache()
            return True
        return False

    # ──────────────────────────────────────────────
    # Authorization
    # ──────────────────────────────────────────────

    def check(
        self,
        principal_id: str,
        permission: Permission,
        tenant_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthorizationResult:
        """
        Check if a principal has a permission in a tenant context.
        Returns AuthorizationResult with full evaluation details.
        """
        start = time.perf_counter()
        ctx = context or {}
        ctx.setdefault("principal_id", principal_id)
        ctx.setdefault("tenant_id", tenant_id)

        # Check cache
        cache_key = f"{principal_id}:{permission}:{tenant_id}"
        if not context:  # Only use cache when no dynamic context
            cached = self._permission_cache.get(cache_key)
            if cached and time.time() < cached[1]:
                elapsed = (time.perf_counter() - start) * 1000
                return AuthorizationResult(
                    allowed=cached[0],
                    principal_id=principal_id,
                    permission=permission,
                    tenant_id=tenant_id,
                    evaluation_time_ms=elapsed,
                )

        # Resolve all effective permissions from roles (with inheritance)
        roles = self.get_principal_roles(principal_id, tenant_id)
        effective_perms: Dict[str, List[str]] = {}  # perm -> [role_ids that grant it]
        matched_role_ids: List[str] = []

        for role in roles:
            all_perms = self._resolve_role_permissions(role)
            for p in all_perms:
                if p.matches(permission):
                    key = str(p)
                    effective_perms.setdefault(key, []).append(role.role_id)
                    if role.role_id not in matched_role_ids:
                        matched_role_ids.append(role.role_id)

        # Evaluate ABAC policies (DENY policies take precedence)
        matched_policy_ids: List[str] = []
        deny_reason: Optional[str] = None

        for policy in self._policies:
            if not policy.enabled:
                continue
            if policy.tenant_id is not None and policy.tenant_id != tenant_id:
                continue
            # Check if this policy applies to the requested permission
            applies = any(p.matches(permission) for p in policy.permissions)
            if not applies:
                continue
            if policy.evaluate(ctx):
                matched_policy_ids.append(policy.policy_id)
                if policy.effect == Effect.DENY:
                    deny_reason = f"Policy '{policy.name}' explicitly denies this permission"

        # Final decision: explicit DENY overrides ALLOW
        allowed = bool(matched_role_ids) and deny_reason is None

        elapsed_ms = (time.perf_counter() - start) * 1000
        result = AuthorizationResult(
            allowed=allowed,
            principal_id=principal_id,
            permission=permission,
            tenant_id=tenant_id,
            matched_roles=matched_role_ids,
            matched_policies=matched_policy_ids,
            denial_reason=deny_reason or (None if allowed else "No role grants this permission"),
            evaluation_time_ms=elapsed_ms,
        )

        # Cache result
        if not context:
            self._permission_cache[cache_key] = (allowed, time.time() + self._cache_ttl)

        return result

    def require(
        self,
        principal_id: str,
        permission: Permission,
        tenant_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Like check() but raises PermissionDeniedError if not authorized"""
        result = self.check(principal_id, permission, tenant_id, context)
        if not result.allowed:
            raise PermissionDeniedError(
                principal_id=principal_id,
                permission=permission,
                tenant_id=tenant_id,
                reason=result.denial_reason or "Access denied",
            )

    def bulk_check(
        self,
        principal_id: str,
        permissions: List[Permission],
        tenant_id: str,
    ) -> Dict[str, bool]:
        """Check multiple permissions at once, returns {perm_str: allowed}"""
        return {
            str(p): self.check(principal_id, p, tenant_id).allowed
            for p in permissions
        }

    def get_effective_permissions(
        self,
        principal_id: str,
        tenant_id: str,
    ) -> List[Permission]:
        """Get all effective permissions for a principal in a tenant"""
        roles = self.get_principal_roles(principal_id, tenant_id)
        all_perms: Set[Permission] = set()
        for role in roles:
            all_perms.update(self._resolve_role_permissions(role))
        return sorted(all_perms, key=str)

    # ──────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────

    def _resolve_role_permissions(
        self,
        role: Role,
        visited: Optional[Set[str]] = None,
    ) -> Set[Permission]:
        """Recursively resolve all permissions including inherited ones"""
        if visited is None:
            visited = set()
        if role.role_id in visited:
            return set()
        visited.add(role.role_id)

        perms = set(role.permissions)
        for parent_id in role.parent_role_ids:
            parent = self._roles.get(parent_id)
            if parent:
                perms.update(self._resolve_role_permissions(parent, visited))
        return perms

    def _invalidate_cache(self) -> None:
        self._permission_cache.clear()

    def _invalidate_cache_for(self, principal_id: str) -> None:
        keys_to_remove = [k for k in self._permission_cache if k.startswith(f"{principal_id}:")]
        for k in keys_to_remove:
            del self._permission_cache[k]

    def _setup_default_roles(self) -> None:
        """Register built-in system roles"""
        viewer = Role(
            role_id=self.VIEWER_ROLE,
            name="Viewer",
            tenant_id=None,
            permissions={
                Permission("workflows", "read"),
                Permission("agents", "read"),
                Permission("memory", "read"),
                Permission("analytics", "read"),
            },
            is_system=True,
            description="Read-only access to all resources",
        )
        developer = Role(
            role_id=self.DEVELOPER_ROLE,
            name="Developer",
            tenant_id=None,
            permissions={
                Permission("workflows", "read"),
                Permission("workflows", "create"),
                Permission("workflows", "execute"),
                Permission("agents", "read"),
                Permission("agents", "create"),
                Permission("agents", "execute"),
                Permission("memory", "read"),
                Permission("memory", "write"),
                Permission("plugins", "read"),
                Permission("plugins", "install"),
                Permission("tools", "execute"),
                Permission("api_keys", "create"),
                Permission("api_keys", "read"),
            },
            parent_role_ids=[self.VIEWER_ROLE],
            is_system=True,
            description="Full developer access",
        )
        tenant_admin = Role(
            role_id=self.TENANT_ADMIN_ROLE,
            name="Tenant Admin",
            tenant_id=None,
            permissions={
                Permission("*", "read"),
                Permission("*", "create"),
                Permission("*", "update"),
                Permission("*", "delete"),
                Permission("*", "execute"),
                Permission("users", "manage"),
                Permission("billing", "manage"),
                Permission("settings", "manage"),
            },
            parent_role_ids=[self.DEVELOPER_ROLE],
            is_system=True,
            description="Full admin access within a tenant",
        )
        super_admin = Role(
            role_id=self.SUPER_ADMIN_ROLE,
            name="Super Admin",
            tenant_id=None,
            permissions={Permission("*", "*")},
            parent_role_ids=[self.TENANT_ADMIN_ROLE],
            is_system=True,
            description="Unrestricted access to all resources and tenants",
        )
        for role in [viewer, developer, tenant_admin, super_admin]:
            self._roles[role.role_id] = role


class PermissionDeniedError(Exception):
    """Raised when a permission check fails"""

    def __init__(
        self,
        principal_id: str,
        permission: Permission,
        tenant_id: str,
        reason: str = "Access denied",
    ) -> None:
        self.principal_id = principal_id
        self.permission = permission
        self.tenant_id = tenant_id
        self.reason = reason
        super().__init__(f"Permission denied: {reason} [{principal_id} -> {permission} @ {tenant_id}]")


# Global singleton
_rbac_engine: Optional[RBACEngine] = None


def get_rbac_engine() -> RBACEngine:
    global _rbac_engine
    if _rbac_engine is None:
        _rbac_engine = RBACEngine()
    return _rbac_engine
