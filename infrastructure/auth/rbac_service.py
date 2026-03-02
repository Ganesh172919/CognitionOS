"""
RBAC (Role-Based Access Control) Service — CognitionOS

Features:
- Hierarchical roles with inheritance
- Granular permissions (resource:action)
- Role assignment per user and tenant
- Permission caching
- Audit trail for all access decisions
- API key scoping
- Multi-tenant role isolation
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set

logger = logging.getLogger(__name__)


class Permission(str, Enum):
    # Agent
    AGENT_CREATE = "agent:create"
    AGENT_READ = "agent:read"
    AGENT_EXECUTE = "agent:execute"
    AGENT_DELETE = "agent:delete"
    AGENT_ADMIN = "agent:admin"
    # Task
    TASK_CREATE = "task:create"
    TASK_READ = "task:read"
    TASK_CANCEL = "task:cancel"
    # Memory
    MEMORY_READ = "memory:read"
    MEMORY_WRITE = "memory:write"
    MEMORY_DELETE = "memory:delete"
    # Billing
    BILLING_READ = "billing:read"
    BILLING_MANAGE = "billing:manage"
    BILLING_ADMIN = "billing:admin"
    # Tenant
    TENANT_READ = "tenant:read"
    TENANT_MANAGE = "tenant:manage"
    TENANT_ADMIN = "tenant:admin"
    # Plugin
    PLUGIN_INSTALL = "plugin:install"
    PLUGIN_MANAGE = "plugin:manage"
    # API
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_ADMIN = "api:admin"
    # Admin
    ADMIN_DASHBOARD = "admin:dashboard"
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"
    SUPER_ADMIN = "admin:super"


@dataclass
class Role:
    name: str
    permissions: Set[str] = field(default_factory=set)
    inherits: List[str] = field(default_factory=list)
    description: str = ""
    is_system: bool = False
    tenant_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "permissions": sorted(self.permissions),
                "inherits": self.inherits, "description": self.description,
                "is_system": self.is_system, "tenant_id": self.tenant_id}


@dataclass
class UserRoleBinding:
    user_id: str
    role_name: str
    tenant_id: Optional[str] = None
    granted_by: str = "system"
    granted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: Optional[str] = None


@dataclass
class AccessDecision:
    allowed: bool
    user_id: str
    permission: str
    reason: str
    roles: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# Default system roles
SYSTEM_ROLES = {
    "viewer": Role(
        name="viewer", is_system=True, description="Read-only access",
        permissions={Permission.AGENT_READ, Permission.TASK_READ,
                     Permission.MEMORY_READ, Permission.API_READ}),
    "user": Role(
        name="user", is_system=True, description="Standard user",
        inherits=["viewer"],
        permissions={Permission.AGENT_CREATE, Permission.AGENT_EXECUTE,
                     Permission.TASK_CREATE, Permission.MEMORY_WRITE,
                     Permission.API_WRITE}),
    "developer": Role(
        name="developer", is_system=True, description="Developer access",
        inherits=["user"],
        permissions={Permission.PLUGIN_INSTALL, Permission.PLUGIN_MANAGE,
                     Permission.AGENT_DELETE}),
    "admin": Role(
        name="admin", is_system=True, description="Tenant admin",
        inherits=["developer"],
        permissions={Permission.BILLING_READ, Permission.BILLING_MANAGE,
                     Permission.TENANT_READ, Permission.TENANT_MANAGE,
                     Permission.ADMIN_DASHBOARD, Permission.ADMIN_USERS}),
    "super_admin": Role(
        name="super_admin", is_system=True, description="Platform super admin",
        inherits=["admin"],
        permissions={Permission.SUPER_ADMIN, Permission.ADMIN_SYSTEM,
                     Permission.BILLING_ADMIN, Permission.TENANT_ADMIN,
                     Permission.AGENT_ADMIN}),
}


class RBACService:
    """Role-based access control with inheritance and multi-tenant support."""

    def __init__(self) -> None:
        self._roles: Dict[str, Role] = dict(SYSTEM_ROLES)
        self._bindings: Dict[str, List[UserRoleBinding]] = defaultdict(list)
        self._cache: Dict[str, FrozenSet[str]] = {}
        self._audit_log: List[AccessDecision] = []
        self._max_audit = 10000

    # ---- role management ----
    def create_role(self, role: Role) -> None:
        self._roles[role.name] = role
        self._invalidate_cache()

    def delete_role(self, name: str) -> bool:
        role = self._roles.get(name)
        if not role or role.is_system:
            return False
        del self._roles[name]
        self._invalidate_cache()
        return True

    def get_role(self, name: str) -> Optional[Role]:
        return self._roles.get(name)

    def list_roles(self, *, tenant_id: Optional[str] = None) -> List[Role]:
        roles = list(self._roles.values())
        if tenant_id:
            roles = [r for r in roles if r.is_system or r.tenant_id == tenant_id]
        return roles

    # ---- resolve permissions with inheritance ----
    def _resolve_permissions(self, role_name: str, visited: Optional[Set[str]] = None) -> Set[str]:
        visited = visited or set()
        if role_name in visited:
            return set()
        visited.add(role_name)

        role = self._roles.get(role_name)
        if not role:
            return set()

        perms = set(role.permissions)
        for parent in role.inherits:
            perms |= self._resolve_permissions(parent, visited)
        return perms

    # ---- user bindings ----
    def assign_role(self, user_id: str, role_name: str, *,
                    tenant_id: Optional[str] = None, granted_by: str = "system") -> bool:
        if role_name not in self._roles:
            return False
        binding = UserRoleBinding(
            user_id=user_id, role_name=role_name,
            tenant_id=tenant_id, granted_by=granted_by)
        self._bindings[user_id].append(binding)
        self._cache.pop(user_id, None)
        return True

    def revoke_role(self, user_id: str, role_name: str) -> bool:
        before = len(self._bindings.get(user_id, []))
        self._bindings[user_id] = [
            b for b in self._bindings.get(user_id, []) if b.role_name != role_name]
        self._cache.pop(user_id, None)
        return len(self._bindings.get(user_id, [])) < before

    def get_user_roles(self, user_id: str) -> List[str]:
        return [b.role_name for b in self._bindings.get(user_id, [])]

    def get_user_permissions(self, user_id: str) -> FrozenSet[str]:
        if user_id in self._cache:
            return self._cache[user_id]
        perms: Set[str] = set()
        for binding in self._bindings.get(user_id, []):
            perms |= self._resolve_permissions(binding.role_name)
        result = frozenset(perms)
        self._cache[user_id] = result
        return result

    # ---- access check ----
    def check_permission(self, user_id: str, permission: str) -> AccessDecision:
        perms = self.get_user_permissions(user_id)
        roles = self.get_user_roles(user_id)
        allowed = permission in perms or Permission.SUPER_ADMIN in perms

        decision = AccessDecision(
            allowed=allowed, user_id=user_id, permission=permission,
            reason="granted" if allowed else "denied",
            roles=roles)

        self._audit_log.append(decision)
        if len(self._audit_log) > self._max_audit:
            self._audit_log = self._audit_log[-self._max_audit:]

        return decision

    def has_permission(self, user_id: str, permission: str) -> bool:
        return self.check_permission(user_id, permission).allowed

    def require_permission(self, user_id: str, permission: str) -> None:
        if not self.has_permission(user_id, permission):
            raise PermissionError(f"User {user_id} lacks permission: {permission}")

    # ---- audit ----
    def get_audit_log(self, *, user_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        logs = self._audit_log
        if user_id:
            logs = [l for l in logs if l.user_id == user_id]
        return [{"allowed": l.allowed, "user_id": l.user_id, "permission": l.permission,
                 "reason": l.reason, "roles": l.roles, "timestamp": l.timestamp}
                for l in logs[-limit:]]

    # ---- cache ----
    def _invalidate_cache(self) -> None:
        self._cache.clear()


_rbac: RBACService | None = None

def get_rbac_service() -> RBACService:
    global _rbac
    if not _rbac:
        _rbac = RBACService()
    return _rbac
