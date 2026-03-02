"""
Comprehensive Audit Trail — CognitionOS

Enterprise-grade audit logging with:
- Immutable audit log entries
- Action tracking (CRUD, auth, admin)
- Actor and resource resolution
- Compliance-ready formatting
- Search and filter capabilities
- Export support
- Retention policies
- Tamper detection via hash chains
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    EXPORT = "export"
    ADMIN = "admin"
    GRANT = "grant"
    REVOKE = "revoke"
    APPROVE = "approve"
    DENY = "deny"
    EXECUTE = "execute"
    DEPLOY = "deploy"
    CONFIGURE = "configure"


class AuditSeverity(str, Enum):
    INFO = "info"
    NOTICE = "notice"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AuditEntry:
    entry_id: str
    action: AuditAction
    actor_id: str
    actor_type: str = "user"
    resource_type: str = ""
    resource_id: str = ""
    tenant_id: str = ""
    description: str = ""
    severity: AuditSeverity = AuditSeverity.INFO
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    ip_address: str = ""
    user_agent: str = ""
    timestamp: float = field(default_factory=time.time)
    hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.entry_id,
            "action": self.action.value,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "tenant_id": self.tenant_id,
            "description": self.description,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "ip_address": self.ip_address,
        }


class AuditTrail:
    """
    Immutable, hash-chained audit trail for compliance and security.
    """

    def __init__(self, *, max_entries: int = 1_000_000):
        self._entries: List[AuditEntry] = []
        self._max_entries = max_entries
        self._last_hash: str = "genesis"
        self._by_actor: Dict[str, List[int]] = defaultdict(list)
        self._by_resource: Dict[str, List[int]] = defaultdict(list)
        self._by_tenant: Dict[str, List[int]] = defaultdict(list)

    def log(self, action: AuditAction, actor_id: str, *,
              resource_type: str = "", resource_id: str = "",
              tenant_id: str = "", description: str = "",
              severity: AuditSeverity = AuditSeverity.INFO,
              old_value: Optional[Dict] = None,
              new_value: Optional[Dict] = None,
              ip_address: str = "",
              metadata: Optional[Dict] = None) -> AuditEntry:
        """Log an audit entry with hash chain integrity."""
        entry = AuditEntry(
            entry_id=uuid.uuid4().hex[:16],
            action=action, actor_id=actor_id,
            resource_type=resource_type, resource_id=resource_id,
            tenant_id=tenant_id, description=description,
            severity=severity, old_value=old_value,
            new_value=new_value, ip_address=ip_address,
            metadata=metadata or {},
        )

        # Hash chain for tamper detection
        chain_data = f"{self._last_hash}:{entry.entry_id}:{entry.action.value}:{entry.timestamp}"
        entry.hash = hashlib.sha256(chain_data.encode()).hexdigest()[:24]
        self._last_hash = entry.hash

        idx = len(self._entries)
        self._entries.append(entry)

        # Index
        self._by_actor[actor_id].append(idx)
        if resource_id:
            self._by_resource[f"{resource_type}:{resource_id}"].append(idx)
        if tenant_id:
            self._by_tenant[tenant_id].append(idx)

        # Retention
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries // 2:]
            self._rebuild_indexes()

        return entry

    # ── Query ──

    def query(self, *, actor_id: Optional[str] = None,
                resource_type: Optional[str] = None,
                resource_id: Optional[str] = None,
                tenant_id: Optional[str] = None,
                action: Optional[AuditAction] = None,
                severity: Optional[AuditSeverity] = None,
                start_time: Optional[float] = None,
                end_time: Optional[float] = None,
                limit: int = 100) -> List[Dict[str, Any]]:
        """Query audit entries with filters."""
        results = self._entries

        if actor_id:
            indexes = self._by_actor.get(actor_id, [])
            results = [self._entries[i] for i in indexes if i < len(self._entries)]

        elif tenant_id:
            indexes = self._by_tenant.get(tenant_id, [])
            results = [self._entries[i] for i in indexes if i < len(self._entries)]

        elif resource_id and resource_type:
            key = f"{resource_type}:{resource_id}"
            indexes = self._by_resource.get(key, [])
            results = [self._entries[i] for i in indexes if i < len(self._entries)]

        if action:
            results = [e for e in results if e.action == action]
        if severity:
            results = [e for e in results if e.severity == severity]
        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]

        return [e.to_dict() for e in results[-limit:]]

    def get_resource_history(self, resource_type: str,
                               resource_id: str) -> List[Dict[str, Any]]:
        key = f"{resource_type}:{resource_id}"
        indexes = self._by_resource.get(key, [])
        return [self._entries[i].to_dict() for i in indexes
                if i < len(self._entries)]

    def get_actor_history(self, actor_id: str, *,
                            limit: int = 100) -> List[Dict[str, Any]]:
        indexes = self._by_actor.get(actor_id, [])
        return [self._entries[i].to_dict() for i in indexes[-limit:]
                if i < len(self._entries)]

    # ── Integrity Verification ──

    def verify_integrity(self) -> Dict[str, Any]:
        """Verify hash chain integrity."""
        if not self._entries:
            return {"valid": True, "entries_checked": 0}

        prev_hash = "genesis"
        broken_at = None

        for i, entry in enumerate(self._entries):
            expected_data = f"{prev_hash}:{entry.entry_id}:{entry.action.value}:{entry.timestamp}"
            expected_hash = hashlib.sha256(expected_data.encode()).hexdigest()[:24]

            if entry.hash != expected_hash:
                broken_at = i
                break
            prev_hash = entry.hash

        return {
            "valid": broken_at is None,
            "entries_checked": len(self._entries),
            "broken_at_index": broken_at,
        }

    # ── Export ──

    def export(self, *, format: str = "json",
                 tenant_id: Optional[str] = None,
                 start_time: Optional[float] = None,
                 end_time: Optional[float] = None) -> str:
        entries = self.query(
            tenant_id=tenant_id,
            start_time=start_time, end_time=end_time,
            limit=50000,
        )
        if format == "json":
            return json.dumps(entries, indent=2, default=str)
        elif format == "csv":
            if not entries:
                return ""
            headers = list(entries[0].keys())
            lines = [",".join(headers)]
            for entry in entries:
                lines.append(",".join(str(entry.get(h, "")) for h in headers))
            return "\n".join(lines)
        return json.dumps(entries, default=str)

    # ── Stats ──

    def get_stats(self) -> Dict[str, Any]:
        by_action = defaultdict(int)
        by_severity = defaultdict(int)
        for entry in self._entries:
            by_action[entry.action.value] += 1
            by_severity[entry.severity.value] += 1

        return {
            "total_entries": len(self._entries),
            "unique_actors": len(self._by_actor),
            "resources_tracked": len(self._by_resource),
            "tenants_tracked": len(self._by_tenant),
            "by_action": dict(sorted(by_action.items(), key=lambda x: -x[1])),
            "by_severity": dict(by_severity),
        }

    def _rebuild_indexes(self):
        self._by_actor.clear()
        self._by_resource.clear()
        self._by_tenant.clear()
        for i, entry in enumerate(self._entries):
            self._by_actor[entry.actor_id].append(i)
            if entry.resource_id:
                key = f"{entry.resource_type}:{entry.resource_id}"
                self._by_resource[key].append(i)
            if entry.tenant_id:
                self._by_tenant[entry.tenant_id].append(i)


# ── Singleton ──
_trail: Optional[AuditTrail] = None


def get_audit_trail() -> AuditTrail:
    global _trail
    if not _trail:
        _trail = AuditTrail()
    return _trail
