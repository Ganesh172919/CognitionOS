"""
Audit Logger — CognitionOS

Immutable audit trail for compliance:
- All user/system actions
- Data access logging
- Change tracking (before/after)
- Tamper detection via hash chains
- Filterable audit queries
- Retention policies
- Export for compliance
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    PERMISSION_CHANGE = "permission_change"
    CONFIG_CHANGE = "config_change"
    API_CALL = "api_call"
    DATA_EXPORT = "data_export"
    DATA_ACCESS = "data_access"
    BILLING_EVENT = "billing_event"
    SECURITY_EVENT = "security_event"
    SYSTEM_EVENT = "system_event"


class AuditSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AuditEntry:
    entry_id: str = field(default_factory=lambda: str(uuid4()))
    action: AuditAction = AuditAction.SYSTEM_EVENT
    severity: AuditSeverity = AuditSeverity.INFO
    actor_id: str = ""  # user or system ID
    actor_type: str = "user"  # user, system, agent, api_key
    tenant_id: str = ""
    resource_type: str = ""
    resource_id: str = ""
    description: str = ""
    before_state: Dict[str, Any] = field(default_factory=dict)
    after_state: Dict[str, Any] = field(default_factory=dict)
    ip_address: str = ""
    user_agent: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hash_chain: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "entry_id": self.entry_id, "action": self.action.value,
            "severity": self.severity.value, "actor_id": self.actor_id,
            "actor_type": self.actor_type, "tenant_id": self.tenant_id,
            "resource_type": self.resource_type, "resource_id": self.resource_id,
            "description": self.description, "timestamp": self.timestamp}
        if self.before_state:
            d["before_state"] = self.before_state
        if self.after_state:
            d["after_state"] = self.after_state
        return d


class AuditLogger:
    """Immutable, chain-linked audit trail."""

    def __init__(self, *, max_entries: int = 100000,
                 retention_days: int = 365) -> None:
        self._entries: List[AuditEntry] = []
        self._max_entries = max_entries
        self._retention_days = retention_days
        self._last_hash = "0" * 64
        self._metrics: Dict[str, int] = defaultdict(int)

    def log(self, action: AuditAction, *, actor_id: str = "",
            actor_type: str = "user", tenant_id: str = "",
            resource_type: str = "", resource_id: str = "",
            description: str = "",
            severity: AuditSeverity = AuditSeverity.INFO,
            before_state: Dict[str, Any] | None = None,
            after_state: Dict[str, Any] | None = None,
            ip_address: str = "", user_agent: str = "",
            metadata: Dict[str, Any] | None = None) -> AuditEntry:

        entry = AuditEntry(
            action=action, severity=severity,
            actor_id=actor_id, actor_type=actor_type,
            tenant_id=tenant_id, resource_type=resource_type,
            resource_id=resource_id, description=description,
            before_state=before_state or {}, after_state=after_state or {},
            ip_address=ip_address, user_agent=user_agent,
            metadata=metadata or {})

        # Hash chain for tamper detection
        chain_data = json.dumps({
            "prev": self._last_hash, "entry_id": entry.entry_id,
            "action": entry.action.value, "actor": entry.actor_id,
            "timestamp": entry.timestamp}, sort_keys=True)
        entry.hash_chain = hashlib.sha256(chain_data.encode()).hexdigest()
        self._last_hash = entry.hash_chain

        self._entries.append(entry)
        self._metrics[action.value] += 1
        self._metrics["total"] += 1

        if severity == AuditSeverity.CRITICAL:
            logger.warning("CRITICAL AUDIT: %s by %s on %s/%s: %s",
                           action.value, actor_id, resource_type,
                           resource_id, description)

        # Enforce limits
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        return entry

    # ---- query ----
    def query(self, *, action: AuditAction | None = None,
              actor_id: str = "", tenant_id: str = "",
              resource_type: str = "", resource_id: str = "",
              severity: AuditSeverity | None = None,
              start_date: str = "", end_date: str = "",
              limit: int = 100) -> List[Dict[str, Any]]:
        results = self._entries
        if action:
            results = [e for e in results if e.action == action]
        if actor_id:
            results = [e for e in results if e.actor_id == actor_id]
        if tenant_id:
            results = [e for e in results if e.tenant_id == tenant_id]
        if resource_type:
            results = [e for e in results if e.resource_type == resource_type]
        if resource_id:
            results = [e for e in results if e.resource_id == resource_id]
        if severity:
            results = [e for e in results if e.severity == severity]
        if start_date:
            results = [e for e in results if e.timestamp >= start_date]
        if end_date:
            results = [e for e in results if e.timestamp <= end_date]
        return [e.to_dict() for e in results[-limit:]]

    # ---- integrity ----
    def verify_chain(self) -> Dict[str, Any]:
        if not self._entries:
            return {"valid": True, "entries_checked": 0}

        prev_hash = "0" * 64
        violations = []
        for i, entry in enumerate(self._entries):
            chain_data = json.dumps({
                "prev": prev_hash, "entry_id": entry.entry_id,
                "action": entry.action.value, "actor": entry.actor_id,
                "timestamp": entry.timestamp}, sort_keys=True)
            expected = hashlib.sha256(chain_data.encode()).hexdigest()
            if entry.hash_chain != expected:
                violations.append({"index": i, "entry_id": entry.entry_id})
            prev_hash = entry.hash_chain

        return {"valid": len(violations) == 0,
                "entries_checked": len(self._entries),
                "violations": violations}

    # ---- export ----
    def export(self, *, format: str = "json", **filters: Any) -> str:
        entries = self.query(**filters, limit=self._max_entries)
        if format == "json":
            return json.dumps(entries, indent=2, default=str)
        elif format == "csv":
            if not entries:
                return ""
            headers = list(entries[0].keys())
            lines = [",".join(headers)]
            for e in entries:
                lines.append(",".join(str(e.get(h, "")).replace(",", ";") for h in headers))
            return "\n".join(lines)
        return json.dumps(entries, default=str)

    # ---- retention ----
    def apply_retention(self) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self._retention_days)).isoformat()
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.timestamp >= cutoff]
        removed = before - len(self._entries)
        if removed:
            logger.info("Audit retention: removed %d entries older than %d days",
                         removed, self._retention_days)
        return removed

    def get_metrics(self) -> Dict[str, Any]:
        return {**dict(self._metrics), "total_entries": len(self._entries),
                "retention_days": self._retention_days}


_logger: AuditLogger | None = None

def get_audit_logger() -> AuditLogger:
    global _logger
    if not _logger:
        _logger = AuditLogger()
    return _logger
