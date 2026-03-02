"""
API Key Management Service — CognitionOS

Features:
- API key generation with scoped permissions
- Key rotation and revocation
- Rate limit association per key
- Usage tracking per key
- Expiration management
- Key hierarchy (master → child keys)
- Audit trail
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class KeyType(str, Enum):
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    TESTING = "testing"
    INTERNAL = "internal"


class KeyStatus(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"
    SUSPENDED = "suspended"


@dataclass
class APIKey:
    key_id: str
    key_hash: str  # store only hash
    prefix: str  # first 8 chars for display
    name: str
    tenant_id: str
    user_id: str
    key_type: KeyType = KeyType.PRODUCTION
    status: KeyStatus = KeyStatus.ACTIVE
    permissions: Set[str] = field(default_factory=set)
    rate_limit_per_minute: int = 60
    rate_limit_per_day: int = 10000
    parent_key_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: Optional[str] = None
    last_used_at: Optional[str] = None
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc).isoformat() > self.expires_at

    @property
    def is_valid(self) -> bool:
        return self.status == KeyStatus.ACTIVE and not self.is_expired

    def to_dict(self, *, include_hash: bool = False) -> Dict[str, Any]:
        d = {
            "key_id": self.key_id, "prefix": self.prefix,
            "name": self.name, "tenant_id": self.tenant_id,
            "key_type": self.key_type.value, "status": self.status.value,
            "permissions": sorted(self.permissions),
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "created_at": self.created_at, "expires_at": self.expires_at,
            "last_used_at": self.last_used_at, "usage_count": self.usage_count}
        if include_hash:
            d["key_hash"] = self.key_hash
        return d


@dataclass
class KeyValidationResult:
    valid: bool
    key_id: Optional[str] = None
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    permissions: Set[str] = field(default_factory=set)
    reason: str = ""
    rate_limit_per_minute: int = 60


class APIKeyManager:
    """Manages API key lifecycle, validation, and usage tracking."""

    KEY_PREFIX = "cog_"
    KEY_LENGTH = 48

    def __init__(self) -> None:
        self._keys: Dict[str, APIKey] = {}  # key_id -> APIKey
        self._hash_index: Dict[str, str] = {}  # key_hash -> key_id
        self._usage: Dict[str, List[float]] = defaultdict(list)  # key_id -> timestamps
        self._audit: List[Dict[str, Any]] = []

    # ---- key generation ----
    def create_key(
        self, *, name: str, tenant_id: str, user_id: str,
        key_type: KeyType = KeyType.PRODUCTION,
        permissions: Set[str] | None = None,
        rate_limit_per_minute: int = 60,
        rate_limit_per_day: int = 10000,
        expires_in_days: int | None = None,
        parent_key_id: str | None = None,
    ) -> tuple[str, APIKey]:
        """Returns (raw_key, api_key_obj). Raw key is shown only once."""
        raw_key = self.KEY_PREFIX + secrets.token_urlsafe(self.KEY_LENGTH)
        key_hash = self._hash_key(raw_key)
        key_id = secrets.token_hex(16)
        prefix = raw_key[:12] + "..."

        expires_at = None
        if expires_in_days:
            expires_at = (datetime.now(timezone.utc) + timedelta(days=expires_in_days)).isoformat()

        api_key = APIKey(
            key_id=key_id, key_hash=key_hash, prefix=prefix,
            name=name, tenant_id=tenant_id, user_id=user_id,
            key_type=key_type, permissions=permissions or set(),
            rate_limit_per_minute=rate_limit_per_minute,
            rate_limit_per_day=rate_limit_per_day,
            parent_key_id=parent_key_id, expires_at=expires_at)

        self._keys[key_id] = api_key
        self._hash_index[key_hash] = key_id
        self._audit_event("create", key_id, tenant_id)
        logger.info("API key created: %s for tenant %s", prefix, tenant_id)
        return raw_key, api_key

    # ---- validation ----
    def validate_key(self, raw_key: str) -> KeyValidationResult:
        key_hash = self._hash_key(raw_key)
        key_id = self._hash_index.get(key_hash)
        if not key_id:
            return KeyValidationResult(valid=False, reason="key_not_found")

        api_key = self._keys.get(key_id)
        if not api_key:
            return KeyValidationResult(valid=False, reason="key_not_found")

        if api_key.status == KeyStatus.REVOKED:
            return KeyValidationResult(valid=False, key_id=key_id, reason="key_revoked")

        if api_key.status == KeyStatus.SUSPENDED:
            return KeyValidationResult(valid=False, key_id=key_id, reason="key_suspended")

        if api_key.is_expired:
            api_key.status = KeyStatus.EXPIRED
            return KeyValidationResult(valid=False, key_id=key_id, reason="key_expired")

        # Rate limit check
        if not self._check_rate_limit(key_id, api_key.rate_limit_per_minute):
            return KeyValidationResult(valid=False, key_id=key_id, reason="rate_limit_exceeded",
                                       rate_limit_per_minute=api_key.rate_limit_per_minute)

        # Record usage
        api_key.last_used_at = datetime.now(timezone.utc).isoformat()
        api_key.usage_count += 1
        self._usage[key_id].append(time.time())

        return KeyValidationResult(
            valid=True, key_id=key_id, tenant_id=api_key.tenant_id,
            user_id=api_key.user_id, permissions=api_key.permissions,
            reason="valid", rate_limit_per_minute=api_key.rate_limit_per_minute)

    def _check_rate_limit(self, key_id: str, limit: int) -> bool:
        now = time.time()
        cutoff = now - 60
        self._usage[key_id] = [t for t in self._usage[key_id] if t > cutoff]
        return len(self._usage[key_id]) < limit

    # ---- key management ----
    def revoke_key(self, key_id: str) -> bool:
        api_key = self._keys.get(key_id)
        if not api_key:
            return False
        api_key.status = KeyStatus.REVOKED
        self._audit_event("revoke", key_id, api_key.tenant_id)
        return True

    def suspend_key(self, key_id: str) -> bool:
        api_key = self._keys.get(key_id)
        if not api_key:
            return False
        api_key.status = KeyStatus.SUSPENDED
        self._audit_event("suspend", key_id, api_key.tenant_id)
        return True

    def reactivate_key(self, key_id: str) -> bool:
        api_key = self._keys.get(key_id)
        if not api_key or api_key.is_expired:
            return False
        api_key.status = KeyStatus.ACTIVE
        self._audit_event("reactivate", key_id, api_key.tenant_id)
        return True

    def rotate_key(self, key_id: str) -> tuple[str, APIKey] | None:
        old = self._keys.get(key_id)
        if not old:
            return None
        old.status = KeyStatus.REVOKED
        return self.create_key(
            name=f"{old.name} (rotated)", tenant_id=old.tenant_id,
            user_id=old.user_id, key_type=old.key_type,
            permissions=old.permissions,
            rate_limit_per_minute=old.rate_limit_per_minute,
            rate_limit_per_day=old.rate_limit_per_day,
            parent_key_id=key_id)

    # ---- queries ----
    def list_keys(self, *, tenant_id: str | None = None,
                  status: KeyStatus | None = None) -> List[Dict[str, Any]]:
        keys = list(self._keys.values())
        if tenant_id:
            keys = [k for k in keys if k.tenant_id == tenant_id]
        if status:
            keys = [k for k in keys if k.status == status]
        return [k.to_dict() for k in keys]

    def get_key(self, key_id: str) -> APIKey | None:
        return self._keys.get(key_id)

    def get_usage_stats(self, key_id: str) -> Dict[str, Any]:
        api_key = self._keys.get(key_id)
        if not api_key:
            return {}
        now = time.time()
        recent = [t for t in self._usage.get(key_id, []) if t > now - 3600]
        return {
            "key_id": key_id, "total_usage": api_key.usage_count,
            "last_hour_requests": len(recent),
            "last_used": api_key.last_used_at}

    # ---- helpers ----
    def _hash_key(self, raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def _audit_event(self, action: str, key_id: str, tenant_id: str) -> None:
        self._audit.append({
            "action": action, "key_id": key_id, "tenant_id": tenant_id,
            "timestamp": datetime.now(timezone.utc).isoformat()})
        if len(self._audit) > 5000:
            self._audit = self._audit[-5000:]

    def get_audit_log(self, *, key_id: str | None = None, limit: int = 100) -> List[Dict[str, Any]]:
        logs = self._audit
        if key_id:
            logs = [l for l in logs if l["key_id"] == key_id]
        return logs[-limit:]


_manager: APIKeyManager | None = None

def get_api_key_manager() -> APIKeyManager:
    global _manager
    if not _manager:
        _manager = APIKeyManager()
    return _manager
