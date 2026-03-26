"""
API Key Manager - Full Lifecycle API Key Management

Handles key generation, rotation, revocation, rate limiting,
scope-based access control, and usage analytics.
"""

import asyncio
import hashlib
import hmac
import logging
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class KeyStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    REVOKED = "revoked"
    EXPIRED = "expired"
    RATE_LIMITED = "rate_limited"


class KeyScope(str, Enum):
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    AGENT_EXECUTE = "agent:execute"
    WORKFLOW_MANAGE = "workflow:manage"
    BILLING_READ = "billing:read"
    BILLING_WRITE = "billing:write"
    ANALYTICS_READ = "analytics:read"
    PLUGIN_MANAGE = "plugin:manage"
    CODE_GENERATE = "code:generate"
    MEMORY_ACCESS = "memory:access"
    ALL = "*"


@dataclass
class RateLimitConfig:
    requests_per_second: int = 10
    requests_per_minute: int = 100
    requests_per_hour: int = 5_000
    requests_per_day: int = 100_000
    burst_limit: int = 50
    concurrent_limit: int = 10


@dataclass
class APIKey:
    """An API key with metadata and access control."""
    key_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    user_id: str = ""
    name: str = ""
    description: str = ""
    prefix: str = ""
    key_hash: str = ""
    status: KeyStatus = KeyStatus.ACTIVE
    scopes: Set[KeyScope] = field(default_factory=lambda: {KeyScope.READ})
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    rotated_from: Optional[str] = None
    revoked_at: Optional[datetime] = None
    revocation_reason: str = ""

    # Usage tracking
    total_requests: int = 0
    total_errors: int = 0
    total_tokens_consumed: int = 0

    # Rate limit state
    _minute_window_start: float = field(default_factory=time.monotonic)
    _minute_count: int = 0
    _hour_window_start: float = field(default_factory=time.monotonic)
    _hour_count: int = 0
    _day_window_start: float = field(default_factory=time.monotonic)
    _day_count: int = 0

    # Metadata
    allowed_ips: List[str] = field(default_factory=list)
    allowed_origins: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        if self.status != KeyStatus.ACTIVE:
            return False
        expires = self.expires_at
        if expires is not None and datetime.utcnow() > expires:
            return False
        return True

    @property
    def masked_key(self) -> str:
        return f"{self.prefix}...{'*' * 20}"

    def has_scope(self, scope: KeyScope) -> bool:
        return KeyScope.ALL in self.scopes or scope in self.scopes

    def to_dict(self) -> Dict[str, Any]:
        last_used = self.last_used_at
        expires = self.expires_at
        return {
            "key_id": self.key_id,
            "name": self.name,
            "prefix": self.prefix,
            "status": self.status.value,
            "scopes": [s.value for s in self.scopes],
            "created_at": self.created_at.isoformat(),
            "last_used_at": last_used.isoformat() if last_used is not None else None,
            "expires_at": expires.isoformat() if expires is not None else None,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "is_active": self.is_active,
        }


@dataclass
class KeyValidationResult:
    valid: bool
    key: Optional[APIKey] = None
    error: str = ""
    remaining_requests: int = 0
    scope_check: bool = True


class APIKeyManager:
    """
    Production API key manager.

    Features:
    - Secure key generation with prefix/hash
    - Scope-based access control
    - Rate limiting per key
    - Key rotation with grace period
    - IP/origin allowlisting
    - Usage analytics
    - Automatic expiration
    - Audit trail
    """

    KEY_PREFIX = "cos_"
    KEY_LENGTH = 48

    def __init__(self, hash_algorithm: str = "sha256"):
        self._keys: Dict[str, APIKey] = {}
        self._key_hash_index: Dict[str, str] = {}  # hash -> key_id
        self._tenant_keys: Dict[str, List[str]] = {}  # tenant_id -> [key_ids]
        self._hash_algorithm = hash_algorithm
        self._audit_log: List[Dict[str, Any]] = []

    # -- Key Generation -----------------------------------------------------

    async def create_key(
        self,
        tenant_id: str,
        user_id: str,
        name: str,
        description: str = "",
        scopes: Optional[Set[KeyScope]] = None,
        rate_limit: Optional[RateLimitConfig] = None,
        expires_in_days: Optional[int] = None,
        allowed_ips: Optional[List[str]] = None,
        allowed_origins: Optional[List[str]] = None,
    ) -> Tuple[str, APIKey]:
        """Create a new API key. Returns (raw_key, api_key_record)."""
        raw_key = self._generate_raw_key()
        key_hash = self._hash_key(raw_key)
        prefix = str(raw_key[0:12])

        key = APIKey(
            tenant_id=tenant_id,
            user_id=user_id,
            name=name,
            description=description,
            prefix=prefix,
            key_hash=key_hash,
            scopes=scopes or {KeyScope.READ},
            rate_limit=rate_limit or RateLimitConfig(),
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days) if expires_in_days else None,
            allowed_ips=allowed_ips or [],
            allowed_origins=allowed_origins or [],
        )

        self._keys[key.key_id] = key
        self._key_hash_index[key_hash] = key.key_id

        if tenant_id not in self._tenant_keys:
            self._tenant_keys[tenant_id] = []
        self._tenant_keys[tenant_id].append(key.key_id)

        self._audit("key_created", key)
        logger.info("Created API key %s for tenant %s", key.key_id, tenant_id)

        return raw_key, key

    async def rotate_key(
        self, key_id: str, grace_period_hours: int = 24
    ) -> Tuple[str, APIKey]:
        """Rotate an API key, keeping the old one active for a grace period."""
        old_key = self._keys.get(key_id)
        if not old_key:
            raise ValueError(f"Key not found: {key_id}")

        # Create new key with same config
        new_raw, new_key = await self.create_key(
            tenant_id=old_key.tenant_id,
            user_id=old_key.user_id,
            name=f"{old_key.name} (rotated)",
            description=old_key.description,
            scopes=old_key.scopes,
            rate_limit=old_key.rate_limit,
            allowed_ips=old_key.allowed_ips,
            allowed_origins=old_key.allowed_origins,
        )
        new_key.rotated_from = key_id

        # Set old key to expire after grace period
        old_key.expires_at = datetime.utcnow() + timedelta(hours=grace_period_hours)
        old_key.metadata["rotated_to"] = new_key.key_id

        self._audit("key_rotated", old_key, {"new_key_id": new_key.key_id})
        return new_raw, new_key

    async def revoke_key(self, key_id: str, reason: str = "") -> bool:
        key = self._keys.get(key_id)
        if not key:
            return False

        key.status = KeyStatus.REVOKED
        key.revoked_at = datetime.utcnow()
        key.revocation_reason = reason

        # Remove from hash index
        self._key_hash_index.pop(key.key_hash, None)

        self._audit("key_revoked", key, {"reason": reason})
        logger.info("Revoked API key %s: %s", key_id, reason)
        return True

    # -- Validation ---------------------------------------------------------

    async def validate_key(
        self,
        raw_key: str,
        required_scope: Optional[KeyScope] = None,
        client_ip: Optional[str] = None,
        origin: Optional[str] = None,
    ) -> KeyValidationResult:
        """Validate an API key and check permissions."""
        key_hash = self._hash_key(raw_key)
        key_id = self._key_hash_index.get(key_hash)

        if not key_id:
            return KeyValidationResult(valid=False, error="invalid_key")

        key = self._keys.get(key_id)
        if not key:
            return KeyValidationResult(valid=False, error="key_not_found")

        # Status check
        if not key.is_active:
            expires = key.expires_at
            if expires is not None and datetime.utcnow() > expires:
                key.status = KeyStatus.EXPIRED
                return KeyValidationResult(valid=False, key=key, error="key_expired")
            return KeyValidationResult(valid=False, key=key, error="key_inactive")

        # IP check
        if key.allowed_ips and client_ip and client_ip not in key.allowed_ips:
            return KeyValidationResult(valid=False, key=key, error="ip_not_allowed")

        # Origin check
        if key.allowed_origins and origin and origin not in key.allowed_origins:
            return KeyValidationResult(valid=False, key=key, error="origin_not_allowed")

        # Scope check
        scope_ok = True
        if required_scope and not key.has_scope(required_scope):
            scope_ok = False

        # Rate limit check
        now = time.monotonic()
        rate_ok, remaining = self._check_rate_limit(key, now)
        if not rate_ok:
            key.status = KeyStatus.RATE_LIMITED
            return KeyValidationResult(
                valid=False, key=key, error="rate_limited",
                remaining_requests=0, scope_check=scope_ok,
            )

        # Update usage
        key.last_used_at = datetime.utcnow()
        key.total_requests += 1

        return KeyValidationResult(
            valid=scope_ok,
            key=key,
            error="" if scope_ok else "insufficient_scope",
            remaining_requests=remaining,
            scope_check=scope_ok,
        )

    def _check_rate_limit(self, key: APIKey, now: float) -> Tuple[bool, int]:
        rl = key.rate_limit

        # Per-minute
        if now - key._minute_window_start > 60:
            key._minute_window_start = now
            key._minute_count = 0
        key._minute_count += 1
        if key._minute_count > rl.requests_per_minute:
            return False, 0

        # Per-hour
        if now - key._hour_window_start > 3600:
            key._hour_window_start = now
            key._hour_count = 0
        key._hour_count += 1
        if key._hour_count > rl.requests_per_hour:
            return False, 0

        # Per-day
        if now - key._day_window_start > 86400:
            key._day_window_start = now
            key._day_count = 0
        key._day_count += 1
        if key._day_count > rl.requests_per_day:
            return False, 0

        remaining = min(
            rl.requests_per_minute - key._minute_count,
            rl.requests_per_hour - key._hour_count,
            rl.requests_per_day - key._day_count,
        )
        return True, remaining

    # -- Queries ------------------------------------------------------------

    def get_key(self, key_id: str) -> Optional[APIKey]:
        return self._keys.get(key_id)

    def get_tenant_keys(self, tenant_id: str) -> List[APIKey]:
        key_ids = self._tenant_keys.get(tenant_id, [])
        return [self._keys[kid] for kid in key_ids if kid in self._keys]

    def get_key_usage(self, key_id: str) -> Dict[str, Any]:
        key = self._keys.get(key_id)
        if not key:
            return {}
        last_used = key.last_used_at
        return {
            "key_id": key_id,
            "total_requests": key.total_requests,
            "total_errors": key.total_errors,
            "total_tokens": key.total_tokens_consumed,
            "last_used": last_used.isoformat() if last_used is not None else None,
            "current_minute_count": key._minute_count,
            "current_hour_count": key._hour_count,
            "current_day_count": key._day_count,
        }

    def get_stats(self) -> Dict[str, Any]:
        active = sum(1 for k in self._keys.values() if k.is_active)
        return {
            "total_keys": len(self._keys),
            "active_keys": active,
            "revoked_keys": sum(1 for k in self._keys.values() if k.status == KeyStatus.REVOKED),
            "expired_keys": sum(1 for k in self._keys.values() if k.status == KeyStatus.EXPIRED),
            "total_requests_all": sum(k.total_requests for k in self._keys.values()),
        }

    # -- Internal -----------------------------------------------------------

    def _generate_raw_key(self) -> str:
        random_part = secrets.token_urlsafe(self.KEY_LENGTH)
        return f"{self.KEY_PREFIX}{random_part}"

    def _hash_key(self, raw_key: str) -> str:
        return hashlib.new(self._hash_algorithm, raw_key.encode()).hexdigest()

    def _audit(self, action: str, key: APIKey, extra: Optional[Dict[str, Any]] = None) -> None:
        entry = {
            "action": action,
            "key_id": key.key_id,
            "tenant_id": key.tenant_id,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if extra:
            entry.update(extra)
        self._audit_log.append(entry)

    def get_audit_log(self, key_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        entries = self._audit_log
        if key_id:
            entries = [e for e in entries if e.get("key_id") == key_id]
        return list(entries[-limit:])
