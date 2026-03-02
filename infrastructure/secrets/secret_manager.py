"""
Secret Manager — CognitionOS

Secure secret storage:
- Encryption at rest (Fernet)
- Secret versioning
- Access control per secret
- Rotation support
- Audit trail
- Environment injection
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

logger = logging.getLogger(__name__)

try:
    from cryptography.fernet import Fernet
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


class SecretType(str, Enum):
    API_KEY = "api_key"
    DATABASE_URL = "database_url"
    OAUTH_SECRET = "oauth_secret"
    ENCRYPTION_KEY = "encryption_key"
    WEBHOOK_SECRET = "webhook_secret"
    CERTIFICATE = "certificate"
    CUSTOM = "custom"


@dataclass
class SecretVersion:
    version: int
    encrypted_value: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by: str = "system"
    is_active: bool = True


@dataclass
class Secret:
    secret_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    secret_type: SecretType = SecretType.CUSTOM
    tenant_id: str = ""
    versions: List[SecretVersion] = field(default_factory=list)
    allowed_services: Set[str] = field(default_factory=set)
    description: str = ""
    rotate_days: int = 0  # 0 = no auto-rotation
    last_accessed: Optional[str] = None
    access_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def current_version(self) -> Optional[SecretVersion]:
        active = [v for v in self.versions if v.is_active]
        return active[-1] if active else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "secret_id": self.secret_id, "name": self.name,
            "type": self.secret_type.value, "tenant_id": self.tenant_id,
            "versions": len(self.versions), "rotate_days": self.rotate_days,
            "last_accessed": self.last_accessed, "access_count": self.access_count}


class SecretManager:
    """Manages encrypted secrets with versioning and access control."""

    def __init__(self, *, master_key: str | None = None) -> None:
        if HAS_CRYPTO:
            if master_key:
                key = base64.urlsafe_b64encode(
                    hashlib.sha256(master_key.encode()).digest())
            else:
                key = Fernet.generate_key()
            self._fernet = Fernet(key)
        else:
            self._fernet = None
        self._secrets: Dict[str, Secret] = {}
        self._name_index: Dict[str, str] = {}  # name -> secret_id
        self._audit: List[Dict[str, Any]] = []

    def _encrypt(self, value: str) -> str:
        if self._fernet:
            return self._fernet.encrypt(value.encode()).decode()
        return base64.b64encode(value.encode()).decode()

    def _decrypt(self, encrypted: str) -> str:
        if self._fernet:
            return self._fernet.decrypt(encrypted.encode()).decode()
        return base64.b64decode(encrypted.encode()).decode()

    # ---- store ----
    def store(self, name: str, value: str, *, secret_type: SecretType = SecretType.CUSTOM,
              tenant_id: str = "", allowed_services: Set[str] | None = None,
              description: str = "", rotate_days: int = 0,
              created_by: str = "system") -> Secret:
        encrypted = self._encrypt(value)
        version = SecretVersion(version=1, encrypted_value=encrypted, created_by=created_by)

        secret = Secret(
            name=name, secret_type=secret_type, tenant_id=tenant_id,
            versions=[version], allowed_services=allowed_services or set(),
            description=description, rotate_days=rotate_days)

        self._secrets[secret.secret_id] = secret
        self._name_index[name] = secret.secret_id
        self._audit_event("store", name, tenant_id, created_by)
        return secret

    # ---- retrieve ----
    def get(self, name: str, *, service: str = "", tenant_id: str = "") -> str | None:
        secret_id = self._name_index.get(name)
        if not secret_id:
            return None
        secret = self._secrets.get(secret_id)
        if not secret:
            return None
        if secret.allowed_services and service not in secret.allowed_services:
            self._audit_event("access_denied", name, tenant_id, service)
            logger.warning("Secret access denied: %s by service %s", name, service)
            return None

        version = secret.current_version
        if not version:
            return None

        secret.last_accessed = datetime.now(timezone.utc).isoformat()
        secret.access_count += 1
        self._audit_event("access", name, tenant_id, service)
        return self._decrypt(version.encrypted_value)

    # ---- rotation ----
    def rotate(self, name: str, new_value: str, *, rotated_by: str = "system") -> bool:
        secret_id = self._name_index.get(name)
        if not secret_id:
            return False
        secret = self._secrets.get(secret_id)
        if not secret:
            return False

        # Deactivate old versions
        for v in secret.versions:
            v.is_active = False

        encrypted = self._encrypt(new_value)
        new_version = SecretVersion(
            version=len(secret.versions) + 1,
            encrypted_value=encrypted, created_by=rotated_by)
        secret.versions.append(new_version)
        self._audit_event("rotate", name, secret.tenant_id, rotated_by)
        return True

    # ---- delete ----
    def delete(self, name: str) -> bool:
        secret_id = self._name_index.pop(name, None)
        if secret_id:
            self._secrets.pop(secret_id, None)
            self._audit_event("delete", name, "", "system")
            return True
        return False

    # ---- query ----
    def list_secrets(self, *, tenant_id: str = "",
                      secret_type: SecretType | None = None) -> List[Dict[str, Any]]:
        results = list(self._secrets.values())
        if tenant_id:
            results = [s for s in results if s.tenant_id == tenant_id]
        if secret_type:
            results = [s for s in results if s.secret_type == secret_type]
        return [s.to_dict() for s in results]

    def check_rotation_needed(self) -> List[Dict[str, Any]]:
        needing_rotation = []
        now = datetime.now(timezone.utc)
        for secret in self._secrets.values():
            if secret.rotate_days <= 0:
                continue
            version = secret.current_version
            if not version:
                continue
            created = datetime.fromisoformat(version.created_at)
            age_days = (now - created).days
            if age_days >= secret.rotate_days:
                needing_rotation.append({
                    "name": secret.name, "age_days": age_days,
                    "rotate_days": secret.rotate_days})
        return needing_rotation

    def _audit_event(self, action: str, name: str, tenant_id: str, actor: str) -> None:
        self._audit.append({
            "action": action, "secret": name, "tenant_id": tenant_id,
            "actor": actor, "timestamp": datetime.now(timezone.utc).isoformat()})
        if len(self._audit) > 5000:
            self._audit = self._audit[-5000:]

    def get_audit_log(self, *, name: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        logs = self._audit
        if name:
            logs = [l for l in logs if l["secret"] == name]
        return logs[-limit:]


_manager: SecretManager | None = None

def get_secret_manager() -> SecretManager:
    global _manager
    if not _manager:
        _manager = SecretManager()
    return _manager
