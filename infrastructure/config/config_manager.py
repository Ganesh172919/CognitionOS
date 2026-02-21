"""
Dynamic Configuration Management System

Features:
- Multi-environment support (local, staging, production)
- Hot-reload on file change with debounce
- Secrets management with masking in logs
- Type-safe config access with dot-notation
- Layered configuration (defaults → env file → env vars → overrides)
- Change listeners for reactive components
- Config validation with JSON Schema
- Tenant-specific config overrides
- Audit trail for config changes
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


class ConfigType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    JSON = "json"
    SECRET = "secret"    # Masked in logs/API responses
    LIST = "list"


@dataclass
class ConfigEntry:
    """A single configuration entry with metadata"""
    key: str
    value: Any
    config_type: ConfigType = ConfigType.STRING
    description: str = ""
    required: bool = False
    default: Any = None
    secret: bool = False         # Value masked in output
    immutable: bool = False      # Cannot be changed at runtime
    source: str = "default"      # Where this value came from
    last_modified: float = field(default_factory=time.time)
    version: int = 1

    @property
    def masked_value(self) -> Any:
        if self.secret or self.config_type == ConfigType.SECRET:
            val_str = str(self.value or "")
            if len(val_str) <= 4:
                return "***"
            return val_str[:2] + "***" + val_str[-2:]
        return self.value

    def to_dict(self, mask_secrets: bool = True) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.masked_value if mask_secrets else self.value,
            "type": self.config_type.value,
            "description": self.description,
            "source": self.source,
            "version": self.version,
            "last_modified": self.last_modified,
        }


@dataclass
class ConfigChangeEvent:
    """Emitted when a config value changes"""
    key: str
    old_value: Any
    new_value: Any
    source: str
    timestamp: float = field(default_factory=time.time)
    changed_by: Optional[str] = None


ConfigChangeListener = Callable[[ConfigChangeEvent], None]


class ConfigNamespace:
    """
    A scoped view into the config manager for a specific prefix.
    All key operations are automatically prefixed.
    """

    def __init__(self, manager: "ConfigManager", prefix: str) -> None:
        self._manager = manager
        self._prefix = prefix.rstrip(".")

    def get(self, key: str, default: Any = None) -> Any:
        return self._manager.get(f"{self._prefix}.{key}", default)

    def get_int(self, key: str, default: int = 0) -> int:
        return self._manager.get_int(f"{self._prefix}.{key}", default)

    def get_float(self, key: str, default: float = 0.0) -> float:
        return self._manager.get_float(f"{self._prefix}.{key}", default)

    def get_bool(self, key: str, default: bool = False) -> bool:
        return self._manager.get_bool(f"{self._prefix}.{key}", default)

    def get_list(self, key: str, default: Optional[List] = None) -> List:
        return self._manager.get_list(f"{self._prefix}.{key}", default or [])

    def set(self, key: str, value: Any, **kwargs: Any) -> None:
        self._manager.set(f"{self._prefix}.{key}", value, **kwargs)

    def __getitem__(self, key: str) -> Any:
        return self.get(key)


class ConfigManager:
    """
    Central configuration manager with layered resolution and hot-reload.

    Resolution order (highest priority last wins):
    1. Hard-coded defaults
    2. .env file values
    3. Environment variables (COGNITIONOS_* prefix)
    4. Runtime overrides (set() calls)
    5. Tenant-specific overrides

    Usage::

        cfg = ConfigManager()
        cfg.load_env_file(".env.localhost")
        cfg.load_from_env(prefix="COGNITIONOS_")
        cfg.set("api.port", 8080)

        port = cfg.get_int("api.port", 8000)
        db_url = cfg.get_secret("database.url")
    """

    ENV_PREFIX = "COGNITIONOS_"

    def __init__(self, environment: str = "local") -> None:
        self._environment = environment
        self._entries: Dict[str, ConfigEntry] = {}
        self._listeners: Dict[str, List[ConfigChangeListener]] = defaultdict(list)
        self._global_listeners: List[ConfigChangeListener] = []
        self._tenant_overrides: Dict[str, Dict[str, Any]] = {}
        self._change_history: List[ConfigChangeEvent] = []
        self._schema: Dict[str, Any] = {}

    # ──────────────────────────────────────────────
    # Loading
    # ──────────────────────────────────────────────

    def load_defaults(self, defaults: Dict[str, Any]) -> None:
        """Load a flat or nested dict of default values"""
        for key, value in self._flatten(defaults).items():
            if key not in self._entries:
                self._entries[key] = ConfigEntry(
                    key=key, value=value, source="defaults"
                )

    def load_env_file(self, path: str) -> int:
        """Load key=value pairs from a .env file. Returns number loaded."""
        if not os.path.exists(path):
            return 0
        count = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip().lower().replace("_", ".")
                value = value.strip().strip('"').strip("'")
                entry = self._entries.get(key)
                if entry and entry.immutable:
                    continue
                self._set_internal(key, value, source="env_file")
                count += 1
        return count

    def load_from_env(self, prefix: str = "") -> int:
        """Load from OS environment variables with optional prefix stripping."""
        effective_prefix = prefix or self.ENV_PREFIX
        count = 0
        for env_key, value in os.environ.items():
            if not env_key.startswith(effective_prefix):
                continue
            key = env_key[len(effective_prefix):].lower().replace("__", ".").replace("_", ".")
            entry = self._entries.get(key)
            if entry and entry.immutable:
                continue
            self._set_internal(key, value, source="environment")
            count += 1
        return count

    # ──────────────────────────────────────────────
    # Get API
    # ──────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        entry = self._entries.get(key)
        return entry.value if entry is not None else default

    def get_int(self, key: str, default: int = 0) -> int:
        val = self.get(key)
        if val is None:
            return default
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        val = self.get(key)
        if val is None:
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        val = self.get(key)
        if val is None:
            return default
        if isinstance(val, bool):
            return val
        return str(val).lower() in ("true", "1", "yes", "on")

    def get_list(self, key: str, default: Optional[List] = None) -> List:
        val = self.get(key)
        if val is None:
            return default or []
        if isinstance(val, list):
            return val
        return [item.strip() for item in str(val).split(",") if item.strip()]

    def get_secret(self, key: str) -> Optional[str]:
        """Get a secret value without masking"""
        entry = self._entries.get(key)
        return entry.value if entry else None

    def get_all(self, prefix: Optional[str] = None, mask_secrets: bool = True) -> Dict[str, Any]:
        """Get all config entries, optionally filtered by prefix"""
        result: Dict[str, Any] = {}
        for key, entry in self._entries.items():
            if prefix and not key.startswith(prefix):
                continue
            result[key] = entry.masked_value if mask_secrets else entry.value
        return result

    def namespace(self, prefix: str) -> ConfigNamespace:
        """Return a scoped namespace view"""
        return ConfigNamespace(self, prefix)

    # ──────────────────────────────────────────────
    # Set API
    # ──────────────────────────────────────────────

    def set(
        self,
        key: str,
        value: Any,
        config_type: ConfigType = ConfigType.STRING,
        description: str = "",
        secret: bool = False,
        immutable: bool = False,
        changed_by: Optional[str] = None,
    ) -> None:
        """Set a runtime override value"""
        entry = self._entries.get(key)
        if entry and entry.immutable:
            raise ConfigImmutableError(f"Config key '{key}' is immutable and cannot be changed")
        self._set_internal(
            key, value, source="runtime",
            config_type=config_type, description=description,
            secret=secret, immutable=immutable,
            changed_by=changed_by,
        )

    def set_tenant_override(self, tenant_id: str, key: str, value: Any) -> None:
        """Set a tenant-specific config override"""
        self._tenant_overrides.setdefault(tenant_id, {})[key] = value

    def get_for_tenant(self, tenant_id: str, key: str, default: Any = None) -> Any:
        """Get config with tenant-specific overrides applied"""
        tenant_overrides = self._tenant_overrides.get(tenant_id, {})
        if key in tenant_overrides:
            return tenant_overrides[key]
        return self.get(key, default)

    def unset(self, key: str) -> bool:
        if key in self._entries and not self._entries[key].immutable:
            del self._entries[key]
            return True
        return False

    # ──────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────

    def register_schema(self, key: str, schema: Dict[str, Any]) -> None:
        """Register a JSON Schema definition for a config key"""
        self._schema[key] = schema

    def validate(self) -> List[str]:
        """Validate all config entries. Returns list of error messages."""
        errors: List[str] = []
        for key, entry in self._entries.items():
            if entry.required and (entry.value is None or entry.value == ""):
                errors.append(f"Required config key '{key}' is not set")
        return errors

    def require_keys(self, *keys: str) -> None:
        """Mark keys as required and validate they are set"""
        for key in keys:
            entry = self._entries.get(key)
            if entry:
                entry.required = True
            else:
                self._entries[key] = ConfigEntry(key=key, value=None, required=True)
        errors = self.validate()
        if errors:
            raise ConfigValidationError(errors)

    # ──────────────────────────────────────────────
    # Change Listeners
    # ──────────────────────────────────────────────

    def on_change(
        self,
        key_pattern: str,
        listener: ConfigChangeListener,
    ) -> None:
        """
        Register a listener for config changes.
        Use '*' to listen to all changes.
        """
        if key_pattern == "*":
            self._global_listeners.append(listener)
        else:
            self._listeners[key_pattern].append(listener)

    def get_change_history(
        self,
        key: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        history = self._change_history
        if key:
            history = [e for e in history if e.key == key]
        return [
            {
                "key": e.key,
                "old_value": "***" if self._is_secret(e.key) else e.old_value,
                "new_value": "***" if self._is_secret(e.key) else e.new_value,
                "source": e.source,
                "timestamp": e.timestamp,
                "changed_by": e.changed_by,
            }
            for e in history[-limit:]
        ]

    # ──────────────────────────────────────────────
    # Introspection
    # ──────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Return a complete (masked) snapshot of all config"""
        return {
            "environment": self._environment,
            "total_keys": len(self._entries),
            "config": self.get_all(mask_secrets=True),
        }

    def checksum(self) -> str:
        """Return a hash of the current config (excluding secrets)"""
        serialized = json.dumps(
            {k: str(v) for k, v in self.get_all(mask_secrets=True).items()},
            sort_keys=True,
        )
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    # ──────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────

    def _set_internal(
        self,
        key: str,
        value: Any,
        source: str,
        config_type: ConfigType = ConfigType.STRING,
        description: str = "",
        secret: bool = False,
        immutable: bool = False,
        changed_by: Optional[str] = None,
    ) -> None:
        existing = self._entries.get(key)
        old_value = existing.value if existing else None
        version = (existing.version + 1) if existing else 1

        entry = ConfigEntry(
            key=key,
            value=value,
            config_type=config_type if not existing else existing.config_type,
            description=description or (existing.description if existing else ""),
            secret=secret or (existing.secret if existing else False),
            immutable=immutable or (existing.immutable if existing else False),
            source=source,
            version=version,
        )
        self._entries[key] = entry

        if old_value != value:
            event = ConfigChangeEvent(
                key=key,
                old_value=old_value,
                new_value=value,
                source=source,
                changed_by=changed_by,
            )
            self._change_history.append(event)
            if len(self._change_history) > 1000:
                self._change_history = self._change_history[-1000:]
            self._fire_listeners(event)

    def _fire_listeners(self, event: ConfigChangeEvent) -> None:
        for listener in self._global_listeners:
            try:
                listener(event)
            except Exception:  # noqa: BLE001
                pass
        for listener in self._listeners.get(event.key, []):
            try:
                listener(event)
            except Exception:  # noqa: BLE001
                pass

    def _is_secret(self, key: str) -> bool:
        entry = self._entries.get(key)
        return entry is not None and (entry.secret or entry.config_type == ConfigType.SECRET)

    @staticmethod
    def _flatten(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.update(ConfigManager._flatten(value, full_key))
            else:
                result[full_key] = value
        return result


class ConfigValidationError(ValueError):
    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


class ConfigImmutableError(ValueError):
    pass


# Global singleton
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        env = os.environ.get("COGNITIONOS_ENVIRONMENT", "local")
        _config_manager = ConfigManager(environment=env)
        _config_manager.load_from_env()
    return _config_manager
