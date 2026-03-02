"""
Configuration Manager — CognitionOS Core Engine

Hierarchical configuration system with:
- Multiple config sources (env, file, remote, defaults)
- Hot-reload with change notification
- Type-safe access with validation
- Secrets management integration
- Per-tenant config overrides
- Config versioning and audit trail
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import (
    Any, Callable, Awaitable, Dict, List, Optional, Set, Type, Union,
)

logger = logging.getLogger(__name__)


class ConfigSource(IntEnum):
    """Configuration source priority (higher = overrides lower)."""
    DEFAULT = 0
    FILE = 10
    ENVIRONMENT = 20
    REMOTE = 30
    TENANT = 40
    RUNTIME = 50


@dataclass
class ConfigEntry:
    key: str
    value: Any
    source: ConfigSource
    value_type: str = "string"
    description: str = ""
    secret: bool = False
    validator: Optional[Callable[[Any], bool]] = None
    updated_at: float = field(default_factory=time.time)

    def validate(self) -> bool:
        if self.validator:
            return self.validator(self.value)
        return True


@dataclass
class ConfigChangeEvent:
    key: str
    old_value: Any
    new_value: Any
    source: ConfigSource
    timestamp: float = field(default_factory=time.time)
    changed_by: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "source": self.source.name,
            "timestamp": self.timestamp, "changed_by": self.changed_by,
        }


class ConfigNamespace:
    """Isolated config namespace with dot-notation access."""

    def __init__(self, prefix: str, store: Dict[str, ConfigEntry]):
        self._prefix = prefix
        self._store = store

    def get(self, key: str, default: Any = None) -> Any:
        full_key = f"{self._prefix}.{key}" if self._prefix else key
        entry = self._store.get(full_key)
        return entry.value if entry else default

    def get_int(self, key: str, default: int = 0) -> int:
        val = self.get(key, default)
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        val = self.get(key, default)
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        val = self.get(key, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes", "on")
        return bool(val)

    def get_list(self, key: str, default: Optional[List] = None) -> List:
        val = self.get(key, default)
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            return [v.strip() for v in val.split(",") if v.strip()]
        return default or []

    def get_dict(self, key: str, default: Optional[Dict] = None) -> Dict:
        val = self.get(key, default)
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, ValueError):
                pass
        return default or {}

    def keys(self) -> List[str]:
        prefix = f"{self._prefix}." if self._prefix else ""
        return [k[len(prefix):] for k in self._store.keys()
                if k.startswith(prefix)]


class ConfigManager:
    """
    Hierarchical configuration manager with hot-reload, validation,
    secrets management, and per-tenant overrides.
    """

    def __init__(self, *, app_name: str = "cognitionos"):
        self._store: Dict[str, ConfigEntry] = {}
        self._tenant_overrides: Dict[str, Dict[str, ConfigEntry]] = defaultdict(dict)
        self._change_listeners: List[Callable[[ConfigChangeEvent], Awaitable[None]]] = []
        self._audit_log: List[ConfigChangeEvent] = []
        self._app_name = app_name
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._reload_lock = asyncio.Lock()
        self._watchers: Dict[str, asyncio.Task] = {}

    # ── Loading ──

    def load_defaults(self, defaults: Dict[str, Any]):
        """Load default configuration values."""
        self._flatten_and_store(defaults, ConfigSource.DEFAULT)

    def load_from_env(self, prefix: str = "COGNITIONOS_"):
        """Load configuration from environment variables."""
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower().replace("__", ".")
                self._set_entry(config_key, value, ConfigSource.ENVIRONMENT)

    def load_from_file(self, file_path: str):
        """Load configuration from JSON or YAML file."""
        path = Path(file_path)
        if not path.exists():
            logger.warning("Config file not found: %s", file_path)
            return

        try:
            content = path.read_text(encoding="utf-8")
            if path.suffix in (".json",):
                data = json.loads(content)
            elif path.suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    data = yaml.safe_load(content)
                except ImportError:
                    logger.warning("PyYAML not installed, skipping YAML config")
                    return
            else:
                logger.warning("Unsupported config format: %s", path.suffix)
                return

            self._flatten_and_store(data, ConfigSource.FILE)
            logger.info("Loaded config from: %s (%d keys)", file_path, len(data))
        except Exception as exc:
            logger.error("Failed to load config from %s: %s", file_path, exc)

    def load_from_dict(self, data: Dict[str, Any],
                        source: ConfigSource = ConfigSource.RUNTIME):
        """Load configuration from a dictionary."""
        self._flatten_and_store(data, source)

    # ── Access ──

    def get(self, key: str, default: Any = None, *,
            tenant_id: Optional[str] = None) -> Any:
        """Get config value with tenant override support."""
        if tenant_id:
            tenant_entry = self._tenant_overrides.get(tenant_id, {}).get(key)
            if tenant_entry:
                return tenant_entry.value
        entry = self._store.get(key)
        return entry.value if entry else default

    def get_int(self, key: str, default: int = 0, **kw) -> int:
        v = self.get(key, default, **kw)
        try:
            return int(v)
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: float = 0.0, **kw) -> float:
        v = self.get(key, default, **kw)
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    def get_bool(self, key: str, default: bool = False, **kw) -> bool:
        v = self.get(key, default, **kw)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return bool(v)

    def get_namespace(self, prefix: str) -> ConfigNamespace:
        """Get a typed namespace accessor."""
        return ConfigNamespace(prefix, self._store)

    def get_all(self, *, include_secrets: bool = False) -> Dict[str, Any]:
        result = {}
        for key, entry in self._store.items():
            if entry.secret and not include_secrets:
                result[key] = "***"
            else:
                result[key] = entry.value
        return result

    # ── Setting ──

    async def set(self, key: str, value: Any, *,
                  source: ConfigSource = ConfigSource.RUNTIME,
                  changed_by: str = "system",
                  tenant_id: Optional[str] = None):
        """Set config value with change notification."""
        old_value = self.get(key, tenant_id=tenant_id)

        if tenant_id:
            self._tenant_overrides[tenant_id][key] = ConfigEntry(
                key=key, value=value, source=ConfigSource.TENANT,
            )
        else:
            self._set_entry(key, value, source)

        change = ConfigChangeEvent(
            key=key, old_value=old_value, new_value=value,
            source=source, changed_by=changed_by,
        )
        self._audit_log.append(change)
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-5000:]

        await self._notify_change(change)

    async def set_many(self, entries: Dict[str, Any], *,
                        source: ConfigSource = ConfigSource.RUNTIME,
                        changed_by: str = "system"):
        for key, value in entries.items():
            await self.set(key, value, source=source, changed_by=changed_by)

    # ── Tenant Overrides ──

    async def set_tenant_override(self, tenant_id: str, key: str, value: Any):
        """Set a tenant-specific config override."""
        await self.set(key, value, source=ConfigSource.TENANT, tenant_id=tenant_id)

    def get_tenant_overrides(self, tenant_id: str) -> Dict[str, Any]:
        overrides = self._tenant_overrides.get(tenant_id, {})
        return {k: v.value for k, v in overrides.items()}

    def clear_tenant_overrides(self, tenant_id: str):
        self._tenant_overrides.pop(tenant_id, None)

    # ── Validation ──

    def register_schema(self, namespace: str, schema: Dict[str, Any]):
        """Register validation schema for a namespace."""
        self._schemas[namespace] = schema

    def validate(self) -> List[str]:
        """Validate all config against registered schemas. Returns errors."""
        errors = []
        for ns, schema in self._schemas.items():
            for key, spec in schema.items():
                full_key = f"{ns}.{key}"
                entry = self._store.get(full_key)

                if spec.get("required") and not entry:
                    errors.append(f"Missing required config: {full_key}")
                    continue

                if entry and entry.validator:
                    if not entry.validate():
                        errors.append(f"Validation failed: {full_key}")

                if entry and "type" in spec:
                    expected = spec["type"]
                    if expected == "int" and not isinstance(entry.value, int):
                        errors.append(f"Type mismatch for {full_key}: expected int")
                    elif expected == "float" and not isinstance(entry.value, (int, float)):
                        errors.append(f"Type mismatch for {full_key}: expected float")
                    elif expected == "bool" and not isinstance(entry.value, bool):
                        errors.append(f"Type mismatch for {full_key}: expected bool")

        return errors

    # ── Hot Reload ──

    async def reload_from_file(self, file_path: str):
        """Hot-reload configuration from a file."""
        async with self._reload_lock:
            old_store = copy.deepcopy(self._store)
            self.load_from_file(file_path)

            # Detect and notify changes
            for key, entry in self._store.items():
                old_entry = old_store.get(key)
                if not old_entry or old_entry.value != entry.value:
                    change = ConfigChangeEvent(
                        key=key,
                        old_value=old_entry.value if old_entry else None,
                        new_value=entry.value,
                        source=ConfigSource.FILE,
                    )
                    await self._notify_change(change)

    def watch_file(self, file_path: str, *, interval: float = 30.0):
        """Start watching a config file for changes."""
        async def _watcher():
            last_mtime = 0.0
            while True:
                try:
                    await asyncio.sleep(interval)
                    path = Path(file_path)
                    if path.exists():
                        mtime = path.stat().st_mtime
                        if mtime > last_mtime:
                            last_mtime = mtime
                            await self.reload_from_file(file_path)
                            logger.info("Config hot-reloaded from: %s", file_path)
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.error("Config watcher error: %s", exc)

        self._watchers[file_path] = asyncio.create_task(_watcher())

    async def stop_watchers(self):
        for path, task in self._watchers.items():
            task.cancel()
        self._watchers.clear()

    # ── Change Listeners ──

    def on_change(self, callback: Callable[[ConfigChangeEvent], Awaitable[None]]):
        self._change_listeners.append(callback)

    async def _notify_change(self, event: ConfigChangeEvent):
        for listener in self._change_listeners:
            try:
                await listener(event)
            except Exception as exc:
                logger.error("Config change listener error: %s", exc)

    # ── Audit ──

    def get_audit_log(self, *, limit: int = 100,
                      key_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        entries = self._audit_log
        if key_filter:
            entries = [e for e in entries if key_filter in e.key]
        return [e.to_dict() for e in entries[-limit:]]

    # ── Internal ──

    def _set_entry(self, key: str, value: Any, source: ConfigSource):
        existing = self._store.get(key)
        if existing and existing.source.value > source.value:
            return  # Don't override higher-priority source
        self._store[key] = ConfigEntry(key=key, value=value, source=source)

    def _flatten_and_store(self, data: Dict[str, Any],
                            source: ConfigSource, prefix: str = ""):
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self._flatten_and_store(value, source, full_key)
            else:
                self._set_entry(full_key, value, source)

    # ── Status ──

    def get_stats(self) -> Dict[str, Any]:
        by_source: Dict[str, int] = defaultdict(int)
        for entry in self._store.values():
            by_source[entry.source.name] += 1
        return {
            "total_keys": len(self._store),
            "by_source": dict(by_source),
            "tenant_overrides": {
                tid: len(overrides) for tid, overrides in self._tenant_overrides.items()
            },
            "watchers": list(self._watchers.keys()),
            "audit_log_size": len(self._audit_log),
        }
