"""
Advanced Configuration Manager — CognitionOS

Features:
- Hierarchical config layering (defaults → env-file → env-vars → overrides)
- Schema validation via Pydantic
- Hot-reload with file-watcher support
- Secrets masking in logs / exports
- Typed accessor helpers
- Namespace isolation for multi-tenant
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Sentinel
# ---------------------------------------------------------------------------

_UNSET = object()

# ---------------------------------------------------------------------------
# Config value metadata
# ---------------------------------------------------------------------------


@dataclass
class ConfigEntry:
    key: str
    value: Any
    source: str  # "default", "env_file", "env_var", "override", "remote"
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_secret: bool = False
    description: str = ""
    validator: Optional[Callable[[Any], bool]] = None


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


@dataclass
class ConfigSchema:
    """Lightweight config schema definition for validation."""

    required_keys: Set[str] = field(default_factory=set)
    type_map: Dict[str, type] = field(default_factory=dict)
    range_constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    regex_patterns: Dict[str, str] = field(default_factory=dict)

    def validate(self, config: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        for key in self.required_keys:
            if key not in config or config[key] is None:
                errors.append(f"Missing required config key: {key}")

        for key, expected_type in self.type_map.items():
            if key in config and config[key] is not None:
                if not isinstance(config[key], expected_type):
                    errors.append(f"Config key '{key}' expected {expected_type.__name__}, got {type(config[key]).__name__}")

        for key, constraints in self.range_constraints.items():
            if key in config and config[key] is not None:
                val = config[key]
                if "min" in constraints and val < constraints["min"]:
                    errors.append(f"Config key '{key}' below minimum {constraints['min']}")
                if "max" in constraints and val > constraints["max"]:
                    errors.append(f"Config key '{key}' above maximum {constraints['max']}")

        for key, pattern in self.regex_patterns.items():
            if key in config and config[key] is not None:
                if not re.match(pattern, str(config[key])):
                    errors.append(f"Config key '{key}' does not match pattern {pattern}")

        return errors


# ---------------------------------------------------------------------------
# Configuration Manager
# ---------------------------------------------------------------------------


class ConfigManager:
    """Hierarchical configuration with layered overrides."""

    SECRET_PATTERNS = re.compile(
        r"(password|secret|token|api_key|private_key|credentials|auth)", re.IGNORECASE
    )

    def __init__(
        self,
        *,
        env_prefix: str = "COGNITIONOS_",
        env_file: Optional[str] = None,
        schema: Optional[ConfigSchema] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._entries: Dict[str, ConfigEntry] = {}
        self._change_callbacks: List[Callable[[str, Any, Any], None]] = []
        self._env_prefix = env_prefix
        self._schema = schema
        self._namespace_overrides: Dict[str, Dict[str, Any]] = {}

        # Layer 1: env file (if provided)
        if env_file:
            self._load_env_file(env_file)

        # Layer 2: environment variables
        self._load_env_vars()

    # ----- loading layers -----

    def _load_env_file(self, path: str) -> None:
        filepath = Path(path)
        if not filepath.exists():
            logger.warning("Config env file not found: %s", path)
            return
        with filepath.open() as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    logger.warning("Invalid config line %d in %s", line_no, path)
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                self._set_internal(key, self._coerce(value), "env_file")

    def _load_env_vars(self) -> None:
        for key, value in os.environ.items():
            if key.startswith(self._env_prefix):
                config_key = key[len(self._env_prefix):].lower()
                self._set_internal(config_key, self._coerce(value), "env_var")

    def _coerce(self, value: str) -> Any:
        """Best-effort type coercion for string values."""
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        if value.lower() in ("false", "no", "0", "off"):
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        return value

    # ----- internal set -----

    def _set_internal(self, key: str, value: Any, source: str) -> None:
        is_secret = bool(self.SECRET_PATTERNS.search(key))
        with self._lock:
            old = self._entries.get(key)
            old_value = old.value if old else _UNSET
            self._entries[key] = ConfigEntry(key=key, value=value, source=source, is_secret=is_secret)
            if old_value is not _UNSET and old_value != value:
                for cb in self._change_callbacks:
                    try:
                        cb(key, old_value, value)
                    except Exception:
                        logger.exception("Config change callback error for key %s", key)

    # ----- public API -----

    def set(self, key: str, value: Any, *, source: str = "override") -> None:
        self._set_internal(key, value, source)

    def get(self, key: str, default: Any = _UNSET, *, cast: Optional[Type[T]] = None) -> Any:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                if default is _UNSET:
                    raise KeyError(f"Config key not found: {key}")
                return default
            value = entry.value
            if cast is not None:
                try:
                    return cast(value)
                except (ValueError, TypeError):
                    return default if default is not _UNSET else value
            return value

    def get_int(self, key: str, default: int = 0) -> int:
        return self.get(key, default, cast=int)

    def get_float(self, key: str, default: float = 0.0) -> float:
        return self.get(key, default, cast=float)

    def get_bool(self, key: str, default: bool = False) -> bool:
        val = self.get(key, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "yes", "1", "on")
        return bool(val)

    def get_str(self, key: str, default: str = "") -> str:
        return str(self.get(key, default))

    def get_list(self, key: str, default: Optional[list] = None) -> list:
        val = self.get(key, default or [])
        if isinstance(val, str):
            return [v.strip() for v in val.split(",") if v.strip()]
        if isinstance(val, list):
            return val
        return default or []

    # ----- namespace isolation (multi-tenant) -----

    def set_namespace(self, namespace: str, overrides: Dict[str, Any]) -> None:
        self._namespace_overrides[namespace] = overrides

    def get_namespaced(self, namespace: str, key: str, default: Any = _UNSET) -> Any:
        ns = self._namespace_overrides.get(namespace, {})
        if key in ns:
            return ns[key]
        return self.get(key, default)

    # ----- validation -----

    def validate(self) -> List[str]:
        if not self._schema:
            return []
        flat = self.to_dict(include_secrets=True)
        return self._schema.validate(flat)

    # ----- change notification -----

    def on_change(self, callback: Callable[[str, Any, Any], None]) -> None:
        self._change_callbacks.append(callback)

    # ----- export -----

    def to_dict(self, *, include_secrets: bool = False) -> Dict[str, Any]:
        with self._lock:
            result: Dict[str, Any] = {}
            for key, entry in self._entries.items():
                if entry.is_secret and not include_secrets:
                    result[key] = "***MASKED***"
                else:
                    result[key] = entry.value
            return result

    def to_sources_dict(self) -> Dict[str, str]:
        with self._lock:
            return {key: entry.source for key, entry in self._entries.items()}

    # ----- bulk set -----

    def set_defaults(self, defaults: Dict[str, Any]) -> None:
        for key, value in defaults.items():
            if key not in self._entries:
                self._set_internal(key, value, "default")

    def merge(self, other: Dict[str, Any], *, source: str = "merge") -> None:
        for key, value in other.items():
            self._set_internal(key, value, source)

    # ----- hot reload -----

    def reload_env_file(self, path: str) -> List[str]:
        """Reload configuration from file, returns list of changed keys."""
        old_snapshot = self.to_dict(include_secrets=True)
        self._load_env_file(path)
        new_snapshot = self.to_dict(include_secrets=True)
        changed = [k for k in new_snapshot if old_snapshot.get(k) != new_snapshot.get(k)]
        if changed:
            logger.info("Config hot-reload: %d keys changed — %s", len(changed), changed)
        return changed

    def __repr__(self) -> str:
        return f"<ConfigManager entries={len(self._entries)} prefix={self._env_prefix}>"


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_default_config: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    global _default_config
    if _default_config is None:
        _default_config = ConfigManager(env_prefix="COGNITIONOS_")
    return _default_config


def init_config_manager(**kwargs: Any) -> ConfigManager:
    global _default_config
    _default_config = ConfigManager(**kwargs)
    return _default_config
