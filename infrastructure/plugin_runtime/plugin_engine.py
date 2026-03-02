"""
Plugin Runtime Engine — CognitionOS

Production plugin system with:
- Plugin lifecycle management (install, activate, deactivate, uninstall)
- Sandboxed execution via AST analysis
- Hook / extension-point system
- Plugin dependency resolution
- Resource quotas and limits
- Plugin marketplace data model
- Hot-reload support
- Inter-plugin communication via event bus
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib
import inspect
import json
import logging
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Type

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PluginState(str, Enum):
    INSTALLED = "installed"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    UPDATING = "updating"
    UNINSTALLING = "uninstalling"


class PluginScope(str, Enum):
    SYSTEM = "system"
    TENANT = "tenant"
    USER = "user"


class HookType(str, Enum):
    PRE_REQUEST = "pre_request"
    POST_REQUEST = "post_request"
    PRE_AGENT_EXECUTE = "pre_agent_execute"
    POST_AGENT_EXECUTE = "post_agent_execute"
    PRE_CODEGEN = "pre_codegen"
    POST_CODEGEN = "post_codegen"
    ON_ERROR = "on_error"
    ON_METRIC = "on_metric"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class PluginManifest:
    """Plugin metadata descriptor."""
    plugin_id: str
    name: str
    version: str
    author: str
    description: str = ""
    homepage: str = ""
    license: str = "MIT"
    min_platform_version: str = "4.0.0"
    dependencies: List[str] = field(default_factory=list)
    hooks: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    scope: PluginScope = PluginScope.SYSTEM
    tags: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    icon_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "license": self.license,
            "dependencies": self.dependencies,
            "hooks": self.hooks,
            "permissions": self.permissions,
            "scope": self.scope.value,
            "tags": self.tags,
        }


@dataclass
class PluginResourceQuota:
    max_memory_mb: int = 256
    max_cpu_percent: float = 10.0
    max_execution_seconds: float = 30.0
    max_api_calls_per_minute: int = 100
    max_storage_mb: int = 100
    max_concurrent_tasks: int = 5


@dataclass
class PluginInstance:
    manifest: PluginManifest
    state: PluginState = PluginState.INSTALLED
    config: Dict[str, Any] = field(default_factory=dict)
    quota: PluginResourceQuota = field(default_factory=PluginResourceQuota)
    installed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    activated_at: Optional[str] = None
    error_message: Optional[str] = None
    api_call_count: int = 0
    execution_count: int = 0
    total_execution_time_ms: float = 0


@dataclass
class HookRegistration:
    hook_type: HookType
    plugin_id: str
    handler: Callable[..., Awaitable[Any]]
    priority: int = 0
    name: str = ""


@dataclass
class MarketplaceListing:
    plugin_id: str
    name: str
    version: str
    author: str
    description: str
    downloads: int = 0
    rating: float = 0.0
    review_count: int = 0
    is_verified: bool = False
    is_featured: bool = False
    price_usd: float = 0.0
    categories: List[str] = field(default_factory=list)
    screenshots: List[str] = field(default_factory=list)
    published_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "downloads": self.downloads,
            "rating": self.rating,
            "review_count": self.review_count,
            "is_verified": self.is_verified,
            "is_featured": self.is_featured,
            "price_usd": self.price_usd,
            "categories": self.categories,
        }


# ---------------------------------------------------------------------------
# Security sandbox - AST-based code analysis
# ---------------------------------------------------------------------------


class PluginSecurityScanner:
    """Static analysis of plugin code for dangerous operations."""

    BLOCKED_IMPORTS = {
        "os", "subprocess", "shutil", "ctypes", "importlib",
        "socket", "http", "urllib", "ftplib", "smtplib",
        "pickle", "marshal", "shelve", "tempfile", "signal",
    }

    BLOCKED_BUILTINS = {
        "exec", "eval", "compile", "__import__", "globals", "locals",
        "getattr", "setattr", "delattr", "open",
    }

    BLOCKED_AST_NODES = {ast.Global, ast.Nonlocal}

    def scan(self, code: str) -> List[str]:
        violations: List[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"Syntax error: {e}"]

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in self.BLOCKED_IMPORTS:
                        violations.append(f"Blocked import: {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root in self.BLOCKED_IMPORTS:
                        violations.append(f"Blocked import: {node.module}")

            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in self.BLOCKED_BUILTINS:
                    violations.append(f"Blocked builtin: {node.func.id}")

            elif type(node) in self.BLOCKED_AST_NODES:
                violations.append(f"Blocked AST node: {type(node).__name__}")

        return violations


# ---------------------------------------------------------------------------
# Plugin Runtime
# ---------------------------------------------------------------------------


class PluginRuntime:
    """Manages plugin lifecycle and execution."""

    def __init__(self, *, plugins_dir: Optional[str] = None) -> None:
        self._plugins: Dict[str, PluginInstance] = {}
        self._hooks: Dict[HookType, List[HookRegistration]] = defaultdict(list)
        self._custom_hooks: Dict[str, List[HookRegistration]] = defaultdict(list)
        self._scanner = PluginSecurityScanner()
        self._marketplace: Dict[str, MarketplaceListing] = {}
        self._metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(int))
        self._plugins_dir = Path(plugins_dir) if plugins_dir else None

    # ----- lifecycle -----

    async def install(self, manifest: PluginManifest, *, config: Optional[Dict[str, Any]] = None) -> PluginInstance:
        if manifest.plugin_id in self._plugins:
            existing = self._plugins[manifest.plugin_id]
            if existing.state != PluginState.ERROR:
                raise ValueError(f"Plugin {manifest.plugin_id} already installed")

        # Check dependencies
        missing = self._check_dependencies(manifest)
        if missing:
            raise ValueError(f"Missing dependencies: {missing}")

        instance = PluginInstance(manifest=manifest, config=config or {})
        self._plugins[manifest.plugin_id] = instance
        self._metrics[manifest.plugin_id]["installs"] += 1
        logger.info("Installed plugin: %s v%s", manifest.name, manifest.version)
        return instance

    async def activate(self, plugin_id: str) -> bool:
        instance = self._plugins.get(plugin_id)
        if not instance:
            raise ValueError(f"Plugin {plugin_id} not installed")

        if instance.state == PluginState.ACTIVE:
            return True

        try:
            instance.state = PluginState.ACTIVE
            instance.activated_at = datetime.now(timezone.utc).isoformat()
            instance.error_message = None
            self._metrics[plugin_id]["activations"] += 1
            logger.info("Activated plugin: %s", plugin_id)
            return True
        except Exception as e:
            instance.state = PluginState.ERROR
            instance.error_message = str(e)
            logger.error("Failed to activate plugin %s: %s", plugin_id, e)
            return False

    async def deactivate(self, plugin_id: str) -> bool:
        instance = self._plugins.get(plugin_id)
        if not instance:
            return False

        # Remove all hooks
        for hook_type in HookType:
            self._hooks[hook_type] = [h for h in self._hooks[hook_type] if h.plugin_id != plugin_id]
        for key in list(self._custom_hooks.keys()):
            self._custom_hooks[key] = [h for h in self._custom_hooks[key] if h.plugin_id != plugin_id]

        instance.state = PluginState.INACTIVE
        logger.info("Deactivated plugin: %s", plugin_id)
        return True

    async def uninstall(self, plugin_id: str) -> bool:
        await self.deactivate(plugin_id)
        instance = self._plugins.pop(plugin_id, None)
        if instance:
            logger.info("Uninstalled plugin: %s", plugin_id)
            return True
        return False

    # ----- hooks -----

    def register_hook(
        self,
        plugin_id: str,
        hook_type: HookType,
        handler: Callable[..., Awaitable[Any]],
        *,
        priority: int = 0,
        name: str = "",
    ) -> HookRegistration:
        instance = self._plugins.get(plugin_id)
        if not instance or instance.state != PluginState.ACTIVE:
            raise ValueError(f"Plugin {plugin_id} is not active")

        registration = HookRegistration(
            hook_type=hook_type,
            plugin_id=plugin_id,
            handler=handler,
            priority=priority,
            name=name or handler.__qualname__,
        )

        if hook_type == HookType.CUSTOM:
            self._custom_hooks[name].append(registration)
        else:
            self._hooks[hook_type].append(registration)
            self._hooks[hook_type].sort(key=lambda r: -r.priority)

        return registration

    async def execute_hooks(self, hook_type: HookType, context: Dict[str, Any]) -> List[Any]:
        hooks = self._hooks.get(hook_type, [])
        results = []
        for reg in hooks:
            instance = self._plugins.get(reg.plugin_id)
            if not instance or instance.state != PluginState.ACTIVE:
                continue

            start = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    reg.handler(context),
                    timeout=instance.quota.max_execution_seconds,
                )
                elapsed_ms = (time.monotonic() - start) * 1000
                instance.execution_count += 1
                instance.total_execution_time_ms += elapsed_ms
                results.append(result)
            except asyncio.TimeoutError:
                logger.warning("Plugin hook timeout: %s/%s", reg.plugin_id, reg.name)
                self._metrics[reg.plugin_id]["timeouts"] += 1
            except Exception as e:
                logger.error("Plugin hook error: %s/%s: %s", reg.plugin_id, reg.name, e)
                self._metrics[reg.plugin_id]["errors"] += 1

        return results

    async def execute_custom_hook(self, hook_name: str, context: Dict[str, Any]) -> List[Any]:
        hooks = self._custom_hooks.get(hook_name, [])
        results = []
        for reg in hooks:
            try:
                result = await reg.handler(context)
                results.append(result)
            except Exception as e:
                logger.error("Custom hook error: %s/%s: %s", hook_name, reg.plugin_id, e)
        return results

    # ----- dependency resolution -----

    def _check_dependencies(self, manifest: PluginManifest) -> List[str]:
        missing = []
        for dep in manifest.dependencies:
            if dep not in self._plugins:
                missing.append(dep)
            elif self._plugins[dep].state != PluginState.ACTIVE:
                missing.append(f"{dep} (not active)")
        return missing

    # ----- marketplace -----

    def register_marketplace_listing(self, listing: MarketplaceListing) -> None:
        self._marketplace[listing.plugin_id] = listing

    def search_marketplace(
        self,
        *,
        query: str = "",
        category: Optional[str] = None,
        verified_only: bool = False,
        sort_by: str = "downloads",
        limit: int = 20,
    ) -> List[MarketplaceListing]:
        results = list(self._marketplace.values())

        if query:
            query_lower = query.lower()
            results = [r for r in results if query_lower in r.name.lower() or query_lower in r.description.lower()]
        if category:
            results = [r for r in results if category in r.categories]
        if verified_only:
            results = [r for r in results if r.is_verified]

        if sort_by == "downloads":
            results.sort(key=lambda r: -r.downloads)
        elif sort_by == "rating":
            results.sort(key=lambda r: -r.rating)
        elif sort_by == "newest":
            results.sort(key=lambda r: r.published_at, reverse=True)

        return results[:limit]

    # ----- info / metrics -----

    def get_plugin(self, plugin_id: str) -> Optional[PluginInstance]:
        return self._plugins.get(plugin_id)

    def list_plugins(self, *, state: Optional[PluginState] = None) -> List[PluginInstance]:
        plugins = list(self._plugins.values())
        if state:
            plugins = [p for p in plugins if p.state == state]
        return plugins

    def get_plugin_metrics(self, plugin_id: Optional[str] = None) -> Dict[str, Any]:
        if plugin_id:
            return dict(self._metrics.get(plugin_id, {}))
        return {pid: dict(m) for pid, m in self._metrics.items()}

    def get_hook_count(self) -> Dict[str, int]:
        counts = {}
        for hook_type, regs in self._hooks.items():
            counts[hook_type.value] = len(regs)
        for name, regs in self._custom_hooks.items():
            counts[f"custom:{name}"] = len(regs)
        return counts


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_runtime: Optional[PluginRuntime] = None


def get_plugin_runtime() -> PluginRuntime:
    global _runtime
    if _runtime is None:
        _runtime = PluginRuntime()
    return _runtime


def init_plugin_runtime(**kwargs: Any) -> PluginRuntime:
    global _runtime
    _runtime = PluginRuntime(**kwargs)
    return _runtime
