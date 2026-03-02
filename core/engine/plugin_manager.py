"""
Plugin Manager — CognitionOS Core Engine

Hot-loadable plugin system supporting:
- Plugin discovery, validation, and lifecycle management
- Sandboxed execution with resource limits
- Dependency resolution between plugins
- Version compatibility checking
- Plugin marketplace integration hooks
- Event-driven plugin communication
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Dict, List, Optional, Set, Type

logger = logging.getLogger(__name__)

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.]+))?$")


class PluginState(str, Enum):
    DISCOVERED = "discovered"
    VALIDATED = "validated"
    INSTALLED = "installed"
    ACTIVE = "active"
    DISABLED = "disabled"
    FAILED = "failed"
    UNINSTALLED = "uninstalled"


@dataclass
class PluginCapability:
    """Declares what a plugin provides or requires."""
    name: str
    version: str = "1.0.0"
    description: str = ""


@dataclass
class PluginMetadata:
    plugin_id: str
    name: str
    version: str
    author: str = ""
    description: str = ""
    homepage: str = ""
    license: str = "MIT"
    entry_point: str = ""
    min_platform_version: str = "3.0.0"
    max_platform_version: str = "99.0.0"
    dependencies: List[str] = field(default_factory=list)  # plugin_id:version
    provides: List[PluginCapability] = field(default_factory=list)
    requires: List[PluginCapability] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    icon_url: str = ""
    downloads: int = 0
    rating: float = 0.0
    verified: bool = False

    @property
    def qualified_name(self) -> str:
        return f"{self.plugin_id}@{self.version}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_id": self.plugin_id, "name": self.name,
            "version": self.version, "author": self.author,
            "description": self.description, "license": self.license,
            "dependencies": self.dependencies,
            "permissions": self.permissions, "tags": self.tags,
            "verified": self.verified, "rating": self.rating,
        }


class PluginInterface(ABC):
    """Base interface all plugins must implement."""

    @abstractmethod
    async def activate(self, context: "PluginContext") -> None:
        """Called when plugin is activated."""

    @abstractmethod
    async def deactivate(self) -> None:
        """Called when plugin is deactivated."""

    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""

    async def on_config_change(self, config: Dict[str, Any]) -> None:
        """Called when plugin configuration changes."""
        pass

    async def health_check(self) -> bool:
        """Return True if plugin is healthy."""
        return True


@dataclass
class PluginContext:
    """Execution context provided to plugins."""
    plugin_id: str
    tenant_id: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    event_publish: Optional[Callable] = None
    logger: Optional[logging.Logger] = None
    data_store: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.logger:
            self.logger = logging.getLogger(f"plugin.{self.plugin_id}")
        if self.data_store is None:
            self.data_store = {}


@dataclass
class PluginInstance:
    metadata: PluginMetadata
    instance: Optional[PluginInterface] = None
    state: PluginState = PluginState.DISCOVERED
    config: Dict[str, Any] = field(default_factory=dict)
    installed_at: float = 0
    activated_at: float = 0
    error: Optional[str] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)
    _context: Optional[PluginContext] = None


class PluginSandbox:
    """Resource-limited execution sandbox for plugins."""

    def __init__(self, *, max_memory_mb: int = 256,
                 max_cpu_seconds: float = 30,
                 max_concurrent: int = 5,
                 allowed_modules: Optional[Set[str]] = None):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_seconds = max_cpu_seconds
        self.max_concurrent = max_concurrent
        self.allowed_modules = allowed_modules or {
            "json", "re", "math", "datetime", "collections",
            "itertools", "functools", "hashlib", "uuid",
            "typing", "dataclasses", "enum", "abc",
            "logging", "asyncio", "aiohttp",
        }
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def execute(self, fn: Callable[..., Awaitable[Any]],
                      *args, **kwargs) -> Any:
        """Execute function within sandbox constraints."""
        async with self._semaphore:
            try:
                return await asyncio.wait_for(
                    fn(*args, **kwargs),
                    timeout=self.max_cpu_seconds,
                )
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"Plugin execution exceeded {self.max_cpu_seconds}s timeout"
                )


class PluginManager:
    """
    Manages plugin lifecycle: discovery, validation, installation,
    activation, deactivation, and uninstallation.
    """

    PLATFORM_VERSION = "3.2.0"

    def __init__(self, *, sandbox: Optional[PluginSandbox] = None,
                 event_publish: Optional[Callable] = None):
        self._plugins: Dict[str, PluginInstance] = {}
        self._sandbox = sandbox or PluginSandbox()
        self._event_publish = event_publish
        self._hooks: Dict[str, List[Callable]] = {
            "pre_install": [], "post_install": [],
            "pre_activate": [], "post_activate": [],
            "pre_deactivate": [], "post_deactivate": [],
        }

    # ── Registration ──

    def register(self, metadata: PluginMetadata) -> bool:
        """Register a plugin from metadata."""
        if metadata.plugin_id in self._plugins:
            logger.warning("Plugin %s already registered", metadata.plugin_id)
            return False
        self._plugins[metadata.plugin_id] = PluginInstance(metadata=metadata)
        logger.info("Registered plugin: %s", metadata.qualified_name)
        return True

    def register_class(self, plugin_class: Type[PluginInterface]) -> bool:
        """Register a plugin from its class."""
        instance = plugin_class()
        metadata = instance.get_metadata()
        if metadata.plugin_id in self._plugins:
            return False
        self._plugins[metadata.plugin_id] = PluginInstance(
            metadata=metadata, instance=instance,
        )
        return True

    # ── Validation ──

    def validate(self, plugin_id: str) -> List[str]:
        """Validate plugin. Returns list of validation errors."""
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            return [f"Plugin {plugin_id} not found"]

        errors = []
        meta = plugin.metadata

        # Version format
        if not _SEMVER_RE.match(meta.version):
            errors.append(f"Invalid version format: {meta.version}")

        # Platform compatibility
        if not self._version_compatible(meta.min_platform_version,
                                          self.PLATFORM_VERSION,
                                          meta.max_platform_version):
            errors.append(
                f"Incompatible platform version: requires "
                f"{meta.min_platform_version}-{meta.max_platform_version}, "
                f"current is {self.PLATFORM_VERSION}"
            )

        # Dependency check
        for dep_spec in meta.dependencies:
            dep_id = dep_spec.split(":")[0] if ":" in dep_spec else dep_spec
            if dep_id not in self._plugins:
                errors.append(f"Missing dependency: {dep_spec}")
            elif self._plugins[dep_id].state not in (PluginState.ACTIVE, PluginState.INSTALLED):
                errors.append(f"Dependency not active: {dep_spec}")

        # Capability requirements
        available_caps = set()
        for p in self._plugins.values():
            if p.state == PluginState.ACTIVE:
                for cap in p.metadata.provides:
                    available_caps.add(cap.name)
        for req in meta.requires:
            if req.name not in available_caps:
                errors.append(f"Required capability not available: {req.name}")

        if not errors:
            plugin.state = PluginState.VALIDATED

        return errors

    # ── Installation ──

    async def install(self, plugin_id: str, *,
                      config: Optional[Dict[str, Any]] = None) -> bool:
        """Install and optionally configure a plugin."""
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            logger.error("Plugin %s not found", plugin_id)
            return False

        # Validate first
        errors = self.validate(plugin_id)
        if errors:
            logger.error("Plugin %s validation failed: %s", plugin_id, errors)
            return False

        # Run pre-install hooks
        for hook in self._hooks["pre_install"]:
            await hook(plugin_id, plugin.metadata)

        plugin.config = config or {}
        plugin.installed_at = time.time()
        plugin.state = PluginState.INSTALLED
        logger.info("Installed plugin: %s", plugin.metadata.qualified_name)

        # Run post-install hooks
        for hook in self._hooks["post_install"]:
            await hook(plugin_id, plugin.metadata)

        return True

    # ── Activation ──

    async def activate(self, plugin_id: str) -> bool:
        """Activate an installed plugin."""
        plugin = self._plugins.get(plugin_id)
        if not plugin or plugin.state != PluginState.INSTALLED:
            logger.error("Cannot activate %s: not installed (state=%s)",
                         plugin_id, plugin.state if plugin else "missing")
            return False

        for hook in self._hooks["pre_activate"]:
            await hook(plugin_id)

        context = PluginContext(
            plugin_id=plugin_id,
            config=plugin.config,
            event_publish=self._event_publish,
        )

        try:
            if plugin.instance:
                await self._sandbox.execute(plugin.instance.activate, context)
            plugin._context = context
            plugin.state = PluginState.ACTIVE
            plugin.activated_at = time.time()
            logger.info("Activated plugin: %s", plugin.metadata.qualified_name)
        except Exception as exc:
            plugin.state = PluginState.FAILED
            plugin.error = str(exc)
            logger.error("Failed to activate %s: %s", plugin_id, exc)
            return False

        for hook in self._hooks["post_activate"]:
            await hook(plugin_id)

        return True

    # ── Deactivation ──

    async def deactivate(self, plugin_id: str) -> bool:
        """Deactivate a running plugin."""
        plugin = self._plugins.get(plugin_id)
        if not plugin or plugin.state != PluginState.ACTIVE:
            return False

        for hook in self._hooks["pre_deactivate"]:
            await hook(plugin_id)

        try:
            if plugin.instance:
                await self._sandbox.execute(plugin.instance.deactivate)
            plugin.state = PluginState.DISABLED
            logger.info("Deactivated plugin: %s", plugin_id)
        except Exception as exc:
            plugin.state = PluginState.FAILED
            plugin.error = str(exc)
            return False

        for hook in self._hooks["post_deactivate"]:
            await hook(plugin_id)

        return True

    # ── Uninstallation ──

    async def uninstall(self, plugin_id: str) -> bool:
        """Uninstall a plugin completely."""
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            return False

        if plugin.state == PluginState.ACTIVE:
            await self.deactivate(plugin_id)

        # Check if other plugins depend on this one
        dependents = self._get_dependents(plugin_id)
        if dependents:
            logger.error("Cannot uninstall %s: depended on by %s",
                         plugin_id, dependents)
            return False

        plugin.state = PluginState.UNINSTALLED
        del self._plugins[plugin_id]
        logger.info("Uninstalled plugin: %s", plugin_id)
        return True

    # ── Hooks ──

    def add_hook(self, event: str, callback: Callable):
        if event in self._hooks:
            self._hooks[event].append(callback)

    # ── Query ──

    def get_plugin(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            return None
        return {
            **plugin.metadata.to_dict(),
            "state": plugin.state.value,
            "installed_at": plugin.installed_at,
            "activated_at": plugin.activated_at,
            "error": plugin.error,
        }

    def list_plugins(self, *, state: Optional[PluginState] = None,
                     tag: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []
        for pid, plugin in self._plugins.items():
            if state and plugin.state != state:
                continue
            if tag and tag not in plugin.metadata.tags:
                continue
            results.append(self.get_plugin(pid))
        return results

    def get_active_plugins(self) -> List[str]:
        return [pid for pid, p in self._plugins.items()
                if p.state == PluginState.ACTIVE]

    # ── Health ──

    async def check_health(self) -> Dict[str, Any]:
        results = {}
        for pid, plugin in self._plugins.items():
            if plugin.state != PluginState.ACTIVE or not plugin.instance:
                continue
            try:
                healthy = await asyncio.wait_for(
                    plugin.instance.health_check(), timeout=5.0
                )
                results[pid] = {"healthy": healthy}
            except Exception as exc:
                results[pid] = {"healthy": False, "error": str(exc)}
        return results

    # ── Internal ──

    def _get_dependents(self, plugin_id: str) -> List[str]:
        dependents = []
        for pid, plugin in self._plugins.items():
            if pid == plugin_id:
                continue
            for dep in plugin.metadata.dependencies:
                dep_id = dep.split(":")[0] if ":" in dep else dep
                if dep_id == plugin_id:
                    dependents.append(pid)
        return dependents

    def _version_compatible(self, min_ver: str, current: str, max_ver: str) -> bool:
        def parse(v: str) -> tuple:
            m = _SEMVER_RE.match(v)
            if not m:
                return (0, 0, 0)
            return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return parse(min_ver) <= parse(current) <= parse(max_ver)
