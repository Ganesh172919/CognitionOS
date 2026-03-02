"""
Engine Orchestrator — CognitionOS Core

Central orchestrator that manages the lifecycle of all platform subsystems.
Provides:
- Ordered startup/shutdown with dependency resolution
- Health monitoring and auto-recovery
- Graceful degradation
- Hot-reload support
- System-wide circuit breakers
- Resource governance
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Dict, List, Optional, Set

from core.engine.event_bus import EventBus, Event, EventPriority

logger = logging.getLogger(__name__)


class EngineState(str, Enum):
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    FAILED = "failed"


class SubsystemState(str, Enum):
    REGISTERED = "registered"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class SubsystemDescriptor:
    name: str
    start_fn: Callable[[], Awaitable[None]]
    stop_fn: Callable[[], Awaitable[None]]
    health_fn: Callable[[], Awaitable[bool]]
    dependencies: List[str] = field(default_factory=list)
    priority: int = 50  # Lower = starts first
    critical: bool = True  # If critical, failure = engine degraded
    state: SubsystemState = SubsystemState.REGISTERED
    start_time: float = 0
    error: Optional[str] = None
    restart_count: int = 0
    max_restarts: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    _failure_count: int = 0
    _last_failure_time: float = 0
    _state: str = "closed"  # closed, open, half_open
    _half_open_calls: int = 0

    @property
    def is_open(self) -> bool:
        if self._state == "open":
            if time.time() - self._last_failure_time > self.recovery_timeout:
                self._state = "half_open"
                self._half_open_calls = 0
                return False
            return True
        return False

    def record_success(self):
        if self._state == "half_open":
            self._half_open_calls += 1
            if self._half_open_calls >= self.half_open_max_calls:
                self._state = "closed"
                self._failure_count = 0
        elif self._state == "closed":
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = "open"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "state": self._state,
            "failure_count": self._failure_count,
            "threshold": self.failure_threshold,
        }


class ResourceGovernor:
    """Track and enforce resource limits across subsystems."""

    def __init__(self, *, max_memory_mb: int = 4096,
                 max_concurrent_tasks: int = 1000,
                 max_connections: int = 500):
        self.max_memory_mb = max_memory_mb
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_connections = max_connections
        self._active_tasks = 0
        self._active_connections = 0
        self._task_semaphore = asyncio.Semaphore(max_concurrent_tasks)

    async def acquire_task_slot(self) -> bool:
        if self._active_tasks >= self.max_concurrent_tasks:
            return False
        await self._task_semaphore.acquire()
        self._active_tasks += 1
        return True

    def release_task_slot(self):
        self._active_tasks = max(0, self._active_tasks - 1)
        self._task_semaphore.release()

    def acquire_connection(self) -> bool:
        if self._active_connections >= self.max_connections:
            return False
        self._active_connections += 1
        return True

    def release_connection(self):
        self._active_connections = max(0, self._active_connections - 1)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "active_tasks": self._active_tasks,
            "max_tasks": self.max_concurrent_tasks,
            "active_connections": self._active_connections,
            "max_connections": self.max_connections,
            "task_utilization_pct": round(self._active_tasks / max(self.max_concurrent_tasks, 1) * 100, 1),
        }


class EngineOrchestrator:
    """
    Central platform orchestrator managing subsystem lifecycle,
    health monitoring, circuit breakers, and resource governance.
    """

    def __init__(self, *, event_bus: Optional[EventBus] = None,
                 health_check_interval: float = 15.0,
                 auto_recovery: bool = True):
        self._state = EngineState.INITIALIZING
        self._subsystems: Dict[str, SubsystemDescriptor] = {}
        self._event_bus = event_bus or EventBus()
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._resource_governor = ResourceGovernor()
        self._health_check_interval = health_check_interval
        self._auto_recovery = auto_recovery
        self._health_task: Optional[asyncio.Task] = None
        self._start_time: float = 0
        self._shutdown_hooks: List[Callable[[], Awaitable[None]]] = []

    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time if self._start_time else 0

    # ── Subsystem Registration ──

    def register_subsystem(self, descriptor: SubsystemDescriptor):
        """Register a subsystem with the engine."""
        self._subsystems[descriptor.name] = descriptor
        self._circuit_breakers[descriptor.name] = CircuitBreaker(name=descriptor.name)
        logger.info("Registered subsystem: %s (priority=%d, critical=%s)",
                     descriptor.name, descriptor.priority, descriptor.critical)

    def add_shutdown_hook(self, hook: Callable[[], Awaitable[None]]):
        self._shutdown_hooks.append(hook)

    # ── Startup ──

    async def start(self):
        """Start all subsystems in dependency-resolved order."""
        self._state = EngineState.STARTING
        self._start_time = time.time()

        await self._event_bus.start(num_workers=8)
        await self._publish_event("engine.starting", {})

        # Topological sort by dependencies and priority
        start_order = self._resolve_start_order()

        for name in start_order:
            sub = self._subsystems[name]
            try:
                sub.state = SubsystemState.STARTING
                sub.start_time = time.time()
                await asyncio.wait_for(sub.start_fn(), timeout=30.0)
                sub.state = SubsystemState.HEALTHY
                logger.info("✓ Started subsystem: %s (%.1fms)",
                             name, (time.time() - sub.start_time) * 1000)
                await self._publish_event("subsystem.started", {"name": name})
            except Exception as exc:
                sub.state = SubsystemState.FAILED
                sub.error = str(exc)
                logger.error("✗ Failed to start subsystem %s: %s", name, exc)
                if sub.critical:
                    self._state = EngineState.DEGRADED
                await self._publish_event("subsystem.failed", {"name": name, "error": str(exc)})

        # Start health monitoring
        self._health_task = asyncio.create_task(self._health_monitor_loop())

        if self._state != EngineState.DEGRADED:
            self._state = EngineState.RUNNING

        elapsed = (time.time() - self._start_time) * 1000
        await self._publish_event("engine.started", {
            "state": self._state.value,
            "elapsed_ms": round(elapsed, 1),
            "subsystems": len(self._subsystems),
        })
        logger.info("Engine started in %.1fms — state: %s", elapsed, self._state.value)

    # ── Shutdown ──

    async def shutdown(self):
        """Gracefully stop all subsystems in reverse order."""
        self._state = EngineState.SHUTTING_DOWN
        await self._publish_event("engine.shutting_down", {})

        # Cancel health monitor
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Run shutdown hooks
        for hook in self._shutdown_hooks:
            try:
                await hook()
            except Exception as exc:
                logger.error("Shutdown hook error: %s", exc)

        # Stop subsystems in reverse order
        stop_order = list(reversed(self._resolve_start_order()))
        for name in stop_order:
            sub = self._subsystems[name]
            if sub.state in (SubsystemState.HEALTHY, SubsystemState.DEGRADED):
                try:
                    await asyncio.wait_for(sub.stop_fn(), timeout=10.0)
                    sub.state = SubsystemState.STOPPED
                    logger.info("Stopped subsystem: %s", name)
                except Exception as exc:
                    logger.error("Error stopping %s: %s", name, exc)

        await self._event_bus.stop()
        self._state = EngineState.STOPPED
        logger.info("Engine stopped")

    # ── Health Monitoring ──

    async def _health_monitor_loop(self):
        """Periodic health check with auto-recovery."""
        while self._state in (EngineState.RUNNING, EngineState.DEGRADED):
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._check_all_health()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Health monitor error: %s", exc)

    async def _check_all_health(self):
        all_healthy = True
        for name, sub in self._subsystems.items():
            if sub.state == SubsystemState.STOPPED:
                continue
            cb = self._circuit_breakers.get(name)
            if cb and cb.is_open:
                all_healthy = False
                continue

            try:
                healthy = await asyncio.wait_for(sub.health_fn(), timeout=5.0)
                if healthy:
                    sub.state = SubsystemState.HEALTHY
                    if cb:
                        cb.record_success()
                else:
                    sub.state = SubsystemState.DEGRADED
                    if cb:
                        cb.record_failure()
                    all_healthy = False
                    if self._auto_recovery:
                        await self._attempt_recovery(name)
            except Exception as exc:
                sub.state = SubsystemState.DEGRADED
                sub.error = str(exc)
                if cb:
                    cb.record_failure()
                all_healthy = False

        self._state = EngineState.RUNNING if all_healthy else EngineState.DEGRADED

    async def _attempt_recovery(self, name: str):
        """Attempt to restart a degraded subsystem."""
        sub = self._subsystems[name]
        if sub.restart_count >= sub.max_restarts:
            logger.warning("Subsystem %s exceeded max restarts (%d)", name, sub.max_restarts)
            return

        logger.info("Attempting recovery of subsystem: %s (attempt %d)",
                     name, sub.restart_count + 1)
        try:
            await sub.stop_fn()
            await asyncio.sleep(1)
            await sub.start_fn()
            sub.state = SubsystemState.HEALTHY
            sub.restart_count += 1
            sub.error = None
            await self._publish_event("subsystem.recovered", {
                "name": name, "restart_count": sub.restart_count,
            })
        except Exception as exc:
            sub.state = SubsystemState.FAILED
            sub.error = str(exc)
            await self._publish_event("subsystem.recovery_failed", {
                "name": name, "error": str(exc),
            })

    # ── Dependency Resolution ──

    def _resolve_start_order(self) -> List[str]:
        """Topological sort based on dependencies, breaking ties by priority."""
        visited: Set[str] = set()
        order: List[str] = []
        visiting: Set[str] = set()

        def visit(name: str):
            if name in visited:
                return
            if name in visiting:
                raise RuntimeError(f"Circular dependency detected involving: {name}")
            visiting.add(name)
            sub = self._subsystems.get(name)
            if sub:
                for dep in sub.dependencies:
                    if dep in self._subsystems:
                        visit(dep)
            visiting.discard(name)
            visited.add(name)
            order.append(name)

        # Sort by priority first, then topological
        sorted_names = sorted(self._subsystems.keys(),
                               key=lambda n: self._subsystems[n].priority)
        for name in sorted_names:
            visit(name)

        return order

    # ── Status & Metrics ──

    def get_status(self) -> Dict[str, Any]:
        return {
            "state": self._state.value,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "subsystems": {
                name: {
                    "state": sub.state.value,
                    "critical": sub.critical,
                    "restart_count": sub.restart_count,
                    "error": sub.error,
                }
                for name, sub in self._subsystems.items()
            },
            "circuit_breakers": {
                name: cb.to_dict()
                for name, cb in self._circuit_breakers.items()
            },
            "resources": self._resource_governor.snapshot(),
            "event_bus": self._event_bus.get_metrics(),
            "metrics": self.gather_metrics(),
        }

    def gather_metrics(self) -> Dict[str, Any]:
        """Produce a snapshot of platform-wide operational metrics."""
        healthy = sum(1 for s in self._subsystems.values()
                      if s.state == SubsystemState.HEALTHY)
        degraded = sum(1 for s in self._subsystems.values()
                       if s.state == SubsystemState.DEGRADED)
        failed = sum(1 for s in self._subsystems.values()
                     if s.state == SubsystemState.FAILED)
        total_restarts = sum(s.restart_count for s in self._subsystems.values())
        open_breakers = sum(1 for cb in self._circuit_breakers.values() if cb.is_open)
        resource_snap = self._resource_governor.snapshot()

        return {
            "subsystem_health": {
                "total": len(self._subsystems),
                "healthy": healthy,
                "degraded": degraded,
                "failed": failed,
                "health_ratio": round(healthy / max(len(self._subsystems), 1), 3),
            },
            "reliability": {
                "total_restarts": total_restarts,
                "open_circuit_breakers": open_breakers,
                "engine_state": self._state.value,
                "uptime_seconds": round(self.uptime_seconds, 1),
            },
            "resources": resource_snap,
        }

    # ── Helpers ──

    async def _publish_event(self, topic: str, payload: Dict[str, Any]):
        event = Event.create(topic, payload, source="engine_orchestrator",
                              priority=EventPriority.HIGH)
        await self._event_bus.publish(event)

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def resource_governor(self) -> ResourceGovernor:
        return self._resource_governor


# ── Singleton ──

_engine: Optional[EngineOrchestrator] = None


def get_engine() -> EngineOrchestrator:
    global _engine
    if not _engine:
        _engine = EngineOrchestrator()
    return _engine
