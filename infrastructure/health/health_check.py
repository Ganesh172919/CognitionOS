"""
Health Check System — CognitionOS

Comprehensive health check infrastructure:
- Component health probes (database, cache, queue, AI)
- Dependency graph checking
- Readiness vs liveness probes (K8s compatible)
- Degraded state detection
- Health history and trends
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ProbeType(str, Enum):
    LIVENESS = "liveness"
    READINESS = "readiness"
    STARTUP = "startup"


@dataclass
class HealthResult:
    component: str
    status: HealthStatus
    message: str = ""
    response_time_ms: float = 0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {"component": self.component, "status": self.status.value,
                "message": self.message, "response_time_ms": round(self.response_time_ms, 2),
                "details": self.details, "timestamp": self.timestamp}


@dataclass
class SystemHealth:
    status: HealthStatus
    components: Dict[str, HealthResult]
    uptime_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "uptime_seconds": round(self.uptime_seconds, 1),
            "timestamp": self.timestamp}


class HealthCheckSystem:
    """Manages component health probes and system-wide health."""

    def __init__(self) -> None:
        self._probes: Dict[str, Callable[..., Awaitable[HealthResult]]] = {}
        self._probe_types: Dict[str, ProbeType] = {}
        self._history: Dict[str, Deque[HealthResult]] = defaultdict(lambda: deque(maxlen=100))
        self._dependencies: Dict[str, List[str]] = {}  # component -> depends_on
        self._start_time = time.monotonic()
        self._last_check: Dict[str, HealthResult] = {}

    def register_probe(self, component: str,
                        probe: Callable[..., Awaitable[HealthResult]], *,
                        probe_type: ProbeType = ProbeType.READINESS,
                        depends_on: List[str] | None = None) -> None:
        self._probes[component] = probe
        self._probe_types[component] = probe_type
        if depends_on:
            self._dependencies[component] = depends_on

    async def check_component(self, component: str) -> HealthResult:
        probe = self._probes.get(component)
        if not probe:
            return HealthResult(component=component, status=HealthStatus.UNKNOWN,
                                 message="No probe registered")
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(probe(), timeout=10.0)
            result.response_time_ms = (time.monotonic() - start) * 1000
        except asyncio.TimeoutError:
            result = HealthResult(component=component, status=HealthStatus.UNHEALTHY,
                                   message="Probe timed out",
                                   response_time_ms=(time.monotonic() - start) * 1000)
        except Exception as e:
            result = HealthResult(component=component, status=HealthStatus.UNHEALTHY,
                                   message=str(e),
                                   response_time_ms=(time.monotonic() - start) * 1000)

        self._history[component].append(result)
        self._last_check[component] = result
        return result

    async def check_all(self) -> SystemHealth:
        tasks = {name: self.check_component(name) for name in self._probes}
        results = {}
        for name, coro in tasks.items():
            results[name] = await coro

        # Determine overall status
        statuses = [r.status for r in results.values()]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.UNKNOWN

        return SystemHealth(
            status=overall, components=results,
            uptime_seconds=time.monotonic() - self._start_time)

    async def liveness(self) -> Dict[str, Any]:
        probes = {n: p for n, p in self._probes.items()
                  if self._probe_types.get(n) == ProbeType.LIVENESS}
        if not probes:
            return {"status": "alive", "uptime": time.monotonic() - self._start_time}
        results = {}
        for name, probe in probes.items():
            results[name] = (await self.check_component(name)).to_dict()
        alive = all(r["status"] == "healthy" for r in results.values())
        return {"status": "alive" if alive else "dead", "components": results}

    async def readiness(self) -> Dict[str, Any]:
        probes = {n: p for n, p in self._probes.items()
                  if self._probe_types.get(n) == ProbeType.READINESS}
        results = {}
        for name in probes:
            results[name] = (await self.check_component(name)).to_dict()
        ready = all(r["status"] == "healthy" for r in results.values())
        return {"ready": ready, "components": results}

    def get_history(self, component: str, limit: int = 50) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in list(self._history.get(component, []))[-limit:]]

    def get_uptime(self) -> float:
        return time.monotonic() - self._start_time

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        return dict(self._dependencies)


_health: HealthCheckSystem | None = None

def get_health_system() -> HealthCheckSystem:
    global _health
    if not _health:
        _health = HealthCheckSystem()
    return _health
