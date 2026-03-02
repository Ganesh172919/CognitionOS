"""
Health Check Aggregator — CognitionOS Production Infrastructure

Aggregates health status from all subsystems into a unified health endpoint:
- Deep health checks per subsystem
- Dependency health mapping
- Readiness and liveness probes
- SLA tracking
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Dict, List, Optional

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
class HealthCheck:
    name: str
    check_fn: Callable[[], Awaitable[bool]]
    critical: bool = True
    timeout: float = 5.0
    interval: float = 30.0
    last_status: HealthStatus = HealthStatus.UNKNOWN
    last_check: float = 0
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    total_checks: int = 0
    total_failures: int = 0


@dataclass
class AggregatedHealth:
    status: HealthStatus
    checks: Dict[str, Dict[str, Any]]
    timestamp: float
    uptime_seconds: float
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "version": self.version,
            "checks": self.checks,
        }


class HealthCheckAggregator:
    """
    Central health check aggregator for all platform subsystems.
    Provides unified /health, /ready, and /live endpoints.
    """

    def __init__(self, *, version: str = "1.0.0", start_time: Optional[float] = None):
        self._checks: Dict[str, HealthCheck] = {}
        self._version = version
        self._start_time = start_time or time.time()
        self._sla_uptime_seconds = 0.0
        self._sla_total_seconds = 0.0
        self._last_overall_status = HealthStatus.UNKNOWN
        logger.info("HealthCheckAggregator initialized (version=%s)", version)

    def register(self, name: str, check_fn: Callable[[], Awaitable[bool]], *,
                 critical: bool = True, timeout: float = 5.0,
                 interval: float = 30.0):
        self._checks[name] = HealthCheck(
            name=name, check_fn=check_fn, critical=critical,
            timeout=timeout, interval=interval,
        )

    async def check_all(self) -> AggregatedHealth:
        """Run all health checks and return aggregated status."""
        results: Dict[str, Dict[str, Any]] = {}
        all_healthy = True
        has_critical_failure = False

        tasks = {
            name: asyncio.create_task(self._run_check(hc))
            for name, hc in self._checks.items()
        }

        for name, task in tasks.items():
            try:
                status = await asyncio.wait_for(task, timeout=10.0)
            except (asyncio.TimeoutError, Exception):
                status = HealthStatus.UNHEALTHY

            hc = self._checks[name]
            results[name] = {
                "status": status.value,
                "critical": hc.critical,
                "last_error": hc.last_error,
                "consecutive_failures": hc.consecutive_failures,
            }

            if status != HealthStatus.HEALTHY:
                all_healthy = False
                if hc.critical:
                    has_critical_failure = True

        overall = (HealthStatus.HEALTHY if all_healthy
                   else HealthStatus.UNHEALTHY if has_critical_failure
                   else HealthStatus.DEGRADED)

        # SLA tracking
        now = time.time()
        if self._sla_total_seconds > 0:
            if self._last_overall_status == HealthStatus.HEALTHY:
                self._sla_uptime_seconds += now - (self._start_time + self._sla_total_seconds)
        self._sla_total_seconds = now - self._start_time
        self._last_overall_status = overall

        return AggregatedHealth(
            status=overall, checks=results,
            timestamp=now, uptime_seconds=now - self._start_time,
            version=self._version,
        )

    async def _run_check(self, hc: HealthCheck) -> HealthStatus:
        try:
            result = await asyncio.wait_for(hc.check_fn(), timeout=hc.timeout)
            hc.total_checks += 1
            hc.last_check = time.time()
            if result:
                hc.last_status = HealthStatus.HEALTHY
                hc.consecutive_failures = 0
                hc.last_error = None
            else:
                hc.last_status = HealthStatus.DEGRADED
                hc.consecutive_failures += 1
                hc.total_failures += 1
        except Exception as exc:
            hc.total_checks += 1
            hc.total_failures += 1
            hc.last_check = time.time()
            hc.last_status = HealthStatus.UNHEALTHY
            hc.consecutive_failures += 1
            hc.last_error = str(exc)
        return hc.last_status

    async def liveness(self) -> Dict[str, Any]:
        """Kubernetes liveness probe — is the process alive?"""
        return {"status": "alive", "uptime": round(time.time() - self._start_time, 1)}

    async def readiness(self) -> Dict[str, Any]:
        """Kubernetes readiness probe — is the service ready to accept traffic?"""
        health = await self.check_all()
        return {
            "ready": health.status != HealthStatus.UNHEALTHY,
            "status": health.status.value,
        }

    def get_sla(self) -> Dict[str, Any]:
        total = time.time() - self._start_time
        uptime_ratio = self._sla_uptime_seconds / total if total > 0 else 1.0
        return {
            "uptime_seconds": round(self._sla_uptime_seconds, 1),
            "total_seconds": round(total, 1),
            "availability_pct": round(uptime_ratio * 100, 3),
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "registered_checks": len(self._checks),
            "overall_status": self._last_overall_status.value,
            "version": self._version,
            "sla": self.get_sla(),
            "checks": {
                name: {
                    "status": hc.last_status.value,
                    "total_checks": hc.total_checks,
                    "total_failures": hc.total_failures,
                    "failure_rate": round(
                        hc.total_failures / max(hc.total_checks, 1), 3
                    ),
                }
                for name, hc in self._checks.items()
            },
        }
