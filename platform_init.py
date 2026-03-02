"""
CognitionOS Platform Initializer

Master bootstrap system that initializes and wires all platform subsystems.
This is the single entry point for starting the entire CognitionOS platform.

Subsystems initialized:
1. Configuration Manager
2. Event Bus
3. Service Registry
4. Plugin Manager
5. Agent Orchestrator
6. Workflow Engine
7. Cost Governance
8. Tenant Isolation
9. Usage Analytics
10. Webhook Engine
11. Growth Automation
12. Benchmark Engine
13. Task Scheduler
14. Test Framework
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PlatformStatus(str, Enum):
    STOPPED = "stopped"
    INITIALIZING = "initializing"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"


@dataclass
class SubsystemInfo:
    name: str
    module: str
    status: str = "not_initialized"
    init_time_ms: float = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "module": self.module,
            "status": self.status,
            "init_time_ms": round(self.init_time_ms, 1),
            "error": self.error,
        }


class CognitionOSPlatform:
    """
    Master platform initializer and lifecycle manager.
    Bootstraps all subsystems in dependency order.
    """

    VERSION = "2.0.0"
    CODENAME = "Prometheus"

    def __init__(self):
        self._status = PlatformStatus.STOPPED
        self._subsystems: Dict[str, SubsystemInfo] = {}
        self._instances: Dict[str, Any] = {}
        self._started_at: Optional[float] = None
        self._init_order = [
            ("config", "core.engine.config_manager", "ConfigManager"),
            ("event_bus", "core.engine.event_bus", "EventBus"),
            ("service_registry", "core.engine.service_registry", "ServiceRegistry"),
            ("plugin_manager", "core.engine.plugin_manager", "PluginManager"),
            ("agent_orchestrator", "core.engine.agent_orchestrator", "AgentOrchestrator"),
            ("workflow_engine", "core.engine.workflow_engine", "WorkflowEngine"),
            ("cost_governance", "infrastructure.cost_governance.cost_engine", "CostGovernanceEngine"),
            ("tenant_isolation", "infrastructure.multi_tenant.tenant_isolation", "TenantIsolationLayer"),
            ("analytics", "infrastructure.analytics.usage_analytics_engine", "UsageAnalyticsEngine"),
            ("webhook_engine", "infrastructure.webhooks.webhook_engine", "WebhookDeliveryEngine"),
            ("growth", "infrastructure.growth.growth_automation", "GrowthAutomationEngine"),
            ("benchmark", "infrastructure.performance.benchmark_engine", "BenchmarkEngine"),
            ("scheduler", "infrastructure.scheduler.task_scheduler", "TaskScheduler"),
        ]

    async def initialize(self) -> Dict[str, Any]:
        """Initialize all platform subsystems in order."""
        self._status = PlatformStatus.INITIALIZING
        self._started_at = time.time()
        results = []

        logger.info("=" * 60)
        logger.info("CognitionOS Platform v%s (%s) — Starting...",
                     self.VERSION, self.CODENAME)
        logger.info("=" * 60)

        for name, module_path, class_name in self._init_order:
            info = SubsystemInfo(name=name, module=module_path)
            start = time.perf_counter()

            try:
                # Dynamic import
                module = __import__(module_path, fromlist=[class_name])
                cls = getattr(module, class_name)
                instance = cls()
                self._instances[name] = instance
                info.status = "running"
                info.init_time_ms = (time.perf_counter() - start) * 1000
                logger.info("  ✓ %s initialized (%.1fms)", name, info.init_time_ms)

            except Exception as exc:
                info.status = "failed"
                info.error = str(exc)
                info.init_time_ms = (time.perf_counter() - start) * 1000
                logger.error("  ✗ %s FAILED: %s", name, exc)

            self._subsystems[name] = info
            results.append(info.to_dict())

        failed = sum(1 for s in self._subsystems.values() if s.status == "failed")
        if failed == 0:
            self._status = PlatformStatus.RUNNING
        elif failed < len(self._subsystems):
            self._status = PlatformStatus.RUNNING  # Degraded but running
        else:
            self._status = PlatformStatus.ERROR

        total_time = (time.time() - self._started_at) * 1000

        logger.info("=" * 60)
        logger.info("Platform ready in %.0fms (%d/%d subsystems)",
                     total_time, len(self._subsystems) - failed,
                     len(self._subsystems))
        logger.info("=" * 60)

        return {
            "status": self._status.value,
            "version": self.VERSION,
            "subsystems": results,
            "total_init_ms": round(total_time, 1),
        }

    async def shutdown(self):
        """Graceful shutdown of all subsystems."""
        self._status = PlatformStatus.SHUTTING_DOWN
        logger.info("CognitionOS shutting down...")

        # Shutdown in reverse order
        for name in reversed(list(self._instances.keys())):
            instance = self._instances[name]
            if hasattr(instance, "shutdown"):
                try:
                    if asyncio.iscoroutinefunction(instance.shutdown):
                        await instance.shutdown()
                    else:
                        instance.shutdown()
                except Exception as exc:
                    logger.error("Error shutting down %s: %s", name, exc)

        self._status = PlatformStatus.STOPPED
        logger.info("CognitionOS shutdown complete.")

    def get_subsystem(self, name: str) -> Any:
        return self._instances.get(name)

    def health_check(self) -> Dict[str, Any]:
        subsystem_health = {}
        for name, info in self._subsystems.items():
            instance = self._instances.get(name)
            health = info.status
            if instance and hasattr(instance, "get_stats"):
                try:
                    stats = instance.get_stats()
                    health = "healthy" if info.status == "running" else info.status
                except Exception:
                    health = "degraded"
            subsystem_health[name] = health

        all_healthy = all(h == "healthy" or h == "running"
                           for h in subsystem_health.values())
        return {
            "status": "healthy" if all_healthy else "degraded",
            "platform_status": self._status.value,
            "version": self.VERSION,
            "uptime_seconds": round(time.time() - self._started_at, 0)
            if self._started_at else 0,
            "subsystems": subsystem_health,
        }

    def get_dashboard(self) -> Dict[str, Any]:
        """Get platform-wide dashboard data."""
        dashboard = {
            "platform": {
                "status": self._status.value,
                "version": self.VERSION,
                "codename": self.CODENAME,
                "uptime_seconds": round(
                    time.time() - self._started_at, 0
                ) if self._started_at else 0,
            },
            "subsystems": {},
        }

        for name, instance in self._instances.items():
            if hasattr(instance, "get_stats"):
                try:
                    dashboard["subsystems"][name] = instance.get_stats()
                except Exception as exc:
                    dashboard["subsystems"][name] = {"error": str(exc)}
            elif hasattr(instance, "get_metrics"):
                try:
                    dashboard["subsystems"][name] = instance.get_metrics()
                except Exception as exc:
                    dashboard["subsystems"][name] = {"error": str(exc)}

        return dashboard


# ── Global Platform Instance ──
_platform: Optional[CognitionOSPlatform] = None


def get_platform() -> CognitionOSPlatform:
    global _platform
    if not _platform:
        _platform = CognitionOSPlatform()
    return _platform


async def bootstrap() -> Dict[str, Any]:
    """Bootstrap the entire CognitionOS platform."""
    platform = get_platform()
    return await platform.initialize()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    )
    asyncio.run(bootstrap())
