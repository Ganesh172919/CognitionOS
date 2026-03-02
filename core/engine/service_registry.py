"""
Service Registry & Discovery — CognitionOS Core Engine

Dynamic service registry with:
- Service registration and deregistration
- Health-aware service discovery
- Load balancing strategies
- Service versioning
- Dependency mapping
- Circuit breaker integration
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ServiceHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class LoadBalanceStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"


@dataclass
class ServiceEndpoint:
    host: str
    port: int
    protocol: str = "http"
    weight: int = 100
    active_connections: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}"


@dataclass
class ServiceDescriptor:
    service_id: str
    name: str
    version: str
    description: str = ""
    endpoints: List[ServiceEndpoint] = field(default_factory=list)
    health: ServiceHealth = ServiceHealth.UNKNOWN
    dependencies: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    health_check_fn: Optional[Callable[[], Awaitable[bool]]] = None
    ttl_seconds: float = 60.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return time.time() - self.last_heartbeat > self.ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_id": self.service_id, "name": self.name,
            "version": self.version, "health": self.health.value,
            "endpoints": [{"url": e.url, "weight": e.weight} for e in self.endpoints],
            "dependencies": self.dependencies,
            "capabilities": self.capabilities,
            "tags": self.tags,
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
            "is_expired": self.is_expired,
        }


class ServiceRegistry:
    """
    Service registry with health-aware discovery, load balancing,
    and automatic deregistration of expired services.
    """

    def __init__(self, *, heartbeat_interval: float = 15.0,
                 cleanup_interval: float = 30.0):
        self._services: Dict[str, ServiceDescriptor] = {}
        self._by_name: Dict[str, List[str]] = defaultdict(list)
        self._by_capability: Dict[str, List[str]] = defaultdict(list)
        self._round_robin_idx: Dict[str, int] = defaultdict(int)
        self._heartbeat_interval = heartbeat_interval
        self._cleanup_interval = cleanup_interval
        self._listeners: List[Callable[[str, str, ServiceDescriptor], Awaitable[None]]] = []
        self._monitor_task: Optional[asyncio.Task] = None

    # ── Registration ──

    def register(self, descriptor: ServiceDescriptor) -> str:
        """Register a service. Returns service_id."""
        sid = descriptor.service_id
        self._services[sid] = descriptor
        if sid not in self._by_name[descriptor.name]:
            self._by_name[descriptor.name].append(sid)
        for cap in descriptor.capabilities:
            if sid not in self._by_capability[cap]:
                self._by_capability[cap].append(sid)

        logger.info("Registered service: %s/%s (id=%s)",
                     descriptor.name, descriptor.version, sid)
        asyncio.get_event_loop().create_task(
            self._notify_listeners("registered", sid, descriptor)
        )
        return sid

    def deregister(self, service_id: str) -> bool:
        svc = self._services.pop(service_id, None)
        if not svc:
            return False
        if service_id in self._by_name.get(svc.name, []):
            self._by_name[svc.name].remove(service_id)
        for cap in svc.capabilities:
            if service_id in self._by_capability.get(cap, []):
                self._by_capability[cap].remove(service_id)
        logger.info("Deregistered service: %s", service_id)
        return True

    def heartbeat(self, service_id: str) -> bool:
        svc = self._services.get(service_id)
        if not svc:
            return False
        svc.last_heartbeat = time.time()
        return True

    # ── Discovery ──

    def discover(self, name: str, *,
                 version: Optional[str] = None,
                 healthy_only: bool = True,
                 strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
                 ) -> Optional[ServiceDescriptor]:
        """Discover a service instance by name with load balancing."""
        candidates = self._get_candidates(name, version=version,
                                            healthy_only=healthy_only)
        if not candidates:
            return None

        if strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin(name, candidates)
        elif strategy == LoadBalanceStrategy.RANDOM:
            return random.choice(candidates)
        elif strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._least_connections(candidates)
        elif strategy == LoadBalanceStrategy.WEIGHTED:
            return self._weighted(candidates)
        return candidates[0]

    def discover_all(self, name: str, *,
                     healthy_only: bool = True) -> List[ServiceDescriptor]:
        """Discover all instances of a service."""
        return self._get_candidates(name, healthy_only=healthy_only)

    def discover_by_capability(self, capability: str) -> List[ServiceDescriptor]:
        """Find services providing a specific capability."""
        sids = self._by_capability.get(capability, [])
        return [self._services[sid] for sid in sids
                if sid in self._services and
                self._services[sid].health != ServiceHealth.UNHEALTHY]

    # ── Health ──

    async def check_health(self, service_id: str) -> ServiceHealth:
        svc = self._services.get(service_id)
        if not svc:
            return ServiceHealth.UNKNOWN
        if svc.health_check_fn:
            try:
                healthy = await asyncio.wait_for(svc.health_check_fn(), timeout=5.0)
                svc.health = ServiceHealth.HEALTHY if healthy else ServiceHealth.UNHEALTHY
            except Exception:
                svc.health = ServiceHealth.UNHEALTHY
        elif svc.is_expired:
            svc.health = ServiceHealth.UNHEALTHY
        return svc.health

    async def check_all_health(self) -> Dict[str, str]:
        results = {}
        for sid in list(self._services.keys()):
            results[sid] = (await self.check_health(sid)).value
        return results

    # ── Monitoring ──

    async def start_monitoring(self):
        if self._monitor_task:
            return
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self):
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    async def _monitor_loop(self):
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired()
                await self.check_all_health()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Service registry monitor error: %s", exc)

    async def _cleanup_expired(self):
        expired = [sid for sid, svc in self._services.items() if svc.is_expired]
        for sid in expired:
            svc = self._services[sid]
            logger.warning("Service expired: %s/%s", svc.name, sid)
            await self._notify_listeners("expired", sid, svc)
            self.deregister(sid)

    # ── Listeners ──

    def add_listener(self, callback: Callable[[str, str, ServiceDescriptor], Awaitable[None]]):
        self._listeners.append(callback)

    async def _notify_listeners(self, event: str, service_id: str,
                                  descriptor: ServiceDescriptor):
        for listener in self._listeners:
            try:
                await listener(event, service_id, descriptor)
            except Exception as exc:
                logger.error("Listener error: %s", exc)

    # ── Load Balancing ──

    def _get_candidates(self, name: str, *,
                         version: Optional[str] = None,
                         healthy_only: bool = True) -> List[ServiceDescriptor]:
        sids = self._by_name.get(name, [])
        candidates = []
        for sid in sids:
            svc = self._services.get(sid)
            if not svc:
                continue
            if healthy_only and svc.health == ServiceHealth.UNHEALTHY:
                continue
            if version and svc.version != version:
                continue
            candidates.append(svc)
        return candidates

    def _round_robin(self, name: str,
                      candidates: List[ServiceDescriptor]) -> ServiceDescriptor:
        idx = self._round_robin_idx[name] % len(candidates)
        self._round_robin_idx[name] = idx + 1
        return candidates[idx]

    def _least_connections(self,
                            candidates: List[ServiceDescriptor]) -> ServiceDescriptor:
        return min(candidates, key=lambda s: sum(
            e.active_connections for e in s.endpoints
        ))

    def _weighted(self, candidates: List[ServiceDescriptor]) -> ServiceDescriptor:
        weights = []
        for svc in candidates:
            w = sum(e.weight for e in svc.endpoints) if svc.endpoints else 1
            weights.append(w)
        return random.choices(candidates, weights=weights, k=1)[0]

    # ── Status ──

    def get_status(self) -> Dict[str, Any]:
        return {
            "total_services": len(self._services),
            "by_health": {
                h.value: sum(1 for s in self._services.values() if s.health == h)
                for h in ServiceHealth
            },
            "services": [s.to_dict() for s in self._services.values()],
        }

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        graph = {}
        for sid, svc in self._services.items():
            graph[svc.name] = svc.dependencies
        return graph
