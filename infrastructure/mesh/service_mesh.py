"""
Service Mesh — CognitionOS

Internal service discovery and communication:
- Service registration and discovery
- Load balancing (round-robin, weighted)
- Service health tracking
- Request routing with version affinity
- Traffic splitting for canary deploys
- Service dependency mapping
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"


class LoadBalanceStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"


@dataclass
class ServiceInstance:
    instance_id: str = field(default_factory=lambda: str(uuid4()))
    service_name: str = ""
    version: str = "1.0.0"
    host: str = "localhost"
    port: int = 8000
    status: ServiceStatus = ServiceStatus.HEALTHY
    weight: int = 100
    metadata: Dict[str, str] = field(default_factory=dict)
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_heartbeat: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    active_connections: int = 0
    total_requests: int = 0
    error_count: int = 0

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id, "service": self.service_name,
            "version": self.version, "url": self.url,
            "status": self.status.value, "weight": self.weight,
            "active_connections": self.active_connections,
            "total_requests": self.total_requests}


@dataclass
class TrafficRule:
    rule_id: str = field(default_factory=lambda: str(uuid4()))
    service_name: str = ""
    version_weights: Dict[str, int] = field(default_factory=dict)  # version -> weight%
    header_matches: Dict[str, str] = field(default_factory=dict)
    is_active: bool = True


class ServiceMesh:
    """Service discovery, load balancing, and traffic management."""

    def __init__(self, *, strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN) -> None:
        self._instances: Dict[str, List[ServiceInstance]] = defaultdict(list)
        self._strategy = strategy
        self._rr_index: Dict[str, int] = defaultdict(int)
        self._traffic_rules: Dict[str, TrafficRule] = {}
        self._metrics: Dict[str, int] = defaultdict(int)

    # ---- registration ----
    def register(self, instance: ServiceInstance) -> str:
        self._instances[instance.service_name].append(instance)
        self._metrics["registrations"] += 1
        logger.info("Service registered: %s v%s @ %s",
                     instance.service_name, instance.version, instance.url)
        return instance.instance_id

    def deregister(self, service_name: str, instance_id: str) -> bool:
        instances = self._instances.get(service_name, [])
        for i, inst in enumerate(instances):
            if inst.instance_id == instance_id:
                instances.pop(i)
                self._metrics["deregistrations"] += 1
                return True
        return False

    def heartbeat(self, service_name: str, instance_id: str) -> bool:
        for inst in self._instances.get(service_name, []):
            if inst.instance_id == instance_id:
                inst.last_heartbeat = datetime.now(timezone.utc).isoformat()
                return True
        return False

    # ---- discovery ----
    def discover(self, service_name: str, *, version: str = "",
                  healthy_only: bool = True) -> List[ServiceInstance]:
        instances = self._instances.get(service_name, [])
        if healthy_only:
            instances = [i for i in instances
                         if i.status in (ServiceStatus.HEALTHY, ServiceStatus.DEGRADED)]
        if version:
            instances = [i for i in instances if i.version == version]
        return instances

    def resolve(self, service_name: str, *, version: str = "",
                headers: Dict[str, str] | None = None) -> Optional[ServiceInstance]:
        # Check traffic rules first
        rule = self._traffic_rules.get(service_name)
        if rule and rule.is_active and rule.version_weights:
            version = self._select_version_by_weight(rule)

        instances = self.discover(service_name, version=version)
        if not instances:
            return None

        self._metrics["resolutions"] += 1
        return self._load_balance(service_name, instances)

    def _load_balance(self, service_name: str,
                       instances: List[ServiceInstance]) -> ServiceInstance:
        if self._strategy == LoadBalanceStrategy.ROUND_ROBIN:
            idx = self._rr_index[service_name] % len(instances)
            self._rr_index[service_name] = idx + 1
            return instances[idx]

        elif self._strategy == LoadBalanceStrategy.WEIGHTED:
            total_weight = sum(i.weight for i in instances)
            r = random.randint(0, total_weight - 1)
            cumulative = 0
            for inst in instances:
                cumulative += inst.weight
                if r < cumulative:
                    return inst
            return instances[-1]

        elif self._strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return min(instances, key=lambda i: i.active_connections)

        else:
            return random.choice(instances)

    def _select_version_by_weight(self, rule: TrafficRule) -> str:
        total = sum(rule.version_weights.values())
        r = random.randint(0, total - 1)
        cumulative = 0
        for version, weight in rule.version_weights.items():
            cumulative += weight
            if r < cumulative:
                return version
        return list(rule.version_weights.keys())[-1]

    # ---- traffic rules ----
    def set_traffic_rule(self, rule: TrafficRule) -> None:
        self._traffic_rules[rule.service_name] = rule

    def remove_traffic_rule(self, service_name: str) -> bool:
        return self._traffic_rules.pop(service_name, None) is not None

    # ---- status ----
    def update_status(self, service_name: str, instance_id: str,
                       status: ServiceStatus) -> bool:
        for inst in self._instances.get(service_name, []):
            if inst.instance_id == instance_id:
                inst.status = status
                return True
        return False

    def list_services(self) -> List[Dict[str, Any]]:
        return [
            {"service": name, "instances": len(insts),
             "healthy": sum(1 for i in insts if i.status == ServiceStatus.HEALTHY)}
            for name, insts in self._instances.items()]

    def get_service_map(self) -> Dict[str, List[Dict[str, Any]]]:
        return {name: [i.to_dict() for i in insts]
                for name, insts in self._instances.items()}

    def get_metrics(self) -> Dict[str, Any]:
        total_instances = sum(len(v) for v in self._instances.values())
        return {**dict(self._metrics), "total_services": len(self._instances),
                "total_instances": total_instances,
                "traffic_rules": len(self._traffic_rules)}


_mesh: ServiceMesh | None = None

def get_service_mesh() -> ServiceMesh:
    global _mesh
    if not _mesh:
        _mesh = ServiceMesh()
    return _mesh
