"""
Health Monitoring Domain - Phase 3

Agent health monitoring system for failure detection and recovery.
Enables real-time health tracking and automated incident management.
"""

from .entities import (
    AgentHealthStatus,
    AgentHealthIncident,
    HealthStatus,
    IncidentSeverity,
    IncidentStatus,
    ResourceMetrics,
    CostMetrics,
    TaskMetrics,
)
from .events import (
    HeartbeatReceived,
    HealthDegraded,
    HealthFailed,
    HealthRecovered,
    IncidentCreated,
    IncidentResolved,
)
from .repositories import AgentHealthRepository, HealthIncidentRepository
from .services import AgentHealthMonitoringService

__all__ = [
    # Entities
    "AgentHealthStatus",
    "AgentHealthIncident",
    "HealthStatus",
    "IncidentSeverity",
    "IncidentStatus",
    "ResourceMetrics",
    "CostMetrics",
    "TaskMetrics",
    # Events
    "HeartbeatReceived",
    "HealthDegraded",
    "HealthFailed",
    "HealthRecovered",
    "IncidentCreated",
    "IncidentResolved",
    # Repositories
    "AgentHealthRepository",
    "HealthIncidentRepository",
    # Services
    "AgentHealthMonitoringService",
]
