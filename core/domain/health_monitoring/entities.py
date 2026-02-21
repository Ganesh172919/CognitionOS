"""
Health Monitoring Domain - Entities

Pure domain entities for health monitoring functionality.
NO external dependencies except Python stdlib.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4


class HealthStatus(str, Enum):
    """Agent health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


class IncidentSeverity(str, Enum):
    """Health incident severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(str, Enum):
    """Health incident status"""
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    IGNORED = "ignored"


# ==================== Value Objects ====================

@dataclass(frozen=True)
class ResourceMetrics:
    """Resource usage metrics snapshot"""
    memory_usage_mb: float
    cpu_usage_percent: float
    working_memory_count: int = 0
    episodic_memory_count: int = 0
    disk_usage_percent: float = 0.0
    network_rx_mb: float = 0.0
    network_tx_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "working_memory_count": self.working_memory_count,
            "episodic_memory_count": self.episodic_memory_count,
            "disk_usage_percent": self.disk_usage_percent,
            "network_rx_mb": self.network_rx_mb,
            "network_tx_mb": self.network_tx_mb,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceMetrics":
        """Create from dictionary"""
        return cls(
            memory_usage_mb=data.get("memory_usage_mb", 0.0),
            cpu_usage_percent=data.get("cpu_usage_percent", 0.0),
            working_memory_count=data.get("working_memory_count", 0),
            episodic_memory_count=data.get("episodic_memory_count", 0),
            disk_usage_percent=data.get("disk_usage_percent", 0.0),
            network_rx_mb=data.get("network_rx_mb", 0.0),
            network_tx_mb=data.get("network_tx_mb", 0.0),
        )


@dataclass(frozen=True)
class CostMetrics:
    """Cost tracking metrics snapshot"""
    cost_consumed: float
    budget_remaining: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "cost_consumed": self.cost_consumed,
            "budget_remaining": self.budget_remaining,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostMetrics":
        """Create from dictionary"""
        return cls(
            cost_consumed=data.get("cost_consumed", 0.0),
            budget_remaining=data.get("budget_remaining", 0.0),
        )


@dataclass(frozen=True)
class TaskMetrics:
    """Task execution metrics snapshot"""
    active_tasks_count: int
    completed_tasks_count: int
    failed_tasks_count: int
    avg_task_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "active_tasks_count": self.active_tasks_count,
            "completed_tasks_count": self.completed_tasks_count,
            "failed_tasks_count": self.failed_tasks_count,
            "avg_task_duration_seconds": self.avg_task_duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskMetrics":
        """Create from dictionary"""
        return cls(
            active_tasks_count=data.get("active_tasks_count", 0),
            completed_tasks_count=data.get("completed_tasks_count", 0),
            failed_tasks_count=data.get("failed_tasks_count", 0),
            avg_task_duration_seconds=data.get("avg_task_duration_seconds", 0.0),
        )


# ==================== Entities ====================

@dataclass
class AgentHealthStatus:
    """
    Agent health status entity for monitoring.
    
    Tracks agent health metrics and status over time.
    Design principles:
    - Real-time health monitoring
    - Failure detection (30s threshold)
    - Automated recovery triggers
    """
    id: UUID
    agent_id: UUID
    workflow_execution_id: UUID
    status: HealthStatus
    last_heartbeat: datetime
    
    # Metrics
    resource_metrics: ResourceMetrics
    cost_metrics: CostMetrics
    task_metrics: TaskMetrics
    
    # Health score (0-1)
    health_score: float
    
    # Error tracking
    error_message: Optional[str] = None
    recovery_attempts: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate health status invariants"""
        if self.health_score < 0.0 or self.health_score > 1.0:
            raise ValueError("Health score must be between 0.0 and 1.0")
        
        if self.recovery_attempts < 0:
            raise ValueError("Recovery attempts cannot be negative")
        
        if self.resource_metrics.memory_usage_mb < 0:
            raise ValueError("Memory usage cannot be negative")
        
        if self.resource_metrics.cpu_usage_percent < 0 or self.resource_metrics.cpu_usage_percent > 100:
            raise ValueError("CPU usage must be between 0 and 100")

    def mark_as_degraded(self, reason: str) -> None:
        """Mark health status as degraded"""
        self.status = HealthStatus.DEGRADED
        self.error_message = reason
        self.updated_at = datetime.utcnow()

    def mark_as_failed(self, error: str) -> None:
        """Mark health status as failed"""
        self.status = HealthStatus.FAILED
        self.error_message = error
        self.updated_at = datetime.utcnow()

    def mark_as_recovering(self) -> None:
        """Mark health status as recovering"""
        if self.status not in [HealthStatus.DEGRADED, HealthStatus.FAILED]:
            raise ValueError(f"Cannot mark as recovering from status: {self.status}")
        self.status = HealthStatus.RECOVERING
        self.recovery_attempts += 1
        self.updated_at = datetime.utcnow()

    def mark_as_healthy(self) -> None:
        """Mark health status as healthy"""
        self.status = HealthStatus.HEALTHY
        self.error_message = None
        self.updated_at = datetime.utcnow()

    def update_heartbeat(self) -> None:
        """Update last heartbeat timestamp"""
        self.last_heartbeat = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def update_metrics(
        self,
        resource_metrics: ResourceMetrics,
        cost_metrics: CostMetrics,
        task_metrics: TaskMetrics,
    ) -> None:
        """Update health metrics"""
        self.resource_metrics = resource_metrics
        self.cost_metrics = cost_metrics
        self.task_metrics = task_metrics
        self.updated_at = datetime.utcnow()

    def update_health_score(self, score: float) -> None:
        """Update health score"""
        if score < 0.0 or score > 1.0:
            raise ValueError("Health score must be between 0.0 and 1.0")
        self.health_score = score
        self.updated_at = datetime.utcnow()

    def is_heartbeat_stale(self, threshold_seconds: int = 30) -> bool:
        """Check if heartbeat is stale (exceeds threshold)"""
        elapsed = (datetime.utcnow() - self.last_heartbeat).total_seconds()
        return elapsed > threshold_seconds

    def is_healthy(self) -> bool:
        """Check if agent is healthy"""
        return self.status == HealthStatus.HEALTHY

    def is_failing(self) -> bool:
        """Check if agent is failing"""
        return self.status in [HealthStatus.DEGRADED, HealthStatus.FAILED]

    def get_failure_rate(self) -> float:
        """Calculate task failure rate"""
        total_tasks = self.task_metrics.completed_tasks_count + self.task_metrics.failed_tasks_count
        if total_tasks == 0:
            return 0.0
        return self.task_metrics.failed_tasks_count / total_tasks

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence"""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "workflow_execution_id": str(self.workflow_execution_id),
            "status": self.status.value,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "resource_metrics": self.resource_metrics.to_dict(),
            "cost_metrics": self.cost_metrics.to_dict(),
            "task_metrics": self.task_metrics.to_dict(),
            "health_score": self.health_score,
            "error_message": self.error_message,
            "recovery_attempts": self.recovery_attempts,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentHealthStatus":
        """Create from dictionary"""
        return cls(
            id=UUID(data["id"]),
            agent_id=UUID(data["agent_id"]),
            workflow_execution_id=UUID(data["workflow_execution_id"]),
            status=HealthStatus(data["status"]),
            last_heartbeat=datetime.fromisoformat(data["last_heartbeat"]),
            resource_metrics=ResourceMetrics.from_dict(data["resource_metrics"]),
            cost_metrics=CostMetrics.from_dict(data["cost_metrics"]),
            task_metrics=TaskMetrics.from_dict(data["task_metrics"]),
            health_score=data["health_score"],
            error_message=data.get("error_message"),
            recovery_attempts=data.get("recovery_attempts", 0),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )

    @classmethod
    def create(
        cls,
        agent_id: UUID,
        workflow_execution_id: UUID,
        resource_metrics: ResourceMetrics,
        cost_metrics: CostMetrics,
        task_metrics: TaskMetrics,
        health_score: float = 1.0,
    ) -> "AgentHealthStatus":
        """Factory method to create a new health status"""
        return cls(
            id=uuid4(),
            agent_id=agent_id,
            workflow_execution_id=workflow_execution_id,
            status=HealthStatus.HEALTHY,
            last_heartbeat=datetime.utcnow(),
            resource_metrics=resource_metrics,
            cost_metrics=cost_metrics,
            task_metrics=task_metrics,
            health_score=health_score,
        )


@dataclass
class AgentHealthIncident:
    """
    Health incident entity for tracking agent issues.
    
    Tracks incidents requiring investigation or remediation.
    """
    id: UUID
    agent_id: UUID
    workflow_execution_id: UUID
    severity: IncidentSeverity
    status: IncidentStatus
    
    # Incident details
    title: str
    description: str
    error_message: Optional[str] = None
    
    # Metrics at incident time
    health_score: float = 0.0
    failure_rate: float = 0.0
    
    # Resolution
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate incident invariants"""
        if self.health_score < 0.0 or self.health_score > 1.0:
            raise ValueError("Health score must be between 0.0 and 1.0")
        
        if self.failure_rate < 0.0 or self.failure_rate > 1.0:
            raise ValueError("Failure rate must be between 0.0 and 1.0")

    def mark_as_investigating(self) -> None:
        """Mark incident as being investigated"""
        if self.status != IncidentStatus.OPEN:
            raise ValueError(f"Cannot mark as investigating from status: {self.status}")
        self.status = IncidentStatus.INVESTIGATING
        self.updated_at = datetime.utcnow()

    def mark_as_resolved(self, resolution: str) -> None:
        """Mark incident as resolved"""
        if self.status == IncidentStatus.RESOLVED:
            raise ValueError("Incident already resolved")
        self.status = IncidentStatus.RESOLVED
        self.resolution_notes = resolution
        self.resolved_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def mark_as_ignored(self, reason: str) -> None:
        """Mark incident as ignored"""
        self.status = IncidentStatus.IGNORED
        self.resolution_notes = reason
        self.updated_at = datetime.utcnow()

    def is_open(self) -> bool:
        """Check if incident is open"""
        return self.status in [IncidentStatus.OPEN, IncidentStatus.INVESTIGATING]

    def is_resolved(self) -> bool:
        """Check if incident is resolved"""
        return self.status == IncidentStatus.RESOLVED

    def is_critical(self) -> bool:
        """Check if incident is critical severity"""
        return self.severity == IncidentSeverity.CRITICAL

    def get_time_to_resolution(self) -> Optional[float]:
        """Get time to resolution in seconds"""
        if not self.resolved_at:
            return None
        return (self.resolved_at - self.created_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence"""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "workflow_execution_id": str(self.workflow_execution_id),
            "severity": self.severity.value,
            "status": self.status.value,
            "title": self.title,
            "description": self.description,
            "error_message": self.error_message,
            "health_score": self.health_score,
            "failure_rate": self.failure_rate,
            "resolution_notes": self.resolution_notes,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentHealthIncident":
        """Create from dictionary"""
        return cls(
            id=UUID(data["id"]),
            agent_id=UUID(data["agent_id"]),
            workflow_execution_id=UUID(data["workflow_execution_id"]),
            severity=IncidentSeverity(data["severity"]),
            status=IncidentStatus(data["status"]),
            title=data["title"],
            description=data["description"],
            error_message=data.get("error_message"),
            health_score=data.get("health_score", 0.0),
            failure_rate=data.get("failure_rate", 0.0),
            resolution_notes=data.get("resolution_notes"),
            resolved_at=datetime.fromisoformat(data["resolved_at"]) if data.get("resolved_at") else None,
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )

    @classmethod
    def create(
        cls,
        agent_id: UUID,
        workflow_execution_id: UUID,
        severity: IncidentSeverity,
        title: str,
        description: str,
        error_message: Optional[str] = None,
        health_score: float = 0.0,
        failure_rate: float = 0.0,
    ) -> "AgentHealthIncident":
        """Factory method to create a new incident"""
        return cls(
            id=uuid4(),
            agent_id=agent_id,
            workflow_execution_id=workflow_execution_id,
            severity=severity,
            status=IncidentStatus.OPEN,
            title=title,
            description=description,
            error_message=error_message,
            health_score=health_score,
            failure_rate=failure_rate,
        )
