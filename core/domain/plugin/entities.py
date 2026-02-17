"""
Plugin Domain - Entities

Pure domain entities for Plugin bounded context.
NO external dependencies except Python stdlib.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


# ==================== Enums ====================

class PluginStatus(str, Enum):
    """Plugin status types"""
    ACTIVE = "active"
    DISABLED = "disabled"
    PENDING_REVIEW = "pending_review"
    DEPRECATED = "deprecated"


class PluginRuntime(str, Enum):
    """Plugin runtime environment types"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    DOCKER = "docker"


class ExecutionStatus(str, Enum):
    """Plugin execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class TrustLevel(str, Enum):
    """Trust level categorization"""
    UNTRUSTED = "untrusted"      # 0-30
    LOW = "low"                  # 31-50
    MEDIUM = "medium"            # 51-70
    HIGH = "high"                # 71-85
    VERIFIED = "verified"        # 86-100


# ==================== Value Objects ====================

@dataclass(frozen=True)
class PluginId:
    """Plugin identifier value object"""
    value: UUID

    def __str__(self) -> str:
        return str(self.value)

    @staticmethod
    def generate() -> "PluginId":
        """Generate a new plugin ID"""
        return PluginId(value=uuid4())


@dataclass(frozen=True)
class VersionInfo:
    """Semantic version information"""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None

    def __post_init__(self):
        if self.major < 0 or self.minor < 0 or self.patch < 0:
            raise ValueError("Version numbers must be non-negative")

    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        return version

    @staticmethod
    def parse(version_str: str) -> "VersionInfo":
        """Parse version string like '1.0.0' or '1.0.0-beta'"""
        parts = version_str.split("-", 1)
        version_parts = parts[0].split(".")
        
        if len(version_parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}")
        
        major, minor, patch = map(int, version_parts)
        prerelease = parts[1] if len(parts) > 1 else None
        
        return VersionInfo(major, minor, patch, prerelease)

    def is_compatible_with(self, other: "VersionInfo") -> bool:
        """Check if this version is compatible with another (same major version)"""
        return self.major == other.major


@dataclass(frozen=True)
class Permission:
    """Plugin permission specification"""
    resource: str
    action: str
    description: str

    def __post_init__(self):
        if not self.resource or not self.action:
            raise ValueError("Permission resource and action cannot be empty")

    def matches(self, resource: str, action: str) -> bool:
        """Check if this permission matches a requested access"""
        # Support wildcard matching
        resource_match = self.resource == "*" or self.resource == resource
        action_match = self.action == "*" or self.action == action
        return resource_match and action_match


@dataclass
class SandboxConfig:
    """Sandbox configuration for plugin execution"""
    network_access: bool = False
    filesystem_access: bool = False
    max_memory_mb: int = 512
    max_cpu_percent: int = 50
    timeout_seconds: int = 300
    allowed_paths: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        if self.max_cpu_percent <= 0 or self.max_cpu_percent > 100:
            raise ValueError("max_cpu_percent must be between 1 and 100")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass
class TrustFactor:
    """Individual factor contributing to trust score"""
    name: str
    score: float  # 0.0 - 1.0
    weight: float  # 0.0 - 1.0
    description: str

    def __post_init__(self):
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Trust factor score must be between 0.0 and 1.0")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("Trust factor weight must be between 0.0 and 1.0")

    @property
    def weighted_score(self) -> float:
        """Get weighted contribution to overall score"""
        return self.score * self.weight


# ==================== Entities ====================

@dataclass
class PluginManifest:
    """
    Plugin manifest containing metadata and configuration.
    
    This is a value object that describes what the plugin is and needs.
    """
    name: str
    version: VersionInfo
    author: str
    description: str
    entry_point: str
    runtime: PluginRuntime
    permissions: List[Permission] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    sandbox_config: SandboxConfig = field(default_factory=SandboxConfig)
    homepage_url: Optional[str] = None
    documentation_url: Optional[str] = None
    source_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.name or not self.author:
            raise ValueError("Plugin name and author are required")
        if not self.entry_point:
            raise ValueError("Plugin entry_point is required")
        if not self.description:
            raise ValueError("Plugin description is required")

    def requires_permission(self, resource: str, action: str) -> bool:
        """Check if plugin requires a specific permission"""
        return any(perm.matches(resource, action) for perm in self.permissions)

    def has_dependency(self, package: str) -> bool:
        """Check if plugin has a specific dependency"""
        return package in self.dependencies


@dataclass
class PluginTrustScore:
    """
    Trust score for a plugin based on multiple factors.
    
    Score range: 0-100
    """
    plugin_id: PluginId
    score: int
    factors: List[TrustFactor] = field(default_factory=list)
    last_calculated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    calculation_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0 <= self.score <= 100:
            raise ValueError("Trust score must be between 0 and 100")

    @property
    def trust_level(self) -> TrustLevel:
        """Get categorical trust level"""
        if self.score <= 30:
            return TrustLevel.UNTRUSTED
        elif self.score <= 50:
            return TrustLevel.LOW
        elif self.score <= 70:
            return TrustLevel.MEDIUM
        elif self.score <= 85:
            return TrustLevel.HIGH
        else:
            return TrustLevel.VERIFIED

    def is_trusted(self, minimum_score: int = 50) -> bool:
        """Check if plugin meets minimum trust threshold"""
        return self.score >= minimum_score

    def get_factor(self, name: str) -> Optional[TrustFactor]:
        """Get specific trust factor by name"""
        return next((f for f in self.factors if f.name == name), None)


@dataclass
class Plugin:
    """
    Plugin aggregate root.
    
    Represents a registered plugin in the system.
    """
    id: PluginId
    manifest: PluginManifest
    status: PluginStatus
    trust_score: Optional[PluginTrustScore] = None
    tenant_id: Optional[UUID] = None  # None for global/marketplace plugins
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    published_at: Optional[datetime] = None
    deprecated_at: Optional[datetime] = None
    install_count: int = 0
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    average_execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.install_count < 0 or self.execution_count < 0:
            raise ValueError("Counts must be non-negative")
        if self.success_count < 0 or self.failure_count < 0:
            raise ValueError("Success/failure counts must be non-negative")

    @staticmethod
    def create(
        manifest: PluginManifest,
        tenant_id: Optional[UUID] = None,
        status: PluginStatus = PluginStatus.PENDING_REVIEW
    ) -> "Plugin":
        """Create a new plugin"""
        return Plugin(
            id=PluginId.generate(),
            manifest=manifest,
            status=status,
            tenant_id=tenant_id
        )

    def activate(self) -> None:
        """Activate the plugin"""
        if self.status == PluginStatus.DEPRECATED:
            raise ValueError("Cannot activate deprecated plugin")
        self.status = PluginStatus.ACTIVE
        self.updated_at = datetime.now(timezone.utc)

    def disable(self) -> None:
        """Disable the plugin"""
        if self.status == PluginStatus.DEPRECATED:
            raise ValueError("Cannot disable deprecated plugin")
        self.status = PluginStatus.DISABLED
        self.updated_at = datetime.now(timezone.utc)

    def deprecate(self) -> None:
        """Mark plugin as deprecated"""
        self.status = PluginStatus.DEPRECATED
        self.deprecated_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def approve(self) -> None:
        """Approve plugin for use"""
        if self.status != PluginStatus.PENDING_REVIEW:
            raise ValueError("Only pending plugins can be approved")
        self.status = PluginStatus.ACTIVE
        self.published_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def update_trust_score(self, trust_score: PluginTrustScore) -> None:
        """Update the plugin's trust score"""
        if trust_score.plugin_id != self.id:
            raise ValueError("Trust score plugin_id mismatch")
        self.trust_score = trust_score
        self.updated_at = datetime.now(timezone.utc)

    def record_installation(self) -> None:
        """Record that the plugin was installed"""
        self.install_count += 1
        self.updated_at = datetime.now(timezone.utc)

    def record_execution(
        self,
        success: bool,
        execution_time_ms: float
    ) -> None:
        """Record a plugin execution"""
        self.execution_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        # Update rolling average execution time
        if self.execution_count > 1:
            self.average_execution_time_ms = (
                (self.average_execution_time_ms * (self.execution_count - 1) + execution_time_ms)
                / self.execution_count
            )
        else:
            self.average_execution_time_ms = execution_time_ms
        
        self.updated_at = datetime.now(timezone.utc)

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 - 1.0)"""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count

    @property
    def is_active(self) -> bool:
        """Check if plugin is active"""
        return self.status == PluginStatus.ACTIVE

    @property
    def is_global(self) -> bool:
        """Check if plugin is a global/marketplace plugin"""
        return self.tenant_id is None

    def can_be_executed(self) -> bool:
        """Check if plugin can be executed"""
        return self.status == PluginStatus.ACTIVE


@dataclass
class PluginExecution:
    """
    Plugin execution record.
    
    Tracks individual plugin execution instances.
    """
    id: UUID
    plugin_id: PluginId
    tenant_id: UUID
    status: ExecutionStatus
    input_data: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_context: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.end_time and self.end_time < self.start_time:
            raise ValueError("end_time cannot be before start_time")

    @staticmethod
    def create(
        plugin_id: PluginId,
        tenant_id: UUID,
        input_data: Dict[str, Any]
    ) -> "PluginExecution":
        """Create a new plugin execution"""
        return PluginExecution(
            id=uuid4(),
            plugin_id=plugin_id,
            tenant_id=tenant_id,
            status=ExecutionStatus.PENDING,
            input_data=input_data,
            start_time=datetime.now(timezone.utc)
        )

    def start(self) -> None:
        """Mark execution as started"""
        if self.status != ExecutionStatus.PENDING:
            raise ValueError(f"Cannot start execution in {self.status} status")
        self.status = ExecutionStatus.RUNNING
        self.start_time = datetime.now(timezone.utc)

    def complete(self, result: Dict[str, Any]) -> None:
        """Mark execution as completed successfully"""
        if self.status != ExecutionStatus.RUNNING:
            raise ValueError(f"Cannot complete execution in {self.status} status")
        self.status = ExecutionStatus.COMPLETED
        self.result = result
        self.end_time = datetime.now(timezone.utc)

    def fail(self, error: str) -> None:
        """Mark execution as failed"""
        if self.status not in [ExecutionStatus.PENDING, ExecutionStatus.RUNNING]:
            raise ValueError(f"Cannot fail execution in {self.status} status")
        self.status = ExecutionStatus.FAILED
        self.error = error
        self.end_time = datetime.now(timezone.utc)

    def timeout(self) -> None:
        """Mark execution as timed out"""
        if self.status != ExecutionStatus.RUNNING:
            raise ValueError(f"Cannot timeout execution in {self.status} status")
        self.status = ExecutionStatus.TIMEOUT
        self.error = "Execution exceeded timeout limit"
        self.end_time = datetime.now(timezone.utc)

    def cancel(self) -> None:
        """Cancel the execution"""
        if self.status not in [ExecutionStatus.PENDING, ExecutionStatus.RUNNING]:
            raise ValueError(f"Cannot cancel execution in {self.status} status")
        self.status = ExecutionStatus.CANCELLED
        self.end_time = datetime.now(timezone.utc)

    def add_log(self, message: str) -> None:
        """Add a log message"""
        timestamp = datetime.now(timezone.utc).isoformat()
        self.logs.append(f"[{timestamp}] {message}")

    def update_resource_usage(self, usage: Dict[str, Any]) -> None:
        """Update resource usage metrics"""
        self.resource_usage.update(usage)

    @property
    def duration_ms(self) -> Optional[float]:
        """Get execution duration in milliseconds"""
        if not self.end_time:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    @property
    def is_completed(self) -> bool:
        """Check if execution is completed"""
        return self.status in [
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.TIMEOUT,
            ExecutionStatus.CANCELLED
        ]

    @property
    def is_successful(self) -> bool:
        """Check if execution was successful"""
        return self.status == ExecutionStatus.COMPLETED


@dataclass
class PluginInstallation:
    """
    Plugin installation record for a tenant.
    
    Tracks which plugins are installed for which tenants.
    """
    id: UUID
    plugin_id: PluginId
    tenant_id: UUID
    installed_at: datetime
    configuration: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_used_at: Optional[datetime] = None
    usage_count: int = 0

    def __post_init__(self):
        if self.usage_count < 0:
            raise ValueError("usage_count must be non-negative")

    @staticmethod
    def create(
        plugin_id: PluginId,
        tenant_id: UUID,
        configuration: Optional[Dict[str, Any]] = None
    ) -> "PluginInstallation":
        """Create a new plugin installation"""
        return PluginInstallation(
            id=uuid4(),
            plugin_id=plugin_id,
            tenant_id=tenant_id,
            installed_at=datetime.now(timezone.utc),
            configuration=configuration or {}
        )

    def enable(self) -> None:
        """Enable the plugin for this tenant"""
        self.enabled = True

    def disable(self) -> None:
        """Disable the plugin for this tenant"""
        self.enabled = False

    def update_configuration(self, configuration: Dict[str, Any]) -> None:
        """Update plugin configuration"""
        self.configuration = configuration

    def record_usage(self) -> None:
        """Record plugin usage"""
        self.usage_count += 1
        self.last_used_at = datetime.now(timezone.utc)

    @property
    def is_active(self) -> bool:
        """Check if installation is active"""
        return self.enabled
