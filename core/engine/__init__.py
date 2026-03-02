"""
CognitionOS Core Engine Layer

Central orchestration engine that coordinates all platform subsystems:
- Event-driven architecture with pub/sub messaging
- Plugin lifecycle management
- Service registry and discovery
- Configuration hot-reload
- Circuit breaker pattern
- AI agent orchestration
- Workflow execution engine
- Autonomous code generation
"""

from core.engine.engine_orchestrator import EngineOrchestrator, EngineState
from core.engine.event_bus import EventBus, Event, EventHandler, EventPriority
from core.engine.plugin_manager import PluginManager, PluginMetadata, PluginState
from core.engine.service_registry import ServiceRegistry, ServiceDescriptor, ServiceHealth
from core.engine.config_manager import ConfigManager, ConfigSource, ConfigChangeEvent
from core.engine.agent_orchestrator import (
    AgentOrchestrator, AgentContext, AgentResponse,
    ToolRegistry, ModelRouter, CostTracker,
    get_agent_orchestrator,
)
from core.engine.workflow_engine import (
    WorkflowEngine, WorkflowDefinition, WorkflowExecution,
    WorkflowStep, StepType, get_workflow_engine,
)
from core.engine.di_container import (
    DependencyContainer, ServiceLifetime, ServiceDescriptor as DIServiceDescriptor,
    CircularDependencyError, ServiceNotRegisteredError,
    get_container, reset_container,
)
from core.engine.distributed_lock import (
    DistributedLockManager, LockInfo, LockType, LockState,
    LeaderElectionResult, LockAcquisitionError,
    get_lock_manager,
)

__all__ = [
    "EngineOrchestrator", "EngineState",
    "EventBus", "Event", "EventHandler", "EventPriority",
    "PluginManager", "PluginMetadata", "PluginState",
    "ServiceRegistry", "ServiceDescriptor", "ServiceHealth",
    "ConfigManager", "ConfigSource", "ConfigChangeEvent",
    "AgentOrchestrator", "AgentContext", "AgentResponse",
    "ToolRegistry", "ModelRouter", "CostTracker",
    "get_agent_orchestrator",
    "WorkflowEngine", "WorkflowDefinition", "WorkflowExecution",
    "WorkflowStep", "StepType", "get_workflow_engine",
    "DependencyContainer", "ServiceLifetime", "DIServiceDescriptor",
    "CircularDependencyError", "ServiceNotRegisteredError",
    "get_container", "reset_container",
    "DistributedLockManager", "LockInfo", "LockType", "LockState",
    "LeaderElectionResult", "LockAcquisitionError",
    "get_lock_manager",
]
