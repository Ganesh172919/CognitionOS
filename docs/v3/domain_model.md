# CognitionOS V3 - Domain Model

**Version**: 3.0
**Date**: 2026-02-11
**Purpose**: Define core domain entities, value objects, and bounded contexts

---

## Overview

This document defines the **ubiquitous language** and **domain model** for CognitionOS. It represents the mental model shared by domain experts and developers.

---

## Bounded Contexts

CognitionOS is organized into 5 main bounded contexts:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Workflow   │────▶│    Agent     │────▶│     Task     │
│   Context    │     │   Context    │     │   Context    │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                     │
       │                    ▼                     │
       │             ┌──────────────┐            │
       └────────────▶│    Memory    │◀───────────┘
                     │   Context    │
                     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Execution   │
                     │   Context    │
                     └──────────────┘
```

---

## 1. Workflow Context

**Responsibility**: Manage workflow definitions, execution, and versioning

### Entities

#### Workflow (Aggregate Root)
```python
@dataclass
class Workflow:
    """
    A workflow is a declarative definition of a multi-step process.

    Invariants:
    - ID + Version must be unique
    - Steps must form a valid DAG (no cycles)
    - Active workflows must have at least one step
    """
    id: WorkflowId
    version: Version
    name: str
    description: Optional[str]
    steps: List[WorkflowStep]
    inputs: List[WorkflowInput]
    outputs: List[WorkflowOutput]
    status: WorkflowStatus
    schedule: Optional[CronSchedule]
    created_by: UserId
    created_at: datetime
    updated_at: datetime

    def can_execute(self) -> bool:
        """Domain rule: Workflow can execute if active and valid."""
        return (
            self.status == WorkflowStatus.ACTIVE and
            len(self.steps) > 0 and
            self._has_valid_dag()
        )

    def validate_dag(self) -> Result[None, ValidationError]:
        """Validate DAG has no cycles."""
        # Cycle detection algorithm
        ...

    def add_step(self, step: WorkflowStep) -> Result[None, WorkflowError]:
        """Add step maintaining DAG validity."""
        if not self._would_create_cycle(step):
            self.steps.append(step)
            return Ok(None)
        return Err(CycleDetected())
```

#### WorkflowExecution
```python
@dataclass
class WorkflowExecution:
    """
    An execution instance of a workflow.

    Invariants:
    - Must reference an existing workflow
    - Status transitions must be valid
    - Cannot restart a completed execution
    """
    id: ExecutionId
    workflow_id: WorkflowId
    workflow_version: Version
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]]
    status: ExecutionStatus
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error: Optional[str]
    user_id: UserId

    def can_transition_to(self, new_status: ExecutionStatus) -> bool:
        """Domain rule: Valid status transitions."""
        valid_transitions = {
            ExecutionStatus.PENDING: [ExecutionStatus.RUNNING, ExecutionStatus.CANCELLED],
            ExecutionStatus.RUNNING: [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED],
            ExecutionStatus.COMPLETED: [],
            ExecutionStatus.FAILED: [ExecutionStatus.RUNNING],  # Can retry
            ExecutionStatus.CANCELLED: []
        }
        return new_status in valid_transitions[self.status]

    def mark_completed(self, outputs: Dict[str, Any]) -> Result[None, ExecutionError]:
        """Mark execution as completed."""
        if not self.can_transition_to(ExecutionStatus.COMPLETED):
            return Err(InvalidStateTransition())

        self.status = ExecutionStatus.COMPLETED
        self.outputs = outputs
        self.completed_at = datetime.utcnow()
        return Ok(None)
```

### Value Objects

#### WorkflowId
```python
@dataclass(frozen=True)
class WorkflowId:
    """Value object for workflow identifier."""
    value: str

    def __post_init__(self):
        if not self.value or len(self.value) > 200:
            raise ValueError("Invalid workflow ID")
```

#### WorkflowStatus
```python
class WorkflowStatus(Enum):
    """Status of a workflow definition."""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
```

#### ExecutionStatus
```python
class ExecutionStatus(Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
```

#### WorkflowStep
```python
@dataclass(frozen=True)
class WorkflowStep:
    """Value object representing a workflow step."""
    id: str
    type: StepType
    name: Optional[str]
    depends_on: List[str]
    params: Dict[str, Any]
    agent_role: Optional[AgentRole]
    timeout: Duration
    retry: RetryPolicy
    condition: Optional[str]
```

### Domain Services

#### WorkflowValidator
```python
class WorkflowValidator:
    """Service for validating workflows."""

    def validate(self, workflow: Workflow) -> Result[None, List[ValidationError]]:
        """Validate workflow against all domain rules."""
        errors = []

        # Check DAG validity
        if not workflow._has_valid_dag():
            errors.append(ValidationError("Workflow contains cycles"))

        # Check all dependencies exist
        step_ids = {step.id for step in workflow.steps}
        for step in workflow.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    errors.append(ValidationError(f"Unknown dependency: {dep}"))

        # Check inputs are defined
        # ... more validations

        return Ok(None) if not errors else Err(errors)
```

#### WorkflowExecutor
```python
class WorkflowExecutor:
    """Service for executing workflows."""

    async def execute(
        self,
        workflow: Workflow,
        inputs: Dict[str, Any],
        user_id: UserId
    ) -> WorkflowExecution:
        """Execute a workflow."""

        # Create execution entity
        execution = WorkflowExecution(
            id=ExecutionId.generate(),
            workflow_id=workflow.id,
            workflow_version=workflow.version,
            inputs=inputs,
            status=ExecutionStatus.PENDING,
            user_id=user_id
        )

        # Execute steps in DAG order
        # ... domain logic for execution

        return execution
```

---

## 2. Agent Context

**Responsibility**: Manage agent lifecycle, assignment, and performance

### Entities

#### Agent (Aggregate Root)
```python
@dataclass
class Agent:
    """
    An AI agent that can perform tasks.

    Invariants:
    - Role must be valid
    - Active agents must have a model configured
    - Confidence threshold must be between 0 and 1
    """
    id: AgentId
    name: str
    role: AgentRole
    model: ModelId
    temperature: float
    capabilities: List[str]
    allowed_tools: List[ToolId]
    failure_strategy: FailureStrategy
    confidence_threshold: float
    status: AgentStatus
    created_at: datetime

    def can_execute_task(self, task: Task) -> bool:
        """Domain rule: Agent can execute if capable and active."""
        return (
            self.status == AgentStatus.ACTIVE and
            all(cap in self.capabilities for cap in task.required_capabilities)
        )

    def assign_to_task(self, task: Task) -> Result[AgentAssignment, AssignmentError]:
        """Assign this agent to a task."""
        if not self.can_execute_task(task):
            return Err(AgentNotCapable())

        return Ok(AgentAssignment(
            agent_id=self.id,
            task_id=task.id,
            assigned_at=datetime.utcnow()
        ))
```

#### AgentMetrics
```python
@dataclass
class AgentMetrics:
    """
    Performance metrics for an agent over a time window.

    This is an entity (not value object) because metrics evolve over time.
    """
    agent_id: AgentId
    time_window: TimeWindow
    task_count: int
    avg_confidence: float
    avg_quality_score: float
    success_rate: float
    avg_latency_ms: int
    total_cost: Money

    def is_performing_well(self, policy: PerformancePolicy) -> bool:
        """Domain rule: Agent is performing well if meets policy."""
        return (
            self.success_rate >= policy.min_success_rate and
            self.avg_confidence >= policy.min_confidence and
            self.avg_quality_score >= policy.min_quality
        )

    def should_be_replaced(self, policy: ReplacementPolicy) -> bool:
        """Domain rule: Agent should be replaced if underperforming."""
        return (
            self.task_count >= policy.min_tasks_for_evaluation and
            not self.is_performing_well(policy.performance_policy)
        )
```

### Value Objects

#### AgentRole
```python
class AgentRole(Enum):
    """Role of an agent in the system."""
    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"
    SUMMARIZER = "summarizer"
    RESEARCHER = "researcher"
    MONITOR = "monitor"
```

#### FailureStrategy
```python
class FailureStrategy(Enum):
    """Strategy for handling agent failures."""
    RETRY = "retry"
    SKIP = "skip"
    REASSIGN = "reassign"
    ESCALATE = "escalate"
```

### Domain Services

#### AgentOrchestrator
```python
class AgentOrchestrator:
    """Service for orchestrating agents."""

    async def assign_task(
        self,
        task: Task,
        available_agents: List[Agent]
    ) -> Result[AgentAssignment, OrchestrationError]:
        """Assign a task to the best available agent."""

        # Filter capable agents
        capable = [a for a in available_agents if a.can_execute_task(task)]

        if not capable:
            return Err(NoCapableAgent())

        # Select best agent (domain logic)
        best_agent = self._select_best_agent(capable, task)

        # Create assignment
        return best_agent.assign_to_task(task)

    def _select_best_agent(self, agents: List[Agent], task: Task) -> Agent:
        """Domain logic for agent selection."""
        # Could use various strategies:
        # - Least loaded
        # - Best performance history
        # - Lowest cost
        # - Random
        ...
```

---

## 3. Memory Context

**Responsibility**: Manage memory storage, retrieval, and lifecycle

### Entities

#### Memory (Aggregate Root)
```python
@dataclass
class Memory:
    """
    A memory is a piece of information stored for later retrieval.

    Invariants:
    - Content must not be empty
    - Embedding must match configured dimensions
    - Namespace must exist
    """
    id: MemoryId
    namespace_id: NamespaceId
    type: MemoryType
    scope: MemoryScope
    content: str
    embedding: EmbeddingVector
    metadata: Dict[str, Any]
    user_id: UserId
    created_at: datetime
    accessed_at: datetime
    access_count: int
    expires_at: Optional[datetime]
    compressed: bool
    archived: bool

    def is_stale(self, policy: LifecyclePolicy) -> bool:
        """Domain rule: Memory is stale if not accessed recently."""
        if not policy.ttl_days:
            return False

        age = datetime.utcnow() - self.accessed_at
        return age.days > policy.ttl_days

    def should_compress(self, policy: LifecyclePolicy) -> bool:
        """Domain rule: Memory should be compressed if old."""
        if self.compressed or not policy.compression_after_days:
            return False

        age = datetime.utcnow() - self.created_at
        return age.days > policy.compression_after_days

    def record_access(self) -> None:
        """Record that memory was accessed."""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1
```

#### MemoryNamespace
```python
@dataclass
class MemoryNamespace:
    """
    A namespace for organizing and isolating memories.

    Invariants:
    - Name must be unique
    - Owner must exist
    """
    id: NamespaceId
    name: str
    description: Optional[str]
    owner_id: UserId
    visibility: Visibility
    lifecycle_policy: LifecyclePolicy
    created_at: datetime

    def can_access(self, user_id: UserId) -> bool:
        """Domain rule: User can access if owner or public."""
        return (
            self.owner_id == user_id or
            self.visibility == Visibility.PUBLIC
        )
```

### Value Objects

#### MemoryType
```python
class MemoryType(Enum):
    """Type of memory based on duration and importance."""
    WORKING = "working"        # Temporary, task-specific
    SHORT_TERM = "short_term"  # Recent, may decay
    LONG_TERM = "long_term"    # Persistent, important
    EPISODIC = "episodic"      # Event-based
    SEMANTIC = "semantic"      # Factual knowledge
```

#### EmbeddingVector
```python
@dataclass(frozen=True)
class EmbeddingVector:
    """Value object for embedding vector."""
    values: List[float]
    dimensions: int

    def __post_init__(self):
        if len(self.values) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dimensions")

    def cosine_similarity(self, other: 'EmbeddingVector') -> float:
        """Calculate cosine similarity with another vector."""
        # Vector math
        ...
```

### Domain Services

#### MemoryRetriever
```python
class MemoryRetriever:
    """Service for retrieving relevant memories."""

    async def retrieve(
        self,
        query: str,
        query_embedding: EmbeddingVector,
        namespace_id: NamespaceId,
        limit: int = 10
    ) -> List[Memory]:
        """Retrieve most relevant memories for query."""

        # This is domain logic for ranking/retrieval
        # Implementation details in infrastructure
        ...
```

---

## 4. Task Context

**Responsibility**: Task planning, decomposition, and dependency management

### Entities

#### Task (Aggregate Root)
```python
@dataclass
class Task:
    """
    A unit of work to be executed.

    Invariants:
    - Description must not be empty
    - Priority must be valid
    - Dependencies must form DAG
    """
    id: TaskId
    name: str
    description: str
    required_capabilities: List[str]
    priority: Priority
    complexity: Complexity
    status: TaskStatus
    depends_on: List[TaskId]
    estimated_duration: Duration
    created_at: datetime
    completed_at: Optional[datetime]

    def can_start(self, completed_tasks: Set[TaskId]) -> bool:
        """Domain rule: Task can start if all dependencies complete."""
        return all(dep in completed_tasks for dep in self.depends_on)

    def mark_completed(self) -> Result[None, TaskError]:
        """Mark task as completed."""
        if self.status != TaskStatus.RUNNING:
            return Err(InvalidStateTransition())

        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        return Ok(None)
```

### Value Objects

#### Priority
```python
class Priority(Enum):
    """Task priority level."""
    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    MINIMAL = 1
```

#### Complexity
```python
class Complexity(Enum):
    """Task complexity estimate."""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"
```

---

## 5. Execution Context

**Responsibility**: Tool execution, sandboxing, and permission management

### Entities

#### ToolExecution
```python
@dataclass
class ToolExecution:
    """
    An execution of a tool.

    Invariants:
    - Tool must exist
    - Permissions must be granted
    - Resource limits must be respected
    """
    id: ExecutionId
    tool_id: ToolId
    params: Dict[str, Any]
    status: ExecutionStatus
    result: Optional[ExecutionResult]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration: Optional[Duration]
    resource_usage: ResourceUsage

    def within_limits(self, limits: ResourceLimits) -> bool:
        """Domain rule: Execution within resource limits."""
        return (
            self.resource_usage.cpu_seconds <= limits.max_cpu_seconds and
            self.resource_usage.memory_mb <= limits.max_memory_mb and
            (self.duration is None or self.duration <= limits.max_duration)
        )
```

### Value Objects

#### ResourceLimits
```python
@dataclass(frozen=True)
class ResourceLimits:
    """Resource limits for tool execution."""
    max_cpu_seconds: int
    max_memory_mb: int
    max_duration: Duration
    allowed_network: bool
    allowed_filesystem: bool
```

---

## Domain Events

Domain events represent things that happened in the domain that domain experts care about.

```python
# Workflow Events
class WorkflowExecutionStarted(DomainEvent):
    execution_id: ExecutionId
    workflow_id: WorkflowId
    user_id: UserId
    timestamp: datetime

class WorkflowExecutionCompleted(DomainEvent):
    execution_id: ExecutionId
    outputs: Dict[str, Any]
    duration: Duration
    timestamp: datetime

# Agent Events
class AgentAssignedToTask(DomainEvent):
    agent_id: AgentId
    task_id: TaskId
    timestamp: datetime

class AgentReplacedDueToPerformance(DomainEvent):
    old_agent_id: AgentId
    new_agent_id: AgentId
    reason: str
    metrics: AgentMetrics
    timestamp: datetime

# Memory Events
class MemoryCreated(DomainEvent):
    memory_id: MemoryId
    namespace_id: NamespaceId
    type: MemoryType
    timestamp: datetime

class MemoryArchived(DomainEvent):
    memory_id: MemoryId
    reason: str
    timestamp: datetime
```

---

## Repository Interfaces

Repositories are defined in the domain but implemented in infrastructure.

```python
class WorkflowRepository(ABC):
    """Repository for workflows."""

    @abstractmethod
    async def get_by_id(self, id: WorkflowId, version: Version) -> Optional[Workflow]:
        pass

    @abstractmethod
    async def save(self, workflow: Workflow) -> None:
        pass

    @abstractmethod
    async def list_active(self) -> List[Workflow]:
        pass

class AgentRepository(ABC):
    """Repository for agents."""

    @abstractmethod
    async def get_by_id(self, id: AgentId) -> Optional[Agent]:
        pass

    @abstractmethod
    async def find_by_role(self, role: AgentRole) -> List[Agent]:
        pass

    @abstractmethod
    async def save(self, agent: Agent) -> None:
        pass
```

---

## Conclusion

This domain model provides:

1. **Ubiquitous Language**: Shared vocabulary between developers and domain experts
2. **Clear Boundaries**: Bounded contexts separate concerns
3. **Business Rules**: Domain logic encapsulated in entities and services
4. **Invariants**: Rules that must always be true
5. **Type Safety**: Value objects prevent invalid states

The domain model is the heart of the system and should evolve as understanding of the domain deepens.

---

**Document Version**: 1.0
**Status**: Active
**Last Updated**: 2026-02-11
