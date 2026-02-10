# CognitionOS V2 - Refactor Plan

**Document Version**: 1.0
**Date**: 2026-02-10
**Purpose**: Define what must be split, modularized, and abstracted for V2
**Status**: Planning Phase

---

## Executive Summary

V2 refactoring focuses on **architectural discipline** without rewriting from scratch. We will:
1. **Split** monolithic components into focused modules
2. **Modularize** tightly coupled code into independent units
3. **Abstract** repeated patterns into reusable libraries
4. **Enforce** contracts between components
5. **Eliminate** hidden coupling

**Guiding Principle**: **Make implicit contracts explicit**

---

## Refactoring Strategy

### What We Will NOT Do
- ❌ Rewrite everything from scratch
- ❌ Change core architecture (microservices stays)
- ❌ Remove core concepts from V1
- ❌ Simplify for convenience
- ❌ Turn this into a tutorial project

### What We WILL Do
- ✅ Split large files into focused modules
- ✅ Extract common patterns into shared libraries
- ✅ Make agent contracts explicit (Pydantic schemas)
- ✅ Formalize workflow definitions (DSL)
- ✅ Enforce service boundaries (no direct DB access across services)
- ✅ Create pluggable abstractions (strategy pattern for agents, tools)

---

## Part A: What Must Be Split

### 1. Agent Orchestrator - Split into Focused Modules

**Current State**: `services/agent-orchestrator/src/main.py` (likely >500 lines)

**Problems**:
- Agent registration mixed with task assignment
- Agent health monitoring mixed with execution tracking
- Single file handles too many responsibilities

**Refactor Plan**:

```
services/agent-orchestrator/
├── src/
│   ├── main.py                    # FastAPI app, routes only (~150 lines)
│   ├── core/
│   │   ├── registry.py            # Agent registration, listing
│   │   ├── scheduler.py           # Task assignment logic
│   │   ├── health_monitor.py     # Agent health checking
│   │   ├── execution_tracker.py  # Track task execution
│   │   └── load_balancer.py      # Load balancing strategy
│   ├── agents/                    # NEW: Typed agent definitions
│   │   ├── base.py                # AgentBase class
│   │   ├── planner.py             # PlannerAgent with schemas
│   │   ├── executor.py            # ExecutorAgent with schemas
│   │   ├── critic.py              # CriticAgent with schemas
│   │   ├── summarizer.py          # SummarizerAgent with schemas
│   │   └── custom.py              # CustomAgent base for extensions
│   ├── strategies/                # NEW: Pluggable strategies
│   │   ├── failure_strategies.py  # RetryStrategy, SkipStrategy, etc.
│   │   ├── assignment_strategies.py # RoundRobin, LeastLoaded, etc.
│   │   └── replacement_strategies.py # When to replace agents
│   └── models/                    # Pydantic models
│       ├── agent.py               # Agent* models
│       ├── assignment.py          # Assignment* models
│       └── metrics.py             # AgentMetrics models
```

**Benefits**:
- Each file has single responsibility
- Easier to test (mock dependencies)
- Easier to extend (add new strategies)
- Clear contracts (Pydantic models in separate files)

---

### 2. Memory Service - Split into Storage + Lifecycle

**Current State**: `services/memory-service/src/main.py` (handles storage and retrieval only)

**Problems**:
- No memory lifecycle management
- No garbage collection
- No compression
- No debugging tools

**Refactor Plan**:

```
services/memory-service/
├── src/
│   ├── main.py                    # FastAPI app, routes (~200 lines)
│   ├── core/
│   │   ├── storage.py             # Memory CRUD operations
│   │   ├── retrieval.py           # Vector search and ranking
│   │   ├── lifecycle.py           # NEW: Memory lifecycle management
│   │   ├── garbage_collector.py  # NEW: GC logic
│   │   ├── compressor.py          # NEW: Memory compression
│   │   ├── namespace_manager.py  # NEW: Namespace isolation
│   │   └── access_control.py     # NEW: Permission enforcement
│   ├── jobs/                      # NEW: Background jobs
│   │   ├── gc_daily.py            # Daily GC job
│   │   ├── compress_old.py        # Compression job
│   │   └── archive_stale.py       # Archival job
│   ├── debug/                     # NEW: Debugging tools
│   │   ├── inspector.py           # Memory inspector CLI
│   │   ├── analyzer.py            # Access pattern analyzer
│   │   └── visualizer.py          # Memory heatmap data
│   └── models/
│       ├── memory.py              # Memory* models
│       ├── namespace.py           # Namespace* models
│       └── lifecycle.py           # LifecyclePolicy models
```

**Benefits**:
- Clear separation of storage vs lifecycle
- Background jobs are standalone modules
- Debugging tools don't pollute core logic
- Easy to add new memory policies

---

### 3. AI Runtime - Split Validation into Separate Service

**Current State**: `services/ai-runtime/src/main.py` + `output_validator.py`

**Problems**:
- Output validation is embedded in AI Runtime
- Should be a separate quality gate service
- Cross-agent verification not possible with current architecture

**Refactor Plan**:

**ai-runtime** (focused on LLM calls only):
```
services/ai-runtime/
├── src/
│   ├── main.py                    # FastAPI app, routes (~150 lines)
│   ├── core/
│   │   ├── llm_router.py          # Route to OpenAI/Anthropic
│   │   ├── openai_client.py       # OpenAI integration
│   │   ├── anthropic_client.py    # Anthropic integration
│   │   ├── cost_tracker.py        # Token usage and cost
│   │   └── fallback_handler.py    # Automatic fallback logic
│   ├── prompts/
│   │   ├── prompt_manager.py      # Prompt versioning
│   │   └── ab_tester.py           # A/B testing
│   └── models/
│       ├── completion.py          # Completion* models
│       └── embedding.py           # Embedding* models
```

**ai-quality-gate** (NEW separate service):
```
services/ai-quality-gate/          # NEW SERVICE (Port 8011)
├── src/
│   ├── main.py                    # FastAPI app, routes
│   ├── core/
│   │   ├── output_validator.py    # MOVED from ai-runtime
│   │   ├── cross_verifier.py      # NEW: Cross-agent verification
│   │   ├── self_critique.py       # NEW: Self-critique loops
│   │   ├── policy_enforcer.py     # NEW: Policy enforcement
│   │   └── quality_scorer.py      # Quality metrics
│   ├── policies/
│   │   ├── default_policy.py      # Default validation policy
│   │   └── custom_policies.py     # User-defined policies
│   └── models/
│       ├── validation.py          # Validation* models
│       └── policy.py              # Policy* models
```

**Benefits**:
- AI Runtime focuses solely on LLM calls
- Quality Gate is a reusable service (can validate any AI output)
- Cross-agent verification becomes possible
- Policies are pluggable and testable

---

### 4. Task Planner - Extract Workflow Engine

**Current State**: `services/task-planner/src/main.py` (workflows implicit in code)

**Problems**:
- Workflows are code logic, not data
- No workflow versioning
- No workflow replay
- No workflow visualization

**Refactor Plan**:

**task-planner** (focused on task decomposition only):
```
services/task-planner/
├── src/
│   ├── main.py                    # FastAPI app, routes (~150 lines)
│   ├── core/
│   │   ├── decomposer.py          # Goal → Tasks
│   │   ├── dag_builder.py         # Build task DAG
│   │   ├── complexity_estimator.py # Estimate task complexity
│   │   └── validator.py           # Validate task plan
│   └── models/
│       ├── task.py                # Task* models
│       └── plan.py                # Plan* models
```

**workflow-engine** (NEW separate service):
```
services/workflow-engine/          # NEW SERVICE (Port 8010)
├── src/
│   ├── main.py                    # FastAPI app, routes
│   ├── core/
│   │   ├── dsl_parser.py          # Parse YAML/JSON DSL
│   │   ├── executor.py            # Execute workflow
│   │   ├── replayer.py            # Replay workflow
│   │   ├── versioner.py           # Workflow versioning
│   │   └── visualizer.py          # Execution graph data
│   ├── workflows/                 # Built-in workflows
│   │   ├── agent_training.yaml
│   │   ├── memory_gc.yaml
│   │   └── health_check.yaml
│   └── models/
│       ├── workflow.py            # Workflow* models
│       ├── execution.py           # Execution* models
│       └── step.py                # Step* models
```

**Benefits**:
- Workflows are declarative (YAML/JSON files)
- Workflows are versioned (v1, v2, etc.)
- Workflows are replayable (for debugging)
- Workflows are visualizable (graph view)

---

### 5. Observability Service - Add Distributed Tracing Layer

**Current State**: `services/observability/src/main.py` (database-only, no integration)

**Problems**:
- Observability service exists but not integrated
- No correlation ID propagation
- No OpenTelemetry integration
- No actual distributed tracing

**Refactor Plan**:

```
services/observability/
├── src/
│   ├── main.py                    # FastAPI app, routes (~150 lines)
│   ├── core/
│   │   ├── metrics_collector.py   # Collect metrics
│   │   ├── trace_collector.py     # Collect traces
│   │   ├── alert_manager.py       # Manage alerts
│   │   └── dashboard_generator.py # Generate dashboard data
│   ├── tracing/                   # NEW: Distributed tracing
│   │   ├── context.py             # Trace context management
│   │   ├── propagator.py          # Correlation ID propagation
│   │   ├── exporter.py            # Export to Jaeger/Zipkin
│   │   └── middleware.py          # FastAPI tracing middleware
│   ├── exporters/                 # NEW: Metric exporters
│   │   ├── prometheus.py          # Prometheus exporter
│   │   └── grafana.py             # Grafana datasource
│   └── models/
│       ├── metric.py              # Metric* models
│       ├── trace.py               # Trace* models
│       └── alert.py               # Alert* models
```

**Shared tracing library** (used by all services):
```
shared/libs/
├── tracing.py                     # NEW: Tracing utilities
│   ├── TraceContext              # Context object
│   ├── inject_trace_id()         # Inject trace ID into headers
│   ├── extract_trace_id()        # Extract trace ID from headers
│   ├── start_span()              # Start a new span
│   └── end_span()                # End a span
```

**Benefits**:
- All services can easily add tracing
- Correlation IDs propagated automatically
- Compatible with OpenTelemetry standard
- Can export to Jaeger, Zipkin, or custom backends

---

## Part B: What Must Be Modularized

### 1. Shared Libraries - Extract Common Patterns

**Current State**: `shared/libs/` has basic utilities

**Missing Patterns** (duplicated across services):
- Database connection management (duplicated in each service)
- HTTP client configuration (duplicated)
- Retry logic (duplicated)
- Rate limiting (duplicated)
- Caching (duplicated)

**Refactor Plan**:

```
shared/libs/
├── __init__.py
├── models.py                      # Pydantic models (existing)
├── config.py                      # Configuration (existing)
├── logger.py                      # Logging (existing)
├── middleware.py                  # FastAPI middleware (existing)
├── auth.py                        # JWT utilities (existing)
├── validators.py                  # Input validation (existing)
├── security.py                    # Security utilities (existing)
├── database.py                    # NEW: Database utilities
│   ├── ConnectionPool            # Pooled connections
│   ├── TransactionContext        # Transaction management
│   └── MigrationRunner           # Migration utilities
├── http_client.py                 # NEW: HTTP client utilities
│   ├── RetryableClient           # HTTP client with retries
│   ├── CircuitBreaker            # Circuit breaker pattern
│   └── RateLimiter               # Rate limiting
├── caching.py                     # NEW: Caching utilities
│   ├── RedisCache                # Redis cache wrapper
│   ├── MemoryCache               # In-memory cache
│   └── cache_decorator           # @cache decorator
├── tracing.py                     # NEW: Distributed tracing
│   ├── TraceContext              # Trace context
│   ├── inject_trace_id()         # Inject trace ID
│   ├── extract_trace_id()        # Extract trace ID
│   ├── start_span()              # Start span
│   └── end_span()                # End span
├── retry.py                       # NEW: Retry utilities
│   ├── retry_decorator           # @retry decorator
│   ├── exponential_backoff()     # Backoff strategy
│   └── RetryConfig               # Retry configuration
└── errors.py                      # NEW: Custom exceptions
    ├── ServiceUnavailableError
    ├── RateLimitExceededError
    ├── ValidationError
    └── AuthenticationError
```

**Example: Database Utilities**

**Before (V1)** - Duplicated in each service:
```python
# services/task-planner/src/main.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

engine = create_async_engine("postgresql://...")
async_session_maker = sessionmaker(engine, class_=AsyncSession)

async def get_task(task_id: UUID):
    async with async_session_maker() as session:
        result = await session.execute(select(Task).filter_by(id=task_id))
        return result.scalar_one_or_none()
```

**After (V2)** - Shared utility:
```python
# shared/libs/database.py
from contextlib import asynccontextmanager

class DatabaseConnection:
    def __init__(self, url: str):
        self.engine = create_async_engine(url, pool_size=10, max_overflow=20)
        self.session_maker = sessionmaker(self.engine, class_=AsyncSession)

    @asynccontextmanager
    async def session(self):
        async with self.session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

# services/task-planner/src/main.py
from shared.libs.database import DatabaseConnection

db = DatabaseConnection(config.DATABASE_URL)

async def get_task(task_id: UUID):
    async with db.session() as session:
        result = await session.execute(select(Task).filter_by(id=task_id))
        return result.scalar_one_or_none()
```

**Benefits**:
- No duplication
- Consistent error handling
- Easy to add features (e.g., connection pooling)
- Testable (mock DatabaseConnection)

---

### 2. Agent Strategies - Make Failure Handling Pluggable

**Current State**: Retry logic hardcoded in agent orchestrator

**Problem**: Can't easily change failure strategies per agent type

**Refactor Plan**:

```python
# services/agent-orchestrator/src/strategies/failure_strategies.py

from abc import ABC, abstractmethod

class FailureStrategy(ABC):
    @abstractmethod
    async def handle_failure(
        self,
        agent: Agent,
        task: Task,
        error: Exception
    ) -> FailureResult:
        pass

class RetryStrategy(FailureStrategy):
    def __init__(self, max_retries: int = 3, backoff_seconds: int = 5):
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    async def handle_failure(self, agent, task, error):
        if task.retry_count < self.max_retries:
            await asyncio.sleep(self.backoff_seconds * (2 ** task.retry_count))
            return FailureResult(action="retry", task=task)
        else:
            return FailureResult(action="fail", error=error)

class SkipStrategy(FailureStrategy):
    async def handle_failure(self, agent, task, error):
        return FailureResult(action="skip", reason="Non-critical task")

class ReassignStrategy(FailureStrategy):
    async def handle_failure(self, agent, task, error):
        new_agent = await self.find_replacement_agent(agent)
        return FailureResult(action="reassign", new_agent=new_agent)

class EscalateStrategy(FailureStrategy):
    async def handle_failure(self, agent, task, error):
        await self.notify_admin(agent, task, error)
        return FailureResult(action="escalate", notified=True)

# Usage
class ExecutorAgent(AgentBase):
    role = AgentRole.EXECUTOR
    failure_strategy = RetryStrategy(max_retries=3)

class CriticAgent(AgentBase):
    role = AgentRole.CRITIC
    failure_strategy = EscalateStrategy()  # Critical failures escalate
```

**Benefits**:
- Pluggable strategies (add new without changing core code)
- Testable (mock strategies)
- Configurable per agent type
- Clear separation of concerns

---

### 3. Tool Runner - Make Tool Execution Pluggable

**Current State**: Tools hardcoded in `tool_runner/src/main.py`

**Problem**: Adding new tools requires editing main file

**Refactor Plan**:

```python
# services/tool-runner/src/tools/base.py

from abc import ABC, abstractmethod
from pydantic import BaseModel

class ToolInput(BaseModel):
    pass

class ToolOutput(BaseModel):
    success: bool
    result: Any
    error: Optional[str] = None

class Tool(ABC):
    name: str
    description: str
    input_schema: Type[ToolInput]
    output_schema: Type[ToolOutput]
    permissions_required: List[str]

    @abstractmethod
    async def execute(self, input: ToolInput) -> ToolOutput:
        pass

# services/tool-runner/src/tools/python_executor.py
class PythonExecutorInput(ToolInput):
    code: str
    timeout: int = 30

class PythonExecutorOutput(ToolOutput):
    stdout: str
    stderr: str
    exit_code: int

class PythonExecutor(Tool):
    name = "execute_python"
    description = "Execute Python code in sandboxed environment"
    input_schema = PythonExecutorInput
    output_schema = PythonExecutorOutput
    permissions_required = ["code_execution"]

    async def execute(self, input: PythonExecutorInput) -> PythonExecutorOutput:
        # Sandboxed execution logic
        result = await run_in_docker("python", input.code, timeout=input.timeout)
        return PythonExecutorOutput(
            success=result.exit_code == 0,
            result=result.stdout,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code
        )

# services/tool-runner/src/registry.py
class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self.tools[name]

    def list(self) -> List[Tool]:
        return list(self.tools.values())

# Usage
registry = ToolRegistry()
registry.register(PythonExecutor())
registry.register(JavaScriptExecutor())
registry.register(HTTPRequest())

tool = registry.get("execute_python")
output = await tool.execute(PythonExecutorInput(code="print('hello')"))
```

**Benefits**:
- Tools are self-contained classes
- Easy to add new tools (just register)
- Input/output schemas enforced
- Permissions checked automatically

---

## Part C: What Must Be Abstracted

### 1. Agent Execution Pattern

**Current Pattern** (duplicated across agent types):
```python
# Duplicated in planner, executor, critic, summarizer
async def execute_task(task: Task):
    # 1. Get relevant memories
    memories = await memory_service.retrieve(task.user_id, task.description)

    # 2. Build prompt
    prompt = build_prompt(task, memories)

    # 3. Call LLM
    response = await ai_runtime.complete(prompt)

    # 4. Validate output
    validation = validate_output(response)

    # 5. Store result
    await store_result(task, response)

    # 6. Log to audit
    await audit_log(task, response)
```

**Abstracted Pattern**:
```python
# services/agent-orchestrator/src/core/agent_executor.py

class AgentExecutor:
    """Abstract agent execution pipeline"""

    async def execute(self, agent: Agent, task: Task) -> ExecutionResult:
        # Standard pipeline for all agents
        context = await self.gather_context(agent, task)
        prompt = await self.build_prompt(agent, task, context)
        response = await self.call_llm(agent, prompt)
        validated = await self.validate_output(agent, response)
        result = await self.store_result(agent, task, validated)
        await self.log_execution(agent, task, result)
        return result

    async def gather_context(self, agent: Agent, task: Task):
        """Override to customize context gathering"""
        return await memory_service.retrieve(task.user_id, task.description)

    async def build_prompt(self, agent: Agent, task: Task, context):
        """Override to customize prompt building"""
        return agent.prompt_template.format(task=task, context=context)

    async def call_llm(self, agent: Agent, prompt: str):
        """Standard LLM call (usually not overridden)"""
        return await ai_runtime.complete(
            role=agent.role,
            prompt=prompt,
            model=agent.model,
            temperature=agent.temperature
        )

    async def validate_output(self, agent: Agent, response):
        """Override to customize validation"""
        return await ai_quality_gate.validate(response, policy=agent.validation_policy)

    async def store_result(self, agent: Agent, task: Task, validated):
        """Standard result storage (usually not overridden)"""
        return await task_service.update_result(task.id, validated.content)

    async def log_execution(self, agent: Agent, task: Task, result):
        """Standard audit logging (usually not overridden)"""
        await audit_log.log(agent=agent, task=task, result=result)
```

**Usage**:
```python
# All agents use the same executor
executor = AgentExecutor()

# Execute any agent
result = await executor.execute(planner_agent, task)
result = await executor.execute(executor_agent, task)
result = await executor.execute(critic_agent, task)
```

**Benefits**:
- No duplication of execution logic
- Consistent error handling
- Consistent logging and auditing
- Easy to add new agents (just define prompt and validation)

---

### 2. Database Query Pattern

**Current Pattern** (duplicated across services):
```python
# Duplicated in task-planner, agent-orchestrator, memory-service, etc.
async def get_by_id(id: UUID):
    async with async_session_maker() as session:
        result = await session.execute(select(Model).filter_by(id=id))
        return result.scalar_one_or_none()

async def list_all():
    async with async_session_maker() as session:
        result = await session.execute(select(Model))
        return result.scalars().all()

async def create(data: dict):
    async with async_session_maker() as session:
        obj = Model(**data)
        session.add(obj)
        await session.commit()
        return obj
```

**Abstracted Pattern**:
```python
# shared/libs/database.py

class Repository(Generic[T]):
    """Generic repository for CRUD operations"""

    def __init__(self, db: DatabaseConnection, model: Type[T]):
        self.db = db
        self.model = model

    async def get_by_id(self, id: UUID) -> Optional[T]:
        async with self.db.session() as session:
            result = await session.execute(select(self.model).filter_by(id=id))
            return result.scalar_one_or_none()

    async def list(self, filters: dict = None, limit: int = 100) -> List[T]:
        async with self.db.session() as session:
            query = select(self.model)
            if filters:
                query = query.filter_by(**filters)
            query = query.limit(limit)
            result = await session.execute(query)
            return result.scalars().all()

    async def create(self, data: dict) -> T:
        async with self.db.session() as session:
            obj = self.model(**data)
            session.add(obj)
            await session.flush()
            await session.refresh(obj)
            return obj

    async def update(self, id: UUID, data: dict) -> Optional[T]:
        async with self.db.session() as session:
            obj = await self.get_by_id(id)
            if obj:
                for key, value in data.items():
                    setattr(obj, key, value)
                await session.flush()
                await session.refresh(obj)
            return obj

    async def delete(self, id: UUID) -> bool:
        async with self.db.session() as session:
            obj = await self.get_by_id(id)
            if obj:
                await session.delete(obj)
                return True
            return False

# Usage
task_repo = Repository(db, Task)
agent_repo = Repository(db, Agent)
memory_repo = Repository(db, Memory)

# All have same interface
task = await task_repo.get_by_id(task_id)
tasks = await task_repo.list(filters={"status": "pending"})
new_task = await task_repo.create({"title": "...", "user_id": "..."})
```

**Benefits**:
- No duplication of CRUD logic
- Consistent error handling
- Easy to add new models (just instantiate Repository)
- Type-safe (Generic[T])

---

### 3. Service Communication Pattern

**Current Pattern** (duplicated service-to-service HTTP calls):
```python
# Duplicated in every service that calls another service
async def call_memory_service(user_id: UUID, query: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://memory-service:8004/retrieve",
            json={"user_id": str(user_id), "query": query}
        )
        response.raise_for_status()
        return response.json()
```

**Abstracted Pattern**:
```python
# shared/libs/service_client.py

class ServiceClient:
    """HTTP client for service-to-service communication"""

    def __init__(self, base_url: str, retry_config: RetryConfig = None):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.retry_config = retry_config or RetryConfig(max_retries=3)

    async def get(self, path: str, **kwargs):
        return await self._request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs):
        return await self._request("POST", path, **kwargs)

    async def _request(self, method: str, path: str, **kwargs):
        url = f"{self.base_url}{path}"

        for attempt in range(self.retry_config.max_retries):
            try:
                # Inject trace ID for distributed tracing
                headers = kwargs.get("headers", {})
                trace_id = get_current_trace_id()
                headers["X-Trace-ID"] = trace_id
                kwargs["headers"] = headers

                response = await self.client.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < self.retry_config.max_retries - 1:
                    # Retry on 5xx errors
                    await asyncio.sleep(exponential_backoff(attempt))
                    continue
                raise

            except httpx.RequestError as e:
                if attempt < self.retry_config.max_retries - 1:
                    # Retry on network errors
                    await asyncio.sleep(exponential_backoff(attempt))
                    continue
                raise

        raise ServiceUnavailableError(f"Failed to call {url} after {self.retry_config.max_retries} retries")

# Usage
memory_client = ServiceClient("http://memory-service:8004")
ai_runtime_client = ServiceClient("http://ai-runtime:8005")

# All service calls have retry logic and tracing
memories = await memory_client.post("/retrieve", json={"user_id": str(user_id), "query": query})
completion = await ai_runtime_client.post("/complete", json={"role": "executor", "prompt": prompt})
```

**Benefits**:
- Automatic retries on transient failures
- Automatic trace ID propagation
- Consistent error handling
- Circuit breaker support (can be added)

---

## Part D: Folder Reorganization

### Current Structure (V1)
```
CognitionOS/
├── services/
│   ├── api-gateway/
│   ├── auth-service/
│   ├── task-planner/
│   ├── agent-orchestrator/
│   ├── memory-service/
│   ├── ai-runtime/
│   ├── tool-runner/
│   ├── audit-log/
│   ├── explainability/
│   └── observability/
├── shared/
│   └── libs/
├── database/
├── prompts/
├── frontend/
├── tests/
├── docs/
└── scripts/
```

### New Structure (V2)
```
CognitionOS/
├── services/                       # Microservices
│   ├── core/                       # V1 services
│   │   ├── api-gateway/
│   │   ├── auth-service/
│   │   ├── task-planner/
│   │   ├── agent-orchestrator/
│   │   ├── memory-service/
│   │   ├── ai-runtime/
│   │   ├── tool-runner/
│   │   ├── audit-log/
│   │   ├── explainability/
│   │   └── observability/
│   └── v2/                         # NEW: V2 services
│       ├── workflow-engine/
│       ├── ai-quality-gate/
│       └── developer-cli/
├── shared/                         # Shared libraries
│   ├── libs/                       # Common utilities
│   │   ├── models.py
│   │   ├── config.py
│   │   ├── logger.py
│   │   ├── middleware.py
│   │   ├── auth.py
│   │   ├── validators.py
│   │   ├── security.py
│   │   ├── database.py             # NEW
│   │   ├── http_client.py          # NEW
│   │   ├── caching.py              # NEW
│   │   ├── tracing.py              # NEW
│   │   ├── retry.py                # NEW
│   │   └── errors.py               # NEW
│   ├── agents/                     # NEW: Shared agent definitions
│   │   ├── base.py
│   │   ├── schemas.py
│   │   └── strategies.py
│   └── contracts/                  # NEW: Service contracts (Pydantic models)
│       ├── task_service.py
│       ├── agent_service.py
│       ├── memory_service.py
│       └── ai_runtime.py
├── database/
│   ├── models.py
│   ├── connection.py
│   ├── migrations/
│   │   ├── v1/                     # V1 migrations
│   │   │   └── 001_initial_schema.sql
│   │   └── v2/                     # NEW: V2 migrations
│       │   ├── 002_workflow_tables.sql
│       │   ├── 003_agent_metrics.sql
│       │   ├── 004_memory_namespaces.sql
│       │   └── 005_quality_gate_tables.sql
│   └── repositories/               # NEW: Database repositories
│       ├── task_repository.py
│       ├── agent_repository.py
│       └── memory_repository.py
├── workflows/                      # NEW: Workflow definitions
│   ├── agent_training.yaml
│   ├── memory_gc.yaml
│   ├── health_check.yaml
│   └── templates/
│       └── workflow_template.yaml
├── prompts/
│   └── versioned/
│       ├── planner/
│       │   ├── v1.md
│       │   └── v2.md               # NEW: Improved prompts
│       ├── executor/
│       ├── critic/
│       └── summarizer/
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   │   ├── dashboards/         # V1 components
│   │   │   └── visualizations/     # NEW: V2 visualizations
│   │   │       ├── AgentGraph.tsx
│   │   │       ├── WorkflowTimeline.tsx
│   │   │       ├── MemoryHeatmap.tsx
│   │   │       ├── FailureDebugger.tsx
│   │   │       └── WhyPanel.tsx
│   │   └── ...
├── tools/                          # NEW: Developer tools
│   ├── dev-cli/
│   │   ├── cognos.py               # Main CLI entry
│   │   ├── commands/
│   │   │   ├── generate.py
│   │   │   ├── lint.py
│   │   │   ├── arch.py
│   │   │   ├── test.py
│   │   │   └── debug.py
│   │   └── templates/
│   │       ├── agent_template.py
│   │       ├── workflow_template.yaml
│   │       └── tool_template.py
│   ├── linters/
│   │   ├── agent_linter.py
│   │   ├── workflow_linter.py
│   │   └── arch_checker.py
│   └── generators/
│       ├── agent_generator.py
│       ├── workflow_generator.py
│       └── tool_generator.py
├── tests/
│   ├── unit/                       # NEW: Unit tests
│   │   ├── services/
│   │   ├── agents/
│   │   └── workflows/
│   ├── integration/
│   │   └── test_integration.py
│   ├── workflows/                  # NEW: Workflow tests
│   │   ├── test_agent_training.py
│   │   └── test_memory_gc.py
│   ├── simulations/                # NEW: Agent simulation tests
│   │   ├── test_planner_agent.py
│   │   └── test_executor_agent.py
│   └── chaos/                      # NEW: Chaos tests
│       ├── test_service_failures.py
│       └── test_network_partition.py
├── docs/
│   ├── verification/               # NEW: V2 verification
│   │   ├── architecture_audit.md
│   │   └── plan_vs_reality.md
│   ├── v2/                         # NEW: V2 planning
│   │   ├── expansion_plan.md
│   │   ├── refactor_plan.md
│   │   └── performance_plan.md
│   ├── architecture.md
│   ├── agent_model.md
│   ├── memory_model.md
│   ├── security.md
│   ├── deployment.md
│   └── contributing.md             # NEW: Developer guide
└── scripts/
    ├── init_database.py
    ├── run_migrations.py           # NEW: Migration runner
    ├── seed_data.py                # NEW: Seed test data
    └── health_check.py             # NEW: Health checker
```

---

## Summary

### Files to Split (5 major splits)
1. ✅ Agent Orchestrator → registry, scheduler, health_monitor, agents/, strategies/
2. ✅ Memory Service → storage, lifecycle, jobs/, debug/
3. ✅ AI Runtime → ai-runtime (LLM only) + ai-quality-gate (NEW service)
4. ✅ Task Planner → task-planner + workflow-engine (NEW service)
5. ✅ Observability → core + tracing/ + exporters/

### Patterns to Modularize (3 major patterns)
1. ✅ Failure handling → Pluggable FailureStrategy classes
2. ✅ Tool execution → Pluggable Tool registry
3. ✅ Assignment logic → Pluggable AssignmentStrategy classes

### Logic to Abstract (3 major abstractions)
1. ✅ Agent execution pipeline → AgentExecutor class
2. ✅ Database operations → Repository[T] generic class
3. ✅ Service communication → ServiceClient with retry/tracing

### Folders to Reorganize (2 major changes)
1. ✅ services/ → services/core/ (V1) + services/v2/ (NEW)
2. ✅ shared/ → shared/libs/ + shared/agents/ + shared/contracts/

---

**Next Document**: `performance_plan.md` - Bottleneck analysis and optimization strategies
