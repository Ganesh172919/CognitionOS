# CognitionOS V2 - Detailed Implementation Plan

**Document Version**: 1.0
**Date**: 2026-02-11
**Status**: Active Implementation
**Priority**: CRITICAL

---

## Executive Summary

This document outlines the detailed implementation plan for completing CognitionOS V1 pending tasks and implementing V2 features as defined in the expansion and refactor plans.

**Current Status:**
- V1: 75% complete (core services operational)
- V2: 10% complete (infrastructure exists but not fully implemented)

**Estimated Timeline:**
- Phase 1 (Critical): 10-12 days
- Phase 2 (High Priority): 4-6 days
- Phase 3 (Medium Priority): 9-12 days
- **Total: 23-30 days**

---

## Implementation Phases

### PHASE 1: Critical Path (Milestone 1)

#### 1.1 Complete Workflow Engine Service (2-3 days)

**Current State:**
- Models defined: `/services/workflow-engine/src/models/__init__.py` ✅
- DSL Parser implemented: `/services/workflow-engine/src/core/dsl_parser.py` ✅
- Executor implemented: `/services/workflow-engine/src/core/executor.py` ✅
- **Missing**: FastAPI service, database persistence, HTTP endpoints

**Tasks:**
1. Create `/services/workflow-engine/src/main.py` with FastAPI app
2. Implement HTTP endpoints:
   - `POST /workflows` - Create workflow definition
   - `GET /workflows/:id` - Get workflow definition
   - `GET /workflows/:id/versions` - List workflow versions
   - `POST /workflows/:id/execute` - Execute workflow
   - `GET /executions/:id` - Get execution status
   - `POST /executions/:id/replay` - Replay execution
   - `GET /executions/:id/graph` - Get execution graph data
3. Create database persistence layer:
   - Workflow repository (CRUD operations)
   - Execution repository
   - Step execution repository
4. Add service clients for integration:
   - Agent Orchestrator client
   - Memory Service client
   - AI Runtime client
   - Tool Runner client
5. Create `requirements.txt` with dependencies
6. Create `Dockerfile` for containerization
7. Add health check endpoint
8. Create service README

**Deliverables:**
- Working FastAPI service on port 8010
- Database persistence for workflows and executions
- Sample workflow execution via API
- Integration with existing services

---

#### 1.2 Create V2 Database Migrations (1 day)

**Current State:**
- Only `001_initial_schema.sql` exists
- No V2 tables in database

**Tasks:**
1. Create `database/migrations/v2/002_workflow_tables.sql`:
   ```sql
   CREATE TABLE workflows (
       id VARCHAR(200) PRIMARY KEY,
       version VARCHAR(50) NOT NULL,
       name VARCHAR(200) NOT NULL,
       description TEXT,
       definition JSONB NOT NULL,
       schedule VARCHAR(100),
       created_by UUID REFERENCES users(id),
       created_at TIMESTAMP NOT NULL DEFAULT NOW(),
       is_active BOOLEAN DEFAULT TRUE,
       UNIQUE(id, version)
   );

   CREATE TABLE workflow_executions (
       id UUID PRIMARY KEY,
       workflow_id VARCHAR(200) NOT NULL,
       workflow_version VARCHAR(50) NOT NULL,
       inputs JSONB NOT NULL,
       outputs JSONB,
       status VARCHAR(50) NOT NULL,
       started_at TIMESTAMP,
       completed_at TIMESTAMP,
       error TEXT,
       user_id UUID REFERENCES users(id),
       created_at TIMESTAMP NOT NULL DEFAULT NOW()
   );

   CREATE TABLE workflow_execution_steps (
       id UUID PRIMARY KEY,
       execution_id UUID REFERENCES workflow_executions(id) ON DELETE CASCADE,
       step_id VARCHAR(200) NOT NULL,
       step_type VARCHAR(100) NOT NULL,
       status VARCHAR(50) NOT NULL,
       started_at TIMESTAMP,
       completed_at TIMESTAMP,
       output JSONB,
       error TEXT,
       agent_id UUID REFERENCES agents(id),
       retry_count INTEGER DEFAULT 0,
       created_at TIMESTAMP NOT NULL DEFAULT NOW()
   );

   CREATE INDEX idx_workflow_executions_workflow_id ON workflow_executions(workflow_id);
   CREATE INDEX idx_workflow_executions_user_id ON workflow_executions(user_id);
   CREATE INDEX idx_workflow_executions_status ON workflow_executions(status);
   CREATE INDEX idx_workflow_execution_steps_execution_id ON workflow_execution_steps(execution_id);
   ```

2. Create `database/migrations/v2/003_agent_metrics.sql`:
   ```sql
   CREATE TABLE agent_metrics (
       id UUID PRIMARY KEY,
       agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
       time_window_start TIMESTAMP NOT NULL,
       time_window_end TIMESTAMP NOT NULL,
       task_count INTEGER NOT NULL,
       avg_confidence FLOAT,
       avg_quality_score FLOAT,
       hallucination_rate FLOAT,
       avg_latency_ms INTEGER,
       p95_latency_ms INTEGER,
       p99_latency_ms INTEGER,
       avg_cost_per_task FLOAT,
       total_tokens_used BIGINT,
       success_rate FLOAT,
       retry_rate FLOAT,
       failure_rate FLOAT,
       created_at TIMESTAMP NOT NULL DEFAULT NOW()
   );

   CREATE INDEX idx_agent_metrics_agent_id ON agent_metrics(agent_id);
   CREATE INDEX idx_agent_metrics_time_window ON agent_metrics(time_window_start, time_window_end);
   ```

3. Create `database/migrations/v2/004_memory_namespaces.sql`:
   ```sql
   CREATE TABLE memory_namespaces (
       id UUID PRIMARY KEY,
       name VARCHAR(200) NOT NULL UNIQUE,
       description TEXT,
       owner_user_id UUID REFERENCES users(id),
       created_at TIMESTAMP NOT NULL DEFAULT NOW()
   );

   CREATE TABLE memory_lifecycle_policies (
       id UUID PRIMARY KEY,
       name VARCHAR(200) NOT NULL UNIQUE,
       namespace_id UUID REFERENCES memory_namespaces(id),
       ttl_days INTEGER,
       compression_after_days INTEGER,
       archive_after_days INTEGER,
       min_access_frequency INTEGER,
       created_at TIMESTAMP NOT NULL DEFAULT NOW()
   );

   -- Add namespace_id to memories table
   ALTER TABLE memories ADD COLUMN namespace_id UUID REFERENCES memory_namespaces(id);
   ALTER TABLE memories ADD COLUMN compressed BOOLEAN DEFAULT FALSE;
   ALTER TABLE memories ADD COLUMN archived BOOLEAN DEFAULT FALSE;

   CREATE INDEX idx_memories_namespace_id ON memories(namespace_id);
   CREATE INDEX idx_memories_compressed ON memories(compressed);
   CREATE INDEX idx_memories_archived ON memories(archived);
   ```

4. Create `database/migrations/v2/005_quality_gate_tables.sql`:
   ```sql
   CREATE TABLE quality_gate_policies (
       id UUID PRIMARY KEY,
       name VARCHAR(200) NOT NULL UNIQUE,
       min_quality_score FLOAT DEFAULT 0.7,
       min_confidence_score FLOAT DEFAULT 0.7,
       require_cross_verification BOOLEAN DEFAULT FALSE,
       require_self_critique BOOLEAN DEFAULT FALSE,
       created_at TIMESTAMP NOT NULL DEFAULT NOW()
   );

   CREATE TABLE quality_gate_results (
       id UUID PRIMARY KEY,
       task_id UUID REFERENCES tasks(id),
       agent_id UUID REFERENCES agents(id),
       policy_id UUID REFERENCES quality_gate_policies(id),
       passed BOOLEAN NOT NULL,
       quality_score FLOAT,
       confidence_score FLOAT,
       failure_reason TEXT,
       checked_at TIMESTAMP NOT NULL DEFAULT NOW()
   );

   CREATE INDEX idx_quality_gate_results_task_id ON quality_gate_results(task_id);
   CREATE INDEX idx_quality_gate_results_agent_id ON quality_gate_results(agent_id);
   CREATE INDEX idx_quality_gate_results_passed ON quality_gate_results(passed);
   ```

5. Create migration runner script `scripts/run_v2_migrations.py`
6. Update `database/models.py` with new V2 models

**Deliverables:**
- 4 new migration files
- Migration runner script
- Updated SQLAlchemy models
- Migrations tested and applied

---

#### 1.3 Implement Typed Agent System (3-4 days)

**Current State:**
- Agents are database records only
- No typed classes or Pydantic schemas

**Tasks:**
1. Create `/shared/libs/agents/` directory structure:
   ```
   shared/libs/agents/
   ├── __init__.py
   ├── base.py           # AgentBase class
   ├── schemas.py        # Input/Output schemas
   ├── strategies.py     # Failure strategies
   └── roles.py          # Role-specific agents
   ```

2. Implement `base.py`:
   ```python
   from abc import ABC, abstractmethod
   from pydantic import BaseModel
   from typing import Type, List, Optional
   from enum import Enum

   class AgentRole(str, Enum):
       PLANNER = "planner"
       EXECUTOR = "executor"
       CRITIC = "critic"
       SUMMARIZER = "summarizer"

   class FailureStrategy(str, Enum):
       RETRY = "retry"
       SKIP = "skip"
       REASSIGN = "reassign"
       ESCALATE = "escalate"

   class AgentBase(ABC):
       role: AgentRole
       model: str
       temperature: float
       input_schema: Type[BaseModel]
       output_schema: Type[BaseModel]
       allowed_tools: List[str]
       failure_strategy: FailureStrategy
       max_retries: int
       retry_delay_seconds: int
       confidence_threshold: float

       @abstractmethod
       async def execute(self, input: BaseModel) -> BaseModel:
           pass
   ```

3. Implement `schemas.py` with agent-specific schemas:
   ```python
   # Planner schemas
   class PlannerAgentInput(BaseModel):
       goal: str
       context: Optional[str] = None
       constraints: List[str] = []

   class PlannerAgentOutput(BaseModel):
       plan: List[str]
       reasoning: str
       confidence: float
       estimated_duration: str

   # Executor schemas
   class ExecutorAgentInput(BaseModel):
       task_description: str
       context: Optional[str] = None
       tools_allowed: List[str] = []

   class ExecutorAgentOutput(BaseModel):
       result: str
       tools_used: List[str]
       confidence: float
       errors: List[str] = []

   # Critic schemas
   class CriticAgentInput(BaseModel):
       content: str
       criteria: List[str]
       context: Optional[str] = None

   class CriticAgentOutput(BaseModel):
       issues: List[str]
       suggestions: List[str]
       confidence: float
       overall_quality: float

   # Summarizer schemas
   class SummarizerAgentInput(BaseModel):
       content: str
       max_length: Optional[int] = None
       style: str = "concise"

   class SummarizerAgentOutput(BaseModel):
       summary: str
       key_points: List[str]
       confidence: float
   ```

4. Implement `roles.py` with concrete agent classes:
   ```python
   class PlannerAgent(AgentBase):
       role = AgentRole.PLANNER
       model = "gpt-4"
       temperature = 0.3
       input_schema = PlannerAgentInput
       output_schema = PlannerAgentOutput
       allowed_tools = ["search", "retrieve_memory"]
       failure_strategy = FailureStrategy.RETRY
       max_retries = 3
       confidence_threshold = 0.7

   class ExecutorAgent(AgentBase):
       role = AgentRole.EXECUTOR
       model = "gpt-4"
       temperature = 0.3
       input_schema = ExecutorAgentInput
       output_schema = ExecutorAgentOutput
       allowed_tools = ["execute_python", "execute_javascript", "http_request"]
       failure_strategy = FailureStrategy.RETRY
       max_retries = 3
       confidence_threshold = 0.7

   # Similar for Critic and Summarizer agents
   ```

5. Implement `strategies.py` with failure handling strategies:
   ```python
   class FailureHandler(ABC):
       @abstractmethod
       async def handle_failure(self, agent: AgentBase, task, error: Exception):
           pass

   class RetryStrategy(FailureHandler):
       async def handle_failure(self, agent, task, error):
           # Retry logic with exponential backoff
           pass

   class SkipStrategy(FailureHandler):
       async def handle_failure(self, agent, task, error):
           # Skip task and continue
           pass

   class ReassignStrategy(FailureHandler):
       async def handle_failure(self, agent, task, error):
           # Assign to different agent
           pass

   class EscalateStrategy(FailureHandler):
       async def handle_failure(self, agent, task, error):
           # Notify admin
           pass
   ```

6. Update `agent-orchestrator` service to use typed agents
7. Add agent performance metrics collection
8. Implement agent replacement logic

**Deliverables:**
- Typed agent system in shared/libs/agents/
- Agent-specific input/output schemas
- Failure strategy implementations
- Updated agent orchestrator
- Agent performance metrics

---

#### 1.4 Add Memory Lifecycle Management (3-4 days)

**Current State:**
- Memory storage and retrieval working
- No lifecycle management or cleanup

**Tasks:**
1. Create `/services/memory-service/src/core/lifecycle.py`:
   ```python
   class MemoryLifecycleManager:
       async def apply_policies(self, namespace_id: UUID):
           # Apply lifecycle policies to memories
           pass

       async def identify_stale_memories(self, ttl_days: int):
           # Find memories older than TTL
           pass

       async def archive_memories(self, memory_ids: List[UUID]):
           # Archive to cold storage
           pass

       async def compress_memories(self, memory_ids: List[UUID]):
           # Compress old memories
           pass
   ```

2. Create `/services/memory-service/src/core/garbage_collector.py`:
   ```python
   class MemoryGarbageCollector:
       async def collect(self, namespace_id: Optional[UUID] = None):
           # Run GC on memories
           pass

       async def calculate_memory_usage(self):
           # Calculate total memory usage
           pass
   ```

3. Create `/services/memory-service/src/core/namespace_manager.py`:
   ```python
   class NamespaceManager:
       async def create_namespace(self, name: str, owner_id: UUID):
           # Create memory namespace
           pass

       async def get_namespace(self, namespace_id: UUID):
           # Get namespace details
           pass

       async def list_namespaces(self, owner_id: Optional[UUID] = None):
           # List namespaces
           pass
   ```

4. Create `/services/memory-service/src/jobs/` directory:
   ```
   jobs/
   ├── __init__.py
   ├── gc_daily.py       # Daily GC job
   ├── compress_old.py   # Compression job
   └── archive_stale.py  # Archival job
   ```

5. Implement background job scheduler
6. Add namespace endpoints to memory service
7. Add lifecycle policy endpoints
8. Add GC trigger endpoint

**Deliverables:**
- Memory lifecycle management
- Namespace support
- Garbage collection
- Compression logic
- Background jobs
- Updated memory service endpoints

---

### PHASE 2: High Priority (Milestone 2)

#### 2.1 Create AI Quality Gate Service (2-3 days)

**Tasks:**
1. Create service directory: `/services/ai-quality-gate/`
2. Move `output_validator.py` from ai-runtime to quality gate
3. Create `main.py` with FastAPI routes
4. Implement cross-agent verification
5. Implement self-critique loops
6. Add policy enforcement
7. Create Docker configuration

**Deliverables:**
- New AI Quality Gate service on port 8011
- Cross-agent verification
- Self-critique functionality
- Policy-based validation

---

#### 2.2 Implement Distributed Tracing (2-3 days)

**Tasks:**
1. Install OpenTelemetry SDK
2. Create tracing utilities in shared/libs/tracing.py
3. Add automatic instrumentation to all services
4. Configure Jaeger/Zipkin exporter
5. Update observability service to accept OTel traces
6. Add trace visualization endpoints

**Deliverables:**
- OpenTelemetry integration
- Distributed tracing across all services
- Jaeger integration
- Trace visualization

---

### PHASE 3: Medium Priority (Milestone 3)

#### 3.1 Create Developer CLI Service (4-5 days)

**Tasks:**
1. Create `/services/developer-cli/` service
2. Implement code generators (agents, workflows, tools)
3. Implement linters
4. Implement architecture checkers
5. Create CLI commands

**Deliverables:**
- Developer CLI service on port 8012
- Code generation tools
- Linting tools
- Architecture enforcement

---

#### 3.2 Create Sample Workflow Files (1 day)

**Tasks:**
1. Create `/workflows/` directory
2. Create `agent_training.yaml`
3. Create `memory_gc.yaml`
4. Create `health_check.yaml`
5. Create `deploy_web_app.yaml` (example from expansion plan)

**Deliverables:**
- 4+ sample workflow YAML files
- Documentation for each workflow

---

#### 3.3 Add Integration Tests (2 days)

**Tasks:**
1. Create workflow execution tests
2. Create typed agent tests
3. Create memory lifecycle tests
4. Create distributed tracing tests

**Deliverables:**
- Comprehensive integration test suite
- CI/CD pipeline updates

---

#### 3.4 Update Infrastructure (1 day)

**Tasks:**
1. Update `docker-compose.yml` with new services
2. Update service ports configuration
3. Create startup scripts
4. Update deployment documentation

**Deliverables:**
- Updated docker-compose.yml
- All services can start together
- Deployment guide updated

---

## Success Criteria

### Phase 1 Complete When:
- [ ] Workflow engine can execute workflows via HTTP API
- [ ] V2 database migrations applied successfully
- [ ] Agents have typed input/output schemas
- [ ] Memory lifecycle management operational
- [ ] All services start without errors

### Phase 2 Complete When:
- [ ] AI Quality Gate service operational
- [ ] Distributed tracing visible in Jaeger
- [ ] Cross-service traces captured correctly

### Phase 3 Complete When:
- [ ] Developer CLI can generate code
- [ ] Sample workflows execute successfully
- [ ] Integration tests pass
- [ ] docker-compose.yml starts all services

---

## Timeline

**Week 1 (Days 1-5):**
- Complete Workflow Engine service
- Create V2 database migrations
- Start typed agent system

**Week 2 (Days 6-10):**
- Complete typed agent system
- Implement memory lifecycle management
- Start AI Quality Gate service

**Week 3 (Days 11-15):**
- Complete AI Quality Gate service
- Implement distributed tracing
- Start Developer CLI

**Week 4 (Days 16-20):**
- Complete Developer CLI
- Create sample workflows
- Add integration tests

**Week 5 (Days 21-25):**
- Infrastructure updates
- Testing and bug fixes
- Documentation updates

---

## Risk Mitigation

**Risk 1: Integration Complexity**
- Mitigation: Start with workflow engine as it's partially complete
- Fallback: Implement services independently first, integrate later

**Risk 2: Database Migration Issues**
- Mitigation: Test migrations on dev database first
- Fallback: Have rollback scripts ready

**Risk 3: Breaking Changes**
- Mitigation: Keep V1 services running, add V2 features incrementally
- Fallback: Feature flags to disable V2 features if issues arise

---

## Notes

- All code must follow existing patterns in the codebase
- Use Pydantic models for all API contracts
- Add comprehensive logging and error handling
- Update documentation as features are implemented
- Commit frequently with clear messages
- Run tests before committing

---

**End of Implementation Plan**
