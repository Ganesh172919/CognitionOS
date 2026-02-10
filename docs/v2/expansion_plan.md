# CognitionOS V2 - System Expansion Plan

**Document Version**: 1.0
**Date**: 2026-02-10
**Purpose**: Define new services, workflows, UI surfaces, and agent capabilities for V2
**Status**: Planning Phase

---

## Executive Summary

V2 transforms CognitionOS from a **functional AI system** into a **production-grade AI operating system** by:
1. **Formalizing workflows** - From implicit code to explicit DSL
2. **Typing agents** - From database records to software entities
3. **Managing memory** - From storage to lifecycle
4. **Visualizing cognition** - From dashboard to debugger
5. **Testing systematically** - From manual to automated
6. **Empowering developers** - From code-only to tooling-assisted

---

## V2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND LAYER (V2)                           │
│  ┌────────────────┐  ┌─────────────────┐  ┌────────────────────┐   │
│  │ Agent Graph    │  │ Workflow Graph  │  │ Memory Heatmap     │   │
│  │ Visualizer     │  │ Timeline        │  │ Inspector          │   │
│  └────────────────┘  └─────────────────┘  └────────────────────┘   │
│  ┌────────────────┐  ┌─────────────────┐  ┌────────────────────┐   │
│  │ Failure        │  │ "Why?" Debug    │  │ Power User         │   │
│  │ Debugger       │  │ Panel           │  │ Mode               │   │
│  └────────────────┘  └─────────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────────┐
│                      NEW SERVICES (V2)                               │
│  ┌────────────────┐  ┌─────────────────┐  ┌────────────────────┐   │
│  │ Workflow Engine│  │ AI Quality Gate │  │ Developer CLI      │   │
│  │ (Port 8010)    │  │ (Port 8011)     │  │ (Port 8012)        │   │
│  └────────────────┘  └─────────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────────┐
│                    ENHANCED SERVICES (V2)                            │
│  ┌────────────────┐  ┌─────────────────┐  ┌────────────────────┐   │
│  │ Agent          │  │ Memory Service  │  │ Observability      │   │
│  │ Orchestrator   │  │ V2              │  │ V2                 │   │
│  │ + Typed Agents │  │ + Lifecycle     │  │ + Distributed      │   │
│  │                │  │ + GC + Namespaces│ │   Tracing          │   │
│  └────────────────┘  └─────────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part A: New Services

### 1. Workflow Engine Service (Port 8010)

**Purpose**: Formalize workflows from implicit code logic to explicit, versioned, replayable definitions

#### Features

**Core Functionality**:
- Workflow definition language (YAML/JSON DSL)
- Workflow versioning (v1, v2, etc.)
- Workflow validation (DAG cycles, missing dependencies)
- Workflow execution engine
- Workflow replay (re-run historical executions)
- Workflow rollback (revert to previous version)

**Workflow DSL Schema**:
```yaml
workflow:
  id: "deploy-web-app-v2"
  version: "2.0.0"
  description: "Deploy web application with tests"

  inputs:
    - name: repo_url
      type: string
      required: true
    - name: environment
      type: enum
      values: [dev, staging, prod]
      default: dev

  outputs:
    - name: deployment_url
      type: string
    - name: deployment_status
      type: enum
      values: [success, partial, failed]

  steps:
    - id: clone_repo
      type: git_clone
      params:
        url: ${{ inputs.repo_url }}
      agent_role: executor
      timeout: 60s

    - id: run_tests
      type: execute_python
      depends_on: [clone_repo]
      params:
        script: pytest tests/
      agent_role: executor
      retry: 3

    - id: build_docker
      type: docker_build
      depends_on: [run_tests]
      params:
        dockerfile: Dockerfile
      agent_role: executor

    - id: deploy
      type: kubernetes_apply
      depends_on: [build_docker]
      params:
        manifest: k8s/deployment.yaml
        environment: ${{ inputs.environment }}
      agent_role: executor
      approval_required: true  # For prod
```

**Execution Graph Visualization**:
- DAG visualization with D3.js
- Step status (pending, running, completed, failed, skipped)
- Real-time progress updates
- Step dependencies clearly shown
- Parallel execution paths highlighted

**Workflow Replay**:
- Save workflow execution state at each step
- Replay from any step
- Replay with different parameters
- Compare replay vs original execution

**API Endpoints**:
- `POST /workflows` - Create workflow definition
- `GET /workflows/:id` - Get workflow definition
- `GET /workflows/:id/versions` - List workflow versions
- `POST /workflows/:id/execute` - Execute workflow
- `GET /executions/:id` - Get execution status
- `POST /executions/:id/replay` - Replay execution
- `POST /executions/:id/rollback` - Rollback to previous version
- `GET /executions/:id/graph` - Get execution graph data

**Database Schema**:
```sql
CREATE TABLE workflows (
    id UUID PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    version VARCHAR(50) NOT NULL,
    definition JSONB NOT NULL,  -- Full workflow DSL
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(name, version)
);

CREATE TABLE workflow_executions (
    id UUID PRIMARY KEY,
    workflow_id UUID REFERENCES workflows(id),
    workflow_version VARCHAR(50) NOT NULL,
    inputs JSONB NOT NULL,
    status VARCHAR(50) NOT NULL,  -- pending, running, completed, failed
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_by UUID REFERENCES users(id)
);

CREATE TABLE workflow_execution_steps (
    id UUID PRIMARY KEY,
    execution_id UUID REFERENCES workflow_executions(id),
    step_id VARCHAR(200) NOT NULL,
    status VARCHAR(50) NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    output JSONB,
    error TEXT,
    agent_id UUID REFERENCES agents(id)
);
```

**Implementation Priority**: **CRITICAL** (Milestone 1)

---

### 2. AI Quality Gate Service (Port 8011)

**Purpose**: Centralized quality firewall between AI output and execution

#### Features

**Output Validation Pipeline**:
1. Schema validation (Pydantic models)
2. Content validation (output_validator.py logic)
3. Cross-agent verification
4. Self-critique loops
5. Confidence thresholding

**Cross-Agent Verification**:
```python
# Example: Executor generates code, Critic verifies
executor_output = ai_runtime.complete(role="executor", prompt=task)
critic_output = ai_runtime.complete(
    role="critic",
    prompt=f"Review this code:\n{executor_output.content}"
)

if critic_output.confidence < 0.7:
    # Fail quality gate
    return QualityGateResult(
        passed=False,
        reason="Critic confidence too low",
        recommendation="Regenerate with different prompt"
    )
```

**Self-Critique Loop**:
```python
# Agent critiques its own output
output_v1 = ai_runtime.complete(role="executor", prompt=task)
critique = ai_runtime.complete(
    role="critic",
    prompt=f"Find flaws in this:\n{output_v1.content}"
)

if critique.has_critical_flaws:
    # Re-generate with critique feedback
    output_v2 = ai_runtime.complete(
        role="executor",
        prompt=f"{task}\n\nPrevious attempt flaws:\n{critique.flaws}\nImprove."
    )
```

**Quality Metrics**:
- Overall quality score (0-1)
- Confidence score (0-1)
- Completeness (0-1)
- Clarity (0-1)
- Coherence (0-1)
- Safety score (0-1)

**API Endpoints**:
- `POST /validate` - Validate AI output
- `POST /verify` - Cross-agent verification
- `POST /critique` - Self-critique loop
- `GET /policies` - Get validation policies
- `POST /policies` - Create validation policy

**Database Schema**:
```sql
CREATE TABLE quality_gate_policies (
    id UUID PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
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
```

**Implementation Priority**: **HIGH** (Milestone 2)

---

### 3. Developer CLI Service (Port 8012)

**Purpose**: Empower developers with CLI tools for code generation, linting, and architecture enforcement

#### Features

**CLI Commands**:
```bash
# Generate new agent
cognos generate agent --name CodeReviewAgent --role critic

# Generate new workflow
cognos generate workflow --name deploy-ml-model

# Generate new tool
cognos generate tool --name slack_notify --language python

# Lint codebase
cognos lint --check agents
cognos lint --check workflows
cognos lint --check architecture

# Architecture enforcement
cognos arch check  # Verify no violations
cognos arch report  # Generate architecture report

# Testing
cognos test workflow --workflow deploy-web-app
cognos test agent --agent PlannerAgent
cognos test chaos --duration 60s

# Debugging
cognos debug workflow --execution-id abc-123
cognos debug agent --agent-id def-456
cognos debug memory --user-id ghi-789
```

**Code Generators**:
1. **Agent Generator**:
   ```bash
   cognos generate agent --name SecurityAuditor --role critic
   ```
   Generates:
   - `services/agent-orchestrator/agents/security_auditor.py`
   - Pydantic input/output schemas
   - Default prompt template
   - Test file

2. **Workflow Generator**:
   ```bash
   cognos generate workflow --name ml-training-pipeline
   ```
   Generates:
   - `workflows/ml_training_pipeline.yaml`
   - Input/output schema
   - Sample test cases

3. **Tool Generator**:
   ```bash
   cognos generate tool --name jira_create_issue
   ```
   Generates:
   - `services/tool-runner/src/tools/jira_create_issue.py`
   - Pydantic param schema
   - Permission requirements
   - Test file

**Linting Rules**:
- Agent schema validation
- Workflow DAG cycle detection
- Tool permission enforcement
- Memory scope validation
- Cross-service dependency checking

**Architecture Enforcement**:
```python
# Example: Detect violations
class ArchitectureChecker:
    def check_service_dependencies(self):
        """Ensure no circular service dependencies"""

    def check_agent_schemas(self):
        """Ensure all agents have input/output schemas"""

    def check_workflow_completeness(self):
        """Ensure all workflows have error handling"""

    def check_memory_scopes(self):
        """Ensure memory operations use proper scopes"""
```

**API Endpoints**:
- `POST /generate/agent` - Generate agent code
- `POST /generate/workflow` - Generate workflow definition
- `POST /generate/tool` - Generate tool code
- `POST /lint/agents` - Lint all agents
- `POST /lint/workflows` - Lint all workflows
- `GET /architecture/report` - Get architecture report
- `POST /architecture/check` - Check for violations

**Implementation Priority**: **MEDIUM** (Milestone 3)

---

## Part B: New Workflows

### 1. Agent Training Workflow

**Purpose**: Train agents on historical executions to improve performance

```yaml
workflow:
  id: "agent-training-pipeline"
  version: "1.0.0"

  steps:
    - id: collect_executions
      type: query_database
      params:
        query: "SELECT * FROM task_execution_logs WHERE status='completed' LIMIT 1000"

    - id: extract_patterns
      type: execute_python
      params:
        script: analyze_successful_executions.py

    - id: generate_prompt_variations
      type: ai_generate
      agent_role: planner
      params:
        prompt: "Generate 5 variations of this prompt: ..."

    - id: a_b_test
      type: prompt_ab_test
      params:
        variations: ${{ steps.generate_prompt_variations.output }}
        duration: 24h

    - id: promote_winner
      type: promote_prompt
      params:
        winner_id: ${{ steps.a_b_test.winner }}
```

---

### 2. Memory Garbage Collection Workflow

**Purpose**: Clean up stale memories, compress old memories, archive inactive memories

```yaml
workflow:
  id: "memory-gc-daily"
  version: "1.0.0"
  schedule: "0 2 * * *"  # 2 AM daily

  steps:
    - id: identify_stale
      type: query_database
      params:
        query: "SELECT id FROM memories WHERE accessed_at < NOW() - INTERVAL '90 days'"

    - id: archive_stale
      type: archive_memories
      params:
        memory_ids: ${{ steps.identify_stale.output }}
        destination: s3://cognos-archive/memories/

    - id: identify_compressible
      type: query_database
      params:
        query: "SELECT id FROM memories WHERE created_at < NOW() - INTERVAL '30 days' AND compressed = false"

    - id: compress_old
      type: compress_memories
      params:
        memory_ids: ${{ steps.identify_compressible.output }}
        compression_ratio: 0.5  # 50% reduction

    - id: update_stats
      type: update_metrics
      params:
        archived_count: ${{ steps.archive_stale.count }}
        compressed_count: ${{ steps.compress_old.count }}
```

---

### 3. System Health Check Workflow

**Purpose**: Continuously monitor system health and auto-remediate issues

```yaml
workflow:
  id: "system-health-check"
  version: "1.0.0"
  schedule: "*/5 * * * *"  # Every 5 minutes

  steps:
    - id: check_service_health
      type: http_request
      params:
        urls:
          - http://api-gateway:8000/health
          - http://auth-service:8001/health
          - http://task-planner:8002/health
          - http://agent-orchestrator:8003/health
          - http://memory-service:8004/health
          - http://ai-runtime:8005/health
          - http://tool-runner:8006/health
          - http://audit-log:8007/health
          - http://explainability:8008/health
          - http://observability:8009/health

    - id: check_database
      type: execute_python
      params:
        script: check_db_connection.py

    - id: check_redis
      type: execute_python
      params:
        script: check_redis_connection.py

    - id: alert_if_unhealthy
      type: send_alert
      condition: ${{ steps.check_service_health.any_failed || steps.check_database.failed || steps.check_redis.failed }}
      params:
        severity: critical
        message: "System health check failed"

    - id: auto_remediate
      type: restart_service
      condition: ${{ steps.check_service_health.any_failed }}
      params:
        service: ${{ steps.check_service_health.failed_service }}
```

---

## Part C: New UI Surfaces

### 1. Agent Graph Visualizer

**Purpose**: Visualize agent execution as a graph, showing thinking process

**Features**:
- Agent nodes (planner, executor, critic, summarizer)
- Edges showing data flow
- Node color by confidence score (green=high, yellow=medium, red=low)
- Click node to see reasoning trace
- Timeline scrubber to see execution over time

**Technology**: React + D3.js + React Flow

**Mockup**:
```
┌────────────────────────────────────────────────┐
│  Agent Execution Graph - Task: Deploy Web App  │
├────────────────────────────────────────────────┤
│                                                │
│    ┌──────────┐                                │
│    │ Planner  │ (confidence: 0.92)             │
│    │  Agent   │                                │
│    └─────┬────┘                                │
│          │ plan                                │
│          ▼                                     │
│    ┌──────────┐                                │
│    │ Executor │ (confidence: 0.87)             │
│    │  Agent   │                                │
│    └─────┬────┘                                │
│          │ code                                │
│          ▼                                     │
│    ┌──────────┐                                │
│    │  Critic  │ (confidence: 0.78) ⚠️          │
│    │  Agent   │                                │
│    └─────┬────┘                                │
│          │ review                              │
│          ▼                                     │
│    ┌──────────┐                                │
│    │ Executor │ (confidence: 0.95) ✅          │
│    │  Agent   │ (v2 - after fixes)             │
│    └──────────┘                                │
│                                                │
│  [◄ Prev] ━━━━━●━━━━━━━━━━━━━━ [Next ►]      │
│           (Timeline: step 4 of 8)              │
└────────────────────────────────────────────────┘
```

---

### 2. Workflow Timeline Graph

**Purpose**: Visualize workflow execution as a Gantt chart / timeline

**Features**:
- Horizontal timeline showing step duration
- Parallel steps shown on same timeline
- Failed steps highlighted in red
- Retry attempts shown as smaller bars
- Hover for step details

**Technology**: React + Recharts

**Mockup**:
```
┌──────────────────────────────────────────────────────────┐
│  Workflow Execution - deploy-web-app-v2                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  clone_repo     ████████ (12s) ✅                       │
│  run_tests      ░░░░░░░░████████████ (23s) ✅           │
│  build_docker   ░░░░░░░░░░░░░░░░░░░░████████ (15s) ✅   │
│  deploy         ░░░░░░░░░░░░░░░░░░░░░░░░░░░████ (8s) ✅ │
│                                                          │
│  ├────┬────┬────┬────┬────┬────┬────┬────┬────┤        │
│  0s  10s  20s  30s  40s  50s  60s  70s  80s  90s       │
│                                                          │
│  Total Duration: 58s                                     │
│  Status: ✅ SUCCESS                                      │
└──────────────────────────────────────────────────────────┘
```

---

### 3. Memory Heatmap Inspector

**Purpose**: Visualize memory usage and access patterns as a heatmap

**Features**:
- Heatmap of memory access frequency (hot=red, cold=blue)
- Memory type breakdown (working, short-term, long-term, episodic, semantic)
- Memory age distribution
- Memory size distribution
- Click cell to see memory details

**Technology**: React + Victory Heatmap

**Mockup**:
```
┌────────────────────────────────────────────────┐
│  Memory Heatmap - User: john@example.com       │
├────────────────────────────────────────────────┤
│                                                │
│  Memory Type │ 0-1d │ 1-7d │ 7-30d │ 30d+    │
│  ───────────────────────────────────────────── │
│  Working     │ 🟥🟥 │ 🟧🟧 │ 🟨🟨  │ 🟦🟦     │
│  Short-term  │ 🟧🟧 │ 🟨🟨 │ 🟦🟦  │ 🟦🟦     │
│  Long-term   │ 🟨🟨 │ 🟦🟦 │ 🟦🟦  │ 🟦🟦     │
│  Episodic    │ 🟧🟧 │ 🟨🟨 │ 🟨🟨  │ 🟦🟦     │
│  Semantic    │ 🟥🟥 │ 🟥🟥 │ 🟧🟧  │ 🟨🟨     │
│                                                │
│  🟥 Hot (>100 accesses)  🟧 Warm (10-100)      │
│  🟨 Cool (1-10)          🟦 Cold (0)           │
│                                                │
│  Total Memories: 1,247                         │
│  Hot Memories: 23  │  Candidates for GC: 892  │
└────────────────────────────────────────────────┘
```

---

### 4. Failure Debugger Panel

**Purpose**: Deep dive into failures with step-by-step debugging

**Features**:
- Failure timeline (what happened when)
- Stack trace (for code execution failures)
- AI reasoning trace (for AI failures)
- Related logs (from observability service)
- Suggested fixes (from AI)

**Technology**: React + Monaco Editor (for code)

**Mockup**:
```
┌────────────────────────────────────────────────┐
│  Failure Debugger - Task: build-docker         │
├────────────────────────────────────────────────┤
│                                                │
│  ❌ FAILED at 2026-02-10 14:32:18 UTC         │
│                                                │
│  Timeline:                                     │
│  ────────────────────────────────────────────  │
│  14:32:10 - Task started                       │
│  14:32:12 - Agent assigned (executor-003)      │
│  14:32:14 - Tool execution started             │
│  14:32:18 - ❌ Tool execution failed           │
│                                                │
│  Error Message:                                │
│  ────────────────────────────────────────────  │
│  Docker build failed: base image not found     │
│  Image: python:3.11-slim                       │
│  Registry: docker.io                           │
│                                                │
│  Stack Trace:                                  │
│  ────────────────────────────────────────────  │
│  File "tool_runner.py", line 142               │
│    subprocess.run(["docker", "build", "."])    │
│  CalledProcessError: Command '['docker', ...]  │
│                                                │
│  AI Reasoning:                                 │
│  ────────────────────────────────────────────  │
│  Planner Agent (confidence: 0.89)              │
│  "I will build a Docker image using the        │
│   Dockerfile in the repository..."            │
│                                                │
│  Suggested Fixes:                              │
│  ────────────────────────────────────────────  │
│  1. ✅ Check Docker registry connectivity      │
│  2. ✅ Verify base image exists                │
│  3. 💡 Use alternative base image              │
│                                                │
│  [Retry with Fix 3] [Mark as Known Issue]     │
└────────────────────────────────────────────────┘
```

---

### 5. "Why Did This Happen?" Panel

**Purpose**: Explain decisions with natural language

**Features**:
- Natural language explanation of decisions
- Confidence breakdown
- Alternatives considered
- Decision rationale
- Timeline of events

**Technology**: React + Markdown renderer

**Mockup**:
```
┌────────────────────────────────────────────────┐
│  Why did the agent choose GPT-4 over Claude?   │
├────────────────────────────────────────────────┤
│                                                │
│  Decision Summary:                             │
│  ────────────────────────────────────────────  │
│  The AI Runtime chose GPT-4 for this task      │
│  because the task required code generation,    │
│  which GPT-4 performs better at according to   │
│  historical metrics (0.92 vs 0.87 quality).    │
│                                                │
│  Factors Considered:                           │
│  ────────────────────────────────────────────  │
│  ✅ Task type: code_generation                 │
│  ✅ Historical quality: GPT-4 (0.92)           │
│  ✅ Cost: $0.03 per 1K tokens (acceptable)     │
│  ✅ Latency: 1.8s P50 (acceptable)             │
│  ⚠️  Budget remaining: $8.42 / $10.00 (84%)    │
│                                                │
│  Alternatives Evaluated:                       │
│  ────────────────────────────────────────────  │
│  1. Claude-3-Sonnet                            │
│     Quality: 0.87 | Cost: $0.003 | Latency: 1.2s │
│     ❌ Rejected: Lower quality score           │
│                                                │
│  2. GPT-3.5-turbo                              │
│     Quality: 0.79 | Cost: $0.0005 | Latency: 0.6s │
│     ❌ Rejected: Below quality threshold (0.80) │
│                                                │
│  Decision Rationale:                           │
│  ────────────────────────────────────────────  │
│  GPT-4 was selected because:                   │
│  • Highest quality score for code generation   │
│  • Within budget constraints                   │
│  • User prefers quality over cost (profile)    │
│                                                │
│  Timeline:                                     │
│  ────────────────────────────────────────────  │
│  14:30:12 - Task received                      │
│  14:30:13 - Model selection started            │
│  14:30:14 - GPT-4 selected (confidence: 0.94)  │
│  14:30:15 - LLM request sent                   │
│  14:30:17 - Response received (2.1s)           │
└────────────────────────────────────────────────┘
```

---

## Part D: New Agent Capabilities

### 1. Typed Agent Entities

**Before (V1)**: Agents are database records
```python
# V1: Agent is just a database row
agent = {
    "id": "uuid",
    "role": "executor",
    "model": "gpt-4",
    "capabilities": ["code", "math"],
    "temperature": 0.3
}
```

**After (V2)**: Agents are typed software entities
```python
# V2: Agent is a typed entity with schemas
from pydantic import BaseModel

class ExecutorAgentInput(BaseModel):
    task_description: str
    context: Optional[str] = None
    tools_allowed: List[str] = []

class ExecutorAgentOutput(BaseModel):
    code: str
    explanation: str
    confidence: float
    tools_used: List[str]

class ExecutorAgent(AgentBase):
    role: AgentRole = AgentRole.EXECUTOR
    model: str = "gpt-4"
    temperature: float = 0.3

    input_schema: Type[BaseModel] = ExecutorAgentInput
    output_schema: Type[BaseModel] = ExecutorAgentOutput

    allowed_tools: List[str] = ["execute_python", "execute_javascript"]

    failure_strategy: FailureStrategy = FailureStrategy.RETRY_WITH_DIFFERENT_PROMPT

    max_retries: int = 3
    retry_delay_seconds: int = 5

    confidence_threshold: float = 0.7

    def execute(self, input: ExecutorAgentInput) -> ExecutorAgentOutput:
        """Execute task with type-safe input/output"""
        ...
```

**Benefits**:
- Type safety (Pydantic validation)
- Clear contracts (input/output schemas)
- Explicit failure strategies
- Testable (mock inputs easily)
- Self-documenting

---

### 2. Agent Performance Metrics

Track agent performance to identify issues and improve over time:

```python
class AgentMetrics(BaseModel):
    agent_id: UUID
    role: AgentRole

    # Quality metrics
    avg_confidence: float
    avg_quality_score: float
    hallucination_rate: float

    # Performance metrics
    avg_latency_ms: int
    p95_latency_ms: int
    p99_latency_ms: int

    # Cost metrics
    avg_cost_per_task: float
    total_tokens_used: int

    # Reliability metrics
    success_rate: float
    retry_rate: float
    failure_rate: float

    # Calculated over
    time_window: timedelta
    task_count: int
```

**Usage**:
```python
# Get agent metrics
metrics = agent_orchestrator.get_agent_metrics(
    agent_id="executor-003",
    time_window=timedelta(days=7)
)

if metrics.success_rate < 0.8:
    # Replace underperforming agent
    agent_orchestrator.replace_agent(
        old_agent_id="executor-003",
        new_agent_config=better_config
    )
```

---

### 3. Agent Replacement Logic

Automatically replace agents that underperform:

```python
class AgentReplacementPolicy(BaseModel):
    min_success_rate: float = 0.8
    max_avg_cost: float = 0.50
    min_avg_confidence: float = 0.7
    max_hallucination_rate: float = 0.1

    evaluation_window: timedelta = timedelta(days=7)
    min_tasks_for_evaluation: int = 10

async def auto_replace_underperforming_agents():
    """Automatically replace agents that don't meet policy"""

    agents = await agent_orchestrator.list_agents()
    policy = AgentReplacementPolicy()

    for agent in agents:
        metrics = await agent_orchestrator.get_agent_metrics(
            agent_id=agent.id,
            time_window=policy.evaluation_window
        )

        if metrics.task_count < policy.min_tasks_for_evaluation:
            continue  # Not enough data

        should_replace = (
            metrics.success_rate < policy.min_success_rate or
            metrics.avg_cost_per_task > policy.max_avg_cost or
            metrics.avg_confidence < policy.min_avg_confidence or
            metrics.hallucination_rate > policy.max_hallucination_rate
        )

        if should_replace:
            # Try different model or temperature
            new_config = optimize_agent_config(agent, metrics)

            await agent_orchestrator.replace_agent(
                old_agent_id=agent.id,
                new_agent_config=new_config,
                reason=f"Underperforming: {metrics}"
            )
```

---

## Summary of Expansions

### New Services (3)
1. ✅ Workflow Engine (Port 8010) - **CRITICAL**
2. ✅ AI Quality Gate (Port 8011) - **HIGH**
3. ✅ Developer CLI (Port 8012) - **MEDIUM**

### New Workflows (3)
1. ✅ Agent Training Pipeline
2. ✅ Memory GC Daily Job
3. ✅ System Health Check

### New UI Surfaces (5)
1. ✅ Agent Graph Visualizer
2. ✅ Workflow Timeline Graph
3. ✅ Memory Heatmap Inspector
4. ✅ Failure Debugger Panel
5. ✅ "Why?" Explanation Panel

### New Agent Capabilities (3)
1. ✅ Typed Agent Entities (Pydantic schemas)
2. ✅ Agent Performance Metrics
3. ✅ Agent Replacement Logic

---

**Total Additions**: 14 major features across 4 categories

**Next Document**: `refactor_plan.md` - What to split, modularize, and abstract
