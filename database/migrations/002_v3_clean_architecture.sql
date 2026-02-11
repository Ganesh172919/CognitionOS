-- CognitionOS V3 Database Schema
-- Migration: 002_v3_clean_architecture
-- Created: 2026-02-11
-- Description: Add tables for V3 clean architecture (Workflow, Agent, Memory, Task, Execution)

-- ============================================================================
-- WORKFLOWS
-- ============================================================================

CREATE TABLE IF NOT EXISTS workflows (
    id VARCHAR(255) NOT NULL,
    version_major INTEGER NOT NULL,
    version_minor INTEGER NOT NULL,
    version_patch INTEGER NOT NULL,
    name VARCHAR(512) NOT NULL,
    description TEXT NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'draft',
    schedule VARCHAR(255),
    tags TEXT[],
    steps JSONB NOT NULL,
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, version_major, version_minor, version_patch)
);

CREATE INDEX idx_workflows_id ON workflows(id);
CREATE INDEX idx_workflows_status ON workflows(status);
CREATE INDEX idx_workflows_created_at ON workflows(created_at);

-- ============================================================================
-- WORKFLOW EXECUTIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS workflow_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id VARCHAR(255) NOT NULL,
    workflow_version_major INTEGER NOT NULL,
    workflow_version_minor INTEGER NOT NULL,
    workflow_version_patch INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    inputs JSONB DEFAULT '{}',
    outputs JSONB,
    error TEXT,
    user_id UUID,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    FOREIGN KEY (workflow_id, workflow_version_major, workflow_version_minor, workflow_version_patch)
        REFERENCES workflows(id, version_major, version_minor, version_patch)
        ON DELETE CASCADE
);

CREATE INDEX idx_workflow_executions_workflow_id ON workflow_executions(workflow_id);
CREATE INDEX idx_workflow_executions_status ON workflow_executions(status);
CREATE INDEX idx_workflow_executions_user_id ON workflow_executions(user_id);
CREATE INDEX idx_workflow_executions_created_at ON workflow_executions(created_at);

-- ============================================================================
-- STEP EXECUTIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS step_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID NOT NULL REFERENCES workflow_executions(id) ON DELETE CASCADE,
    step_id VARCHAR(255) NOT NULL,
    step_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    output JSONB,
    error TEXT,
    agent_id UUID,
    retry_count INTEGER DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_step_executions_execution_id ON step_executions(execution_id);
CREATE INDEX idx_step_executions_status ON step_executions(status);
CREATE INDEX idx_step_executions_step_id ON step_executions(step_id);

-- ============================================================================
-- AGENTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_definitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) UNIQUE NOT NULL,
    role VARCHAR(50) NOT NULL,
    version VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    default_capabilities TEXT[] NOT NULL,
    default_tools JSONB NOT NULL,
    model_config JSONB NOT NULL,
    default_budget JSONB NOT NULL,
    system_prompt_template TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE (name, version)
);

CREATE INDEX idx_agent_definitions_name ON agent_definitions(name);
CREATE INDEX idx_agent_definitions_role ON agent_definitions(role);

CREATE TABLE IF NOT EXISTS agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    definition_id UUID NOT NULL REFERENCES agent_definitions(id),
    role VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    capabilities JSONB NOT NULL,
    tools JSONB NOT NULL,
    model_config JSONB NOT NULL,
    budget_limits JSONB NOT NULL,
    budget_usage JSONB NOT NULL DEFAULT '{"tokens_used": 0, "cost_usd": 0.0, "time_seconds": 0.0, "tool_executions": 0}',
    status VARCHAR(50) NOT NULL DEFAULT 'created',
    failure_strategy VARCHAR(50) NOT NULL DEFAULT 'retry',
    current_task_id UUID,
    granted_permissions TEXT[] DEFAULT '{}',
    system_prompt TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_active TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_agents_definition_id ON agents(definition_id);
CREATE INDEX idx_agents_status ON agents(status);
CREATE INDEX idx_agents_current_task_id ON agents(current_task_id);
CREATE INDEX idx_agents_last_active ON agents(last_active);

-- ============================================================================
-- MEMORY SYSTEM V3
-- ============================================================================

CREATE TABLE IF NOT EXISTS memories_v3 (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    content TEXT NOT NULL,
    memory_type VARCHAR(50) NOT NULL,
    scope VARCHAR(50) NOT NULL,
    namespace VARCHAR(512) NOT NULL,
    embedding vector(1536),
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    importance VARCHAR(50) NOT NULL DEFAULT 'medium',
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    access_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    accessed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_memories_v3_user_id ON memories_v3(user_id);
CREATE INDEX idx_memories_v3_type ON memories_v3(memory_type);
CREATE INDEX idx_memories_v3_scope ON memories_v3(scope);
CREATE INDEX idx_memories_v3_namespace ON memories_v3(namespace);
CREATE INDEX idx_memories_v3_status ON memories_v3(status);
CREATE INDEX idx_memories_v3_accessed_at ON memories_v3(accessed_at);

-- Vector similarity index (IVFFlat for faster approximate search)
CREATE INDEX idx_memories_v3_embedding ON memories_v3 USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ============================================================================
-- MEMORY COLLECTIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS memory_collections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(512) NOT NULL,
    description TEXT NOT NULL,
    namespace VARCHAR(512) NOT NULL,
    memory_ids UUID[] DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_memory_collections_namespace ON memory_collections(namespace);

-- ============================================================================
-- MEMORY LIFECYCLE POLICIES
-- ============================================================================

CREATE TABLE IF NOT EXISTS memory_lifecycle_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(512) NOT NULL,
    namespace VARCHAR(512) NOT NULL UNIQUE,
    compress_after_days INTEGER,
    archive_after_days INTEGER,
    delete_after_days INTEGER,
    importance_threshold VARCHAR(50) NOT NULL DEFAULT 'low',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_memory_policies_namespace ON memory_lifecycle_policies(namespace);

-- ============================================================================
-- TASKS V3
-- ============================================================================

CREATE TABLE IF NOT EXISTS tasks_v3 (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    session_id UUID NOT NULL,
    goal_id UUID NOT NULL,
    name VARCHAR(512) NOT NULL,
    description TEXT NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    priority VARCHAR(50) NOT NULL DEFAULT 'medium',
    required_capabilities TEXT[] DEFAULT '{}',
    dependencies UUID[] DEFAULT '{}',
    assigned_agent_id UUID,
    input_data JSONB DEFAULT '{}',
    output_data JSONB,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_tasks_v3_user_id ON tasks_v3(user_id);
CREATE INDEX idx_tasks_v3_status ON tasks_v3(status);
CREATE INDEX idx_tasks_v3_assigned_agent_id ON tasks_v3(assigned_agent_id);
CREATE INDEX idx_tasks_v3_created_at ON tasks_v3(created_at);

-- ============================================================================
-- EXECUTION TRACES
-- ============================================================================

CREATE TABLE IF NOT EXISTS execution_traces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    entity_id UUID NOT NULL,
    user_id UUID NOT NULL,
    parent_trace_id UUID REFERENCES execution_traces(id),
    inputs JSONB DEFAULT '{}',
    outputs JSONB,
    error TEXT,
    metadata JSONB DEFAULT '{}',
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_execution_traces_type ON execution_traces(execution_type);
CREATE INDEX idx_execution_traces_status ON execution_traces(status);
CREATE INDEX idx_execution_traces_entity_id ON execution_traces(entity_id);
CREATE INDEX idx_execution_traces_user_id ON execution_traces(user_id);
CREATE INDEX idx_execution_traces_parent ON execution_traces(parent_trace_id);
CREATE INDEX idx_execution_traces_created_at ON execution_traces(created_at);
