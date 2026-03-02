-- Migration 010: Agent Runs (Single-Agent Runtime)
-- Date: 2026-02-27
-- Description: Persisted autonomous agent runs, step logs, artifacts, evaluations, and memory links

-- ============================================================================
-- AGENT RUNS
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id),

    requirement TEXT NOT NULL,
    status VARCHAR(50) NOT NULL CHECK (
        status IN ('created', 'queued', 'running', 'validating', 'completed', 'failed', 'cancelled')
    ),

    -- Budgets and usage are JSON for forward compatibility.
    budgets JSONB NOT NULL DEFAULT '{}'::jsonb,
    usage JSONB NOT NULL DEFAULT '{}'::jsonb,

    error TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_tenant_id_created_at ON agent_runs(tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_runs_status_created_at ON agent_runs(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_runs_user_id_created_at ON agent_runs(user_id, created_at DESC);

-- ============================================================================
-- AGENT RUN STEPS
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_run_steps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,

    step_index INTEGER NOT NULL,
    step_type VARCHAR(50) NOT NULL CHECK (
        step_type IN ('analysis', 'planning', 'execution', 'validation', 'evaluation', 'tool', 'codegen')
    ),
    status VARCHAR(50) NOT NULL CHECK (
        status IN ('pending', 'running', 'completed', 'failed', 'skipped')
    ),

    input JSONB NOT NULL DEFAULT '{}'::jsonb,
    output JSONB,
    tool_calls JSONB NOT NULL DEFAULT '[]'::jsonb,

    tokens_used INTEGER NOT NULL DEFAULT 0,
    cost_usd NUMERIC(12, 6) NOT NULL DEFAULT 0,
    duration_ms INTEGER,

    error TEXT,

    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_agent_run_step UNIQUE (run_id, step_index)
);

CREATE INDEX IF NOT EXISTS idx_agent_run_steps_run_id_index ON agent_run_steps(run_id, step_index);
CREATE INDEX IF NOT EXISTS idx_agent_run_steps_status ON agent_run_steps(status);

-- ============================================================================
-- AGENT RUN ARTIFACTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_run_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    step_id UUID REFERENCES agent_run_steps(id) ON DELETE SET NULL,

    kind VARCHAR(50) NOT NULL CHECK (kind IN ('code', 'patch', 'test', 'log', 'report', 'binary')),
    name VARCHAR(255) NOT NULL,
    content_type VARCHAR(100) NOT NULL DEFAULT 'text/plain',

    content_text TEXT,
    content_bytes BYTEA,
    sha256 VARCHAR(64),
    size_bytes INTEGER,
    storage_url TEXT,

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_run_artifacts_run_id ON agent_run_artifacts(run_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_run_artifacts_step_id ON agent_run_artifacts(step_id);
CREATE INDEX IF NOT EXISTS idx_agent_run_artifacts_kind ON agent_run_artifacts(kind);

-- ============================================================================
-- AGENT RUN EVALUATIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_run_evaluations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,

    success BOOLEAN NOT NULL DEFAULT FALSE,
    confidence NUMERIC(5, 4) NOT NULL DEFAULT 0,
    quality_scores JSONB NOT NULL DEFAULT '{}'::jsonb,
    policy_violations JSONB NOT NULL DEFAULT '[]'::jsonb,
    retry_plan JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_run_evaluations_run_id ON agent_run_evaluations(run_id, created_at DESC);

-- ============================================================================
-- AGENT RUN MEMORY LINKS
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_run_memory_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,

    memory_id UUID NOT NULL,
    memory_tier VARCHAR(50) NOT NULL,
    relation VARCHAR(50) NOT NULL DEFAULT 'used', -- used/promoted/evicted

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_run_memory_links_run_id ON agent_run_memory_links(run_id, created_at DESC);

