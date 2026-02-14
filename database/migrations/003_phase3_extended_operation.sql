-- Migration 003: Phase 3 - Extended Agent Operation Infrastructure
-- Date: 2026-02-14
-- Description: Adds support for checkpoint/resume, hierarchical memory,
--              health monitoring, and cost governance for 24+ hour autonomous operations

-- ============================================================================
-- 1. CHECKPOINT & RESUME SYSTEM
-- ============================================================================

-- Checkpoints for workflow state persistence and recovery
CREATE TABLE IF NOT EXISTS checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_execution_id UUID NOT NULL REFERENCES workflow_executions(id) ON DELETE CASCADE,
    checkpoint_number INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Execution state snapshot
    execution_state JSONB NOT NULL,
    dag_progress JSONB NOT NULL,
    memory_snapshot_ref VARCHAR(500),
    active_tasks JSONB,
    budget_state JSONB,
    
    -- Metadata
    checkpoint_size_bytes BIGINT,
    compression_enabled BOOLEAN DEFAULT TRUE,
    metadata JSONB,
    
    -- Constraints
    UNIQUE(workflow_execution_id, checkpoint_number)
);

CREATE INDEX idx_checkpoints_workflow ON checkpoints(workflow_execution_id);
CREATE INDEX idx_checkpoints_created ON checkpoints(created_at DESC);
CREATE INDEX idx_checkpoints_number ON checkpoints(workflow_execution_id, checkpoint_number);

COMMENT ON TABLE checkpoints IS 'Workflow execution checkpoints for recovery and resume';
COMMENT ON COLUMN checkpoints.execution_state IS 'Current execution state including variables and context';
COMMENT ON COLUMN checkpoints.dag_progress IS 'DAG execution progress including completed/pending nodes';
COMMENT ON COLUMN checkpoints.memory_snapshot_ref IS 'Reference to memory snapshot in object storage';

-- ============================================================================
-- 2. HIERARCHICAL MEMORY SYSTEM (L1, L2, L3)
-- ============================================================================

-- L1: Working Memory (fast, recent, high-importance)
CREATE TABLE IF NOT EXISTS working_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    workflow_execution_id UUID REFERENCES workflow_executions(id) ON DELETE CASCADE,
    
    -- Memory content
    content TEXT NOT NULL,
    embedding vector(1536),
    
    -- Importance and lifecycle
    importance_score FLOAT DEFAULT 0.5 CHECK (importance_score >= 0 AND importance_score <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    memory_type VARCHAR(50),
    tags TEXT[],
    metadata JSONB
);

CREATE INDEX idx_working_memory_agent ON working_memory(agent_id);
CREATE INDEX idx_working_memory_importance ON working_memory(importance_score DESC);
CREATE INDEX idx_working_memory_created ON working_memory(created_at DESC);
CREATE INDEX idx_working_memory_expires ON working_memory(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX idx_working_memory_embedding ON working_memory USING ivfflat (embedding vector_cosine_ops);

COMMENT ON TABLE working_memory IS 'L1 Working Memory: Recent high-importance items (~1K capacity, <10ms retrieval)';
COMMENT ON COLUMN working_memory.importance_score IS 'Importance score 0-1, influences retention and promotion to L2';

-- L2: Episodic Memory (compressed, clustered, medium-term)
CREATE TABLE IF NOT EXISTS episodic_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    
    -- Clustering and summarization
    cluster_id UUID,
    summary TEXT NOT NULL,
    embedding vector(1536),
    
    -- Compression metadata
    compression_ratio FLOAT,
    source_memory_ids UUID[],
    source_memory_count INTEGER,
    
    -- Importance and lifecycle
    importance_score FLOAT DEFAULT 0.5 CHECK (importance_score >= 0 AND importance_score <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    
    -- Metadata
    temporal_period TSTZRANGE,
    tags TEXT[],
    metadata JSONB
);

CREATE INDEX idx_episodic_memory_agent ON episodic_memory(agent_id);
CREATE INDEX idx_episodic_memory_cluster ON episodic_memory(cluster_id);
CREATE INDEX idx_episodic_memory_importance ON episodic_memory(importance_score DESC);
CREATE INDEX idx_episodic_memory_created ON episodic_memory(created_at DESC);
CREATE INDEX idx_episodic_memory_embedding ON episodic_memory USING ivfflat (embedding vector_cosine_ops);

COMMENT ON TABLE episodic_memory IS 'L2 Episodic Memory: Compressed summaries of clusters (~10K capacity)';
COMMENT ON COLUMN episodic_memory.summary IS 'LLM-generated summary of source memories';
COMMENT ON COLUMN episodic_memory.compression_ratio IS 'Compression ratio achieved (source size / summary size)';

-- L3: Long-Term Memory (unlimited, cold storage, archived knowledge)
CREATE TABLE IF NOT EXISTS longterm_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    
    -- Knowledge content
    knowledge_type VARCHAR(50),
    title VARCHAR(500),
    content TEXT NOT NULL,
    embedding vector(1536),
    
    -- Importance and lifecycle
    importance_score FLOAT DEFAULT 0.5 CHECK (importance_score >= 0 AND importance_score <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    archived_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    source_type VARCHAR(50), -- episodic_compression, manual_entry, learned_pattern
    tags TEXT[],
    metadata JSONB
);

CREATE INDEX idx_longterm_memory_agent ON longterm_memory(agent_id);
CREATE INDEX idx_longterm_memory_type ON longterm_memory(knowledge_type);
CREATE INDEX idx_longterm_memory_importance ON longterm_memory(importance_score DESC);
CREATE INDEX idx_longterm_memory_archived ON longterm_memory(archived_at) WHERE archived_at IS NOT NULL;
CREATE INDEX idx_longterm_memory_embedding ON longterm_memory USING ivfflat (embedding vector_cosine_ops);

COMMENT ON TABLE longterm_memory IS 'L3 Long-Term Memory: Unlimited cold storage for high-value knowledge';
COMMENT ON COLUMN longterm_memory.knowledge_type IS 'Type of knowledge: fact, pattern, procedure, concept, etc.';

-- Memory lifecycle tracking
CREATE TABLE IF NOT EXISTS memory_lifecycle_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    
    -- Event details
    event_type VARCHAR(50) NOT NULL, -- created, accessed, promoted, demoted, compressed, archived
    memory_tier VARCHAR(10) NOT NULL, -- L1, L2, L3
    memory_id UUID NOT NULL,
    
    -- Context
    reason TEXT,
    importance_score_before FLOAT,
    importance_score_after FLOAT,
    
    -- Timing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB
);

CREATE INDEX idx_memory_lifecycle_agent ON memory_lifecycle_events(agent_id);
CREATE INDEX idx_memory_lifecycle_type ON memory_lifecycle_events(event_type);
CREATE INDEX idx_memory_lifecycle_created ON memory_lifecycle_events(created_at DESC);

COMMENT ON TABLE memory_lifecycle_events IS 'Audit log of memory tier transitions and lifecycle events';

-- ============================================================================
-- 3. AGENT HEALTH MONITORING
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_health_status (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    workflow_execution_id UUID REFERENCES workflow_executions(id) ON DELETE CASCADE,
    
    -- Health status
    status VARCHAR(20) NOT NULL CHECK (status IN ('healthy', 'degraded', 'failed', 'recovering')),
    last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Resource metrics
    memory_usage_mb FLOAT,
    cpu_usage_percent FLOAT,
    working_memory_count INTEGER,
    episodic_memory_count INTEGER,
    
    -- Cost metrics
    cost_consumed DECIMAL(10, 2),
    budget_remaining DECIMAL(10, 2),
    
    -- Task metrics
    active_tasks_count INTEGER DEFAULT 0,
    completed_tasks_count INTEGER DEFAULT 0,
    failed_tasks_count INTEGER DEFAULT 0,
    
    -- Overall health score (0-1)
    health_score FLOAT CHECK (health_score >= 0 AND health_score <= 1),
    
    -- Metadata
    error_message TEXT,
    recovery_attempts INTEGER DEFAULT 0,
    metadata JSONB,
    
    -- Timing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_agent_health_agent ON agent_health_status(agent_id);
CREATE INDEX idx_agent_health_workflow ON agent_health_status(workflow_execution_id);
CREATE INDEX idx_agent_health_heartbeat ON agent_health_status(last_heartbeat DESC);
CREATE INDEX idx_agent_health_status ON agent_health_status(status);
CREATE INDEX idx_agent_health_score ON agent_health_status(health_score);

COMMENT ON TABLE agent_health_status IS 'Real-time agent health monitoring with 15-second heartbeat';
COMMENT ON COLUMN agent_health_status.health_score IS 'Composite health score based on all metrics (0=critical, 1=perfect)';

-- Health incidents
CREATE TABLE IF NOT EXISTS agent_health_incidents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    workflow_execution_id UUID REFERENCES workflow_executions(id) ON DELETE CASCADE,
    
    -- Incident details
    incident_type VARCHAR(50) NOT NULL, -- heartbeat_failure, memory_overflow, cost_breach, task_failure
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    description TEXT NOT NULL,
    
    -- Resolution
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'investigating', 'resolved', 'ignored')),
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolution_notes TEXT,
    
    -- Recovery
    recovery_action VARCHAR(100), -- restart, checkpoint_restore, workflow_halt, etc.
    recovery_successful BOOLEAN,
    
    -- Timing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Metadata
    metrics_snapshot JSONB,
    metadata JSONB
);

CREATE INDEX idx_health_incidents_agent ON agent_health_incidents(agent_id);
CREATE INDEX idx_health_incidents_status ON agent_health_incidents(status);
CREATE INDEX idx_health_incidents_severity ON agent_health_incidents(severity);
CREATE INDEX idx_health_incidents_created ON agent_health_incidents(created_at DESC);

COMMENT ON TABLE agent_health_incidents IS 'Health incident tracking and resolution history';

-- ============================================================================
-- 4. COST GOVERNANCE & BUDGET TRACKING
-- ============================================================================

CREATE TABLE IF NOT EXISTS workflow_budget (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_execution_id UUID NOT NULL REFERENCES workflow_executions(id) ON DELETE CASCADE,
    
    -- Budget allocation
    allocated_budget DECIMAL(10, 2) NOT NULL CHECK (allocated_budget >= 0),
    consumed_budget DECIMAL(10, 2) DEFAULT 0 CHECK (consumed_budget >= 0),
    
    -- Thresholds
    warning_threshold DECIMAL(10, 2), -- Default 80% of allocated
    critical_threshold DECIMAL(10, 2), -- Default 95% of allocated
    
    -- Status
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'warning', 'critical', 'exhausted', 'completed')),
    
    -- Actions taken
    warnings_sent INTEGER DEFAULT 0,
    halt_triggered_at TIMESTAMP WITH TIME ZONE,
    suspended_at TIMESTAMP WITH TIME ZONE,
    
    -- Timing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB
);

CREATE INDEX idx_workflow_budget_execution ON workflow_budget(workflow_execution_id);
CREATE INDEX idx_workflow_budget_status ON workflow_budget(status);
CREATE INDEX idx_workflow_budget_consumed ON workflow_budget(consumed_budget DESC);

COMMENT ON TABLE workflow_budget IS 'Per-workflow budget allocation and enforcement';
COMMENT ON COLUMN workflow_budget.status IS 'Budget status: active (<80%), warning (80-95%), critical (95-100%), exhausted (100%)';

-- Detailed cost tracking
CREATE TABLE IF NOT EXISTS cost_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_execution_id UUID NOT NULL REFERENCES workflow_executions(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES agents(id) ON DELETE CASCADE,
    
    -- Operation details
    operation_type VARCHAR(50) NOT NULL, -- llm_call, storage, compute, memory_operation
    provider VARCHAR(50),
    model VARCHAR(100),
    
    -- Resource usage
    tokens_used INTEGER,
    execution_time_ms INTEGER,
    memory_bytes BIGINT,
    
    -- Cost calculation
    cost DECIMAL(10, 6) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    
    -- Context
    task_id UUID,
    step_name VARCHAR(200),
    
    -- Timing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Metadata
    request_payload JSONB,
    response_metadata JSONB,
    metadata JSONB
);

CREATE INDEX idx_cost_tracking_workflow ON cost_tracking(workflow_execution_id);
CREATE INDEX idx_cost_tracking_agent ON cost_tracking(agent_id);
CREATE INDEX idx_cost_tracking_operation ON cost_tracking(operation_type);
CREATE INDEX idx_cost_tracking_created ON cost_tracking(created_at DESC);
CREATE INDEX idx_cost_tracking_provider_model ON cost_tracking(provider, model);

COMMENT ON TABLE cost_tracking IS 'Granular cost tracking for all operations (LLM, storage, compute)';
COMMENT ON COLUMN cost_tracking.cost IS 'Cost in smallest currency unit (e.g., cents for USD)';

-- Cost aggregations (materialized view for performance)
CREATE MATERIALIZED VIEW IF NOT EXISTS cost_summary_by_workflow AS
SELECT 
    workflow_execution_id,
    COUNT(*) as total_operations,
    SUM(cost) as total_cost,
    SUM(CASE WHEN operation_type = 'llm_call' THEN cost ELSE 0 END) as llm_cost,
    SUM(CASE WHEN operation_type = 'storage' THEN cost ELSE 0 END) as storage_cost,
    SUM(CASE WHEN operation_type = 'compute' THEN cost ELSE 0 END) as compute_cost,
    SUM(tokens_used) as total_tokens,
    MIN(created_at) as first_operation_at,
    MAX(created_at) as last_operation_at
FROM cost_tracking
GROUP BY workflow_execution_id;

CREATE UNIQUE INDEX idx_cost_summary_workflow ON cost_summary_by_workflow(workflow_execution_id);

COMMENT ON MATERIALIZED VIEW cost_summary_by_workflow IS 'Pre-aggregated cost summary per workflow for fast queries';

-- ============================================================================
-- 5. PHASE 3 CONFIGURATION & FEATURE FLAGS
-- ============================================================================

CREATE TABLE IF NOT EXISTS phase3_config (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_key VARCHAR(100) NOT NULL UNIQUE,
    config_value JSONB NOT NULL,
    description TEXT,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert default Phase 3 configurations
INSERT INTO phase3_config (config_key, config_value, description) VALUES
('checkpoint.enabled', 'true', 'Enable checkpoint/resume functionality'),
('checkpoint.interval_seconds', '300', 'Checkpoint creation interval (5 minutes)'),
('checkpoint.max_per_workflow', '100', 'Maximum checkpoints to retain per workflow'),
('checkpoint.compression_enabled', 'true', 'Enable checkpoint compression'),

('memory.l1.max_items', '1000', 'L1 Working Memory maximum items'),
('memory.l2.max_items', '10000', 'L2 Episodic Memory maximum items'),
('memory.l3.enabled', 'true', 'Enable L3 Long-Term Memory'),
('memory.compression_interval_minutes', '60', 'Memory compression interval'),
('memory.importance_threshold_l1_l2', '0.7', 'Importance threshold for L1→L2 promotion'),
('memory.importance_threshold_l2_l3', '0.8', 'Importance threshold for L2→L3 promotion'),

('health.heartbeat_interval_seconds', '15', 'Agent heartbeat interval'),
('health.failure_threshold_seconds', '30', 'No heartbeat failure threshold'),
('health.auto_recovery_enabled', 'true', 'Enable automatic recovery'),
('health.max_recovery_attempts', '3', 'Maximum automatic recovery attempts'),

('cost.warning_threshold_percent', '80', 'Budget warning threshold'),
('cost.critical_threshold_percent', '95', 'Budget critical threshold'),
('cost.auto_halt_enabled', 'true', 'Automatically halt at critical threshold'),
('cost.auto_suspend_at_100', 'true', 'Automatically suspend at 100% budget')
ON CONFLICT (config_key) DO NOTHING;

CREATE INDEX idx_phase3_config_key ON phase3_config(config_key);
CREATE INDEX idx_phase3_config_enabled ON phase3_config(enabled);

COMMENT ON TABLE phase3_config IS 'Phase 3 feature configuration and feature flags';

-- ============================================================================
-- 6. UTILITY FUNCTIONS
-- ============================================================================

-- Function to calculate memory tier for an item based on importance and age
CREATE OR REPLACE FUNCTION calculate_memory_tier(
    importance_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER
) RETURNS VARCHAR(2) AS $$
DECLARE
    age_hours FLOAT;
    tier VARCHAR(2);
BEGIN
    age_hours := EXTRACT(EPOCH FROM (NOW() - created_at)) / 3600;
    
    -- High importance + recent + frequently accessed = L1
    IF importance_score >= 0.7 AND age_hours < 24 AND access_count > 5 THEN
        tier := 'L1';
    -- Medium importance or older = L2
    ELSIF importance_score >= 0.5 OR age_hours < 168 THEN -- 1 week
        tier := 'L2';
    -- Low importance or very old = L3
    ELSE
        tier := 'L3';
    END IF;
    
    RETURN tier;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to calculate agent health score
CREATE OR REPLACE FUNCTION calculate_health_score(
    memory_usage_mb FLOAT,
    cost_consumed DECIMAL,
    budget_remaining DECIMAL,
    failed_tasks_count INTEGER,
    active_tasks_count INTEGER
) RETURNS FLOAT AS $$
DECLARE
    score FLOAT := 1.0;
    budget_usage_pct FLOAT;
BEGIN
    -- Deduct for high memory usage (>1.5GB = 20% penalty)
    IF memory_usage_mb > 1500 THEN
        score := score - 0.2;
    END IF;
    
    -- Deduct for budget consumption
    IF budget_remaining > 0 THEN
        budget_usage_pct := cost_consumed / (cost_consumed + budget_remaining);
        IF budget_usage_pct > 0.95 THEN
            score := score - 0.3;
        ELSIF budget_usage_pct > 0.8 THEN
            score := score - 0.15;
        END IF;
    END IF;
    
    -- Deduct for failed tasks
    IF failed_tasks_count > 0 AND active_tasks_count > 0 THEN
        score := score - (failed_tasks_count::FLOAT / active_tasks_count::FLOAT) * 0.5;
    END IF;
    
    -- Ensure score stays in valid range
    RETURN GREATEST(0.0, LEAST(1.0, score));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ============================================================================
-- 7. TRIGGERS FOR AUTO-UPDATE
-- ============================================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_agent_health_status_updated_at
    BEFORE UPDATE ON agent_health_status
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workflow_budget_updated_at
    BEFORE UPDATE ON workflow_budget
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_phase3_config_updated_at
    BEFORE UPDATE ON phase3_config
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- 8. GRANTS AND PERMISSIONS
-- ============================================================================

-- Grant appropriate permissions (adjust based on your user setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO cognition_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO cognition_app;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

-- Log migration completion
DO $$
BEGIN
    RAISE NOTICE 'Migration 003: Phase 3 Extended Operation Infrastructure - COMPLETE';
    RAISE NOTICE 'Added: Checkpoints, Hierarchical Memory (L1/L2/L3), Health Monitoring, Cost Governance';
END $$;
