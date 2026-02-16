-- Migration 005: Phase 5 V4 Evolution - Performance & Resilience
-- CognitionOS V4: Multi-Layer LLM Caching, Vector Search Optimization, Cost Tracking

-- ========================================
-- Part 1: LLM Multi-Layer Caching
-- ========================================

-- L1/L2: Exact match cache (Redis + PostgreSQL)
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key VARCHAR(255) PRIMARY KEY,
    request_hash VARCHAR(255) NOT NULL,
    messages JSONB NOT NULL,
    model VARCHAR(100) NOT NULL,
    response_content TEXT NOT NULL,
    provider VARCHAR(50) NOT NULL,
    usage JSONB NOT NULL,
    latency_ms INTEGER NOT NULL,
    cost_usd DECIMAL(10, 6),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    accessed_at TIMESTAMP NOT NULL DEFAULT NOW(),
    access_count INTEGER NOT NULL DEFAULT 1,
    ttl_seconds INTEGER NOT NULL DEFAULT 3600,
    metadata JSONB,
    INDEX idx_llm_cache_model (model),
    INDEX idx_llm_cache_created_at (created_at),
    INDEX idx_llm_cache_request_hash (request_hash)
);

-- L3: Semantic similarity cache with pgvector
CREATE TABLE IF NOT EXISTS llm_semantic_cache (
    cache_key VARCHAR(255) PRIMARY KEY,
    request_hash VARCHAR(255) NOT NULL,
    messages JSONB NOT NULL,
    model VARCHAR(100) NOT NULL,
    response_content TEXT NOT NULL,
    provider VARCHAR(50) NOT NULL,
    usage JSONB NOT NULL,
    latency_ms INTEGER NOT NULL,
    cost_usd DECIMAL(10, 6),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    accessed_at TIMESTAMP NOT NULL DEFAULT NOW(),
    access_count INTEGER NOT NULL DEFAULT 1,
    ttl_seconds INTEGER NOT NULL DEFAULT 86400,
    embedding vector(1536),  -- OpenAI embedding dimension
    metadata JSONB
);

-- Create HNSW index for fast semantic search
CREATE INDEX IF NOT EXISTS idx_llm_semantic_embedding_hnsw 
ON llm_semantic_cache 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Create IVFFlat index for large-scale semantic search
CREATE INDEX IF NOT EXISTS idx_llm_semantic_embedding_ivfflat 
ON llm_semantic_cache 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 1000);

-- Additional indexes for semantic cache
CREATE INDEX IF NOT EXISTS idx_llm_semantic_model ON llm_semantic_cache(model);
CREATE INDEX IF NOT EXISTS idx_llm_semantic_created_at ON llm_semantic_cache(created_at);
CREATE INDEX IF NOT EXISTS idx_llm_semantic_accessed_at ON llm_semantic_cache(accessed_at);
CREATE INDEX IF NOT EXISTS idx_llm_semantic_access_count ON llm_semantic_cache(access_count);

-- Cache statistics tracking
CREATE TABLE IF NOT EXISTS llm_cache_stats (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    layer VARCHAR(20) NOT NULL,  -- l1_redis, l2_database, l3_semantic, l4_llm_api
    total_requests INTEGER NOT NULL DEFAULT 0,
    cache_hits INTEGER NOT NULL DEFAULT 0,
    cache_misses INTEGER NOT NULL DEFAULT 0,
    hit_rate DECIMAL(5, 4),
    avg_latency_ms DECIMAL(10, 2),
    total_cost_usd DECIMAL(10, 6),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_cache_stats_timestamp ON llm_cache_stats(timestamp);
CREATE INDEX IF NOT EXISTS idx_cache_stats_layer ON llm_cache_stats(layer);

-- ========================================
-- Part 2: Enhanced Vector Search
-- ========================================

-- Optimize existing memory tables with better indexes
-- Add namespace partitioning
ALTER TABLE memories ADD COLUMN IF NOT EXISTS namespace VARCHAR(100) DEFAULT 'default';
CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);

-- Add metadata-only index for fast filtering
CREATE INDEX IF NOT EXISTS idx_memories_metadata_gin ON memories USING gin(metadata jsonb_path_ops);

-- Composite index for common queries
CREATE INDEX IF NOT EXISTS idx_memories_user_namespace_created 
ON memories(user_id, namespace, created_at DESC);

-- ========================================
-- Part 3: Cost Tracking & Budgeting
-- ========================================

-- Per-request cost tracking
CREATE TABLE IF NOT EXISTS llm_cost_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    user_id UUID REFERENCES users(id),
    workflow_id UUID,
    task_id UUID,
    request_id VARCHAR(255),
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    cost_usd DECIMAL(10, 6) NOT NULL,
    latency_ms INTEGER,
    cache_hit BOOLEAN DEFAULT FALSE,
    cache_layer VARCHAR(20),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_cost_timestamp ON llm_cost_tracking(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_cost_user_id ON llm_cost_tracking(user_id);
CREATE INDEX IF NOT EXISTS idx_cost_workflow_id ON llm_cost_tracking(workflow_id);
CREATE INDEX IF NOT EXISTS idx_cost_provider_model ON llm_cost_tracking(provider, model);

-- User budget system
CREATE TABLE IF NOT EXISTS user_budgets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID UNIQUE REFERENCES users(id),
    total_budget_usd DECIMAL(10, 2) NOT NULL,
    used_budget_usd DECIMAL(10, 2) NOT NULL DEFAULT 0.00,
    soft_limit_usd DECIMAL(10, 2),  -- Warning threshold
    hard_limit_usd DECIMAL(10, 2),  -- Block threshold
    reset_period VARCHAR(20) NOT NULL DEFAULT 'monthly',  -- daily, weekly, monthly
    last_reset TIMESTAMP NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata JSONB
);

-- Budget alerts
CREATE TABLE IF NOT EXISTS budget_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    alert_type VARCHAR(50) NOT NULL,  -- soft_limit, hard_limit, overage
    threshold_usd DECIMAL(10, 2) NOT NULL,
    current_spend_usd DECIMAL(10, 2) NOT NULL,
    triggered_at TIMESTAMP NOT NULL DEFAULT NOW(),
    acknowledged BOOLEAN DEFAULT FALSE,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_budget_alerts_user_id ON budget_alerts(user_id);
CREATE INDEX IF NOT EXISTS idx_budget_alerts_triggered_at ON budget_alerts(triggered_at DESC);

-- ========================================
-- Part 4: Circuit Breaker State
-- ========================================

-- Circuit breaker state persistence
CREATE TABLE IF NOT EXISTS circuit_breaker_state (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    state VARCHAR(20) NOT NULL,  -- closed, open, half_open
    failure_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    last_failure_at TIMESTAMP,
    last_success_at TIMESTAMP,
    opened_at TIMESTAMP,
    last_state_change TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata JSONB
);

-- Circuit breaker metrics history
CREATE TABLE IF NOT EXISTS circuit_breaker_events (
    id SERIAL PRIMARY KEY,
    circuit_name VARCHAR(100) NOT NULL,
    event_type VARCHAR(50) NOT NULL,  -- state_change, failure, success, rejected
    old_state VARCHAR(20),
    new_state VARCHAR(20),
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_cb_events_name ON circuit_breaker_events(circuit_name);
CREATE INDEX IF NOT EXISTS idx_cb_events_timestamp ON circuit_breaker_events(timestamp DESC);

-- ========================================
-- Part 5: Performance Optimization Indexes
-- ========================================

-- Task execution optimization
CREATE INDEX IF NOT EXISTS idx_tasks_status_created ON tasks(status, created_at DESC) 
WHERE status IN ('pending', 'running');

-- Workflow execution optimization
CREATE INDEX IF NOT EXISTS idx_task_exec_logs_task_created ON task_execution_logs(task_id, created_at DESC);

-- Memory retrieval optimization
CREATE INDEX IF NOT EXISTS idx_memories_user_created ON memories(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(metadata->'importance') 
WHERE metadata->>'importance' IS NOT NULL;

-- Agent assignment optimization
CREATE INDEX IF NOT EXISTS idx_agent_assignments_status ON agent_task_assignments(status, assigned_at DESC);

-- ========================================
-- Part 6: Helper Functions
-- ========================================

-- Function to calculate cache hit rate
CREATE OR REPLACE FUNCTION calculate_cache_hit_rate(
    layer_name VARCHAR,
    time_window INTERVAL DEFAULT '1 hour'
) RETURNS DECIMAL(5, 4) AS $$
DECLARE
    hit_rate DECIMAL(5, 4);
BEGIN
    SELECT 
        CASE 
            WHEN SUM(total_requests) > 0 THEN
                SUM(cache_hits)::DECIMAL / SUM(total_requests)::DECIMAL
            ELSE 0
        END INTO hit_rate
    FROM llm_cache_stats
    WHERE layer = layer_name
    AND timestamp > NOW() - time_window;
    
    RETURN COALESCE(hit_rate, 0);
END;
$$ LANGUAGE plpgsql;

-- Function to check budget status
CREATE OR REPLACE FUNCTION check_budget_status(
    p_user_id UUID
) RETURNS TABLE(
    budget_usd DECIMAL(10, 2),
    used_usd DECIMAL(10, 2),
    remaining_usd DECIMAL(10, 2),
    utilization_percent DECIMAL(5, 2),
    status VARCHAR(20)
) AS $$
DECLARE
    v_budget user_budgets%ROWTYPE;
    v_remaining DECIMAL(10, 2);
    v_utilization DECIMAL(5, 2);
    v_status VARCHAR(20);
BEGIN
    -- Get budget
    SELECT * INTO v_budget FROM user_budgets WHERE user_id = p_user_id;
    
    IF NOT FOUND THEN
        RETURN QUERY SELECT 
            0::DECIMAL(10,2), 
            0::DECIMAL(10,2), 
            0::DECIMAL(10,2), 
            0::DECIMAL(5,2), 
            'no_budget'::VARCHAR;
        RETURN;
    END IF;
    
    -- Calculate metrics
    v_remaining := v_budget.total_budget_usd - v_budget.used_budget_usd;
    v_utilization := (v_budget.used_budget_usd / v_budget.total_budget_usd) * 100;
    
    -- Determine status
    IF v_budget.used_budget_usd >= v_budget.hard_limit_usd THEN
        v_status := 'blocked';
    ELSIF v_budget.used_budget_usd >= v_budget.soft_limit_usd THEN
        v_status := 'warning';
    ELSE
        v_status := 'ok';
    END IF;
    
    RETURN QUERY SELECT 
        v_budget.total_budget_usd,
        v_budget.used_budget_usd,
        v_remaining,
        v_utilization,
        v_status;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- Part 7: Sample Data & Configuration
-- ========================================

-- Insert default cache statistics
INSERT INTO llm_cache_stats (layer, total_requests, cache_hits, cache_misses, hit_rate, avg_latency_ms)
VALUES 
    ('l1_redis', 0, 0, 0, 0, 0),
    ('l2_database', 0, 0, 0, 0, 0),
    ('l3_semantic', 0, 0, 0, 0, 0),
    ('l4_llm_api', 0, 0, 0, 0, 0)
ON CONFLICT DO NOTHING;

-- Insert default circuit breaker state
INSERT INTO circuit_breaker_state (name, state)
VALUES 
    ('openai_api', 'closed'),
    ('anthropic_api', 'closed'),
    ('redis_cache', 'closed'),
    ('database', 'closed')
ON CONFLICT (name) DO NOTHING;

-- ========================================
-- Migration Complete
-- ========================================

COMMENT ON TABLE llm_cache IS 'Phase 5.2: L1/L2 exact match LLM cache';
COMMENT ON TABLE llm_semantic_cache IS 'Phase 5.2: L3 semantic similarity LLM cache';
COMMENT ON TABLE llm_cost_tracking IS 'Phase 5.3: Per-request cost tracking';
COMMENT ON TABLE user_budgets IS 'Phase 5.3: User budget management';
COMMENT ON TABLE circuit_breaker_state IS 'Phase 5.3: Circuit breaker state persistence';
