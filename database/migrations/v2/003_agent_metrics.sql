-- ============================================================================
-- Migration: 003 - Agent Metrics Tables
-- Version: V2
-- Date: 2026-02-11
-- Description: Add agent performance metrics tracking for V2 agent evolution
-- ============================================================================

-- Agent metrics table: Track agent performance over time windows
CREATE TABLE agent_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    time_window_start TIMESTAMP NOT NULL,
    time_window_end TIMESTAMP NOT NULL,
    task_count INTEGER NOT NULL DEFAULT 0,

    -- Quality metrics
    avg_confidence FLOAT,
    avg_quality_score FLOAT,
    hallucination_rate FLOAT,

    -- Performance metrics
    avg_latency_ms INTEGER,
    p50_latency_ms INTEGER,
    p95_latency_ms INTEGER,
    p99_latency_ms INTEGER,

    -- Cost metrics
    avg_cost_per_task FLOAT,
    total_cost FLOAT,
    total_tokens_used BIGINT,

    -- Reliability metrics
    success_rate FLOAT,
    retry_rate FLOAT,
    failure_rate FLOAT,
    timeout_rate FLOAT,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_time_window CHECK (time_window_end > time_window_start),
    CONSTRAINT valid_rates CHECK (
        success_rate >= 0 AND success_rate <= 1 AND
        retry_rate >= 0 AND retry_rate <= 1 AND
        failure_rate >= 0 AND failure_rate <= 1 AND
        timeout_rate >= 0 AND timeout_rate <= 1
    )
);

-- Agent performance history: Detailed performance tracking
CREATE TABLE agent_performance_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    task_id UUID REFERENCES tasks(id),

    -- Performance data
    quality_score FLOAT,
    confidence FLOAT,
    latency_ms INTEGER,
    token_count INTEGER,
    cost FLOAT,

    -- Status
    status VARCHAR(50) NOT NULL,
    error_type VARCHAR(100),
    retry_count INTEGER DEFAULT 0,

    -- Timestamps
    executed_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_scores CHECK (
        (quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 1)) AND
        (confidence IS NULL OR (confidence >= 0 AND confidence <= 1))
    )
);

-- Agent replacement log: Track agent replacements for underperformance
CREATE TABLE agent_replacement_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    old_agent_id UUID REFERENCES agents(id),
    new_agent_id UUID REFERENCES agents(id),

    -- Replacement reason
    reason TEXT NOT NULL,
    trigger_type VARCHAR(50) NOT NULL CHECK (trigger_type IN ('manual', 'automatic', 'policy')),

    -- Metrics that triggered replacement
    metrics_snapshot JSONB,

    -- Timestamps
    replaced_at TIMESTAMP NOT NULL DEFAULT NOW(),
    replaced_by UUID REFERENCES users(id)
);

-- Indexes for performance
CREATE INDEX idx_agent_metrics_agent_id ON agent_metrics(agent_id);
CREATE INDEX idx_agent_metrics_time_window ON agent_metrics(time_window_start, time_window_end);
CREATE INDEX idx_agent_metrics_created_at ON agent_metrics(created_at DESC);

CREATE INDEX idx_agent_performance_history_agent_id ON agent_performance_history(agent_id);
CREATE INDEX idx_agent_performance_history_task_id ON agent_performance_history(task_id);
CREATE INDEX idx_agent_performance_history_status ON agent_performance_history(status);
CREATE INDEX idx_agent_performance_history_executed_at ON agent_performance_history(executed_at DESC);

CREATE INDEX idx_agent_replacement_log_old_agent ON agent_replacement_log(old_agent_id);
CREATE INDEX idx_agent_replacement_log_new_agent ON agent_replacement_log(new_agent_id);
CREATE INDEX idx_agent_replacement_log_replaced_at ON agent_replacement_log(replaced_at DESC);

-- Trigger for updated_at
CREATE TRIGGER update_agent_metrics_updated_at
    BEFORE UPDATE ON agent_metrics
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments
COMMENT ON TABLE agent_metrics IS 'Aggregated agent performance metrics over time windows';
COMMENT ON TABLE agent_performance_history IS 'Detailed performance data for each task execution';
COMMENT ON TABLE agent_replacement_log IS 'History of agent replacements due to performance issues';
COMMENT ON COLUMN agent_metrics.hallucination_rate IS 'Percentage of responses flagged as hallucinations';
COMMENT ON COLUMN agent_metrics.p50_latency_ms IS '50th percentile latency in milliseconds';
COMMENT ON COLUMN agent_metrics.p95_latency_ms IS '95th percentile latency in milliseconds';
COMMENT ON COLUMN agent_metrics.p99_latency_ms IS '99th percentile latency in milliseconds';
