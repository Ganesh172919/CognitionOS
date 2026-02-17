-- CognitionOS Execution Persistence
-- Migration: 008_execution_persistence
-- Created: 2026-02-17
-- Description: P0 execution persistence for deterministic execution, replay, and resume

-- ============================================================================
-- EXECUTION PERSISTENCE (Step Attempts with Idempotency)
-- ============================================================================

-- Track every attempt to execute a step (before/after)
CREATE TABLE IF NOT EXISTS step_execution_attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    step_execution_id UUID NOT NULL REFERENCES step_executions(id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL,
    idempotency_key VARCHAR(255) NOT NULL UNIQUE,  -- For retry-safe operations

    -- State before execution
    inputs JSONB NOT NULL,
    agent_id UUID,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- State after execution
    outputs JSONB,
    error TEXT,
    status VARCHAR(50) NOT NULL,  -- success, failed, timeout
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER,

    -- For deterministic replay
    is_deterministic BOOLEAN DEFAULT TRUE,
    nondeterminism_flags TEXT[],  -- e.g., ['external_api', 'timestamp', 'random']

    -- Request/response payloads for replay
    request_payload JSONB,
    response_payload JSONB,
    response_hash VARCHAR(64),  -- SHA-256 hash for comparison

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_step_attempt UNIQUE (step_execution_id, attempt_number)
);

CREATE INDEX idx_step_attempts_execution_id ON step_execution_attempts(step_execution_id);
CREATE INDEX idx_step_attempts_idempotency_key ON step_execution_attempts(idempotency_key);
CREATE INDEX idx_step_attempts_status ON step_execution_attempts(status);
CREATE INDEX idx_step_attempts_started_at ON step_execution_attempts(started_at);

-- ============================================================================
-- EXECUTION SNAPSHOTS (for Resume)
-- ============================================================================

-- Periodic snapshots of execution state for fast resume
CREATE TABLE IF NOT EXISTS execution_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID NOT NULL REFERENCES workflow_executions(id) ON DELETE CASCADE,
    snapshot_type VARCHAR(50) NOT NULL,  -- 'checkpoint', 'before_step', 'after_step'

    -- Complete execution state
    workflow_state JSONB NOT NULL,  -- Full workflow execution state
    step_states JSONB NOT NULL,  -- All step execution states
    variables JSONB NOT NULL DEFAULT '{}',  -- Execution context variables

    -- Progress tracking
    completed_steps TEXT[] DEFAULT '{}',
    pending_steps TEXT[] DEFAULT '{}',
    failed_steps TEXT[] DEFAULT '{}',

    -- Snapshot metadata
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_by VARCHAR(50) DEFAULT 'system',  -- 'system' or 'user'
    snapshot_size_bytes INTEGER,

    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_execution_snapshots_execution_id ON execution_snapshots(execution_id);
CREATE INDEX idx_execution_snapshots_type ON execution_snapshots(snapshot_type);
CREATE INDEX idx_execution_snapshots_created_at ON execution_snapshots(created_at);

-- ============================================================================
-- REPLAY SESSIONS
-- ============================================================================

-- Track replay executions for comparison with original
CREATE TABLE IF NOT EXISTS replay_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    original_execution_id UUID NOT NULL REFERENCES workflow_executions(id),
    replay_execution_id UUID NOT NULL REFERENCES workflow_executions(id),

    -- Replay configuration
    replay_mode VARCHAR(50) NOT NULL,  -- 'full', 'from_step', 'failed_only'
    start_from_step VARCHAR(255),
    use_cached_outputs BOOLEAN DEFAULT TRUE,

    -- Comparison results
    match_percentage DECIMAL(5,2),  -- % of outputs that matched
    divergence_details JSONB DEFAULT '{}',

    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Metadata
    triggered_by UUID,  -- user_id
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_replay_sessions_original_execution ON replay_sessions(original_execution_id);
CREATE INDEX idx_replay_sessions_replay_execution ON replay_sessions(replay_execution_id);
CREATE INDEX idx_replay_sessions_status ON replay_sessions(status);

-- ============================================================================
-- UNIFIED ERROR MODEL
-- ============================================================================

-- Standardized error tracking across all services
CREATE TABLE IF NOT EXISTS execution_errors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Error classification
    error_code VARCHAR(100) NOT NULL,  -- e.g., 'WORKFLOW_VALIDATION_ERROR'
    error_category VARCHAR(50) NOT NULL,  -- 'validation', 'execution', 'external', 'timeout'
    severity VARCHAR(20) NOT NULL,  -- 'low', 'medium', 'high', 'critical'
    is_retryable BOOLEAN NOT NULL DEFAULT FALSE,

    -- Error details
    message TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    stack_trace TEXT,

    -- Context
    service_name VARCHAR(100),
    correlation_id UUID NOT NULL,  -- Links to execution_id or trace_id
    execution_id UUID REFERENCES workflow_executions(id) ON DELETE CASCADE,
    step_execution_id UUID REFERENCES step_executions(id) ON DELETE CASCADE,
    user_id UUID,

    -- Retry information
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    next_retry_at TIMESTAMP WITH TIME ZONE,

    -- Resolution
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolution_notes TEXT,

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_execution_errors_correlation_id ON execution_errors(correlation_id);
CREATE INDEX idx_execution_errors_execution_id ON execution_errors(execution_id);
CREATE INDEX idx_execution_errors_step_execution_id ON execution_errors(step_execution_id);
CREATE INDEX idx_execution_errors_error_code ON execution_errors(error_code);
CREATE INDEX idx_execution_errors_category ON execution_errors(error_category);
CREATE INDEX idx_execution_errors_severity ON execution_errors(severity);
CREATE INDEX idx_execution_errors_retryable ON execution_errors(is_retryable);
CREATE INDEX idx_execution_errors_resolved ON execution_errors(resolved);
CREATE INDEX idx_execution_errors_created_at ON execution_errors(created_at);

-- ============================================================================
-- CORRELATION IDS
-- ============================================================================

-- Add correlation_id to existing tables for distributed tracing
ALTER TABLE workflow_executions ADD COLUMN IF NOT EXISTS correlation_id UUID DEFAULT gen_random_uuid();
ALTER TABLE step_executions ADD COLUMN IF NOT EXISTS correlation_id UUID DEFAULT gen_random_uuid();

CREATE INDEX IF NOT EXISTS idx_workflow_executions_correlation ON workflow_executions(correlation_id);
CREATE INDEX IF NOT EXISTS idx_step_executions_correlation ON step_executions(correlation_id);

-- ============================================================================
-- EXECUTION LOCKS (for distributed execution)
-- ============================================================================

-- Prevent concurrent execution of same workflow/step
CREATE TABLE IF NOT EXISTS execution_locks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    lock_key VARCHAR(255) NOT NULL UNIQUE,  -- e.g., 'execution:{execution_id}:step:{step_id}'
    lock_holder VARCHAR(255) NOT NULL,  -- worker/process ID
    acquired_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_execution_locks_key ON execution_locks(lock_key);
CREATE INDEX idx_execution_locks_expires_at ON execution_locks(expires_at);

-- Clean up expired locks (run periodically)
CREATE OR REPLACE FUNCTION cleanup_expired_locks()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM execution_locks WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEWS FOR QUERY CONVENIENCE
-- ============================================================================

-- View: Execution with latest attempt
CREATE OR REPLACE VIEW step_executions_with_latest_attempt AS
SELECT
    se.*,
    sea.attempt_number AS latest_attempt,
    sea.idempotency_key AS latest_idempotency_key,
    sea.is_deterministic,
    sea.duration_ms AS latest_duration_ms
FROM step_executions se
LEFT JOIN LATERAL (
    SELECT * FROM step_execution_attempts
    WHERE step_execution_id = se.id
    ORDER BY attempt_number DESC
    LIMIT 1
) sea ON true;

-- View: Execution timeline
CREATE OR REPLACE VIEW execution_timeline AS
SELECT
    we.id AS execution_id,
    we.workflow_id,
    we.status AS execution_status,
    se.id AS step_execution_id,
    se.step_id,
    se.status AS step_status,
    sea.attempt_number,
    sea.started_at,
    sea.completed_at,
    sea.duration_ms,
    sea.status AS attempt_status
FROM workflow_executions we
JOIN step_executions se ON se.execution_id = we.id
LEFT JOIN step_execution_attempts sea ON sea.step_execution_id = se.id
ORDER BY we.created_at DESC, se.step_id, sea.attempt_number;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE step_execution_attempts IS 'P0: Tracks every execution attempt for idempotency and replay';
COMMENT ON TABLE execution_snapshots IS 'P0: Periodic snapshots for fast resume from last checkpoint';
COMMENT ON TABLE replay_sessions IS 'P0: Tracks replay executions and comparison with originals';
COMMENT ON TABLE execution_errors IS 'P0: Unified error model across all services';
COMMENT ON TABLE execution_locks IS 'P0: Distributed locks to prevent concurrent execution';
