-- Migration 006: Phase 6 Intelligence Layer - Advanced Learning & Self-Optimization
-- CognitionOS V5: Meta-Learning, Adaptive Optimization, Anomaly Detection

-- ========================================
-- Part 1: Execution History Tracking
-- ========================================

-- Comprehensive execution history for meta-learning
CREATE TABLE IF NOT EXISTS execution_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID,
    task_id UUID,
    task_type VARCHAR(100) NOT NULL,
    model_used VARCHAR(100) NOT NULL,
    cache_layer_hit VARCHAR(20),  -- l1_redis, l2_database, l3_semantic, l4_llm_api, null
    execution_time_ms INTEGER NOT NULL,
    cost_usd DECIMAL(10, 6) NOT NULL,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    context JSONB,  -- Additional context like task complexity, user preferences, etc.
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_execution_history_workflow ON execution_history(workflow_id);
CREATE INDEX IF NOT EXISTS idx_execution_history_task_type ON execution_history(task_type);
CREATE INDEX IF NOT EXISTS idx_execution_history_model ON execution_history(model_used);
CREATE INDEX IF NOT EXISTS idx_execution_history_created ON execution_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_execution_history_success ON execution_history(success);
CREATE INDEX IF NOT EXISTS idx_execution_history_cache_hit ON execution_history(cache_layer_hit);

-- Composite indexes for common analytics queries
CREATE INDEX IF NOT EXISTS idx_execution_history_task_model 
ON execution_history(task_type, model_used, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_execution_history_context_gin 
ON execution_history USING gin(context jsonb_path_ops);

-- ========================================
-- Part 2: ML Models & Training
-- ========================================

-- ML model registry
CREATE TABLE IF NOT EXISTS ml_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,  -- cache_optimizer, model_router, anomaly_detector, etc.
    accuracy DECIMAL(5, 4),
    precision_score DECIMAL(5, 4),
    recall_score DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    trained_at TIMESTAMP NOT NULL DEFAULT NOW(),
    training_samples INTEGER,
    model_artifact BYTEA,  -- Serialized model (pickle/joblib)
    hyperparameters JSONB,
    feature_importance JSONB,
    validation_metrics JSONB,
    status VARCHAR(20) NOT NULL DEFAULT 'training',  -- training, deployed, archived
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_ml_models_name_version ON ml_models(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_ml_models_type ON ml_models(model_type);
CREATE INDEX IF NOT EXISTS idx_ml_models_status ON ml_models(status);
CREATE INDEX IF NOT EXISTS idx_ml_models_trained_at ON ml_models(trained_at DESC);

-- Unique constraint for active models
CREATE UNIQUE INDEX IF NOT EXISTS idx_ml_models_active_unique 
ON ml_models(model_name, model_type) 
WHERE status = 'deployed';

-- ML training data
CREATE TABLE IF NOT EXISTS ml_training_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_type VARCHAR(50) NOT NULL,
    feature_vector JSONB NOT NULL,
    label VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_ml_training_model_type ON ml_training_data(model_type);
CREATE INDEX IF NOT EXISTS idx_ml_training_created ON ml_training_data(created_at DESC);

-- ========================================
-- Part 3: Adaptive Configuration
-- ========================================

-- Dynamic configuration that adapts based on learning
CREATE TABLE IF NOT EXISTS adaptive_config (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    optimization_score DECIMAL(10, 6),  -- How much this config improved performance
    previous_value JSONB,
    applied_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP,
    confidence_level DECIMAL(5, 4),  -- 0.0 to 1.0
    evaluation_period_hours INTEGER DEFAULT 24,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_adaptive_config_key ON adaptive_config(config_key);
CREATE INDEX IF NOT EXISTS idx_adaptive_config_applied ON adaptive_config(applied_at DESC);
CREATE INDEX IF NOT EXISTS idx_adaptive_config_score ON adaptive_config(optimization_score DESC NULLS LAST);

-- Configuration history for rollback
CREATE TABLE IF NOT EXISTS adaptive_config_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_key VARCHAR(100) NOT NULL,
    old_value JSONB,
    new_value JSONB NOT NULL,
    optimization_score DECIMAL(10, 6),
    changed_at TIMESTAMP NOT NULL DEFAULT NOW(),
    changed_by VARCHAR(100),  -- system, user, ml_model
    reason TEXT,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_adaptive_config_hist_key ON adaptive_config_history(config_key);
CREATE INDEX IF NOT EXISTS idx_adaptive_config_hist_changed ON adaptive_config_history(changed_at DESC);

-- ========================================
-- Part 4: Performance Anomaly Detection
-- ========================================

-- Baseline performance metrics
CREATE TABLE IF NOT EXISTS performance_baselines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,  -- latency, cost, error_rate, cache_hit_rate
    baseline_value DECIMAL(15, 6) NOT NULL,
    std_deviation DECIMAL(15, 6),
    min_value DECIMAL(15, 6),
    max_value DECIMAL(15, 6),
    percentile_95 DECIMAL(15, 6),
    percentile_99 DECIMAL(15, 6),
    sample_count INTEGER NOT NULL,
    calculated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    valid_until TIMESTAMP,
    context JSONB,  -- Context like time_of_day, task_type, etc.
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_perf_baselines_metric ON performance_baselines(metric_name, metric_type);
CREATE INDEX IF NOT EXISTS idx_perf_baselines_calculated ON performance_baselines(calculated_at DESC);

-- Unique constraint for active baselines
CREATE UNIQUE INDEX IF NOT EXISTS idx_perf_baselines_unique 
ON performance_baselines(metric_name, metric_type, context) 
WHERE valid_until IS NULL OR valid_until > NOW();

-- Detected anomalies
CREATE TABLE IF NOT EXISTS performance_anomalies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    expected_value DECIMAL(15, 6),
    actual_value DECIMAL(15, 6) NOT NULL,
    deviation_percent DECIMAL(10, 2),
    severity VARCHAR(20) NOT NULL,  -- info, warning, critical
    detected_at TIMESTAMP NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMP,
    resolution VARCHAR(50),  -- auto_recovered, manual_fix, false_positive
    root_cause TEXT,
    remediation_action TEXT,
    context JSONB,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_perf_anomalies_metric ON performance_anomalies(metric_name, metric_type);
CREATE INDEX IF NOT EXISTS idx_perf_anomalies_detected ON performance_anomalies(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_perf_anomalies_severity ON performance_anomalies(severity);
CREATE INDEX IF NOT EXISTS idx_perf_anomalies_unresolved ON performance_anomalies(resolved_at) 
WHERE resolved_at IS NULL;

-- ========================================
-- Part 5: Self-Healing Actions
-- ========================================

-- Record of self-healing actions taken
CREATE TABLE IF NOT EXISTS self_healing_actions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    action_type VARCHAR(50) NOT NULL,  -- circuit_breaker_reset, cache_clear, restart_service, etc.
    trigger_type VARCHAR(50) NOT NULL,  -- anomaly, circuit_open, manual
    trigger_id UUID,  -- Reference to anomaly or other trigger
    action_details JSONB NOT NULL,
    initiated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    success BOOLEAN,
    error_message TEXT,
    impact_assessment JSONB,  -- Before/after metrics
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_self_healing_type ON self_healing_actions(action_type);
CREATE INDEX IF NOT EXISTS idx_self_healing_initiated ON self_healing_actions(initiated_at DESC);
CREATE INDEX IF NOT EXISTS idx_self_healing_success ON self_healing_actions(success);

-- ========================================
-- Part 6: Intelligent Router Decisions
-- ========================================

-- Track model routing decisions for learning
CREATE TABLE IF NOT EXISTS model_routing_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID,
    task_type VARCHAR(100) NOT NULL,
    task_complexity DECIMAL(5, 4),  -- 0.0 to 1.0
    available_models JSONB NOT NULL,  -- List of models considered
    selected_model VARCHAR(100) NOT NULL,
    selection_reason VARCHAR(200),
    predicted_cost DECIMAL(10, 6),
    actual_cost DECIMAL(10, 6),
    predicted_quality DECIMAL(5, 4),
    actual_quality DECIMAL(5, 4),  -- Based on success/failure
    decision_confidence DECIMAL(5, 4),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_routing_decisions_task_type ON model_routing_decisions(task_type);
CREATE INDEX IF NOT EXISTS idx_routing_decisions_model ON model_routing_decisions(selected_model);
CREATE INDEX IF NOT EXISTS idx_routing_decisions_created ON model_routing_decisions(created_at DESC);

-- Composite index for model performance analysis
CREATE INDEX IF NOT EXISTS idx_routing_decisions_analysis 
ON model_routing_decisions(task_type, selected_model, created_at DESC);

-- ========================================
-- Part 7: Cache Optimization Tracking
-- ========================================

-- Track cache TTL optimization decisions
CREATE TABLE IF NOT EXISTS cache_optimization_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cache_layer VARCHAR(20) NOT NULL,
    cache_key_pattern VARCHAR(255),
    old_ttl_seconds INTEGER NOT NULL,
    new_ttl_seconds INTEGER NOT NULL,
    optimization_reason TEXT,
    predicted_hit_rate DECIMAL(5, 4),
    actual_hit_rate DECIMAL(5, 4),
    cost_impact DECIMAL(10, 6),  -- Positive = savings, Negative = increase
    applied_at TIMESTAMP NOT NULL DEFAULT NOW(),
    evaluated_at TIMESTAMP,
    kept BOOLEAN,  -- Whether the optimization was kept or reverted
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_cache_opt_layer ON cache_optimization_decisions(cache_layer);
CREATE INDEX IF NOT EXISTS idx_cache_opt_applied ON cache_optimization_decisions(applied_at DESC);
CREATE INDEX IF NOT EXISTS idx_cache_opt_kept ON cache_optimization_decisions(kept);

-- ========================================
-- Part 8: Helper Functions
-- ========================================

-- Calculate execution success rate by task type
CREATE OR REPLACE FUNCTION calculate_task_success_rate(
    p_task_type VARCHAR,
    time_window INTERVAL DEFAULT '24 hours'
) RETURNS DECIMAL(5, 4) AS $$
DECLARE
    success_rate DECIMAL(5, 4);
BEGIN
    SELECT 
        CASE 
            WHEN COUNT(*) > 0 THEN
                COUNT(*) FILTER (WHERE success = true)::DECIMAL / COUNT(*)::DECIMAL
            ELSE 0
        END INTO success_rate
    FROM execution_history
    WHERE task_type = p_task_type
    AND created_at > NOW() - time_window;
    
    RETURN COALESCE(success_rate, 0);
END;
$$ LANGUAGE plpgsql;

-- Get optimal model for task type based on historical performance
CREATE OR REPLACE FUNCTION get_optimal_model(
    p_task_type VARCHAR,
    max_cost_usd DECIMAL DEFAULT NULL
) RETURNS TABLE(
    model VARCHAR,
    avg_cost DECIMAL(10, 6),
    success_rate DECIMAL(5, 4),
    avg_execution_time INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        eh.model_used,
        AVG(eh.cost_usd)::DECIMAL(10, 6) as avg_cost,
        (COUNT(*) FILTER (WHERE eh.success = true)::DECIMAL / COUNT(*)::DECIMAL)::DECIMAL(5, 4) as success_rate,
        AVG(eh.execution_time_ms)::INTEGER as avg_execution_time
    FROM execution_history eh
    WHERE eh.task_type = p_task_type
    AND eh.created_at > NOW() - INTERVAL '7 days'
    AND (max_cost_usd IS NULL OR eh.cost_usd <= max_cost_usd)
    GROUP BY eh.model_used
    HAVING COUNT(*) >= 10  -- Minimum sample size
    ORDER BY success_rate DESC, avg_cost ASC
    LIMIT 3;
END;
$$ LANGUAGE plpgsql;

-- Detect if current performance is anomalous
CREATE OR REPLACE FUNCTION is_performance_anomalous(
    p_metric_name VARCHAR,
    p_metric_type VARCHAR,
    p_current_value DECIMAL,
    p_threshold_std_devs DECIMAL DEFAULT 3.0
) RETURNS BOOLEAN AS $$
DECLARE
    v_baseline performance_baselines%ROWTYPE;
    v_deviation DECIMAL;
    v_is_anomaly BOOLEAN;
BEGIN
    -- Get active baseline
    SELECT * INTO v_baseline
    FROM performance_baselines
    WHERE metric_name = p_metric_name
    AND metric_type = p_metric_type
    AND (valid_until IS NULL OR valid_until > NOW())
    ORDER BY calculated_at DESC
    LIMIT 1;
    
    IF NOT FOUND THEN
        RETURN false;  -- No baseline, can't detect anomalies
    END IF;
    
    -- Calculate deviation in standard deviations
    IF v_baseline.std_deviation > 0 THEN
        v_deviation := ABS(p_current_value - v_baseline.baseline_value) / v_baseline.std_deviation;
        v_is_anomaly := v_deviation > p_threshold_std_devs;
    ELSE
        -- Check if value is outside min/max range
        v_is_anomaly := p_current_value < v_baseline.min_value OR p_current_value > v_baseline.max_value;
    END IF;
    
    RETURN v_is_anomaly;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- Part 9: Initial Data & Configuration
-- ========================================

-- Insert default adaptive configurations
INSERT INTO adaptive_config (config_key, config_value, confidence_level)
VALUES 
    ('cache.l1_redis.default_ttl_seconds', '300', 0.5),
    ('cache.l2_database.default_ttl_seconds', '3600', 0.5),
    ('cache.l3_semantic.default_ttl_seconds', '86400', 0.5),
    ('cache.l3_semantic.similarity_threshold', '0.92', 0.5),
    ('router.default_model', '"gpt-3.5-turbo"', 0.5),
    ('router.complexity_threshold_gpt4', '0.7', 0.5),
    ('anomaly.detection_threshold_std_devs', '3.0', 0.8),
    ('self_healing.enabled', 'true', 0.9),
    ('self_healing.circuit_breaker_auto_reset', 'true', 0.9)
ON CONFLICT (config_key) DO NOTHING;

-- ========================================
-- Migration Complete
-- ========================================

COMMENT ON TABLE execution_history IS 'Phase 6.1: Comprehensive execution tracking for meta-learning';
COMMENT ON TABLE ml_models IS 'Phase 6.1: ML model registry and versioning';
COMMENT ON TABLE adaptive_config IS 'Phase 6.1: Self-optimizing configuration system';
COMMENT ON TABLE performance_baselines IS 'Phase 6.2: Performance baseline tracking for anomaly detection';
COMMENT ON TABLE performance_anomalies IS 'Phase 6.2: Detected performance anomalies';
COMMENT ON TABLE self_healing_actions IS 'Phase 6.2: Self-healing action history';
COMMENT ON TABLE model_routing_decisions IS 'Phase 6.3: Intelligent model routing decisions';
COMMENT ON TABLE cache_optimization_decisions IS 'Phase 6.3: Adaptive cache optimization tracking';
