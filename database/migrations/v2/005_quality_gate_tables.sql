-- ============================================================================
-- Migration: 005 - AI Quality Gate Tables
-- Version: V2
-- Date: 2026-02-11
-- Description: Add AI quality gate tables for cross-agent verification and quality control
-- ============================================================================

-- Quality gate policies table: Define quality validation policies
CREATE TABLE quality_gate_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL UNIQUE,
    description TEXT,

    -- Quality thresholds
    min_quality_score FLOAT DEFAULT 0.7 CHECK (min_quality_score >= 0 AND min_quality_score <= 1),
    min_confidence_score FLOAT DEFAULT 0.7 CHECK (min_confidence_score >= 0 AND min_confidence_score <= 1),
    min_completeness FLOAT DEFAULT 0.8 CHECK (min_completeness >= 0 AND min_completeness <= 1),
    min_clarity FLOAT DEFAULT 0.7 CHECK (min_clarity >= 0 AND min_clarity <= 1),
    max_hallucination_rate FLOAT DEFAULT 0.1 CHECK (max_hallucination_rate >= 0 AND max_hallucination_rate <= 1),

    -- Verification requirements
    require_cross_verification BOOLEAN DEFAULT FALSE,
    require_self_critique BOOLEAN DEFAULT FALSE,
    cross_verification_agent_role VARCHAR(50),
    min_cross_verification_count INTEGER DEFAULT 1,

    -- Applicability
    agent_roles TEXT[],
    task_types TEXT[],

    -- Status
    is_active BOOLEAN DEFAULT TRUE,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Quality gate results table: Record quality gate evaluations
CREATE TABLE quality_gate_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES tasks(id),
    agent_id UUID REFERENCES agents(id),
    policy_id UUID REFERENCES quality_gate_policies(id),

    -- Input
    content TEXT NOT NULL,
    content_type VARCHAR(50),

    -- Evaluation results
    passed BOOLEAN NOT NULL,
    quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 1),
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    completeness_score FLOAT CHECK (completeness_score >= 0 AND completeness_score <= 1),
    clarity_score FLOAT CHECK (clarity_score >= 0 AND clarity_score <= 1),
    coherence_score FLOAT CHECK (coherence_score >= 0 AND coherence_score <= 1),
    safety_score FLOAT CHECK (safety_score >= 0 AND safety_score <= 1),

    -- Failure details
    failure_reason TEXT,
    violations JSONB,

    -- Recommendation
    recommendation VARCHAR(50) CHECK (recommendation IN ('accept', 'reject', 'regenerate', 'modify')),
    suggested_improvements TEXT,

    checked_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Cross-agent verifications table: Track cross-agent verification results
CREATE TABLE cross_agent_verifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    quality_gate_result_id UUID NOT NULL REFERENCES quality_gate_results(id) ON DELETE CASCADE,

    -- Original output
    original_agent_id UUID NOT NULL REFERENCES agents(id),
    original_content TEXT NOT NULL,

    -- Verification agent
    verification_agent_id UUID NOT NULL REFERENCES agents(id),
    verification_result TEXT,
    verification_passed BOOLEAN NOT NULL,

    -- Verification details
    issues_found TEXT[],
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),

    verified_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Self-critique sessions table: Track self-critique loops
CREATE TABLE self_critique_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    quality_gate_result_id UUID NOT NULL REFERENCES quality_gate_results(id) ON DELETE CASCADE,

    -- Agent
    agent_id UUID NOT NULL REFERENCES agents(id),

    -- Iterations
    iteration_count INTEGER NOT NULL DEFAULT 0,
    max_iterations INTEGER DEFAULT 3,

    -- Results
    final_content TEXT,
    improvement_score FLOAT,
    critical_flaws_found TEXT[],

    -- Status
    status VARCHAR(50) NOT NULL CHECK (status IN ('in_progress', 'completed', 'max_iterations_reached')),

    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- Self-critique iterations table: Individual critique iterations
CREATE TABLE self_critique_iterations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES self_critique_sessions(id) ON DELETE CASCADE,
    iteration_number INTEGER NOT NULL,

    -- Content
    content TEXT NOT NULL,
    critique TEXT NOT NULL,

    -- Scores
    quality_score FLOAT,
    flaws_detected TEXT[],

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    UNIQUE(session_id, iteration_number)
);

-- Indexes for performance
CREATE INDEX idx_quality_gate_policies_active ON quality_gate_policies(is_active);
CREATE INDEX idx_quality_gate_policies_agent_roles ON quality_gate_policies USING GIN(agent_roles);

CREATE INDEX idx_quality_gate_results_task_id ON quality_gate_results(task_id);
CREATE INDEX idx_quality_gate_results_agent_id ON quality_gate_results(agent_id);
CREATE INDEX idx_quality_gate_results_policy_id ON quality_gate_results(policy_id);
CREATE INDEX idx_quality_gate_results_passed ON quality_gate_results(passed);
CREATE INDEX idx_quality_gate_results_checked_at ON quality_gate_results(checked_at DESC);

CREATE INDEX idx_cross_agent_verifications_result_id ON cross_agent_verifications(quality_gate_result_id);
CREATE INDEX idx_cross_agent_verifications_original_agent ON cross_agent_verifications(original_agent_id);
CREATE INDEX idx_cross_agent_verifications_verification_agent ON cross_agent_verifications(verification_agent_id);
CREATE INDEX idx_cross_agent_verifications_passed ON cross_agent_verifications(verification_passed);

CREATE INDEX idx_self_critique_sessions_result_id ON self_critique_sessions(quality_gate_result_id);
CREATE INDEX idx_self_critique_sessions_agent_id ON self_critique_sessions(agent_id);
CREATE INDEX idx_self_critique_sessions_status ON self_critique_sessions(status);

CREATE INDEX idx_self_critique_iterations_session_id ON self_critique_iterations(session_id);

-- Trigger for updated_at
CREATE TRIGGER update_quality_gate_policies_updated_at
    BEFORE UPDATE ON quality_gate_policies
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create default quality gate policy
INSERT INTO quality_gate_policies (name, description, min_quality_score, min_confidence_score)
VALUES (
    'default_policy',
    'Default quality gate policy for all agents',
    0.7,
    0.7
);

-- Comments
COMMENT ON TABLE quality_gate_policies IS 'Quality validation policies for AI outputs';
COMMENT ON TABLE quality_gate_results IS 'Results of quality gate evaluations';
COMMENT ON TABLE cross_agent_verifications IS 'Cross-agent verification results';
COMMENT ON TABLE self_critique_sessions IS 'Self-critique loop sessions';
COMMENT ON TABLE self_critique_iterations IS 'Individual iterations within self-critique sessions';
COMMENT ON COLUMN quality_gate_results.violations IS 'JSONB array of specific policy violations';
COMMENT ON COLUMN cross_agent_verifications.verification_result IS 'Detailed verification output from critic agent';
COMMENT ON COLUMN self_critique_sessions.improvement_score IS 'How much the output improved through self-critique';
