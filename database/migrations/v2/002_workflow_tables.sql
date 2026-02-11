-- ============================================================================
-- Migration: 002 - Workflow Tables
-- Version: V2
-- Date: 2026-02-11
-- Description: Add workflow engine tables for declarative workflow execution
-- ============================================================================

-- Workflows table: Stores workflow definitions
CREATE TABLE workflows (
    id VARCHAR(200) NOT NULL,
    version VARCHAR(50) NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    definition JSONB NOT NULL,
    schedule VARCHAR(100),
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    tags TEXT[],
    PRIMARY KEY (id, version)
);

-- Workflow executions table: Stores workflow execution instances
CREATE TABLE workflow_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id VARCHAR(200) NOT NULL,
    workflow_version VARCHAR(50) NOT NULL,
    inputs JSONB NOT NULL DEFAULT '{}',
    outputs JSONB,
    status VARCHAR(50) NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error TEXT,
    user_id UUID REFERENCES users(id),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Workflow execution steps table: Stores individual step executions
CREATE TABLE workflow_execution_steps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID NOT NULL REFERENCES workflow_executions(id) ON DELETE CASCADE,
    step_id VARCHAR(200) NOT NULL,
    step_type VARCHAR(100) NOT NULL,
    step_name VARCHAR(200),
    status VARCHAR(50) NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped', 'cancelled')),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    output JSONB,
    error TEXT,
    agent_id UUID REFERENCES agents(id),
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_workflows_name ON workflows(name);
CREATE INDEX idx_workflows_created_by ON workflows(created_by);
CREATE INDEX idx_workflows_is_active ON workflows(is_active);
CREATE INDEX idx_workflows_schedule ON workflows(schedule) WHERE schedule IS NOT NULL;
CREATE INDEX idx_workflows_tags ON workflows USING GIN(tags);

CREATE INDEX idx_workflow_executions_workflow ON workflow_executions(workflow_id, workflow_version);
CREATE INDEX idx_workflow_executions_user_id ON workflow_executions(user_id);
CREATE INDEX idx_workflow_executions_status ON workflow_executions(status);
CREATE INDEX idx_workflow_executions_created_at ON workflow_executions(created_at DESC);
CREATE INDEX idx_workflow_executions_started_at ON workflow_executions(started_at) WHERE started_at IS NOT NULL;

CREATE INDEX idx_workflow_execution_steps_execution_id ON workflow_execution_steps(execution_id);
CREATE INDEX idx_workflow_execution_steps_step_id ON workflow_execution_steps(step_id);
CREATE INDEX idx_workflow_execution_steps_status ON workflow_execution_steps(status);
CREATE INDEX idx_workflow_execution_steps_agent_id ON workflow_execution_steps(agent_id) WHERE agent_id IS NOT NULL;

-- Triggers for updated_at timestamps
CREATE TRIGGER update_workflows_updated_at
    BEFORE UPDATE ON workflows
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workflow_executions_updated_at
    BEFORE UPDATE ON workflow_executions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_workflow_execution_steps_updated_at
    BEFORE UPDATE ON workflow_execution_steps
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments
COMMENT ON TABLE workflows IS 'Workflow definitions (DSL)';
COMMENT ON TABLE workflow_executions IS 'Workflow execution instances';
COMMENT ON TABLE workflow_execution_steps IS 'Individual step executions within workflows';
COMMENT ON COLUMN workflows.definition IS 'Full workflow definition in JSONB format';
COMMENT ON COLUMN workflows.schedule IS 'Cron expression for scheduled workflows';
COMMENT ON COLUMN workflow_executions.inputs IS 'Input parameters for workflow execution';
COMMENT ON COLUMN workflow_executions.outputs IS 'Output values from workflow execution';
COMMENT ON COLUMN workflow_execution_steps.output IS 'Output data from step execution';
