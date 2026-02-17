-- Migration 009: Multi-Tenancy and Billing Infrastructure
-- Date: 2026-02-17
-- Description: Add comprehensive multi-tenancy support with billing, subscriptions, and usage metering

-- ============================================================================
-- TENANTS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(255) UNIQUE NOT NULL,
    status VARCHAR(50) NOT NULL CHECK (status IN ('active', 'suspended', 'trial', 'churned', 'pending')),
    subscription_tier VARCHAR(50) NOT NULL CHECK (subscription_tier IN ('free', 'pro', 'team', 'enterprise')),
    
    -- Settings (stored as JSONB for flexibility)
    settings JSONB NOT NULL DEFAULT '{}'::jsonb,
    
    -- Ownership
    owner_user_id UUID REFERENCES users(id),
    billing_email VARCHAR(255),
    
    -- Lifecycle timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    trial_ends_at TIMESTAMP,
    suspended_at TIMESTAMP,
    suspended_reason TEXT,
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Indexes
    CONSTRAINT unique_slug UNIQUE (slug)
);

CREATE INDEX idx_tenants_status ON tenants(status);
CREATE INDEX idx_tenants_subscription_tier ON tenants(subscription_tier);
CREATE INDEX idx_tenants_owner_user_id ON tenants(owner_user_id);
CREATE INDEX idx_tenants_created_at ON tenants(created_at);

-- ============================================================================
-- SUBSCRIPTIONS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    
    -- Subscription details
    tier VARCHAR(50) NOT NULL CHECK (tier IN ('free', 'pro', 'team', 'enterprise')),
    status VARCHAR(50) NOT NULL CHECK (status IN ('active', 'trialing', 'past_due', 'canceled', 'unpaid', 'paused')),
    
    -- Stripe integration
    stripe_subscription_id VARCHAR(255) UNIQUE,
    stripe_customer_id VARCHAR(255),
    
    -- Billing period
    current_period_start TIMESTAMP NOT NULL,
    current_period_end TIMESTAMP NOT NULL,
    trial_start TIMESTAMP,
    trial_end TIMESTAMP,
    
    -- Cancellation
    canceled_at TIMESTAMP,
    cancel_at_period_end BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Pricing
    amount_cents INTEGER NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'usd',
    billing_cycle VARCHAR(20) NOT NULL CHECK (billing_cycle IN ('monthly', 'yearly')),
    
    -- Payment method (stored as JSONB)
    payment_method JSONB,
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    
    CONSTRAINT unique_tenant_active_subscription UNIQUE (tenant_id, status) WHERE status = 'active'
);

CREATE INDEX idx_subscriptions_tenant_id ON subscriptions(tenant_id);
CREATE INDEX idx_subscriptions_status ON subscriptions(status);
CREATE INDEX idx_subscriptions_stripe_subscription_id ON subscriptions(stripe_subscription_id);
CREATE INDEX idx_subscriptions_trial_end ON subscriptions(trial_end) WHERE trial_end IS NOT NULL;
CREATE INDEX idx_subscriptions_current_period_end ON subscriptions(current_period_end);

-- ============================================================================
-- INVOICES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS invoices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    subscription_id UUID NOT NULL REFERENCES subscriptions(id) ON DELETE CASCADE,
    
    -- Invoice details
    status VARCHAR(50) NOT NULL CHECK (status IN ('draft', 'open', 'paid', 'void', 'uncollectible')),
    invoice_number VARCHAR(100) UNIQUE NOT NULL,
    
    -- Amounts in cents
    amount_cents INTEGER NOT NULL,
    amount_paid_cents INTEGER NOT NULL DEFAULT 0,
    amount_due_cents INTEGER NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'usd',
    
    -- Stripe integration
    stripe_invoice_id VARCHAR(255) UNIQUE,
    
    -- Billing period
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    
    -- Due date and payment
    due_date TIMESTAMP,
    paid_at TIMESTAMP,
    
    -- Line items (stored as JSONB)
    line_items JSONB NOT NULL DEFAULT '[]'::jsonb,
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_invoices_tenant_id ON invoices(tenant_id);
CREATE INDEX idx_invoices_subscription_id ON invoices(subscription_id);
CREATE INDEX idx_invoices_status ON invoices(status);
CREATE INDEX idx_invoices_stripe_invoice_id ON invoices(stripe_invoice_id);
CREATE INDEX idx_invoices_due_date ON invoices(due_date) WHERE due_date IS NOT NULL;
CREATE INDEX idx_invoices_paid_at ON invoices(paid_at) WHERE paid_at IS NOT NULL;
CREATE INDEX idx_invoices_created_at ON invoices(created_at);

-- ============================================================================
-- USAGE RECORDS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS usage_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    
    -- Resource tracking
    resource_type VARCHAR(100) NOT NULL,  -- 'executions', 'tokens', 'storage', 'api_calls', 'plugin_calls'
    quantity DECIMAL(20, 6) NOT NULL,
    unit VARCHAR(50) NOT NULL,
    
    -- Timestamp
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Partitioning by month for scalability
CREATE INDEX idx_usage_records_tenant_id_timestamp ON usage_records(tenant_id, timestamp DESC);
CREATE INDEX idx_usage_records_resource_type ON usage_records(resource_type);
CREATE INDEX idx_usage_records_timestamp ON usage_records(timestamp DESC);

-- ============================================================================
-- TENANT ISOLATION: Add tenant_id to existing tables
-- ============================================================================

-- Add tenant_id to workflows table
ALTER TABLE workflows ADD COLUMN IF NOT EXISTS tenant_id UUID REFERENCES tenants(id);
CREATE INDEX IF NOT EXISTS idx_workflows_tenant_id ON workflows(tenant_id);

-- Add tenant_id to agents table
ALTER TABLE agents ADD COLUMN IF NOT EXISTS tenant_id UUID REFERENCES tenants(id);
CREATE INDEX IF NOT EXISTS idx_agents_tenant_id ON agents(tenant_id);

-- Add tenant_id to memory_entries table
ALTER TABLE memory_entries ADD COLUMN IF NOT EXISTS tenant_id UUID REFERENCES tenants(id);
CREATE INDEX IF NOT EXISTS idx_memory_entries_tenant_id ON memory_entries(tenant_id);

-- Add tenant_id to checkpoints table
ALTER TABLE checkpoints ADD COLUMN IF NOT EXISTS tenant_id UUID REFERENCES tenants(id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_tenant_id ON checkpoints(tenant_id);

-- Add tenant_id to executions table
ALTER TABLE executions ADD COLUMN IF NOT EXISTS tenant_id UUID REFERENCES tenants(id);
CREATE INDEX IF NOT EXISTS idx_executions_tenant_id ON executions(tenant_id);

-- Add tenant_id to users table (for multi-tenant user management)
ALTER TABLE users ADD COLUMN IF NOT EXISTS tenant_id UUID REFERENCES tenants(id);
CREATE INDEX IF NOT EXISTS idx_users_tenant_id ON users(tenant_id);

-- ============================================================================
-- API KEYS TABLE (for API authentication)
-- ============================================================================

CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    
    -- Key details
    key_hash VARCHAR(255) NOT NULL UNIQUE,  -- Hashed API key
    key_prefix VARCHAR(20) NOT NULL,  -- First few chars for identification
    name VARCHAR(255) NOT NULL,
    
    -- Permissions
    scopes TEXT[] NOT NULL DEFAULT '{}',  -- Array of permission scopes
    
    -- Rate limiting
    rate_limit_per_minute INTEGER DEFAULT 60,
    
    -- Status
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    last_used_at TIMESTAMP,
    
    -- Expiration
    expires_at TIMESTAMP,
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    created_by_user_id UUID REFERENCES users(id),
    revoked_at TIMESTAMP,
    revoked_by_user_id UUID REFERENCES users(id),
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_api_keys_tenant_id ON api_keys(tenant_id);
CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_key_prefix ON api_keys(key_prefix);
CREATE INDEX idx_api_keys_is_active ON api_keys(is_active) WHERE is_active = TRUE;

-- ============================================================================
-- RATE LIMIT TRACKING TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS rate_limit_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    resource_key VARCHAR(255) NOT NULL,  -- e.g., 'api_calls', 'executions'
    
    -- Tracking window
    window_start TIMESTAMP NOT NULL,
    window_duration_seconds INTEGER NOT NULL,
    
    -- Counts
    request_count INTEGER NOT NULL DEFAULT 0,
    blocked_count INTEGER NOT NULL DEFAULT 0,
    
    -- Last update
    last_request_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT unique_rate_limit_window UNIQUE (tenant_id, resource_key, window_start)
);

CREATE INDEX idx_rate_limit_tracking_tenant_resource ON rate_limit_tracking(tenant_id, resource_key, window_start);
CREATE INDEX idx_rate_limit_tracking_window_start ON rate_limit_tracking(window_start);

-- ============================================================================
-- FEATURE FLAGS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS feature_flags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Feature details
    key VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Rollout configuration
    enabled_globally BOOLEAN NOT NULL DEFAULT FALSE,
    enabled_tiers TEXT[] DEFAULT '{}',  -- Array of enabled tiers
    enabled_tenant_ids UUID[] DEFAULT '{}',  -- Specific tenants
    
    -- Rollout percentage (0-100)
    rollout_percentage INTEGER CHECK (rollout_percentage >= 0 AND rollout_percentage <= 100),
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_feature_flags_key ON feature_flags(key);
CREATE INDEX idx_feature_flags_enabled_globally ON feature_flags(enabled_globally);

-- ============================================================================
-- AUDIT LOG FOR BILLING EVENTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS billing_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    
    -- Event details
    event_type VARCHAR(100) NOT NULL,  -- 'subscription_created', 'tier_upgraded', 'invoice_paid', etc.
    entity_type VARCHAR(100) NOT NULL,  -- 'subscription', 'invoice', 'usage_record'
    entity_id UUID NOT NULL,
    
    -- Actor
    user_id UUID REFERENCES users(id),
    
    -- Changes
    changes JSONB,  -- Before/after values
    
    -- Timestamp
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_billing_audit_log_tenant_id ON billing_audit_log(tenant_id);
CREATE INDEX idx_billing_audit_log_event_type ON billing_audit_log(event_type);
CREATE INDEX idx_billing_audit_log_entity_type_id ON billing_audit_log(entity_type, entity_id);
CREATE INDEX idx_billing_audit_log_timestamp ON billing_audit_log(timestamp DESC);

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_tenants_updated_at BEFORE UPDATE ON tenants
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_subscriptions_updated_at BEFORE UPDATE ON subscriptions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- DEFAULT DATA
-- ============================================================================

-- Insert default feature flags
INSERT INTO feature_flags (key, name, description, enabled_globally, enabled_tiers)
VALUES 
    ('plugins_enabled', 'Plugin System', 'Enable plugin execution and marketplace', FALSE, ARRAY['pro', 'team', 'enterprise']),
    ('custom_models', 'Custom LLM Models', 'Allow custom LLM model configuration', FALSE, ARRAY['team', 'enterprise']),
    ('priority_execution', 'Priority Execution', 'Execute tasks with higher priority', FALSE, ARRAY['pro', 'team', 'enterprise']),
    ('advanced_analytics', 'Advanced Analytics', 'Access to advanced analytics and insights', FALSE, ARRAY['team', 'enterprise']),
    ('api_v4', 'API v4 Access', 'Access to next-generation API features', FALSE, ARRAY[])
ON CONFLICT (key) DO NOTHING;

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE tenants IS 'Core multi-tenancy table - represents isolated customer environments';
COMMENT ON TABLE subscriptions IS 'Subscription management with Stripe integration';
COMMENT ON TABLE invoices IS 'Invoice generation and payment tracking';
COMMENT ON TABLE usage_records IS 'Metered usage tracking for billing';
COMMENT ON TABLE api_keys IS 'API key management for programmatic access';
COMMENT ON TABLE rate_limit_tracking IS 'Real-time rate limiting enforcement';
COMMENT ON TABLE feature_flags IS 'Feature flag system for gradual rollouts';
COMMENT ON TABLE billing_audit_log IS 'Immutable audit trail for billing events';
