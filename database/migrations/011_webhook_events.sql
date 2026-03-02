-- Migration 011: Webhook Events (Stripe Idempotency)
-- Date: 2026-02-28
-- Description: Persist webhook events for idempotent processing and retry management

-- ============================================================================
-- WEBHOOK EVENTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS webhook_events (
    id BIGSERIAL PRIMARY KEY,

    -- Stripe event identification
    event_id VARCHAR(255) UNIQUE NOT NULL,
    event_type VARCHAR(100) NOT NULL,

    -- Processing status
    status VARCHAR(20) NOT NULL CHECK (status IN ('pending', 'processed', 'failed', 'retrying')),

    -- Processing result / error details
    result JSONB,
    error_message TEXT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,
    failed_at TIMESTAMP WITH TIME ZONE,
    last_retry_at TIMESTAMP WITH TIME ZONE,

    -- Retry management
    retry_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_webhook_events_status_retry ON webhook_events(status, retry_count);
CREATE INDEX IF NOT EXISTS idx_webhook_events_type_status ON webhook_events(event_type, status);
CREATE INDEX IF NOT EXISTS idx_webhook_events_created_at ON webhook_events(created_at);

