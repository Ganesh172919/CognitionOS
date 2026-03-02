-- Migration 012: Usage Records Idempotency
-- Date: 2026-02-28
-- Description: Add event_id to usage_records for idempotent metering inserts

ALTER TABLE usage_records
    ADD COLUMN IF NOT EXISTS event_id VARCHAR(255);

-- Enforce idempotency per-tenant for deterministic events (e.g., retries).
CREATE UNIQUE INDEX IF NOT EXISTS uq_usage_records_tenant_event_id
    ON usage_records (tenant_id, event_id);

CREATE INDEX IF NOT EXISTS idx_usage_records_event_id
    ON usage_records (event_id)
    WHERE event_id IS NOT NULL;
