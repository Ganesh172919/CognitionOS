-- ============================================================================
-- Migration: 004 - Memory Namespaces and Lifecycle
-- Version: V2
-- Date: 2026-02-11
-- Description: Add memory namespaces and lifecycle management for V2 memory system
-- ============================================================================

-- Memory namespaces table: Logical grouping of memories
CREATE TABLE memory_namespaces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL UNIQUE,
    description TEXT,
    owner_user_id UUID REFERENCES users(id),

    -- Access control
    visibility VARCHAR(50) NOT NULL DEFAULT 'private' CHECK (visibility IN ('private', 'shared', 'public')),
    allowed_users UUID[],

    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Memory lifecycle policies table: Define retention and archival policies
CREATE TABLE memory_lifecycle_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL UNIQUE,
    namespace_id UUID REFERENCES memory_namespaces(id) ON DELETE CASCADE,
    description TEXT,

    -- TTL policies
    ttl_days INTEGER,
    min_access_frequency INTEGER,

    -- Compression policies
    compression_after_days INTEGER,
    compression_ratio FLOAT DEFAULT 0.5 CHECK (compression_ratio > 0 AND compression_ratio <= 1),

    -- Archival policies
    archive_after_days INTEGER,
    archive_destination VARCHAR(500),

    -- Deletion policies
    delete_archived_after_days INTEGER,

    -- Status
    is_active BOOLEAN DEFAULT TRUE,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_ttl_order CHECK (
        (ttl_days IS NULL OR compression_after_days IS NULL OR ttl_days > compression_after_days) AND
        (compression_after_days IS NULL OR archive_after_days IS NULL OR archive_after_days > compression_after_days)
    )
);

-- Add namespace and lifecycle fields to memories table
ALTER TABLE memories ADD COLUMN namespace_id UUID REFERENCES memory_namespaces(id);
ALTER TABLE memories ADD COLUMN compressed BOOLEAN DEFAULT FALSE;
ALTER TABLE memories ADD COLUMN archived BOOLEAN DEFAULT FALSE;
ALTER TABLE memories ADD COLUMN archived_at TIMESTAMP;
ALTER TABLE memories ADD COLUMN archive_location VARCHAR(500);
ALTER TABLE memories ADD COLUMN original_size_bytes INTEGER;
ALTER TABLE memories ADD COLUMN compressed_size_bytes INTEGER;

-- Memory garbage collection runs table: Track GC operations
CREATE TABLE memory_gc_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace_id UUID REFERENCES memory_namespaces(id),

    -- Statistics
    memories_scanned INTEGER NOT NULL DEFAULT 0,
    memories_deleted INTEGER NOT NULL DEFAULT 0,
    memories_compressed INTEGER NOT NULL DEFAULT 0,
    memories_archived INTEGER NOT NULL DEFAULT 0,

    -- Size metrics
    bytes_freed BIGINT,
    bytes_compressed BIGINT,
    bytes_archived BIGINT,

    -- Timing
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    duration_seconds INTEGER,

    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
    error TEXT
);

-- Indexes for performance
CREATE INDEX idx_memory_namespaces_owner ON memory_namespaces(owner_user_id);
CREATE INDEX idx_memory_namespaces_visibility ON memory_namespaces(visibility);

CREATE INDEX idx_memory_lifecycle_policies_namespace ON memory_lifecycle_policies(namespace_id);
CREATE INDEX idx_memory_lifecycle_policies_active ON memory_lifecycle_policies(is_active);

CREATE INDEX idx_memories_namespace_id ON memories(namespace_id);
CREATE INDEX idx_memories_compressed ON memories(compressed);
CREATE INDEX idx_memories_archived ON memories(archived);
CREATE INDEX idx_memories_lifecycle ON memories(namespace_id, compressed, archived, accessed_at);

CREATE INDEX idx_memory_gc_runs_namespace ON memory_gc_runs(namespace_id);
CREATE INDEX idx_memory_gc_runs_started_at ON memory_gc_runs(started_at DESC);
CREATE INDEX idx_memory_gc_runs_status ON memory_gc_runs(status);

-- Triggers for updated_at
CREATE TRIGGER update_memory_namespaces_updated_at
    BEFORE UPDATE ON memory_namespaces
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_memory_lifecycle_policies_updated_at
    BEFORE UPDATE ON memory_lifecycle_policies
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create default namespace for existing memories
INSERT INTO memory_namespaces (name, description, visibility)
VALUES ('default', 'Default namespace for existing memories', 'private');

-- Assign existing memories to default namespace
UPDATE memories
SET namespace_id = (SELECT id FROM memory_namespaces WHERE name = 'default')
WHERE namespace_id IS NULL;

-- Comments
COMMENT ON TABLE memory_namespaces IS 'Logical grouping and isolation of memories';
COMMENT ON TABLE memory_lifecycle_policies IS 'Retention, compression, and archival policies for memories';
COMMENT ON TABLE memory_gc_runs IS 'History of memory garbage collection operations';
COMMENT ON COLUMN memories.compressed IS 'Whether memory content has been compressed';
COMMENT ON COLUMN memories.archived IS 'Whether memory has been archived to cold storage';
COMMENT ON COLUMN memories.archive_location IS 'S3/cold storage location if archived';
COMMENT ON COLUMN memory_lifecycle_policies.compression_ratio IS 'Target compression ratio (0-1)';
COMMENT ON COLUMN memory_lifecycle_policies.archive_destination IS 'S3 bucket or cold storage destination';
