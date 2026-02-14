-- Migration 004: Phase 4 - Massive-Scale Planning Engine (Task Decomposition)
-- Date: 2026-02-14
-- Description: Adds support for hierarchical task decomposition with 10,000+ nodes,
--              100+ depth levels, dependency management, and cycle detection.

-- ============================================================================
-- 1. TASK DECOMPOSITION SYSTEM
-- ============================================================================

-- Task decompositions for hierarchical planning
CREATE TABLE IF NOT EXISTS task_decompositions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_execution_id UUID NOT NULL REFERENCES workflow_executions(id) ON DELETE CASCADE,
    
    -- Root task information
    root_task_name VARCHAR(500) NOT NULL,
    root_task_description TEXT,
    root_node_id UUID,
    
    -- Strategy and statistics
    strategy VARCHAR(50) NOT NULL CHECK (strategy IN ('breadth_first', 'depth_first', 'hybrid', 'adaptive')),
    total_nodes INTEGER DEFAULT 0,
    max_depth_reached INTEGER DEFAULT 0,
    leaf_node_count INTEGER DEFAULT 0,
    
    -- All node IDs for quick access (optimized for 10K+ nodes)
    all_node_ids UUID[],
    
    -- Status flags
    is_complete BOOLEAN DEFAULT FALSE,
    has_cycles BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    metadata JSONB
);

CREATE INDEX idx_task_decompositions_workflow ON task_decompositions(workflow_execution_id);
CREATE INDEX idx_task_decompositions_created ON task_decompositions(created_at DESC);
CREATE INDEX idx_task_decompositions_complete ON task_decompositions(is_complete) WHERE is_complete = FALSE;
CREATE INDEX idx_task_decompositions_cycles ON task_decompositions(has_cycles) WHERE has_cycles = TRUE;

COMMENT ON TABLE task_decompositions IS 'Hierarchical task decompositions supporting 10,000+ interconnected tasks';
COMMENT ON COLUMN task_decompositions.strategy IS 'Decomposition strategy: breadth_first, depth_first, hybrid, or adaptive';
COMMENT ON COLUMN task_decompositions.all_node_ids IS 'Array of all node IDs for efficient lookups (optimized for large graphs)';

-- ============================================================================
-- 2. TASK NODES
-- ============================================================================

-- Task nodes for hierarchical decomposition tree
CREATE TABLE IF NOT EXISTS task_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    decomposition_id UUID NOT NULL REFERENCES task_decompositions(id) ON DELETE CASCADE,
    
    -- Task information
    name VARCHAR(500) NOT NULL,
    description TEXT,
    
    -- Hierarchy (supports 100+ depth levels)
    parent_id UUID,
    depth_level INTEGER NOT NULL CHECK (depth_level >= 0 AND depth_level <= 200),
    child_node_ids UUID[] DEFAULT '{}',
    
    -- Task properties
    estimated_complexity FLOAT NOT NULL CHECK (estimated_complexity >= 0 AND estimated_complexity <= 1),
    is_leaf BOOLEAN DEFAULT TRUE,
    actual_subtask_count INTEGER DEFAULT 0,
    
    -- Status lifecycle
    status VARCHAR(50) NOT NULL DEFAULT 'pending' CHECK (
        status IN ('pending', 'decomposing', 'decomposed', 'ready', 'blocked', 'failed')
    ),
    
    -- Dependencies (stored as JSONB array for flexibility)
    -- Format: [{"from_node_id": UUID, "to_node_id": UUID, "dependency_type": "sequential|parallel|conditional|resource", "condition": "...", "metadata": {...}}]
    dependencies JSONB DEFAULT '[]',
    
    -- Tags and metadata
    tags TEXT[],
    metadata JSONB,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE,
    decomposed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_task_nodes_decomposition ON task_nodes(decomposition_id);
CREATE INDEX idx_task_nodes_parent ON task_nodes(parent_id) WHERE parent_id IS NOT NULL;
CREATE INDEX idx_task_nodes_depth ON task_nodes(decomposition_id, depth_level);
CREATE INDEX idx_task_nodes_leaf ON task_nodes(decomposition_id, is_leaf) WHERE is_leaf = TRUE;
CREATE INDEX idx_task_nodes_status ON task_nodes(decomposition_id, status);
CREATE INDEX idx_task_nodes_created ON task_nodes(created_at DESC);
CREATE INDEX idx_task_nodes_complexity ON task_nodes(estimated_complexity DESC);

-- GIN index for dependencies JSONB for efficient cycle detection
CREATE INDEX idx_task_nodes_dependencies ON task_nodes USING gin(dependencies);

COMMENT ON TABLE task_nodes IS 'Task nodes in hierarchical decomposition tree (supports 100+ depth, 10K+ nodes)';
COMMENT ON COLUMN task_nodes.depth_level IS 'Depth in hierarchy (0 = root, max 200 for safety)';
COMMENT ON COLUMN task_nodes.estimated_complexity IS 'Complexity score 0-1 (drives decomposition decisions)';
COMMENT ON COLUMN task_nodes.dependencies IS 'Array of dependency objects with type (sequential, parallel, conditional, resource)';
COMMENT ON COLUMN task_nodes.status IS 'Lifecycle: pending → decomposing → decomposed → ready (or blocked/failed)';

-- ============================================================================
-- 3. UTILITY FUNCTIONS FOR TASK DECOMPOSITION
-- ============================================================================

-- Function to calculate effective complexity considering child tasks
CREATE OR REPLACE FUNCTION calculate_effective_complexity(
    p_node_id UUID
) RETURNS FLOAT AS $$
DECLARE
    v_base_complexity FLOAT;
    v_child_count INTEGER;
    v_effective_complexity FLOAT;
BEGIN
    -- Get base complexity and child count
    SELECT estimated_complexity, actual_subtask_count
    INTO v_base_complexity, v_child_count
    FROM task_nodes
    WHERE id = p_node_id;
    
    -- If leaf node, return base complexity
    IF v_child_count = 0 THEN
        RETURN v_base_complexity;
    END IF;
    
    -- Effective complexity is weighted average of base and aggregated children
    -- More children = lower effective complexity (work is distributed)
    v_effective_complexity := v_base_complexity / (1.0 + (v_child_count * 0.1));
    
    RETURN GREATEST(v_effective_complexity, 0.1);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION calculate_effective_complexity IS 'Calculate effective task complexity considering decomposition into subtasks';

-- Function to detect if a task should be decomposed further
CREATE OR REPLACE FUNCTION should_decompose_task(
    p_node_id UUID,
    p_complexity_threshold FLOAT DEFAULT 0.3
) RETURNS BOOLEAN AS $$
DECLARE
    v_node RECORD;
BEGIN
    SELECT 
        estimated_complexity,
        is_leaf,
        status,
        depth_level
    INTO v_node
    FROM task_nodes
    WHERE id = p_node_id;
    
    -- Cannot decompose if not found
    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;
    
    -- Should decompose if:
    -- 1. Complexity above threshold
    -- 2. Is currently a leaf
    -- 3. Status is pending or ready
    -- 4. Not at max depth
    RETURN (
        v_node.estimated_complexity >= p_complexity_threshold AND
        v_node.is_leaf = TRUE AND
        v_node.status IN ('pending', 'ready') AND
        v_node.depth_level < 150
    );
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION should_decompose_task IS 'Determine if task should be decomposed based on complexity and status';

-- Function to get decomposition tree statistics
CREATE OR REPLACE FUNCTION get_decomposition_statistics(
    p_decomposition_id UUID
) RETURNS TABLE (
    total_nodes BIGINT,
    max_depth INTEGER,
    leaf_count BIGINT,
    avg_complexity FLOAT,
    pending_count BIGINT,
    decomposed_count BIGINT,
    ready_count BIGINT,
    blocked_count BIGINT,
    failed_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT as total_nodes,
        MAX(depth_level)::INTEGER as max_depth,
        COUNT(CASE WHEN is_leaf THEN 1 END)::BIGINT as leaf_count,
        AVG(estimated_complexity)::FLOAT as avg_complexity,
        COUNT(CASE WHEN status = 'pending' THEN 1 END)::BIGINT as pending_count,
        COUNT(CASE WHEN status = 'decomposed' THEN 1 END)::BIGINT as decomposed_count,
        COUNT(CASE WHEN status = 'ready' THEN 1 END)::BIGINT as ready_count,
        COUNT(CASE WHEN status = 'blocked' THEN 1 END)::BIGINT as blocked_count,
        COUNT(CASE WHEN status = 'failed' THEN 1 END)::BIGINT as failed_count
    FROM task_nodes
    WHERE decomposition_id = p_decomposition_id;
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION get_decomposition_statistics IS 'Get comprehensive statistics for a task decomposition';

-- ============================================================================
-- 4. INDEXES FOR PERFORMANCE (10,000+ NODE OPTIMIZATION)
-- ============================================================================

-- Composite index for common query patterns
CREATE INDEX idx_task_nodes_decomp_status_depth ON task_nodes(decomposition_id, status, depth_level);
CREATE INDEX idx_task_nodes_decomp_leaf_depth ON task_nodes(decomposition_id, is_leaf, depth_level) WHERE is_leaf = TRUE;

-- BRIN index for created_at (efficient for time-series data)
CREATE INDEX idx_task_nodes_created_brin ON task_nodes USING brin(created_at);

-- ============================================================================
-- 5. CONSTRAINTS AND TRIGGERS
-- ============================================================================

-- Ensure parent exists in same decomposition (if parent_id is set)
CREATE OR REPLACE FUNCTION validate_task_node_parent() RETURNS TRIGGER AS $$
BEGIN
    IF NEW.parent_id IS NOT NULL THEN
        -- Check parent exists in same decomposition
        IF NOT EXISTS (
            SELECT 1 FROM task_nodes
            WHERE id = NEW.parent_id
            AND decomposition_id = NEW.decomposition_id
        ) THEN
            RAISE EXCEPTION 'Parent task node % does not exist in decomposition %',
                NEW.parent_id, NEW.decomposition_id;
        END IF;
        
        -- Parent depth should be child depth - 1
        DECLARE
            v_parent_depth INTEGER;
        BEGIN
            SELECT depth_level INTO v_parent_depth
            FROM task_nodes
            WHERE id = NEW.parent_id;
            
            IF v_parent_depth != NEW.depth_level - 1 THEN
                RAISE EXCEPTION 'Invalid depth level: parent depth % should be %',
                    v_parent_depth, NEW.depth_level - 1;
            END IF;
        END;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER task_node_validate_parent
    BEFORE INSERT OR UPDATE ON task_nodes
    FOR EACH ROW
    EXECUTE FUNCTION validate_task_node_parent();

COMMENT ON TRIGGER task_node_validate_parent ON task_nodes IS 'Validates parent-child relationship consistency';

-- ============================================================================
-- 6. INITIAL DATA AND CONFIGURATION
-- ============================================================================

-- No initial data needed - tables are ready for Phase 4 decomposition operations

-- ============================================================================
-- 7. MIGRATION COMPLETE
-- ============================================================================

-- Record migration
INSERT INTO schema_migrations (version, description, applied_at)
VALUES (
    '004',
    'Phase 4 - Massive-Scale Planning Engine (Task Decomposition)',
    NOW()
)
ON CONFLICT (version) DO NOTHING;
