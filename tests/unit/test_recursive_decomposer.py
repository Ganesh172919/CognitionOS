"""
Unit Tests for RecursiveDecomposer Service

Tests for RecursiveDecomposer domain service business logic.
"""

import pytest
from uuid import uuid4

from core.domain.task_decomposition.entities import (
    TaskNode,
    TaskDecomposition,
    TaskNodeStatus,
    DecompositionStrategy,
)
from core.domain.task_decomposition.services import RecursiveDecomposer


class TestRecursiveDecomposer:
    """Tests for RecursiveDecomposer service"""
    
    @pytest.fixture
    def decomposer(self):
        """Create decomposer instance with default max depth"""
        return RecursiveDecomposer(max_depth=150)
    
    @pytest.fixture
    def small_decomposer(self):
        """Create decomposer with smaller max depth for testing"""
        return RecursiveDecomposer(max_depth=10)
    
    @pytest.fixture
    def sample_decomposition(self):
        """Create sample decomposition"""
        return TaskDecomposition.create(
            workflow_execution_id=uuid4(),
            root_task_name="Test Task",
            root_task_description="Test description",
            strategy=DecompositionStrategy.HYBRID
        )
    
    @pytest.fixture
    def sample_task_node(self, sample_decomposition):
        """Create sample task node"""
        return TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Sample Task",
            description="Sample task description",
            depth_level=0,
            estimated_complexity=0.8
        )
    
    def test_can_decompose_with_decomposable_task(self, decomposer, sample_task_node):
        """Test can_decompose returns True for decomposable task"""
        # Task is pending, is_leaf=True, and complexity > 0.3
        assert decomposer.can_decompose(sample_task_node) is True
    
    def test_can_decompose_at_max_depth(self, small_decomposer, sample_decomposition):
        """Test can_decompose returns False at max depth"""
        task_at_max = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Deep Task",
            description="Task at max depth",
            depth_level=10,  # At max_depth
            estimated_complexity=0.8
        )
        
        assert small_decomposer.can_decompose(task_at_max) is False
    
    def test_can_decompose_beyond_max_depth(self, small_decomposer, sample_decomposition):
        """Test can_decompose returns False beyond max depth"""
        task_beyond_max = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Too Deep Task",
            description="Task beyond max depth",
            depth_level=11,  # Beyond max_depth
            estimated_complexity=0.8
        )
        
        assert small_decomposer.can_decompose(task_beyond_max) is False
    
    def test_can_decompose_non_decomposable_task(self, decomposer, sample_task_node):
        """Test can_decompose returns False for non-decomposable task"""
        # Mark as non-leaf (already decomposed)
        sample_task_node.add_child(uuid4())
        
        assert decomposer.can_decompose(sample_task_node) is False
    
    def test_can_decompose_low_complexity_task(self, decomposer, sample_decomposition):
        """Test can_decompose returns False for low complexity task"""
        low_complexity_task = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Simple Task",
            description="Low complexity task",
            depth_level=0,
            estimated_complexity=0.2  # Below 0.3 threshold
        )
        
        assert decomposer.can_decompose(low_complexity_task) is False
    
    def test_should_decompose_above_threshold(self, decomposer, sample_task_node):
        """Test should_decompose returns True for task above complexity threshold"""
        # complexity = 0.8, threshold = 0.3
        assert decomposer.should_decompose(sample_task_node, complexity_threshold=0.3) is True
    
    def test_should_decompose_below_threshold(self, decomposer, sample_decomposition):
        """Test should_decompose returns False for task below complexity threshold"""
        task = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Simple Task",
            description="Below threshold",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        assert decomposer.should_decompose(task, complexity_threshold=0.7) is False
    
    def test_should_decompose_at_threshold(self, decomposer, sample_decomposition):
        """Test should_decompose returns True at exact threshold"""
        task = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Task at Threshold",
            description="Exactly at threshold",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        assert decomposer.should_decompose(task, complexity_threshold=0.5) is True
    
    def test_estimate_subtask_count_breadth_first(self, decomposer, sample_task_node):
        """Test estimate_subtask_count for breadth-first strategy"""
        count = decomposer.estimate_subtask_count(
            sample_task_node,
            DecompositionStrategy.BREADTH_FIRST
        )
        
        # Formula: 5 + (0.8 * 10) = 13
        assert count == 13
    
    def test_estimate_subtask_count_depth_first(self, decomposer, sample_task_node):
        """Test estimate_subtask_count for depth-first strategy"""
        count = decomposer.estimate_subtask_count(
            sample_task_node,
            DecompositionStrategy.DEPTH_FIRST
        )
        
        # Formula: 2 + (0.8 * 3) = 4
        assert count == 4
    
    def test_estimate_subtask_count_hybrid(self, decomposer, sample_task_node):
        """Test estimate_subtask_count for hybrid strategy"""
        count = decomposer.estimate_subtask_count(
            sample_task_node,
            DecompositionStrategy.HYBRID
        )
        
        # Formula: 3 + (0.8 * 5) = 7
        assert count == 7
    
    def test_estimate_subtask_count_adaptive(self, decomposer, sample_decomposition):
        """Test estimate_subtask_count for adaptive strategy adapts by depth"""
        task_shallow = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Shallow Task",
            description="At depth 0",
            depth_level=0,
            estimated_complexity=0.8
        )
        
        count_shallow = decomposer.estimate_subtask_count(
            task_shallow,
            DecompositionStrategy.ADAPTIVE
        )
        
        task_deep = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Deep Task",
            description="At depth 75",
            depth_level=75,
            estimated_complexity=0.8
        )
        
        count_deep = decomposer.estimate_subtask_count(
            task_deep,
            DecompositionStrategy.ADAPTIVE
        )
        
        # Shallow should have more subtasks than deep due to depth factor
        assert count_shallow > count_deep
    
    def test_decompose_task_creates_subtasks(self, decomposer, sample_task_node, sample_decomposition):
        """Test decompose_task creates correct subtasks"""
        subtask_specs = [
            {"name": "Subtask 1", "description": "First subtask", "complexity": 0.4, "tags": ["test"]},
            {"name": "Subtask 2", "description": "Second subtask", "complexity": 0.5, "tags": []},
            {"name": "Subtask 3", "description": "Third subtask", "complexity": 0.3, "tags": ["critical"]},
        ]
        
        subtasks = decomposer.decompose_task(
            sample_task_node,
            sample_decomposition,
            subtask_specs
        )
        
        assert len(subtasks) == 3
        assert all(isinstance(task, TaskNode) for task in subtasks)
        assert subtasks[0].name == "Subtask 1"
        assert subtasks[0].estimated_complexity == 0.4
        assert subtasks[0].tags == ["test"]
    
    def test_decompose_task_sets_parent_references(self, decomposer, sample_task_node, sample_decomposition):
        """Test decompose_task sets correct parent references"""
        subtask_specs = [
            {"name": "Subtask 1", "description": "First subtask", "complexity": 0.4},
            {"name": "Subtask 2", "description": "Second subtask", "complexity": 0.5},
        ]
        
        subtasks = decomposer.decompose_task(
            sample_task_node,
            sample_decomposition,
            subtask_specs
        )
        
        # All subtasks should reference parent
        for subtask in subtasks:
            assert subtask.parent_id == sample_task_node.id.value
            assert subtask.depth_level == sample_task_node.depth_level + 1
    
    def test_decompose_task_updates_parent_status(self, decomposer, sample_task_node, sample_decomposition):
        """Test decompose_task updates parent task status"""
        subtask_specs = [
            {"name": "Subtask 1", "description": "First subtask", "complexity": 0.4},
        ]
        
        assert sample_task_node.status == TaskNodeStatus.PENDING
        
        decomposer.decompose_task(
            sample_task_node,
            sample_decomposition,
            subtask_specs
        )
        
        assert sample_task_node.status == TaskNodeStatus.DECOMPOSED
        assert sample_task_node.is_leaf is False
    
    def test_decompose_task_registers_with_decomposition(self, decomposer, sample_task_node, sample_decomposition):
        """Test decompose_task registers subtasks with decomposition"""
        initial_nodes = sample_decomposition.total_nodes
        
        subtask_specs = [
            {"name": "Subtask 1", "description": "First subtask", "complexity": 0.4},
            {"name": "Subtask 2", "description": "Second subtask", "complexity": 0.5},
        ]
        
        subtasks = decomposer.decompose_task(
            sample_task_node,
            sample_decomposition,
            subtask_specs
        )
        
        # Decomposition should track all new subtasks
        assert sample_decomposition.total_nodes == initial_nodes + 2
        for subtask in subtasks:
            assert subtask.id.value in sample_decomposition.all_node_ids
    
    def test_decompose_task_registers_leaf_nodes(self, decomposer, sample_task_node, sample_decomposition):
        """Test decompose_task registers low-complexity subtasks as leaves"""
        initial_leaves = sample_decomposition.leaf_node_count
        
        subtask_specs = [
            {"name": "Leaf Subtask", "description": "Low complexity", "complexity": 0.2},
            {"name": "Non-leaf Subtask", "description": "High complexity", "complexity": 0.8},
        ]
        
        decomposer.decompose_task(
            sample_task_node,
            sample_decomposition,
            subtask_specs
        )
        
        # Only the low-complexity subtask should be registered as leaf
        assert sample_decomposition.leaf_node_count == initial_leaves + 1
    
    def test_decompose_task_raises_for_non_decomposable(self, decomposer, sample_decomposition):
        """Test decompose_task raises error for non-decomposable task"""
        non_decomposable = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Non-decomposable",
            description="Already decomposed",
            depth_level=0,
            estimated_complexity=0.8
        )
        non_decomposable.add_child(uuid4())  # Make it non-decomposable
        
        with pytest.raises(ValueError, match="Cannot decompose task"):
            decomposer.decompose_task(
                non_decomposable,
                sample_decomposition,
                [{"name": "Should Fail", "description": "Test", "complexity": 0.5}]
            )
    
    def test_decompose_task_raises_at_max_depth(self, small_decomposer, sample_decomposition):
        """Test decompose_task raises error at max depth"""
        task_at_max = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="At Max Depth",
            description="Cannot decompose further",
            depth_level=10,
            estimated_complexity=0.8
        )
        
        with pytest.raises(ValueError, match="Cannot decompose task at depth 10"):
            small_decomposer.decompose_task(
                task_at_max,
                sample_decomposition,
                [{"name": "Should Fail", "description": "Test", "complexity": 0.5}]
            )
    
    def test_decompose_task_with_empty_specifications(self, decomposer, sample_task_node, sample_decomposition):
        """Test decompose_task with empty subtask specifications"""
        subtasks = decomposer.decompose_task(
            sample_task_node,
            sample_decomposition,
            []
        )
        
        assert len(subtasks) == 0
        assert sample_task_node.status == TaskNodeStatus.DECOMPOSED
        # With no children added, is_leaf remains True
        assert sample_task_node.is_leaf is True
    
    def test_decompose_task_with_minimal_specs(self, decomposer, sample_task_node, sample_decomposition):
        """Test decompose_task uses defaults for missing specification fields"""
        minimal_specs = [
            {},  # No fields provided
        ]
        
        subtasks = decomposer.decompose_task(
            sample_task_node,
            sample_decomposition,
            minimal_specs
        )
        
        assert len(subtasks) == 1
        assert subtasks[0].name == f"Subtask of {sample_task_node.name}"
        assert subtasks[0].description == ""
        assert subtasks[0].estimated_complexity == pytest.approx(0.5)
        assert subtasks[0].tags == []
