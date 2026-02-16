"""
Unit Tests for IntegrityEnforcer Service

Tests for IntegrityEnforcer domain service business logic.
"""

import pytest
from uuid import uuid4

from core.domain.task_decomposition.entities import (
    TaskNode,
    TaskDecomposition,
    Dependency,
    DependencyType,
    DecompositionStrategy,
)
from core.domain.task_decomposition.services import IntegrityEnforcer


class TestIntegrityEnforcer:
    """Tests for IntegrityEnforcer service"""
    
    @pytest.fixture
    def enforcer(self):
        """Create integrity enforcer instance"""
        return IntegrityEnforcer()
    
    @pytest.fixture
    def sample_decomposition(self):
        """Create sample decomposition"""
        return TaskDecomposition.create(
            workflow_execution_id=uuid4(),
            root_task_name="Test Task",
            root_task_description="Test description",
            strategy=DecompositionStrategy.HYBRID
        )
    
    def test_validate_decomposition_valid(self, enforcer, sample_decomposition):
        """Test validate_decomposition with valid decomposition"""
        root = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Root",
            description="Root task",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        sample_decomposition.set_root_node(root.id.value)
        
        nodes = {root.id.value: root}
        
        is_valid, errors = enforcer.validate_decomposition(sample_decomposition, nodes)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_decomposition_missing_root(self, enforcer, sample_decomposition):
        """Test validate_decomposition detects missing root node"""
        is_valid, errors = enforcer.validate_decomposition(sample_decomposition, {})
        
        assert is_valid is False
        assert any("no root node" in err.lower() for err in errors)
    
    def test_validate_decomposition_root_not_in_nodes(self, enforcer, sample_decomposition):
        """Test validate_decomposition detects root node not in nodes dict"""
        sample_decomposition.root_node_id = uuid4()
        
        is_valid, errors = enforcer.validate_decomposition(sample_decomposition, {})
        
        assert is_valid is False
        assert any("Root node" in err and "not found" in err for err in errors)
    
    def test_validate_decomposition_node_count_mismatch(self, enforcer, sample_decomposition):
        """Test validate_decomposition detects node count mismatch"""
        root = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Root",
            description="Root task",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        sample_decomposition.set_root_node(root.id.value)
        sample_decomposition.total_nodes = 5  # Mismatch (should be 1)
        
        nodes = {root.id.value: root}
        
        is_valid, errors = enforcer.validate_decomposition(sample_decomposition, nodes)
        
        assert is_valid is False
        assert any("Node count mismatch" in err for err in errors)
    
    def test_validate_decomposition_with_cycles(self, enforcer, sample_decomposition):
        """Test validate_decomposition detects cycles"""
        node1 = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Node 1",
            description="First node",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        node2 = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Node 2",
            description="Second node",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        # Create cycle
        node2.add_dependency(
            Dependency(
                from_node_id=node1.id.value,
                to_node_id=node2.id.value,
                dependency_type=DependencyType.SEQUENTIAL
            )
        )
        
        node1.add_dependency(
            Dependency(
                from_node_id=node2.id.value,
                to_node_id=node1.id.value,
                dependency_type=DependencyType.SEQUENTIAL
            )
        )
        
        sample_decomposition.set_root_node(node1.id.value)
        sample_decomposition.register_node(node2.id.value, 0)
        
        nodes = {
            node1.id.value: node1,
            node2.id.value: node2,
        }
        
        is_valid, errors = enforcer.validate_decomposition(sample_decomposition, nodes)
        
        assert is_valid is False
        assert any("Cycle detected" in err for err in errors)
        assert sample_decomposition.has_cycles is True
    
    def test_validate_decomposition_with_dependency_errors(self, enforcer, sample_decomposition):
        """Test validate_decomposition detects dependency validation errors"""
        node1 = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Node 1",
            description="First node",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        # Add self-referencing dependency (invalid)
        node1.add_dependency(
            Dependency(
                from_node_id=node1.id.value,
                to_node_id=node1.id.value,
                dependency_type=DependencyType.SEQUENTIAL
            )
        )
        
        sample_decomposition.set_root_node(node1.id.value)
        
        nodes = {node1.id.value: node1}
        
        is_valid, errors = enforcer.validate_decomposition(sample_decomposition, nodes)
        
        assert is_valid is False
        assert any("Node 1" in err for err in errors)
    
    def test_validate_decomposition_missing_parent(self, enforcer, sample_decomposition):
        """Test validate_decomposition detects missing parent node"""
        child = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Child",
            description="Child with missing parent",
            parent_id=uuid4(),  # Parent doesn't exist
            depth_level=1,
            estimated_complexity=0.5
        )
        
        sample_decomposition.set_root_node(child.id.value)
        
        nodes = {child.id.value: child}
        
        is_valid, errors = enforcer.validate_decomposition(sample_decomposition, nodes)
        
        assert is_valid is False
        assert any("missing parent" in err.lower() for err in errors)
    
    def test_validate_decomposition_incorrect_depth(self, enforcer, sample_decomposition):
        """Test validate_decomposition detects incorrect depth levels"""
        parent = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Parent",
            description="Parent task",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        child = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Child",
            description="Child task",
            parent_id=parent.id.value,
            depth_level=5,  # Should be 1, not 5
            estimated_complexity=0.5
        )
        
        sample_decomposition.set_root_node(parent.id.value)
        sample_decomposition.register_node(child.id.value, 1)
        
        nodes = {
            parent.id.value: parent,
            child.id.value: child,
        }
        
        is_valid, errors = enforcer.validate_decomposition(sample_decomposition, nodes)
        
        assert is_valid is False
        assert any("incorrect depth" in err.lower() for err in errors)
    
    def test_enforce_parent_child_consistency_valid(self, enforcer, sample_decomposition):
        """Test enforce_parent_child_consistency with valid relationships"""
        parent = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Parent",
            description="Parent task",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        child1 = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Child 1",
            description="First child",
            parent_id=parent.id.value,
            depth_level=1,
            estimated_complexity=0.5
        )
        
        child2 = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Child 2",
            description="Second child",
            parent_id=parent.id.value,
            depth_level=1,
            estimated_complexity=0.5
        )
        
        parent.add_child(child1.id.value)
        parent.add_child(child2.id.value)
        
        errors = enforcer.enforce_parent_child_consistency(parent, [child1, child2])
        
        assert len(errors) == 0
    
    def test_enforce_parent_child_consistency_wrong_parent_reference(self, enforcer, sample_decomposition):
        """Test enforce_parent_child_consistency detects wrong parent reference"""
        parent = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Parent",
            description="Parent task",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        child = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Child",
            description="Child with wrong parent ref",
            parent_id=uuid4(),  # Wrong parent
            depth_level=1,
            estimated_complexity=0.5
        )
        
        parent.add_child(child.id.value)
        
        errors = enforcer.enforce_parent_child_consistency(parent, [child])
        
        assert len(errors) > 0
        assert any("does not reference parent" in err for err in errors)
    
    def test_enforce_parent_child_consistency_incorrect_depth(self, enforcer, sample_decomposition):
        """Test enforce_parent_child_consistency detects incorrect depth"""
        parent = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Parent",
            description="Parent task",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        child = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Child",
            description="Child with wrong depth",
            parent_id=parent.id.value,
            depth_level=3,  # Should be 1
            estimated_complexity=0.5
        )
        
        parent.add_child(child.id.value)
        
        errors = enforcer.enforce_parent_child_consistency(parent, [child])
        
        assert len(errors) > 0
        assert any("incorrect depth" in err.lower() for err in errors)
    
    def test_enforce_parent_child_consistency_missing_child_reference(self, enforcer, sample_decomposition):
        """Test enforce_parent_child_consistency detects missing child references"""
        parent = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Parent",
            description="Parent task",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        child = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Child",
            description="Child not in parent's list",
            parent_id=parent.id.value,
            depth_level=1,
            estimated_complexity=0.5
        )
        
        # Don't add child to parent's child_node_ids
        
        errors = enforcer.enforce_parent_child_consistency(parent, [child])
        
        assert len(errors) > 0
        assert any("missing child references" in err.lower() for err in errors)
    
    def test_enforce_parent_child_consistency_extra_child_reference(self, enforcer, sample_decomposition):
        """Test enforce_parent_child_consistency detects extra child references"""
        parent = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Parent",
            description="Parent task",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        child = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Child",
            description="Actual child",
            parent_id=parent.id.value,
            depth_level=1,
            estimated_complexity=0.5
        )
        
        parent.add_child(child.id.value)
        parent.add_child(uuid4())  # Extra child reference
        
        errors = enforcer.enforce_parent_child_consistency(parent, [child])
        
        assert len(errors) > 0
        assert any("extra child references" in err.lower() for err in errors)
    
    def test_enforce_parent_child_consistency_empty_children(self, enforcer, sample_decomposition):
        """Test enforce_parent_child_consistency with empty children list"""
        parent = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Parent",
            description="Parent with no children",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        errors = enforcer.enforce_parent_child_consistency(parent, [])
        
        # No children provided, parent should have no child references
        assert len(errors) == 0
    
    def test_validate_decomposition_complex_hierarchy(self, enforcer, sample_decomposition):
        """Test validate_decomposition with complex multi-level hierarchy"""
        root = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Root",
            description="Root task",
            depth_level=0,
            estimated_complexity=0.8
        )
        
        child1 = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Child 1",
            description="First child",
            parent_id=root.id.value,
            depth_level=1,
            estimated_complexity=0.5
        )
        
        child2 = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Child 2",
            description="Second child",
            parent_id=root.id.value,
            depth_level=1,
            estimated_complexity=0.5
        )
        
        grandchild = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Grandchild",
            description="Grandchild task",
            parent_id=child1.id.value,
            depth_level=2,
            estimated_complexity=0.3
        )
        
        root.add_child(child1.id.value)
        root.add_child(child2.id.value)
        child1.add_child(grandchild.id.value)
        
        sample_decomposition.set_root_node(root.id.value)
        sample_decomposition.register_node(child1.id.value, 1)
        sample_decomposition.register_node(child2.id.value, 1)
        sample_decomposition.register_node(grandchild.id.value, 2)
        
        nodes = {
            root.id.value: root,
            child1.id.value: child1,
            child2.id.value: child2,
            grandchild.id.value: grandchild,
        }
        
        is_valid, errors = enforcer.validate_decomposition(sample_decomposition, nodes)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_decomposition_multiple_errors(self, enforcer, sample_decomposition):
        """Test validate_decomposition accumulates multiple errors"""
        # Create nodes with multiple issues
        root = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Root",
            description="Root task",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        # Child with missing parent
        orphan = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Orphan",
            description="Orphan child",
            parent_id=uuid4(),
            depth_level=1,
            estimated_complexity=0.5
        )
        
        # Child with wrong depth
        wrong_depth = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Wrong Depth",
            description="Wrong depth child",
            parent_id=root.id.value,
            depth_level=5,
            estimated_complexity=0.5
        )
        
        sample_decomposition.set_root_node(root.id.value)
        sample_decomposition.register_node(orphan.id.value, 1)
        sample_decomposition.register_node(wrong_depth.id.value, 1)
        sample_decomposition.total_nodes = 10  # Also wrong count
        
        nodes = {
            root.id.value: root,
            orphan.id.value: orphan,
            wrong_depth.id.value: wrong_depth,
        }
        
        is_valid, errors = enforcer.validate_decomposition(sample_decomposition, nodes)
        
        assert is_valid is False
        assert len(errors) >= 3  # At least: count mismatch, missing parent, wrong depth
