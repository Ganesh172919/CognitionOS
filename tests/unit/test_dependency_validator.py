"""
Unit Tests for DependencyValidator Service

Tests for DependencyValidator domain service business logic.
"""

import pytest
from uuid import uuid4, UUID

from core.domain.task_decomposition.entities import (
    TaskNode,
    TaskDecomposition,
    Dependency,
    DependencyType,
    DecompositionStrategy,
)
from core.domain.task_decomposition.services import DependencyValidator


class TestDependencyValidator:
    """Tests for DependencyValidator service"""
    
    @pytest.fixture
    def validator(self):
        """Create dependency validator instance"""
        return DependencyValidator()
    
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
    def sample_nodes(self, sample_decomposition):
        """Create sample nodes for testing"""
        node1 = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Task 1",
            description="First task",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        node2 = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Task 2",
            description="Second task",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        node3 = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Task 3",
            description="Third task",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        return {
            node1.id.value: node1,
            node2.id.value: node2,
            node3.id.value: node3,
        }
    
    def test_validate_dependency_valid_sequential(self, validator, sample_nodes):
        """Test validate_dependency with valid sequential dependency"""
        node_ids = list(sample_nodes.keys())
        dependency = Dependency(
            from_node_id=node_ids[0],
            to_node_id=node_ids[1],
            dependency_type=DependencyType.SEQUENTIAL
        )
        
        errors = validator.validate_dependency(dependency, sample_nodes)
        
        assert len(errors) == 0
    
    def test_validate_dependency_valid_parallel(self, validator, sample_nodes):
        """Test validate_dependency with valid parallel dependency"""
        node_ids = list(sample_nodes.keys())
        dependency = Dependency(
            from_node_id=node_ids[0],
            to_node_id=node_ids[1],
            dependency_type=DependencyType.PARALLEL
        )
        
        errors = validator.validate_dependency(dependency, sample_nodes)
        
        assert len(errors) == 0
    
    def test_validate_dependency_missing_from_node(self, validator, sample_nodes):
        """Test validate_dependency with missing from_node"""
        node_ids = list(sample_nodes.keys())
        missing_id = uuid4()
        
        dependency = Dependency(
            from_node_id=missing_id,
            to_node_id=node_ids[0],
            dependency_type=DependencyType.SEQUENTIAL
        )
        
        errors = validator.validate_dependency(dependency, sample_nodes)
        
        assert len(errors) == 1
        assert "From node" in errors[0]
        assert "not found" in errors[0]
    
    def test_validate_dependency_missing_to_node(self, validator, sample_nodes):
        """Test validate_dependency with missing to_node"""
        node_ids = list(sample_nodes.keys())
        missing_id = uuid4()
        
        dependency = Dependency(
            from_node_id=node_ids[0],
            to_node_id=missing_id,
            dependency_type=DependencyType.SEQUENTIAL
        )
        
        errors = validator.validate_dependency(dependency, sample_nodes)
        
        assert len(errors) == 1
        assert "To node" in errors[0]
        assert "not found" in errors[0]
    
    def test_validate_dependency_both_nodes_missing(self, validator, sample_nodes):
        """Test validate_dependency with both nodes missing"""
        missing_from = uuid4()
        missing_to = uuid4()
        
        dependency = Dependency(
            from_node_id=missing_from,
            to_node_id=missing_to,
            dependency_type=DependencyType.SEQUENTIAL
        )
        
        errors = validator.validate_dependency(dependency, sample_nodes)
        
        assert len(errors) == 2
        assert any("From node" in err for err in errors)
        assert any("To node" in err for err in errors)
    
    def test_validate_dependency_self_reference(self, validator, sample_nodes):
        """Test validate_dependency detects self-reference"""
        node_id = list(sample_nodes.keys())[0]
        
        dependency = Dependency(
            from_node_id=node_id,
            to_node_id=node_id,
            dependency_type=DependencyType.SEQUENTIAL
        )
        
        errors = validator.validate_dependency(dependency, sample_nodes)
        
        assert len(errors) == 1
        assert "cannot depend on itself" in errors[0]
    
    def test_validate_dependency_parent_child_cycle(self, validator, sample_decomposition):
        """Test validate_dependency detects parent-child cycle"""
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
            depth_level=1,
            estimated_complexity=0.5
        )
        
        nodes = {
            parent.id.value: parent,
            child.id.value: child,
        }
        
        # Try to create dependency from parent to child where child is parent's descendant (not allowed)
        dependency = Dependency(
            from_node_id=parent.id.value,
            to_node_id=child.id.value,
            dependency_type=DependencyType.SEQUENTIAL
        )
        
        errors = validator.validate_dependency(dependency, nodes)
        
        assert len(errors) == 1
        assert "cycle" in errors[0].lower()
    
    def test_validate_dependency_conditional_without_condition(self, validator, sample_nodes):
        """Test validate_dependency detects conditional without condition"""
        node_ids = list(sample_nodes.keys())
        
        dependency = Dependency(
            from_node_id=node_ids[0],
            to_node_id=node_ids[1],
            dependency_type=DependencyType.CONDITIONAL,
            condition=None
        )
        
        errors = validator.validate_dependency(dependency, sample_nodes)
        
        assert len(errors) == 1
        assert "Conditional dependency requires a condition" in errors[0]
    
    def test_validate_dependency_conditional_with_condition(self, validator, sample_nodes):
        """Test validate_dependency accepts conditional with condition"""
        node_ids = list(sample_nodes.keys())
        
        dependency = Dependency(
            from_node_id=node_ids[0],
            to_node_id=node_ids[1],
            dependency_type=DependencyType.CONDITIONAL,
            condition="if success"
        )
        
        errors = validator.validate_dependency(dependency, sample_nodes)
        
        assert len(errors) == 0
    
    def test_validate_all_dependencies_no_errors(self, validator, sample_nodes):
        """Test validate_all_dependencies with valid dependencies"""
        node_ids = list(sample_nodes.keys())
        
        # Add valid dependencies
        sample_nodes[node_ids[1]].add_dependency(
            Dependency(
                from_node_id=node_ids[0],
                to_node_id=node_ids[1],
                dependency_type=DependencyType.SEQUENTIAL
            )
        )
        
        sample_nodes[node_ids[2]].add_dependency(
            Dependency(
                from_node_id=node_ids[1],
                to_node_id=node_ids[2],
                dependency_type=DependencyType.SEQUENTIAL
            )
        )
        
        all_errors = validator.validate_all_dependencies(sample_nodes)
        
        assert len(all_errors) == 0
    
    def test_validate_all_dependencies_with_errors(self, validator, sample_nodes):
        """Test validate_all_dependencies detects multiple errors"""
        node_ids = list(sample_nodes.keys())
        
        # Add self-reference dependency
        sample_nodes[node_ids[0]].add_dependency(
            Dependency(
                from_node_id=node_ids[0],
                to_node_id=node_ids[0],
                dependency_type=DependencyType.SEQUENTIAL
            )
        )
        
        # Add dependency to missing node
        sample_nodes[node_ids[1]].add_dependency(
            Dependency(
                from_node_id=uuid4(),
                to_node_id=node_ids[1],
                dependency_type=DependencyType.SEQUENTIAL
            )
        )
        
        all_errors = validator.validate_all_dependencies(sample_nodes)
        
        assert len(all_errors) == 2
        assert node_ids[0] in all_errors
        assert node_ids[1] in all_errors
    
    def test_validate_all_dependencies_empty_nodes(self, validator):
        """Test validate_all_dependencies with empty nodes dictionary"""
        all_errors = validator.validate_all_dependencies({})
        
        assert len(all_errors) == 0
    
    def test_is_descendant_direct_child(self, validator, sample_decomposition):
        """Test _is_descendant detects direct child"""
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
            depth_level=1,
            estimated_complexity=0.5
        )
        
        nodes = {
            parent.id.value: parent,
            child.id.value: child,
        }
        
        assert validator._is_descendant(parent.id.value, child, nodes) is True
    
    def test_is_descendant_indirect_descendant(self, validator, sample_decomposition):
        """Test _is_descendant detects indirect descendant"""
        grandparent = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Grandparent",
            description="Grandparent task",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        parent = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Parent",
            description="Parent task",
            parent_id=grandparent.id.value,
            depth_level=1,
            estimated_complexity=0.5
        )
        
        child = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Child",
            description="Child task",
            parent_id=parent.id.value,
            depth_level=2,
            estimated_complexity=0.5
        )
        
        nodes = {
            grandparent.id.value: grandparent,
            parent.id.value: parent,
            child.id.value: child,
        }
        
        assert validator._is_descendant(grandparent.id.value, child, nodes) is True
    
    def test_is_descendant_not_descendant(self, validator, sample_decomposition):
        """Test _is_descendant returns False for non-descendant"""
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
        
        nodes = {
            node1.id.value: node1,
            node2.id.value: node2,
        }
        
        assert validator._is_descendant(node1.id.value, node2, nodes) is False
    
    def test_is_descendant_root_node(self, validator, sample_decomposition):
        """Test _is_descendant returns False for root node"""
        root = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Root",
            description="Root task",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        nodes = {root.id.value: root}
        
        assert validator._is_descendant(uuid4(), root, nodes) is False
    
    def test_is_descendant_missing_parent(self, validator, sample_decomposition):
        """Test _is_descendant handles missing parent gracefully"""
        orphan = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Orphan",
            description="Task with missing parent",
            parent_id=uuid4(),  # Parent doesn't exist in nodes
            depth_level=1,
            estimated_complexity=0.5
        )
        
        nodes = {orphan.id.value: orphan}
        
        assert validator._is_descendant(uuid4(), orphan, nodes) is False
