"""
Unit Tests for Task Decomposition Domain Entities

Tests for Phase 4 hierarchical task decomposition entities.
"""

import pytest
from datetime import datetime
from uuid import uuid4

from core.domain.task_decomposition.entities import (
    TaskNode,
    TaskNodeId,
    TaskDecomposition,
    Dependency,
    TaskNodeStatus,
    DependencyType,
    DecompositionStrategy,
)


class TestTaskNodeId:
    """Tests for TaskNodeId value object"""
    
    def test_task_node_id_generation(self):
        """Test generating a task node ID"""
        node_id = TaskNodeId.generate()
        assert node_id is not None
        assert str(node_id) == str(node_id.value)
    
    def test_task_node_id_immutable(self):
        """Test that task node ID is immutable"""
        node_id = TaskNodeId.generate()
        with pytest.raises(AttributeError):
            node_id.value = uuid4()


class TestDependency:
    """Tests for Dependency value object"""
    
    def test_dependency_creation(self):
        """Test creating a dependency"""
        from_id = uuid4()
        to_id = uuid4()
        
        dep = Dependency(
            from_node_id=from_id,
            to_node_id=to_id,
            dependency_type=DependencyType.SEQUENTIAL
        )
        
        assert dep.from_node_id == from_id
        assert dep.to_node_id == to_id
        assert dep.dependency_type == DependencyType.SEQUENTIAL
    
    def test_dependency_with_condition(self):
        """Test creating conditional dependency"""
        dep = Dependency(
            from_node_id=uuid4(),
            to_node_id=uuid4(),
            dependency_type=DependencyType.CONDITIONAL,
            condition="status == 'success'"
        )
        
        assert dep.condition == "status == 'success'"
    
    def test_dependency_serialization(self):
        """Test dependency to_dict and from_dict"""
        from_id = uuid4()
        to_id = uuid4()
        
        dep = Dependency(
            from_node_id=from_id,
            to_node_id=to_id,
            dependency_type=DependencyType.PARALLEL,
            metadata={"priority": "high"}
        )
        
        data = dep.to_dict()
        restored = Dependency.from_dict(data)
        
        assert restored.from_node_id == from_id
        assert restored.to_node_id == to_id
        assert restored.dependency_type == DependencyType.PARALLEL
        assert restored.metadata["priority"] == "high"


class TestTaskNode:
    """Tests for TaskNode entity"""
    
    def test_task_node_creation(self):
        """Test creating a task node"""
        decomp_id = uuid4()
        node = TaskNode.create(
            decomposition_id=decomp_id,
            name="Test Task",
            description="A test task",
            depth_level=0,
            estimated_complexity=0.7
        )
        
        assert node.decomposition_id == decomp_id
        assert node.name == "Test Task"
        assert node.depth_level == 0
        assert node.estimated_complexity == 0.7
        assert node.status == TaskNodeStatus.PENDING
        assert node.is_leaf is True
    
    def test_task_node_with_parent(self):
        """Test creating a child task node"""
        decomp_id = uuid4()
        parent_id = uuid4()
        
        node = TaskNode.create(
            decomposition_id=decomp_id,
            name="Child Task",
            description="A child task",
            parent_id=parent_id,
            depth_level=1
        )
        
        assert node.parent_id == parent_id
        assert node.depth_level == 1
    
    def test_task_node_depth_validation(self):
        """Test task node depth validation"""
        decomp_id = uuid4()
        
        # Negative depth should fail
        with pytest.raises(ValueError, match="cannot be negative"):
            TaskNode.create(
                decomposition_id=decomp_id,
                name="Invalid",
                description="Invalid depth",
                depth_level=-1
            )
        
        # Depth > 200 should fail
        with pytest.raises(ValueError, match="exceeds maximum"):
            TaskNode.create(
                decomposition_id=decomp_id,
                name="Too Deep",
                description="Too deep",
                depth_level=250
            )
    
    def test_task_node_complexity_validation(self):
        """Test task node complexity validation"""
        decomp_id = uuid4()
        
        # Complexity < 0 should fail
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            TaskNode.create(
                decomposition_id=decomp_id,
                name="Invalid",
                description="Invalid complexity",
                estimated_complexity=-0.1
            )
        
        # Complexity > 1 should fail
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            TaskNode.create(
                decomposition_id=decomp_id,
                name="Invalid",
                description="Invalid complexity",
                estimated_complexity=1.5
            )
    
    def test_task_node_add_child(self):
        """Test adding children to task node"""
        node = TaskNode.create(
            decomposition_id=uuid4(),
            name="Parent",
            description="Parent task",
            depth_level=0
        )
        
        assert node.is_leaf is True
        assert len(node.child_node_ids) == 0
        
        child_id = uuid4()
        node.add_child(child_id)
        
        assert node.is_leaf is False
        assert len(node.child_node_ids) == 1
        assert child_id in node.child_node_ids
        assert node.actual_subtask_count == 1
    
    def test_task_node_add_dependency(self):
        """Test adding dependencies to task node"""
        node = TaskNode.create(
            decomposition_id=uuid4(),
            name="Task",
            description="Task with dependencies",
            depth_level=0
        )
        
        assert len(node.dependencies) == 0
        
        dep = Dependency(
            from_node_id=uuid4(),
            to_node_id=node.id.value,
            dependency_type=DependencyType.SEQUENTIAL
        )
        
        node.add_dependency(dep)
        assert len(node.dependencies) == 1
        
        # Adding same dependency again should not duplicate
        node.add_dependency(dep)
        assert len(node.dependencies) == 1
    
    def test_task_node_remove_dependency(self):
        """Test removing dependencies"""
        node = TaskNode.create(
            decomposition_id=uuid4(),
            name="Task",
            description="Task",
            depth_level=0
        )
        
        from_id = uuid4()
        to_id = node.id.value
        
        dep = Dependency(
            from_node_id=from_id,
            to_node_id=to_id,
            dependency_type=DependencyType.SEQUENTIAL
        )
        
        node.add_dependency(dep)
        assert len(node.dependencies) == 1
        
        removed = node.remove_dependency(from_id, to_id)
        assert removed is True
        assert len(node.dependencies) == 0
    
    def test_task_node_status_transitions(self):
        """Test task node status transitions"""
        node = TaskNode.create(
            decomposition_id=uuid4(),
            name="Task",
            description="Task",
            depth_level=0
        )
        
        # Initial status
        assert node.status == TaskNodeStatus.PENDING
        
        # Pending -> Decomposing
        node.mark_decomposing()
        assert node.status == TaskNodeStatus.DECOMPOSING
        
        # Decomposing -> Decomposed
        node.mark_decomposed()
        assert node.status == TaskNodeStatus.DECOMPOSED
        assert node.decomposed_at is not None
    
    def test_task_node_invalid_status_transition(self):
        """Test invalid status transitions"""
        node = TaskNode.create(
            decomposition_id=uuid4(),
            name="Task",
            description="Task",
            depth_level=0
        )
        
        # Cannot mark as decomposed without being decomposing first
        with pytest.raises(ValueError, match="Cannot mark as decomposed"):
            node.mark_decomposed()
    
    def test_task_node_mark_ready(self):
        """Test marking task as ready"""
        node = TaskNode.create(
            decomposition_id=uuid4(),
            name="Task",
            description="Task",
            depth_level=0
        )
        
        node.mark_ready()
        assert node.status == TaskNodeStatus.READY
    
    def test_task_node_mark_blocked(self):
        """Test marking task as blocked"""
        node = TaskNode.create(
            decomposition_id=uuid4(),
            name="Task",
            description="Task",
            depth_level=0
        )
        
        node.mark_blocked("Waiting for resources")
        assert node.status == TaskNodeStatus.BLOCKED
        assert node.metadata["blocked_reason"] == "Waiting for resources"
    
    def test_task_node_mark_failed(self):
        """Test marking task as failed"""
        node = TaskNode.create(
            decomposition_id=uuid4(),
            name="Task",
            description="Task",
            depth_level=0
        )
        
        node.mark_failed("Execution error")
        assert node.status == TaskNodeStatus.FAILED
        assert node.metadata["error"] == "Execution error"
    
    def test_task_node_is_decomposable(self):
        """Test checking if task is decomposable"""
        node = TaskNode.create(
            decomposition_id=uuid4(),
            name="Task",
            description="Task",
            depth_level=0,
            estimated_complexity=0.8
        )
        
        # High complexity, pending, leaf -> decomposable
        assert node.is_decomposable() is True
        
        # After decomposing, not decomposable
        node.mark_decomposing()
        assert node.is_decomposable() is False
        
        # Low complexity -> not decomposable
        simple_node = TaskNode.create(
            decomposition_id=uuid4(),
            name="Simple",
            description="Simple task",
            depth_level=0,
            estimated_complexity=0.2
        )
        assert simple_node.is_decomposable() is False
    
    def test_task_node_serialization(self):
        """Test task node to_dict and from_dict"""
        decomp_id = uuid4()
        node = TaskNode.create(
            decomposition_id=decomp_id,
            name="Test Task",
            description="Description",
            depth_level=5,
            estimated_complexity=0.6,
            tags=["important", "urgent"]
        )
        
        # Add a child and dependency
        child_id = uuid4()
        node.add_child(child_id)
        
        dep = Dependency(
            from_node_id=uuid4(),
            to_node_id=node.id.value,
            dependency_type=DependencyType.PARALLEL
        )
        node.add_dependency(dep)
        
        # Serialize and deserialize
        data = node.to_dict()
        restored = TaskNode.from_dict(data)
        
        assert restored.decomposition_id == decomp_id
        assert restored.name == "Test Task"
        assert restored.depth_level == 5
        assert restored.estimated_complexity == 0.6
        assert len(restored.child_node_ids) == 1
        assert len(restored.dependencies) == 1
        assert "important" in restored.tags


class TestTaskDecomposition:
    """Tests for TaskDecomposition entity"""
    
    def test_task_decomposition_creation(self):
        """Test creating a task decomposition"""
        workflow_id = uuid4()
        decomp = TaskDecomposition.create(
            workflow_execution_id=workflow_id,
            root_task_name="Build System",
            root_task_description="Complete system build",
            strategy=DecompositionStrategy.HYBRID
        )
        
        assert decomp.workflow_execution_id == workflow_id
        assert decomp.root_task_name == "Build System"
        assert decomp.strategy == DecompositionStrategy.HYBRID
        assert decomp.total_nodes == 0
        assert decomp.is_complete is False
    
    def test_task_decomposition_register_node(self):
        """Test registering nodes in decomposition"""
        decomp = TaskDecomposition.create(
            workflow_execution_id=uuid4(),
            root_task_name="Task",
            root_task_description="Task"
        )
        
        node_id1 = uuid4()
        node_id2 = uuid4()
        
        decomp.register_node(node_id1, depth=0)
        assert decomp.total_nodes == 1
        assert decomp.max_depth_reached == 0
        
        decomp.register_node(node_id2, depth=5)
        assert decomp.total_nodes == 2
        assert decomp.max_depth_reached == 5
    
    def test_task_decomposition_set_root_node(self):
        """Test setting root node"""
        decomp = TaskDecomposition.create(
            workflow_execution_id=uuid4(),
            root_task_name="Task",
            root_task_description="Task"
        )
        
        root_id = uuid4()
        decomp.set_root_node(root_id)
        
        assert decomp.root_node_id == root_id
        assert decomp.total_nodes == 1
        assert root_id in decomp.all_node_ids
    
    def test_task_decomposition_register_leaf_node(self):
        """Test registering leaf nodes"""
        decomp = TaskDecomposition.create(
            workflow_execution_id=uuid4(),
            root_task_name="Task",
            root_task_description="Task"
        )
        
        assert decomp.leaf_node_count == 0
        
        decomp.register_leaf_node(uuid4())
        assert decomp.leaf_node_count == 1
        
        decomp.register_leaf_node(uuid4())
        assert decomp.leaf_node_count == 2
    
    def test_task_decomposition_mark_complete(self):
        """Test marking decomposition as complete"""
        decomp = TaskDecomposition.create(
            workflow_execution_id=uuid4(),
            root_task_name="Task",
            root_task_description="Task"
        )
        
        assert decomp.is_complete is False
        assert decomp.completed_at is None
        
        decomp.mark_complete()
        
        assert decomp.is_complete is True
        assert decomp.completed_at is not None
    
    def test_task_decomposition_mark_has_cycles(self):
        """Test marking decomposition as having cycles"""
        decomp = TaskDecomposition.create(
            workflow_execution_id=uuid4(),
            root_task_name="Task",
            root_task_description="Task"
        )
        
        assert decomp.has_cycles is False
        
        decomp.mark_has_cycles()
        
        assert decomp.has_cycles is True
    
    def test_task_decomposition_get_statistics(self):
        """Test getting decomposition statistics"""
        decomp = TaskDecomposition.create(
            workflow_execution_id=uuid4(),
            root_task_name="Task",
            root_task_description="Task"
        )
        
        # Add some nodes
        decomp.register_node(uuid4(), depth=0)
        decomp.register_node(uuid4(), depth=1)
        decomp.register_node(uuid4(), depth=2)
        decomp.register_leaf_node(uuid4())
        decomp.register_leaf_node(uuid4())
        
        stats = decomp.get_statistics()
        
        assert stats["total_nodes"] == 3
        assert stats["max_depth_reached"] == 2
        assert stats["leaf_node_count"] == 2
        assert stats["is_complete"] is False
        assert stats["has_cycles"] is False
        assert stats["strategy"] == "hybrid"
    
    def test_task_decomposition_serialization(self):
        """Test decomposition to_dict and from_dict"""
        workflow_id = uuid4()
        decomp = TaskDecomposition.create(
            workflow_execution_id=workflow_id,
            root_task_name="Build OS",
            root_task_description="Build operating system",
            strategy=DecompositionStrategy.DEPTH_FIRST
        )
        
        # Add nodes
        root_id = uuid4()
        decomp.set_root_node(root_id)
        decomp.register_node(uuid4(), depth=1)
        decomp.register_leaf_node(uuid4())
        decomp.mark_complete()
        
        # Serialize and deserialize
        data = decomp.to_dict()
        restored = TaskDecomposition.from_dict(data)
        
        assert restored.workflow_execution_id == workflow_id
        assert restored.root_task_name == "Build OS"
        assert restored.strategy == DecompositionStrategy.DEPTH_FIRST
        assert restored.total_nodes == 2
        assert restored.leaf_node_count == 1
        assert restored.is_complete is True
        assert restored.root_node_id == root_id
