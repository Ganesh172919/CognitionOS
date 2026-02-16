"""
Unit Tests for CycleDetector Service

Tests for CycleDetector domain service business logic.
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
from core.domain.task_decomposition.services import CycleDetector


class TestCycleDetector:
    """Tests for CycleDetector service"""
    
    @pytest.fixture
    def detector(self):
        """Create cycle detector instance"""
        return CycleDetector()
    
    @pytest.fixture
    def sample_decomposition(self):
        """Create sample decomposition"""
        return TaskDecomposition.create(
            workflow_execution_id=uuid4(),
            root_task_name="Test Task",
            root_task_description="Test description",
            strategy=DecompositionStrategy.HYBRID
        )
    
    def test_detect_cycles_no_cycles(self, detector, sample_decomposition):
        """Test detect_cycles with acyclic graph"""
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
        
        node3 = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Node 3",
            description="Third node",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        # Linear dependency: 1 -> 2 -> 3
        node2.add_dependency(
            Dependency(
                from_node_id=node1.id.value,
                to_node_id=node2.id.value,
                dependency_type=DependencyType.SEQUENTIAL
            )
        )
        
        node3.add_dependency(
            Dependency(
                from_node_id=node2.id.value,
                to_node_id=node3.id.value,
                dependency_type=DependencyType.SEQUENTIAL
            )
        )
        
        nodes = {
            node1.id.value: node1,
            node2.id.value: node2,
            node3.id.value: node3,
        }
        
        cycles = detector.detect_cycles(nodes)
        
        assert len(cycles) == 0
    
    def test_detect_cycles_simple_cycle(self, detector, sample_decomposition):
        """Test detect_cycles detects simple two-node cycle"""
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
        
        # Create cycle: 1 -> 2 -> 1
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
        
        nodes = {
            node1.id.value: node1,
            node2.id.value: node2,
        }
        
        cycles = detector.detect_cycles(nodes)
        
        assert len(cycles) > 0
        # Cycle should contain both nodes
        assert any(node1.id.value in cycle and node2.id.value in cycle for cycle in cycles)
    
    def test_detect_cycles_self_loop(self, detector, sample_decomposition):
        """Test detect_cycles detects self-loop"""
        node = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Self Loop Node",
            description="Node with self-loop",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        # Create self-loop
        node.add_dependency(
            Dependency(
                from_node_id=node.id.value,
                to_node_id=node.id.value,
                dependency_type=DependencyType.SEQUENTIAL
            )
        )
        
        nodes = {node.id.value: node}
        
        cycles = detector.detect_cycles(nodes)
        
        assert len(cycles) > 0
        # Cycle should contain the node
        assert any(node.id.value in cycle for cycle in cycles)
    
    def test_detect_cycles_three_node_cycle(self, detector, sample_decomposition):
        """Test detect_cycles detects three-node cycle"""
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
        
        node3 = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Node 3",
            description="Third node",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        # Create cycle: 1 -> 2 -> 3 -> 1
        node2.add_dependency(
            Dependency(
                from_node_id=node1.id.value,
                to_node_id=node2.id.value,
                dependency_type=DependencyType.SEQUENTIAL
            )
        )
        
        node3.add_dependency(
            Dependency(
                from_node_id=node2.id.value,
                to_node_id=node3.id.value,
                dependency_type=DependencyType.SEQUENTIAL
            )
        )
        
        node1.add_dependency(
            Dependency(
                from_node_id=node3.id.value,
                to_node_id=node1.id.value,
                dependency_type=DependencyType.SEQUENTIAL
            )
        )
        
        nodes = {
            node1.id.value: node1,
            node2.id.value: node2,
            node3.id.value: node3,
        }
        
        cycles = detector.detect_cycles(nodes)
        
        assert len(cycles) > 0
        # Cycle should contain all three nodes
        assert any(
            node1.id.value in cycle and node2.id.value in cycle and node3.id.value in cycle
            for cycle in cycles
        )
    
    def test_detect_cycles_multiple_cycles(self, detector, sample_decomposition):
        """Test detect_cycles detects multiple independent cycles"""
        # First cycle: 1 -> 2 -> 1
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
        
        # Second cycle: 3 -> 4 -> 3
        node3 = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Node 3",
            description="Third node",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        node4 = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Node 4",
            description="Fourth node",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        node4.add_dependency(
            Dependency(
                from_node_id=node3.id.value,
                to_node_id=node4.id.value,
                dependency_type=DependencyType.SEQUENTIAL
            )
        )
        
        node3.add_dependency(
            Dependency(
                from_node_id=node4.id.value,
                to_node_id=node3.id.value,
                dependency_type=DependencyType.SEQUENTIAL
            )
        )
        
        nodes = {
            node1.id.value: node1,
            node2.id.value: node2,
            node3.id.value: node3,
            node4.id.value: node4,
        }
        
        cycles = detector.detect_cycles(nodes)
        
        # Should detect both cycles
        assert len(cycles) >= 2
    
    def test_detect_cycles_empty_graph(self, detector):
        """Test detect_cycles with empty graph"""
        cycles = detector.detect_cycles({})
        
        assert len(cycles) == 0
    
    def test_detect_cycles_single_node_no_dependencies(self, detector, sample_decomposition):
        """Test detect_cycles with single isolated node"""
        node = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Isolated Node",
            description="No dependencies",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        nodes = {node.id.value: node}
        
        cycles = detector.detect_cycles(nodes)
        
        assert len(cycles) == 0
    
    def test_has_cycle_returns_true_with_cycle(self, detector, sample_decomposition):
        """Test has_cycle returns True when cycle exists"""
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
        
        nodes = {
            node1.id.value: node1,
            node2.id.value: node2,
        }
        
        assert detector.has_cycle(nodes) is True
    
    def test_has_cycle_returns_false_without_cycle(self, detector, sample_decomposition):
        """Test has_cycle returns False when no cycle exists"""
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
        
        # Linear dependency (no cycle)
        node2.add_dependency(
            Dependency(
                from_node_id=node1.id.value,
                to_node_id=node2.id.value,
                dependency_type=DependencyType.SEQUENTIAL
            )
        )
        
        nodes = {
            node1.id.value: node1,
            node2.id.value: node2,
        }
        
        assert detector.has_cycle(nodes) is False
    
    def test_get_cycle_description_with_valid_nodes(self, detector, sample_decomposition):
        """Test get_cycle_description generates readable description"""
        node1 = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Task A",
            description="First task",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        node2 = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Task B",
            description="Second task",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        node3 = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Task C",
            description="Third task",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        nodes = {
            node1.id.value: node1,
            node2.id.value: node2,
            node3.id.value: node3,
        }
        
        cycle = [node1.id.value, node2.id.value, node3.id.value]
        description = detector.get_cycle_description(cycle, nodes)
        
        assert "Task A" in description
        assert "Task B" in description
        assert "Task C" in description
        assert "→" in description
    
    def test_get_cycle_description_with_missing_nodes(self, detector, sample_decomposition):
        """Test get_cycle_description handles missing nodes gracefully"""
        node1 = TaskNode.create(
            decomposition_id=sample_decomposition.id,
            name="Task A",
            description="First task",
            depth_level=0,
            estimated_complexity=0.5
        )
        
        missing_id = uuid4()
        
        nodes = {node1.id.value: node1}
        
        cycle = [node1.id.value, missing_id]
        description = detector.get_cycle_description(cycle, nodes)
        
        assert "Task A" in description
        assert str(missing_id) in description
        assert "→" in description
    
    def test_get_cycle_description_empty_cycle(self, detector):
        """Test get_cycle_description with empty cycle"""
        description = detector.get_cycle_description([], {})
        
        assert description == ""
    
    def test_detect_cycles_complex_graph_with_cycle(self, detector, sample_decomposition):
        """Test detect_cycles with complex graph containing one cycle"""
        nodes_list = []
        for i in range(5):
            node = TaskNode.create(
                decomposition_id=sample_decomposition.id,
                name=f"Node {i}",
                description=f"Node {i} description",
                depth_level=0,
                estimated_complexity=0.5
            )
            nodes_list.append(node)
        
        # Create cycle: 0 -> 1 -> 2 -> 0
        # (Dependencies show "depends on", so reverse direction)
        nodes_list[1].add_dependency(
            Dependency(
                from_node_id=nodes_list[0].id.value,
                to_node_id=nodes_list[1].id.value,
                dependency_type=DependencyType.SEQUENTIAL
            )
        )
        
        nodes_list[2].add_dependency(
            Dependency(
                from_node_id=nodes_list[1].id.value,
                to_node_id=nodes_list[2].id.value,
                dependency_type=DependencyType.SEQUENTIAL
            )
        )
        
        # Create cycle back to 0
        nodes_list[0].add_dependency(
            Dependency(
                from_node_id=nodes_list[2].id.value,
                to_node_id=nodes_list[0].id.value,
                dependency_type=DependencyType.SEQUENTIAL
            )
        )
        
        # Add some non-cycle nodes
        nodes_list[3].add_dependency(
            Dependency(
                from_node_id=nodes_list[0].id.value,
                to_node_id=nodes_list[3].id.value,
                dependency_type=DependencyType.SEQUENTIAL
            )
        )
        
        nodes_list[4].add_dependency(
            Dependency(
                from_node_id=nodes_list[3].id.value,
                to_node_id=nodes_list[4].id.value,
                dependency_type=DependencyType.SEQUENTIAL
            )
        )
        
        nodes = {node.id.value: node for node in nodes_list}
        
        cycles = detector.detect_cycles(nodes)
        
        assert len(cycles) > 0
