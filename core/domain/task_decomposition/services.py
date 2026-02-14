"""
Task Decomposition Domain - Services

Domain services for hierarchical task decomposition.
"""

from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

from .entities import (
    TaskNode,
    TaskDecomposition,
    Dependency,
    DependencyType,
    DecompositionStrategy,
)


class RecursiveDecomposer:
    """
    Domain service for recursive task decomposition.
    
    Decomposes tasks into subtasks recursively up to 100+ depth levels.
    """
    
    def __init__(self, max_depth: int = 150):
        """
        Initialize recursive decomposer.
        
        Args:
            max_depth: Maximum decomposition depth (default: 150)
        """
        self.max_depth = max_depth
    
    def can_decompose(self, task_node: TaskNode) -> bool:
        """
        Check if task node can be decomposed further.
        
        Args:
            task_node: Task node to check
            
        Returns:
            True if can decompose, False otherwise
        """
        return (
            task_node.is_decomposable() and
            task_node.depth_level < self.max_depth
        )
    
    def should_decompose(
        self,
        task_node: TaskNode,
        complexity_threshold: float = 0.3
    ) -> bool:
        """
        Determine if task should be decomposed based on complexity.
        
        Args:
            task_node: Task node to evaluate
            complexity_threshold: Minimum complexity to warrant decomposition
            
        Returns:
            True if should decompose, False otherwise
        """
        return (
            task_node.estimated_complexity >= complexity_threshold and
            self.can_decompose(task_node)
        )
    
    def estimate_subtask_count(
        self,
        task_node: TaskNode,
        strategy: DecompositionStrategy
    ) -> int:
        """
        Estimate number of subtasks for decomposition.
        
        Args:
            task_node: Task node to decompose
            strategy: Decomposition strategy
            
        Returns:
            Estimated subtask count
        """
        base_complexity = task_node.estimated_complexity
        
        # Strategy affects decomposition granularity
        if strategy == DecompositionStrategy.BREADTH_FIRST:
            # More subtasks, less depth
            return int(5 + (base_complexity * 10))
        elif strategy == DecompositionStrategy.DEPTH_FIRST:
            # Fewer subtasks, more depth
            return int(2 + (base_complexity * 3))
        elif strategy == DecompositionStrategy.HYBRID:
            # Balanced approach
            return int(3 + (base_complexity * 5))
        else:  # ADAPTIVE
            # Adapts based on depth
            depth_factor = 1.0 - (task_node.depth_level / self.max_depth)
            return int(3 + (base_complexity * 7 * depth_factor))
    
    def decompose_task(
        self,
        task_node: TaskNode,
        decomposition: TaskDecomposition,
        subtask_specifications: List[Dict[str, any]]
    ) -> List[TaskNode]:
        """
        Decompose a task into subtasks.
        
        Args:
            task_node: Parent task node
            decomposition: Decomposition context
            subtask_specifications: List of subtask specifications
            
        Returns:
            List of created subtask nodes
        """
        if not self.can_decompose(task_node):
            raise ValueError(f"Cannot decompose task at depth {task_node.depth_level}")
        
        task_node.mark_decomposing()
        
        subtasks = []
        child_depth = task_node.depth_level + 1
        
        for spec in subtask_specifications:
            subtask = TaskNode.create(
                decomposition_id=decomposition.id,
                name=spec.get("name", f"Subtask of {task_node.name}"),
                description=spec.get("description", ""),
                parent_id=task_node.id.value,
                depth_level=child_depth,
                estimated_complexity=spec.get("complexity", 0.5),
                tags=spec.get("tags", []),
            )
            
            task_node.add_child(subtask.id.value)
            subtasks.append(subtask)
            
            # Register with decomposition
            decomposition.register_node(subtask.id.value, child_depth)
            
            if subtask.estimated_complexity < 0.3:  # Likely a leaf
                decomposition.register_leaf_node(subtask.id.value)
        
        task_node.mark_decomposed()
        
        return subtasks


class DependencyValidator:
    """
    Domain service for validating task dependencies.
    
    Ensures dependency integrity and logical consistency.
    """
    
    def validate_dependency(
        self,
        dependency: Dependency,
        all_nodes: Dict[UUID, TaskNode]
    ) -> List[str]:
        """
        Validate a single dependency.
        
        Args:
            dependency: Dependency to validate
            all_nodes: All nodes in decomposition
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check nodes exist
        if dependency.from_node_id not in all_nodes:
            errors.append(f"From node {dependency.from_node_id} not found")
        
        if dependency.to_node_id not in all_nodes:
            errors.append(f"To node {dependency.to_node_id} not found")
        
        if errors:
            return errors
        
        from_node = all_nodes[dependency.from_node_id]
        to_node = all_nodes[dependency.to_node_id]
        
        # Cannot depend on self
        if dependency.from_node_id == dependency.to_node_id:
            errors.append("Task cannot depend on itself")
        
        # Cannot depend on descendants (would create cycle)
        if self._is_descendant(from_node.id.value, to_node, all_nodes):
            errors.append(f"Dependency creates cycle: {from_node.name} → {to_node.name}")
        
        # Conditional dependencies need condition
        if dependency.dependency_type == DependencyType.CONDITIONAL:
            if not dependency.condition:
                errors.append("Conditional dependency requires a condition")
        
        return errors
    
    def _is_descendant(
        self,
        ancestor_id: UUID,
        node: TaskNode,
        all_nodes: Dict[UUID, TaskNode]
    ) -> bool:
        """Check if node is a descendant of ancestor_id"""
        if node.parent_id is None:
            return False
        
        if node.parent_id == ancestor_id:
            return True
        
        if node.parent_id in all_nodes:
            parent = all_nodes[node.parent_id]
            return self._is_descendant(ancestor_id, parent, all_nodes)
        
        return False
    
    def validate_all_dependencies(
        self,
        nodes: Dict[UUID, TaskNode]
    ) -> Dict[UUID, List[str]]:
        """
        Validate all dependencies in a set of nodes.
        
        Args:
            nodes: Dictionary of node_id -> TaskNode
            
        Returns:
            Dictionary of node_id -> list of errors
        """
        all_errors = {}
        
        for node_id, node in nodes.items():
            node_errors = []
            
            for dependency in node.dependencies:
                errors = self.validate_dependency(dependency, nodes)
                node_errors.extend(errors)
            
            if node_errors:
                all_errors[node_id] = node_errors
        
        return all_errors


class CycleDetector:
    """
    Domain service for detecting cycles in task dependency graph.
    
    Uses depth-first search to detect circular dependencies.
    """
    
    def detect_cycles(
        self,
        nodes: Dict[UUID, TaskNode]
    ) -> List[List[UUID]]:
        """
        Detect all cycles in the dependency graph.
        
        Args:
            nodes: Dictionary of node_id -> TaskNode
            
        Returns:
            List of cycles (each cycle is a list of node IDs)
        """
        visited = set()
        recursion_stack = set()
        cycles = []
        
        def dfs(node_id: UUID, path: List[UUID]) -> None:
            """Depth-first search to detect cycles"""
            if node_id in recursion_stack:
                # Found a cycle
                cycle_start = path.index(node_id)
                cycle = path[cycle_start:] + [node_id]
                cycles.append(cycle)
                return
            
            if node_id in visited:
                return
            
            visited.add(node_id)
            recursion_stack.add(node_id)
            
            if node_id in nodes:
                node = nodes[node_id]
                for dependency in node.dependencies:
                    dfs(dependency.from_node_id, path + [node_id])
            
            recursion_stack.remove(node_id)
        
        # Check all nodes
        for node_id in nodes.keys():
            if node_id not in visited:
                dfs(node_id, [])
        
        return cycles
    
    def has_cycle(self, nodes: Dict[UUID, TaskNode]) -> bool:
        """
        Quick check if any cycles exist.
        
        Args:
            nodes: Dictionary of node_id -> TaskNode
            
        Returns:
            True if cycles exist, False otherwise
        """
        cycles = self.detect_cycles(nodes)
        return len(cycles) > 0
    
    def get_cycle_description(
        self,
        cycle: List[UUID],
        nodes: Dict[UUID, TaskNode]
    ) -> str:
        """
        Get human-readable description of a cycle.
        
        Args:
            cycle: List of node IDs in cycle
            nodes: Dictionary of node_id -> TaskNode
            
        Returns:
            Cycle description
        """
        names = []
        for node_id in cycle:
            if node_id in nodes:
                names.append(nodes[node_id].name)
            else:
                names.append(str(node_id))
        
        return " → ".join(names)


class IntegrityEnforcer:
    """
    Domain service for enforcing decomposition integrity.
    
    Ensures logical consistency across the decomposition tree.
    """
    
    def __init__(self):
        self.dependency_validator = DependencyValidator()
        self.cycle_detector = CycleDetector()
    
    def validate_decomposition(
        self,
        decomposition: TaskDecomposition,
        nodes: Dict[UUID, TaskNode]
    ) -> Tuple[bool, List[str]]:
        """
        Validate entire decomposition for integrity.
        
        Args:
            decomposition: Decomposition to validate
            nodes: All nodes in decomposition
            
        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        
        # Check root node exists
        if decomposition.root_node_id is None:
            errors.append("Decomposition has no root node")
        elif decomposition.root_node_id not in nodes:
            errors.append(f"Root node {decomposition.root_node_id} not found")
        
        # Check node count consistency
        if len(nodes) != decomposition.total_nodes:
            errors.append(
                f"Node count mismatch: {len(nodes)} nodes but "
                f"decomposition reports {decomposition.total_nodes}"
            )
        
        # Validate all dependencies
        dependency_errors = self.dependency_validator.validate_all_dependencies(nodes)
        for node_id, node_errors in dependency_errors.items():
            for error in node_errors:
                errors.append(f"Node {nodes[node_id].name}: {error}")
        
        # Check for cycles
        cycles = self.cycle_detector.detect_cycles(nodes)
        if cycles:
            decomposition.mark_has_cycles()
            for cycle in cycles:
                cycle_desc = self.cycle_detector.get_cycle_description(cycle, nodes)
                errors.append(f"Cycle detected: {cycle_desc}")
        
        # Check depth consistency
        for node_id, node in nodes.items():
            if node.parent_id is not None:
                if node.parent_id not in nodes:
                    errors.append(f"Node {node.name} has missing parent {node.parent_id}")
                else:
                    parent = nodes[node.parent_id]
                    expected_depth = parent.depth_level + 1
                    if node.depth_level != expected_depth:
                        errors.append(
                            f"Node {node.name} has incorrect depth: "
                            f"{node.depth_level} (expected {expected_depth})"
                        )
        
        return len(errors) == 0, errors
    
    def enforce_parent_child_consistency(
        self,
        parent: TaskNode,
        children: List[TaskNode]
    ) -> List[str]:
        """
        Enforce parent-child relationship consistency.
        
        Args:
            parent: Parent task node
            children: Child task nodes
            
        Returns:
            List of consistency errors
        """
        errors = []
        
        # All children should reference parent
        for child in children:
            if child.parent_id != parent.id.value:
                errors.append(
                    f"Child {child.name} does not reference parent {parent.name}"
                )
            
            # Child depth should be parent depth + 1
            expected_depth = parent.depth_level + 1
            if child.depth_level != expected_depth:
                errors.append(
                    f"Child {child.name} has incorrect depth: "
                    f"{child.depth_level} (expected {expected_depth})"
                )
        
        # Parent should reference all children
        child_ids = {child.id.value for child in children}
        parent_child_ids = set(parent.child_node_ids)
        
        missing = child_ids - parent_child_ids
        if missing:
            errors.append(f"Parent missing child references: {missing}")
        
        extra = parent_child_ids - child_ids
        if extra:
            errors.append(f"Parent has extra child references: {extra}")
        
        return errors
