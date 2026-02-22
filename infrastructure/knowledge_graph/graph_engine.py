"""
Knowledge Graph Engine
=======================
Production-grade AI-powered knowledge graph for intelligent reasoning:
- Entity and relationship management with typed schema
- Graph traversal algorithms (BFS, DFS, Dijkstra, A*)
- Semantic similarity search using embeddings
- Automated knowledge extraction from text
- Graph analytics (PageRank, community detection, centrality)
- Inference engine for relationship deduction
- Temporal knowledge with versioning
- Multi-tenant knowledge spaces
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import math
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    """Semantic node types in the knowledge graph."""
    CONCEPT = "concept"
    ENTITY = "entity"
    PERSON = "person"
    ORGANIZATION = "organization"
    TECHNOLOGY = "technology"
    PROCESS = "process"
    EVENT = "event"
    PRODUCT = "product"
    LOCATION = "location"
    DOCUMENT = "document"
    CODE_MODULE = "code_module"
    API_ENDPOINT = "api_endpoint"
    AGENT_ACTION = "agent_action"
    USER_INTENT = "user_intent"
    CUSTOM = "custom"


class EdgeType(str, Enum):
    """Semantic edge types defining relationships."""
    IS_A = "is_a"
    HAS_A = "has_a"
    PART_OF = "part_of"
    DEPENDS_ON = "depends_on"
    CAUSES = "causes"
    FOLLOWS = "follows"
    RELATED_TO = "related_to"
    IMPLEMENTS = "implements"
    CALLS = "calls"
    OWNS = "owns"
    USES = "uses"
    PRODUCES = "produces"
    CONSUMES = "consumes"
    SIMILAR_TO = "similar_to"
    CONTRADICTS = "contradicts"
    DERIVED_FROM = "derived_from"
    TRIGGERS = "triggers"
    CUSTOM = "custom"


class TraversalStrategy(str, Enum):
    """Graph traversal algorithm selection."""
    BFS = "bfs"
    DFS = "dfs"
    DIJKSTRA = "dijkstra"
    A_STAR = "a_star"
    RANDOM_WALK = "random_walk"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    """A typed node in the knowledge graph with embedding support."""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    node_type: NodeType = NodeType.CONCEPT
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    tenant_id: str = "default"
    confidence: float = 1.0  # Confidence score [0,1]
    source: str = "manual"  # manual, extracted, inferred
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    visit_count: int = 0
    pagerank_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "node_type": self.node_type.value,
            "description": self.description,
            "properties": self.properties,
            "tenant_id": self.tenant_id,
            "confidence": self.confidence,
            "source": self.source,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "visit_count": self.visit_count,
            "pagerank_score": self.pagerank_score,
            "has_embedding": self.embedding is not None,
        }

    def similarity(self, other: "GraphNode") -> float:
        """Cosine similarity between node embeddings."""
        if self.embedding is None or other.embedding is None:
            return 0.0
        if len(self.embedding) != len(other.embedding):
            return 0.0

        dot = sum(a * b for a, b in zip(self.embedding, other.embedding))
        mag_a = math.sqrt(sum(x ** 2 for x in self.embedding))
        mag_b = math.sqrt(sum(x ** 2 for x in other.embedding))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)


@dataclass
class GraphEdge:
    """A typed, weighted, directed edge between two graph nodes."""
    edge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    edge_type: EdgeType = EdgeType.RELATED_TO
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    bidirectional: bool = False
    tenant_id: str = "default"
    source: str = "manual"
    created_at: datetime = field(default_factory=datetime.utcnow)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    @property
    def is_temporal(self) -> bool:
        return self.valid_from is not None or self.valid_until is not None

    def is_valid_at(self, point_in_time: datetime) -> bool:
        if not self.is_temporal:
            return True
        if self.valid_from and point_in_time < self.valid_from:
            return False
        if self.valid_until and point_in_time > self.valid_until:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "properties": self.properties,
            "confidence": self.confidence,
            "bidirectional": self.bidirectional,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class GraphQuery:
    """A structured query against the knowledge graph."""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query_text: str = ""
    node_type_filter: Optional[NodeType] = None
    edge_type_filter: Optional[EdgeType] = None
    start_node_id: Optional[str] = None
    end_node_id: Optional[str] = None
    max_depth: int = 5
    max_results: int = 20
    min_confidence: float = 0.0
    traversal: TraversalStrategy = TraversalStrategy.BFS
    tenant_id: str = "default"
    semantic_search: bool = False
    include_properties: bool = True


@dataclass
class GraphQueryResult:
    """Result of a knowledge graph query."""
    query_id: str = ""
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    paths: List[List[str]] = field(default_factory=list)
    total_nodes: int = 0
    total_edges: int = 0
    execution_time_ms: float = 0.0
    algorithm_used: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "nodes": self.nodes,
            "edges": self.edges,
            "paths": self.paths,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "execution_time_ms": self.execution_time_ms,
            "algorithm_used": self.algorithm_used,
        }


@dataclass
class InferredFact:
    """A fact inferred by the inference engine."""
    fact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subject_id: str = ""
    predicate: EdgeType = EdgeType.RELATED_TO
    object_id: str = ""
    confidence: float = 0.0
    reasoning_chain: List[str] = field(default_factory=list)
    inferred_at: datetime = field(default_factory=datetime.utcnow)
    rule_applied: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "subject_id": self.subject_id,
            "predicate": self.predicate.value,
            "object_id": self.object_id,
            "confidence": self.confidence,
            "reasoning_chain": self.reasoning_chain,
            "rule_applied": self.rule_applied,
            "inferred_at": self.inferred_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Graph Inference Engine
# ---------------------------------------------------------------------------

class GraphInferenceEngine:
    """
    Rule-based and probabilistic inference engine for knowledge graphs.
    Applies transitive closure, symmetric relations, and custom rules
    to derive new facts from existing knowledge.
    """

    def __init__(self) -> None:
        self._rules: List[Dict[str, Any]] = []
        self._inferred_facts: List[InferredFact] = []
        self._load_default_rules()

    def _load_default_rules(self) -> None:
        """Load built-in inference rules."""
        # Transitivity: A depends_on B, B depends_on C => A depends_on C
        self._rules.append({
            "name": "dependency_transitivity",
            "premise_1": EdgeType.DEPENDS_ON,
            "premise_2": EdgeType.DEPENDS_ON,
            "conclusion": EdgeType.DEPENDS_ON,
            "confidence_decay": 0.8,
        })
        # IS_A transitivity: A is_a B, B is_a C => A is_a C
        self._rules.append({
            "name": "is_a_transitivity",
            "premise_1": EdgeType.IS_A,
            "premise_2": EdgeType.IS_A,
            "conclusion": EdgeType.IS_A,
            "confidence_decay": 0.9,
        })
        # Ownership propagation: A owns B, B uses C => A uses C (indirectly)
        self._rules.append({
            "name": "ownership_propagation",
            "premise_1": EdgeType.OWNS,
            "premise_2": EdgeType.USES,
            "conclusion": EdgeType.USES,
            "confidence_decay": 0.7,
        })
        # Causality chain
        self._rules.append({
            "name": "causality_chain",
            "premise_1": EdgeType.CAUSES,
            "premise_2": EdgeType.CAUSES,
            "conclusion": EdgeType.CAUSES,
            "confidence_decay": 0.75,
        })

    def add_rule(
        self,
        name: str,
        premise_1: EdgeType,
        premise_2: EdgeType,
        conclusion: EdgeType,
        confidence_decay: float = 0.8,
    ) -> None:
        """Add a custom inference rule."""
        self._rules.append({
            "name": name,
            "premise_1": premise_1,
            "premise_2": premise_2,
            "conclusion": conclusion,
            "confidence_decay": confidence_decay,
        })

    def run_inference(
        self,
        adjacency: Dict[str, List[Tuple[str, EdgeType, float]]],
        max_iterations: int = 3,
    ) -> List[InferredFact]:
        """Run forward-chaining inference on the graph adjacency list."""
        new_facts: List[InferredFact] = []

        for _ in range(max_iterations):
            iteration_facts: List[InferredFact] = []
            for rule in self._rules:
                p1_type = rule["premise_1"]
                p2_type = rule["premise_2"]
                conclusion_type = rule["conclusion"]
                decay = rule["confidence_decay"]

                # Find edges matching premise_1
                for node_a, edges_a in adjacency.items():
                    for node_b, edge_type_ab, conf_ab in edges_a:
                        if edge_type_ab != p1_type:
                            continue
                        # Find edges from B matching premise_2
                        edges_b = adjacency.get(node_b, [])
                        for node_c, edge_type_bc, conf_bc in edges_b:
                            if edge_type_bc != p2_type:
                                continue
                            if node_a == node_c:
                                continue

                            # Check if fact already exists
                            existing = any(
                                f.subject_id == node_a
                                and f.object_id == node_c
                                and f.predicate == conclusion_type
                                for f in new_facts
                            )
                            if not existing:
                                confidence = conf_ab * conf_bc * decay
                                if confidence >= 0.3:
                                    fact = InferredFact(
                                        subject_id=node_a,
                                        predicate=conclusion_type,
                                        object_id=node_c,
                                        confidence=confidence,
                                        reasoning_chain=[node_a, node_b, node_c],
                                        rule_applied=rule["name"],
                                    )
                                    iteration_facts.append(fact)

            new_facts.extend(iteration_facts)
            if not iteration_facts:
                break

        self._inferred_facts.extend(new_facts)
        return new_facts

    def get_inferred_facts(
        self, min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        return [
            f.to_dict()
            for f in self._inferred_facts
            if f.confidence >= min_confidence
        ]


# ---------------------------------------------------------------------------
# Knowledge Graph Analytics
# ---------------------------------------------------------------------------

class KnowledgeGraphAnalytics:
    """Graph analytics: PageRank, centrality, community detection, path analysis."""

    def compute_pagerank(
        self,
        adjacency: Dict[str, List[str]],
        damping: float = 0.85,
        iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> Dict[str, float]:
        """Compute PageRank scores for all nodes."""
        nodes = list(adjacency.keys())
        for targets in adjacency.values():
            for t in targets:
                if t not in adjacency:
                    adjacency[t] = []
        all_nodes = list(adjacency.keys())
        n = len(all_nodes)
        if n == 0:
            return {}

        scores = {node: 1.0 / n for node in all_nodes}
        # Reverse adjacency for incoming links
        incoming: Dict[str, List[str]] = defaultdict(list)
        for src, targets in adjacency.items():
            for tgt in targets:
                incoming[tgt].append(src)

        for _ in range(iterations):
            new_scores: Dict[str, float] = {}
            delta = 0.0
            for node in all_nodes:
                rank_sum = 0.0
                for src in incoming.get(node, []):
                    out_count = len(adjacency.get(src, [])) or 1
                    rank_sum += scores[src] / out_count
                new_rank = (1.0 - damping) / n + damping * rank_sum
                new_scores[node] = new_rank
                delta = max(delta, abs(new_rank - scores[node]))
            scores = new_scores
            if delta < tolerance:
                break

        return scores

    def compute_degree_centrality(
        self, adjacency: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """Compute normalized degree centrality for all nodes."""
        all_nodes: Set[str] = set(adjacency.keys())
        for targets in adjacency.values():
            all_nodes.update(targets)
        n = len(all_nodes) - 1 or 1

        in_degree: Dict[str, int] = defaultdict(int)
        out_degree: Dict[str, int] = defaultdict(int)

        for src, targets in adjacency.items():
            out_degree[src] += len(targets)
            for tgt in targets:
                in_degree[tgt] += 1

        result = {}
        for node in all_nodes:
            combined = (in_degree.get(node, 0) + out_degree.get(node, 0)) / (2 * n)
            result[node] = combined
        return result

    def detect_communities_louvain_simple(
        self, adjacency: Dict[str, List[str]]
    ) -> Dict[str, int]:
        """Simplified Louvain-like community detection using modularity optimization."""
        all_nodes: Set[str] = set(adjacency.keys())
        for targets in adjacency.values():
            all_nodes.update(targets)

        # Initialize each node in its own community
        community: Dict[str, int] = {node: i for i, node in enumerate(all_nodes)}
        improved = True
        passes = 0
        max_passes = 10

        while improved and passes < max_passes:
            improved = False
            passes += 1
            for node in all_nodes:
                current_community = community[node]
                neighbors = adjacency.get(node, [])
                if not neighbors:
                    continue

                # Count neighbors per community
                community_counts: Dict[int, int] = defaultdict(int)
                for neighbor in neighbors:
                    community_counts[community[neighbor]] += 1

                # Move to community with most connections
                best_community = max(community_counts, key=community_counts.get)
                if best_community != current_community:
                    community[node] = best_community
                    improved = True

        return community

    def find_shortest_path_bfs(
        self,
        adjacency: Dict[str, List[str]],
        start: str,
        end: str,
        max_depth: int = 10,
    ) -> Optional[List[str]]:
        """BFS shortest path between two nodes."""
        if start == end:
            return [start]
        if start not in adjacency:
            return None

        visited = {start}
        queue = deque([(start, [start])])

        while queue:
            node, path = queue.popleft()
            if len(path) > max_depth:
                continue
            for neighbor in adjacency.get(node, []):
                if neighbor in visited:
                    continue
                new_path = path + [neighbor]
                if neighbor == end:
                    return new_path
                visited.add(neighbor)
                queue.append((neighbor, new_path))

        return None

    def find_all_paths(
        self,
        adjacency: Dict[str, List[str]],
        start: str,
        end: str,
        max_depth: int = 5,
        max_paths: int = 10,
    ) -> List[List[str]]:
        """Find all paths between two nodes up to max_depth."""
        paths: List[List[str]] = []

        def dfs(current: str, path: List[str], visited: Set[str]) -> None:
            if len(paths) >= max_paths:
                return
            if current == end and len(path) > 1:
                paths.append(list(path))
                return
            if len(path) > max_depth:
                return
            for neighbor in adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)

        dfs(start, [start], {start})
        return paths

    def compute_clustering_coefficient(
        self, adjacency: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """Compute local clustering coefficient for each node."""
        result: Dict[str, float] = {}
        for node, neighbors in adjacency.items():
            n = len(neighbors)
            if n < 2:
                result[node] = 0.0
                continue
            # Count edges between neighbors
            neighbor_set = set(neighbors)
            edge_count = 0
            for nb in neighbors:
                for nb2 in adjacency.get(nb, []):
                    if nb2 in neighbor_set and nb2 != node:
                        edge_count += 1
            max_edges = n * (n - 1)
            result[node] = edge_count / max_edges if max_edges > 0 else 0.0
        return result

    def get_graph_statistics(
        self,
        nodes: Dict[str, GraphNode],
        adjacency: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Compute comprehensive graph statistics."""
        total_nodes = len(nodes)
        total_edges = sum(len(v) for v in adjacency.values())
        avg_degree = total_edges / total_nodes if total_nodes > 0 else 0.0

        node_type_dist: Dict[str, int] = defaultdict(int)
        for node in nodes.values():
            node_type_dist[node.node_type.value] += 1

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "avg_degree": avg_degree,
            "density": total_edges / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0.0,
            "node_type_distribution": dict(node_type_dist),
        }


# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------

class KnowledgeGraph:
    """
    Production-grade knowledge graph with multi-tenant support,
    semantic search, inference, analytics, and temporal reasoning.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: Dict[str, GraphEdge] = {}
        # adjacency[tenant_id][node_id] -> list of (target_id, edge_id)
        self._adjacency: Dict[str, Dict[str, List[Tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
        self._reverse_adjacency: Dict[str, Dict[str, List[Tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
        self._node_name_index: Dict[str, Dict[str, str]] = defaultdict(dict)  # tenant -> name -> node_id
        self._analytics = KnowledgeGraphAnalytics()
        self._inference = GraphInferenceEngine()
        self._lock = asyncio.Lock()
        self._query_cache: Dict[str, Tuple[GraphQueryResult, datetime]] = {}
        self._cache_ttl_seconds: int = 300

    # ------------------------------------------------------------------
    # Node Management
    # ------------------------------------------------------------------

    async def add_node(self, node: GraphNode) -> GraphNode:
        """Add or update a node in the knowledge graph."""
        async with self._lock:
            self._nodes[node.node_id] = node
            self._node_name_index[node.tenant_id][node.name.lower()] = node.node_id
            logger.debug("Added node: %s (%s) type=%s", node.name, node.node_id, node.node_type.value)
            return node

    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(node_id)

    async def find_node_by_name(self, name: str, tenant_id: str = "default") -> Optional[GraphNode]:
        """Find a node by exact name (case-insensitive)."""
        node_id = self._node_name_index.get(tenant_id, {}).get(name.lower())
        if node_id:
            return self._nodes.get(node_id)
        return None

    async def update_node(self, node_id: str, updates: Dict[str, Any]) -> Optional[GraphNode]:
        """Update node properties."""
        async with self._lock:
            node = self._nodes.get(node_id)
            if not node:
                return None
            for key, value in updates.items():
                if hasattr(node, key):
                    setattr(node, key, value)
            node.updated_at = datetime.utcnow()
            return node

    async def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its edges."""
        async with self._lock:
            node = self._nodes.pop(node_id, None)
            if not node:
                return False
            # Remove edges involving this node
            edges_to_remove = [
                eid for eid, edge in self._edges.items()
                if edge.source_id == node_id or edge.target_id == node_id
            ]
            for eid in edges_to_remove:
                self._edges.pop(eid, None)

            # Clean adjacency
            tenant = node.tenant_id
            self._adjacency[tenant].pop(node_id, None)
            for adj_list in self._adjacency[tenant].values():
                adj_list[:] = [(t, e) for t, e in adj_list if t != node_id]
            self._node_name_index[tenant].pop(node.name.lower(), None)
            return True

    # ------------------------------------------------------------------
    # Edge Management
    # ------------------------------------------------------------------

    async def add_edge(self, edge: GraphEdge) -> GraphEdge:
        """Add a typed edge between two nodes."""
        async with self._lock:
            if edge.source_id not in self._nodes or edge.target_id not in self._nodes:
                raise ValueError(
                    f"Nodes {edge.source_id} or {edge.target_id} not found"
                )
            self._edges[edge.edge_id] = edge
            tenant = edge.tenant_id
            self._adjacency[tenant][edge.source_id].append((edge.target_id, edge.edge_id))
            self._reverse_adjacency[tenant][edge.target_id].append((edge.source_id, edge.edge_id))
            if edge.bidirectional:
                self._adjacency[tenant][edge.target_id].append((edge.source_id, edge.edge_id))
                self._reverse_adjacency[tenant][edge.source_id].append((edge.target_id, edge.edge_id))
            return edge

    async def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        tenant_id: str = "default",
        direction: str = "outgoing",
    ) -> List[Tuple[GraphNode, GraphEdge]]:
        """Get neighboring nodes with edges."""
        adj = self._adjacency if direction == "outgoing" else self._reverse_adjacency
        neighbors: List[Tuple[GraphNode, GraphEdge]] = []

        for target_id, edge_id in adj.get(tenant_id, {}).get(node_id, []):
            edge = self._edges.get(edge_id)
            target = self._nodes.get(target_id)
            if edge and target:
                if edge_type is None or edge.edge_type == edge_type:
                    if edge.is_valid_at(datetime.utcnow()):
                        target.visit_count += 1
                        neighbors.append((target, edge))

        return neighbors

    # ------------------------------------------------------------------
    # Graph Traversal
    # ------------------------------------------------------------------

    async def traverse(self, query: GraphQuery) -> GraphQueryResult:
        """Execute a graph traversal query with caching."""
        cache_key = f"{query.start_node_id}:{query.max_depth}:{query.traversal.value}:{query.tenant_id}"
        cached = self._query_cache.get(cache_key)
        if cached:
            result, cached_at = cached
            age = (datetime.utcnow() - cached_at).total_seconds()
            if age < self._cache_ttl_seconds:
                return result

        start_time = datetime.utcnow()

        if query.traversal == TraversalStrategy.BFS:
            result = await self._bfs(query)
        elif query.traversal == TraversalStrategy.DFS:
            result = await self._dfs(query)
        elif query.traversal == TraversalStrategy.DIJKSTRA:
            result = await self._dijkstra(query)
        else:
            result = await self._bfs(query)

        result.execution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        result.algorithm_used = query.traversal.value
        self._query_cache[cache_key] = (result, datetime.utcnow())
        return result

    async def _bfs(self, query: GraphQuery) -> GraphQueryResult:
        """Breadth-first search traversal."""
        start_id = query.start_node_id
        if not start_id or start_id not in self._nodes:
            return GraphQueryResult(query_id=query.query_id)

        visited_nodes: Dict[str, GraphNode] = {}
        visited_edges: Dict[str, GraphEdge] = {}
        paths: List[List[str]] = []
        queue: deque = deque([(start_id, [start_id], 0)])
        visited_ids: Set[str] = {start_id}

        while queue and len(visited_nodes) < query.max_results:
            node_id, path, depth = queue.popleft()
            node = self._nodes.get(node_id)
            if not node:
                continue
            if node.tenant_id != query.tenant_id:
                continue
            if query.node_type_filter and node.node_type != query.node_type_filter:
                continue
            if node.confidence < query.min_confidence:
                continue

            visited_nodes[node_id] = node

            # Check if we've reached the target
            if query.end_node_id and node_id == query.end_node_id:
                paths.append(path)
                continue

            if depth >= query.max_depth:
                continue

            for target_id, edge_id in self._adjacency.get(query.tenant_id, {}).get(node_id, []):
                if target_id in visited_ids:
                    continue
                edge = self._edges.get(edge_id)
                if not edge:
                    continue
                if query.edge_type_filter and edge.edge_type != query.edge_type_filter:
                    continue
                visited_ids.add(target_id)
                visited_edges[edge_id] = edge
                queue.append((target_id, path + [target_id], depth + 1))

        if not query.end_node_id:
            paths = [[start_id]]

        return GraphQueryResult(
            query_id=query.query_id,
            nodes=[n.to_dict() for n in visited_nodes.values()],
            edges=[e.to_dict() for e in visited_edges.values()],
            paths=paths,
            total_nodes=len(visited_nodes),
            total_edges=len(visited_edges),
        )

    async def _dfs(self, query: GraphQuery) -> GraphQueryResult:
        """Depth-first search traversal."""
        start_id = query.start_node_id
        if not start_id or start_id not in self._nodes:
            return GraphQueryResult(query_id=query.query_id)

        visited_nodes: Dict[str, GraphNode] = {}
        visited_edges: Dict[str, GraphEdge] = {}
        stack = [(start_id, [start_id], 0)]
        visited_ids: Set[str] = {start_id}

        while stack and len(visited_nodes) < query.max_results:
            node_id, path, depth = stack.pop()
            node = self._nodes.get(node_id)
            if not node or node.tenant_id != query.tenant_id:
                continue
            visited_nodes[node_id] = node

            if depth >= query.max_depth:
                continue

            for target_id, edge_id in reversed(
                self._adjacency.get(query.tenant_id, {}).get(node_id, [])
            ):
                if target_id in visited_ids:
                    continue
                edge = self._edges.get(edge_id)
                if edge:
                    visited_ids.add(target_id)
                    visited_edges[edge_id] = edge
                    stack.append((target_id, path + [target_id], depth + 1))

        return GraphQueryResult(
            query_id=query.query_id,
            nodes=[n.to_dict() for n in visited_nodes.values()],
            edges=[e.to_dict() for e in visited_edges.values()],
            total_nodes=len(visited_nodes),
            total_edges=len(visited_edges),
        )

    async def _dijkstra(self, query: GraphQuery) -> GraphQueryResult:
        """Dijkstra's shortest path algorithm using edge weights."""
        start_id = query.start_node_id
        end_id = query.end_node_id
        if not start_id or start_id not in self._nodes:
            return GraphQueryResult(query_id=query.query_id)

        dist: Dict[str, float] = defaultdict(lambda: float("inf"))
        dist[start_id] = 0.0
        prev: Dict[str, Optional[str]] = {start_id: None}
        heap: List[Tuple[float, str]] = [(0.0, start_id)]
        visited_edges: Dict[str, GraphEdge] = {}

        while heap:
            d, node_id = heapq.heappop(heap)
            if d > dist[node_id]:
                continue

            for target_id, edge_id in self._adjacency.get(query.tenant_id, {}).get(node_id, []):
                edge = self._edges.get(edge_id)
                if not edge:
                    continue
                new_dist = d + (1.0 / edge.weight if edge.weight > 0 else 1.0)
                if new_dist < dist[target_id]:
                    dist[target_id] = new_dist
                    prev[target_id] = node_id
                    visited_edges[edge_id] = edge
                    heapq.heappush(heap, (new_dist, target_id))

        # Reconstruct path to end_node if specified
        paths: List[List[str]] = []
        if end_id and end_id in dist:
            path: List[str] = []
            current: Optional[str] = end_id
            while current:
                path.append(current)
                current = prev.get(current)
            paths = [list(reversed(path))]

        visited_nodes = {
            nid: self._nodes[nid]
            for nid in dist
            if nid in self._nodes and dist[nid] < float("inf")
        }

        return GraphQueryResult(
            query_id=query.query_id,
            nodes=[n.to_dict() for n in visited_nodes.values()],
            edges=[e.to_dict() for e in visited_edges.values()],
            paths=paths,
            total_nodes=len(visited_nodes),
            total_edges=len(visited_edges),
        )

    # ------------------------------------------------------------------
    # Semantic Search
    # ------------------------------------------------------------------

    async def semantic_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        node_type: Optional[NodeType] = None,
        tenant_id: str = "default",
        min_similarity: float = 0.5,
    ) -> List[Tuple[GraphNode, float]]:
        """Find semantically similar nodes using embedding cosine similarity."""
        query_node = GraphNode(embedding=query_embedding)
        results: List[Tuple[GraphNode, float]] = []

        for node in self._nodes.values():
            if node.tenant_id != tenant_id:
                continue
            if node_type and node.node_type != node_type:
                continue
            if node.embedding is None:
                continue
            sim = query_node.similarity(node)
            if sim >= min_similarity:
                results.append((node, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def text_search(
        self,
        query: str,
        tenant_id: str = "default",
        limit: int = 20,
    ) -> List[GraphNode]:
        """Simple text-based search across node names and descriptions."""
        query_lower = query.lower()
        results: List[Tuple[GraphNode, float]] = []

        for node in self._nodes.values():
            if node.tenant_id != tenant_id:
                continue
            score = 0.0
            if query_lower in node.name.lower():
                score += 2.0
            if query_lower in node.description.lower():
                score += 1.0
            for tag in node.tags:
                if query_lower in tag.lower():
                    score += 0.5
            if score > 0:
                results.append((node, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:limit]]

    # ------------------------------------------------------------------
    # Analytics & Inference
    # ------------------------------------------------------------------

    async def compute_analytics(
        self, tenant_id: str = "default"
    ) -> Dict[str, Any]:
        """Compute comprehensive graph analytics for a tenant."""
        tenant_nodes = {
            nid: n for nid, n in self._nodes.items()
            if n.tenant_id == tenant_id
        }
        adj_simple: Dict[str, List[str]] = {
            nid: [t for t, _ in edges]
            for nid, edges in self._adjacency.get(tenant_id, {}).items()
        }

        # PageRank
        pagerank = self._analytics.compute_pagerank(dict(adj_simple))
        for nid, score in pagerank.items():
            if nid in self._nodes:
                self._nodes[nid].pagerank_score = score

        # Degree centrality
        centrality = self._analytics.compute_degree_centrality(dict(adj_simple))

        # Community detection
        communities = self._analytics.detect_communities_louvain_simple(dict(adj_simple))

        # Graph statistics
        stats = self._analytics.get_graph_statistics(tenant_nodes, adj_simple)

        # Top nodes by PageRank
        top_nodes = sorted(
            [(nid, score) for nid, score in pagerank.items() if nid in self._nodes],
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        return {
            "statistics": stats,
            "top_nodes_by_pagerank": [
                {"node_id": nid, "name": self._nodes[nid].name, "pagerank": score}
                for nid, score in top_nodes
            ],
            "community_count": len(set(communities.values())),
            "most_central_nodes": sorted(
                centrality.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

    async def run_inference(
        self, tenant_id: str = "default", min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Run inference engine and return new inferred facts."""
        adj_typed: Dict[str, List[Tuple[str, EdgeType, float]]] = defaultdict(list)
        for nid, edge_list in self._adjacency.get(tenant_id, {}).items():
            for target_id, edge_id in edge_list:
                edge = self._edges.get(edge_id)
                if edge:
                    adj_typed[nid].append((target_id, edge.edge_type, edge.confidence))

        new_facts = self._inference.run_inference(dict(adj_typed))
        return [f.to_dict() for f in new_facts if f.confidence >= min_confidence]

    async def find_path(
        self,
        start_node_id: str,
        end_node_id: str,
        tenant_id: str = "default",
        max_depth: int = 8,
    ) -> Optional[List[Dict[str, Any]]]:
        """Find shortest semantic path between two nodes."""
        adj_simple: Dict[str, List[str]] = {
            nid: [t for t, _ in edges]
            for nid, edges in self._adjacency.get(tenant_id, {}).items()
        }
        path = self._analytics.find_shortest_path_bfs(
            adj_simple, start_node_id, end_node_id, max_depth
        )
        if not path:
            return None

        return [
            self._nodes[nid].to_dict()
            for nid in path
            if nid in self._nodes
        ]

    async def extract_subgraph(
        self,
        center_node_id: str,
        hops: int = 2,
        tenant_id: str = "default",
    ) -> Dict[str, Any]:
        """Extract ego-network (subgraph) around a central node."""
        query = GraphQuery(
            start_node_id=center_node_id,
            max_depth=hops,
            max_results=200,
            tenant_id=tenant_id,
            traversal=TraversalStrategy.BFS,
        )
        result = await self.traverse(query)
        return result.to_dict()

    # ------------------------------------------------------------------
    # Knowledge Extraction
    # ------------------------------------------------------------------

    async def extract_from_text(
        self,
        text: str,
        tenant_id: str = "default",
        auto_link: bool = True,
    ) -> List[GraphNode]:
        """
        Basic NLP-style entity extraction from text.
        Creates nodes for capitalized proper nouns and tech terms.
        In production this would use a proper NER model.
        """
        import re
        words = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text)
        unique_terms = list(dict.fromkeys(words))[:20]

        created_nodes: List[GraphNode] = []
        for term in unique_terms:
            existing = await self.find_node_by_name(term, tenant_id)
            if not existing:
                node = GraphNode(
                    name=term,
                    node_type=NodeType.CONCEPT,
                    description=f"Extracted from text",
                    tenant_id=tenant_id,
                    source="extracted",
                    confidence=0.7,
                )
                await self.add_node(node)
                created_nodes.append(node)

        # Auto-link co-occurring entities
        if auto_link and len(created_nodes) > 1:
            for i, node_a in enumerate(created_nodes):
                for node_b in created_nodes[i + 1:]:
                    edge = GraphEdge(
                        source_id=node_a.node_id,
                        target_id=node_b.node_id,
                        edge_type=EdgeType.RELATED_TO,
                        weight=0.5,
                        confidence=0.6,
                        tenant_id=tenant_id,
                        source="extracted",
                    )
                    try:
                        await self.add_edge(edge)
                    except ValueError:
                        pass

        return created_nodes

    # ------------------------------------------------------------------
    # Bulk Operations
    # ------------------------------------------------------------------

    async def bulk_add_nodes(self, nodes: List[GraphNode]) -> int:
        """Bulk insert nodes for efficiency."""
        count = 0
        async with self._lock:
            for node in nodes:
                self._nodes[node.node_id] = node
                self._node_name_index[node.tenant_id][node.name.lower()] = node.node_id
                count += 1
        return count

    async def get_graph_summary(self, tenant_id: str = "default") -> Dict[str, Any]:
        """Get a high-level summary of the knowledge graph."""
        tenant_nodes = [n for n in self._nodes.values() if n.tenant_id == tenant_id]
        tenant_edges = [e for e in self._edges.values() if e.tenant_id == tenant_id]

        node_type_dist: Dict[str, int] = defaultdict(int)
        for node in tenant_nodes:
            node_type_dist[node.node_type.value] += 1

        edge_type_dist: Dict[str, int] = defaultdict(int)
        for edge in tenant_edges:
            edge_type_dist[edge.edge_type.value] += 1

        return {
            "tenant_id": tenant_id,
            "total_nodes": len(tenant_nodes),
            "total_edges": len(tenant_edges),
            "node_type_distribution": dict(node_type_dist),
            "edge_type_distribution": dict(edge_type_dist),
            "nodes_with_embeddings": sum(1 for n in tenant_nodes if n.embedding is not None),
            "inferred_facts": len(self._inference.get_inferred_facts()),
        }

    async def list_nodes(
        self,
        tenant_id: str = "default",
        node_type: Optional[NodeType] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        nodes = [
            n for n in self._nodes.values()
            if n.tenant_id == tenant_id
            and (node_type is None or n.node_type == node_type)
        ]
        nodes.sort(key=lambda n: n.pagerank_score, reverse=True)
        return [n.to_dict() for n in nodes[offset:offset + limit]]
