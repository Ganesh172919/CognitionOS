"""
Memory Hierarchy Domain - Services

Domain services for memory hierarchy orchestration and business logic.
NO external dependencies except Python stdlib.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID, uuid4
import math

from .entities import (
    WorkingMemory,
    EpisodicMemory,
    LongTermMemory,
    MemoryTier,
    MemoryEmbedding,
    KnowledgeType,
    SourceType,
)
from .repositories import (
    WorkingMemoryRepository,
    EpisodicMemoryRepository,
    LongTermMemoryRepository,
)
from .events import (
    L1ToL2Promotion,
    L2ToL3Promotion,
    MemoryEvicted,
    MemoryCompressed,
    MemoryClusterCreated,
)


class MemoryTierManager:
    """
    Domain service for managing memory tier transitions.
    
    Handles promotion, demotion, and eviction across L1/L2/L3 tiers.
    """

    def __init__(
        self,
        l1_repository: WorkingMemoryRepository,
        l2_repository: EpisodicMemoryRepository,
        l3_repository: LongTermMemoryRepository,
    ):
        """
        Initialize memory tier manager.
        
        Args:
            l1_repository: Working memory repository
            l2_repository: Episodic memory repository
            l3_repository: Long-term memory repository
        """
        self.l1_repository = l1_repository
        self.l2_repository = l2_repository
        self.l3_repository = l3_repository

    async def promote_l1_to_l2(
        self,
        source_memories: List[WorkingMemory],
        cluster_id: str,
        summary: str,
        embedding: Optional[MemoryEmbedding] = None,
    ) -> Tuple[EpisodicMemory, L1ToL2Promotion]:
        """
        Promote L1 working memories to L2 episodic memory.
        
        Compresses multiple related working memories into a single
        episodic memory with a summary.
        
        Args:
            source_memories: List of working memories to compress
            cluster_id: Cluster identifier
            summary: Compressed summary
            embedding: Optional embedding for the summary
            
        Returns:
            Tuple of (created episodic memory, promotion event)
        """
        if not source_memories:
            raise ValueError("Cannot promote empty memory list")

        # Calculate temporal period
        timestamps = [m.created_at for m in source_memories]
        temporal_start = min(timestamps)
        temporal_end = max(timestamps)

        # Calculate average importance
        avg_importance = sum(m.importance_score for m in source_memories) / len(source_memories)

        # Create episodic memory
        agent_id = source_memories[0].agent_id
        source_ids = [m.id for m in source_memories]

        episodic = EpisodicMemory.create(
            agent_id=agent_id,
            cluster_id=cluster_id,
            summary=summary,
            source_memory_ids=source_ids,
            temporal_start=temporal_start,
            temporal_end=temporal_end,
            importance_score=avg_importance,
            embedding=embedding,
        )

        # Calculate compression ratio
        original_length = sum(len(m.content) for m in source_memories)
        episodic.calculate_compression_ratio(original_length)

        # Save episodic memory
        await self.l2_repository.save(episodic)

        # Create promotion event
        event = L1ToL2Promotion(
            event_id=uuid4(),
            source_memory_ids=source_ids,
            episodic_memory_id=episodic.id,
            agent_id=agent_id,
            cluster_id=cluster_id,
            compression_ratio=episodic.compression_ratio,
        )

        return episodic, event

    async def promote_l2_to_l3(
        self,
        episodic_memory: EpisodicMemory,
        knowledge_type: KnowledgeType,
        title: str,
        content: Optional[str] = None,
        embedding: Optional[MemoryEmbedding] = None,
    ) -> Tuple[LongTermMemory, L2ToL3Promotion]:
        """
        Promote L2 episodic memory to L3 long-term memory.
        
        Extracts durable knowledge from episodic memory.
        
        Args:
            episodic_memory: Episodic memory to promote
            knowledge_type: Type of knowledge being extracted
            title: Title for the long-term memory
            content: Optional content (defaults to episodic summary)
            embedding: Optional embedding
            
        Returns:
            Tuple of (created long-term memory, promotion event)
        """
        # Use episodic summary as content if not provided
        final_content = content or episodic_memory.summary

        # Create long-term memory
        longterm = LongTermMemory.create(
            agent_id=episodic_memory.agent_id,
            knowledge_type=knowledge_type,
            title=title,
            content=final_content,
            source_type=SourceType.EPISODIC_COMPRESSION,
            importance_score=episodic_memory.importance_score,
            embedding=embedding or episodic_memory.embedding,
            source_references=[episodic_memory.id],
        )

        # Save long-term memory
        await self.l3_repository.save(longterm)

        # Create promotion event
        event = L2ToL3Promotion(
            event_id=uuid4(),
            episodic_memory_id=episodic_memory.id,
            longterm_memory_id=longterm.id,
            agent_id=episodic_memory.agent_id,
            source_type=SourceType.EPISODIC_COMPRESSION,
            importance_score=longterm.importance_score,
        )

        return longterm, event

    async def demote_l2_to_l1(
        self,
        episodic_memory: EpisodicMemory,
    ) -> List[WorkingMemory]:
        """
        Demote L2 episodic memory back to L1 working memories.
        
        Expands compressed episodic memory if source memories are still available.
        This is a recovery mechanism if episodic compression was premature.
        
        Args:
            episodic_memory: Episodic memory to demote
            
        Returns:
            List of recovered working memories
        """
        # In practice, this would retrieve original working memories
        # if they haven't been deleted. For now, return empty list.
        # Implementation would query l1_repository for source_memory_ids
        return []

    async def evict_from_l1(
        self,
        agent_id: UUID,
        workflow_execution_id: UUID,
        max_count: int = 10,
    ) -> Tuple[List[UUID], List[MemoryEvicted]]:
        """
        Evict least recently used memories from L1.
        
        Uses LRU (Least Recently Used) strategy to free up working memory.
        
        Args:
            agent_id: Agent ID
            workflow_execution_id: Workflow execution ID
            max_count: Maximum number of memories to evict
            
        Returns:
            Tuple of (evicted memory IDs, eviction events)
        """
        # Get LRU candidates
        candidates = await self.l1_repository.find_lru_candidates(
            agent_id=agent_id,
            workflow_execution_id=workflow_execution_id,
            limit=max_count,
        )

        evicted_ids = []
        events = []

        for memory in candidates:
            # Delete the memory
            await self.l1_repository.delete(memory.id)
            evicted_ids.append(memory.id)

            # Create eviction event
            event = MemoryEvicted(
                event_id=uuid4(),
                memory_id=memory.id,
                agent_id=agent_id,
                tier=MemoryTier.L1_WORKING,
                reason="lru_eviction",
                access_count=memory.access_count,
                age_hours=memory.calculate_age_hours(),
            )
            events.append(event)

        return evicted_ids, events


class MemoryImportanceScorer:
    """
    Domain service for calculating memory importance scores.
    
    Analyzes access patterns, recency, and content signals to determine
    which memories should be promoted or evicted.
    """

    def __init__(
        self,
        access_weight: float = 0.3,
        recency_weight: float = 0.4,
        content_weight: float = 0.3,
    ):
        """
        Initialize importance scorer.
        
        Args:
            access_weight: Weight for access count factor
            recency_weight: Weight for recency factor
            content_weight: Weight for content signals
        """
        if not math.isclose(access_weight + recency_weight + content_weight, 1.0, abs_tol=1e-9):
            raise ValueError("Weights must sum to 1.0")

        self.access_weight = access_weight
        self.recency_weight = recency_weight
        self.content_weight = content_weight

    def calculate_importance(
        self,
        memory: WorkingMemory,
        max_access_count: int = 100,
        max_age_hours: float = 24.0,
    ) -> float:
        """
        Calculate importance score for a working memory.
        
        Combines access frequency, recency, and content signals.
        
        Args:
            memory: Working memory to score
            max_access_count: Maximum access count for normalization
            max_age_hours: Maximum age in hours for normalization
            
        Returns:
            Importance score (0-1)
        """
        # Access frequency component (normalized)
        access_score = min(memory.access_count / max_access_count, 1.0)

        # Recency component (inverse of age)
        age_hours = memory.calculate_age_hours()
        recency_score = max(0.0, 1.0 - (age_hours / max_age_hours))

        # Content component (based on memory type and existing score)
        # High-value types get bonus
        content_bonus = 0.0
        if memory.memory_type.value in ["action", "result"]:
            content_bonus = 0.2
        elif memory.memory_type.value == "reasoning":
            content_bonus = 0.3

        content_score = min(memory.importance_score + content_bonus, 1.0)

        # Weighted combination
        final_score = (
            self.access_weight * access_score +
            self.recency_weight * recency_score +
            self.content_weight * content_score
        )

        return min(final_score, 1.0)

    async def update_importance_scores(
        self,
        memories: List[WorkingMemory],
        l1_repository: WorkingMemoryRepository,
    ) -> int:
        """
        Update importance scores for a batch of memories.
        
        Args:
            memories: List of working memories
            l1_repository: Repository for saving updates
            
        Returns:
            Number of memories updated
        """
        if not memories:
            return 0

        # Calculate max values for normalization
        max_access = max((m.access_count for m in memories), default=1)
        max_age = max((m.calculate_age_hours() for m in memories), default=1.0)

        updated_count = 0
        for memory in memories:
            new_score = self.calculate_importance(
                memory=memory,
                max_access_count=max_access,
                max_age_hours=max_age,
            )

            # Update if score changed significantly
            if abs(new_score - memory.importance_score) > 0.05:
                memory.update_importance_score(new_score)
                await l1_repository.save(memory)
                updated_count += 1

        return updated_count


class MemoryCompressionService:
    """
    Domain service for memory compression operations.
    
    Handles semantic clustering and compression of memories.
    Note: LLM integration would be at the application layer.
    """

    def __init__(self, min_cluster_size: int = 3, max_cluster_size: int = 10):
        """
        Initialize compression service.
        
        Args:
            min_cluster_size: Minimum memories for a cluster
            max_cluster_size: Maximum memories in a cluster
        """
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size

    async def compress_memories(
        self,
        memories: List[WorkingMemory],
        summary: str,
        cluster_id: Optional[str] = None,
    ) -> Tuple[str, MemoryCompressed]:
        """
        Compress a group of memories into a summary.
        
        In practice, this would use an LLM to generate the summary.
        This is a placeholder that validates the compression.
        
        Args:
            memories: Memories to compress
            summary: Compressed summary text
            cluster_id: Optional cluster identifier
            
        Returns:
            Tuple of (cluster_id, compression event)
        """
        if len(memories) < self.min_cluster_size:
            raise ValueError(f"Cluster too small: {len(memories)} < {self.min_cluster_size}")

        if len(memories) > self.max_cluster_size:
            raise ValueError(f"Cluster too large: {len(memories)} > {self.max_cluster_size}")

        # Generate cluster ID if not provided
        final_cluster_id = cluster_id or f"cluster_{uuid4()}"

        # Calculate compression metrics
        original_length = sum(len(m.content) for m in memories)
        compressed_length = len(summary)
        compression_ratio = compressed_length / original_length if original_length > 0 else 0.0

        # Create compression event
        event = MemoryCompressed(
            event_id=uuid4(),
            source_memory_ids=[m.id for m in memories],
            compressed_memory_id=uuid4(),  # Placeholder
            agent_id=memories[0].agent_id,
            original_length=original_length,
            compressed_length=compressed_length,
            compression_ratio=compression_ratio,
        )

        return final_cluster_id, event

    def calculate_semantic_similarity(
        self,
        embedding1: MemoryEmbedding,
        embedding2: MemoryEmbedding,
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First memory embedding
            embedding2: Second memory embedding
            
        Returns:
            Cosine similarity (0-1)
        """
        if embedding1.dimension != embedding2.dimension:
            raise ValueError("Embeddings must have same dimension")

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(embedding1.vector, embedding2.vector))
        norm1 = math.sqrt(sum(a * a for a in embedding1.vector))
        norm2 = math.sqrt(sum(b * b for b in embedding2.vector))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return max(0.0, min(similarity, 1.0))

    async def cluster_related_memories(
        self,
        memories: List[WorkingMemory],
        similarity_threshold: float = 0.7,
    ) -> Dict[str, List[WorkingMemory]]:
        """
        Cluster related memories based on semantic similarity.
        
        Simple greedy clustering algorithm. In practice, would use
        more sophisticated clustering (e.g., HDBSCAN, OPTICS).
        
        Args:
            memories: Memories to cluster
            similarity_threshold: Minimum similarity for clustering
            
        Returns:
            Dictionary mapping cluster_id to list of memories
        """
        if not memories:
            return {}

        # Filter memories with embeddings
        embeddable = [m for m in memories if m.embedding is not None]
        if not embeddable:
            return {}

        clusters: Dict[str, List[WorkingMemory]] = {}
        assigned = set()

        for i, memory in enumerate(embeddable):
            if memory.id in assigned:
                continue

            # Start new cluster
            cluster_id = f"cluster_{uuid4()}"
            cluster = [memory]
            assigned.add(memory.id)

            # Find similar memories
            for j, other in enumerate(embeddable):
                if i == j or other.id in assigned:
                    continue

                if memory.embedding and other.embedding:
                    similarity = self.calculate_semantic_similarity(
                        memory.embedding,
                        other.embedding,
                    )

                    if similarity >= similarity_threshold:
                        cluster.append(other)
                        assigned.add(other.id)

                        # Stop if cluster is full
                        if len(cluster) >= self.max_cluster_size:
                            break

            # Only keep cluster if it meets minimum size
            if len(cluster) >= self.min_cluster_size:
                clusters[cluster_id] = cluster

        return clusters
