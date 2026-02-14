"""
Memory Hierarchy Application - Use Cases

Application layer use cases for Memory Hierarchy bounded context.
Orchestrates domain entities and coordinates with infrastructure.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from core.domain.memory_hierarchy import (
    WorkingMemory,
    EpisodicMemory,
    LongTermMemory,
    MemoryTier,
    MemoryType,
    KnowledgeType,
    MemoryEmbedding,
    WorkingMemoryRepository,
    EpisodicMemoryRepository,
    LongTermMemoryRepository,
    MemoryTierManager,
    MemoryImportanceScorer,
    MemoryCompressionService,
    L1ToL2Promotion,
    L2ToL3Promotion,
    MemoryEvicted,
    MemoryAccessed,
)


# ==================== DTOs (Data Transfer Objects) ====================

@dataclass
class StoreMemoryCommand:
    """Command to store a new memory in L1"""
    agent_id: UUID
    workflow_execution_id: UUID
    content: str
    importance_score: float = 0.5
    embedding: Optional[Dict[str, Any]] = None
    memory_type: str = "observation"
    tags: List[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate command"""
        if not 0.0 <= self.importance_score <= 1.0:
            raise ValueError("Importance score must be between 0 and 1")
        if not self.content:
            raise ValueError("Content cannot be empty")


@dataclass
class RetrieveMemoryQuery:
    """Query to retrieve memories from a tier"""
    agent_id: UUID
    tier: str = "l1_working"
    workflow_execution_id: Optional[UUID] = None
    similarity_threshold: Optional[float] = None
    min_importance: Optional[float] = None
    limit: int = 100
    tags: Optional[List[str]] = None


@dataclass
class PromoteMemoriesCommand:
    """Command to promote memories between tiers"""
    agent_id: UUID
    tier_from: str
    tier_to: str
    memory_ids: Optional[List[UUID]] = None
    min_importance: float = 0.7
    min_access_count: int = 3
    min_age_hours: float = 1.0
    cluster_id: Optional[str] = None
    summary: Optional[str] = None
    knowledge_type: Optional[str] = None
    title: Optional[str] = None

    def __post_init__(self):
        """Validate command"""
        valid_tiers = ["l1_working", "l2_episodic", "l3_longterm"]
        if self.tier_from not in valid_tiers or self.tier_to not in valid_tiers:
            raise ValueError(f"Invalid tier. Must be one of {valid_tiers}")


@dataclass
class EvictMemoriesCommand:
    """Command to evict low priority memories from L1"""
    agent_id: UUID
    workflow_execution_id: UUID
    max_count: int = 10
    strategy: str = "lru"

    def __post_init__(self):
        """Validate command"""
        if self.max_count <= 0:
            raise ValueError("Max count must be positive")
        if self.strategy not in ["lru", "low_importance"]:
            raise ValueError("Strategy must be 'lru' or 'low_importance'")


@dataclass
class MemoryStatisticsQuery:
    """Query to get memory statistics"""
    agent_id: UUID
    include_archived: bool = False


@dataclass
class SearchMemoriesQuery:
    """Query to search memories across tiers"""
    agent_id: UUID
    query_embedding: Dict[str, Any]
    top_k: int = 10
    tiers: List[str] = field(default_factory=lambda: ["l1_working", "l2_episodic", "l3_longterm"])
    similarity_threshold: float = 0.7

    def __post_init__(self):
        """Validate query"""
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold must be between 0 and 1")


@dataclass
class UpdateImportanceCommand:
    """Command to update importance scores"""
    agent_id: UUID
    workflow_execution_id: Optional[UUID] = None
    memory_ids: Optional[List[UUID]] = None


@dataclass
class MemoryResult:
    """Result of memory operation"""
    id: UUID
    tier: str
    content: str
    importance_score: float
    created_at: str
    memory_type: Optional[str] = None
    access_count: Optional[int] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MemoryStatisticsResult:
    """Result of memory statistics query"""
    l1_count: int
    l2_count: int
    l3_count: int
    total_size_mb: float
    l1_avg_importance: float = 0.0
    l2_avg_importance: float = 0.0
    l3_avg_importance: float = 0.0
    l1_avg_age_hours: float = 0.0


# ==================== Use Cases ====================

class StoreWorkingMemoryUseCase:
    """
    Use Case: Store a new memory in L1 working memory.

    Orchestrates:
    1. Create working memory entity
    2. Save to L1 repository
    3. Publish domain event
    """

    def __init__(
        self,
        l1_repository: WorkingMemoryRepository,
        event_publisher: Optional[Any] = None
    ):
        self.l1_repository = l1_repository
        self.event_publisher = event_publisher

    async def execute(self, command: StoreMemoryCommand) -> MemoryResult:
        """
        Store a new working memory.

        Args:
            command: Store memory command

        Returns:
            MemoryResult with memory details

        Raises:
            ValueError: If memory creation fails
        """
        # Parse embedding if provided
        embedding = None
        if command.embedding:
            embedding = MemoryEmbedding(
                vector=command.embedding.get("vector", []),
                model=command.embedding.get("model", "unknown"),
                dimension=command.embedding.get("dimension", 0),
            )

        # Parse memory type
        memory_type = MemoryType(command.memory_type)

        # Create working memory
        memory = WorkingMemory.create(
            agent_id=command.agent_id,
            workflow_execution_id=command.workflow_execution_id,
            content=command.content,
            importance_score=command.importance_score,
            embedding=embedding,
            memory_type=memory_type,
            tags=command.tags,
            expires_at=command.expires_at,
            metadata=command.metadata,
        )

        # Save to repository
        await self.l1_repository.save(memory)

        # Publish event
        if self.event_publisher:
            from uuid import uuid4
            event = MemoryAccessed(
                event_id=uuid4(),
                memory_id=memory.id,
                agent_id=memory.agent_id,
                tier=MemoryTier.L1_WORKING,
                access_count=0,
            )
            await self.event_publisher.publish(event)

        return MemoryResult(
            id=memory.id,
            tier=MemoryTier.L1_WORKING.value,
            content=memory.content,
            importance_score=memory.importance_score,
            created_at=memory.created_at.isoformat(),
            memory_type=memory.memory_type.value,
            access_count=memory.access_count,
            tags=memory.tags,
            metadata=memory.metadata,
        )


class RetrieveWorkingMemoryUseCase:
    """
    Use Case: Retrieve memories from L1 with filters.

    Orchestrates:
    1. Query L1 repository with filters
    2. Update access tracking
    3. Return results
    """

    def __init__(
        self,
        l1_repository: WorkingMemoryRepository,
        event_publisher: Optional[Any] = None
    ):
        self.l1_repository = l1_repository
        self.event_publisher = event_publisher

    async def execute(self, query: RetrieveMemoryQuery) -> List[MemoryResult]:
        """
        Retrieve working memories.

        Args:
            query: Retrieve memory query

        Returns:
            List of memory results
        """
        # Get memories based on query
        if query.min_importance is not None:
            memories = await self.l1_repository.find_by_importance(
                agent_id=query.agent_id,
                min_importance=query.min_importance,
                limit=query.limit,
            )
        else:
            memories = await self.l1_repository.find_by_agent(
                agent_id=query.agent_id,
                workflow_execution_id=query.workflow_execution_id,
                limit=query.limit,
            )

        # Update access tracking
        for memory in memories:
            memory.update_access()
            await self.l1_repository.save(memory)

        # Convert to results
        results = []
        for memory in memories:
            result = MemoryResult(
                id=memory.id,
                tier=MemoryTier.L1_WORKING.value,
                content=memory.content,
                importance_score=memory.importance_score,
                created_at=memory.created_at.isoformat(),
                memory_type=memory.memory_type.value,
                access_count=memory.access_count,
                tags=memory.tags,
                metadata=memory.metadata,
            )
            results.append(result)

        return results


class PromoteMemoriesToL2UseCase:
    """
    Use Case: Promote memories from L1 to L2.

    Orchestrates:
    1. Find candidate memories
    2. Cluster related memories
    3. Compress to episodic memory
    4. Publish promotion events
    """

    def __init__(
        self,
        tier_manager: MemoryTierManager,
        compression_service: MemoryCompressionService,
        l1_repository: WorkingMemoryRepository,
        event_publisher: Optional[Any] = None
    ):
        self.tier_manager = tier_manager
        self.compression_service = compression_service
        self.l1_repository = l1_repository
        self.event_publisher = event_publisher

    async def execute(self, command: PromoteMemoriesCommand) -> List[MemoryResult]:
        """
        Promote L1 memories to L2.

        Args:
            command: Promote memories command

        Returns:
            List of created episodic memories

        Raises:
            ValueError: If promotion fails
        """
        if command.tier_from != "l1_working" or command.tier_to != "l2_episodic":
            raise ValueError("This use case only handles L1 → L2 promotion")

        # Get candidate memories
        if command.memory_ids:
            # Specific memories
            memories = []
            for memory_id in command.memory_ids:
                memory = await self.l1_repository.find_by_id(memory_id)
                if memory:
                    memories.append(memory)
        else:
            # Find by importance
            memories = await self.l1_repository.find_by_importance(
                agent_id=command.agent_id,
                min_importance=command.min_importance,
                limit=100,
            )

            # Filter by promotion criteria
            memories = [
                m for m in memories
                if m.should_promote_to_l2(
                    min_importance=command.min_importance,
                    min_access_count=command.min_access_count,
                    min_age_hours=command.min_age_hours,
                )
            ]

        if not memories:
            return []

        # Cluster related memories
        clusters = await self.compression_service.cluster_related_memories(
            memories=memories,
            similarity_threshold=0.7,
        )

        results = []
        for cluster_id, cluster_memories in clusters.items():
            # Generate summary (in production, use LLM)
            summary = command.summary or f"Summary of {len(cluster_memories)} memories"

            # Promote to L2
            episodic, event = await self.tier_manager.promote_l1_to_l2(
                source_memories=cluster_memories,
                cluster_id=cluster_id,
                summary=summary,
            )

            # Publish event
            if self.event_publisher:
                await self.event_publisher.publish(event)

            # Delete source memories from L1
            source_ids = [m.id for m in cluster_memories]
            await self.l1_repository.delete_batch(source_ids)

            results.append(MemoryResult(
                id=episodic.id,
                tier=MemoryTier.L2_EPISODIC.value,
                content=episodic.summary,
                importance_score=episodic.importance_score,
                created_at=episodic.created_at.isoformat(),
                metadata={"cluster_id": cluster_id, "source_count": len(cluster_memories)},
            ))

        return results


class PromoteMemoriesToL3UseCase:
    """
    Use Case: Promote memories from L2 to L3.

    Orchestrates:
    1. Find candidate episodic memories
    2. Extract knowledge
    3. Create long-term memory
    4. Publish promotion events
    """

    def __init__(
        self,
        tier_manager: MemoryTierManager,
        l2_repository: EpisodicMemoryRepository,
        event_publisher: Optional[Any] = None
    ):
        self.tier_manager = tier_manager
        self.l2_repository = l2_repository
        self.event_publisher = event_publisher

    async def execute(self, command: PromoteMemoriesCommand) -> List[MemoryResult]:
        """
        Promote L2 memories to L3.

        Args:
            command: Promote memories command

        Returns:
            List of created long-term memories

        Raises:
            ValueError: If promotion fails
        """
        if command.tier_from != "l2_episodic" or command.tier_to != "l3_longterm":
            raise ValueError("This use case only handles L2 → L3 promotion")

        if not command.knowledge_type or not command.title:
            raise ValueError("knowledge_type and title required for L3 promotion")

        # Get candidate memories
        if command.memory_ids:
            memories = []
            for memory_id in command.memory_ids:
                memory = await self.l2_repository.find_by_id(memory_id)
                if memory:
                    memories.append(memory)
        else:
            memories = await self.l2_repository.find_by_importance(
                agent_id=command.agent_id,
                min_importance=command.min_importance,
                limit=100,
            )

        if not memories:
            return []

        results = []
        knowledge_type = KnowledgeType(command.knowledge_type)

        for episodic in memories:
            # Promote to L3
            longterm, event = await self.tier_manager.promote_l2_to_l3(
                episodic_memory=episodic,
                knowledge_type=knowledge_type,
                title=command.title,
                content=episodic.summary,
            )

            # Publish event
            if self.event_publisher:
                await self.event_publisher.publish(event)

            results.append(MemoryResult(
                id=longterm.id,
                tier=MemoryTier.L3_LONGTERM.value,
                content=longterm.content,
                importance_score=longterm.importance_score,
                created_at=longterm.created_at.isoformat(),
                metadata={"knowledge_type": longterm.knowledge_type.value, "title": longterm.title},
            ))

        return results


class EvictLowPriorityMemoriesUseCase:
    """
    Use Case: Evict low priority memories from L1.

    Orchestrates:
    1. Find eviction candidates using LRU
    2. Delete memories
    3. Publish eviction events
    """

    def __init__(
        self,
        tier_manager: MemoryTierManager,
        event_publisher: Optional[Any] = None
    ):
        self.tier_manager = tier_manager
        self.event_publisher = event_publisher

    async def execute(self, command: EvictMemoriesCommand) -> int:
        """
        Evict low priority memories.

        Args:
            command: Evict memories command

        Returns:
            Number of memories evicted

        Raises:
            ValueError: If eviction fails
        """
        # Evict using LRU strategy
        evicted_ids, events = await self.tier_manager.evict_from_l1(
            agent_id=command.agent_id,
            workflow_execution_id=command.workflow_execution_id,
            max_count=command.max_count,
        )

        # Publish events
        if self.event_publisher:
            for event in events:
                await self.event_publisher.publish(event)

        return len(evicted_ids)


class GetMemoryStatisticsUseCase:
    """
    Use Case: Get memory tier statistics.

    Orchestrates:
    1. Query counts from each tier
    2. Calculate averages
    3. Return statistics
    """

    def __init__(
        self,
        l1_repository: WorkingMemoryRepository,
        l2_repository: EpisodicMemoryRepository,
        l3_repository: LongTermMemoryRepository,
    ):
        self.l1_repository = l1_repository
        self.l2_repository = l2_repository
        self.l3_repository = l3_repository

    async def execute(self, query: MemoryStatisticsQuery) -> MemoryStatisticsResult:
        """
        Get memory statistics.

        Args:
            query: Memory statistics query

        Returns:
            Memory statistics result
        """
        # Get counts
        l1_count = await self.l1_repository.count_by_agent(query.agent_id)
        l2_count = await self.l2_repository.count_by_agent(query.agent_id)
        l3_count = await self.l3_repository.count_by_agent(
            query.agent_id,
            include_archived=query.include_archived,
        )

        # Get memories for averages
        l1_memories = await self.l1_repository.find_by_agent(
            agent_id=query.agent_id,
            limit=1000,
        )
        l2_memories = await self.l2_repository.find_by_agent(
            agent_id=query.agent_id,
            limit=1000,
        )
        l3_memories = await self.l3_repository.find_by_agent(
            agent_id=query.agent_id,
            include_archived=query.include_archived,
            limit=1000,
        )

        # Calculate averages
        l1_avg_importance = 0.0
        l1_avg_age_hours = 0.0
        if l1_memories:
            l1_avg_importance = sum(m.importance_score for m in l1_memories) / len(l1_memories)
            l1_avg_age_hours = sum(m.calculate_age_hours() for m in l1_memories) / len(l1_memories)

        l2_avg_importance = 0.0
        if l2_memories:
            l2_avg_importance = sum(m.importance_score for m in l2_memories) / len(l2_memories)

        l3_avg_importance = 0.0
        if l3_memories:
            l3_avg_importance = sum(m.importance_score for m in l3_memories) / len(l3_memories)

        # Calculate total size (rough estimate)
        total_size_mb = (
            sum(len(m.content) for m in l1_memories) +
            sum(len(m.summary) for m in l2_memories) +
            sum(len(m.content) for m in l3_memories)
        ) / (1024 * 1024)

        return MemoryStatisticsResult(
            l1_count=l1_count,
            l2_count=l2_count,
            l3_count=l3_count,
            total_size_mb=total_size_mb,
            l1_avg_importance=l1_avg_importance,
            l2_avg_importance=l2_avg_importance,
            l3_avg_importance=l3_avg_importance,
            l1_avg_age_hours=l1_avg_age_hours,
        )


class SearchMemoriesAcrossTiersUseCase:
    """
    Use Case: Semantic search across memory tiers.

    Orchestrates:
    1. Search each tier using embeddings
    2. Calculate similarity scores
    3. Merge and rank results
    """

    def __init__(
        self,
        l1_repository: WorkingMemoryRepository,
        l2_repository: EpisodicMemoryRepository,
        l3_repository: LongTermMemoryRepository,
        compression_service: MemoryCompressionService,
    ):
        self.l1_repository = l1_repository
        self.l2_repository = l2_repository
        self.l3_repository = l3_repository
        self.compression_service = compression_service

    async def execute(self, query: SearchMemoriesQuery) -> List[MemoryResult]:
        """
        Search memories across tiers.

        Args:
            query: Search memories query

        Returns:
            List of matching memories ranked by similarity

        Note:
            This is a simplified implementation. Production would use
            vector database for efficient similarity search.
        """
        # Parse query embedding
        query_embedding = MemoryEmbedding(
            vector=query.query_embedding.get("vector", []),
            model=query.query_embedding.get("model", "unknown"),
            dimension=query.query_embedding.get("dimension", 0),
        )

        results = []

        # Search L1
        if "l1_working" in query.tiers:
            l1_memories = await self.l1_repository.find_by_agent(
                agent_id=query.agent_id,
                limit=1000,
            )
            for memory in l1_memories:
                if memory.embedding:
                    similarity = self.compression_service.calculate_semantic_similarity(
                        query_embedding,
                        memory.embedding,
                    )
                    if similarity >= query.similarity_threshold:
                        results.append((similarity, MemoryResult(
                            id=memory.id,
                            tier=MemoryTier.L1_WORKING.value,
                            content=memory.content,
                            importance_score=memory.importance_score,
                            created_at=memory.created_at.isoformat(),
                            memory_type=memory.memory_type.value,
                            access_count=memory.access_count,
                            metadata={"similarity": similarity},
                        )))

        # Search L2
        if "l2_episodic" in query.tiers:
            l2_memories = await self.l2_repository.find_by_agent(
                agent_id=query.agent_id,
                limit=1000,
            )
            for memory in l2_memories:
                if memory.embedding:
                    similarity = self.compression_service.calculate_semantic_similarity(
                        query_embedding,
                        memory.embedding,
                    )
                    if similarity >= query.similarity_threshold:
                        results.append((similarity, MemoryResult(
                            id=memory.id,
                            tier=MemoryTier.L2_EPISODIC.value,
                            content=memory.summary,
                            importance_score=memory.importance_score,
                            created_at=memory.created_at.isoformat(),
                            metadata={"similarity": similarity, "cluster_id": memory.cluster_id},
                        )))

        # Search L3
        if "l3_longterm" in query.tiers:
            l3_memories = await self.l3_repository.find_by_agent(
                agent_id=query.agent_id,
                include_archived=False,
                limit=1000,
            )
            for memory in l3_memories:
                if memory.embedding:
                    similarity = self.compression_service.calculate_semantic_similarity(
                        query_embedding,
                        memory.embedding,
                    )
                    if similarity >= query.similarity_threshold:
                        results.append((similarity, MemoryResult(
                            id=memory.id,
                            tier=MemoryTier.L3_LONGTERM.value,
                            content=memory.content,
                            importance_score=memory.importance_score,
                            created_at=memory.created_at.isoformat(),
                            metadata={"similarity": similarity, "title": memory.title},
                        )))

        # Sort by similarity (descending) and take top_k
        results.sort(key=lambda x: x[0], reverse=True)
        return [result for _, result in results[:query.top_k]]


class UpdateMemoryImportanceUseCase:
    """
    Use Case: Recalculate importance scores for memories.

    Orchestrates:
    1. Get memories to update
    2. Recalculate importance scores
    3. Save updated scores
    """

    def __init__(
        self,
        importance_scorer: MemoryImportanceScorer,
        l1_repository: WorkingMemoryRepository,
    ):
        self.importance_scorer = importance_scorer
        self.l1_repository = l1_repository

    async def execute(self, command: UpdateImportanceCommand) -> int:
        """
        Update importance scores.

        Args:
            command: Update importance command

        Returns:
            Number of memories updated
        """
        # Get memories to update
        if command.memory_ids:
            memories = []
            for memory_id in command.memory_ids:
                memory = await self.l1_repository.find_by_id(memory_id)
                if memory:
                    memories.append(memory)
        else:
            memories = await self.l1_repository.find_by_agent(
                agent_id=command.agent_id,
                workflow_execution_id=command.workflow_execution_id,
                limit=1000,
            )

        # Update importance scores
        updated_count = await self.importance_scorer.update_importance_scores(
            memories=memories,
            l1_repository=self.l1_repository,
        )

        return updated_count
