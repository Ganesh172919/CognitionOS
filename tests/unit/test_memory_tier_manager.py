"""
Unit Tests for Memory Tier Manager Service

Tests for MemoryTierManager business logic and tier transitions.
"""

import pytest
from unittest.mock import AsyncMock
from uuid import uuid4
from datetime import datetime, timedelta

from core.domain.memory_hierarchy.entities import (
    WorkingMemory,
    EpisodicMemory,
    LongTermMemory,
    MemoryTier,
    MemoryType,
    MemoryEmbedding,
    KnowledgeType,
    SourceType,
)
from core.domain.memory_hierarchy.services import MemoryTierManager


class TestMemoryTierManager:
    """Tests for MemoryTierManager service"""
    
    @pytest.fixture
    def mock_l1_repository(self):
        """Create mock L1 repository"""
        repository = AsyncMock()
        repository.save = AsyncMock()
        repository.delete = AsyncMock()
        repository.find_lru_candidates = AsyncMock(return_value=[])
        return repository
    
    @pytest.fixture
    def mock_l2_repository(self):
        """Create mock L2 repository"""
        repository = AsyncMock()
        repository.save = AsyncMock()
        return repository
    
    @pytest.fixture
    def mock_l3_repository(self):
        """Create mock L3 repository"""
        repository = AsyncMock()
        repository.save = AsyncMock()
        return repository
    
    @pytest.fixture
    def tier_manager(self, mock_l1_repository, mock_l2_repository, mock_l3_repository):
        """Create tier manager instance"""
        return MemoryTierManager(
            l1_repository=mock_l1_repository,
            l2_repository=mock_l2_repository,
            l3_repository=mock_l3_repository
        )
    
    @pytest.fixture
    def sample_working_memories(self):
        """Create sample working memories"""
        agent_id = uuid4()
        workflow_id = uuid4()
        
        memories = []
        for i in range(5):
            memory = WorkingMemory.create(
                agent_id=agent_id,
                workflow_execution_id=workflow_id,
                content=f"Test memory content {i}",
                memory_type=MemoryType.OBSERVATION,
                importance_score=0.7 + (i * 0.05),
                ttl_hours=1
            )
            # Simulate different creation times
            memory.created_at = datetime.utcnow() - timedelta(minutes=i*10)
            memories.append(memory)
        
        return memories
    
    @pytest.mark.asyncio
    async def test_promote_l1_to_l2_success(
        self, tier_manager, mock_l2_repository, sample_working_memories
    ):
        """Test successful L1 to L2 promotion"""
        cluster_id = "cluster_123"
        summary = "Compressed summary of related observations"
        
        # Promote memories
        episodic, event = await tier_manager.promote_l1_to_l2(
            source_memories=sample_working_memories,
            cluster_id=cluster_id,
            summary=summary
        )
        
        # Verify episodic memory was created
        assert isinstance(episodic, EpisodicMemory)
        assert episodic.cluster_id == cluster_id
        assert episodic.summary == summary
        assert len(episodic.source_memory_ids) == 5
        
        # Verify temporal period
        assert episodic.temporal_start is not None
        assert episodic.temporal_end is not None
        assert episodic.temporal_start <= episodic.temporal_end
        
        # Verify importance score (average)
        expected_avg = sum(m.importance_score for m in sample_working_memories) / 5
        assert episodic.importance_score == pytest.approx(expected_avg, rel=0.01)
        
        # Verify compression ratio was calculated
        assert episodic.compression_ratio > 0.0
        
        # Verify save was called
        mock_l2_repository.save.assert_called_once_with(episodic)
        
        # Verify event was created
        assert event.source_memory_ids == [m.id for m in sample_working_memories]
        assert event.episodic_memory_id == episodic.id
        assert event.cluster_id == cluster_id
    
    @pytest.mark.asyncio
    async def test_promote_l1_to_l2_with_embedding(
        self, tier_manager, mock_l2_repository, sample_working_memories
    ):
        """Test L1 to L2 promotion with embedding"""
        embedding = MemoryEmbedding(
            vector=[0.1, 0.2, 0.3],
            model="text-embedding-ada-002",
            dimensions=3
        )
        
        episodic, event = await tier_manager.promote_l1_to_l2(
            source_memories=sample_working_memories,
            cluster_id="cluster_456",
            summary="Test summary",
            embedding=embedding
        )
        
        # Verify embedding was preserved
        assert episodic.embedding == embedding
        assert episodic.embedding.model == "text-embedding-ada-002"
    
    @pytest.mark.asyncio
    async def test_promote_l1_to_l2_empty_list_raises_error(self, tier_manager):
        """Test that promoting empty list raises ValueError"""
        with pytest.raises(ValueError, match="Cannot promote empty memory list"):
            await tier_manager.promote_l1_to_l2(
                source_memories=[],
                cluster_id="cluster_789",
                summary="Should fail"
            )
    
    @pytest.mark.asyncio
    async def test_promote_l2_to_l3_success(
        self, tier_manager, mock_l3_repository
    ):
        """Test successful L2 to L3 promotion"""
        agent_id = uuid4()
        
        # Create episodic memory
        episodic = EpisodicMemory.create(
            agent_id=agent_id,
            cluster_id="cluster_abc",
            summary="Important episodic knowledge",
            source_memory_ids=[uuid4(), uuid4()],
            temporal_start=datetime.utcnow() - timedelta(hours=2),
            temporal_end=datetime.utcnow(),
            importance_score=0.85
        )
        
        # Promote to L3
        longterm, event = await tier_manager.promote_l2_to_l3(
            episodic_memory=episodic,
            knowledge_type=KnowledgeType.PROCEDURAL,
            title="Learned Procedure",
            content="Step-by-step procedure learned from experience"
        )
        
        # Verify long-term memory was created
        assert isinstance(longterm, LongTermMemory)
        assert longterm.knowledge_type == KnowledgeType.PROCEDURAL
        assert longterm.title == "Learned Procedure"
        assert longterm.content == "Step-by-step procedure learned from experience"
        assert longterm.source_type == SourceType.EPISODIC_COMPRESSION
        assert longterm.importance_score == 0.85
        assert episodic.id in longterm.source_references
        
        # Verify save was called
        mock_l3_repository.save.assert_called_once_with(longterm)
        
        # Verify event
        assert event.episodic_memory_id == episodic.id
        assert event.longterm_memory_id == longterm.id
    
    @pytest.mark.asyncio
    async def test_promote_l2_to_l3_uses_episodic_summary_as_default(
        self, tier_manager, mock_l3_repository
    ):
        """Test L2 to L3 promotion uses episodic summary when content not provided"""
        episodic = EpisodicMemory.create(
            agent_id=uuid4(),
            cluster_id="cluster_xyz",
            summary="This should become the content",
            source_memory_ids=[uuid4()],
            temporal_start=datetime.utcnow(),
            temporal_end=datetime.utcnow(),
            importance_score=0.75
        )
        
        longterm, event = await tier_manager.promote_l2_to_l3(
            episodic_memory=episodic,
            knowledge_type=KnowledgeType.SEMANTIC,
            title="Semantic Knowledge"
            # Note: no content provided
        )
        
        # Should use episodic summary as content
        assert longterm.content == "This should become the content"
    
    @pytest.mark.asyncio
    async def test_promote_l2_to_l3_preserves_embedding(
        self, tier_manager, mock_l3_repository
    ):
        """Test L2 to L3 promotion preserves embedding from episodic memory"""
        embedding = MemoryEmbedding(
            vector=[0.5, 0.6, 0.7],
            model="text-embedding-ada-002",
            dimensions=3
        )
        
        episodic = EpisodicMemory.create(
            agent_id=uuid4(),
            cluster_id="cluster_emb",
            summary="Summary with embedding",
            source_memory_ids=[uuid4()],
            temporal_start=datetime.utcnow(),
            temporal_end=datetime.utcnow(),
            importance_score=0.8,
            embedding=embedding
        )
        
        longterm, event = await tier_manager.promote_l2_to_l3(
            episodic_memory=episodic,
            knowledge_type=KnowledgeType.FACTUAL,
            title="Factual Knowledge"
        )
        
        # Embedding should be preserved
        assert longterm.embedding == embedding
    
    @pytest.mark.asyncio
    async def test_evict_from_l1_success(
        self, tier_manager, mock_l1_repository
    ):
        """Test successful L1 eviction"""
        agent_id = uuid4()
        workflow_id = uuid4()
        
        # Create LRU candidates
        candidates = []
        for i in range(3):
            memory = WorkingMemory.create(
                agent_id=agent_id,
                workflow_execution_id=workflow_id,
                content=f"Old memory {i}",
                memory_type=MemoryType.OBSERVATION,
                importance_score=0.5,
                ttl_hours=1
            )
            memory.created_at = datetime.utcnow() - timedelta(hours=10)
            memory.access_count = 1
            candidates.append(memory)
        
        mock_l1_repository.find_lru_candidates.return_value = candidates
        
        # Evict memories
        evicted_ids, events = await tier_manager.evict_from_l1(
            agent_id=agent_id,
            workflow_execution_id=workflow_id,
            max_count=10
        )
        
        # Verify eviction
        assert len(evicted_ids) == 3
        assert len(events) == 3
        
        # Verify delete was called for each memory
        assert mock_l1_repository.delete.call_count == 3
        
        # Verify events
        for event in events:
            assert event.tier == MemoryTier.L1_WORKING
            assert event.reason == "lru_eviction"
            assert event.agent_id == agent_id
    
    @pytest.mark.asyncio
    async def test_evict_from_l1_respects_max_count(
        self, tier_manager, mock_l1_repository
    ):
        """Test that eviction respects max_count parameter"""
        agent_id = uuid4()
        workflow_id = uuid4()
        
        # Find LRU should be called with correct limit
        await tier_manager.evict_from_l1(
            agent_id=agent_id,
            workflow_execution_id=workflow_id,
            max_count=5
        )
        
        mock_l1_repository.find_lru_candidates.assert_called_once_with(
            agent_id=agent_id,
            workflow_execution_id=workflow_id,
            limit=5
        )
    
    @pytest.mark.asyncio
    async def test_evict_from_l1_empty_candidates(
        self, tier_manager, mock_l1_repository
    ):
        """Test eviction with no LRU candidates"""
        mock_l1_repository.find_lru_candidates.return_value = []
        
        evicted_ids, events = await tier_manager.evict_from_l1(
            agent_id=uuid4(),
            workflow_execution_id=uuid4(),
            max_count=10
        )
        
        # Should return empty lists
        assert evicted_ids == []
        assert events == []
        assert mock_l1_repository.delete.call_count == 0
