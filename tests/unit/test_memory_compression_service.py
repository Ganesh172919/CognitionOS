"""
Unit Tests for Memory Compression Service

Tests for MemoryCompressionService clustering and compression logic.
"""

import pytest
from uuid import uuid4
import math

from core.domain.memory_hierarchy.entities import (
    WorkingMemory,
    MemoryType,
    MemoryEmbedding,
)
from core.domain.memory_hierarchy.services import MemoryCompressionService


class TestMemoryCompressionService:
    """Tests for MemoryCompressionService"""
    
    @pytest.fixture
    def compression_service(self):
        """Create compression service with default settings"""
        return MemoryCompressionService(
            min_cluster_size=3,
            max_cluster_size=10
        )
    
    @pytest.fixture
    def sample_memories(self):
        """Create sample working memories"""
        agent_id = uuid4()
        workflow_id = uuid4()
        
        memories = []
        for i in range(5):
            memory = WorkingMemory.create(
                agent_id=agent_id,
                workflow_execution_id=workflow_id,
                content=f"Memory content {i} with some text to compress",
                memory_type=MemoryType.OBSERVATION,
                importance_score=0.7,
                ttl_hours=1
            )
            memories.append(memory)
        
        return memories
    
    def test_compression_service_initialization(self):
        """Test compression service initializes correctly"""
        service = MemoryCompressionService(
            min_cluster_size=5,
            max_cluster_size=15
        )
        
        assert service.min_cluster_size == 5
        assert service.max_cluster_size == 15
    
    @pytest.mark.asyncio
    async def test_compress_memories_success(
        self, compression_service, sample_memories
    ):
        """Test successful memory compression"""
        summary = "Compressed summary of observations"
        
        cluster_id, event = await compression_service.compress_memories(
            memories=sample_memories,
            summary=summary
        )
        
        # Verify cluster ID was generated
        assert cluster_id is not None
        assert cluster_id.startswith("cluster_")
        
        # Verify event details
        assert len(event.source_memory_ids) == 5
        assert event.original_length > 0
        assert event.compressed_length == len(summary)
        assert 0.0 <= event.compression_ratio <= 1.0
    
    @pytest.mark.asyncio
    async def test_compress_memories_with_custom_cluster_id(
        self, compression_service, sample_memories
    ):
        """Test compression with custom cluster ID"""
        custom_cluster_id = "my_custom_cluster"
        summary = "Test summary"
        
        cluster_id, event = await compression_service.compress_memories(
            memories=sample_memories,
            summary=summary,
            cluster_id=custom_cluster_id
        )
        
        # Should use provided cluster ID
        assert cluster_id == custom_cluster_id
    
    @pytest.mark.asyncio
    async def test_compress_memories_too_small_raises_error(
        self, compression_service
    ):
        """Test that compressing too few memories raises error"""
        agent_id = uuid4()
        workflow_id = uuid4()
        
        # Only 2 memories (min is 3)
        memories = [
            WorkingMemory.create(
                agent_id=agent_id,
                workflow_execution_id=workflow_id,
                content="Memory 1",
                memory_type=MemoryType.OBSERVATION,
                importance_score=0.7,
                ttl_hours=1
            ),
            WorkingMemory.create(
                agent_id=agent_id,
                workflow_execution_id=workflow_id,
                content="Memory 2",
                memory_type=MemoryType.OBSERVATION,
                importance_score=0.7,
                ttl_hours=1
            )
        ]
        
        with pytest.raises(ValueError, match="Cluster too small"):
            await compression_service.compress_memories(
                memories=memories,
                summary="Should fail"
            )
    
    @pytest.mark.asyncio
    async def test_compress_memories_too_large_raises_error(
        self, compression_service
    ):
        """Test that compressing too many memories raises error"""
        agent_id = uuid4()
        workflow_id = uuid4()
        
        # 11 memories (max is 10)
        memories = []
        for i in range(11):
            memory = WorkingMemory.create(
                agent_id=agent_id,
                workflow_execution_id=workflow_id,
                content=f"Memory {i}",
                memory_type=MemoryType.OBSERVATION,
                importance_score=0.7,
                ttl_hours=1
            )
            memories.append(memory)
        
        with pytest.raises(ValueError, match="Cluster too large"):
            await compression_service.compress_memories(
                memories=memories,
                summary="Should fail"
            )
    
    @pytest.mark.asyncio
    async def test_compress_memories_calculates_compression_ratio(
        self, compression_service
    ):
        """Test that compression ratio is calculated correctly"""
        agent_id = uuid4()
        workflow_id = uuid4()
        
        memories = []
        for i in range(3):
            memory = WorkingMemory.create(
                agent_id=agent_id,
                workflow_execution_id=workflow_id,
                content="x" * 100,  # 100 chars each
                memory_type=MemoryType.OBSERVATION,
                importance_score=0.7,
                ttl_hours=1
            )
            memories.append(memory)
        
        summary = "y" * 50  # 50 chars compressed
        
        cluster_id, event = await compression_service.compress_memories(
            memories=memories,
            summary=summary
        )
        
        # Original: 300 chars, Compressed: 50 chars
        # Ratio should be 50/300 = 0.166...
        assert event.original_length == 300
        assert event.compressed_length == 50
        assert event.compression_ratio == pytest.approx(0.1667, rel=0.01)
    
    def test_calculate_semantic_similarity_identical_vectors(
        self, compression_service
    ):
        """Test similarity calculation for identical vectors"""
        embedding1 = MemoryEmbedding(
            vector=[0.5, 0.5, 0.5],
            model="test-model",
            dimensions=3
        )
        embedding2 = MemoryEmbedding(
            vector=[0.5, 0.5, 0.5],
            model="test-model",
            dimensions=3
        )
        
        similarity = compression_service.calculate_semantic_similarity(
            embedding1, embedding2
        )
        
        # Identical vectors should have similarity of 1.0
        assert similarity == pytest.approx(1.0, rel=0.01)
    
    def test_calculate_semantic_similarity_orthogonal_vectors(
        self, compression_service
    ):
        """Test similarity calculation for orthogonal vectors"""
        embedding1 = MemoryEmbedding(
            vector=[1.0, 0.0, 0.0],
            model="test-model",
            dimensions=3
        )
        embedding2 = MemoryEmbedding(
            vector=[0.0, 1.0, 0.0],
            model="test-model",
            dimensions=3
        )
        
        similarity = compression_service.calculate_semantic_similarity(
            embedding1, embedding2
        )
        
        # Orthogonal vectors should have similarity of 0.0
        assert similarity == pytest.approx(0.0, abs=0.01)
    
    def test_calculate_semantic_similarity_opposite_vectors(
        self, compression_service
    ):
        """Test similarity calculation for opposite vectors"""
        embedding1 = MemoryEmbedding(
            vector=[1.0, 0.0, 0.0],
            model="test-model",
            dimensions=3
        )
        embedding2 = MemoryEmbedding(
            vector=[-1.0, 0.0, 0.0],
            model="test-model",
            dimensions=3
        )
        
        similarity = compression_service.calculate_semantic_similarity(
            embedding1, embedding2
        )
        
        # Opposite vectors have negative cosine, but we clamp to 0.0
        assert similarity >= 0.0
    
    def test_calculate_semantic_similarity_different_dimensions_raises_error(
        self, compression_service
    ):
        """Test that different dimension embeddings raise error"""
        embedding1 = MemoryEmbedding(
            vector=[1.0, 0.0],
            model="test-model",
            dimensions=2
        )
        embedding2 = MemoryEmbedding(
            vector=[1.0, 0.0, 0.0],
            model="test-model",
            dimensions=3
        )
        
        with pytest.raises(ValueError, match="same dimension"):
            compression_service.calculate_semantic_similarity(
                embedding1, embedding2
            )
    
    def test_calculate_semantic_similarity_zero_vectors(
        self, compression_service
    ):
        """Test similarity calculation with zero vectors"""
        embedding1 = MemoryEmbedding(
            vector=[0.0, 0.0, 0.0],
            model="test-model",
            dimensions=3
        )
        embedding2 = MemoryEmbedding(
            vector=[1.0, 0.0, 0.0],
            model="test-model",
            dimensions=3
        )
        
        similarity = compression_service.calculate_semantic_similarity(
            embedding1, embedding2
        )
        
        # Zero vector has no direction, similarity should be 0.0
        assert similarity == 0.0
    
    @pytest.mark.asyncio
    async def test_cluster_related_memories_with_similar_embeddings(
        self, compression_service
    ):
        """Test clustering memories with similar embeddings"""
        agent_id = uuid4()
        workflow_id = uuid4()
        
        # Create memories with similar embeddings
        memories = []
        for i in range(4):
            memory = WorkingMemory.create(
                agent_id=agent_id,
                workflow_execution_id=workflow_id,
                content=f"Memory {i}",
                memory_type=MemoryType.OBSERVATION,
                importance_score=0.7,
                ttl_hours=1
            )
            # Similar vectors
            memory.embedding = MemoryEmbedding(
                vector=[0.9, 0.1, 0.0],
                model="test-model",
                dimensions=3
            )
            memories.append(memory)
        
        clusters = await compression_service.cluster_related_memories(
            memories=memories,
            similarity_threshold=0.7
        )
        
        # All should be in one cluster (they're very similar)
        assert len(clusters) >= 1
    
    @pytest.mark.asyncio
    async def test_cluster_related_memories_empty_list(
        self, compression_service
    ):
        """Test clustering empty list returns empty dict"""
        clusters = await compression_service.cluster_related_memories(
            memories=[],
            similarity_threshold=0.7
        )
        
        assert clusters == {}
    
    @pytest.mark.asyncio
    async def test_cluster_related_memories_no_embeddings(
        self, compression_service
    ):
        """Test clustering memories without embeddings returns empty dict"""
        agent_id = uuid4()
        workflow_id = uuid4()
        
        memories = []
        for i in range(3):
            memory = WorkingMemory.create(
                agent_id=agent_id,
                workflow_execution_id=workflow_id,
                content=f"Memory {i}",
                memory_type=MemoryType.OBSERVATION,
                importance_score=0.7,
                ttl_hours=1
            )
            # No embedding set
            memories.append(memory)
        
        clusters = await compression_service.cluster_related_memories(
            memories=memories,
            similarity_threshold=0.7
        )
        
        # No embeddings means no clustering
        assert clusters == {}
