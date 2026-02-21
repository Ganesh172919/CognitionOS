"""
Unit Tests for Memory Hierarchy Domain Entities

Tests for hierarchical memory (L1/L2/L3) domain entities.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from core.domain.memory_hierarchy.entities import (
    WorkingMemory,
    EpisodicMemory,
    LongTermMemory,
    MemoryTier,
    MemoryType,
    KnowledgeType,
)


class TestWorkingMemory:
    """Tests for WorkingMemory (L1) entity"""
    
    def test_working_memory_creation(self):
        """Test creating L1 working memory"""
        agent_id = uuid4()
        memory = WorkingMemory.create(
            agent_id=agent_id,
            content="Important fact about the system",
            embedding=[0.1] * 1536,
            importance_score=0.8,
            memory_type=MemoryType.FACT
        )
        
        assert memory.agent_id == agent_id
        assert memory.content == "Important fact about the system"
        assert memory.importance_score == 0.8
        assert memory.memory_type == MemoryType.FACT
        assert memory.access_count == 0
    
    def test_working_memory_update_access(self):
        """Test updating memory access"""
        agent_id = uuid4()
        memory = WorkingMemory.create(
            agent_id=agent_id,
            content="Test memory",
            embedding=[0.1] * 1536,
            importance_score=0.5
        )
        
        original_count = memory.access_count
        original_time = memory.last_accessed_at
        
        memory.update_access()
        
        assert memory.access_count == original_count + 1
        assert memory.last_accessed_at > original_time
    
    def test_working_memory_update_importance_score(self):
        """Test updating importance score"""
        agent_id = uuid4()
        memory = WorkingMemory.create(
            agent_id=agent_id,
            content="Test memory",
            embedding=[0.1] * 1536,
            importance_score=0.5
        )
        
        memory.update_importance_score(0.9)
        assert memory.importance_score == 0.9
    
    def test_working_memory_update_importance_score_validation(self):
        """Test importance score validation"""
        agent_id = uuid4()
        memory = WorkingMemory.create(
            agent_id=agent_id,
            content="Test memory",
            embedding=[0.1] * 1536,
            importance_score=0.5
        )
        
        # Score must be between 0 and 1
        with pytest.raises(ValueError, match="Importance score must be between 0 and 1"):
            memory.update_importance_score(1.5)
        
        with pytest.raises(ValueError, match="Importance score must be between 0 and 1"):
            memory.update_importance_score(-0.1)
    
    def test_working_memory_calculate_age_hours(self):
        """Test calculating memory age in hours"""
        agent_id = uuid4()
        memory = WorkingMemory.create(
            agent_id=agent_id,
            content="Old memory",
            embedding=[0.1] * 1536,
            importance_score=0.5
        )
        
        # Set created_at to 2 hours ago
        memory.created_at = datetime.utcnow() - timedelta(hours=2)
        
        age = memory.calculate_age_hours()
        assert age >= 1.9  # Allow for small timing differences
        assert age <= 2.1
    
    def test_working_memory_should_promote_to_l2(self):
        """Test L2 promotion criteria"""
        agent_id = uuid4()
        
        # High importance, frequently accessed memory should promote
        high_importance_memory = WorkingMemory.create(
            agent_id=agent_id,
            content="Important memory",
            embedding=[0.1] * 1536,
            importance_score=0.85
        )
        high_importance_memory.access_count = 15
        
        assert high_importance_memory.should_promote_to_l2() is True
        
        # Low importance memory should not promote
        low_importance_memory = WorkingMemory.create(
            agent_id=agent_id,
            content="Low importance",
            embedding=[0.1] * 1536,
            importance_score=0.3
        )
        low_importance_memory.access_count = 2
        
        assert low_importance_memory.should_promote_to_l2() is False
    
    def test_working_memory_is_expired(self):
        """Test checking if memory has expired"""
        agent_id = uuid4()
        memory = WorkingMemory.create(
            agent_id=agent_id,
            content="Expiring memory",
            embedding=[0.1] * 1536,
            importance_score=0.5
        )
        
        # No expiration set
        assert memory.is_expired() is False
        
        # Set expiration in past
        memory.expires_at = datetime.utcnow() - timedelta(hours=1)
        assert memory.is_expired() is True
        
        # Set expiration in future
        memory.expires_at = datetime.utcnow() + timedelta(hours=1)
        assert memory.is_expired() is False


class TestEpisodicMemory:
    """Tests for EpisodicMemory (L2) entity"""
    
    def test_episodic_memory_creation(self):
        """Test creating L2 episodic memory"""
        agent_id = uuid4()
        memory = EpisodicMemory.create(
            agent_id=agent_id,
            summary="Summary of multiple related facts",
            embedding=[0.2] * 1536,
            importance_score=0.75
        )
        
        assert memory.agent_id == agent_id
        assert memory.summary == "Summary of multiple related facts"
        assert memory.importance_score == 0.75
        assert memory.source_memory_count == 0
    
    def test_episodic_memory_add_source_memory(self):
        """Test adding source memory reference"""
        agent_id = uuid4()
        memory = EpisodicMemory.create(
            agent_id=agent_id,
            summary="Compressed summary",
            embedding=[0.2] * 1536,
            importance_score=0.7
        )
        
        source_id1 = uuid4()
        source_id2 = uuid4()
        
        memory.add_source_memory(source_id1)
        assert memory.source_memory_count == 1
        assert source_id1 in memory.source_memory_ids
        
        memory.add_source_memory(source_id2)
        assert memory.source_memory_count == 2
    
    def test_episodic_memory_calculate_compression_ratio(self):
        """Test calculating compression ratio"""
        agent_id = uuid4()
        memory = EpisodicMemory.create(
            agent_id=agent_id,
            summary="Compressed",  # 10 chars
            embedding=[0.2] * 1536,
            importance_score=0.7
        )
        
        original_size = 100  # Original content was 100 chars
        compressed_size = 10  # Summary is 10 chars
        
        ratio = memory.calculate_compression_ratio(original_size, compressed_size)
        assert ratio == 0.1  # 10/100 = 0.1


class TestLongTermMemory:
    """Tests for LongTermMemory (L3) entity"""
    
    def test_longterm_memory_creation(self):
        """Test creating L3 long-term memory"""
        agent_id = uuid4()
        memory = LongTermMemory.create(
            agent_id=agent_id,
            knowledge_type=KnowledgeType.FACTUAL,
            title="Important System Fact",
            content="Detailed description of an important system fact",
            embedding=[0.3] * 1536,
            importance_score=0.9
        )
        
        assert memory.agent_id == agent_id
        assert memory.knowledge_type == KnowledgeType.FACTUAL
        assert memory.title == "Important System Fact"
        assert memory.importance_score == 0.9
        assert memory.archived_at is None
    
    def test_longterm_memory_mark_archived(self):
        """Test marking memory as archived"""
        agent_id = uuid4()
        memory = LongTermMemory.create(
            agent_id=agent_id,
            knowledge_type=KnowledgeType.PROCEDURAL,
            title="Old Procedure",
            content="Outdated procedure description",
            embedding=[0.3] * 1536,
            importance_score=0.5
        )
        
        assert not memory.is_archived()
        
        memory.mark_archived()
        
        assert memory.is_archived()
        assert memory.archived_at is not None
    
    def test_longterm_memory_is_archived(self):
        """Test checking if memory is archived"""
        agent_id = uuid4()
        memory = LongTermMemory.create(
            agent_id=agent_id,
            knowledge_type=KnowledgeType.CONCEPTUAL,
            title="Active Concept",
            content="Current concept description",
            embedding=[0.3] * 1536,
            importance_score=0.8
        )
        
        # Not archived by default
        assert memory.is_archived() is False
        
        # Mark as archived
        memory.mark_archived()
        assert memory.is_archived() is True
    
    def test_longterm_memory_knowledge_types(self):
        """Test different knowledge types"""
        agent_id = uuid4()
        
        factual = LongTermMemory.create(
            agent_id=agent_id,
            knowledge_type=KnowledgeType.FACTUAL,
            title="Fact",
            content="A fact",
            embedding=[0.1] * 1536,
            importance_score=0.8
        )
        
        procedural = LongTermMemory.create(
            agent_id=agent_id,
            knowledge_type=KnowledgeType.PROCEDURAL,
            title="Procedure",
            content="A procedure",
            embedding=[0.2] * 1536,
            importance_score=0.7
        )
        
        conceptual = LongTermMemory.create(
            agent_id=agent_id,
            knowledge_type=KnowledgeType.CONCEPTUAL,
            title="Concept",
            content="A concept",
            embedding=[0.3] * 1536,
            importance_score=0.9
        )
        
        strategic = LongTermMemory.create(
            agent_id=agent_id,
            knowledge_type=KnowledgeType.STRATEGIC,
            title="Strategy",
            content="A strategy",
            embedding=[0.4] * 1536,
            importance_score=0.85
        )
        
        assert factual.knowledge_type == KnowledgeType.FACTUAL
        assert procedural.knowledge_type == KnowledgeType.PROCEDURAL
        assert conceptual.knowledge_type == KnowledgeType.CONCEPTUAL
        assert strategic.knowledge_type == KnowledgeType.STRATEGIC


class TestMemoryTier:
    """Tests for MemoryTier enum"""
    
    def test_memory_tier_values(self):
        """Test memory tier enum values"""
        assert MemoryTier.L1_WORKING.value == "l1_working"
        assert MemoryTier.L2_EPISODIC.value == "l2_episodic"
        assert MemoryTier.L3_LONGTERM.value == "l3_longterm"
