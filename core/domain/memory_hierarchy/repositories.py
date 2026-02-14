"""
Memory Hierarchy Domain - Repositories

Repository interfaces for memory hierarchy persistence.
NO external dependencies except Python stdlib.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from .entities import WorkingMemory, EpisodicMemory, LongTermMemory, MemoryTier


class WorkingMemoryRepository(ABC):
    """Repository interface for L1 working memory persistence"""

    @abstractmethod
    async def save(self, memory: WorkingMemory) -> None:
        """
        Save a working memory.
        
        Args:
            memory: Working memory to save
        """
        pass

    @abstractmethod
    async def find_by_id(self, memory_id: UUID) -> Optional[WorkingMemory]:
        """
        Find a working memory by ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            WorkingMemory if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_agent(
        self,
        agent_id: UUID,
        workflow_execution_id: Optional[UUID] = None,
        limit: int = 100,
    ) -> List[WorkingMemory]:
        """
        Find working memories by agent.
        
        Args:
            agent_id: Agent ID
            workflow_execution_id: Optional workflow execution ID filter
            limit: Maximum number of results
            
        Returns:
            List of working memories
        """
        pass

    @abstractmethod
    async def find_by_importance(
        self,
        agent_id: UUID,
        min_importance: float,
        limit: int = 100,
    ) -> List[WorkingMemory]:
        """
        Find working memories by importance score.
        
        Args:
            agent_id: Agent ID
            min_importance: Minimum importance score
            limit: Maximum number of results
            
        Returns:
            List of working memories
        """
        pass

    @abstractmethod
    async def find_lru_candidates(
        self,
        agent_id: UUID,
        workflow_execution_id: UUID,
        limit: int = 10,
    ) -> List[WorkingMemory]:
        """
        Find LRU (Least Recently Used) eviction candidates.
        
        Args:
            agent_id: Agent ID
            workflow_execution_id: Workflow execution ID
            limit: Maximum number of candidates
            
        Returns:
            List of working memories ordered by LRU
        """
        pass

    @abstractmethod
    async def find_expired(
        self,
        agent_id: UUID,
        current_time: datetime,
    ) -> List[WorkingMemory]:
        """
        Find expired working memories.
        
        Args:
            agent_id: Agent ID
            current_time: Current time for comparison
            
        Returns:
            List of expired working memories
        """
        pass

    @abstractmethod
    async def delete(self, memory_id: UUID) -> None:
        """
        Delete a working memory.
        
        Args:
            memory_id: Memory ID to delete
        """
        pass

    @abstractmethod
    async def delete_batch(self, memory_ids: List[UUID]) -> int:
        """
        Delete multiple working memories.
        
        Args:
            memory_ids: List of memory IDs to delete
            
        Returns:
            Number of memories deleted
        """
        pass

    @abstractmethod
    async def count_by_agent(
        self,
        agent_id: UUID,
        workflow_execution_id: Optional[UUID] = None,
    ) -> int:
        """
        Count working memories for an agent.
        
        Args:
            agent_id: Agent ID
            workflow_execution_id: Optional workflow execution ID filter
            
        Returns:
            Count of working memories
        """
        pass


class EpisodicMemoryRepository(ABC):
    """Repository interface for L2 episodic memory persistence"""

    @abstractmethod
    async def save(self, memory: EpisodicMemory) -> None:
        """
        Save an episodic memory.
        
        Args:
            memory: Episodic memory to save
        """
        pass

    @abstractmethod
    async def find_by_id(self, memory_id: UUID) -> Optional[EpisodicMemory]:
        """
        Find an episodic memory by ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            EpisodicMemory if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_agent(
        self,
        agent_id: UUID,
        limit: int = 100,
    ) -> List[EpisodicMemory]:
        """
        Find episodic memories by agent.
        
        Args:
            agent_id: Agent ID
            limit: Maximum number of results
            
        Returns:
            List of episodic memories
        """
        pass

    @abstractmethod
    async def find_by_cluster(
        self,
        agent_id: UUID,
        cluster_id: str,
    ) -> Optional[EpisodicMemory]:
        """
        Find episodic memory by cluster ID.
        
        Args:
            agent_id: Agent ID
            cluster_id: Cluster ID
            
        Returns:
            EpisodicMemory if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_importance(
        self,
        agent_id: UUID,
        min_importance: float,
        limit: int = 100,
    ) -> List[EpisodicMemory]:
        """
        Find episodic memories by importance score.
        
        Args:
            agent_id: Agent ID
            min_importance: Minimum importance score
            limit: Maximum number of results
            
        Returns:
            List of episodic memories
        """
        pass

    @abstractmethod
    async def find_by_temporal_range(
        self,
        agent_id: UUID,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100,
    ) -> List[EpisodicMemory]:
        """
        Find episodic memories within a temporal range.
        
        Args:
            agent_id: Agent ID
            start_time: Start of temporal range
            end_time: End of temporal range
            limit: Maximum number of results
            
        Returns:
            List of episodic memories
        """
        pass

    @abstractmethod
    async def delete(self, memory_id: UUID) -> None:
        """
        Delete an episodic memory.
        
        Args:
            memory_id: Memory ID to delete
        """
        pass

    @abstractmethod
    async def count_by_agent(self, agent_id: UUID) -> int:
        """
        Count episodic memories for an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Count of episodic memories
        """
        pass


class LongTermMemoryRepository(ABC):
    """Repository interface for L3 long-term memory persistence"""

    @abstractmethod
    async def save(self, memory: LongTermMemory) -> None:
        """
        Save a long-term memory.
        
        Args:
            memory: Long-term memory to save
        """
        pass

    @abstractmethod
    async def find_by_id(self, memory_id: UUID) -> Optional[LongTermMemory]:
        """
        Find a long-term memory by ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            LongTermMemory if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_agent(
        self,
        agent_id: UUID,
        include_archived: bool = False,
        limit: int = 100,
    ) -> List[LongTermMemory]:
        """
        Find long-term memories by agent.
        
        Args:
            agent_id: Agent ID
            include_archived: Include archived memories
            limit: Maximum number of results
            
        Returns:
            List of long-term memories
        """
        pass

    @abstractmethod
    async def find_by_knowledge_type(
        self,
        agent_id: UUID,
        knowledge_type: str,
        limit: int = 100,
    ) -> List[LongTermMemory]:
        """
        Find long-term memories by knowledge type.
        
        Args:
            agent_id: Agent ID
            knowledge_type: Knowledge type filter
            limit: Maximum number of results
            
        Returns:
            List of long-term memories
        """
        pass

    @abstractmethod
    async def find_by_importance(
        self,
        agent_id: UUID,
        min_importance: float,
        limit: int = 100,
    ) -> List[LongTermMemory]:
        """
        Find long-term memories by importance score.
        
        Args:
            agent_id: Agent ID
            min_importance: Minimum importance score
            limit: Maximum number of results
            
        Returns:
            List of long-term memories
        """
        pass

    @abstractmethod
    async def find_archived(
        self,
        agent_id: UUID,
        limit: int = 100,
    ) -> List[LongTermMemory]:
        """
        Find archived long-term memories.
        
        Args:
            agent_id: Agent ID
            limit: Maximum number of results
            
        Returns:
            List of archived long-term memories
        """
        pass

    @abstractmethod
    async def search_by_title(
        self,
        agent_id: UUID,
        title_query: str,
        limit: int = 100,
    ) -> List[LongTermMemory]:
        """
        Search long-term memories by title.
        
        Args:
            agent_id: Agent ID
            title_query: Title search query
            limit: Maximum number of results
            
        Returns:
            List of matching long-term memories
        """
        pass

    @abstractmethod
    async def delete(self, memory_id: UUID) -> None:
        """
        Delete a long-term memory.
        
        Args:
            memory_id: Memory ID to delete
        """
        pass

    @abstractmethod
    async def count_by_agent(
        self,
        agent_id: UUID,
        include_archived: bool = False,
    ) -> int:
        """
        Count long-term memories for an agent.
        
        Args:
            agent_id: Agent ID
            include_archived: Include archived memories
            
        Returns:
            Count of long-term memories
        """
        pass
