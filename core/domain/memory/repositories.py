"""
Memory Domain - Repository Interfaces

Pure interfaces for memory persistence.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from .entities import (
    Memory,
    MemoryCollection,
    MemoryLifecyclePolicy,
    MemoryId,
    MemoryType,
    MemoryScope,
    MemoryStatus,
    MemoryNamespace,
    Embedding
)


class MemoryRepository(ABC):
    """
    Repository interface for Memory aggregate.

    Handles persistence and retrieval of memories.
    """

    @abstractmethod
    async def save(self, memory: Memory) -> None:
        """
        Persist memory.

        Args:
            memory: Memory to save
        """
        pass

    @abstractmethod
    async def get_by_id(self, memory_id: MemoryId) -> Optional[Memory]:
        """
        Retrieve memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_by_user(
        self,
        user_id: UUID,
        scope: Optional[MemoryScope] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100
    ) -> List[Memory]:
        """
        Get memories for a user.

        Args:
            user_id: User identifier
            scope: Optional scope filter
            memory_type: Optional type filter
            limit: Maximum number of memories to return

        Returns:
            List of memories
        """
        pass

    @abstractmethod
    async def get_by_namespace(
        self,
        namespace: MemoryNamespace,
        limit: int = 100
    ) -> List[Memory]:
        """
        Get memories in a namespace.

        Args:
            namespace: Namespace
            limit: Maximum number to return

        Returns:
            List of memories in namespace
        """
        pass

    @abstractmethod
    async def semantic_search(
        self,
        query_embedding: Embedding,
        user_id: Optional[UUID] = None,
        scope: Optional[MemoryScope] = None,
        namespace: Optional[MemoryNamespace] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[tuple[Memory, float]]:
        """
        Semantic search using vector similarity.

        Args:
            query_embedding: Query vector
            user_id: Optional user filter
            scope: Optional scope filter
            namespace: Optional namespace filter
            limit: Maximum results
            similarity_threshold: Minimum similarity score

        Returns:
            List of (memory, similarity_score) tuples sorted by similarity
        """
        pass

    @abstractmethod
    async def search_by_content(
        self,
        query: str,
        user_id: Optional[UUID] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100
    ) -> List[Memory]:
        """
        Full-text search on memory content.

        Args:
            query: Search query
            user_id: Optional user filter
            memory_type: Optional type filter
            limit: Maximum results

        Returns:
            List of matching memories
        """
        pass

    @abstractmethod
    async def get_by_status(
        self,
        status: MemoryStatus,
        limit: int = 100
    ) -> List[Memory]:
        """
        Get memories by status.

        Args:
            status: Memory status
            limit: Maximum results

        Returns:
            List of memories with status
        """
        pass

    @abstractmethod
    async def get_stale_memories(
        self,
        threshold_days: int = 90,
        limit: int = 100
    ) -> List[Memory]:
        """
        Get stale memories not accessed within threshold.

        Args:
            threshold_days: Days since last access
            limit: Maximum results

        Returns:
            List of stale memories
        """
        pass

    @abstractmethod
    async def delete(self, memory_id: MemoryId) -> bool:
        """
        Hard delete memory.

        Args:
            memory_id: Memory identifier

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def count_by_user(self, user_id: UUID) -> int:
        """
        Count memories for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of memories
        """
        pass


class MemoryCollectionRepository(ABC):
    """
    Repository interface for MemoryCollection aggregate.

    Handles persistence of memory collections.
    """

    @abstractmethod
    async def save(self, collection: MemoryCollection) -> None:
        """
        Persist memory collection.

        Args:
            collection: Collection to save
        """
        pass

    @abstractmethod
    async def get_by_id(self, collection_id: UUID) -> Optional[MemoryCollection]:
        """
        Retrieve collection by ID.

        Args:
            collection_id: Collection identifier

        Returns:
            Collection if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_by_namespace(
        self,
        namespace: MemoryNamespace
    ) -> List[MemoryCollection]:
        """
        Get collections in a namespace.

        Args:
            namespace: Namespace

        Returns:
            List of collections
        """
        pass

    @abstractmethod
    async def get_containing_memory(
        self,
        memory_id: MemoryId
    ) -> List[MemoryCollection]:
        """
        Get all collections containing a specific memory.

        Args:
            memory_id: Memory identifier

        Returns:
            List of collections containing the memory
        """
        pass

    @abstractmethod
    async def delete(self, collection_id: UUID) -> bool:
        """
        Delete collection.

        Args:
            collection_id: Collection identifier

        Returns:
            True if deleted, False if not found
        """
        pass


class MemoryLifecyclePolicyRepository(ABC):
    """
    Repository interface for MemoryLifecyclePolicy aggregate.

    Handles persistence of lifecycle policies.
    """

    @abstractmethod
    async def save(self, policy: MemoryLifecyclePolicy) -> None:
        """
        Persist lifecycle policy.

        Args:
            policy: Policy to save
        """
        pass

    @abstractmethod
    async def get_by_id(self, policy_id: UUID) -> Optional[MemoryLifecyclePolicy]:
        """
        Retrieve policy by ID.

        Args:
            policy_id: Policy identifier

        Returns:
            Policy if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_by_namespace(
        self,
        namespace: MemoryNamespace
    ) -> Optional[MemoryLifecyclePolicy]:
        """
        Get policy for a namespace.

        Args:
            namespace: Namespace

        Returns:
            Policy if found, None otherwise
        """
        pass

    @abstractmethod
    async def list_all(self) -> List[MemoryLifecyclePolicy]:
        """
        List all policies.

        Returns:
            List of all policies
        """
        pass

    @abstractmethod
    async def delete(self, policy_id: UUID) -> bool:
        """
        Delete policy.

        Args:
            policy_id: Policy identifier

        Returns:
            True if deleted, False if not found
        """
        pass
