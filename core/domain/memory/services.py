"""
Memory Domain - Domain Services

Domain services for memory-related business logic.
"""

from typing import List, Optional, Tuple
from datetime import datetime, timedelta

from .entities import (
    Memory,
    MemoryCollection,
    MemoryLifecyclePolicy,
    MemoryId,
    MemoryImportance,
    MemoryStatus,
    Embedding
)


class MemoryIndexer:
    """
    Domain service for indexing and embedding memories.

    Coordinates generation of vector embeddings for semantic search.
    """

    @staticmethod
    def needs_embedding(memory: Memory) -> bool:
        """Check if memory needs embedding generation"""
        return memory.embedding is None and memory.status == MemoryStatus.ACTIVE

    @staticmethod
    def is_embedding_stale(memory: Memory) -> bool:
        """Check if embedding is outdated relative to content"""
        # If content was updated after embedding would have been generated
        # (This is a simplified check - real implementation would track embedding timestamp)
        return memory.embedding is not None and memory.updated_at > memory.created_at


class MemoryDeduplicator:
    """
    Domain service for detecting and merging duplicate memories.

    Prevents storage of redundant information.
    """

    @staticmethod
    def find_duplicates(
        memory: Memory,
        candidates: List[Memory],
        similarity_threshold: float = 0.95
    ) -> List[Memory]:
        """
        Find duplicate memories based on semantic similarity.

        Args:
            memory: Memory to check
            candidates: Candidate memories to compare against
            similarity_threshold: Threshold for considering duplicate

        Returns:
            List of likely duplicate memories
        """
        duplicates = []

        for candidate in candidates:
            if candidate.id == memory.id:
                continue

            # Check exact content match first
            if memory.content.strip() == candidate.content.strip():
                duplicates.append(candidate)
                continue

            # Check semantic similarity if embeddings available
            similarity = memory.similarity_to(candidate)
            if similarity >= similarity_threshold:
                duplicates.append(candidate)

        return duplicates

    @staticmethod
    def should_merge(memory1: Memory, memory2: Memory) -> bool:
        """
        Determine if two memories should be merged.

        Merge if they are very similar and in same namespace.
        """
        if memory1.namespace.full_path() != memory2.namespace.full_path():
            return False

        if memory1.memory_type != memory2.memory_type:
            return False

        # Very high similarity threshold for auto-merge
        return memory1.similarity_to(memory2) >= 0.98


class MemoryGarbageCollector:
    """
    Domain service for memory lifecycle management.

    Applies policies to compress, archive, and delete old memories.
    """

    @staticmethod
    def apply_policy(
        memories: List[Memory],
        policy: MemoryLifecyclePolicy
    ) -> Tuple[List[Memory], List[Memory], List[Memory]]:
        """
        Apply lifecycle policy to memories.

        Returns:
            Tuple of (to_compress, to_archive, to_delete) memory lists
        """
        to_compress = []
        to_archive = []
        to_delete = []

        for memory in memories:
            if policy.should_delete(memory):
                to_delete.append(memory)
            elif policy.should_archive(memory):
                to_archive.append(memory)
            elif policy.should_compress(memory):
                to_compress.append(memory)

        return (to_compress, to_archive, to_delete)

    @staticmethod
    def get_cleanup_candidates(
        memories: List[Memory],
        max_memories: int
    ) -> List[Memory]:
        """
        Get memories that are candidates for cleanup.

        Selects least important, least accessed, oldest memories.

        Args:
            memories: All memories
            max_memories: Maximum to keep

        Returns:
            Memories to clean up
        """
        if len(memories) <= max_memories:
            return []

        # Score memories for cleanup (higher score = more likely to clean up)
        def cleanup_score(memory: Memory) -> Tuple[int, int, float]:
            importance_scores = {
                MemoryImportance.CRITICAL: 0,
                MemoryImportance.HIGH: 1,
                MemoryImportance.MEDIUM: 2,
                MemoryImportance.LOW: 3
            }

            return (
                importance_scores.get(memory.importance, 2),  # Lower importance = higher score
                -memory.access_count,  # Fewer accesses = higher score
                memory.age_days()  # Older = higher score
            )

        # Sort by cleanup score (descending)
        sorted_memories = sorted(memories, key=cleanup_score, reverse=True)

        # Return excess memories
        num_to_clean = len(memories) - max_memories
        return sorted_memories[:num_to_clean]


class MemoryRetrieval:
    """
    Domain service for intelligent memory retrieval.

    Implements retrieval strategies beyond simple similarity.
    """

    @staticmethod
    def rank_by_relevance(
        memories: List[Memory],
        recency_weight: float = 0.3,
        frequency_weight: float = 0.3,
        importance_weight: float = 0.4
    ) -> List[Memory]:
        """
        Rank memories by relevance considering multiple factors.

        Args:
            memories: Memories to rank
            recency_weight: Weight for recency score
            frequency_weight: Weight for access frequency
            importance_weight: Weight for importance

        Returns:
            Memories sorted by relevance (descending)
        """
        def relevance_score(memory: Memory) -> float:
            # Recency score (0-1, higher is more recent)
            days_since_access = (datetime.utcnow() - memory.accessed_at).days
            recency = 1.0 / (1.0 + days_since_access)

            # Frequency score (normalized access count)
            max_access = max(m.access_count for m in memories) or 1
            frequency = memory.access_count / max_access

            # Importance score (0-1)
            importance_scores = {
                MemoryImportance.LOW: 0.25,
                MemoryImportance.MEDIUM: 0.5,
                MemoryImportance.HIGH: 0.75,
                MemoryImportance.CRITICAL: 1.0
            }
            importance = importance_scores.get(memory.importance, 0.5)

            return (
                recency_weight * recency +
                frequency_weight * frequency +
                importance_weight * importance
            )

        return sorted(memories, key=relevance_score, reverse=True)

    @staticmethod
    def filter_by_freshness(
        memories: List[Memory],
        max_age_days: int = 30
    ) -> List[Memory]:
        """
        Filter memories to only fresh ones.

        Args:
            memories: Memories to filter
            max_age_days: Maximum age in days

        Returns:
            Memories within age threshold
        """
        threshold = timedelta(days=max_age_days)
        now = datetime.utcnow()

        return [
            memory for memory in memories
            if (now - memory.created_at) <= threshold
        ]


class MemoryNamespaceManager:
    """
    Domain service for managing memory namespaces.

    Provides hierarchical organization of memories.
    """

    @staticmethod
    def is_descendant(namespace: str, ancestor: str) -> bool:
        """
        Check if namespace is a descendant of ancestor.

        Example: "user.preferences.ui" is descendant of "user.preferences"
        """
        return namespace.startswith(f"{ancestor}.")

    @staticmethod
    def get_parent(namespace: str) -> Optional[str]:
        """
        Get parent namespace.

        Example: "user.preferences.ui" -> "user.preferences"
        """
        parts = namespace.split(".")
        if len(parts) <= 1:
            return None
        return ".".join(parts[:-1])

    @staticmethod
    def get_descendants(
        namespace: str,
        all_namespaces: List[str]
    ) -> List[str]:
        """Get all descendant namespaces"""
        return [
            ns for ns in all_namespaces
            if MemoryNamespaceManager.is_descendant(ns, namespace)
        ]
