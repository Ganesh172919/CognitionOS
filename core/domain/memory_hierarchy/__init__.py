"""
Memory Hierarchy Domain

Hierarchical memory system (L1/L2/L3) for CognitionOS.

This module provides:
- L1 Working Memory: Fast, short-term memory for active operations
- L2 Episodic Memory: Compressed clusters of related working memories
- L3 Long-Term Memory: Persistent knowledge and learnings

Components:
- entities: Memory tier entities (WorkingMemory, EpisodicMemory, LongTermMemory)
- services: Domain services for tier management, importance scoring, compression
- events: Domain events for memory lifecycle operations
- repositories: Repository interfaces for persistence

Design Principles:
- Clean architecture with zero external dependencies
- Domain-driven design patterns
- Automatic tier promotion/demotion based on importance
- LRU eviction for capacity management
- Semantic clustering and compression
"""

from .entities import (
    # Enums
    MemoryTier,
    MemoryType,
    KnowledgeType,
    SourceType,
    # Value Objects
    MemoryEmbedding,
    # Entities
    WorkingMemory,
    EpisodicMemory,
    LongTermMemory,
)

from .services import (
    MemoryTierManager,
    MemoryImportanceScorer,
    MemoryCompressionService,
)

from .events import (
    MemoryPromoted,
    L1ToL2Promotion,
    L2ToL3Promotion,
    MemoryDemoted,
    MemoryEvicted,
    MemoryCompressed,
    MemoryClusterCreated,
    MemoryAccessed,
    MemoryArchived,
)

from .repositories import (
    WorkingMemoryRepository,
    EpisodicMemoryRepository,
    LongTermMemoryRepository,
)

__all__ = [
    # Enums
    "MemoryTier",
    "MemoryType",
    "KnowledgeType",
    "SourceType",
    # Value Objects
    "MemoryEmbedding",
    # Entities
    "WorkingMemory",
    "EpisodicMemory",
    "LongTermMemory",
    # Services
    "MemoryTierManager",
    "MemoryImportanceScorer",
    "MemoryCompressionService",
    # Events
    "MemoryPromoted",
    "L1ToL2Promotion",
    "L2ToL3Promotion",
    "MemoryDemoted",
    "MemoryEvicted",
    "MemoryCompressed",
    "MemoryClusterCreated",
    "MemoryAccessed",
    "MemoryArchived",
    # Repositories
    "WorkingMemoryRepository",
    "EpisodicMemoryRepository",
    "LongTermMemoryRepository",
]
