"""
Memory Domain Package

Exports all domain entities, value objects, services, and events
for the Memory bounded context.
"""

# Entities and Value Objects
from .entities import (
    Memory,
    MemoryCollection,
    MemoryLifecyclePolicy,
    MemoryId,
    Embedding,
    MemoryNamespace,
    MemoryType,
    MemoryScope,
    MemoryStatus,
    MemoryImportance
)

# Repository Interfaces
from .repositories import (
    MemoryRepository,
    MemoryCollectionRepository,
    MemoryLifecyclePolicyRepository
)

# Domain Services
from .services import (
    MemoryIndexer,
    MemoryDeduplicator,
    MemoryGarbageCollector,
    MemoryRetrieval,
    MemoryNamespaceManager
)

# Domain Events
from .events import (
    DomainEvent,
    MemoryCreated,
    MemoryAccessed,
    MemoryContentUpdated,
    MemoryEmbeddingGenerated,
    MemoryImportanceChanged,
    MemoryCompressed,
    MemoryArchived,
    MemoryDeleted,
    MemoryCollectionCreated,
    MemoryAddedToCollection,
    MemoryRemovedFromCollection,
    LifecyclePolicyCreated,
    LifecyclePolicyApplied,
    DuplicateMemoriesDetected,
    MemoriesMerged
)

__all__ = [
    # Entities
    "Memory",
    "MemoryCollection",
    "MemoryLifecyclePolicy",
    # Value Objects
    "MemoryId",
    "Embedding",
    "MemoryNamespace",
    # Enums
    "MemoryType",
    "MemoryScope",
    "MemoryStatus",
    "MemoryImportance",
    # Repositories
    "MemoryRepository",
    "MemoryCollectionRepository",
    "MemoryLifecyclePolicyRepository",
    # Services
    "MemoryIndexer",
    "MemoryDeduplicator",
    "MemoryGarbageCollector",
    "MemoryRetrieval",
    "MemoryNamespaceManager",
    # Events
    "DomainEvent",
    "MemoryCreated",
    "MemoryAccessed",
    "MemoryContentUpdated",
    "MemoryEmbeddingGenerated",
    "MemoryImportanceChanged",
    "MemoryCompressed",
    "MemoryArchived",
    "MemoryDeleted",
    "MemoryCollectionCreated",
    "MemoryAddedToCollection",
    "MemoryRemovedFromCollection",
    "LifecyclePolicyCreated",
    "LifecyclePolicyApplied",
    "DuplicateMemoriesDetected",
    "MemoriesMerged",
]
