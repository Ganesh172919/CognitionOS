"""
Memory Domain - Domain Events

Events representing state changes in the Memory domain.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID

from .entities import MemoryId, MemoryType, MemoryScope, MemoryImportance


@dataclass(frozen=True)
class DomainEvent:
    """Base class for all domain events"""
    occurred_at: datetime
    event_id: UUID


# ==================== Memory Events ====================

@dataclass(frozen=True)
class MemoryCreated(DomainEvent):
    """Event: A new memory was created"""
    memory_id: MemoryId
    user_id: UUID
    memory_type: MemoryType
    scope: MemoryScope
    namespace: str


@dataclass(frozen=True)
class MemoryAccessed(DomainEvent):
    """Event: A memory was accessed"""
    memory_id: MemoryId
    user_id: UUID
    access_count: int


@dataclass(frozen=True)
class MemoryContentUpdated(DomainEvent):
    """Event: Memory content was updated"""
    memory_id: MemoryId
    user_id: UUID
    old_content_length: int
    new_content_length: int


@dataclass(frozen=True)
class MemoryEmbeddingGenerated(DomainEvent):
    """Event: Embedding was generated for memory"""
    memory_id: MemoryId
    model: str
    dimensions: int


@dataclass(frozen=True)
class MemoryImportanceChanged(DomainEvent):
    """Event: Memory importance level changed"""
    memory_id: MemoryId
    old_importance: MemoryImportance
    new_importance: MemoryImportance


@dataclass(frozen=True)
class MemoryCompressed(DomainEvent):
    """Event: Memory was compressed"""
    memory_id: MemoryId
    compression_ratio: float


@dataclass(frozen=True)
class MemoryArchived(DomainEvent):
    """Event: Memory was archived"""
    memory_id: MemoryId
    reason: Optional[str]


@dataclass(frozen=True)
class MemoryDeleted(DomainEvent):
    """Event: Memory was deleted"""
    memory_id: MemoryId
    reason: Optional[str]


# ==================== Collection Events ====================

@dataclass(frozen=True)
class MemoryCollectionCreated(DomainEvent):
    """Event: A memory collection was created"""
    collection_id: UUID
    name: str
    namespace: str


@dataclass(frozen=True)
class MemoryAddedToCollection(DomainEvent):
    """Event: Memory was added to collection"""
    collection_id: UUID
    memory_id: MemoryId


@dataclass(frozen=True)
class MemoryRemovedFromCollection(DomainEvent):
    """Event: Memory was removed from collection"""
    collection_id: UUID
    memory_id: MemoryId


# ==================== Lifecycle Policy Events ====================

@dataclass(frozen=True)
class LifecyclePolicyCreated(DomainEvent):
    """Event: A lifecycle policy was created"""
    policy_id: UUID
    namespace: str
    compress_after_days: Optional[int]
    archive_after_days: Optional[int]
    delete_after_days: Optional[int]


@dataclass(frozen=True)
class LifecyclePolicyApplied(DomainEvent):
    """Event: Lifecycle policy was applied"""
    policy_id: UUID
    memories_compressed: int
    memories_archived: int
    memories_deleted: int


# ==================== Deduplication Events ====================

@dataclass(frozen=True)
class DuplicateMemoriesDetected(DomainEvent):
    """Event: Duplicate memories were detected"""
    memory_id: MemoryId
    duplicate_ids: list[MemoryId]
    similarity_score: float


@dataclass(frozen=True)
class MemoriesMerged(DomainEvent):
    """Event: Multiple memories were merged"""
    source_memory_ids: list[MemoryId]
    target_memory_id: MemoryId
