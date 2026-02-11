"""
Memory Domain - Entities

Pure domain entities for Memory bounded context.
NO external dependencies except Python stdlib.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


# ==================== Enums ====================

class MemoryType(str, Enum):
    """Types of memory content"""
    FACT = "fact"
    PATTERN = "pattern"
    PREFERENCE = "preference"
    SKILL = "skill"
    EXPERIENCE = "experience"
    KNOWLEDGE = "knowledge"


class MemoryScope(str, Enum):
    """Scope of memory visibility"""
    USER = "user"  # User-specific memory
    SESSION = "session"  # Session-specific memory
    AGENT = "agent"  # Agent-specific memory
    GLOBAL = "global"  # System-wide memory


class MemoryStatus(str, Enum):
    """Memory lifecycle status"""
    ACTIVE = "active"
    COMPRESSED = "compressed"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MemoryImportance(str, Enum):
    """Memory importance level"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ==================== Value Objects ====================

@dataclass(frozen=True)
class MemoryId:
    """Memory identifier value object"""
    value: UUID

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Embedding:
    """Vector embedding value object"""
    vector: tuple[float, ...]  # Immutable tuple
    model: str
    dimensions: int

    def __post_init__(self):
        if len(self.vector) != self.dimensions:
            raise ValueError(f"Vector length {len(self.vector)} doesn't match dimensions {self.dimensions}")

    @classmethod
    def from_list(cls, values: List[float], model: str) -> "Embedding":
        """Create embedding from list of floats"""
        return cls(
            vector=tuple(values),
            model=model,
            dimensions=len(values)
        )

    def to_list(self) -> List[float]:
        """Convert to list for serialization"""
        return list(self.vector)

    def cosine_similarity(self, other: "Embedding") -> float:
        """Calculate cosine similarity with another embedding"""
        if self.dimensions != other.dimensions:
            raise ValueError("Cannot compare embeddings of different dimensions")

        import math

        # Dot product
        dot_product = sum(a * b for a, b in zip(self.vector, other.vector))

        # Magnitudes
        mag_a = math.sqrt(sum(a * a for a in self.vector))
        mag_b = math.sqrt(sum(b * b for b in other.vector))

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot_product / (mag_a * mag_b)


@dataclass(frozen=True)
class MemoryNamespace:
    """Namespace for organizing memories"""
    name: str
    parent: Optional[str] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Namespace name cannot be empty")

    def full_path(self) -> str:
        """Get full namespace path"""
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name


# ==================== Entities ====================

@dataclass
class Memory:
    """
    Memory aggregate root.

    Represents a single memory entry with content, embedding, and metadata.

    Invariants:
    - Memory must have content
    - Embedding dimensions must match if provided
    - Cannot modify archived or deleted memories
    - Access count increases monotonically
    """
    id: MemoryId
    user_id: UUID
    content: str
    memory_type: MemoryType
    scope: MemoryScope
    namespace: MemoryNamespace
    embedding: Optional[Embedding] = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    importance: MemoryImportance = MemoryImportance.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate memory invariants"""
        if not self.content or len(self.content.strip()) == 0:
            raise ValueError("Memory content cannot be empty")
        if self.access_count < 0:
            raise ValueError("Access count cannot be negative")

    def can_modify(self) -> bool:
        """Business rule: Check if memory can be modified"""
        return self.status == MemoryStatus.ACTIVE

    def access(self) -> None:
        """
        Business rule: Record memory access.

        Increases access count and updates accessed timestamp.
        """
        self.access_count += 1
        self.accessed_at = datetime.utcnow()

    def update_content(self, new_content: str) -> None:
        """
        Business rule: Update memory content.

        Can only update ACTIVE memories.
        Clears embedding since content changed.
        """
        if not self.can_modify():
            raise ValueError(f"Cannot modify memory in {self.status} status")

        if not new_content or len(new_content.strip()) == 0:
            raise ValueError("New content cannot be empty")

        self.content = new_content
        self.embedding = None  # Clear embedding since content changed
        self.updated_at = datetime.utcnow()

    def set_embedding(self, embedding: Embedding) -> None:
        """
        Business rule: Set memory embedding.

        Can only set embedding for ACTIVE memories.
        """
        if not self.can_modify():
            raise ValueError(f"Cannot set embedding for memory in {self.status} status")

        self.embedding = embedding
        self.updated_at = datetime.utcnow()

    def add_tag(self, tag: str) -> None:
        """Add tag to memory (idempotent)"""
        if not self.can_modify():
            raise ValueError(f"Cannot modify memory in {self.status} status")

        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()

    def remove_tag(self, tag: str) -> None:
        """Remove tag from memory"""
        if not self.can_modify():
            raise ValueError(f"Cannot modify memory in {self.status} status")

        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()

    def set_importance(self, importance: MemoryImportance) -> None:
        """Update memory importance level"""
        if not self.can_modify():
            raise ValueError(f"Cannot modify memory in {self.status} status")

        self.importance = importance
        self.updated_at = datetime.utcnow()

    def compress(self, compression_ratio: float = 0.5) -> None:
        """
        Business rule: Compress memory.

        Can only compress ACTIVE memories.
        Compression reduces content size while preserving key information.
        """
        if self.status != MemoryStatus.ACTIVE:
            raise ValueError(f"Cannot compress memory in {self.status} status")

        if not (0 < compression_ratio <= 1.0):
            raise ValueError("Compression ratio must be between 0 and 1")

        # Mark as compressed (actual compression would be done by application layer)
        self.status = MemoryStatus.COMPRESSED
        self.updated_at = datetime.utcnow()

    def archive(self) -> None:
        """
        Business rule: Archive memory.

        Can archive ACTIVE or COMPRESSED memories.
        Archived memories are read-only.
        """
        if self.status not in [MemoryStatus.ACTIVE, MemoryStatus.COMPRESSED]:
            raise ValueError(f"Cannot archive memory in {self.status} status")

        self.status = MemoryStatus.ARCHIVED
        self.updated_at = datetime.utcnow()

    def mark_deleted(self) -> None:
        """
        Business rule: Mark memory as deleted.

        Soft delete - memory remains but is marked as deleted.
        """
        if self.status == MemoryStatus.DELETED:
            raise ValueError("Memory is already deleted")

        self.status = MemoryStatus.DELETED
        self.updated_at = datetime.utcnow()

    def is_stale(self, threshold_days: int = 90) -> bool:
        """
        Domain logic: Check if memory is stale.

        Memory is stale if not accessed within threshold days.
        """
        from datetime import timedelta
        threshold = timedelta(days=threshold_days)
        return (datetime.utcnow() - self.accessed_at) > threshold

    def similarity_to(self, other: "Memory") -> float:
        """
        Domain logic: Calculate similarity to another memory.

        Returns cosine similarity if both have embeddings, 0.0 otherwise.
        """
        if self.embedding is None or other.embedding is None:
            return 0.0

        return self.embedding.cosine_similarity(other.embedding)

    def age_days(self) -> int:
        """Calculate memory age in days"""
        return (datetime.utcnow() - self.created_at).days


@dataclass
class MemoryCollection:
    """
    Memory collection entity.

    Groups related memories together.
    """
    id: UUID
    name: str
    description: str
    namespace: MemoryNamespace
    memory_ids: List[MemoryId] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_memory(self, memory_id: MemoryId) -> None:
        """Add memory to collection (idempotent)"""
        if memory_id not in self.memory_ids:
            self.memory_ids.append(memory_id)
            self.updated_at = datetime.utcnow()

    def remove_memory(self, memory_id: MemoryId) -> None:
        """Remove memory from collection"""
        if memory_id in self.memory_ids:
            self.memory_ids.remove(memory_id)
            self.updated_at = datetime.utcnow()

    def contains(self, memory_id: MemoryId) -> bool:
        """Check if collection contains memory"""
        return memory_id in self.memory_ids

    def size(self) -> int:
        """Get number of memories in collection"""
        return len(self.memory_ids)

    def is_empty(self) -> bool:
        """Check if collection is empty"""
        return len(self.memory_ids) == 0


@dataclass
class MemoryLifecyclePolicy:
    """
    Memory lifecycle policy entity.

    Defines rules for memory compression, archival, and deletion.
    """
    id: UUID
    name: str
    namespace: MemoryNamespace
    compress_after_days: Optional[int] = 30
    archive_after_days: Optional[int] = 180
    delete_after_days: Optional[int] = 365
    importance_threshold: MemoryImportance = MemoryImportance.LOW
    created_at: datetime = field(default_factory=datetime.utcnow)

    def should_compress(self, memory: Memory) -> bool:
        """
        Domain logic: Check if memory should be compressed.

        Memory should be compressed if:
        - Policy specifies compression threshold
        - Memory is older than threshold
        - Memory is ACTIVE status
        - Memory importance is at or below threshold
        """
        if self.compress_after_days is None:
            return False

        if memory.status != MemoryStatus.ACTIVE:
            return False

        if memory.age_days() < self.compress_after_days:
            return False

        # Check importance level
        importance_levels = {
            MemoryImportance.LOW: 1,
            MemoryImportance.MEDIUM: 2,
            MemoryImportance.HIGH: 3,
            MemoryImportance.CRITICAL: 4
        }

        return importance_levels[memory.importance] <= importance_levels[self.importance_threshold]

    def should_archive(self, memory: Memory) -> bool:
        """
        Domain logic: Check if memory should be archived.

        Memory should be archived if:
        - Policy specifies archive threshold
        - Memory is older than threshold
        - Memory is ACTIVE or COMPRESSED
        """
        if self.archive_after_days is None:
            return False

        if memory.status not in [MemoryStatus.ACTIVE, MemoryStatus.COMPRESSED]:
            return False

        return memory.age_days() >= self.archive_after_days

    def should_delete(self, memory: Memory) -> bool:
        """
        Domain logic: Check if memory should be deleted.

        Memory should be deleted if:
        - Policy specifies deletion threshold
        - Memory is older than threshold
        - Memory has low access count (rarely used)
        """
        if self.delete_after_days is None:
            return False

        if memory.status == MemoryStatus.DELETED:
            return False

        if memory.age_days() < self.delete_after_days:
            return False

        # Don't delete important or frequently accessed memories
        if memory.importance in [MemoryImportance.CRITICAL, MemoryImportance.HIGH]:
            return False

        if memory.access_count > 10:  # Arbitrary threshold for "frequently used"
            return False

        return True
