"""
Memory Hierarchy Domain - Events

Domain events for memory hierarchy operations.
NO external dependencies except Python stdlib.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from .entities import MemoryTier, SourceType


# ==================== Promotion Events ====================

@dataclass
class MemoryPromoted:
    """
    Event emitted when a memory is promoted to a higher tier.
    
    L1 → L2: Working memory promoted to episodic
    L2 → L3: Episodic memory promoted to long-term
    """
    event_id: UUID
    memory_id: UUID
    agent_id: UUID
    from_tier: MemoryTier
    to_tier: MemoryTier
    importance_score: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": str(self.event_id),
            "timestamp": self.timestamp.isoformat(),
            "memory_id": str(self.memory_id),
            "agent_id": str(self.agent_id),
            "from_tier": self.from_tier.value,
            "to_tier": self.to_tier.value,
            "importance_score": self.importance_score,
            "reason": self.reason,
            "metadata": self.metadata,
        }


@dataclass
class L1ToL2Promotion:
    """Event for L1 → L2 promotion (working to episodic)"""
    event_id: UUID
    source_memory_ids: List[UUID]
    episodic_memory_id: UUID
    agent_id: UUID
    cluster_id: str
    compression_ratio: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": str(self.event_id),
            "timestamp": self.timestamp.isoformat(),
            "source_memory_ids": [str(mid) for mid in self.source_memory_ids],
            "episodic_memory_id": str(self.episodic_memory_id),
            "agent_id": str(self.agent_id),
            "cluster_id": self.cluster_id,
            "compression_ratio": self.compression_ratio,
            "metadata": self.metadata,
        }


@dataclass
class L2ToL3Promotion:
    """Event for L2 → L3 promotion (episodic to long-term)"""
    event_id: UUID
    episodic_memory_id: UUID
    longterm_memory_id: UUID
    agent_id: UUID
    source_type: SourceType
    importance_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": str(self.event_id),
            "timestamp": self.timestamp.isoformat(),
            "episodic_memory_id": str(self.episodic_memory_id),
            "longterm_memory_id": str(self.longterm_memory_id),
            "agent_id": str(self.agent_id),
            "source_type": self.source_type.value,
            "importance_score": self.importance_score,
            "metadata": self.metadata,
        }


# ==================== Demotion Events ====================

@dataclass
class MemoryDemoted:
    """Event emitted when a memory is demoted to a lower tier"""
    event_id: UUID
    memory_id: UUID
    agent_id: UUID
    from_tier: MemoryTier
    to_tier: MemoryTier
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": str(self.event_id),
            "timestamp": self.timestamp.isoformat(),
            "memory_id": str(self.memory_id),
            "agent_id": str(self.agent_id),
            "from_tier": self.from_tier.value,
            "to_tier": self.to_tier.value,
            "reason": self.reason,
            "metadata": self.metadata,
        }


# ==================== Eviction Events ====================

@dataclass
class MemoryEvicted:
    """Event emitted when a memory is evicted from L1"""
    event_id: UUID
    memory_id: UUID
    agent_id: UUID
    tier: MemoryTier
    reason: str
    access_count: int
    age_hours: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": str(self.event_id),
            "timestamp": self.timestamp.isoformat(),
            "memory_id": str(self.memory_id),
            "agent_id": str(self.agent_id),
            "tier": self.tier.value,
            "reason": self.reason,
            "access_count": self.access_count,
            "age_hours": self.age_hours,
            "metadata": self.metadata,
        }


# ==================== Compression Events ====================

@dataclass
class MemoryCompressed:
    """Event emitted when memories are compressed"""
    event_id: UUID
    source_memory_ids: List[UUID]
    compressed_memory_id: UUID
    agent_id: UUID
    original_length: int
    compressed_length: int
    compression_ratio: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": str(self.event_id),
            "timestamp": self.timestamp.isoformat(),
            "source_memory_ids": [str(mid) for mid in self.source_memory_ids],
            "compressed_memory_id": str(self.compressed_memory_id),
            "agent_id": str(self.agent_id),
            "original_length": self.original_length,
            "compressed_length": self.compressed_length,
            "compression_ratio": self.compression_ratio,
            "metadata": self.metadata,
        }


# ==================== Clustering Events ====================

@dataclass
class MemoryClusterCreated:
    """Event emitted when a memory cluster is created"""
    event_id: UUID
    cluster_id: str
    agent_id: UUID
    memory_ids: List[UUID]
    cluster_size: int
    avg_similarity: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": str(self.event_id),
            "timestamp": self.timestamp.isoformat(),
            "cluster_id": self.cluster_id,
            "agent_id": str(self.agent_id),
            "memory_ids": [str(mid) for mid in self.memory_ids],
            "cluster_size": self.cluster_size,
            "avg_similarity": self.avg_similarity,
            "metadata": self.metadata,
        }


# ==================== Access Events ====================

@dataclass
class MemoryAccessed:
    """Event emitted when a memory is accessed"""
    event_id: UUID
    memory_id: UUID
    agent_id: UUID
    tier: MemoryTier
    access_count: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": str(self.event_id),
            "timestamp": self.timestamp.isoformat(),
            "memory_id": str(self.memory_id),
            "agent_id": str(self.agent_id),
            "tier": self.tier.value,
            "access_count": self.access_count,
            "metadata": self.metadata,
        }


# ==================== Archive Events ====================

@dataclass
class MemoryArchived:
    """Event emitted when a long-term memory is archived"""
    event_id: UUID
    memory_id: UUID
    agent_id: UUID
    importance_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_id": str(self.event_id),
            "timestamp": self.timestamp.isoformat(),
            "memory_id": str(self.memory_id),
            "agent_id": str(self.agent_id),
            "importance_score": self.importance_score,
            "metadata": self.metadata,
        }
