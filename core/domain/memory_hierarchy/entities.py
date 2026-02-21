"""
Memory Hierarchy Domain - Entities

Pure domain entities for hierarchical memory (L1/L2/L3) functionality.
NO external dependencies except Python stdlib.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4


class MemoryTier(str, Enum):
    """Memory tier levels"""
    L1_WORKING = "l1_working"
    L2_EPISODIC = "l2_episodic"
    L3_LONGTERM = "l3_longterm"


class MemoryType(str, Enum):
    """Types of working memory"""
    OBSERVATION = "observation"
    ACTION = "action"
    RESULT = "result"
    REASONING = "reasoning"
    METADATA = "metadata"
    FACT = "fact"


class KnowledgeType(str, Enum):
    """Types of long-term knowledge"""
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CONCEPTUAL = "conceptual"
    STRATEGIC = "strategic"
    SEMANTIC = "semantic"


class SourceType(str, Enum):
    """Source of long-term memory"""
    EPISODIC_COMPRESSION = "episodic_compression"
    MANUAL_ENTRY = "manual_entry"
    LEARNED_PATTERN = "learned_pattern"


# ==================== Value Objects ====================

@dataclass(frozen=True)
class MemoryEmbedding:
    """Memory embedding value object"""
    vector: List[float]
    model: str
    dimensions: int = 0

    def __post_init__(self):
        """Validate embedding invariants"""
        # If dimensions is not set, infer from vector length
        if self.dimensions == 0:
            object.__setattr__(self, 'dimensions', len(self.vector))
        if len(self.vector) != self.dimensions:
            raise ValueError(f"Vector length {len(self.vector)} does not match dimensions {self.dimensions}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "vector": self.vector,
            "model": self.model,
            "dimensions": self.dimensions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEmbedding":
        """Create from dictionary"""
        vector = data["vector"]
        return cls(
            vector=vector,
            model=data.get("model", "unknown"),
            dimensions=data.get("dimensions", data.get("dimension", len(vector))),
        )


# ==================== Entities ====================

@dataclass
class WorkingMemory:
    """
    L1 Working Memory entity.
    
    Short-term memory for active workflow execution context.
    Design principles:
    - Fast access for current operations
    - LRU eviction when capacity reached
    - Promotion to L2 based on importance and access patterns
    """
    id: UUID
    agent_id: UUID
    workflow_execution_id: UUID
    
    # Content
    content: str
    embedding: Optional[MemoryEmbedding]
    
    # Importance and access tracking
    importance_score: float
    access_count: int = 0
    last_accessed_at: datetime = field(default_factory=datetime.utcnow)
    
    # Lifecycle
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    # Classification
    memory_type: MemoryType = MemoryType.OBSERVATION
    tags: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate working memory invariants"""
        if not 0.0 <= self.importance_score <= 1.0:
            raise ValueError("Importance score must be between 0 and 1")
        
        if self.access_count < 0:
            raise ValueError("Access count cannot be negative")

    @classmethod
    def create(
        cls,
        agent_id: UUID,
        workflow_execution_id: Optional[UUID] = None,
        content: str = "",
        importance_score: float = 0.5,
        embedding: Optional[Union[MemoryEmbedding, List[float]]] = None,
        memory_type: MemoryType = MemoryType.OBSERVATION,
        tags: Optional[List[str]] = None,
        expires_at: Optional[datetime] = None,
        ttl_hours: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "WorkingMemory":
        """
        Factory method to create a new working memory.
        
        Args:
            agent_id: Agent ID
            workflow_execution_id: Workflow execution ID (optional)
            content: Memory content
            importance_score: Importance score (0-1)
            embedding: Optional embedding (MemoryEmbedding or raw vector)
            memory_type: Type of memory
            tags: Optional tags
            expires_at: Optional expiration time
            ttl_hours: Optional TTL in hours (alternative to expires_at)
            metadata: Optional metadata
            
        Returns:
            New WorkingMemory instance
        """
        if ttl_hours is not None and expires_at is None:
            expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)
        if isinstance(embedding, list):
            embedding = MemoryEmbedding(vector=embedding, model="unknown")
        if workflow_execution_id is None:
            workflow_execution_id = uuid4()
        return cls(
            id=uuid4(),
            agent_id=agent_id,
            workflow_execution_id=workflow_execution_id,
            content=content,
            embedding=embedding,
            importance_score=importance_score,
            memory_type=memory_type,
            tags=tags or [],
            expires_at=expires_at,
            metadata=metadata or {},
        )

    def update_access(self) -> None:
        """Update access tracking when memory is accessed"""
        self.access_count += 1
        self.last_accessed_at = datetime.utcnow()

    def update_importance_score(self, new_score: float) -> None:
        """
        Update importance score with validation.
        
        Args:
            new_score: New importance score (0-1)
            
        Raises:
            ValueError: If score is not between 0 and 1
        """
        if not 0.0 <= new_score <= 1.0:
            raise ValueError(f"Importance score must be between 0 and 1, got {new_score}")
        self.importance_score = new_score

    def calculate_age_hours(self) -> float:
        """Calculate age of memory in hours"""
        age_delta = datetime.utcnow() - self.created_at
        return age_delta.total_seconds() / 3600

    def should_promote_to_l2(
        self,
        min_importance: float = 0.7,
        min_access_count: int = 3,
        min_age_hours: float = 0.0,
    ) -> bool:
        """
        Determine if memory should be promoted to L2.
        
        Args:
            min_importance: Minimum importance score
            min_access_count: Minimum access count
            min_age_hours: Minimum age in hours
            
        Returns:
            True if should be promoted
        """
        return (
            self.importance_score >= min_importance
            and self.access_count >= min_access_count
            and self.calculate_age_hours() >= min_age_hours
        )

    def is_expired(self) -> bool:
        """Check if memory has expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "workflow_execution_id": str(self.workflow_execution_id),
            "content": self.content,
            "embedding": self.embedding.to_dict() if self.embedding else None,
            "importance_score": self.importance_score,
            "access_count": self.access_count,
            "last_accessed_at": self.last_accessed_at.isoformat(),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "memory_type": self.memory_type.value,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkingMemory":
        """Create from dictionary"""
        return cls(
            id=UUID(data["id"]),
            agent_id=UUID(data["agent_id"]),
            workflow_execution_id=UUID(data["workflow_execution_id"]),
            content=data["content"],
            embedding=MemoryEmbedding.from_dict(data["embedding"]) if data.get("embedding") else None,
            importance_score=data["importance_score"],
            access_count=data.get("access_count", 0),
            last_accessed_at=datetime.fromisoformat(data["last_accessed_at"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            memory_type=MemoryType(data["memory_type"]),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EpisodicMemory:
    """
    L2 Episodic Memory entity.
    
    Compressed clusters of related working memories.
    Design principles:
    - Semantic clustering of related L1 memories
    - LLM-based compression and summarization
    - Temporal coherence within episodes
    """
    id: UUID
    agent_id: UUID
    cluster_id: str
    
    # Compressed content
    summary: str
    embedding: Optional[MemoryEmbedding]
    
    # Compression metadata
    compression_ratio: float
    source_memory_ids: List[UUID]
    source_memory_count: int
    
    # Importance
    importance_score: float
    
    # Temporal context
    temporal_period: Dict[str, datetime]
    
    # Lifecycle
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate episodic memory invariants"""
        if not 0.0 <= self.importance_score <= 1.0:
            raise ValueError("Importance score must be between 0 and 1")
        
        if self.compression_ratio < 0:
            raise ValueError("Compression ratio cannot be negative")
        
        if self.source_memory_count != len(self.source_memory_ids):
            raise ValueError("Source memory count must match IDs length")
        
        if "start" not in self.temporal_period or "end" not in self.temporal_period:
            raise ValueError("Temporal period must have start and end")

    @classmethod
    def create(
        cls,
        agent_id: UUID,
        summary: str,
        cluster_id: Optional[str] = None,
        source_memory_ids: Optional[List[UUID]] = None,
        temporal_start: Optional[datetime] = None,
        temporal_end: Optional[datetime] = None,
        importance_score: float = 0.6,
        embedding: Optional[Union[MemoryEmbedding, List[float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "EpisodicMemory":
        """
        Factory method to create a new episodic memory.
        
        Args:
            agent_id: Agent ID
            summary: Compressed summary
            cluster_id: Cluster identifier (optional, auto-generated if not provided)
            source_memory_ids: Source working memory IDs
            temporal_start: Start of temporal period
            temporal_end: End of temporal period
            importance_score: Importance score (0-1)
            embedding: Optional embedding (MemoryEmbedding or raw vector)
            metadata: Optional metadata
            
        Returns:
            New EpisodicMemory instance
        """
        if isinstance(embedding, list):
            embedding = MemoryEmbedding(vector=embedding, model="unknown")
        if cluster_id is None:
            cluster_id = str(uuid4())
        if source_memory_ids is None:
            source_memory_ids = []
        if temporal_start is None:
            temporal_start = datetime.utcnow()
        if temporal_end is None:
            temporal_end = datetime.utcnow()
        return cls(
            id=uuid4(),
            agent_id=agent_id,
            cluster_id=cluster_id,
            summary=summary,
            embedding=embedding,
            compression_ratio=0.0,
            source_memory_ids=source_memory_ids,
            source_memory_count=len(source_memory_ids),
            importance_score=importance_score,
            temporal_period={"start": temporal_start, "end": temporal_end},
            metadata=metadata or {},
        )

    @property
    def temporal_start(self) -> Optional[datetime]:
        """Get temporal period start"""
        return self.temporal_period.get("start")

    @property
    def temporal_end(self) -> Optional[datetime]:
        """Get temporal period end"""
        return self.temporal_period.get("end")

    def add_source_memory(self, memory_id: UUID) -> None:
        """Add a source memory ID to this episode"""
        if memory_id not in self.source_memory_ids:
            self.source_memory_ids.append(memory_id)
            self.source_memory_count = len(self.source_memory_ids)
            self.updated_at = datetime.utcnow()

    def calculate_compression_ratio(self, original_total_length: int, compressed_length: Optional[int] = None) -> float:
        """
        Calculate compression ratio based on original content.
        
        Args:
            original_total_length: Total length of source memories
            compressed_length: Length of compressed content (uses summary length if not provided)
            
        Returns:
            Compression ratio (compressed/original)
        """
        if original_total_length == 0:
            return 0.0
        
        if compressed_length is None:
            compressed_length = len(self.summary)
        
        self.compression_ratio = compressed_length / original_total_length
        return self.compression_ratio

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "cluster_id": self.cluster_id,
            "summary": self.summary,
            "embedding": self.embedding.to_dict() if self.embedding else None,
            "compression_ratio": self.compression_ratio,
            "source_memory_ids": [str(mid) for mid in self.source_memory_ids],
            "source_memory_count": self.source_memory_count,
            "importance_score": self.importance_score,
            "temporal_period": {
                "start": self.temporal_period["start"].isoformat(),
                "end": self.temporal_period["end"].isoformat(),
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodicMemory":
        """Create from dictionary"""
        return cls(
            id=UUID(data["id"]),
            agent_id=UUID(data["agent_id"]),
            cluster_id=data["cluster_id"],
            summary=data["summary"],
            embedding=MemoryEmbedding.from_dict(data["embedding"]) if data.get("embedding") else None,
            compression_ratio=data["compression_ratio"],
            source_memory_ids=[UUID(mid) for mid in data["source_memory_ids"]],
            source_memory_count=data["source_memory_count"],
            importance_score=data["importance_score"],
            temporal_period={
                "start": datetime.fromisoformat(data["temporal_period"]["start"]),
                "end": datetime.fromisoformat(data["temporal_period"]["end"]),
            },
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class LongTermMemory:
    """
    L3 Long-Term Memory entity.
    
    Persistent knowledge extracted from episodic memories.
    Design principles:
    - High-value, durable knowledge
    - Archival of important patterns and learnings
    - Manual or automated knowledge extraction
    """
    id: UUID
    agent_id: UUID
    
    # Knowledge content
    knowledge_type: KnowledgeType
    title: str
    content: str
    embedding: Optional[MemoryEmbedding]
    
    # Importance and source
    importance_score: float
    source_type: SourceType
    source_references: List[UUID] = field(default_factory=list)
    
    # Lifecycle
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    archived_at: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate long-term memory invariants"""
        if not 0.0 <= self.importance_score <= 1.0:
            raise ValueError("Importance score must be between 0 and 1")
        
        if not self.title:
            raise ValueError("Title cannot be empty")
        
        if not self.content:
            raise ValueError("Content cannot be empty")

    @classmethod
    def create(
        cls,
        agent_id: UUID,
        knowledge_type: KnowledgeType,
        title: str,
        content: str,
        source_type: SourceType = SourceType.MANUAL_ENTRY,
        importance_score: float = 0.8,
        embedding: Optional[Union[MemoryEmbedding, List[float]]] = None,
        source_references: Optional[List[UUID]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "LongTermMemory":
        """
        Factory method to create a new long-term memory.
        
        Args:
            agent_id: Agent ID
            knowledge_type: Type of knowledge
            title: Knowledge title
            content: Knowledge content
            source_type: Source of knowledge
            importance_score: Importance score (0-1)
            embedding: Optional embedding (MemoryEmbedding or raw vector)
            source_references: Optional source reference IDs
            metadata: Optional metadata
            
        Returns:
            New LongTermMemory instance
        """
        if isinstance(embedding, list):
            embedding = MemoryEmbedding(vector=embedding, model="unknown")
        return cls(
            id=uuid4(),
            agent_id=agent_id,
            knowledge_type=knowledge_type,
            title=title,
            content=content,
            embedding=embedding,
            importance_score=importance_score,
            source_type=source_type,
            source_references=source_references or [],
            metadata=metadata or {},
        )

    def mark_archived(self) -> None:
        """Mark this memory as archived"""
        if not self.archived_at:
            self.archived_at = datetime.utcnow()
            self.updated_at = datetime.utcnow()

    def is_archived(self) -> bool:
        """Check if memory is archived"""
        return self.archived_at is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "knowledge_type": self.knowledge_type.value,
            "title": self.title,
            "content": self.content,
            "embedding": self.embedding.to_dict() if self.embedding else None,
            "importance_score": self.importance_score,
            "source_type": self.source_type.value,
            "source_references": [str(ref) for ref in self.source_references],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "archived_at": self.archived_at.isoformat() if self.archived_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LongTermMemory":
        """Create from dictionary"""
        return cls(
            id=UUID(data["id"]),
            agent_id=UUID(data["agent_id"]),
            knowledge_type=KnowledgeType(data["knowledge_type"]),
            title=data["title"],
            content=data["content"],
            embedding=MemoryEmbedding.from_dict(data["embedding"]) if data.get("embedding") else None,
            importance_score=data["importance_score"],
            source_type=SourceType(data["source_type"]),
            source_references=[UUID(ref) for ref in data.get("source_references", [])],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            archived_at=datetime.fromisoformat(data["archived_at"]) if data.get("archived_at") else None,
            metadata=data.get("metadata", {}),
        )
