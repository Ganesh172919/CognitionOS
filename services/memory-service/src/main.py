"""
Memory Service.

Manages multi-layer memory storage and retrieval for CognitionOS agents.
"""

import sys
import os

# Add shared libs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from uuid import UUID, uuid4
import math

from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, Field, validator

from shared.libs.config import MemoryServiceConfig, load_config
from shared.libs.logger import setup_logger, get_contextual_logger
from shared.libs.models import (
    Memory, MemoryType, MemoryScope, ShortTermMemory,
    EpisodicMemory, ErrorResponse
)
from shared.libs.middleware import (
    TracingMiddleware,
    LoggingMiddleware,
    ErrorHandlingMiddleware
)


# Configuration
config = load_config(MemoryServiceConfig)
logger = setup_logger(__name__, level=config.log_level)

# FastAPI app
app = FastAPI(
    title="CognitionOS Memory Service",
    version=config.service_version,
    description="Multi-layer memory storage and retrieval"
)

# Add middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(TracingMiddleware)


# ============================================================================
# Request/Response Models
# ============================================================================

class StoreMemoryRequest(BaseModel):
    """Request to store a new memory."""
    user_id: UUID
    content: str
    memory_type: MemoryType
    scope: MemoryScope = MemoryScope.USER
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str = "user_input"
    confidence: float = 1.0
    is_sensitive: bool = False

    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v


class RetrieveMemoriesRequest(BaseModel):
    """Request to retrieve relevant memories."""
    user_id: UUID
    query: str
    k: int = Field(default=5, ge=1, le=50)
    memory_types: Optional[List[MemoryType]] = None
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class EmbedRequest(BaseModel):
    """Request to generate embeddings."""
    texts: List[str]


class MemoryResponse(BaseModel):
    """Memory with relevance score."""
    memory: Memory
    relevance_score: float


# ============================================================================
# In-Memory Storage (Replace with Database)
# ============================================================================

# In production, use PostgreSQL + pgvector
memories_db: Dict[UUID, Memory] = {}
short_term_db: Dict[UUID, ShortTermMemory] = {}
episodic_db: Dict[UUID, EpisodicMemory] = {}


# ============================================================================
# Memory Storage
# ============================================================================

class MemoryStore:
    """
    Memory storage and retrieval engine.

    In production, this would use PostgreSQL with pgvector extension.
    """

    def __init__(self):
        self.logger = get_contextual_logger(__name__, component="MemoryStore")

    async def store(
        self,
        user_id: UUID,
        content: str,
        memory_type: MemoryType,
        scope: MemoryScope,
        metadata: Dict[str, Any],
        source: str,
        confidence: float,
        is_sensitive: bool
    ) -> Memory:
        """
        Store a new memory.

        Args:
            user_id: User who owns this memory
            content: Memory content
            memory_type: Type of memory
            scope: USER or SESSION
            metadata: Additional metadata
            source: Where this memory came from
            confidence: Confidence score (0.0 to 1.0)
            is_sensitive: Whether memory contains PII

        Returns:
            Stored memory with generated embedding
        """
        self.logger.info(
            "Storing memory",
            extra={
                "user_id": str(user_id),
                "type": memory_type.value,
                "scope": scope.value
            }
        )

        # Generate embedding (simulated for now)
        # In production: embedding = await openai.Embedding.create(input=content)
        embedding = self._generate_embedding(content)

        # Extract entities and keywords (simulated)
        entities = self._extract_entities(content)
        keywords = self._extract_keywords(content)

        # Check for duplicates
        similar = await self._find_similar(user_id, embedding, threshold=0.95)
        if similar:
            self.logger.info("Similar memory exists, updating instead of creating")
            return await self.update(similar[0].id, content, confidence)

        # Create memory
        memory = Memory(
            user_id=user_id,
            content=content,
            embedding=embedding,
            memory_type=memory_type,
            scope=scope,
            metadata={
                **metadata,
                "entities": entities,
                "keywords": keywords
            },
            source=source,
            confidence=confidence,
            is_sensitive=is_sensitive
        )

        # Store in database
        memories_db[memory.id] = memory

        self.logger.info(
            "Memory stored",
            extra={"memory_id": str(memory.id)}
        )

        return memory

    async def retrieve(
        self,
        user_id: UUID,
        query: str,
        k: int = 5,
        memory_types: Optional[List[MemoryType]] = None,
        min_confidence: float = 0.0
    ) -> List[MemoryResponse]:
        """
        Retrieve relevant memories.

        Args:
            user_id: User ID
            query: Search query
            k: Number of results
            memory_types: Filter by memory types
            min_confidence: Minimum confidence threshold

        Returns:
            List of memories with relevance scores
        """
        self.logger.info(
            "Retrieving memories",
            extra={
                "user_id": str(user_id),
                "query": query[:50],
                "k": k
            }
        )

        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        # Filter by user and type
        candidates = [
            mem for mem in memories_db.values()
            if mem.user_id == user_id
            and not mem.deleted
            and (not memory_types or mem.memory_type in memory_types)
            and mem.confidence >= min_confidence
        ]

        # Calculate similarity scores
        scored_memories = []
        for memory in candidates:
            similarity = self._cosine_similarity(query_embedding, memory.embedding)

            # Apply time decay
            age_days = (datetime.utcnow() - memory.created_at).days
            decay_factor = math.exp(-age_days / 30)  # 30-day half-life

            # Apply access frequency boost
            frequency_boost = math.log(1 + memory.access_count) * 0.1

            # Final score
            score = similarity * decay_factor * (1 + frequency_boost)

            scored_memories.append((memory, score))

        # Sort by score and take top k
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        top_memories = scored_memories[:k]

        # Update access statistics
        for memory, score in top_memories:
            memory.last_accessed = datetime.utcnow()
            memory.access_count += 1

        results = [
            MemoryResponse(memory=mem, relevance_score=score)
            for mem, score in top_memories
        ]

        self.logger.info(
            "Retrieved memories",
            extra={"count": len(results)}
        )

        return results

    async def update(
        self,
        memory_id: UUID,
        new_content: str,
        confidence_adjustment: float = 0.0
    ) -> Memory:
        """Update existing memory."""
        if memory_id not in memories_db:
            raise ValueError(f"Memory not found: {memory_id}")

        memory = memories_db[memory_id]

        # Generate new embedding
        new_embedding = self._generate_embedding(new_content)

        # Update memory
        memory.content = new_content
        memory.embedding = new_embedding
        memory.version += 1
        memory.confidence = max(0.0, min(1.0, memory.confidence + confidence_adjustment))
        memory.updated_at = datetime.utcnow()

        self.logger.info(
            "Memory updated",
            extra={"memory_id": str(memory_id), "version": memory.version}
        )

        return memory

    async def delete(self, memory_id: UUID, soft_delete: bool = True):
        """Delete a memory."""
        if memory_id not in memories_db:
            raise ValueError(f"Memory not found: {memory_id}")

        if soft_delete:
            memories_db[memory_id].deleted = True
            self.logger.info("Memory soft-deleted", extra={"memory_id": str(memory_id)})
        else:
            del memories_db[memory_id]
            self.logger.info("Memory hard-deleted", extra={"memory_id": str(memory_id)})

    async def _find_similar(
        self,
        user_id: UUID,
        embedding: List[float],
        threshold: float = 0.95
    ) -> List[Memory]:
        """Find similar memories."""
        similar = []
        for memory in memories_db.values():
            if memory.user_id == user_id and not memory.deleted:
                similarity = self._cosine_similarity(embedding, memory.embedding)
                if similarity >= threshold:
                    similar.append(memory)
        return similar

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        In production, this would call OpenAI API:
        response = await openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']
        """
        # Simulated embedding (1536 dimensions for ada-002)
        import random
        random.seed(hash(text) % (2**32))
        return [random.random() for _ in range(1536)]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text (simulated)."""
        # In production, use spaCy or similar NLP library
        # For now, just extract capitalized words
        words = text.split()
        entities = [w for w in words if w and w[0].isupper() and len(w) > 2]
        return list(set(entities))[:10]  # Limit to 10

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (simulated)."""
        # In production, use TF-IDF or similar
        # For now, just extract longer words
        words = text.lower().split()
        keywords = [w for w in words if len(w) > 5]
        return list(set(keywords))[:10]  # Limit to 10


# ============================================================================
# API Endpoints
# ============================================================================

memory_store = MemoryStore()


@app.post("/memories", response_model=Memory, status_code=status.HTTP_201_CREATED)
async def store_memory(request: StoreMemoryRequest):
    """
    Store a new memory.

    Creates a memory with vector embedding for semantic search.
    """
    log = get_contextual_logger(
        __name__,
        action="store_memory",
        user_id=str(request.user_id)
    )

    try:
        memory = await memory_store.store(
            user_id=request.user_id,
            content=request.content,
            memory_type=request.memory_type,
            scope=request.scope,
            metadata=request.metadata,
            source=request.source,
            confidence=request.confidence,
            is_sensitive=request.is_sensitive
        )

        log.info("Memory stored successfully")
        return memory

    except Exception as e:
        log.error("Failed to store memory", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store memory: {str(e)}"
        )


@app.post("/retrieve", response_model=List[MemoryResponse])
async def retrieve_memories(request: RetrieveMemoriesRequest):
    """
    Retrieve relevant memories.

    Uses semantic search to find memories relevant to the query.
    """
    log = get_contextual_logger(
        __name__,
        action="retrieve_memories",
        user_id=str(request.user_id)
    )

    try:
        results = await memory_store.retrieve(
            user_id=request.user_id,
            query=request.query,
            k=request.k,
            memory_types=request.memory_types,
            min_confidence=request.min_confidence
        )

        log.info(
            "Memories retrieved",
            extra={"count": len(results)}
        )

        return results

    except Exception as e:
        log.error("Failed to retrieve memories", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memories: {str(e)}"
        )


@app.get("/memories/{memory_id}", response_model=Memory)
async def get_memory(memory_id: UUID):
    """Get a specific memory by ID."""
    if memory_id not in memories_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory not found: {memory_id}"
        )

    memory = memories_db[memory_id]
    if memory.deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory has been deleted: {memory_id}"
        )

    return memory


@app.put("/memories/{memory_id}", response_model=Memory)
async def update_memory(
    memory_id: UUID,
    content: str,
    confidence_adjustment: float = 0.0
):
    """Update a memory's content and confidence."""
    try:
        memory = await memory_store.update(
            memory_id=memory_id,
            new_content=content,
            confidence_adjustment=confidence_adjustment
        )
        return memory
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: UUID, hard_delete: bool = False):
    """Delete a memory (soft delete by default)."""
    try:
        await memory_store.delete(memory_id, soft_delete=not hard_delete)
        return {"message": "Memory deleted successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "memory-service",
        "version": config.service_version,
        "timestamp": datetime.utcnow().isoformat(),
        "total_memories": len([m for m in memories_db.values() if not m.deleted])
    }


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize service on startup."""
    logger.info(
        "Memory Service starting",
        extra={
            "version": config.service_version,
            "environment": config.environment
        }
    )

    # In production, initialize database connection and pgvector


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Memory Service shutting down")

    # In production, close database connections


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.host,
        port=getattr(config, 'port', 8004),
        log_level=config.log_level.lower()
    )
