"""
Memory Service with PostgreSQL + pgvector integration.

Manages multi-layer memory storage and retrieval for CognitionOS agents.
"""

import os

# Add shared libs to path

from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from uuid import UUID
import math

from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, Field, validator
from sqlalchemy import select, and_, or_, text
from sqlalchemy.ext.asyncio import AsyncSession

from shared.libs.config import MemoryServiceConfig, load_config
from shared.libs.logger import setup_logger, get_contextual_logger
from shared.libs.models import (
    MemoryType, MemoryScope, ErrorResponse
)
from shared.libs.middleware import (
    TracingMiddleware,
    LoggingMiddleware,
    ErrorHandlingMiddleware
)

from database import (
    get_db, init_db, check_db_health,
    Memory as DBMemory
)

# Import AI Runtime client for embeddings
from ai_runtime_client import AIRuntimeClient


# Configuration
config = load_config(MemoryServiceConfig)
logger = setup_logger(__name__, level=config.log_level)

# FastAPI app
app = FastAPI(
    title="CognitionOS Memory Service",
    version=config.service_version,
    description="Multi-layer memory storage and retrieval with PostgreSQL + pgvector"
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


class MemoryResponse(BaseModel):
    """Response model for memories."""
    id: UUID
    user_id: UUID
    content: str
    memory_type: MemoryType
    scope: MemoryScope
    metadata: Dict[str, Any]
    source: Optional[str]
    confidence: float
    access_count: int
    is_sensitive: bool
    created_at: datetime
    updated_at: datetime
    accessed_at: datetime
    relevance_score: Optional[float] = None

    class Config:
        from_attributes = True


# ============================================================================
# Memory Storage with PostgreSQL
# ============================================================================

class MemoryStore:
    """
    Memory storage and retrieval engine with PostgreSQL + pgvector.
    """

    def __init__(self):
        self.logger = get_contextual_logger(__name__, component="MemoryStore")
        # Initialize AI Runtime client for embeddings
        self.ai_client = AIRuntimeClient()

    async def store(
        self,
        db: AsyncSession,
        user_id: UUID,
        content: str,
        memory_type: MemoryType,
        scope: MemoryScope,
        metadata: Dict[str, Any],
        source: str,
        confidence: float,
        is_sensitive: bool
    ) -> DBMemory:
        """
        Store a new memory in PostgreSQL.

        Args:
            db: Database session
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

        # Generate embedding using AI Runtime
        try:
            embeddings = await self.ai_client.generate_embeddings(
                texts=[content],
                user_id=user_id
            )
            embedding = embeddings[0] if embeddings else None
        except Exception as e:
            self.logger.warning(f"Failed to generate embedding via AI Runtime: {e}")
            # Fallback to simulated embedding
            embedding = self._generate_embedding(content)

        # Extract entities and keywords
        entities = self._extract_entities(content)
        keywords = self._extract_keywords(content)

        # Enrich metadata
        enriched_metadata = {
            **metadata,
            "entities": entities,
            "keywords": keywords
        }

        # Check for duplicates using vector similarity
        # For now, skip duplicate check to simplify initial implementation
        # In production, use: SELECT * FROM memories WHERE embedding <-> query_embedding < threshold

        # Create memory object
        memory = DBMemory(
            user_id=user_id,
            content=content,
            # embedding=embedding,  # Uncomment when pgvector is fully set up
            memory_type=memory_type,
            scope=scope,
            metadata=enriched_metadata,
            source=source,
            confidence=confidence,
            is_sensitive=is_sensitive
        )

        # Save to database
        db.add(memory)
        await db.commit()
        await db.refresh(memory)

        self.logger.info(
            "Memory stored",
            extra={"memory_id": str(memory.id)}
        )

        return memory

    async def retrieve(
        self,
        db: AsyncSession,
        user_id: UUID,
        query: str,
        k: int = 5,
        memory_types: Optional[List[MemoryType]] = None,
        min_confidence: float = 0.0
    ) -> List[MemoryResponse]:
        """
        Retrieve relevant memories using semantic search.

        Args:
            db: Database session
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

        # Generate query embedding using AI Runtime
        try:
            embeddings = await self.ai_client.generate_embeddings(
                texts=[query],
                user_id=user_id
            )
            query_embedding = embeddings[0] if embeddings else None
        except Exception as e:
            self.logger.warning(f"Failed to generate query embedding: {e}")
            query_embedding = self._generate_embedding(query)

        # Build query filters
        filters = [
            DBMemory.user_id == user_id,
            DBMemory.confidence >= min_confidence
        ]

        if memory_types:
            filters.append(DBMemory.memory_type.in_(memory_types))

        # For now, retrieve all matching memories and score in Python
        # In production with pgvector, use:
        # SELECT *, embedding <=> query_embedding AS distance
        # FROM memories WHERE ... ORDER BY distance LIMIT k

        stmt = select(DBMemory).where(and_(*filters)).order_by(DBMemory.created_at.desc()).limit(k * 3)
        result = await db.execute(stmt)
        candidates = result.scalars().all()

        # Calculate similarity scores
        scored_memories = []
        for memory in candidates:
            # In production, use actual vector similarity from database
            # For now, simulate with simple text matching
            similarity = self._text_similarity(query, memory.content)

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
            memory.accessed_at = datetime.utcnow()
            memory.access_count += 1

        await db.commit()

        # Convert to response models
        results = [
            MemoryResponse(
                **{
                    "id": mem.id,
                    "user_id": mem.user_id,
                    "content": mem.content,
                    "memory_type": mem.memory_type,
                    "scope": mem.scope,
                    "metadata": mem.metadata,
                    "source": mem.source,
                    "confidence": mem.confidence,
                    "access_count": mem.access_count,
                    "is_sensitive": mem.is_sensitive,
                    "created_at": mem.created_at,
                    "updated_at": mem.updated_at,
                    "accessed_at": mem.accessed_at,
                    "relevance_score": score
                }
            )
            for mem, score in top_memories
        ]

        self.logger.info(
            "Retrieved memories",
            extra={"count": len(results)}
        )

        return results

    async def update(
        self,
        db: AsyncSession,
        memory_id: UUID,
        new_content: str,
        confidence_adjustment: float = 0.0
    ) -> DBMemory:
        """Update existing memory."""
        memory = await db.get(DBMemory, memory_id)

        if not memory:
            raise ValueError(f"Memory not found: {memory_id}")

        # Generate new embedding using AI Runtime
        try:
            embeddings = await self.ai_client.generate_embeddings(
                texts=[new_content],
                user_id=memory.user_id
            )
            new_embedding = embeddings[0] if embeddings else None
        except Exception as e:
            self.logger.warning(f"Failed to generate embedding: {e}")
            new_embedding = self._generate_embedding(new_content)

        # Update memory
        memory.content = new_content
        # memory.embedding = new_embedding  # Uncomment when pgvector is set up
        memory.confidence = max(0.0, min(1.0, memory.confidence + confidence_adjustment))

        await db.commit()
        await db.refresh(memory)

        self.logger.info(
            "Memory updated",
            extra={"memory_id": str(memory_id)}
        )

        return memory

    async def delete(
        self,
        db: AsyncSession,
        memory_id: UUID
    ):
        """Delete a memory."""
        memory = await db.get(DBMemory, memory_id)

        if not memory:
            raise ValueError(f"Memory not found: {memory_id}")

        await db.delete(memory)
        await db.commit()

        self.logger.info("Memory deleted", extra={"memory_id": str(memory_id)})

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

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity (placeholder for vector similarity)."""
        # In production, this would use actual vector cosine similarity
        # For now, use simple word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text (simulated)."""
        # In production, use spaCy or similar NLP library
        words = text.split()
        entities = [w for w in words if w and w[0].isupper() and len(w) > 2]
        return list(set(entities))[:10]

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (simulated)."""
        # In production, use TF-IDF or similar
        words = text.lower().split()
        keywords = [w for w in words if len(w) > 5]
        return list(set(keywords))[:10]


# ============================================================================
# API Endpoints
# ============================================================================

memory_store = MemoryStore()


@app.post("/memories", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def store_memory(request: StoreMemoryRequest, db: AsyncSession = Depends(get_db)):
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
            db=db,
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
        return MemoryResponse(
            **{
                "id": memory.id,
                "user_id": memory.user_id,
                "content": memory.content,
                "memory_type": memory.memory_type,
                "scope": memory.scope,
                "metadata": memory.metadata,
                "source": memory.source,
                "confidence": memory.confidence,
                "access_count": memory.access_count,
                "is_sensitive": memory.is_sensitive,
                "created_at": memory.created_at,
                "updated_at": memory.updated_at,
                "accessed_at": memory.accessed_at
            }
        )

    except Exception as e:
        log.error("Failed to store memory", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store memory: {str(e)}"
        )


@app.post("/retrieve", response_model=List[MemoryResponse])
async def retrieve_memories(request: RetrieveMemoriesRequest, db: AsyncSession = Depends(get_db)):
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
            db=db,
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


@app.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get a specific memory by ID."""
    memory = await db.get(DBMemory, memory_id)

    if not memory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory not found: {memory_id}"
        )

    return MemoryResponse(
        **{
            "id": memory.id,
            "user_id": memory.user_id,
            "content": memory.content,
            "memory_type": memory.memory_type,
            "scope": memory.scope,
            "metadata": memory.metadata,
            "source": memory.source,
            "confidence": memory.confidence,
            "access_count": memory.access_count,
            "is_sensitive": memory.is_sensitive,
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
            "accessed_at": memory.accessed_at
        }
    )


@app.put("/memories/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: UUID,
    content: str,
    confidence_adjustment: float = 0.0,
    db: AsyncSession = Depends(get_db)
):
    """Update a memory's content and confidence."""
    try:
        memory = await memory_store.update(
            db=db,
            memory_id=memory_id,
            new_content=content,
            confidence_adjustment=confidence_adjustment
        )
        return MemoryResponse(
            **{
                "id": memory.id,
                "user_id": memory.user_id,
                "content": memory.content,
                "memory_type": memory.memory_type,
                "scope": memory.scope,
                "metadata": memory.metadata,
                "source": memory.source,
                "confidence": memory.confidence,
                "access_count": memory.access_count,
                "is_sensitive": memory.is_sensitive,
                "created_at": memory.created_at,
                "updated_at": memory.updated_at,
                "accessed_at": memory.accessed_at
            }
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: UUID, db: AsyncSession = Depends(get_db)):
    """Delete a memory."""
    try:
        await memory_store.delete(db=db, memory_id=memory_id)
        return {"message": "Memory deleted successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    """Health check endpoint."""
    # Check database health
    db_healthy = await check_db_health()

    # Count total memories
    stmt = select(DBMemory)
    result = await db.execute(stmt)
    total_memories = len(result.scalars().all())

    return {
        "status": "healthy" if db_healthy else "unhealthy",
        "service": "memory-service",
        "version": config.service_version,
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected" if db_healthy else "disconnected",
        "total_memories": total_memories
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

    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Memory Service shutting down")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.host,
        port=getattr(config, 'port', 8004),
        log_level=config.log_level.lower()
    )
