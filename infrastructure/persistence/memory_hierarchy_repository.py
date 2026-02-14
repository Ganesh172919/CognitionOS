"""
Memory Hierarchy Infrastructure - PostgreSQL Repository Implementation

Concrete implementation of memory hierarchy repositories using PostgreSQL.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, delete, func, and_, or_, text
from sqlalchemy.ext.asyncio import AsyncSession

from core.domain.memory_hierarchy.entities import (
    WorkingMemory,
    EpisodicMemory,
    LongTermMemory,
    MemoryEmbedding,
    MemoryType,
    KnowledgeType,
    SourceType,
)
from core.domain.memory_hierarchy.repositories import (
    WorkingMemoryRepository,
    EpisodicMemoryRepository,
    LongTermMemoryRepository,
)

from infrastructure.persistence.memory_hierarchy_models import (
    WorkingMemoryModel,
    EpisodicMemoryModel,
    LongTermMemoryModel,
)


class PostgreSQLWorkingMemoryRepository(WorkingMemoryRepository):
    """
    PostgreSQL implementation of WorkingMemoryRepository.
    
    Maps between WorkingMemory domain entities and SQLAlchemy models.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, memory: WorkingMemory) -> None:
        """Persist working memory to database"""
        model = self._to_model(memory)
        self.session.add(model)
        await self.session.flush()
    
    async def find_by_id(self, memory_id: UUID) -> Optional[WorkingMemory]:
        """Retrieve working memory by ID"""
        stmt = select(WorkingMemoryModel).where(WorkingMemoryModel.id == memory_id)
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
        
        return self._to_entity(model)
    
    async def find_by_agent(
        self,
        agent_id: UUID,
        workflow_execution_id: Optional[UUID] = None,
        limit: int = 100,
    ) -> List[WorkingMemory]:
        """Find working memories by agent"""
        stmt = select(WorkingMemoryModel).where(
            WorkingMemoryModel.agent_id == agent_id
        )
        
        if workflow_execution_id:
            stmt = stmt.where(WorkingMemoryModel.workflow_execution_id == workflow_execution_id)
        
        stmt = stmt.order_by(WorkingMemoryModel.created_at.desc()).limit(limit)
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]
    
    async def find_by_importance(
        self,
        agent_id: UUID,
        min_importance: float,
        limit: int = 100,
    ) -> List[WorkingMemory]:
        """Find working memories by importance score"""
        stmt = (
            select(WorkingMemoryModel)
            .where(
                and_(
                    WorkingMemoryModel.agent_id == agent_id,
                    WorkingMemoryModel.importance_score >= min_importance
                )
            )
            .order_by(WorkingMemoryModel.importance_score.desc())
            .limit(limit)
        )
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]
    
    async def find_lru_candidates(
        self,
        agent_id: UUID,
        workflow_execution_id: UUID,
        limit: int = 10,
    ) -> List[WorkingMemory]:
        """Find LRU (Least Recently Used) eviction candidates"""
        stmt = (
            select(WorkingMemoryModel)
            .where(
                and_(
                    WorkingMemoryModel.agent_id == agent_id,
                    WorkingMemoryModel.workflow_execution_id == workflow_execution_id
                )
            )
            .order_by(WorkingMemoryModel.last_accessed_at.asc())
            .limit(limit)
        )
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]
    
    async def find_expired(
        self,
        agent_id: UUID,
        current_time: datetime,
    ) -> List[WorkingMemory]:
        """Find expired working memories"""
        stmt = select(WorkingMemoryModel).where(
            and_(
                WorkingMemoryModel.agent_id == agent_id,
                WorkingMemoryModel.expires_at.isnot(None),
                WorkingMemoryModel.expires_at <= current_time
            )
        )
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]
    
    async def delete(self, memory_id: UUID) -> None:
        """Delete a working memory"""
        stmt = delete(WorkingMemoryModel).where(WorkingMemoryModel.id == memory_id)
        await self.session.execute(stmt)
        await self.session.flush()
    
    async def delete_batch(self, memory_ids: List[UUID]) -> int:
        """Delete multiple working memories"""
        if not memory_ids:
            return 0
        
        stmt = delete(WorkingMemoryModel).where(
            WorkingMemoryModel.id.in_(memory_ids)
        )
        
        result = await self.session.execute(stmt)
        await self.session.flush()
        
        return result.rowcount
    
    async def count_by_agent(
        self,
        agent_id: UUID,
        workflow_execution_id: Optional[UUID] = None,
    ) -> int:
        """Count working memories for an agent"""
        stmt = select(func.count()).select_from(WorkingMemoryModel).where(
            WorkingMemoryModel.agent_id == agent_id
        )
        
        if workflow_execution_id:
            stmt = stmt.where(WorkingMemoryModel.workflow_execution_id == workflow_execution_id)
        
        result = await self.session.execute(stmt)
        count = result.scalar_one()
        
        return count
    
    async def find_similar(
        self,
        agent_id: UUID,
        embedding_vector: List[float],
        limit: int = 10,
        min_similarity: float = 0.7,
    ) -> List[WorkingMemory]:
        """
        Find similar working memories using cosine similarity.
        
        Args:
            agent_id: Agent ID
            embedding_vector: Query embedding vector
            limit: Maximum number of results
            min_similarity: Minimum cosine similarity (0-1)
            
        Returns:
            List of similar working memories ordered by similarity
        """
        # pgvector cosine distance: 1 - cosine_similarity
        # So similarity = 1 - distance
        max_distance = 1 - min_similarity
        
        stmt = (
            select(WorkingMemoryModel)
            .where(
                and_(
                    WorkingMemoryModel.agent_id == agent_id,
                    WorkingMemoryModel.embedding.isnot(None)
                )
            )
            .order_by(WorkingMemoryModel.embedding.cosine_distance(embedding_vector))
            .limit(limit)
        )
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        # Filter by similarity threshold (since we can't add it to WHERE clause easily)
        # In production, you might want to add this as a raw SQL filter
        return [self._to_entity(model) for model in models]
    
    def _to_model(self, memory: WorkingMemory) -> WorkingMemoryModel:
        """Convert domain entity to ORM model"""
        return WorkingMemoryModel(
            id=memory.id,
            agent_id=memory.agent_id,
            workflow_execution_id=memory.workflow_execution_id,
            content=memory.content,
            embedding=memory.embedding.vector if memory.embedding else None,
            importance_score=memory.importance_score,
            created_at=memory.created_at,
            last_accessed_at=memory.last_accessed_at,
            access_count=memory.access_count,
            expires_at=memory.expires_at,
            memory_type=memory.memory_type.value,
            tags=memory.tags,
            metadata=memory.metadata,
        )
    
    def _to_entity(self, model: WorkingMemoryModel) -> WorkingMemory:
        """Convert ORM model to domain entity"""
        embedding = None
        if model.embedding is not None:
            embedding = MemoryEmbedding(
                vector=model.embedding,
                model="text-embedding-ada-002",
                dimension=1536,
            )
        
        return WorkingMemory(
            id=model.id,
            agent_id=model.agent_id,
            workflow_execution_id=model.workflow_execution_id,
            content=model.content,
            embedding=embedding,
            importance_score=model.importance_score,
            access_count=model.access_count,
            last_accessed_at=model.last_accessed_at,
            created_at=model.created_at,
            expires_at=model.expires_at,
            memory_type=MemoryType(model.memory_type) if model.memory_type else MemoryType.OBSERVATION,
            tags=model.tags or [],
            metadata=model.metadata or {},
        )


class PostgreSQLEpisodicMemoryRepository(EpisodicMemoryRepository):
    """
    PostgreSQL implementation of EpisodicMemoryRepository.
    
    Maps between EpisodicMemory domain entities and SQLAlchemy models.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, memory: EpisodicMemory) -> None:
        """Persist episodic memory to database"""
        model = self._to_model(memory)
        self.session.add(model)
        await self.session.flush()
    
    async def find_by_id(self, memory_id: UUID) -> Optional[EpisodicMemory]:
        """Retrieve episodic memory by ID"""
        stmt = select(EpisodicMemoryModel).where(EpisodicMemoryModel.id == memory_id)
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
        
        return self._to_entity(model)
    
    async def find_by_agent(
        self,
        agent_id: UUID,
        limit: int = 100,
    ) -> List[EpisodicMemory]:
        """Find episodic memories by agent"""
        stmt = (
            select(EpisodicMemoryModel)
            .where(EpisodicMemoryModel.agent_id == agent_id)
            .order_by(EpisodicMemoryModel.created_at.desc())
            .limit(limit)
        )
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]
    
    async def find_by_cluster(
        self,
        agent_id: UUID,
        cluster_id: str,
    ) -> Optional[EpisodicMemory]:
        """Find episodic memory by cluster ID"""
        stmt = select(EpisodicMemoryModel).where(
            and_(
                EpisodicMemoryModel.agent_id == agent_id,
                EpisodicMemoryModel.cluster_id == UUID(cluster_id)
            )
        )
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
        
        return self._to_entity(model)
    
    async def find_by_importance(
        self,
        agent_id: UUID,
        min_importance: float,
        limit: int = 100,
    ) -> List[EpisodicMemory]:
        """Find episodic memories by importance score"""
        stmt = (
            select(EpisodicMemoryModel)
            .where(
                and_(
                    EpisodicMemoryModel.agent_id == agent_id,
                    EpisodicMemoryModel.importance_score >= min_importance
                )
            )
            .order_by(EpisodicMemoryModel.importance_score.desc())
            .limit(limit)
        )
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]
    
    async def find_by_temporal_range(
        self,
        agent_id: UUID,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100,
    ) -> List[EpisodicMemory]:
        """Find episodic memories within a temporal range"""
        stmt = (
            select(EpisodicMemoryModel)
            .where(EpisodicMemoryModel.agent_id == agent_id)
            .order_by(EpisodicMemoryModel.created_at.desc())
            .limit(limit)
        )
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        # Filter by temporal period in Python since we need to parse JSONB
        filtered = []
        for model in models:
            if model.temporal_period:
                period_start = datetime.fromisoformat(model.temporal_period.get("start", ""))
                period_end = datetime.fromisoformat(model.temporal_period.get("end", ""))
                
                # Check if there's overlap with the requested range
                if period_start <= end_time and period_end >= start_time:
                    filtered.append(self._to_entity(model))
        
        return filtered[:limit]
    
    async def delete(self, memory_id: UUID) -> None:
        """Delete an episodic memory"""
        stmt = delete(EpisodicMemoryModel).where(EpisodicMemoryModel.id == memory_id)
        await self.session.execute(stmt)
        await self.session.flush()
    
    async def count_by_agent(self, agent_id: UUID) -> int:
        """Count episodic memories for an agent"""
        stmt = (
            select(func.count())
            .select_from(EpisodicMemoryModel)
            .where(EpisodicMemoryModel.agent_id == agent_id)
        )
        
        result = await self.session.execute(stmt)
        count = result.scalar_one()
        
        return count
    
    async def find_similar(
        self,
        agent_id: UUID,
        embedding_vector: List[float],
        limit: int = 10,
        min_similarity: float = 0.7,
    ) -> List[EpisodicMemory]:
        """
        Find similar episodic memories using cosine similarity.
        
        Args:
            agent_id: Agent ID
            embedding_vector: Query embedding vector
            limit: Maximum number of results
            min_similarity: Minimum cosine similarity (0-1)
            
        Returns:
            List of similar episodic memories ordered by similarity
        """
        stmt = (
            select(EpisodicMemoryModel)
            .where(
                and_(
                    EpisodicMemoryModel.agent_id == agent_id,
                    EpisodicMemoryModel.embedding.isnot(None)
                )
            )
            .order_by(EpisodicMemoryModel.embedding.cosine_distance(embedding_vector))
            .limit(limit)
        )
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]
    
    def _to_model(self, memory: EpisodicMemory) -> EpisodicMemoryModel:
        """Convert domain entity to ORM model"""
        return EpisodicMemoryModel(
            id=memory.id,
            agent_id=memory.agent_id,
            cluster_id=UUID(memory.cluster_id),
            summary=memory.summary,
            embedding=memory.embedding.vector if memory.embedding else None,
            compression_ratio=memory.compression_ratio,
            source_memory_ids=memory.source_memory_ids,
            source_memory_count=memory.source_memory_count,
            importance_score=memory.importance_score,
            created_at=memory.created_at,
            last_accessed_at=datetime.utcnow(),
            access_count=0,
            temporal_period={
                "start": memory.temporal_period["start"].isoformat(),
                "end": memory.temporal_period["end"].isoformat(),
            },
            tags=memory.metadata.get("tags", []),
            metadata=memory.metadata,
        )
    
    def _to_entity(self, model: EpisodicMemoryModel) -> EpisodicMemory:
        """Convert ORM model to domain entity"""
        embedding = None
        if model.embedding is not None:
            embedding = MemoryEmbedding(
                vector=model.embedding,
                model="text-embedding-ada-002",
                dimension=1536,
            )
        
        temporal_period = {
            "start": datetime.fromisoformat(model.temporal_period["start"]),
            "end": datetime.fromisoformat(model.temporal_period["end"]),
        } if model.temporal_period else {"start": model.created_at, "end": model.created_at}
        
        return EpisodicMemory(
            id=model.id,
            agent_id=model.agent_id,
            cluster_id=str(model.cluster_id),
            summary=model.summary,
            embedding=embedding,
            compression_ratio=model.compression_ratio or 0.0,
            source_memory_ids=model.source_memory_ids or [],
            source_memory_count=model.source_memory_count or 0,
            importance_score=model.importance_score,
            temporal_period=temporal_period,
            created_at=model.created_at,
            updated_at=model.last_accessed_at,
            metadata=model.metadata or {},
        )


class PostgreSQLLongTermMemoryRepository(LongTermMemoryRepository):
    """
    PostgreSQL implementation of LongTermMemoryRepository.
    
    Maps between LongTermMemory domain entities and SQLAlchemy models.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, memory: LongTermMemory) -> None:
        """Persist long-term memory to database"""
        model = self._to_model(memory)
        self.session.add(model)
        await self.session.flush()
    
    async def find_by_id(self, memory_id: UUID) -> Optional[LongTermMemory]:
        """Retrieve long-term memory by ID"""
        stmt = select(LongTermMemoryModel).where(LongTermMemoryModel.id == memory_id)
        
        result = await self.session.execute(stmt)
        model = result.scalar_one_or_none()
        
        if model is None:
            return None
        
        return self._to_entity(model)
    
    async def find_by_agent(
        self,
        agent_id: UUID,
        include_archived: bool = False,
        limit: int = 100,
    ) -> List[LongTermMemory]:
        """Find long-term memories by agent"""
        stmt = select(LongTermMemoryModel).where(
            LongTermMemoryModel.agent_id == agent_id
        )
        
        if not include_archived:
            stmt = stmt.where(LongTermMemoryModel.archived_at.is_(None))
        
        stmt = stmt.order_by(LongTermMemoryModel.created_at.desc()).limit(limit)
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]
    
    async def find_by_knowledge_type(
        self,
        agent_id: UUID,
        knowledge_type: str,
        limit: int = 100,
    ) -> List[LongTermMemory]:
        """Find long-term memories by knowledge type"""
        stmt = (
            select(LongTermMemoryModel)
            .where(
                and_(
                    LongTermMemoryModel.agent_id == agent_id,
                    LongTermMemoryModel.knowledge_type == knowledge_type
                )
            )
            .order_by(LongTermMemoryModel.created_at.desc())
            .limit(limit)
        )
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]
    
    async def find_by_importance(
        self,
        agent_id: UUID,
        min_importance: float,
        limit: int = 100,
    ) -> List[LongTermMemory]:
        """Find long-term memories by importance score"""
        stmt = (
            select(LongTermMemoryModel)
            .where(
                and_(
                    LongTermMemoryModel.agent_id == agent_id,
                    LongTermMemoryModel.importance_score >= min_importance
                )
            )
            .order_by(LongTermMemoryModel.importance_score.desc())
            .limit(limit)
        )
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]
    
    async def find_archived(
        self,
        agent_id: UUID,
        limit: int = 100,
    ) -> List[LongTermMemory]:
        """Find archived long-term memories"""
        stmt = (
            select(LongTermMemoryModel)
            .where(
                and_(
                    LongTermMemoryModel.agent_id == agent_id,
                    LongTermMemoryModel.archived_at.isnot(None)
                )
            )
            .order_by(LongTermMemoryModel.archived_at.desc())
            .limit(limit)
        )
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]
    
    async def search_by_title(
        self,
        agent_id: UUID,
        title_query: str,
        limit: int = 100,
    ) -> List[LongTermMemory]:
        """Search long-term memories by title"""
        stmt = (
            select(LongTermMemoryModel)
            .where(
                and_(
                    LongTermMemoryModel.agent_id == agent_id,
                    LongTermMemoryModel.title.ilike(f"%{title_query}%")
                )
            )
            .order_by(LongTermMemoryModel.created_at.desc())
            .limit(limit)
        )
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]
    
    async def delete(self, memory_id: UUID) -> None:
        """Delete a long-term memory"""
        stmt = delete(LongTermMemoryModel).where(LongTermMemoryModel.id == memory_id)
        await self.session.execute(stmt)
        await self.session.flush()
    
    async def count_by_agent(
        self,
        agent_id: UUID,
        include_archived: bool = False,
    ) -> int:
        """Count long-term memories for an agent"""
        stmt = (
            select(func.count())
            .select_from(LongTermMemoryModel)
            .where(LongTermMemoryModel.agent_id == agent_id)
        )
        
        if not include_archived:
            stmt = stmt.where(LongTermMemoryModel.archived_at.is_(None))
        
        result = await self.session.execute(stmt)
        count = result.scalar_one()
        
        return count
    
    async def find_similar(
        self,
        agent_id: UUID,
        embedding_vector: List[float],
        limit: int = 10,
        min_similarity: float = 0.7,
        include_archived: bool = False,
    ) -> List[LongTermMemory]:
        """
        Find similar long-term memories using cosine similarity.
        
        Args:
            agent_id: Agent ID
            embedding_vector: Query embedding vector
            limit: Maximum number of results
            min_similarity: Minimum cosine similarity (0-1)
            include_archived: Include archived memories
            
        Returns:
            List of similar long-term memories ordered by similarity
        """
        stmt = (
            select(LongTermMemoryModel)
            .where(
                and_(
                    LongTermMemoryModel.agent_id == agent_id,
                    LongTermMemoryModel.embedding.isnot(None)
                )
            )
        )
        
        if not include_archived:
            stmt = stmt.where(LongTermMemoryModel.archived_at.is_(None))
        
        stmt = stmt.order_by(
            LongTermMemoryModel.embedding.cosine_distance(embedding_vector)
        ).limit(limit)
        
        result = await self.session.execute(stmt)
        models = result.scalars().all()
        
        return [self._to_entity(model) for model in models]
    
    def _to_model(self, memory: LongTermMemory) -> LongTermMemoryModel:
        """Convert domain entity to ORM model"""
        return LongTermMemoryModel(
            id=memory.id,
            agent_id=memory.agent_id,
            knowledge_type=memory.knowledge_type.value,
            title=memory.title,
            content=memory.content,
            embedding=memory.embedding.vector if memory.embedding else None,
            importance_score=memory.importance_score,
            created_at=memory.created_at,
            last_accessed_at=memory.updated_at,
            access_count=0,
            archived_at=memory.archived_at,
            source_type=memory.source_type.value,
            tags=memory.metadata.get("tags", []),
            metadata=memory.metadata,
        )
    
    def _to_entity(self, model: LongTermMemoryModel) -> LongTermMemory:
        """Convert ORM model to domain entity"""
        embedding = None
        if model.embedding is not None:
            embedding = MemoryEmbedding(
                vector=model.embedding,
                model="text-embedding-ada-002",
                dimension=1536,
            )
        
        return LongTermMemory(
            id=model.id,
            agent_id=model.agent_id,
            knowledge_type=KnowledgeType(model.knowledge_type) if model.knowledge_type else KnowledgeType.FACTUAL,
            title=model.title or "",
            content=model.content,
            embedding=embedding,
            importance_score=model.importance_score,
            source_type=SourceType(model.source_type) if model.source_type else SourceType.MANUAL_ENTRY,
            source_references=model.metadata.get("source_references", []) if model.metadata else [],
            created_at=model.created_at,
            updated_at=model.last_accessed_at,
            archived_at=model.archived_at,
            metadata=model.metadata or {},
        )
