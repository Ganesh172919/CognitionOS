"""
Memory Hierarchy API Routes

Provides REST endpoints for hierarchical memory management.
"""

import os
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.application.memory_hierarchy.use_cases import (
    StoreMemoryCommand,
    RetrieveMemoryQuery,
    PromoteMemoriesCommand,
    EvictMemoriesCommand,
    MemoryStatisticsQuery,
    SearchMemoriesQuery,
    UpdateImportanceCommand,
    StoreWorkingMemoryUseCase,
    RetrieveWorkingMemoryUseCase,
    PromoteMemoriesToL2UseCase,
    PromoteMemoriesToL3UseCase,
    EvictLowPriorityMemoriesUseCase,
    GetMemoryStatisticsUseCase,
    SearchMemoriesAcrossTiersUseCase,
    UpdateMemoryImportanceUseCase,
)
from services.api.src.schemas.phase3 import (
    StoreMemoryRequest,
    RetrieveMemoryRequest,
    PromoteMemoriesRequest,
    EvictMemoriesRequest,
    MemoryStatisticsRequest,
    SearchMemoriesRequest,
    UpdateImportanceRequest,
    MemoryResponse,
    MemoryListResponse,
    MemoryStatisticsResponse,
    EvictMemoriesResponse,
    UpdateImportanceResponse,
)
from services.api.src.dependencies.injection import (
    get_db_session,
    get_store_memory_use_case,
    get_retrieve_memory_use_case,
    get_promote_l2_use_case,
    get_promote_l3_use_case,
    get_evict_memories_use_case,
    get_memory_statistics_use_case,
    get_search_memories_use_case,
    get_update_importance_use_case,
)


router = APIRouter(prefix="/api/v3/memory", tags=["Hierarchical Memory"])


@router.post(
    "/working",
    response_model=MemoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Store memory in L1 working memory",
    description="Store a new memory in L1 working memory tier with optional embedding and metadata",
)
async def store_working_memory(
    request: StoreMemoryRequest,
    session: AsyncSession = Depends(get_db_session),
) -> MemoryResponse:
    """Store a new working memory in L1"""
    try:
        # Get use case
        use_case = await get_store_memory_use_case(session)
        
        # Convert request to command
        command = StoreMemoryCommand(
            agent_id=request.agent_id,
            workflow_execution_id=request.workflow_execution_id,
            content=request.content,
            importance_score=request.importance_score,
            embedding=request.embedding,
            memory_type=request.memory_type,
            tags=request.tags,
            expires_at=request.expires_at,
            metadata=request.metadata,
        )
        
        # Execute use case
        result = await use_case.execute(command)
        
        return MemoryResponse(
            id=result.id,
            tier=result.tier,
            content=result.content,
            importance_score=result.importance_score,
            created_at=result.created_at,
            memory_type=result.memory_type,
            access_count=result.access_count,
            tags=result.tags,
            metadata=result.metadata,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store working memory: {str(e)}"
        )


@router.get(
    "/working/{agent_id}",
    response_model=MemoryListResponse,
    summary="Retrieve L1 memories for agent",
    description="Retrieve working memories from L1 for a specific agent with optional filters",
)
async def retrieve_working_memory(
    agent_id: UUID,
    workflow_execution_id: Optional[UUID] = None,
    min_importance: Optional[float] = None,
    limit: int = 100,
    session: AsyncSession = Depends(get_db_session),
) -> MemoryListResponse:
    """Retrieve working memories for an agent"""
    try:
        # Get use case
        use_case = await get_retrieve_memory_use_case(session)
        
        # Create query
        query = RetrieveMemoryQuery(
            agent_id=agent_id,
            tier="l1_working",
            workflow_execution_id=workflow_execution_id,
            min_importance=min_importance,
            limit=limit,
        )
        
        # Execute use case
        results = await use_case.execute(query)
        
        return MemoryListResponse(
            memories=[
                MemoryResponse(
                    id=r.id,
                    tier=r.tier,
                    content=r.content,
                    importance_score=r.importance_score,
                    created_at=r.created_at,
                    memory_type=r.memory_type,
                    access_count=r.access_count,
                    tags=r.tags,
                    metadata=r.metadata,
                )
                for r in results
            ],
            total=len(results),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve working memories: {str(e)}"
        )


@router.post(
    "/promote/l2",
    response_model=MemoryListResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Promote L1→L2 (compress)",
    description="Promote working memories from L1 to L2 episodic memory with clustering and compression",
)
async def promote_to_l2(
    request: PromoteMemoriesRequest,
    session: AsyncSession = Depends(get_db_session),
) -> MemoryListResponse:
    """Promote memories from L1 to L2"""
    try:
        # Get use case
        use_case = await get_promote_l2_use_case(session)
        
        # Convert request to command
        command = PromoteMemoriesCommand(
            agent_id=request.agent_id,
            tier_from="l1_working",
            tier_to="l2_episodic",
            memory_ids=request.memory_ids,
            min_importance=request.min_importance,
            min_access_count=request.min_access_count,
            min_age_hours=request.min_age_hours,
            cluster_id=request.cluster_id,
            summary=request.summary,
        )
        
        # Execute use case
        results = await use_case.execute(command)
        
        return MemoryListResponse(
            memories=[
                MemoryResponse(
                    id=r.id,
                    tier=r.tier,
                    content=r.content,
                    importance_score=r.importance_score,
                    created_at=r.created_at,
                    metadata=r.metadata,
                )
                for r in results
            ],
            total=len(results),
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to promote memories to L2: {str(e)}"
        )


@router.post(
    "/promote/l3",
    response_model=MemoryListResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Promote L2→L3 (archive)",
    description="Promote episodic memories from L2 to L3 long-term memory for archival",
)
async def promote_to_l3(
    request: PromoteMemoriesRequest,
    session: AsyncSession = Depends(get_db_session),
) -> MemoryListResponse:
    """Promote memories from L2 to L3"""
    try:
        # Get use case
        use_case = await get_promote_l3_use_case(session)
        
        # Convert request to command
        command = PromoteMemoriesCommand(
            agent_id=request.agent_id,
            tier_from="l2_episodic",
            tier_to="l3_longterm",
            memory_ids=request.memory_ids,
            min_importance=request.min_importance,
            knowledge_type=request.knowledge_type,
            title=request.title,
        )
        
        # Execute use case
        results = await use_case.execute(command)
        
        return MemoryListResponse(
            memories=[
                MemoryResponse(
                    id=r.id,
                    tier=r.tier,
                    content=r.content,
                    importance_score=r.importance_score,
                    created_at=r.created_at,
                    metadata=r.metadata,
                )
                for r in results
            ],
            total=len(results),
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to promote memories to L3: {str(e)}"
        )


@router.post(
    "/evict",
    response_model=EvictMemoriesResponse,
    status_code=status.HTTP_200_OK,
    summary="Evict low-priority L1 memories",
    description="Evict low-priority memories from L1 using LRU or low-importance strategy",
)
async def evict_memories(
    request: EvictMemoriesRequest,
    session: AsyncSession = Depends(get_db_session),
) -> EvictMemoriesResponse:
    """Evict low-priority memories from L1"""
    try:
        # Get use case
        use_case = await get_evict_memories_use_case(session)
        
        # Convert request to command
        command = EvictMemoriesCommand(
            agent_id=request.agent_id,
            workflow_execution_id=request.workflow_execution_id,
            max_count=request.max_count,
            strategy=request.strategy,
        )
        
        # Execute use case
        evicted_count = await use_case.execute(command)
        
        return EvictMemoriesResponse(
            evicted_count=evicted_count,
            strategy=request.strategy,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to evict memories: {str(e)}"
        )


@router.get(
    "/statistics/{agent_id}",
    response_model=MemoryStatisticsResponse,
    summary="Get memory tier statistics",
    description="Retrieve comprehensive statistics for all memory tiers for a specific agent",
)
async def get_memory_statistics(
    agent_id: UUID,
    include_archived: bool = False,
    session: AsyncSession = Depends(get_db_session),
) -> MemoryStatisticsResponse:
    """Get memory tier statistics for an agent"""
    try:
        # Get use case
        use_case = await get_memory_statistics_use_case(session)
        
        # Create query
        query = MemoryStatisticsQuery(
            agent_id=agent_id,
            include_archived=include_archived,
        )
        
        # Execute use case
        result = await use_case.execute(query)
        
        return MemoryStatisticsResponse(
            l1_count=result.l1_count,
            l2_count=result.l2_count,
            l3_count=result.l3_count,
            total_size_mb=result.total_size_mb,
            l1_avg_importance=result.l1_avg_importance,
            l2_avg_importance=result.l2_avg_importance,
            l3_avg_importance=result.l3_avg_importance,
            l1_avg_age_hours=result.l1_avg_age_hours,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memory statistics: {str(e)}"
        )


@router.post(
    "/search",
    response_model=MemoryListResponse,
    summary="Semantic search across all tiers",
    description="Perform semantic search across all memory tiers using embedding similarity",
)
async def search_memories(
    request: SearchMemoriesRequest,
    session: AsyncSession = Depends(get_db_session),
) -> MemoryListResponse:
    """Search memories across all tiers"""
    try:
        # Get use case
        use_case = await get_search_memories_use_case(session)
        
        # Convert request to query
        query = SearchMemoriesQuery(
            agent_id=request.agent_id,
            query_embedding=request.query_embedding,
            top_k=request.top_k,
            tiers=request.tiers,
            similarity_threshold=request.similarity_threshold,
        )
        
        # Execute use case
        results = await use_case.execute(query)
        
        return MemoryListResponse(
            memories=[
                MemoryResponse(
                    id=r.id,
                    tier=r.tier,
                    content=r.content,
                    importance_score=r.importance_score,
                    created_at=r.created_at,
                    memory_type=r.memory_type,
                    access_count=r.access_count,
                    tags=r.tags,
                    metadata=r.metadata,
                )
                for r in results
            ],
            total=len(results),
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search memories: {str(e)}"
        )


@router.post(
    "/importance/update",
    response_model=UpdateImportanceResponse,
    status_code=status.HTTP_200_OK,
    summary="Update importance scores",
    description="Recalculate and update importance scores for memories",
)
async def update_importance_scores(
    request: UpdateImportanceRequest,
    session: AsyncSession = Depends(get_db_session),
) -> UpdateImportanceResponse:
    """Update importance scores for memories"""
    try:
        # Get use case
        use_case = await get_update_importance_use_case(session)
        
        # Convert request to command
        command = UpdateImportanceCommand(
            agent_id=request.agent_id,
            workflow_execution_id=request.workflow_execution_id,
            memory_ids=request.memory_ids,
        )
        
        # Execute use case
        updated_count = await use_case.execute(command)
        
        return UpdateImportanceResponse(
            updated_count=updated_count,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update importance scores: {str(e)}"
        )
