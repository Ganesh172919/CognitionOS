"""
Memory Hierarchy Application Layer

Application layer for hierarchical memory management (L1/L2/L3).
Exports use cases and DTOs for memory operations.
"""

from .use_cases import (
    # DTOs - Commands
    StoreMemoryCommand,
    RetrieveMemoryQuery,
    PromoteMemoriesCommand,
    EvictMemoriesCommand,
    MemoryStatisticsQuery,
    SearchMemoriesQuery,
    UpdateImportanceCommand,
    
    # DTOs - Results
    MemoryResult,
    MemoryStatisticsResult,
    
    # Use Cases
    StoreWorkingMemoryUseCase,
    RetrieveWorkingMemoryUseCase,
    PromoteMemoriesToL2UseCase,
    PromoteMemoriesToL3UseCase,
    EvictLowPriorityMemoriesUseCase,
    GetMemoryStatisticsUseCase,
    SearchMemoriesAcrossTiersUseCase,
    UpdateMemoryImportanceUseCase,
)

__all__ = [
    # Commands
    "StoreMemoryCommand",
    "RetrieveMemoryQuery",
    "PromoteMemoriesCommand",
    "EvictMemoriesCommand",
    "MemoryStatisticsQuery",
    "SearchMemoriesQuery",
    "UpdateImportanceCommand",
    
    # Results
    "MemoryResult",
    "MemoryStatisticsResult",
    
    # Use Cases
    "StoreWorkingMemoryUseCase",
    "RetrieveWorkingMemoryUseCase",
    "PromoteMemoriesToL2UseCase",
    "PromoteMemoriesToL3UseCase",
    "EvictLowPriorityMemoriesUseCase",
    "GetMemoryStatisticsUseCase",
    "SearchMemoriesAcrossTiersUseCase",
    "UpdateMemoryImportanceUseCase",
]
