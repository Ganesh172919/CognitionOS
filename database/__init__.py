"""
Database package for CognitionOS
"""

from .models import (
    Base,
    User,
    Session,
    Task,
    TaskExecutionLog,
    Memory,
    Agent,
    AgentTaskAssignment,
    Tool,
    ToolExecution,
    Conversation,
    Message,
    APIUsage,
    LLMUsage,
    # Enums
    TaskStatus,
    TaskComplexity,
    MemoryType,
    MemoryScope,
    AgentRole,
    AgentStatus,
    ToolType,
)

from .connection import (
    async_engine,
    sync_engine,
    AsyncSessionLocal,
    SessionLocal,
    init_db,
    get_db,
    get_db_context,
    get_sync_db,
    close_db,
    check_db_health,
)

__all__ = [
    # Models
    "Base",
    "User",
    "Session",
    "Task",
    "TaskExecutionLog",
    "Memory",
    "Agent",
    "AgentTaskAssignment",
    "Tool",
    "ToolExecution",
    "Conversation",
    "Message",
    "APIUsage",
    "LLMUsage",
    # Enums
    "TaskStatus",
    "TaskComplexity",
    "MemoryType",
    "MemoryScope",
    "AgentRole",
    "AgentStatus",
    "ToolType",
    # Connection
    "async_engine",
    "sync_engine",
    "AsyncSessionLocal",
    "SessionLocal",
    "init_db",
    "get_db",
    "get_db_context",
    "get_sync_db",
    "close_db",
    "check_db_health",
]
