#!/usr/bin/env python3
"""
Database initialization script for CognitionOS

Initializes the database with default agents, tools, and configurations.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import (
    init_db, get_db_context, Agent, Tool, AgentRole,
    AgentStatus, ToolType
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_default_agents():
    """Create default AI agents"""
    logger.info("Creating default agents...")

    default_agents = [
        {
            "name": "Primary Planner",
            "role": AgentRole.PLANNER,
            "capabilities": ["task_decomposition", "dependency_analysis", "estimation"],
            "prompt_version": "v1",
            "model": "gpt-4-turbo-preview",
            "temperature": 0.7,
            "max_tokens": 4096,
        },
        {
            "name": "Code Executor",
            "role": AgentRole.EXECUTOR,
            "capabilities": ["python_execution", "javascript_execution", "file_operations"],
            "prompt_version": "v1",
            "model": "gpt-4-turbo-preview",
            "temperature": 0.3,
            "max_tokens": 8192,
        },
        {
            "name": "Quality Critic",
            "role": AgentRole.CRITIC,
            "capabilities": ["code_review", "security_analysis", "quality_assessment"],
            "prompt_version": "v1",
            "model": "gpt-4-turbo-preview",
            "temperature": 0.5,
            "max_tokens": 4096,
        },
        {
            "name": "Context Summarizer",
            "role": AgentRole.SUMMARIZER,
            "capabilities": ["text_summarization", "context_compression", "entity_extraction"],
            "prompt_version": "v1",
            "model": "gpt-3.5-turbo",
            "temperature": 0.4,
            "max_tokens": 2048,
        },
    ]

    async with get_db_context() as db:
        for agent_data in default_agents:
            agent = Agent(
                status=AgentStatus.IDLE,
                **agent_data
            )
            db.add(agent)
            logger.info(f"Created agent: {agent.name} ({agent.role.value})")

        await db.commit()

    logger.info(f"Created {len(default_agents)} default agents")


async def create_default_tools():
    """Create default execution tools"""
    logger.info("Creating default tools...")

    default_tools = [
        {
            "name": "execute_python",
            "type": ToolType.PYTHON,
            "description": "Execute Python code in a sandboxed environment",
            "parameters_schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "timeout": {"type": "integer", "default": 30, "description": "Timeout in seconds"}
                },
                "required": ["code"]
            },
            "timeout_seconds": 30,
            "retry_count": 0,
        },
        {
            "name": "execute_javascript",
            "type": ToolType.JAVASCRIPT,
            "description": "Execute JavaScript code in Node.js environment",
            "parameters_schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "JavaScript code to execute"},
                    "timeout": {"type": "integer", "default": 30, "description": "Timeout in seconds"}
                },
                "required": ["code"]
            },
            "timeout_seconds": 30,
            "retry_count": 0,
        },
        {
            "name": "http_request",
            "type": ToolType.HTTP,
            "description": "Make HTTP/HTTPS requests to external APIs",
            "parameters_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to request"},
                    "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"], "default": "GET"},
                    "headers": {"type": "object", "description": "HTTP headers"},
                    "body": {"type": "object", "description": "Request body (for POST/PUT/PATCH)"},
                    "timeout": {"type": "integer", "default": 30}
                },
                "required": ["url"]
            },
            "timeout_seconds": 30,
            "retry_count": 3,
        },
        {
            "name": "read_file",
            "type": ToolType.FILE,
            "description": "Read contents of a file",
            "parameters_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "encoding": {"type": "string", "default": "utf-8"}
                },
                "required": ["path"]
            },
            "timeout_seconds": 10,
            "retry_count": 0,
        },
        {
            "name": "write_file",
            "type": ToolType.FILE,
            "description": "Write content to a file",
            "parameters_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                    "mode": {"type": "string", "enum": ["write", "append"], "default": "write"}
                },
                "required": ["path", "content"]
            },
            "timeout_seconds": 10,
            "retry_count": 0,
        },
        {
            "name": "sql_query",
            "type": ToolType.SQL,
            "description": "Execute SQL query (SELECT only for safety)",
            "parameters_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"},
                    "database": {"type": "string", "description": "Database name"},
                    "timeout": {"type": "integer", "default": 30}
                },
                "required": ["query", "database"]
            },
            "timeout_seconds": 30,
            "retry_count": 0,
        },
        {
            "name": "search_web",
            "type": ToolType.SEARCH,
            "description": "Search the internet for information",
            "parameters_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20}
                },
                "required": ["query"]
            },
            "timeout_seconds": 30,
            "retry_count": 3,
        },
    ]

    async with get_db_context() as db:
        for tool_data in default_tools:
            tool = Tool(**tool_data)
            db.add(tool)
            logger.info(f"Created tool: {tool.name} ({tool.type.value})")

        await db.commit()

    logger.info(f"Created {len(default_tools)} default tools")


async def main():
    """Main initialization function"""
    logger.info("=" * 60)
    logger.info("CognitionOS Database Initialization")
    logger.info("=" * 60)

    # Step 1: Initialize database schema
    logger.info("\n[1/3] Initializing database schema...")
    try:
        await init_db()
        logger.info("✓ Database schema initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize schema: {e}")
        return 1

    # Step 2: Create default agents
    logger.info("\n[2/3] Creating default agents...")
    try:
        await create_default_agents()
        logger.info("✓ Default agents created")
    except Exception as e:
        logger.error(f"✗ Failed to create agents: {e}")
        return 1

    # Step 3: Create default tools
    logger.info("\n[3/3] Creating default tools...")
    try:
        await create_default_tools()
        logger.info("✓ Default tools created")
    except Exception as e:
        logger.error(f"✗ Failed to create tools: {e}")
        return 1

    logger.info("\n" + "=" * 60)
    logger.info("Database initialization complete!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("  1. Start all services: docker-compose up -d")
    logger.info("  2. Check service health: curl http://localhost:8000/health")
    logger.info("  3. Create a user account via API Gateway")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
