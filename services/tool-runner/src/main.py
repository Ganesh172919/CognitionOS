"""
Tool Runner Service.

Provides sandboxed execution environment for agent tools.
"""

import sys
import os

# Add shared libs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from datetime import datetime
from typing import List, Dict, Optional, Any
from uuid import UUID, uuid4
from enum import Enum
import subprocess
import json

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from shared.libs.config import ToolRunnerConfig, load_config
from shared.libs.logger import setup_logger, get_contextual_logger
from shared.libs.models import ErrorResponse
from shared.libs.middleware import (
    TracingMiddleware,
    LoggingMiddleware,
    ErrorHandlingMiddleware
)


# Configuration
config = load_config(ToolRunnerConfig)
logger = setup_logger(__name__, level=config.log_level)

# FastAPI app
app = FastAPI(
    title="CognitionOS Tool Runner",
    version=config.service_version,
    description="Sandboxed tool execution service"
)

# Add middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(TracingMiddleware)


# ============================================================================
# Request/Response Models
# ============================================================================

class ToolExecutionRequest(BaseModel):
    """Request to execute a tool."""
    tool_name: str
    parameters: Dict[str, Any]
    user_id: UUID
    agent_id: UUID
    permissions: List[str] = Field(default_factory=list)
    timeout_seconds: Optional[int] = None


class ToolExecutionResult(BaseModel):
    """Result of tool execution."""
    execution_id: UUID
    tool_name: str
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    duration_seconds: float
    stdout: Optional[str] = None
    stderr: Optional[str] = None


# ============================================================================
# Tool Definitions
# ============================================================================

TOOLS = {
    "execute_python": {
        "description": "Execute Python code in sandbox",
        "required_permissions": ["code_execution"],
        "parameters": {"code": "string"},
        "timeout": 30
    },
    "http_request": {
        "description": "Make HTTP request",
        "required_permissions": ["network"],
        "parameters": {"url": "string", "method": "string", "body": "object"},
        "timeout": 10
    },
    "read_file": {
        "description": "Read file contents",
        "required_permissions": ["filesystem_read"],
        "parameters": {"path": "string"},
        "timeout": 5
    },
    "write_file": {
        "description": "Write file contents",
        "required_permissions": ["filesystem_write"],
        "parameters": {"path": "string", "content": "string"},
        "timeout": 5
    },
    "search_web": {
        "description": "Search the web",
        "required_permissions": ["network"],
        "parameters": {"query": "string", "num_results": "integer"},
        "timeout": 15
    },
    "sql_query": {
        "description": "Execute SQL query",
        "required_permissions": ["database_read"],
        "parameters": {"query": "string"},
        "timeout": 10
    }
}


# ============================================================================
# Sandbox Executor
# ============================================================================

class SandboxExecutor:
    """
    Executes tools in sandboxed environment.

    Uses Docker for isolation in production.
    Falls back to subprocess for local development.
    """

    def __init__(self):
        self.logger = get_contextual_logger(__name__, component="SandboxExecutor")
        self.use_docker = config.sandbox_enabled

    async def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: int,
        permissions: List[str]
    ) -> ToolExecutionResult:
        """
        Execute a tool in sandbox.

        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            timeout: Execution timeout
            permissions: User permissions

        Returns:
            Execution result
        """
        execution_id = uuid4()
        start_time = datetime.utcnow()

        self.logger.info(
            "Executing tool",
            extra={
                "execution_id": str(execution_id),
                "tool": tool_name,
                "sandbox": self.use_docker
            }
        )

        try:
            # Check permissions
            tool_def = TOOLS.get(tool_name)
            if not tool_def:
                raise ValueError(f"Unknown tool: {tool_name}")

            required_perms = tool_def["required_permissions"]
            if not all(perm in permissions for perm in required_perms):
                raise PermissionError(
                    f"Missing permissions: {set(required_perms) - set(permissions)}"
                )

            # Execute based on tool type
            if tool_name == "execute_python":
                result = await self._execute_python(parameters, timeout)
            elif tool_name == "http_request":
                result = await self._execute_http(parameters, timeout)
            elif tool_name == "read_file":
                result = await self._execute_read_file(parameters, timeout)
            elif tool_name == "write_file":
                result = await self._execute_write_file(parameters, timeout)
            elif tool_name == "search_web":
                result = await self._execute_web_search(parameters, timeout)
            elif tool_name == "sql_query":
                result = await self._execute_sql(parameters, timeout)
            else:
                result = {"error": f"Tool not implemented: {tool_name}"}

            duration = (datetime.utcnow() - start_time).total_seconds()

            return ToolExecutionResult(
                execution_id=execution_id,
                tool_name=tool_name,
                success="error" not in result,
                output=result.get("output"),
                error=result.get("error"),
                duration_seconds=duration,
                stdout=result.get("stdout"),
                stderr=result.get("stderr")
            )

        except PermissionError as e:
            self.logger.error("Permission denied", extra={"error": str(e)})
            duration = (datetime.utcnow() - start_time).total_seconds()
            return ToolExecutionResult(
                execution_id=execution_id,
                tool_name=tool_name,
                success=False,
                error=str(e),
                duration_seconds=duration
            )
        except Exception as e:
            self.logger.error("Execution failed", extra={"error": str(e)})
            duration = (datetime.utcnow() - start_time).total_seconds()
            return ToolExecutionResult(
                execution_id=execution_id,
                tool_name=tool_name,
                success=False,
                error=str(e),
                duration_seconds=duration
            )

    async def _execute_python(
        self,
        parameters: Dict[str, Any],
        timeout: int
    ) -> Dict[str, Any]:
        """Execute Python code."""
        code = parameters.get("code", "")

        if self.use_docker:
            # In production, run in Docker container
            # docker run --rm --network=none --memory=512m python:3.11-slim python -c "code"
            return {
                "output": "[Docker execution not implemented in demo]",
                "stdout": "Simulated Python execution"
            }
        else:
            # For demo, simulate execution
            return {
                "output": f"Executed Python code:\n{code[:100]}...",
                "stdout": "Code executed successfully"
            }

    async def _execute_http(
        self,
        parameters: Dict[str, Any],
        timeout: int
    ) -> Dict[str, Any]:
        """Make HTTP request."""
        url = parameters.get("url")
        method = parameters.get("method", "GET")

        # In production, use httpx with proper timeout and error handling
        return {
            "output": {
                "status_code": 200,
                "body": f"[Simulated {method} request to {url}]"
            }
        }

    async def _execute_read_file(
        self,
        parameters: Dict[str, Any],
        timeout: int
    ) -> Dict[str, Any]:
        """Read file contents."""
        path = parameters.get("path")

        # Validate path (prevent directory traversal)
        if ".." in path or path.startswith("/"):
            return {"error": "Invalid file path"}

        return {
            "output": f"[Simulated file read from {path}]"
        }

    async def _execute_write_file(
        self,
        parameters: Dict[str, Any],
        timeout: int
    ) -> Dict[str, Any]:
        """Write file contents."""
        path = parameters.get("path")
        content = parameters.get("content")

        # Validate path
        if ".." in path or path.startswith("/"):
            return {"error": "Invalid file path"}

        return {
            "output": f"[Simulated file write to {path}]"
        }

    async def _execute_web_search(
        self,
        parameters: Dict[str, Any],
        timeout: int
    ) -> Dict[str, Any]:
        """Search the web."""
        query = parameters.get("query")
        num_results = parameters.get("num_results", 5)

        # In production, integrate with search API (Google, Bing, etc.)
        return {
            "output": [
                {"title": f"Result {i}", "url": f"https://example.com/{i}"}
                for i in range(num_results)
            ]
        }

    async def _execute_sql(
        self,
        parameters: Dict[str, Any],
        timeout: int
    ) -> Dict[str, Any]:
        """Execute SQL query."""
        query = parameters.get("query")

        # Validate query (only SELECT allowed for database_read permission)
        if not query.strip().upper().startswith("SELECT"):
            return {"error": "Only SELECT queries allowed with database_read permission"}

        return {
            "output": [
                {"id": 1, "name": "Sample Row 1"},
                {"id": 2, "name": "Sample Row 2"}
            ]
        }


# ============================================================================
# API Endpoints
# ============================================================================

executor = SandboxExecutor()


@app.post("/execute", response_model=ToolExecutionResult)
async def execute_tool(request: ToolExecutionRequest):
    """
    Execute a tool in sandboxed environment.

    Checks permissions, enforces limits, and returns results.
    """
    log = get_contextual_logger(
        __name__,
        action="execute_tool",
        tool=request.tool_name,
        user_id=str(request.user_id),
        agent_id=str(request.agent_id)
    )

    # Get tool definition
    tool_def = TOOLS.get(request.tool_name)
    if not tool_def:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown tool: {request.tool_name}"
        )

    # Use default timeout if not specified
    timeout = request.timeout_seconds or tool_def["timeout"]
    timeout = min(timeout, config.max_tool_timeout)  # Enforce max timeout

    # Execute
    result = await executor.execute(
        tool_name=request.tool_name,
        parameters=request.parameters,
        timeout=timeout,
        permissions=request.permissions
    )

    log.info(
        "Tool execution completed",
        extra={
            "execution_id": str(result.execution_id),
            "success": result.success,
            "duration": result.duration_seconds
        }
    )

    return result


@app.get("/tools")
async def list_tools():
    """List available tools and their requirements."""
    return TOOLS


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "tool-runner",
        "version": config.service_version,
        "timestamp": datetime.utcnow().isoformat(),
        "sandbox_enabled": config.sandbox_enabled,
        "available_tools": len(TOOLS)
    }


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize service on startup."""
    logger.info(
        "Tool Runner starting",
        extra={
            "version": config.service_version,
            "sandbox_enabled": config.sandbox_enabled,
            "max_concurrent": config.max_concurrent_executions
        }
    )

    if not config.sandbox_enabled:
        logger.warning("Running WITHOUT sandboxing - FOR DEVELOPMENT ONLY")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Tool Runner shutting down")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.host,
        port=getattr(config, 'port', 8006),
        log_level=config.log_level.lower()
    )
