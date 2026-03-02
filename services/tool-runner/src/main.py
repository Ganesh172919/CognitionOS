"""
Tool Runner Service.

Provides sandboxed execution environment for agent tools.
"""

import asyncio
import base64
import html
import os
import re
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from datetime import datetime
from typing import List, Dict, Optional, Any
from uuid import UUID, uuid4
import subprocess
import json

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

import httpx

from shared.libs.config import ToolRunnerConfig, load_config
from shared.libs.logger import setup_logger, get_contextual_logger
from shared.libs.models import ErrorResponse
from shared.libs.middleware import (
    TracingMiddleware,
    LoggingMiddleware,
    ErrorHandlingMiddleware
)

# Import audit client
from audit_client import AuditClient


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
    Integrates with audit-log service for comprehensive logging.
    """

    def __init__(self):
        self.logger = get_contextual_logger(__name__, component="SandboxExecutor")
        self.use_docker = config.sandbox_enabled and config.execution_mode.lower() == "docker"
        self.audit_client = AuditClient()
        self._semaphore = asyncio.Semaphore(max(1, int(config.max_concurrent_executions)))

        allowed_roots_env = os.getenv("SANDBOX_ALLOWED_ROOTS", "").strip()
        roots = [r.strip() for r in allowed_roots_env.split(",") if r.strip()] if allowed_roots_env else []
        if not roots:
            roots = ["/tmp/cognitionos_sandbox"]
        self._sandbox_root = Path(roots[0]).resolve()
        self._sandbox_root.mkdir(parents=True, exist_ok=True)

        self._max_file_bytes = int(os.getenv("SANDBOX_MAX_FILE_BYTES", "1048576"))  # 1 MiB default

        allowlist_env = os.getenv("SANDBOX_NETWORK_ALLOWLIST", "").strip()
        self._network_allowlist = [h.strip().lower() for h in allowlist_env.split(",") if h.strip()]

    def _resolve_path(self, rel_path: str) -> Path:
        if not isinstance(rel_path, str) or not rel_path.strip():
            raise ValueError("path is required")
        if os.path.isabs(rel_path):
            raise ValueError("absolute paths are not allowed")

        candidate = (self._sandbox_root / rel_path).resolve()
        try:
            if not candidate.is_relative_to(self._sandbox_root):
                raise ValueError("path escapes sandbox root")
        except AttributeError:
            root = str(self._sandbox_root)
            if not str(candidate).startswith(root.rstrip(os.sep) + os.sep):
                raise ValueError("path escapes sandbox root")

        return candidate

    def _validate_network_target(self, url: str) -> None:
        if not config.sandbox_network_enabled:
            raise PermissionError("Sandbox network is disabled")

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise PermissionError("Only http/https URLs are allowed")
        if not parsed.hostname:
            raise ValueError("Invalid URL: missing hostname")

        if self._network_allowlist:
            host = parsed.hostname.lower()
            if host not in self._network_allowlist and not any(
                host.endswith("." + allowed) for allowed in self._network_allowlist
            ):
                raise PermissionError("Target hostname not in allowlist")

    async def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: int,
        permissions: List[str],
        user_id: UUID,
        agent_id: UUID
    ) -> ToolExecutionResult:
        """
        Execute a tool in sandbox with audit logging.

        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            timeout: Execution timeout
            permissions: User permissions
            user_id: User ID for audit
            agent_id: Agent ID for audit

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

        await self._semaphore.acquire()
        try:
            # Check permissions
            tool_def = TOOLS.get(tool_name)
            if not tool_def:
                raise ValueError(f"Unknown tool: {tool_name}")

            required_perms = tool_def["required_permissions"]
            if not all(perm in permissions for perm in required_perms):
                # Log permission denied to audit
                await self.audit_client.log_permission_denied(
                    tool_name=tool_name,
                    user_id=user_id,
                    agent_id=agent_id,
                    required_permissions=required_perms,
                    granted_permissions=permissions
                )

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
            duration_ms = int(duration * 1000)

            success = "error" not in result

            # Log to audit service
            await self.audit_client.log_tool_execution(
                tool_name=tool_name,
                user_id=user_id,
                agent_id=agent_id,
                parameters=parameters,
                permissions=required_perms,
                outcome="success" if success else "failure",
                success=success,
                duration_ms=duration_ms,
                error=result.get("error"),
                output=result.get("output")
            )

            return ToolExecutionResult(
                execution_id=execution_id,
                tool_name=tool_name,
                success=success,
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
            duration_ms = int(duration * 1000)

            # Log failure to audit
            await self.audit_client.log_tool_execution(
                tool_name=tool_name,
                user_id=user_id,
                agent_id=agent_id,
                parameters=parameters,
                permissions=permissions,
                outcome="error",
                success=False,
                duration_ms=duration_ms,
                error=str(e)
            )

            return ToolExecutionResult(
                execution_id=execution_id,
                tool_name=tool_name,
                success=False,
                error=str(e),
                duration_seconds=duration
            )
        finally:
            try:
                self._semaphore.release()
            except Exception:
                pass

    async def _execute_python(
        self,
        parameters: Dict[str, Any],
        timeout: int
    ) -> Dict[str, Any]:
        """Execute Python code."""
        code = parameters.get("code", "")
        if not isinstance(code, str):
            return {"error": "Invalid parameter: code must be a string"}

        with tempfile.TemporaryDirectory(prefix="cogpy_") as tmpdir:
            tmp_path = Path(tmpdir)
            script_path = tmp_path / "main.py"
            script_path.write_text(code, encoding="utf-8")

            if self.use_docker:
                cmd: List[str] = [
                    "docker",
                    "run",
                    "--rm",
                    "--pids-limit",
                    "256",
                    "--memory",
                    config.sandbox_memory_limit,
                    "--cpus",
                    str(config.sandbox_cpu_limit),
                    "--security-opt",
                    "no-new-privileges",
                    "--cap-drop",
                    "ALL",
                    "-e",
                    "PYTHONDONTWRITEBYTECODE=1",
                ]
                if not config.sandbox_network_enabled:
                    cmd.extend(["--network", "none"])

                cmd.extend(
                    [
                        "-v",
                        f"{script_path.parent}:/work:ro",
                        "-w",
                        "/work",
                        config.docker_base_image,
                        "python",
                        "-I",
                        "-u",
                        "main.py",
                    ]
                )
            else:
                cmd = [sys.executable, "-I", "-u", str(script_path)]

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(tmp_path),
                )
            except subprocess.TimeoutExpired:
                return {"error": f"Python execution timed out after {timeout}s"}
            except FileNotFoundError as exc:
                return {"error": f"Execution backend not available: {exc}"}

            stdout = (proc.stdout or "")[:200_000]
            stderr = (proc.stderr or "")[:200_000]
            if proc.returncode != 0:
                return {
                    "error": f"Python exited with code {proc.returncode}",
                    "stdout": stdout,
                    "stderr": stderr,
                    "output": {"exit_code": proc.returncode},
                }

            return {"output": {"exit_code": 0}, "stdout": stdout, "stderr": stderr}

    async def _execute_http(
        self,
        parameters: Dict[str, Any],
        timeout: int
    ) -> Dict[str, Any]:
        """Make HTTP request."""
        url = parameters.get("url")
        method = (parameters.get("method") or "GET").upper()
        headers = parameters.get("headers") or {}
        body = parameters.get("body")

        if not isinstance(url, str) or not url:
            return {"error": "Invalid parameter: url is required"}

        self._validate_network_target(url)

        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}:
            return {"error": f"Unsupported method: {method}"}

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout), follow_redirects=False) as client:
                resp = await client.request(
                    method,
                    url,
                    headers=headers if isinstance(headers, dict) else None,
                    json=body if isinstance(body, (dict, list)) else None,
                    content=body if isinstance(body, (str, bytes)) else None,
                )
        except Exception as exc:  # noqa: BLE001
            return {"error": f"HTTP request failed: {exc}"}

        text_body = resp.text
        if len(text_body) > 50_000:
            text_body = text_body[:50_000] + "\n...[truncated]..."

        return {
            "output": {
                "status_code": resp.status_code,
                "headers": dict(resp.headers),
                "body": text_body,
            }
        }

    async def _execute_read_file(
        self,
        parameters: Dict[str, Any],
        timeout: int
    ) -> Dict[str, Any]:
        """Read file contents."""
        rel_path = parameters.get("path")
        try:
            target = self._resolve_path(rel_path)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

        if not target.exists():
            return {"error": "File not found"}
        if target.is_dir():
            return {"error": "Path is a directory"}

        data = target.read_bytes()
        if len(data) > self._max_file_bytes:
            return {"error": f"File too large (>{self._max_file_bytes} bytes)"}

        try:
            text = data.decode("utf-8")
            return {"output": {"path": str(rel_path), "encoding": "utf-8", "content": text, "size_bytes": len(data)}}
        except UnicodeDecodeError:
            b64 = base64.b64encode(data).decode("ascii")
            return {"output": {"path": str(rel_path), "encoding": "base64", "content": b64, "size_bytes": len(data)}}

    async def _execute_write_file(
        self,
        parameters: Dict[str, Any],
        timeout: int
    ) -> Dict[str, Any]:
        """Write file contents."""
        rel_path = parameters.get("path")
        content = parameters.get("content")

        if not isinstance(content, str):
            return {"error": "Invalid parameter: content must be a string"}

        try:
            target = self._resolve_path(rel_path)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        if len(data) > self._max_file_bytes:
            return {"error": f"Content too large (>{self._max_file_bytes} bytes)"}

        target.write_text(content, encoding="utf-8")
        return {"output": {"path": str(rel_path), "bytes_written": len(data)}}

    async def _execute_web_search(
        self,
        parameters: Dict[str, Any],
        timeout: int
    ) -> Dict[str, Any]:
        """Search the web."""
        query = parameters.get("query")
        num_results = int(parameters.get("num_results", 5))
        if not isinstance(query, str) or not query.strip():
            return {"error": "Invalid parameter: query is required"}

        if not config.sandbox_network_enabled:
            return {"error": "Sandbox network is disabled"}

        url = f"https://duckduckgo.com/html/?{httpx.QueryParams({'q': query})}"
        self._validate_network_target(url)

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout), follow_redirects=False) as client:
                resp = await client.get(url, headers={"User-Agent": "CognitionOS-ToolRunner/1.0"})
                resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Search request failed: {exc}"}

        pattern = re.compile(r'<a[^>]+class=\"result__a\"[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>')
        results = []
        for match in pattern.finditer(resp.text):
            href = match.group(1)
            title_html = match.group(2)
            title = re.sub(r"<.*?>", "", title_html)
            title = html.unescape(title).strip()
            results.append({"title": title, "url": href})
            if len(results) >= max(1, min(20, num_results)):
                break

        return {"output": results}

    async def _execute_sql(
        self,
        parameters: Dict[str, Any],
        timeout: int
    ) -> Dict[str, Any]:
        """Execute SQL query."""
        query = parameters.get("query")
        if not isinstance(query, str) or not query.strip():
            return {"error": "Invalid parameter: query is required"}

        normalized = query.strip().upper()
        if not (normalized.startswith("SELECT") or normalized.startswith("EXPLAIN")):
            return {"error": "Only SELECT/EXPLAIN queries allowed with database_read permission"}
        if ";" in query.strip().rstrip(";"):
            return {"error": "Multiple statements are not allowed"}

        try:
            import asyncpg
        except Exception as exc:  # noqa: BLE001
            return {"error": f"asyncpg not available: {exc}"}

        try:
            conn = await asyncpg.connect(config.database_url, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Failed to connect to database: {exc}"}

        try:
            rows = await conn.fetch(query, timeout=timeout)
            max_rows = int(os.getenv("SANDBOX_SQL_MAX_ROWS", "200"))
            out = [dict(r) for r in rows[: max(1, min(5000, max_rows))]]
            return {"output": out}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"SQL query failed: {exc}"}
        finally:
            try:
                await conn.close()
            except Exception:
                pass


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

    # Execute with audit logging
    result = await executor.execute(
        tool_name=request.tool_name,
        parameters=request.parameters,
        timeout=timeout,
        permissions=request.permissions,
        user_id=request.user_id,
        agent_id=request.agent_id
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
