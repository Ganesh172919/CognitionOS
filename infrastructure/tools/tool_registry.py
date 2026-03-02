"""
Tool Registry — CognitionOS

Dynamic tool management for AI agents:
- Tool registration with schemas
- Parameter validation
- Permission-based access
- Usage tracking
- Tool versioning
- Tool composition (pipelines)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
from uuid import uuid4

logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    CODE_GENERATION = "code_generation"
    FILE_OPERATIONS = "file_operations"
    DATA_ANALYSIS = "data_analysis"
    WEB_SEARCH = "web_search"
    DATABASE = "database"
    API_INTEGRATION = "api_integration"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    CUSTOM = "custom"


class ToolStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DISABLED = "disabled"
    BETA = "beta"


@dataclass
class ToolParameter:
    name: str
    param_type: str  # string, int, float, bool, list, dict
    description: str = ""
    required: bool = True
    default: Any = None
    enum_values: List[str] = field(default_factory=list)


@dataclass
class ToolDefinition:
    tool_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    category: ToolCategory = ToolCategory.CUSTOM
    status: ToolStatus = ToolStatus.ACTIVE
    parameters: List[ToolParameter] = field(default_factory=list)
    required_permissions: Set[str] = field(default_factory=set)
    max_execution_time: float = 60.0
    rate_limit_per_minute: int = 60
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "version": self.version, "category": self.category.value,
            "parameters": [
                {"name": p.name, "type": p.param_type,
                 "description": p.description, "required": p.required}
                for p in self.parameters]}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id, "name": self.name,
            "description": self.description, "version": self.version,
            "category": self.category.value, "status": self.status.value,
            "parameter_count": len(self.parameters), "tags": self.tags}


@dataclass
class ToolExecution:
    execution_id: str = field(default_factory=lambda: str(uuid4()))
    tool_name: str = ""
    agent_id: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    success: bool = False
    duration_ms: float = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ToolPipeline:
    pipeline_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)  # [{tool_name, params_mapping}]
    description: str = ""


class ToolRegistry:
    """Manages tool definitions, execution, and usage tracking."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable[..., Awaitable[Any]]] = {}
        self._executions: List[ToolExecution] = []
        self._pipelines: Dict[str, ToolPipeline] = {}
        self._usage: Dict[str, int] = defaultdict(int)

    # ---- registration ----
    def register(self, definition: ToolDefinition,
                 handler: Callable[..., Awaitable[Any]]) -> str:
        self._tools[definition.name] = definition
        self._handlers[definition.name] = handler
        logger.info("Tool registered: %s v%s", definition.name, definition.version)
        return definition.tool_id

    def unregister(self, tool_name: str) -> bool:
        self._tools.pop(tool_name, None)
        self._handlers.pop(tool_name, None)
        return True

    # ---- validation ----
    def validate_params(self, tool_name: str, params: Dict[str, Any]) -> List[str]:
        definition = self._tools.get(tool_name)
        if not definition:
            return [f"Tool not found: {tool_name}"]

        errors = []
        for p in definition.parameters:
            if p.required and p.name not in params:
                errors.append(f"Missing required parameter: {p.name}")
            if p.name in params and p.enum_values:
                if params[p.name] not in p.enum_values:
                    errors.append(f"{p.name} must be one of: {p.enum_values}")
        return errors

    # ---- execution ----
    async def execute(self, tool_name: str, params: Dict[str, Any], *,
                      agent_id: str = "") -> ToolExecution:
        import time as _time

        definition = self._tools.get(tool_name)
        handler = self._handlers.get(tool_name)
        execution = ToolExecution(tool_name=tool_name, agent_id=agent_id,
                                   parameters=params)

        if not definition or not handler:
            execution.error = f"Tool not found: {tool_name}"
            self._executions.append(execution)
            return execution

        if definition.status == ToolStatus.DISABLED:
            execution.error = "Tool is disabled"
            self._executions.append(execution)
            return execution

        errors = self.validate_params(tool_name, params)
        if errors:
            execution.error = ", ".join(errors)
            self._executions.append(execution)
            return execution

        start = _time.monotonic()
        try:
            import asyncio
            result = await asyncio.wait_for(
                handler(**params), timeout=definition.max_execution_time)
            execution.result = result
            execution.success = True
        except asyncio.TimeoutError:
            execution.error = f"Timeout after {definition.max_execution_time}s"
        except Exception as e:
            execution.error = str(e)

        execution.duration_ms = (_time.monotonic() - start) * 1000
        self._executions.append(execution)
        self._usage[tool_name] += 1

        if len(self._executions) > 10000:
            self._executions = self._executions[-10000:]

        return execution

    # ---- pipelines ----
    def create_pipeline(self, pipeline: ToolPipeline) -> str:
        self._pipelines[pipeline.pipeline_id] = pipeline
        return pipeline.pipeline_id

    async def execute_pipeline(self, pipeline_id: str, initial_params: Dict[str, Any], *,
                                agent_id: str = "") -> List[ToolExecution]:
        pipeline = self._pipelines.get(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline not found: {pipeline_id}")

        results = []
        context = dict(initial_params)
        for step in pipeline.steps:
            tool_name = step["tool_name"]
            param_mapping = step.get("params", {})
            params = {}
            for k, v in param_mapping.items():
                if isinstance(v, str) and v.startswith("$"):
                    params[k] = context.get(v[1:], v)
                else:
                    params[k] = v

            execution = await self.execute(tool_name, params, agent_id=agent_id)
            results.append(execution)
            if execution.success and isinstance(execution.result, dict):
                context.update(execution.result)
            elif not execution.success:
                break
        return results

    # ---- query ----
    def list_tools(self, *, category: ToolCategory | None = None,
                   status: ToolStatus | None = None) -> List[Dict[str, Any]]:
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        if status:
            tools = [t for t in tools if t.status == status]
        return [t.to_dict() for t in tools]

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()
                if t.status == ToolStatus.ACTIVE]

    def get_execution_history(self, *, tool_name: str = "",
                               agent_id: str = "", limit: int = 50) -> List[Dict[str, Any]]:
        execs = self._executions
        if tool_name:
            execs = [e for e in execs if e.tool_name == tool_name]
        if agent_id:
            execs = [e for e in execs if e.agent_id == agent_id]
        return [{"tool": e.tool_name, "success": e.success,
                 "duration_ms": round(e.duration_ms, 2),
                 "error": e.error, "timestamp": e.timestamp}
                for e in execs[-limit:]]

    def get_usage_stats(self) -> Dict[str, int]:
        return dict(self._usage)

    def get_metrics(self) -> Dict[str, Any]:
        total = len(self._executions)
        success = sum(1 for e in self._executions if e.success)
        return {"total_tools": len(self._tools), "total_executions": total,
                "success_rate_pct": round(success / max(1, total) * 100, 2),
                "pipelines": len(self._pipelines)}


_registry: ToolRegistry | None = None

def get_tool_registry() -> ToolRegistry:
    global _registry
    if not _registry:
        _registry = ToolRegistry()
    return _registry
