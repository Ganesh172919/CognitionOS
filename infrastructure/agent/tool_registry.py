"""
Agent Tool Registry

Dynamic tool discovery, registration, and sandboxed execution for autonomous agents.
Tools are typed, versioned, and schema-validated using JSON Schema.
Supports sync and async execution with timeout enforcement.
"""

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union


class ToolCategory(str, Enum):
    """Functional categories for tools"""
    CODE = "code"
    FILE_SYSTEM = "file_system"
    WEB = "web"
    DATABASE = "database"
    AI = "ai"
    SYSTEM = "system"
    COMMUNICATION = "communication"
    DATA = "data"
    ANALYTICS = "analytics"
    UTILITY = "utility"


class ToolStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DISABLED = "disabled"
    EXPERIMENTAL = "experimental"


@dataclass
class ToolParameter:
    """Definition of a single tool parameter"""
    name: str
    type: str           # JSON Schema type: string, number, boolean, array, object
    description: str
    required: bool = True
    default: Any = None
    enum_values: Optional[List[Any]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None

    def to_json_schema(self) -> Dict[str, Any]:
        schema: Dict[str, Any] = {"type": self.type, "description": self.description}
        if self.enum_values:
            schema["enum"] = self.enum_values
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.min_length is not None:
            schema["minLength"] = self.min_length
        if self.max_length is not None:
            schema["maxLength"] = self.max_length
        if self.pattern:
            schema["pattern"] = self.pattern
        if self.default is not None and not self.required:
            schema["default"] = self.default
        return schema


@dataclass
class ToolDefinition:
    """Complete definition of a registered tool"""
    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter]
    returns: str                      # Description of the return value
    version: str = "1.0.0"
    status: ToolStatus = ToolStatus.ACTIVE
    requires_approval: bool = False   # Human-in-the-loop gate
    max_execution_time_s: float = 30.0
    idempotent: bool = True
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_openai_tool_spec(self) -> Dict[str, Any]:
        """Export as OpenAI function-calling schema"""
        required = [p.name for p in self.parameters if p.required]
        properties = {p.name: p.to_json_schema() for p in self.parameters}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_anthropic_tool_spec(self) -> Dict[str, Any]:
        """Export as Anthropic tool-use schema"""
        required = [p.name for p in self.parameters if p.required]
        properties = {p.name: p.to_json_schema() for p in self.parameters}
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


@dataclass
class ToolExecutionResult:
    """Result of tool execution"""
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }


class ToolValidationError(ValueError):
    """Raised when tool input validation fails"""


class ToolNotFoundError(KeyError):
    """Raised when a requested tool is not registered"""


class ToolExecutionError(RuntimeError):
    """Raised when tool execution fails"""


class InputValidator:
    """JSON Schema-based input validator for tool parameters"""

    def validate(
        self,
        tool: ToolDefinition,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate and coerce inputs. Returns cleaned inputs."""
        cleaned: Dict[str, Any] = {}

        for param in tool.parameters:
            value = inputs.get(param.name)
            if value is None:
                if param.required:
                    raise ToolValidationError(
                        f"Missing required parameter '{param.name}' for tool '{tool.name}'"
                    )
                if param.default is not None:
                    cleaned[param.name] = param.default
                continue

            value = self._coerce(param, value)

            if param.minimum is not None and isinstance(value, (int, float)):
                if value < param.minimum:
                    raise ToolValidationError(
                        f"Parameter '{param.name}' is below minimum {param.minimum}"
                    )
            if param.maximum is not None and isinstance(value, (int, float)):
                if value > param.maximum:
                    raise ToolValidationError(
                        f"Parameter '{param.name}' exceeds maximum {param.maximum}"
                    )
            if param.enum_values is not None and value not in param.enum_values:
                raise ToolValidationError(
                    f"Parameter '{param.name}' must be one of {param.enum_values}"
                )

            cleaned[param.name] = value

        known = {p.name for p in tool.parameters}
        for key in inputs:
            if key not in known:
                raise ToolValidationError(
                    f"Unknown parameter '{key}' for tool '{tool.name}'"
                )

        return cleaned

    @staticmethod
    def _coerce(param: ToolParameter, value: Any) -> Any:
        try:
            if param.type == "number":
                return float(value)
            if param.type == "integer":
                return int(value)
            if param.type == "boolean":
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes")
                return bool(value)
            if param.type == "string":
                return str(value)
        except (ValueError, TypeError) as exc:
            raise ToolValidationError(
                f"Cannot coerce '{param.name}' to {param.type}: {exc}"
            ) from exc
        return value


ExecutorFn = Callable[..., Union[Any, Awaitable[Any]]]


class ToolRegistry:
    """
    Central registry for all agent tools.

    Features:
    - Registration with JSON Schema validation
    - Sync/async execution with timeout
    - Approval gate for sensitive tools
    - Execution metrics per tool
    - Category-based discovery
    - OpenAI / Anthropic spec export
    """

    def __init__(
        self,
        approval_callback: Optional[Callable[[str, Dict], Awaitable[bool]]] = None,
    ):
        self._tools: Dict[str, ToolDefinition] = {}
        self._executors: Dict[str, ExecutorFn] = {}
        self._metrics: Dict[str, Dict[str, Any]] = {}
        self._validator = InputValidator()
        self._approval_callback = approval_callback

    # ──────────────────────────────────────────────
    # Registration
    # ──────────────────────────────────────────────

    def register(
        self,
        definition: ToolDefinition,
        executor: ExecutorFn,
    ) -> None:
        """Register a tool with its executor function"""
        if definition.name in self._tools:
            raise ValueError(f"Tool '{definition.name}' is already registered")
        self._tools[definition.name] = definition
        self._executors[definition.name] = executor
        self._metrics[definition.name] = {
            "call_count": 0,
            "success_count": 0,
            "error_count": 0,
            "total_time_ms": 0.0,
            "avg_time_ms": 0.0,
        }

    def deregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._executors.pop(name, None)
        self._metrics.pop(name, None)

    def set_status(self, name: str, status: ToolStatus) -> None:
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' not found")
        self._tools[name].status = status

    # ──────────────────────────────────────────────
    # Discovery
    # ──────────────────────────────────────────────

    def get(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' not found in registry")
        return self._tools[name]

    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        status: Optional[ToolStatus] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ToolDefinition]:
        results = list(self._tools.values())
        if category:
            results = [t for t in results if t.category == category]
        if status:
            results = [t for t in results if t.status == status]
        if tags:
            results = [t for t in results if all(tag in t.tags for tag in tags)]
        return results

    def to_openai_tools(
        self,
        category: Optional[ToolCategory] = None,
    ) -> List[Dict[str, Any]]:
        tools = self.list_tools(category=category, status=ToolStatus.ACTIVE)
        return [t.to_openai_tool_spec() for t in tools]

    def to_anthropic_tools(
        self,
        category: Optional[ToolCategory] = None,
    ) -> List[Dict[str, Any]]:
        tools = self.list_tools(category=category, status=ToolStatus.ACTIVE)
        return [t.to_anthropic_tool_spec() for t in tools]

    # ──────────────────────────────────────────────
    # Execution
    # ──────────────────────────────────────────────

    async def execute(
        self,
        name: str,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolExecutionResult:
        """Validate inputs and execute a registered tool."""
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' not found")

        tool_def = self._tools[name]

        if tool_def.status == ToolStatus.DISABLED:
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                output=None,
                error=f"Tool '{name}' is disabled",
            )

        try:
            cleaned_inputs = self._validator.validate(tool_def, inputs)
        except ToolValidationError as exc:
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                output=None,
                error=str(exc),
            )

        if tool_def.requires_approval:
            approved = await self._request_approval(name, cleaned_inputs)
            if not approved:
                return ToolExecutionResult(
                    tool_name=name,
                    success=False,
                    output=None,
                    error=f"Execution of '{name}' was not approved",
                )

        start = time.perf_counter()
        try:
            executor = self._executors[name]
            if inspect.iscoroutinefunction(executor):
                result = await asyncio.wait_for(
                    executor(**cleaned_inputs),
                    timeout=tool_def.max_execution_time_s,
                )
            else:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: executor(**cleaned_inputs)),
                    timeout=tool_def.max_execution_time_s,
                )
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._record_metric(name, success=True, elapsed_ms=elapsed_ms)
            return ToolExecutionResult(
                tool_name=name,
                success=True,
                output=result,
                execution_time_ms=elapsed_ms,
                metadata={"context": context or {}},
            )
        except asyncio.TimeoutError:
            elapsed_ms = tool_def.max_execution_time_s * 1000
            self._record_metric(name, success=False, elapsed_ms=elapsed_ms)
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                output=None,
                error=f"Tool '{name}' exceeded timeout of {tool_def.max_execution_time_s}s",
                execution_time_ms=elapsed_ms,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._record_metric(name, success=False, elapsed_ms=elapsed_ms)
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                output=None,
                error=f"Tool execution failed: {exc}",
                execution_time_ms=elapsed_ms,
            )

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self._metrics)

    # ──────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────

    def _record_metric(self, name: str, success: bool, elapsed_ms: float) -> None:
        m = self._metrics[name]
        m["call_count"] += 1
        m["total_time_ms"] += elapsed_ms
        m["avg_time_ms"] = m["total_time_ms"] / m["call_count"]
        if success:
            m["success_count"] += 1
        else:
            m["error_count"] += 1

    async def _request_approval(self, name: str, inputs: Dict[str, Any]) -> bool:
        if self._approval_callback:
            return await self._approval_callback(name, inputs)
        return False


def tool(
    name: str,
    description: str,
    category: ToolCategory = ToolCategory.UTILITY,
    parameters: Optional[List[ToolParameter]] = None,
    returns: str = "Any",
    version: str = "1.0.0",
    status: ToolStatus = ToolStatus.ACTIVE,
    requires_approval: bool = False,
    max_execution_time_s: float = 30.0,
    tags: Optional[List[str]] = None,
) -> Callable:
    """
    Decorator that registers a function as an agent tool.

    Usage::

        @tool(
            name="search_web",
            description="Search the internet for information",
            category=ToolCategory.WEB,
            parameters=[ToolParameter("query", "string", "Search query")],
            returns="List of search results",
        )
        async def search_web(query: str) -> List[Dict]:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        inferred_params = parameters or []
        if not inferred_params:
            sig = inspect.signature(fn)
            for pname, param in sig.parameters.items():
                ann = param.annotation
                py_type = "string"
                if ann in (int,):
                    py_type = "integer"
                elif ann in (float,):
                    py_type = "number"
                elif ann in (bool,):
                    py_type = "boolean"
                inferred_params.append(
                    ToolParameter(
                        name=pname,
                        type=py_type,
                        description=f"Parameter {pname}",
                        required=param.default is inspect.Parameter.empty,
                        default=None if param.default is inspect.Parameter.empty else param.default,
                    )
                )

        defn = ToolDefinition(
            name=name,
            description=description,
            category=category,
            parameters=inferred_params,
            returns=returns,
            version=version,
            status=status,
            requires_approval=requires_approval,
            max_execution_time_s=max_execution_time_s,
            tags=tags or [],
        )
        fn._tool_definition = defn  # type: ignore[attr-defined]
        return fn

    return decorator


def build_registry_from_module(
    module: Any,
    approval_callback: Optional[Callable] = None,
) -> "ToolRegistry":
    """Auto-discover and register all @tool-decorated functions from a module."""
    registry = ToolRegistry(approval_callback=approval_callback)
    for attr_name in dir(module):
        fn = getattr(module, attr_name)
        if callable(fn) and hasattr(fn, "_tool_definition"):
            registry.register(fn._tool_definition, fn)
    return registry
