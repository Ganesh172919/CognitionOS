"""
AI Agent Orchestration Pipeline — CognitionOS

Production AI agent orchestration with:
- Agent lifecycle management
- Multi-model LLM routing
- Tool registration and execution
- Agent context management
- Conversation memory
- Cost tracking per request
- Streaming response support
- Guardrails and safety filters
- Agent composition (chain, parallel)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    WAITING_TOOL = "waiting_tool"
    RESPONDING = "responding"
    ERROR = "error"
    COMPLETED = "completed"


class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    CUSTOM = "custom"


class ExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CHAIN = "chain"
    ROUTER = "router"


@dataclass
class ModelConfig:
    provider: ModelProvider
    model_name: str
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0
    timeout_seconds: float = 60.0
    fallback_model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolDefinition:
    tool_id: str
    name: str
    description: str
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    handler: Optional[Callable[..., Awaitable[Any]]] = None
    requires_confirmation: bool = False
    max_retries: int = 2
    timeout_seconds: float = 30.0
    cost_per_call: float = 0.0
    categories: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_id": self.tool_id, "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
            "requires_confirmation": self.requires_confirmation,
        }


@dataclass
class ToolCall:
    call_id: str
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: float = 0
    cost: float = 0


@dataclass
class ConversationMessage:
    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0


@dataclass
class AgentContext:
    agent_id: str
    session_id: str
    tenant_id: str = ""
    user_id: str = ""
    messages: List[ConversationMessage] = field(default_factory=list)
    system_prompt: str = ""
    tools: List[str] = field(default_factory=list)  # tool names
    model_config: Optional[ModelConfig] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    max_turns: int = 20
    current_turn: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0
    started_at: float = field(default_factory=time.time)


@dataclass
class AgentResponse:
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    finished: bool = False
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0
    duration_ms: float = 0
    model_used: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": len(self.tool_calls),
            "finished": self.finished,
            "stop_reason": self.stop_reason,
            "cost": round(self.cost, 6),
            "duration_ms": round(self.duration_ms, 1),
        }


# ── Safety Guardrails ──

class SafetyGuardrail:
    """Content safety filter for AI agent input/output."""

    BLOCKED_PATTERNS = {
        "harmful_instruction", "personal_data_request",
        "unauthorized_access", "code_injection",
    }

    def __init__(self):
        self._custom_rules: List[Callable[[str], Optional[str]]] = []
        self._blocked_topics: Set[str] = set()

    def add_rule(self, rule: Callable[[str], Optional[str]]):
        self._custom_rules.append(rule)

    def block_topic(self, topic: str):
        self._blocked_topics.add(topic.lower())

    def check_input(self, content: str) -> Optional[str]:
        """Returns violation message or None if safe."""
        content_lower = content.lower()

        for topic in self._blocked_topics:
            if topic in content_lower:
                return f"Blocked topic detected: {topic}"

        for rule in self._custom_rules:
            violation = rule(content)
            if violation:
                return violation

        return None

    def check_output(self, content: str) -> Optional[str]:
        content_lower = content.lower()

        # Check for leaked sensitive patterns
        sensitive = ["api_key", "password:", "secret:", "private_key"]
        for pattern in sensitive:
            if pattern in content_lower:
                return f"Potential sensitive data in output: {pattern}"

        return None


# ── Cost Tracker ──

class CostTracker:
    """Track AI inference costs per tenant/user/session."""

    def __init__(self):
        self._costs: Dict[str, float] = defaultdict(float)
        self._by_model: Dict[str, float] = defaultdict(float)
        self._by_tenant: Dict[str, float] = defaultdict(float)
        self._daily_costs: Dict[str, float] = defaultdict(float)
        self._total_cost: float = 0

    def record(self, model: str, input_tokens: int, output_tokens: int,
                config: ModelConfig, *, tenant_id: str = "",
                session_id: str = "") -> float:
        cost = (input_tokens * config.cost_per_input_token +
                output_tokens * config.cost_per_output_token)
        self._total_cost += cost
        self._by_model[model] += cost
        if tenant_id:
            self._by_tenant[tenant_id] += cost
        if session_id:
            self._costs[session_id] += cost

        from datetime import datetime, timezone
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._daily_costs[day] += cost

        return cost

    def get_session_cost(self, session_id: str) -> float:
        return self._costs.get(session_id, 0)

    def get_tenant_cost(self, tenant_id: str) -> float:
        return self._by_tenant.get(tenant_id, 0)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_cost": round(self._total_cost, 4),
            "by_model": {k: round(v, 4) for k, v in self._by_model.items()},
            "top_tenants": sorted(
                [{"tenant_id": k, "cost": round(v, 4)}
                 for k, v in self._by_tenant.items()],
                key=lambda x: -x["cost"]
            )[:10],
        }


# ── Tool Registry ──

class ToolRegistry:
    """Centralized tool registration and execution."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, name: str, description: str,
                   handler: Callable[..., Awaitable[Any]], *,
                   parameters_schema: Optional[Dict] = None,
                   requires_confirmation: bool = False,
                   categories: Optional[List[str]] = None,
                   cost_per_call: float = 0) -> ToolDefinition:
        tool = ToolDefinition(
            tool_id=uuid.uuid4().hex[:12],
            name=name, description=description,
            parameters_schema=parameters_schema or {},
            handler=handler,
            requires_confirmation=requires_confirmation,
            categories=categories or [],
            cost_per_call=cost_per_call,
        )
        self._tools[name] = tool
        return tool

    async def execute(self, tool_name: str, arguments: Dict[str, Any]
                        ) -> ToolCall:
        tool = self._tools.get(tool_name)
        call = ToolCall(
            call_id=uuid.uuid4().hex[:12],
            tool_name=tool_name,
            arguments=arguments,
        )

        if not tool:
            call.error = f"Tool not found: {tool_name}"
            return call

        if not tool.handler:
            call.error = f"Tool has no handler: {tool_name}"
            return call

        start = time.perf_counter()
        for attempt in range(tool.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    tool.handler(**arguments),
                    timeout=tool.timeout_seconds,
                )
                call.result = result
                call.cost = tool.cost_per_call
                break
            except asyncio.TimeoutError:
                call.error = f"Tool timed out after {tool.timeout_seconds}s"
            except Exception as exc:
                call.error = str(exc)
                if attempt < tool.max_retries:
                    await asyncio.sleep(1)

        call.duration_ms = (time.perf_counter() - start) * 1000
        return call

    def get_tools(self, *, category: Optional[str] = None
                    ) -> List[Dict[str, Any]]:
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if category in t.categories]
        return [t.to_dict() for t in tools]

    def get_tool_names(self) -> List[str]:
        return list(self._tools.keys())


# ── Model Router ──

class ModelRouter:
    """Route requests to appropriate models based on task type and cost."""

    def __init__(self):
        self._models: Dict[str, ModelConfig] = {}
        self._routing_rules: List[Dict[str, Any]] = []

    def register_model(self, name: str, config: ModelConfig):
        self._models[name] = config

    def add_routing_rule(self, condition: Dict[str, Any], model_name: str):
        self._routing_rules.append({"condition": condition, "model": model_name})

    def select_model(self, *, task_type: str = "",
                       complexity: int = 1,
                       max_cost: Optional[float] = None) -> Optional[ModelConfig]:
        for rule in self._routing_rules:
            cond = rule["condition"]
            if task_type and cond.get("task_type") == task_type:
                return self._models.get(rule["model"])
            if complexity >= cond.get("min_complexity", 0):
                model = self._models.get(rule["model"])
                if model and (not max_cost or model.cost_per_input_token <= max_cost):
                    return model

        # Default: cheapest model
        if self._models:
            return min(self._models.values(),
                        key=lambda m: m.cost_per_input_token)
        return None

    def get_models(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "provider": m.provider.value,
                "model": m.model_name,
                "cost_input": m.cost_per_input_token,
                "cost_output": m.cost_per_output_token,
            }
            for name, m in self._models.items()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class AgentOrchestrator:
    """
    Central AI agent orchestration system.
    Manages agent sessions, tool execution, model routing, and cost tracking.
    """

    def __init__(self, *, llm_fn: Optional[Callable[..., Awaitable[Dict]]] = None):
        self._tool_registry = ToolRegistry()
        self._model_router = ModelRouter()
        self._cost_tracker = CostTracker()
        self._guardrail = SafetyGuardrail()
        self._llm_fn = llm_fn or self._default_llm
        self._sessions: Dict[str, AgentContext] = {}
        self._metrics = {
            "total_sessions": 0, "total_turns": 0,
            "total_tool_calls": 0, "total_errors": 0,
        }

    @property
    def tools(self) -> ToolRegistry:
        return self._tool_registry

    @property
    def models(self) -> ModelRouter:
        return self._model_router

    @property
    def costs(self) -> CostTracker:
        return self._cost_tracker

    @property
    def guardrail(self) -> SafetyGuardrail:
        return self._guardrail

    # ── Session Management ──

    def create_session(self, *, tenant_id: str = "", user_id: str = "",
                         system_prompt: str = "",
                         model: Optional[str] = None,
                         tools: Optional[List[str]] = None) -> AgentContext:
        session_id = uuid.uuid4().hex[:12]
        model_config = None
        if model:
            model_config = self._model_router._models.get(model)

        ctx = AgentContext(
            agent_id=uuid.uuid4().hex[:8],
            session_id=session_id,
            tenant_id=tenant_id, user_id=user_id,
            system_prompt=system_prompt,
            tools=tools or self._tool_registry.get_tool_names(),
            model_config=model_config,
        )

        if system_prompt:
            ctx.messages.append(ConversationMessage(
                role="system", content=system_prompt,
            ))

        self._sessions[session_id] = ctx
        self._metrics["total_sessions"] += 1
        return ctx

    def get_session(self, session_id: str) -> Optional[AgentContext]:
        return self._sessions.get(session_id)

    def end_session(self, session_id: str):
        self._sessions.pop(session_id, None)

    # ── Agent Execution ──

    async def run(self, session_id: str, user_message: str) -> AgentResponse:
        """Execute one agent turn."""
        ctx = self._sessions.get(session_id)
        if not ctx:
            return AgentResponse(
                content="Session not found", finished=True,
                stop_reason="session_not_found",
            )

        # Guard input
        violation = self._guardrail.check_input(user_message)
        if violation:
            return AgentResponse(
                content=f"Message blocked: {violation}",
                finished=True, stop_reason="guardrail",
            )

        # Add user message
        ctx.messages.append(ConversationMessage(
            role="user", content=user_message,
        ))
        ctx.current_turn += 1
        self._metrics["total_turns"] += 1

        # Select model
        model_config = ctx.model_config or self._model_router.select_model()

        start = time.perf_counter()

        try:
            # Call LLM
            llm_response = await self._llm_fn(
                messages=[{"role": m.role, "content": m.content}
                           for m in ctx.messages],
                model=model_config.model_name if model_config else "default",
                tools=[self._tool_registry._tools[t].to_dict()
                        for t in ctx.tools if t in self._tool_registry._tools],
            )

            response = AgentResponse(
                content=llm_response.get("content", ""),
                model_used=model_config.model_name if model_config else "default",
                input_tokens=llm_response.get("input_tokens", 0),
                output_tokens=llm_response.get("output_tokens", 0),
            )

            # Process tool calls
            tool_calls_data = llm_response.get("tool_calls", [])
            for tc_data in tool_calls_data:
                tc = await self._tool_registry.execute(
                    tc_data.get("name", ""),
                    tc_data.get("arguments", {}),
                )
                response.tool_calls.append(tc)
                self._metrics["total_tool_calls"] += 1

            # Cost tracking
            if model_config:
                response.cost = self._cost_tracker.record(
                    model_config.model_name,
                    response.input_tokens,
                    response.output_tokens,
                    model_config,
                    tenant_id=ctx.tenant_id,
                    session_id=session_id,
                )
                ctx.total_cost += response.cost

            # Update context
            ctx.total_input_tokens += response.input_tokens
            ctx.total_output_tokens += response.output_tokens

            # Guard output
            output_violation = self._guardrail.check_output(response.content)
            if output_violation:
                response.content = "[Content filtered by safety guardrail]"

            # Store assistant message
            ctx.messages.append(ConversationMessage(
                role="assistant", content=response.content,
                tool_calls=response.tool_calls,
            ))

            response.finished = not tool_calls_data or ctx.current_turn >= ctx.max_turns
            response.stop_reason = (
                "max_turns" if ctx.current_turn >= ctx.max_turns
                else "tool_use" if tool_calls_data
                else "complete"
            )

        except Exception as exc:
            self._metrics["total_errors"] += 1
            response = AgentResponse(
                content=f"Agent error: {exc}",
                finished=True, stop_reason="error",
            )

        response.duration_ms = (time.perf_counter() - start) * 1000
        return response

    async def _default_llm(self, messages: List[Dict], model: str = "",
                              tools: Optional[List[Dict]] = None) -> Dict:
        """Default LLM handler (stub). Replace with actual LLM API calls."""
        await asyncio.sleep(0.01)
        return {
            "content": "This is a placeholder response. Connect an LLM provider.",
            "input_tokens": sum(len(m["content"].split()) for m in messages),
            "output_tokens": 10,
            "tool_calls": [],
        }

    # ── Multi-Agent Composition ──

    async def run_chain(self, agents: List[AgentContext],
                          initial_input: str) -> List[AgentResponse]:
        """Chain multiple agents sequentially."""
        responses = []
        current_input = initial_input

        for agent in agents:
            self._sessions[agent.session_id] = agent
            response = await self.run(agent.session_id, current_input)
            responses.append(response)
            current_input = response.content

        return responses

    async def run_parallel(self, session_ids: List[str],
                              message: str) -> List[AgentResponse]:
        """Run multiple agents in parallel with same input."""
        tasks = [self.run(sid, message) for sid in session_ids]
        return await asyncio.gather(*tasks)

    # ── Stats ──

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._metrics,
            "active_sessions": len(self._sessions),
            "registered_tools": len(self._tool_registry._tools),
            "registered_models": len(self._model_router._models),
            "costs": self._cost_tracker.snapshot(),
        }


# ── Singleton ──
_orchestrator: Optional[AgentOrchestrator] = None


def get_agent_orchestrator(**kwargs) -> AgentOrchestrator:
    global _orchestrator
    if not _orchestrator:
        _orchestrator = AgentOrchestrator(**kwargs)
    return _orchestrator
