"""
Intelligence API Routes

Exposes the new intelligence layer:
- LLM Router: smart model selection, cost estimation
- Agent Tool Registry: tool discovery and execution
- Vector Memory: semantic memory search
- Telemetry: real-time metrics and alerts
- Rate Limiter: quota management
- Billing Meter: usage tracking and cost reporting
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v3/intelligence", tags=["Intelligence"])


# ──────────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────────


class LLMRouteRequest(BaseModel):
    messages: List[Dict[str, str]]
    strategy: str = "balanced"
    required_tags: Optional[List[str]] = None
    budget_usd: Optional[float] = None
    max_tokens: int = 1024
    temperature: float = 0.7


class LLMRouteDecisionResponse(BaseModel):
    model_id: str
    provider: str
    strategy_used: str
    estimated_cost_usd: float
    estimated_latency_ms: float
    confidence: float
    fallback_models: List[str]
    reasoning: str


class ToolExecuteRequest(BaseModel):
    tool_name: str
    inputs: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


class ToolExecuteResponse(BaseModel):
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time_ms: float


class MemoryStoreRequest(BaseModel):
    content: str
    memory_type: str = "observation"
    tier: str = "episodic"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: Optional[List[str]] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MemorySearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    tier: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    min_importance: float = 0.0
    min_similarity: float = 0.0


class AgentExecuteRequest(BaseModel):
    goal: str
    steps: List[Dict[str, Any]]
    variables: Optional[Dict[str, Any]] = None
    budget_usd: Optional[float] = None
    agent_id: str = "default"


class RateLimitCheckRequest(BaseModel):
    identity: str
    rule_name: str
    consume: float = 1.0


class BillingRecordRequest(BaseModel):
    tenant_id: str
    dimension: str
    quantity: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# ──────────────────────────────────────────────────────────────────────────────
# Lazy singletons
# ──────────────────────────────────────────────────────────────────────────────

_router_instance = None
_registry_instance = None
_memory_instance = None
_telemetry_instance = None
_rate_limiter_instance = None
_billing_meter_instance = None


def _get_llm_router():
    global _router_instance
    if _router_instance is None:
        from infrastructure.llm.router import IntelligentLLMRouter, RoutingStrategy
        _router_instance = IntelligentLLMRouter(default_strategy=RoutingStrategy.BALANCED)
    return _router_instance


def _get_tool_registry():
    global _registry_instance
    if _registry_instance is None:
        from infrastructure.agent.tool_registry import ToolRegistry
        _registry_instance = ToolRegistry()
    return _registry_instance


def _get_memory_store():
    global _memory_instance
    if _memory_instance is None:
        from infrastructure.agent.vector_memory import VectorMemoryStore
        _memory_instance = VectorMemoryStore()
    return _memory_instance


def _get_telemetry():
    global _telemetry_instance
    if _telemetry_instance is None:
        from infrastructure.telemetry import get_telemetry
        _telemetry_instance = get_telemetry()
    return _telemetry_instance


def _get_rate_limiter():
    global _rate_limiter_instance
    if _rate_limiter_instance is None:
        from infrastructure.middleware.advanced_rate_limiter import (
            build_pro_tier_limiter,
        )
        _rate_limiter_instance = build_pro_tier_limiter()
    return _rate_limiter_instance


def _get_billing_meter():
    global _billing_meter_instance
    if _billing_meter_instance is None:
        from infrastructure.billing.metering import BillingMeter, build_default_pricing, SubscriptionTier
        meter = BillingMeter()
        for tier, rules in build_default_pricing().items():
            meter.set_pricing(tier, rules)
        _billing_meter_instance = meter
    return _billing_meter_instance


# ──────────────────────────────────────────────────────────────────────────────
# LLM Router endpoints
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/llm/decide", response_model=LLMRouteDecisionResponse)
async def llm_routing_decision(request: LLMRouteRequest) -> LLMRouteDecisionResponse:
    """
    Get an LLM routing decision without executing the request.
    Returns the recommended model, cost estimate, and fallback chain.
    """
    from infrastructure.llm.router import IntelligentLLMRouter, RoutingStrategy
    from infrastructure.llm.provider import LLMRequest

    llm_router = _get_llm_router()
    strategy_map = {
        "cost_optimized": RoutingStrategy.COST_OPTIMIZED,
        "latency_optimized": RoutingStrategy.LATENCY_OPTIMIZED,
        "quality_optimized": RoutingStrategy.QUALITY_OPTIMIZED,
        "balanced": RoutingStrategy.BALANCED,
        "round_robin": RoutingStrategy.ROUND_ROBIN,
        "least_loaded": RoutingStrategy.LEAST_LOADED,
    }
    strategy = strategy_map.get(request.strategy, RoutingStrategy.BALANCED)

    llm_req = LLMRequest(
        messages=request.messages,
        model="",
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    decision = llm_router.decide(
        llm_req,
        strategy=strategy,
        required_tags=request.required_tags,
        budget_usd=request.budget_usd,
    )
    return LLMRouteDecisionResponse(
        model_id=decision.model_profile.model_id,
        provider=decision.model_profile.provider.value,
        strategy_used=decision.strategy_used.value,
        estimated_cost_usd=decision.estimated_cost_usd,
        estimated_latency_ms=decision.estimated_latency_ms,
        confidence=decision.confidence,
        fallback_models=decision.fallback_models,
        reasoning=decision.reasoning,
    )


@router.get("/llm/health")
async def llm_router_health() -> Dict[str, Any]:
    """Get health status of all registered LLM models"""
    return _get_llm_router().get_health_status()


@router.get("/llm/metrics")
async def llm_router_metrics() -> Dict[str, Any]:
    """Get aggregate LLM routing metrics"""
    return _get_llm_router().get_metrics()


# ──────────────────────────────────────────────────────────────────────────────
# Tool Registry endpoints
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/tools")
async def list_tools(
    category: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """List all registered agent tools"""
    from infrastructure.agent.tool_registry import ToolCategory, ToolStatus

    registry = _get_tool_registry()
    category_enum = ToolCategory(category) if category else None
    status_enum = ToolStatus(status_filter) if status_filter else None

    tools = registry.list_tools(category=category_enum, status=status_enum)
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "category": t.category.value,
                "status": t.status.value,
                "version": t.version,
                "parameters": [
                    {"name": p.name, "type": p.type, "required": p.required}
                    for p in t.parameters
                ],
                "tags": t.tags,
            }
            for t in tools
        ],
        "total": len(tools),
    }


@router.get("/tools/openai-spec")
async def tools_openai_spec() -> List[Dict[str, Any]]:
    """Export all active tools in OpenAI function-calling format"""
    return _get_tool_registry().to_openai_tools()


@router.get("/tools/anthropic-spec")
async def tools_anthropic_spec() -> List[Dict[str, Any]]:
    """Export all active tools in Anthropic tool-use format"""
    return _get_tool_registry().to_anthropic_tools()


@router.post("/tools/execute", response_model=ToolExecuteResponse)
async def execute_tool(request: ToolExecuteRequest) -> ToolExecuteResponse:
    """Execute a registered agent tool"""
    registry = _get_tool_registry()
    try:
        result = await registry.execute(
            request.tool_name, request.inputs, context=request.context
        )
        return ToolExecuteResponse(
            tool_name=result.tool_name,
            success=result.success,
            output=result.output,
            error=result.error,
            execution_time_ms=result.execution_time_ms,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/tools/metrics")
async def tool_metrics() -> Dict[str, Any]:
    """Get per-tool execution metrics"""
    return _get_tool_registry().get_metrics()


# ──────────────────────────────────────────────────────────────────────────────
# Vector Memory endpoints
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/memory/store")
async def store_memory(request: MemoryStoreRequest) -> Dict[str, Any]:
    """Store a new memory entry"""
    from infrastructure.agent.vector_memory import MemoryType, MemoryTier

    memory = _get_memory_store()
    try:
        memory_type = MemoryType(request.memory_type)
        tier = MemoryTier(request.tier)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    entry = memory.store(
        content=request.content,
        memory_type=memory_type,
        tier=tier,
        importance=request.importance,
        metadata=request.metadata or {},
        tags=request.tags or [],
        agent_id=request.agent_id,
        session_id=request.session_id,
    )
    return {
        "entry_id": entry.entry_id,
        "fingerprint": entry.fingerprint,
        "tier": entry.tier.value,
        "memory_type": entry.memory_type.value,
        "importance": entry.importance,
    }


@router.post("/memory/search")
async def search_memory(request: MemorySearchRequest) -> Dict[str, Any]:
    """Semantic search over stored memories"""
    from infrastructure.agent.vector_memory import MemoryTier

    memory = _get_memory_store()
    tier_enum = None
    if request.tier:
        try:
            tier_enum = MemoryTier(request.tier)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid tier: {request.tier}") from exc

    results = memory.search(
        query=request.query,
        top_k=request.top_k,
        tier=tier_enum,
        agent_id=request.agent_id,
        session_id=request.session_id,
        min_importance=request.min_importance,
        min_similarity=request.min_similarity,
    )
    return {
        "query": request.query,
        "results": [r.to_dict() for r in results],
        "total": len(results),
    }


@router.get("/memory/stats")
async def memory_stats() -> Dict[str, Any]:
    """Get memory store statistics"""
    return _get_memory_store().stats()


@router.get("/memory/context")
async def memory_context(
    query: str = Query(..., description="Query to build context for"),
    agent_id: Optional[str] = Query(None),
    max_tokens: int = Query(1000, ge=100, le=8000),
) -> Dict[str, Any]:
    """Build memory context string for LLM injection"""
    context = _get_memory_store().build_context(
        query=query, max_tokens=max_tokens, agent_id=agent_id
    )
    return {"query": query, "context": context, "char_count": len(context)}


@router.delete("/memory/tier/{tier_name}")
async def clear_memory_tier(
    tier_name: str,
    agent_id: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Clear all memories in a specific tier"""
    from infrastructure.agent.vector_memory import MemoryTier

    try:
        tier = MemoryTier(tier_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid tier: {tier_name}") from exc

    count = _get_memory_store().clear_tier(tier, agent_id=agent_id)
    return {"tier": tier_name, "cleared_count": count}


# ──────────────────────────────────────────────────────────────────────────────
# Agent Execution endpoints
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/agent/execute")
async def execute_agent(request: AgentExecuteRequest) -> Dict[str, Any]:
    """Execute an autonomous agent with a plan of steps"""
    from infrastructure.agent.execution_engine import (
        AgentExecutionEngine,
        ExecutionStep,
        StepType,
    )

    engine = AgentExecutionEngine(
        tool_registry=_get_tool_registry(),
        memory_store=_get_memory_store(),
    )

    steps = []
    for s in request.steps:
        try:
            step_type = StepType(s.get("step_type", "think"))
        except ValueError:
            step_type = StepType.THINK
        steps.append(ExecutionStep(
            step_id=s.get("step_id", str(time.time())),
            name=s.get("name", "unnamed"),
            step_type=step_type,
            payload=s.get("payload", {}),
            depends_on=s.get("depends_on", []),
            max_retries=s.get("max_retries", 1),
            timeout_s=s.get("timeout_s", 30.0),
            on_failure=s.get("on_failure", "abort"),
        ))

    ctx = await engine.execute(
        steps=steps,
        goal=request.goal,
        agent_id=request.agent_id,
        variables=request.variables,
        budget_usd=request.budget_usd,
    )
    return ctx.summary()


# ──────────────────────────────────────────────────────────────────────────────
# Telemetry endpoints
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/telemetry/snapshot")
async def telemetry_snapshot() -> Dict[str, Any]:
    """Get full telemetry snapshot"""
    return _get_telemetry().snapshot()


@router.get("/telemetry/prometheus")
async def telemetry_prometheus() -> StreamingResponse:
    """Render metrics in Prometheus text format"""
    text = _get_telemetry().prometheus_text()
    return StreamingResponse(
        iter([text]),
        media_type="text/plain; version=0.0.4",
    )


@router.get("/telemetry/samples")
async def telemetry_samples(
    metric_name: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=1000),
) -> Dict[str, Any]:
    """Get recent metric samples"""
    samples = _get_telemetry().recent_samples(name=metric_name, limit=limit)
    return {"samples": samples, "total": len(samples)}


@router.get("/telemetry/alerts")
async def telemetry_alerts() -> Dict[str, Any]:
    """Get currently active alerts"""
    alerts = _get_telemetry().get_active_alerts()
    return {"alerts": [a.to_dict() for a in alerts], "total": len(alerts)}


@router.get("/telemetry/stream")
async def telemetry_stream(
    interval_s: float = Query(5.0, ge=1.0, le=60.0),
) -> StreamingResponse:
    """Server-Sent Events stream of real-time telemetry"""
    tc = _get_telemetry()
    return StreamingResponse(
        tc.sse_stream(interval_s=interval_s),
        media_type="text/event-stream",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Rate Limiter endpoints
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/rate-limit/check")
async def check_rate_limit(request: RateLimitCheckRequest) -> Dict[str, Any]:
    """Check rate limit quota for an identity"""
    result = _get_rate_limiter().check(
        request.identity, request.rule_name, consume=request.consume
    )
    return result.to_dict()


@router.get("/rate-limit/quota/{identity}")
async def get_quota_status(identity: str) -> Dict[str, Any]:
    """Get full quota status for an identity"""
    return _get_rate_limiter().get_quota_status(identity)


# ──────────────────────────────────────────────────────────────────────────────
# Billing Meter endpoints
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/billing/record")
async def record_billing_usage(request: BillingRecordRequest) -> Dict[str, Any]:
    """Record a usage event for billing"""
    from infrastructure.billing.metering import BillingDimension

    try:
        dimension = BillingDimension(request.dimension)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid dimension: {request.dimension}") from exc

    record = _get_billing_meter().record(
        tenant_id=request.tenant_id,
        dimension=dimension,
        quantity=request.quantity,
        user_id=request.user_id,
        session_id=request.session_id,
        metadata=request.metadata,
    )
    return record.to_dict()


@router.get("/billing/usage/{tenant_id}")
async def get_billing_usage(
    tenant_id: str,
    period_days: int = Query(30, ge=1, le=365),
) -> Dict[str, Any]:
    """Get billing usage for a tenant"""
    return _get_billing_meter().get_current_usage(tenant_id, period_days=period_days)


@router.get("/billing/usage/{tenant_id}/history")
async def get_billing_history(
    tenant_id: str,
    days: int = Query(7, ge=1, le=90),
) -> Dict[str, Any]:
    """Get daily usage history for a tenant"""
    history = _get_billing_meter().get_usage_history(tenant_id, days=days)
    return {"tenant_id": tenant_id, "history": history, "days": days}


@router.get("/billing/usage/{tenant_id}/forecast")
async def forecast_billing(tenant_id: str) -> Dict[str, Any]:
    """Forecast end-of-month billing cost"""
    return _get_billing_meter().forecast_monthly_cost(tenant_id)


@router.get("/billing/invoice/{tenant_id}")
async def generate_invoice(
    tenant_id: str,
    period_days: int = Query(30, ge=1, le=365),
) -> Dict[str, Any]:
    """Generate a draft invoice for a tenant"""
    return _get_billing_meter().generate_invoice(tenant_id, period_days=period_days)
