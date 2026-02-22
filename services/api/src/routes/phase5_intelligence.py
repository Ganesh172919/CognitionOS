"""
Phase 5 Intelligence API Routes
================================
REST endpoints for:
  - Federated Learning Engine      /api/v3/phase5/federated-learning/
  - Vector Search Engine           /api/v3/phase5/vector-search/
  - Prompt Engineering Platform    /api/v3/phase5/prompt-engineering/
  - Distributed Tracing Engine     /api/v3/phase5/tracing/
  - Zero-Trust Security Engine     /api/v3/phase5/zero-trust/
  - Load Testing Engine            /api/v3/phase5/load-testing/
  - Developer SDK Portal           /api/v3/phase5/sdk-portal/
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from infrastructure.federated_learning import (
    FederatedLearningEngine,
    FederationConfig,
    FederatedClient,
    ClientGradient,
    ModelWeights,
    DifferentialPrivacyConfig,
    AggregationAlgorithm,
    ClientSelectionStrategy,
)
from infrastructure.vector_search import (
    VectorSearchEngine,
    VectorRecord,
    SearchQuery,
    DistanceMetric,
)
from infrastructure.prompt_engineering import (
    PromptEngineeringPlatform,
    PromptVariable,
    PromptChain,
    ChainNode,
    ChainNodeType,
    ProviderModel,
)
from infrastructure.distributed_tracing import (
    DistributedTracingEngine,
    SpanKind,
    SamplingStrategy,
)
from infrastructure.zero_trust import (
    ZeroTrustSecurityEngine,
    AccessSubject,
    AccessResource,
    AccessRequest,
    SecurityPolicy,
    NetworkPolicy,
    NetworkPolicyAction,
    PolicyEffect,
    TrustLevel,
)
from infrastructure.load_testing import (
    LoadTestingEngine,
    LoadTestScenario,
    LoadStage,
    SLAAssertion,
    LoadPattern,
)
from infrastructure.sdk_portal import (
    DeveloperSDKPortal,
    APIEndpoint,
    APIParameter,
    SDKLanguage,
    APIStatus,
)

router = APIRouter(prefix="/api/v3/phase5", tags=["Phase 5 Intelligence"])

# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------
_federated_engine = FederatedLearningEngine()
_vector_engine = VectorSearchEngine()
_prompt_platform = PromptEngineeringPlatform()
_tracing_engine = DistributedTracingEngine(
    sampling_strategy=SamplingStrategy.ADAPTIVE, sampling_ratio=0.1
)
_zero_trust_engine = ZeroTrustSecurityEngine()
_load_test_engine = LoadTestingEngine()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_FEDERATION_ARCHITECTURE: Dict[str, Any] = {
    "layers": {
        "dense1": {"units": 64, "input_dim": 32},
        "dense2": {"units": 10, "input_dim": 64},
    }
}
_sdk_portal = DeveloperSDKPortal()


# ===========================================================================
# Pydantic request schemas
# ===========================================================================

class CreateFederationRequest(BaseModel):
    model_id: str
    rounds: int = 50
    clients_per_round: int = 5
    min_clients: int = 2
    aggregation: str = "fed_avg"
    dp_enabled: bool = True
    dp_epsilon: float = 1.0
    architecture: Optional[Dict[str, Any]] = None


class RegisterClientRequest(BaseModel):
    client_id: str
    tenant_id: str
    num_samples: int = 1000
    compute_capacity: float = 0.8
    bandwidth_mbps: float = 10.0


class SubmitGradientRequest(BaseModel):
    client_id: str
    tenant_id: str
    layer_name: str
    weights: List[float]
    num_samples: int = 100
    loss: float = 0.5


class VectorUpsertRequest(BaseModel):
    records: List[Dict[str, Any]]
    namespace: str = "default"
    tenant_id: str = "global"


class VectorSearchRequest(BaseModel):
    vector: List[float]
    top_k: int = 10
    namespace: str = "default"
    tenant_id: str = "global"
    metadata_filter: Optional[Dict[str, Any]] = None
    text_query: Optional[str] = None
    distance_metric: str = "cosine"
    min_score: float = 0.0


class RAGRequest(BaseModel):
    query: str
    vector: List[float]
    namespace: str = "default"
    tenant_id: str = "global"
    top_k: int = 5
    metadata_filter: Optional[Dict[str, Any]] = None


class CreatePromptRequest(BaseModel):
    name: str
    template: str
    description: str = ""
    category: str = "general"
    tenant_id: str = "global"
    tags: List[str] = []
    system_message: Optional[str] = None
    model: str = "gpt-4-turbo"


class ExecutePromptRequest(BaseModel):
    prompt_id: str
    variables: Dict[str, Any] = {}
    tenant_id: str = "global"
    version_id: Optional[str] = None


class CreateSpanRequest(BaseModel):
    operation_name: str
    service_name: str
    parent_trace_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    kind: str = "internal"
    tenant_id: str = "global"
    attributes: Dict[str, Any] = {}


class EndSpanRequest(BaseModel):
    span_id: str
    trace_id: str
    is_error: bool = False
    error_message: Optional[str] = None
    attributes: Dict[str, Any] = {}


class AuthorizeRequest(BaseModel):
    subject_id: str
    subject_type: str = "user"
    roles: List[str] = []
    resource_type: str
    resource_id: str
    action: str
    tenant_id: str = "global"
    ip_address: Optional[str] = None
    mfa_verified: bool = False
    trust_level: str = "medium"


class AddPolicyRequest(BaseModel):
    name: str
    effect: str = "allow"
    subjects: List[str] = []
    resources: List[str] = []
    actions: List[str] = []
    priority: int = 100
    tenant_id: str = "global"


class CreateLoadTestRequest(BaseModel):
    name: str
    base_url: str = "http://localhost:8000"
    endpoints: List[Dict[str, Any]]
    pattern: str = "steady_state"
    max_vus: int = 50
    duration_seconds: int = 60
    sla_p95_ms: Optional[float] = None
    sla_error_rate: Optional[float] = None


class RegisterDeveloperRequest(BaseModel):
    email: str
    name: str
    organization: str = ""
    tier: str = "free"


# ===========================================================================
# Federated Learning Endpoints
# ===========================================================================

@router.post("/federated-learning/federations")
async def create_federation(req: CreateFederationRequest) -> Dict[str, Any]:
    """Create a new federated learning job."""
    try:
        agg = AggregationAlgorithm(req.aggregation)
    except ValueError:
        agg = AggregationAlgorithm.FED_AVG

    dp = DifferentialPrivacyConfig(enabled=req.dp_enabled, epsilon=req.dp_epsilon)
    config = FederationConfig(
        model_id=req.model_id,
        rounds=req.rounds,
        clients_per_round=req.clients_per_round,
        min_clients=req.min_clients,
        aggregation=agg,
        dp_config=dp,
    )
    arch = req.architecture or _DEFAULT_FEDERATION_ARCHITECTURE
    model = await _federated_engine.create_federation(config, arch)
    return {
        "model_id": model.model_id,
        "status": model.status.value,
        "total_rounds": model.total_rounds,
        "dp_enabled": req.dp_enabled,
    }


@router.post("/federated-learning/clients")
async def register_client(req: RegisterClientRequest) -> Dict[str, Any]:
    """Register a federated learning client."""
    client = FederatedClient(
        client_id=req.client_id,
        tenant_id=req.tenant_id,
        num_samples=req.num_samples,
        compute_capacity=req.compute_capacity,
        bandwidth_mbps=req.bandwidth_mbps,
    )
    await _federated_engine.register_client(client)
    return {"client_id": req.client_id, "registered": True}


@router.post("/federated-learning/rounds/{model_id}")
async def start_round(model_id: str) -> Dict[str, Any]:
    """Start a new training round for a federated model."""
    try:
        round_obj = await _federated_engine.start_round(model_id)
        return {
            "round_id": round_obj.round_id,
            "round_number": round_obj.round_number,
            "selected_clients": round_obj.selected_clients,
            "status": round_obj.status.value,
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/federated-learning/rounds/{round_id}/gradients")
async def submit_gradient(round_id: str, req: SubmitGradientRequest) -> Dict[str, Any]:
    """Submit local gradient for aggregation."""
    gradient = ClientGradient(
        client_id=req.client_id,
        tenant_id=req.tenant_id,
        round_number=0,
        layer_gradients=[ModelWeights(req.layer_name, req.weights, [len(req.weights)])],
        num_samples=req.num_samples,
        loss=req.loss,
    )
    try:
        await _federated_engine.submit_gradient(round_id, gradient)
        return {"accepted": True, "round_id": round_id}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/federated-learning/rounds/{round_id}/aggregate")
async def aggregate_round(round_id: str) -> Dict[str, Any]:
    """Aggregate gradients and update global model."""
    try:
        round_obj = await _federated_engine.aggregate_round(round_id)
        return {
            "round_id": round_id,
            "status": round_obj.status.value,
            "global_loss": round_obj.global_loss,
            "participants": len(round_obj.received_gradients),
            "privacy_budget_used": round_obj.privacy_budget_used,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/federated-learning/models/{model_id}/privacy-report")
async def get_privacy_report(model_id: str) -> Dict[str, Any]:
    """Get differential privacy budget report."""
    report = await _federated_engine.get_privacy_report(model_id)
    if not report:
        raise HTTPException(status_code=404, detail="Model not found")
    return report


@router.get("/federated-learning/summary")
async def federated_summary() -> Dict[str, Any]:
    """Get federated learning engine summary."""
    return await _federated_engine.get_federation_summary()


# ===========================================================================
# Vector Search Endpoints
# ===========================================================================

@router.post("/vector-search/upsert")
async def vector_upsert(req: VectorUpsertRequest) -> Dict[str, Any]:
    """Upsert vectors into the store."""
    records = []
    for r in req.records:
        records.append(VectorRecord(
            record_id=r.get("id", str(uuid.uuid4())),
            vector=r.get("vector", []),
            metadata=r.get("metadata", {}),
            text=r.get("text"),
            namespace=req.namespace,
            tenant_id=req.tenant_id,
        ))
    result = await _vector_engine.upsert(records)
    return result


@router.post("/vector-search/search")
async def vector_search(req: VectorSearchRequest) -> Dict[str, Any]:
    """Search for similar vectors."""
    try:
        metric = DistanceMetric(req.distance_metric)
    except ValueError:
        metric = DistanceMetric.COSINE

    query = SearchQuery(
        vector=req.vector,
        top_k=req.top_k,
        namespace=req.namespace,
        tenant_id=req.tenant_id,
        metadata_filter=req.metadata_filter,
        text_query=req.text_query,
        distance_metric=metric,
        min_score=req.min_score,
    )
    results = await _vector_engine.search(query)
    return {
        "results": [
            {
                "id": r.record_id,
                "score": r.score,
                "distance": r.distance,
                "metadata": r.metadata,
                "text": r.text,
                "rank": r.rank,
            }
            for r in results
        ],
        "count": len(results),
    }


@router.post("/vector-search/rag")
async def rag_retrieve(req: RAGRequest) -> Dict[str, Any]:
    """RAG pipeline: retrieve relevant chunks and build augmented prompt."""
    ctx = await _vector_engine.rag_retrieve(
        req.query, req.vector, req.namespace, req.tenant_id, req.top_k, req.metadata_filter
    )
    return {
        "query": ctx.query,
        "augmented_prompt": ctx.augmented_prompt,
        "token_count": ctx.token_count,
        "retrieved_count": len(ctx.retrieved_chunks),
        "reranked_count": len(ctx.reranked_chunks),
        "retrieval_latency_ms": round(ctx.retrieval_latency_ms, 2),
        "total_latency_ms": round(ctx.total_latency_ms, 2),
    }


@router.get("/vector-search/namespaces/{tenant_id}")
async def list_namespaces(tenant_id: str) -> Dict[str, Any]:
    """List vector namespaces for a tenant."""
    namespaces = await _vector_engine.list_namespaces(tenant_id)
    return {"tenant_id": tenant_id, "namespaces": namespaces}


@router.get("/vector-search/summary")
async def vector_summary() -> Dict[str, Any]:
    """Get vector engine summary."""
    return await _vector_engine.get_engine_summary()


# ===========================================================================
# Prompt Engineering Endpoints
# ===========================================================================

@router.post("/prompt-engineering/prompts")
async def create_prompt(req: CreatePromptRequest) -> Dict[str, Any]:
    """Create a new versioned prompt."""
    try:
        model = ProviderModel(req.model)
    except ValueError:
        model = ProviderModel.GPT_4_TURBO

    prompt = await _prompt_platform.create_prompt(
        name=req.name,
        template=req.template,
        description=req.description,
        category=req.category,
        tenant_id=req.tenant_id,
        tags=req.tags,
        system_message=req.system_message,
        model=model,
    )
    return {
        "prompt_id": prompt.prompt_id,
        "name": prompt.name,
        "active_version_id": prompt.active_version_id,
    }


@router.post("/prompt-engineering/execute")
async def execute_prompt(req: ExecutePromptRequest) -> Dict[str, Any]:
    """Execute a prompt with variable substitution."""
    try:
        execution = await _prompt_platform.execute_prompt(
            req.prompt_id, req.variables, req.tenant_id, req.version_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return {
        "execution_id": execution.execution_id,
        "rendered_prompt": execution.rendered_prompt,
        "output": execution.raw_output,
        "input_tokens": execution.input_tokens,
        "output_tokens": execution.output_tokens,
        "cost_usd": execution.cost_usd,
        "latency_ms": execution.latency_ms,
        "safety_flags": execution.safety_flags,
    }


@router.get("/prompt-engineering/prompts")
async def list_prompts(tenant_id: str = "global", query: str = "") -> Dict[str, Any]:
    """List prompts with optional search."""
    prompts = _prompt_platform.library.search(query, tenant_id=tenant_id)
    return {
        "prompts": [
            {
                "id": p.prompt_id,
                "name": p.name,
                "category": p.category,
                "versions": len(p.versions),
                "executions": p.total_executions,
                "tags": p.tags,
            }
            for p in prompts
        ]
    }


@router.post("/prompt-engineering/ab-test")
async def create_ab_test(
    prompt_id: str,
    name: str,
    version_a: str,
    version_b: str,
    traffic_split: float = 0.5,
) -> Dict[str, Any]:
    """Create A/B test between two prompt versions."""
    test = _prompt_platform.ab_manager.create_test(prompt_id, name, version_a, version_b, traffic_split)
    return {"test_id": test.test_id, "status": test.status.value}


@router.get("/prompt-engineering/ab-test/{test_id}/analyze")
async def analyze_ab_test(test_id: str) -> Dict[str, Any]:
    """Get A/B test analysis."""
    return _prompt_platform.ab_manager.analyze(test_id)


@router.get("/prompt-engineering/cost-report")
async def prompt_cost_report(tenant_id: Optional[str] = None) -> Dict[str, Any]:
    """Get prompt execution cost report."""
    return _prompt_platform.get_cost_report(tenant_id)


@router.get("/prompt-engineering/summary")
async def prompt_summary() -> Dict[str, Any]:
    """Get prompt engineering platform summary."""
    return _prompt_platform.get_platform_summary()


# ===========================================================================
# Distributed Tracing Endpoints
# ===========================================================================

@router.post("/tracing/spans")
async def start_span(req: CreateSpanRequest) -> Dict[str, Any]:
    """Start a new tracing span."""
    try:
        kind = SpanKind(req.kind)
    except ValueError:
        kind = SpanKind.INTERNAL

    parent_ctx = None
    if req.parent_trace_id and req.parent_span_id:
        from infrastructure.distributed_tracing import TraceContext
        parent_ctx = TraceContext(req.parent_trace_id, req.parent_span_id)

    span, ctx = _tracing_engine.start_span(
        operation_name=req.operation_name,
        service_name=req.service_name,
        parent_context=parent_ctx,
        kind=kind,
        attributes=req.attributes,
        tenant_id=req.tenant_id,
    )
    return {
        "span_id": span.span_id,
        "trace_id": span.trace_id,
        "traceparent": ctx.to_traceparent(),
    }


@router.put("/tracing/spans/{span_id}/end")
async def end_span(span_id: str, req: EndSpanRequest) -> Dict[str, Any]:
    """End an active span."""
    span = _tracing_engine._active_spans.get(span_id)
    if not span:
        raise HTTPException(status_code=404, detail="Active span not found")
    for k, v in req.attributes.items():
        span.set_attribute(k, v)
    error = Exception(req.error_message) if req.is_error and req.error_message else None
    _tracing_engine.end_span(span, error)
    return {"span_id": span_id, "duration_ms": round(span.duration_ms, 3), "status": span.status.value}


@router.get("/tracing/traces/{trace_id}")
async def get_trace(trace_id: str) -> Dict[str, Any]:
    """Get a trace by ID."""
    trace = _tracing_engine.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {
        "trace_id": trace_id,
        "spans": len(trace.spans),
        "services": trace.service_names,
        "duration_ms": round(trace.total_duration_ms, 2),
        "has_errors": trace.has_errors,
    }


@router.get("/tracing/traces/{trace_id}/waterfall")
async def get_waterfall(trace_id: str) -> Dict[str, Any]:
    """Get waterfall timeline for a trace."""
    waterfall = _tracing_engine.get_waterfall(trace_id)
    if not waterfall:
        raise HTTPException(status_code=404, detail="Trace not found or has no spans")
    return {"trace_id": trace_id, "timeline": waterfall}


@router.get("/tracing/search")
async def search_traces(
    service: Optional[str] = None,
    operation: Optional[str] = None,
    has_error: Optional[bool] = None,
    min_duration_ms: Optional[float] = None,
    tenant_id: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """Search traces by filters."""
    traces = _tracing_engine.search_traces(
        service=service, operation=operation, has_error=has_error,
        min_duration_ms=min_duration_ms, tenant_id=tenant_id, limit=limit,
    )
    return {
        "traces": [
            {
                "trace_id": t.trace_id,
                "services": t.service_names,
                "duration_ms": round(t.total_duration_ms, 2),
                "spans": len(t.spans),
                "has_errors": t.has_errors,
            }
            for t in traces
        ]
    }


@router.get("/tracing/latency/{service}/{operation}")
async def get_latency_percentiles(service: str, operation: str) -> Dict[str, Any]:
    """Get latency percentiles for a service operation."""
    return _tracing_engine.get_latency_percentiles(service, operation)


@router.get("/tracing/service-graph")
async def get_service_graph() -> Dict[str, Any]:
    """Get service dependency graph built from traces."""
    return _tracing_engine.get_service_graph()


@router.get("/tracing/summary")
async def tracing_summary() -> Dict[str, Any]:
    """Get tracing engine summary."""
    return _tracing_engine.get_engine_summary()


# ===========================================================================
# Zero-Trust Security Endpoints
# ===========================================================================

@router.post("/zero-trust/authorize")
async def authorize_request(req: AuthorizeRequest) -> Dict[str, Any]:
    """Evaluate zero-trust authorization for a request."""
    try:
        trust_level = TrustLevel(req.trust_level)
    except ValueError:
        trust_level = TrustLevel.MEDIUM

    subject = AccessSubject(
        subject_id=req.subject_id,
        subject_type=req.subject_type,
        roles=req.roles,
        ip_address=req.ip_address,
        mfa_verified=req.mfa_verified,
        trust_level=trust_level,
    )
    resource = AccessResource(
        resource_type=req.resource_type,
        resource_id=req.resource_id,
        owner_tenant_id=req.tenant_id,
    )
    access_req = AccessRequest(
        subject=subject,
        resource=resource,
        action=req.action,
        tenant_id=req.tenant_id,
    )
    decision = await _zero_trust_engine.authorize(access_req)
    return {
        "allowed": decision.effect == PolicyEffect.ALLOW,
        "effect": decision.effect.value,
        "reason": decision.reason,
        "trust_score": decision.trust_score,
        "step_up_required": decision.step_up_required,
        "decision_latency_ms": round(decision.decision_latency_ms, 3),
    }


@router.post("/zero-trust/policies")
async def add_security_policy(req: AddPolicyRequest) -> Dict[str, Any]:
    """Add a new security policy."""
    try:
        effect = PolicyEffect(req.effect)
    except ValueError:
        effect = PolicyEffect.ALLOW

    policy = SecurityPolicy(
        name=req.name,
        effect=effect,
        subjects=req.subjects,
        resources=req.resources,
        actions=req.actions,
        priority=req.priority,
        tenant_id=req.tenant_id,
    )
    _zero_trust_engine.policy_engine.add_policy(policy)
    return {"policy_id": policy.policy_id, "name": policy.name, "effect": effect.value}


@router.post("/zero-trust/certificates/issue")
async def issue_certificate(service_name: str, namespace: str = "default", ttl_seconds: int = 3600) -> Dict[str, Any]:
    """Issue a workload identity certificate (SVID)."""
    identity = _zero_trust_engine.certificate_authority.issue(service_name, namespace, ttl_seconds)
    return {
        "spiffe_id": identity.spiffe_id,
        "serial": identity.certificate_serial,
        "fingerprint": identity.fingerprint,
        "expires_at": identity.expires_at,
        "ttl_seconds": int(identity.ttl_seconds),
    }


@router.delete("/zero-trust/certificates/{serial}/revoke")
async def revoke_certificate(serial: str) -> Dict[str, Any]:
    """Revoke a workload identity certificate."""
    success = _zero_trust_engine.certificate_authority.revoke(serial)
    return {"serial": serial, "revoked": success}


@router.post("/zero-trust/network/check")
async def check_network_access(
    source: str, dest: str, port: Optional[int] = None, protocol: str = "tcp"
) -> Dict[str, Any]:
    """Check if a service-to-service network connection is allowed."""
    allowed = _zero_trust_engine.check_network_access(source, dest, port, protocol)
    return {"source": source, "dest": dest, "allowed": allowed}


@router.get("/zero-trust/audit-log")
async def get_audit_log(
    subject_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    effect: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """Get security audit log."""
    records = _zero_trust_engine.get_audit_log(subject_id, tenant_id, effect, limit)
    return {
        "records": [
            {
                "record_id": r.record_id,
                "subject_id": r.subject_id,
                "action": r.action,
                "effect": r.effect,
                "reason": r.reason,
                "trust_score": r.trust_score,
                "timestamp": r.timestamp,
            }
            for r in records
        ]
    }


@router.get("/zero-trust/summary")
async def zero_trust_summary() -> Dict[str, Any]:
    """Get zero-trust engine summary."""
    return _zero_trust_engine.get_security_summary()


# ===========================================================================
# Load Testing Endpoints
# ===========================================================================

@router.post("/load-testing/scenarios")
async def create_load_test(req: CreateLoadTestRequest) -> Dict[str, Any]:
    """Create a load test scenario."""
    try:
        pattern = LoadPattern(req.pattern)
    except ValueError:
        pattern = LoadPattern.STEADY_STATE

    stages = _load_test_engine.generate_scenario_from_spec(pattern, req.max_vus, req.duration_seconds)
    sla_assertions = []
    if req.sla_p95_ms:
        sla_assertions.append(SLAAssertion(metric="p95_latency_ms", operator="lt", threshold=req.sla_p95_ms, name="P95 Latency"))
    if req.sla_error_rate:
        sla_assertions.append(SLAAssertion(metric="error_rate", operator="lt", threshold=req.sla_error_rate, name="Error Rate"))

    scenario = await _load_test_engine.create_scenario(
        name=req.name,
        base_url=req.base_url,
        endpoints=req.endpoints,
        stages=stages,
        sla_assertions=sla_assertions,
    )
    return {
        "scenario_id": scenario.scenario_id,
        "name": scenario.name,
        "stages": len(scenario.stages),
        "sla_assertions": len(scenario.sla_assertions),
    }


@router.post("/load-testing/scenarios/{scenario_id}/run")
async def run_load_test(scenario_id: str, max_requests: int = 200) -> Dict[str, Any]:
    """Execute a load test scenario."""
    try:
        metrics = await _load_test_engine.run_test(scenario_id, max_requests=max_requests)
        return await _load_test_engine.get_test_report(metrics.run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/load-testing/runs/{run_id}/report")
async def get_test_report(run_id: str) -> Dict[str, Any]:
    """Get a detailed load test report."""
    try:
        return await _load_test_engine.get_test_report(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/load-testing/runs/{run_id}/set-baseline")
async def set_baseline(run_id: str, scenario_name: str) -> Dict[str, Any]:
    """Set a test run as the performance baseline for regression detection."""
    success = _load_test_engine.set_baseline(scenario_name, run_id)
    return {"success": success, "run_id": run_id, "scenario": scenario_name}


@router.get("/load-testing/regressions/{run_id}")
async def detect_regressions(run_id: str, scenario_name: str) -> Dict[str, Any]:
    """Detect performance regressions against the baseline."""
    regressions = _load_test_engine.detect_regressions(scenario_name, run_id)
    return {"run_id": run_id, "regressions": regressions, "regression_count": len(regressions)}


@router.get("/load-testing/summary")
async def load_test_summary() -> Dict[str, Any]:
    """Get load testing engine summary."""
    return _load_test_engine.get_engine_summary()


# ===========================================================================
# Developer SDK Portal Endpoints
# ===========================================================================

@router.post("/sdk-portal/developers")
async def register_developer(req: RegisterDeveloperRequest) -> Dict[str, Any]:
    """Register a new developer account."""
    account = await _sdk_portal.register_developer(req.email, req.name, req.organization, req.tier)
    return {"account_id": account.account_id, "email": account.email, "tier": account.tier}


@router.post("/sdk-portal/developers/{account_id}/onboard")
async def complete_onboarding(account_id: str) -> Dict[str, Any]:
    """Complete developer onboarding and provision API key."""
    try:
        return await _sdk_portal.complete_onboarding(account_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/sdk-portal/sdk/{language}")
async def generate_sdk(language: str, tags: Optional[str] = None) -> Dict[str, Any]:
    """Generate SDK code for the specified language."""
    try:
        lang = SDKLanguage(language)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")

    tags_filter = tags.split(",") if tags else None
    code = _sdk_portal.generate_sdk(lang, tags_filter)
    return {"language": language, "code": code, "lines": len(code.splitlines())}


@router.get("/sdk-portal/openapi")
async def get_openapi_spec() -> Dict[str, Any]:
    """Get OpenAPI 3.0 specification."""
    return _sdk_portal.generate_openapi_spec()


@router.get("/sdk-portal/postman")
async def get_postman_collection() -> Dict[str, Any]:
    """Get Postman collection export."""
    return _sdk_portal.generate_postman_collection()


@router.post("/sdk-portal/sandbox/execute")
async def sandbox_execute(
    account_id: str,
    endpoint_path: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a request in the interactive sandbox."""
    req = await _sdk_portal.execute_sandbox_request(account_id, endpoint_path, method, params, body)
    return {
        "request_id": req.request_id,
        "status": req.response_status,
        "response": req.response_body,
        "latency_ms": round(req.latency_ms, 2),
    }


@router.get("/sdk-portal/snippet")
async def get_code_snippet(endpoint_path: str, method: str = "GET", language: str = "curl") -> Dict[str, Any]:
    """Get a code snippet for an endpoint."""
    try:
        lang = SDKLanguage(language)
    except ValueError:
        lang = SDKLanguage.CURL
    snippet = _sdk_portal.get_code_snippet(endpoint_path, method, lang)
    return {"endpoint": endpoint_path, "language": language, "snippet": snippet}


@router.get("/sdk-portal/changelog")
async def get_changelog(limit: int = 10) -> Dict[str, Any]:
    """Get API changelog."""
    entries = _sdk_portal.get_changelog(limit)
    return {
        "entries": [
            {"version": e.version, "date": e.date, "changes": e.changes}
            for e in entries
        ]
    }


@router.get("/sdk-portal/summary")
async def sdk_portal_summary() -> Dict[str, Any]:
    """Get developer SDK portal summary."""
    return _sdk_portal.get_portal_summary()


# ===========================================================================
# Phase 5 Aggregate Health
# ===========================================================================

@router.get("/health")
async def phase5_health() -> Dict[str, Any]:
    """Aggregate health check for all Phase 5 systems."""
    fed_summary = await _federated_engine.get_federation_summary()
    vec_summary = await _vector_engine.get_engine_summary()
    tr_summary = _tracing_engine.get_engine_summary()
    zt_summary = _zero_trust_engine.get_security_summary()
    lt_summary = _load_test_engine.get_engine_summary()
    portal_summary = _sdk_portal.get_portal_summary()
    prompt_summary = _prompt_platform.get_platform_summary()

    return {
        "status": "healthy",
        "phase": 5,
        "systems": {
            "federated_learning": {"models": fed_summary["total_models"], "clients": fed_summary["total_clients"]},
            "vector_search": {"indexes": vec_summary["total_indexes"], "vectors": vec_summary["total_vectors"]},
            "prompt_engineering": {"prompts": prompt_summary["total_prompts"], "executions": prompt_summary["total_executions"]},
            "distributed_tracing": {"traces": tr_summary["total_traces"], "spans": tr_summary["total_spans"]},
            "zero_trust": {"policies": zt_summary["policies"], "requests": zt_summary["total_requests"]},
            "load_testing": {"scenarios": lt_summary["total_scenarios"], "runs": lt_summary["total_runs"]},
            "sdk_portal": {"endpoints": portal_summary["registered_endpoints"], "developers": portal_summary["registered_developers"]},
        },
    }
