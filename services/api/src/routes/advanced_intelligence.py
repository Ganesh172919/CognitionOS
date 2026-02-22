"""
Advanced Intelligence Systems API Routes
==========================================
API endpoints for Phase 4 enterprise systems:
- ML Pipeline & Model Registry
- Knowledge Graph Engine
- Self-Healing Infrastructure
- GraphQL Gateway
- Data Lakehouse
- Tenant Compliance Automation
- API Monetization Engine
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status as http_status
from pydantic import BaseModel

from infrastructure.ml_pipeline import (
    ModelRegistry,
    FeatureStore,
    ExperimentTracker,
    ModelVersion,
    ModelMetrics,
    ModelFramework,
    ModelStage,
    Feature,
    FeatureType,
    ExperimentRun,
)
from infrastructure.knowledge_graph import (
    KnowledgeGraph,
    GraphNode,
    GraphEdge,
    GraphQuery,
    NodeType,
    EdgeType,
    TraversalStrategy,
)
from infrastructure.self_healing import (
    SelfHealingEngine,
    HealthProbe,
    HealingAction,
    HealingPolicy,
    ProbeType,
    ActionType,
    IncidentState,
)
from infrastructure.graphql import (
    GraphQLGateway,
    SchemaRegistry,
)
from infrastructure.data_lakehouse import (
    DataLakehouse,
    LakehouseTable,
    DataSchema,
    ColumnDefinition,
    ColumnDataType,
    DataFormat,
    ETLJob,
)
from infrastructure.tenant_compliance import (
    TenantComplianceEngine,
    ComplianceFramework,
    DataResidencyRegion,
    PrivacyRequestType,
    AuditEvent,
)
from infrastructure.api_monetization import (
    APIMonetizationEngine,
    UsageMetric,
    InvoiceStatus,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v3/advanced-intelligence", tags=["Advanced Intelligence Systems"])

# ---------------------------------------------------------------------------
# Singleton managers
# ---------------------------------------------------------------------------

_model_registry = ModelRegistry()
_feature_store = FeatureStore()
_experiment_tracker = ExperimentTracker()
_knowledge_graph = KnowledgeGraph()
_self_healing = SelfHealingEngine()
_graphql_gateway = GraphQLGateway()
_data_lakehouse = DataLakehouse()
_compliance_engine = TenantComplianceEngine()
_monetization_engine = APIMonetizationEngine()

# Initialize GraphQL gateway on startup
_graphql_initialized = False


async def _ensure_graphql_initialized() -> None:
    global _graphql_initialized
    if not _graphql_initialized:
        await _graphql_gateway.initialize()
        _graphql_initialized = True


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class RegisterModelRequest(BaseModel):
    name: str
    description: str = ""
    owner: str = "system"
    tags: Dict[str, str] = {}


class CreateModelVersionRequest(BaseModel):
    model_name: str
    framework: str = "custom"
    hyperparameters: Dict[str, Any] = {}
    feature_schema: Dict[str, str] = {}
    description: str = ""
    training_dataset_uri: str = ""
    metrics: Optional[Dict[str, float]] = None
    created_by: str = "system"


class PromoteVersionRequest(BaseModel):
    target_stage: str = "production"
    traffic_percentage: float = 100.0


class ABTestRequest(BaseModel):
    champion_version_id: str
    challenger_version_id: str
    challenger_traffic_pct: float = 10.0


class RecordPredictionRequest(BaseModel):
    version_id: str
    prediction: Any
    ground_truth: Optional[Any] = None
    latency_ms: float = 0.0


class RegisterFeatureRequest(BaseModel):
    name: str
    feature_group: str
    feature_type: str = "numerical"
    description: str = ""
    entity_key: str = "user_id"
    ttl_seconds: int = 86400


class MaterializeFeaturesRequest(BaseModel):
    feature_group: str
    entity_key: str
    entity_value: str
    feature_values: Dict[str, Any]


class GetOnlineFeaturesRequest(BaseModel):
    feature_names: List[str]
    entity_key: str
    entity_value: str


class CreateExperimentRequest(BaseModel):
    name: str
    description: str = ""
    tags: Dict[str, str] = {}


class StartRunRequest(BaseModel):
    experiment_id: str
    run_name: str = ""
    parameters: Dict[str, Any] = {}


class LogMetricRequest(BaseModel):
    name: str
    value: float
    step: int = 0


class AddNodeRequest(BaseModel):
    name: str
    node_type: str = "concept"
    description: str = ""
    properties: Dict[str, Any] = {}
    tenant_id: str = "default"
    confidence: float = 1.0
    tags: List[str] = []


class AddEdgeRequest(BaseModel):
    source_id: str
    target_id: str
    edge_type: str = "related_to"
    weight: float = 1.0
    bidirectional: bool = False
    tenant_id: str = "default"


class TraverseGraphRequest(BaseModel):
    start_node_id: str
    max_depth: int = 5
    max_results: int = 20
    traversal: str = "bfs"
    end_node_id: Optional[str] = None
    edge_type_filter: Optional[str] = None
    node_type_filter: Optional[str] = None
    tenant_id: str = "default"


class RegisterProbeRequest(BaseModel):
    service_name: str
    probe_type: str = "http"
    target: str = ""
    interval_seconds: int = 30
    timeout_seconds: int = 10
    failure_threshold: int = 3


class RegisterPolicyRequest(BaseModel):
    service_name: str
    auto_heal: bool = True
    actions: List[Dict[str, Any]] = []


class ExecuteGraphQLRequest(BaseModel):
    query: str
    variables: Dict[str, Any] = {}
    operation_name: Optional[str] = None
    tenant_id: str = "default"


class CreateTableRequest(BaseModel):
    name: str
    database: str = "default"
    description: str = ""
    format: str = "delta"
    tenant_id: str = "default"
    owner: str = "system"
    schema_columns: Optional[List[Dict[str, Any]]] = None


class InsertRowsRequest(BaseModel):
    table_key: str
    rows: List[Dict[str, Any]]


class QueryLakehouseRequest(BaseModel):
    sql: str
    database: str = "default"


class RegisterETLJobRequest(BaseModel):
    name: str
    source_table: str
    target_table: str
    transform_sql: str = ""
    schedule_cron: str = ""


class InitComplianceRequest(BaseModel):
    tenant_id: str
    frameworks: List[str]
    data_residency: Optional[List[str]] = None


class SubmitPrivacyRequestSchema(BaseModel):
    tenant_id: str
    subject_id: str
    subject_email: str
    request_type: str
    data_categories: List[str] = []


class LogAuditEventRequest(BaseModel):
    tenant_id: str
    actor_id: str
    action: str
    resource_type: str
    resource_id: str
    outcome: str = "success"
    ip_address: str = ""
    metadata: Dict[str, Any] = {}


class OnboardTenantRequest(BaseModel):
    tenant_id: str
    tier_name: str = "free"
    initial_credits_usd: float = 0.0


class RecordUsageRequest(BaseModel):
    tenant_id: str
    metric: str
    quantity: float = 1.0
    endpoint: str = ""


class CreateAPIKeyRequest(BaseModel):
    tenant_id: str
    user_id: str
    name: str
    scopes: List[str] = ["read", "write"]
    rate_limit_per_minute: int = 1000
    expires_in_days: Optional[int] = None


# ===========================================================================
# ML PIPELINE ENDPOINTS
# ===========================================================================

@router.post("/ml/models")
async def register_model(req: RegisterModelRequest) -> Dict[str, Any]:
    """Register a new ML model in the registry."""
    try:
        model = await _model_registry.register_model(
            name=req.name,
            description=req.description,
            tags=req.tags,
            owner=req.owner,
        )
        return {"status": "success", "model": model}
    except ValueError as e:
        raise HTTPException(http_status.HTTP_409_CONFLICT, detail=str(e))


@router.post("/ml/models/{model_name}/versions")
async def create_model_version(
    model_name: str, req: CreateModelVersionRequest
) -> Dict[str, Any]:
    """Create a new model version."""
    metrics = None
    if req.metrics:
        metrics = ModelMetrics(**req.metrics)
    try:
        framework = ModelFramework(req.framework)
    except ValueError:
        framework = ModelFramework.CUSTOM
    version = await _model_registry.create_version(
        model_name=model_name,
        framework=framework,
        hyperparameters=req.hyperparameters,
        feature_schema=req.feature_schema,
        metrics=metrics,
        description=req.description,
        training_dataset_uri=req.training_dataset_uri,
        created_by=req.created_by,
    )
    return {"status": "success", "version": version.to_dict()}


@router.get("/ml/models")
async def list_models() -> Dict[str, Any]:
    """List all registered models."""
    models = await _model_registry.list_models()
    return {"models": models, "total": len(models)}


@router.get("/ml/models/{model_name}/versions")
async def list_model_versions(
    model_name: str,
    stage: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """List versions for a model."""
    stage_filter = ModelStage(stage) if stage else None
    versions = await _model_registry.list_versions(model_name, stage_filter)
    return {"versions": versions, "total": len(versions)}


@router.post("/ml/models/{model_name}/versions/{version_id}/promote")
async def promote_model_version(
    model_name: str, version_id: str, req: PromoteVersionRequest
) -> Dict[str, Any]:
    """Promote a model version to a new stage."""
    try:
        stage = ModelStage(req.target_stage)
    except ValueError:
        raise HTTPException(http_status.HTTP_400_BAD_REQUEST, detail=f"Invalid stage: {req.target_stage}")
    version = await _model_registry.promote_version(
        model_name, version_id, stage, req.traffic_percentage
    )
    return {"status": "success", "version": version.to_dict()}


@router.post("/ml/models/{model_name}/ab-test")
async def setup_ab_test(model_name: str, req: ABTestRequest) -> Dict[str, Any]:
    """Configure A/B test between champion and challenger models."""
    config = await _model_registry.setup_ab_test(
        model_name,
        req.champion_version_id,
        req.challenger_version_id,
        req.challenger_traffic_pct,
    )
    return {"status": "success", "ab_test": config}


@router.post("/ml/models/{model_name}/predictions")
async def record_prediction(model_name: str, req: RecordPredictionRequest) -> Dict[str, Any]:
    """Record a model prediction for drift detection."""
    await _model_registry.record_prediction(
        model_name, req.version_id, req.prediction, req.ground_truth, req.latency_ms
    )
    return {"status": "recorded"}


@router.get("/ml/models/{model_name}/serving-stats")
async def get_model_serving_stats(model_name: str) -> Dict[str, Any]:
    """Get real-time serving statistics for a model."""
    return await _model_registry.get_serving_stats(model_name)


@router.get("/ml/models/{model_name}/versions/{version_id}/lineage")
async def get_model_lineage(model_name: str, version_id: str) -> Dict[str, Any]:
    """Get the training lineage for a model version."""
    lineage = await _model_registry.get_model_lineage(model_name, version_id)
    return {"lineage": lineage, "depth": len(lineage)}


@router.get("/ml/registry/summary")
async def get_registry_summary() -> Dict[str, Any]:
    """Get high-level model registry summary."""
    return await _model_registry.get_registry_summary()


# Feature Store
@router.post("/ml/features")
async def register_feature(req: RegisterFeatureRequest) -> Dict[str, Any]:
    """Register a new feature in the feature store."""
    try:
        feat_type = FeatureType(req.feature_type)
    except ValueError:
        feat_type = FeatureType.NUMERICAL
    feature = Feature(
        name=req.name,
        feature_group=req.feature_group,
        feature_type=feat_type,
        description=req.description,
        entity_key=req.entity_key,
        ttl_seconds=req.ttl_seconds,
    )
    feature = await _feature_store.register_feature(feature)
    return {"status": "success", "feature": feature.to_dict()}


@router.post("/ml/features/materialize")
async def materialize_features(req: MaterializeFeaturesRequest) -> Dict[str, Any]:
    """Materialize feature values to the online store."""
    await _feature_store.materialize_features(
        req.feature_group, req.entity_key, req.entity_value, req.feature_values
    )
    return {"status": "materialized", "feature_count": len(req.feature_values)}


@router.post("/ml/features/get-online")
async def get_online_features(req: GetOnlineFeaturesRequest) -> Dict[str, Any]:
    """Get real-time features for inference."""
    features = await _feature_store.get_online_features(
        req.feature_names, req.entity_key, req.entity_value
    )
    return {"features": features, "requested": len(req.feature_names)}


@router.get("/ml/features")
async def list_features(feature_group: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    """List all registered features."""
    features = await _feature_store.list_features(feature_group)
    return {"features": features, "total": len(features)}


# Experiments
@router.post("/ml/experiments")
async def create_experiment(req: CreateExperimentRequest) -> Dict[str, Any]:
    """Create a new ML experiment."""
    exp_id = await _experiment_tracker.create_experiment(req.name, req.description, req.tags)
    return {"status": "created", "experiment_id": exp_id}


@router.post("/ml/experiments/runs")
async def start_experiment_run(req: StartRunRequest) -> Dict[str, Any]:
    """Start a new experiment run."""
    run = await _experiment_tracker.start_run(
        req.experiment_id, req.run_name, req.parameters
    )
    return {"status": "started", "run": run.to_dict()}


@router.post("/ml/experiments/runs/{run_id}/metrics")
async def log_run_metric(run_id: str, req: LogMetricRequest) -> Dict[str, Any]:
    """Log a metric for an experiment run."""
    await _experiment_tracker.log_metric(run_id, req.name, req.value, req.step)
    return {"status": "logged"}


@router.post("/ml/experiments/runs/{run_id}/end")
async def end_experiment_run(
    run_id: str,
    run_status: str = Query(default="completed"),
) -> Dict[str, Any]:
    """End an experiment run."""
    run = await _experiment_tracker.end_run(run_id, run_status)
    return {"status": "ended", "run": run.to_dict()}


@router.get("/ml/experiments/{experiment_id}/runs")
async def list_experiment_runs(experiment_id: str, limit: int = 20) -> Dict[str, Any]:
    """List runs for an experiment."""
    runs = await _experiment_tracker.list_runs(experiment_id, limit)
    return {"runs": runs, "total": len(runs)}


# ===========================================================================
# KNOWLEDGE GRAPH ENDPOINTS
# ===========================================================================

@router.post("/knowledge-graph/nodes")
async def add_graph_node(req: AddNodeRequest) -> Dict[str, Any]:
    """Add a node to the knowledge graph."""
    try:
        node_type = NodeType(req.node_type)
    except ValueError:
        node_type = NodeType.CONCEPT
    node = GraphNode(
        name=req.name,
        node_type=node_type,
        description=req.description,
        properties=req.properties,
        tenant_id=req.tenant_id,
        confidence=req.confidence,
        tags=req.tags,
    )
    node = await _knowledge_graph.add_node(node)
    return {"status": "added", "node": node.to_dict()}


@router.get("/knowledge-graph/nodes/{node_id}")
async def get_graph_node(node_id: str) -> Dict[str, Any]:
    """Get a knowledge graph node by ID."""
    node = await _knowledge_graph.get_node(node_id)
    if not node:
        raise HTTPException(http_status.HTTP_404_NOT_FOUND, detail=f"Node {node_id} not found")
    return node.to_dict()


@router.post("/knowledge-graph/edges")
async def add_graph_edge(req: AddEdgeRequest) -> Dict[str, Any]:
    """Add an edge between two knowledge graph nodes."""
    try:
        edge_type = EdgeType(req.edge_type)
    except ValueError:
        edge_type = EdgeType.RELATED_TO
    edge = GraphEdge(
        source_id=req.source_id,
        target_id=req.target_id,
        edge_type=edge_type,
        weight=req.weight,
        bidirectional=req.bidirectional,
        tenant_id=req.tenant_id,
    )
    try:
        edge = await _knowledge_graph.add_edge(edge)
    except ValueError as e:
        raise HTTPException(http_status.HTTP_400_BAD_REQUEST, detail=str(e))
    return {"status": "added", "edge": edge.to_dict()}


@router.post("/knowledge-graph/traverse")
async def traverse_knowledge_graph(req: TraverseGraphRequest) -> Dict[str, Any]:
    """Traverse the knowledge graph from a starting node."""
    try:
        traversal = TraversalStrategy(req.traversal)
    except ValueError:
        traversal = TraversalStrategy.BFS
    query = GraphQuery(
        start_node_id=req.start_node_id,
        end_node_id=req.end_node_id,
        max_depth=req.max_depth,
        max_results=req.max_results,
        traversal=traversal,
        tenant_id=req.tenant_id,
    )
    result = await _knowledge_graph.traverse(query)
    return result.to_dict()


@router.get("/knowledge-graph/search")
async def search_knowledge_graph(
    q: str = Query(..., description="Search query"),
    tenant_id: str = Query(default="default"),
    limit: int = Query(default=20),
) -> Dict[str, Any]:
    """Search nodes in the knowledge graph by text."""
    nodes = await _knowledge_graph.text_search(q, tenant_id, limit)
    return {"nodes": [n.to_dict() for n in nodes], "total": len(nodes)}


@router.get("/knowledge-graph/analytics")
async def get_graph_analytics(
    tenant_id: str = Query(default="default"),
) -> Dict[str, Any]:
    """Get knowledge graph analytics (PageRank, centrality, communities)."""
    return await _knowledge_graph.compute_analytics(tenant_id)


@router.post("/knowledge-graph/inference")
async def run_graph_inference(
    tenant_id: str = Query(default="default"),
    min_confidence: float = Query(default=0.5),
) -> Dict[str, Any]:
    """Run inference engine to discover new facts."""
    facts = await _knowledge_graph.run_inference(tenant_id, min_confidence)
    return {"inferred_facts": facts, "total": len(facts)}


@router.post("/knowledge-graph/extract-from-text")
async def extract_from_text(
    text: str,
    tenant_id: str = Query(default="default"),
) -> Dict[str, Any]:
    """Extract entities from text and add to knowledge graph."""
    nodes = await _knowledge_graph.extract_from_text(text, tenant_id)
    return {"extracted_nodes": [n.to_dict() for n in nodes], "total": len(nodes)}


@router.get("/knowledge-graph/summary")
async def get_graph_summary(
    tenant_id: str = Query(default="default"),
) -> Dict[str, Any]:
    """Get knowledge graph summary."""
    return await _knowledge_graph.get_graph_summary(tenant_id)


@router.get("/knowledge-graph/nodes")
async def list_graph_nodes(
    tenant_id: str = Query(default="default"),
    node_type: Optional[str] = Query(default=None),
    limit: int = Query(default=50),
    offset: int = Query(default=0),
) -> Dict[str, Any]:
    """List knowledge graph nodes."""
    nt = NodeType(node_type) if node_type else None
    nodes = await _knowledge_graph.list_nodes(tenant_id, nt, limit, offset)
    return {"nodes": nodes, "total": len(nodes)}


@router.get("/knowledge-graph/path")
async def find_graph_path(
    start_node_id: str,
    end_node_id: str,
    tenant_id: str = Query(default="default"),
    max_depth: int = Query(default=8),
) -> Dict[str, Any]:
    """Find path between two knowledge graph nodes."""
    path = await _knowledge_graph.find_path(start_node_id, end_node_id, tenant_id, max_depth)
    if path is None:
        return {"path": None, "found": False}
    return {"path": path, "found": True, "hops": len(path) - 1}


# ===========================================================================
# SELF-HEALING ENDPOINTS
# ===========================================================================

@router.post("/self-healing/probes")
async def register_health_probe(req: RegisterProbeRequest) -> Dict[str, Any]:
    """Register a health probe for a service."""
    try:
        probe_type = ProbeType(req.probe_type)
    except ValueError:
        probe_type = ProbeType.HTTP
    probe = HealthProbe(
        service_name=req.service_name,
        probe_type=probe_type,
        target=req.target,
        interval_seconds=req.interval_seconds,
        timeout_seconds=req.timeout_seconds,
        consecutive_failures_threshold=req.failure_threshold,
    )
    probe = await _self_healing.register_probe(probe)
    return {"status": "registered", "probe": probe.to_dict()}


@router.post("/self-healing/policies")
async def register_healing_policy(req: RegisterPolicyRequest) -> Dict[str, Any]:
    """Register an auto-healing policy for a service."""
    actions = []
    for action_config in req.actions:
        try:
            action_type = ActionType(action_config.get("action_type", "restart_service"))
        except ValueError:
            action_type = ActionType.RESTART_SERVICE
        actions.append(HealingAction(
            service_name=req.service_name,
            action_type=action_type,
            priority=action_config.get("priority", 50),
            parameters=action_config.get("parameters", {}),
        ))
    policy = HealingPolicy(
        service_name=req.service_name,
        auto_heal=req.auto_heal,
        actions=actions,
    )
    policy = await _self_healing.register_policy(policy)
    return {"status": "registered", "policy_id": policy.policy_id}


@router.post("/self-healing/probe-cycle")
async def run_probe_cycle() -> Dict[str, Any]:
    """Execute a probe cycle for all registered services."""
    results = await _self_healing.run_probe_cycle()
    return {"status": "completed", "results": results}


@router.get("/self-healing/system-health")
async def get_system_health_report() -> Dict[str, Any]:
    """Get the overall system health report."""
    report = await _self_healing.get_system_health_report()
    return report.to_dict()


@router.get("/self-healing/services")
async def list_monitored_services() -> Dict[str, Any]:
    """List all monitored services and their health status."""
    services = await _self_healing.list_services()
    return {"services": services, "total": len(services)}


@router.get("/self-healing/services/{service_name}")
async def get_service_health(service_name: str) -> Dict[str, Any]:
    """Get health status for a specific service."""
    health = await _self_healing.get_service_health(service_name)
    if not health:
        raise HTTPException(http_status.HTTP_404_NOT_FOUND, detail=f"Service {service_name} not monitored")
    return health


@router.get("/self-healing/incidents")
async def list_incidents(
    incident_state: Optional[str] = Query(default=None),
    limit: int = Query(default=20),
) -> Dict[str, Any]:
    """List incidents."""
    state_filter = IncidentState(incident_state) if incident_state else None
    incidents = await _self_healing.list_incidents(state_filter, limit)
    return {"incidents": incidents, "total": len(incidents)}


@router.post("/self-healing/incidents/{incident_id}/resolve")
async def resolve_incident(
    incident_id: str,
    summary: str = Query(default="Manually resolved"),
    root_cause: str = Query(default=""),
) -> Dict[str, Any]:
    """Resolve an incident."""
    incident = await _self_healing.resolve_incident(incident_id, summary, root_cause)
    if not incident:
        raise HTTPException(http_status.HTTP_404_NOT_FOUND, detail="Incident not found")
    return {"status": "resolved", "incident": incident.to_dict()}


@router.get("/self-healing/statistics")
async def get_healing_statistics() -> Dict[str, Any]:
    """Get overall self-healing statistics."""
    return await _self_healing.get_healing_statistics()


# ===========================================================================
# GRAPHQL GATEWAY ENDPOINTS
# ===========================================================================

@router.post("/graphql")
async def execute_graphql(req: ExecuteGraphQLRequest) -> Dict[str, Any]:
    """Execute a GraphQL query or mutation."""
    await _ensure_graphql_initialized()
    result = await _graphql_gateway.execute(
        query=req.query,
        variables=req.variables,
        operation_name=req.operation_name,
        context={"tenant_id": req.tenant_id},
        tenant_id=req.tenant_id,
    )
    return result.to_dict()


@router.get("/graphql/schema")
async def get_graphql_schema(schema_name: str = Query(default="default")) -> Dict[str, Any]:
    """Get the GraphQL Schema Definition Language (SDL)."""
    await _ensure_graphql_initialized()
    sdl = await _graphql_gateway.get_schema_sdl(schema_name)
    return {"sdl": sdl}


@router.get("/graphql/introspect")
async def introspect_graphql_schema() -> Dict[str, Any]:
    """Introspect the GraphQL schema."""
    await _ensure_graphql_initialized()
    return await _graphql_gateway.introspect()


@router.get("/graphql/stats")
async def get_graphql_gateway_stats() -> Dict[str, Any]:
    """Get GraphQL gateway operational statistics."""
    await _ensure_graphql_initialized()
    return await _graphql_gateway.get_gateway_stats()


@router.get("/graphql/history")
async def get_graphql_execution_history(limit: int = Query(default=20)) -> Dict[str, Any]:
    """Get recent GraphQL execution history."""
    await _ensure_graphql_initialized()
    history = await _graphql_gateway.get_execution_history(limit)
    return {"history": history, "total": len(history)}


# ===========================================================================
# DATA LAKEHOUSE ENDPOINTS
# ===========================================================================

@router.post("/lakehouse/tables")
async def create_lakehouse_table(req: CreateTableRequest) -> Dict[str, Any]:
    """Create a new table in the data lakehouse."""
    schema = None
    if req.schema_columns:
        columns = []
        for col_def in req.schema_columns:
            try:
                dtype = ColumnDataType(col_def.get("data_type", "string"))
            except ValueError:
                dtype = ColumnDataType.STRING
            columns.append(ColumnDefinition(
                name=col_def["name"],
                data_type=dtype,
                nullable=col_def.get("nullable", True),
                description=col_def.get("description", ""),
            ))
        schema = DataSchema(name=req.name, columns=columns)

    try:
        fmt = DataFormat(req.format)
    except ValueError:
        fmt = DataFormat.DELTA

    table = LakehouseTable(
        name=req.name,
        database=req.database,
        description=req.description,
        format=fmt,
        tenant_id=req.tenant_id,
        owner=req.owner,
        schema=schema,
    )
    try:
        table = await _data_lakehouse.create_table(table)
    except ValueError as e:
        raise HTTPException(http_status.HTTP_409_CONFLICT, detail=str(e))
    return {"status": "created", "table": table.to_dict()}


@router.get("/lakehouse/tables")
async def list_lakehouse_tables(
    database: str = Query(default="default"),
    tenant_id: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """List all tables in a lakehouse database."""
    tables = await _data_lakehouse.list_tables(database, tenant_id)
    return {"tables": tables, "total": len(tables)}


@router.post("/lakehouse/tables/{table_key}/insert")
async def insert_lakehouse_rows(table_key: str, req: InsertRowsRequest) -> Dict[str, Any]:
    """Insert rows into a lakehouse table."""
    result = await _data_lakehouse.insert_rows(table_key, req.rows)
    return {"status": "inserted", **result}


@router.get("/lakehouse/tables/{table_key}/scan")
async def scan_lakehouse_table(
    table_key: str,
    limit: Optional[int] = Query(default=100),
) -> Dict[str, Any]:
    """Scan all rows from a lakehouse table."""
    rows = await _data_lakehouse.scan_table(table_key, limit=limit)
    return {"rows": rows, "total": len(rows)}


@router.post("/lakehouse/query")
async def query_lakehouse(req: QueryLakehouseRequest) -> Dict[str, Any]:
    """Execute a SQL query against the lakehouse."""
    try:
        rows, plan = await _data_lakehouse.query(req.sql, req.database)
        return {
            "rows": rows,
            "row_count": len(rows),
            "plan": plan.to_dict(),
        }
    except Exception as e:
        raise HTTPException(http_status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/lakehouse/tables/{table_key}/compact")
async def compact_lakehouse_table(table_key: str) -> Dict[str, Any]:
    """Compact small partitions in a table."""
    result = await _data_lakehouse.compact_table(table_key)
    return {"status": "compacted", **result}


@router.get("/lakehouse/tables/{table_key}/stats")
async def get_table_stats(table_key: str) -> Dict[str, Any]:
    """Get statistics for a lakehouse table."""
    return await _data_lakehouse.get_table_stats(table_key)


@router.get("/lakehouse/summary")
async def get_lakehouse_summary() -> Dict[str, Any]:
    """Get high-level lakehouse summary."""
    return await _data_lakehouse.get_lakehouse_summary()


@router.post("/lakehouse/etl/jobs")
async def register_etl_job(req: RegisterETLJobRequest) -> Dict[str, Any]:
    """Register an ETL pipeline job."""
    job = ETLJob(
        name=req.name,
        source_table=req.source_table,
        target_table=req.target_table,
        transform_sql=req.transform_sql,
        schedule_cron=req.schedule_cron,
    )
    job = await _data_lakehouse.etl.register_job(job)
    return {"status": "registered", "job": job.to_dict()}


@router.post("/lakehouse/etl/jobs/{job_id}/run")
async def run_etl_job(job_id: str) -> Dict[str, Any]:
    """Execute an ETL job."""
    result = await _data_lakehouse.etl.run_job(job_id, _data_lakehouse)
    return result


@router.get("/lakehouse/catalog/search")
async def search_data_catalog(
    q: str = Query(...),
    asset_type: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Search the data catalog."""
    results = _data_lakehouse.catalog.search(q, asset_type)
    return {"results": results, "total": len(results)}


@router.get("/lakehouse/query-history")
async def get_query_history(limit: int = Query(default=20)) -> Dict[str, Any]:
    """Get recent query execution history."""
    history = _data_lakehouse.query_engine.get_query_history(limit)
    return {"history": history, "total": len(history)}


# ===========================================================================
# TENANT COMPLIANCE ENDPOINTS
# ===========================================================================

@router.post("/compliance/initialize")
async def initialize_compliance(req: InitComplianceRequest) -> Dict[str, Any]:
    """Initialize compliance controls for a tenant."""
    frameworks = []
    for f in req.frameworks:
        try:
            frameworks.append(ComplianceFramework(f))
        except ValueError:
            pass
    residency = []
    for r in (req.data_residency or []):
        try:
            residency.append(DataResidencyRegion(r))
        except ValueError:
            pass
    result = await _compliance_engine.initialize_tenant_compliance(
        req.tenant_id, frameworks, residency or None
    )
    return {"status": "initialized", **result}


@router.post("/compliance/{tenant_id}/assess/{framework}")
async def run_compliance_assessment(tenant_id: str, framework: str) -> Dict[str, Any]:
    """Run a compliance assessment for a tenant and framework."""
    try:
        fw = ComplianceFramework(framework)
    except ValueError:
        raise HTTPException(http_status.HTTP_400_BAD_REQUEST, detail=f"Unknown framework: {framework}")
    assessment = await _compliance_engine.run_automated_assessment(tenant_id, fw)
    return {"status": "completed", "assessment": assessment.to_dict()}


@router.get("/compliance/{tenant_id}/dashboard")
async def get_compliance_dashboard(tenant_id: str) -> Dict[str, Any]:
    """Get compliance dashboard for a tenant."""
    return await _compliance_engine.get_compliance_dashboard(tenant_id)


@router.get("/compliance/{tenant_id}/violations")
async def get_compliance_violations(
    tenant_id: str,
    resolved: bool = Query(default=False),
) -> Dict[str, Any]:
    """Get compliance violations for a tenant."""
    violations = await _compliance_engine.get_violations(tenant_id, resolved)
    return {"violations": violations, "total": len(violations)}


@router.post("/compliance/violations/{violation_id}/resolve")
async def resolve_compliance_violation(
    violation_id: str,
    notes: str = Query(default="Remediation completed"),
) -> Dict[str, Any]:
    """Resolve a compliance violation."""
    violation = await _compliance_engine.resolve_violation(violation_id, notes)
    if not violation:
        raise HTTPException(http_status.HTTP_404_NOT_FOUND, detail="Violation not found")
    return {"status": "resolved", "violation": violation.to_dict()}


@router.post("/compliance/privacy/requests")
async def submit_privacy_request(req: SubmitPrivacyRequestSchema) -> Dict[str, Any]:
    """Submit a GDPR/CCPA privacy request."""
    try:
        req_type = PrivacyRequestType(req.request_type)
    except ValueError:
        raise HTTPException(http_status.HTTP_400_BAD_REQUEST, detail=f"Invalid request type: {req.request_type}")
    request = await _compliance_engine.privacy.submit_privacy_request(
        req.tenant_id, req.subject_id, req.subject_email, req_type, req.data_categories
    )
    return {"status": "submitted", "request": request.to_dict()}


@router.get("/compliance/{tenant_id}/privacy/requests")
async def list_privacy_requests(
    tenant_id: str,
    request_status: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """List privacy requests for a tenant."""
    requests = await _compliance_engine.privacy.list_requests(tenant_id, request_status)
    return {"requests": requests, "total": len(requests)}


@router.post("/compliance/audit/events")
async def log_audit_event(req: LogAuditEventRequest) -> Dict[str, Any]:
    """Log an audit trail event."""
    event = AuditEvent(
        tenant_id=req.tenant_id,
        actor_id=req.actor_id,
        action=req.action,
        resource_type=req.resource_type,
        resource_id=req.resource_id,
        outcome=req.outcome,
        ip_address=req.ip_address,
        metadata=req.metadata,
    )
    event = await _compliance_engine.audit_trail.log_event(event)
    return {"status": "logged", "event_id": event.event_id, "hash": event.hash}


@router.get("/compliance/{tenant_id}/audit/verify")
async def verify_audit_integrity(tenant_id: str) -> Dict[str, Any]:
    """Verify the integrity of the audit trail."""
    return await _compliance_engine.audit_trail.verify_integrity(tenant_id)


@router.get("/compliance/{tenant_id}/audit/search")
async def search_audit_events(
    tenant_id: str,
    actor_id: Optional[str] = Query(default=None),
    action: Optional[str] = Query(default=None),
    resource_type: Optional[str] = Query(default=None),
    limit: int = Query(default=50),
) -> Dict[str, Any]:
    """Search audit trail events."""
    events = await _compliance_engine.audit_trail.search_events(
        tenant_id, actor_id, action, resource_type, limit=limit
    )
    return {"events": events, "total": len(events)}


# ===========================================================================
# API MONETIZATION ENDPOINTS
# ===========================================================================

@router.post("/monetization/tenants/onboard")
async def onboard_monetization_tenant(req: OnboardTenantRequest) -> Dict[str, Any]:
    """Onboard a new tenant to the monetization system."""
    result = await _monetization_engine.onboard_tenant(
        req.tenant_id, req.tier_name, req.initial_credits_usd
    )
    return {"status": "onboarded", **result}


@router.post("/monetization/usage/record")
async def record_api_usage(req: RecordUsageRequest) -> Dict[str, Any]:
    """Record API usage for billing."""
    try:
        metric = UsageMetric(req.metric)
    except ValueError:
        raise HTTPException(http_status.HTTP_400_BAD_REQUEST, detail=f"Invalid metric: {req.metric}")
    result = await _monetization_engine.record_api_usage(
        req.tenant_id, metric, req.quantity, endpoint=req.endpoint
    )
    return result


@router.get("/monetization/tenants/{tenant_id}/billing")
async def get_tenant_billing_summary(tenant_id: str) -> Dict[str, Any]:
    """Get billing summary for a tenant."""
    return await _monetization_engine.get_tenant_billing_summary(tenant_id)


@router.post("/monetization/tenants/{tenant_id}/invoice")
async def generate_tenant_invoice(
    tenant_id: str,
    year: int = Query(default=2026),
    month: int = Query(default=1),
) -> Dict[str, Any]:
    """Generate a monthly invoice for a tenant."""
    invoice = await _monetization_engine.generate_monthly_invoice(tenant_id, year, month)
    return {"status": "generated", "invoice": invoice.to_dict()}


@router.post("/monetization/tenants/{tenant_id}/upgrade")
async def upgrade_tenant_tier(
    tenant_id: str,
    new_tier: str = Query(...),
) -> Dict[str, Any]:
    """Upgrade a tenant's subscription tier."""
    try:
        result = await _monetization_engine.upgrade_tier(tenant_id, new_tier)
        return {"status": "upgraded", **result}
    except ValueError as e:
        raise HTTPException(http_status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/monetization/tenants/{tenant_id}/credits")
async def add_tenant_credits(
    tenant_id: str,
    amount: float = Query(..., description="Credits in USD"),
    reason: str = Query(default=""),
) -> Dict[str, Any]:
    """Add billing credits to a tenant account."""
    total = await _monetization_engine.add_credits(tenant_id, amount, reason)
    return {"status": "added", "total_credits_usd": total}


@router.get("/monetization/pricing-tiers")
async def list_pricing_tiers() -> Dict[str, Any]:
    """List all available pricing tiers."""
    tiers = _monetization_engine.list_pricing_tiers()
    return {"tiers": tiers, "total": len(tiers)}


@router.get("/monetization/revenue/metrics")
async def get_revenue_metrics() -> Dict[str, Any]:
    """Get high-level revenue metrics (MRR, ARR, churn)."""
    return _monetization_engine.get_revenue_metrics()


@router.post("/monetization/api-keys")
async def create_api_key(req: CreateAPIKeyRequest) -> Dict[str, Any]:
    """Create a new API key."""
    api_key, raw_key = await _monetization_engine.api_keys.create_key(
        tenant_id=req.tenant_id,
        user_id=req.user_id,
        name=req.name,
        scopes=req.scopes,
        rate_limit_per_minute=req.rate_limit_per_minute,
        expires_in_days=req.expires_in_days,
    )
    return {
        "status": "created",
        "key": api_key.to_dict(),
        "raw_key": raw_key,  # Only shown once
        "warning": "Store this key securely. It will not be shown again.",
    }


@router.get("/monetization/api-keys")
async def list_api_keys(tenant_id: str = Query(...)) -> Dict[str, Any]:
    """List API keys for a tenant."""
    keys = await _monetization_engine.api_keys.list_keys(tenant_id)
    return {"keys": keys, "total": len(keys)}


@router.delete("/monetization/api-keys/{key_id}")
async def revoke_api_key(key_id: str) -> Dict[str, Any]:
    """Revoke an API key."""
    success = await _monetization_engine.api_keys.revoke_key(key_id)
    if not success:
        raise HTTPException(http_status.HTTP_404_NOT_FOUND, detail="API key not found")
    return {"status": "revoked", "key_id": key_id}
