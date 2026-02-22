"""
Phase 6 Intelligence API Routes — Cognitive AI Engine, Multi-Agent Coordination,
Real-Time Collaboration, Intelligent Data Mesh, AI Security Operations,
Enterprise Workflow Automation, and Performance Profiling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from infrastructure.cognitive_engine import (
    CognitiveEngine,
    ReasoningStrategy,
)
from infrastructure.multi_agent import (
    AgentRole,
    ConsensusAlgorithm,
    DelegationStrategy,
    MultiAgentCoordinator,
)
from infrastructure.collaboration import (
    CollaborationEngine,
    ConflictStrategy,
    DocumentType,
    OperationType,
)
from infrastructure.data_mesh import (
    AccessPolicy,
    DataAssetType,
    DataDomain,
    DataMeshEngine,
)
from infrastructure.ai_security import (
    EventCategory,
    SecurityOperationsCenter,
    ThreatIndicatorType,
    ThreatSeverity,
)
from infrastructure.workflow_automation import (
    ExecutionStatus,
    TriggerType,
    WorkflowAutomationEngine,
    WorkflowStatus,
)
from infrastructure.profiling import PerformanceProfiler

router = APIRouter(prefix="/api/v3/phase6", tags=["Phase 6 Intelligence"])

# ─────────────────────── Singleton Instances ────────────────────────────────
_cognitive_engine = CognitiveEngine()
_multi_agent_coordinator = MultiAgentCoordinator()
_collaboration_engine = CollaborationEngine()
_data_mesh = DataMeshEngine()
_soc = SecurityOperationsCenter()
_workflow_engine = WorkflowAutomationEngine()
_profiler = PerformanceProfiler()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — COGNITIVE AI ENGINE
# ══════════════════════════════════════════════════════════════════════════════


class CognitiveTaskRequest(BaseModel):
    task: str
    strategy: str = Field(default="plan_and_solve")
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    max_iterations: int = Field(default=3, ge=1, le=5)


class DecomposeRequest(BaseModel):
    goal: str
    strategy: str = Field(default="plan_and_solve")
    max_depth: int = Field(default=4, ge=1, le=6)
    context: Optional[Dict[str, Any]] = None


class ReasonRequest(BaseModel):
    task: str
    context: Optional[Dict[str, Any]] = None
    max_steps: int = Field(default=8, ge=1, le=15)


class ValidateOutputRequest(BaseModel):
    output: str
    task: str
    context: Optional[Dict[str, Any]] = None


@router.post("/cognitive/process")
async def cognitive_process_task(req: CognitiveTaskRequest) -> Dict[str, Any]:
    """Process a task through the full cognitive pipeline: plan → reason → evaluate → reflect."""
    try:
        strategy = ReasoningStrategy(req.strategy)
    except ValueError:
        raise HTTPException(400, f"Invalid strategy: {req.strategy}")
    result = await _cognitive_engine.process_task(
        task=req.task,
        strategy=strategy,
        session_id=req.session_id,
        context=req.context,
        max_iterations=req.max_iterations,
    )
    return result


@router.post("/cognitive/decompose")
async def cognitive_decompose(req: DecomposeRequest) -> Dict[str, Any]:
    """Decompose a high-level goal into a structured execution plan."""
    try:
        strategy = ReasoningStrategy(req.strategy)
    except ValueError:
        raise HTTPException(400, f"Invalid strategy: {req.strategy}")
    plan = _cognitive_engine.decomposer.decompose(
        goal=req.goal,
        strategy=strategy,
        max_depth=req.max_depth,
        context=req.context,
    )
    return plan.to_dict()


@router.post("/cognitive/reason/chain-of-thought")
async def cognitive_chain_of_thought(req: ReasonRequest) -> Dict[str, Any]:
    """Apply chain-of-thought reasoning to produce a structured reasoning trace."""
    trace = _cognitive_engine.cot_reasoner.reason(
        task=req.task,
        context=req.context,
        max_steps=req.max_steps,
    )
    return trace.to_dict()


@router.post("/cognitive/reason/tree-of-thought")
async def cognitive_tree_of_thought(req: ReasonRequest) -> Dict[str, Any]:
    """Apply tree-of-thought reasoning with beam search over multiple branches."""
    result = _cognitive_engine.tot_reasoner.reason(
        task=req.task,
        context=req.context,
    )
    return result


@router.post("/cognitive/validate-output")
async def cognitive_validate_output(req: ValidateOutputRequest) -> Dict[str, Any]:
    """Validate an AI-generated output for hallucinations and quality."""
    return _cognitive_engine.hallucination_reducer.validate_output(
        output=req.output,
        context=req.context or {},
        task=req.task,
    )


@router.post("/cognitive/sessions")
async def cognitive_create_session() -> Dict[str, Any]:
    """Create a new cognitive processing session."""
    sid = _cognitive_engine.create_session()
    return {"session_id": sid, "status": "created"}


@router.get("/cognitive/sessions/{session_id}")
async def cognitive_get_session(session_id: str) -> Dict[str, Any]:
    """Get details of a cognitive session."""
    session = _cognitive_engine.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return {"session_id": session_id, **session}


@router.get("/cognitive/sessions")
async def cognitive_list_sessions() -> Dict[str, Any]:
    """List all active cognitive sessions."""
    sessions = _cognitive_engine.list_sessions()
    return {"sessions": sessions, "count": len(sessions)}


@router.get("/cognitive/stats")
async def cognitive_stats() -> Dict[str, Any]:
    """Get cognitive engine statistics."""
    return _cognitive_engine.get_statistics()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MULTI-AGENT COORDINATION
# ══════════════════════════════════════════════════════════════════════════════


class RegisterAgentRequest(BaseModel):
    name: str
    role: str
    capabilities: List[Dict[str, Any]] = Field(default_factory=list)
    max_load: float = Field(default=1.0, ge=0.1, le=10.0)
    metadata: Optional[Dict[str, Any]] = None


class DelegateTaskRequest(BaseModel):
    description: str
    required_capabilities: List[str] = Field(default_factory=list)
    payload: Dict[str, Any] = Field(default_factory=dict)
    delegated_by: str = Field(default="system")
    strategy: str = Field(default="capability_match")
    priority: int = Field(default=5, ge=1, le=10)


class ConsensusRequest(BaseModel):
    proposal: Dict[str, Any]
    proposer_id: str
    algorithm: str = Field(default="majority_vote")
    auto_vote: bool = True


class CompleteTaskRequest(BaseModel):
    task_id: str
    result: Any = None
    success: bool = True


@router.post("/agents/register")
async def register_agent(req: RegisterAgentRequest) -> Dict[str, Any]:
    """Register a new agent with the multi-agent coordinator."""
    try:
        role = AgentRole(req.role)
    except ValueError:
        raise HTTPException(400, f"Invalid role: {req.role}")
    agent_id = _multi_agent_coordinator.register_agent(
        name=req.name,
        role=role,
        capabilities=req.capabilities,
        max_load=req.max_load,
        metadata=req.metadata,
    )
    return {"agent_id": agent_id, "name": req.name, "role": req.role}


@router.delete("/agents/{agent_id}")
async def deregister_agent(agent_id: str) -> Dict[str, Any]:
    """Deregister an agent from the coordinator."""
    success = _multi_agent_coordinator.deregister_agent(agent_id)
    return {"success": success, "agent_id": agent_id}


@router.get("/agents")
async def list_agents() -> Dict[str, Any]:
    """List all registered agents with their status and capabilities."""
    agents = _multi_agent_coordinator.list_agents()
    return {"agents": agents, "count": len(agents)}


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str) -> Dict[str, Any]:
    """Get details of a specific agent."""
    agent = _multi_agent_coordinator.get_agent(agent_id)
    if agent is None:
        raise HTTPException(404, "Agent not found")
    return agent


@router.post("/agents/tasks/delegate")
async def delegate_task(req: DelegateTaskRequest) -> Dict[str, Any]:
    """Delegate a task to the most suitable available agent."""
    try:
        strategy = DelegationStrategy(req.strategy)
    except ValueError:
        raise HTTPException(400, f"Invalid delegation strategy: {req.strategy}")
    result = await _multi_agent_coordinator.delegate_task(
        description=req.description,
        required_capabilities=req.required_capabilities,
        payload=req.payload,
        delegated_by=req.delegated_by,
        strategy=strategy,
        priority=req.priority,
    )
    return result


@router.post("/agents/tasks/complete")
async def complete_task(req: CompleteTaskRequest) -> Dict[str, Any]:
    """Mark a delegated task as completed."""
    success = _multi_agent_coordinator.delegator.complete_task(
        task_id=req.task_id,
        result=req.result,
        success=req.success,
    )
    return {"success": success, "task_id": req.task_id}


@router.get("/agents/tasks/{task_id}")
async def get_task(task_id: str) -> Dict[str, Any]:
    """Get details of a delegated task."""
    task = _multi_agent_coordinator.delegator.get_task(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    return task.to_dict()


@router.post("/agents/consensus")
async def run_consensus(req: ConsensusRequest) -> Dict[str, Any]:
    """Run a consensus round among registered agents."""
    try:
        algorithm = ConsensusAlgorithm(req.algorithm)
    except ValueError:
        raise HTTPException(400, f"Invalid consensus algorithm: {req.algorithm}")
    result = await _multi_agent_coordinator.run_consensus(
        proposal=req.proposal,
        proposer_id=req.proposer_id,
        algorithm=algorithm,
        auto_vote=req.auto_vote,
    )
    return result


@router.get("/agents/system/health")
async def agent_system_health() -> Dict[str, Any]:
    """Get multi-agent system health metrics."""
    return _multi_agent_coordinator.get_system_health()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — REAL-TIME COLLABORATION
# ══════════════════════════════════════════════════════════════════════════════


class CreateDocumentRequest(BaseModel):
    title: str
    doc_type: str = Field(default="text")
    owner_id: str
    initial_content: Any = ""
    conflict_strategy: str = Field(default="operational_transform")


class CreateSessionRequest(BaseModel):
    document_id: str
    creator_id: str


class JoinSessionRequest(BaseModel):
    session_id: str
    user_id: str
    display_name: str


class ApplyOperationRequest(BaseModel):
    doc_id: str
    user_id: str
    op_type: str = Field(default="insert")
    position: int = Field(default=0, ge=0)
    content: Any = ""
    length: Optional[int] = None
    client_revision: Optional[int] = None


class UpdateCursorRequest(BaseModel):
    user_id: str
    doc_id: str
    position: int = Field(default=0, ge=0)
    selection_start: Optional[int] = None
    selection_end: Optional[int] = None


@router.post("/collab/documents")
async def create_document(req: CreateDocumentRequest) -> Dict[str, Any]:
    """Create a new collaborative document."""
    try:
        doc_type = DocumentType(req.doc_type)
        strategy = ConflictStrategy(req.conflict_strategy)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    doc = _collaboration_engine.create_document(
        title=req.title,
        doc_type=doc_type,
        owner_id=req.owner_id,
        initial_content=req.initial_content,
        conflict_strategy=strategy,
    )
    return doc.to_dict()


@router.get("/collab/documents")
async def list_documents(
    owner_id: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """List collaborative documents."""
    docs = _collaboration_engine.list_documents(owner_id)
    return {"documents": docs, "count": len(docs)}


@router.get("/collab/documents/{doc_id}")
async def get_document(doc_id: str) -> Dict[str, Any]:
    """Get a collaborative document."""
    doc = _collaboration_engine.get_document(doc_id)
    if doc is None:
        raise HTTPException(404, "Document not found")
    return doc.to_dict()


@router.post("/collab/sessions")
async def create_session(req: CreateSessionRequest) -> Dict[str, Any]:
    """Create a new collaboration session for a document."""
    session = _collaboration_engine.create_session(req.document_id, req.creator_id)
    if session is None:
        raise HTTPException(404, "Document not found")
    return session.to_dict()


@router.post("/collab/sessions/join")
async def join_session(req: JoinSessionRequest) -> Dict[str, Any]:
    """Join an existing collaboration session."""
    result = _collaboration_engine.join_session(
        session_id=req.session_id,
        user_id=req.user_id,
        display_name=req.display_name,
    )
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Failed to join session"))
    return result


@router.post("/collab/operations")
async def apply_operation(req: ApplyOperationRequest) -> Dict[str, Any]:
    """Apply an editing operation to a collaborative document."""
    try:
        op_type = OperationType(req.op_type)
    except ValueError:
        raise HTTPException(400, f"Invalid operation type: {req.op_type}")
    result = _collaboration_engine.apply_operation(
        doc_id=req.doc_id,
        user_id=req.user_id,
        op_type=op_type,
        position=req.position,
        content=req.content,
        length=req.length,
        client_revision=req.client_revision,
    )
    if not result.get("success"):
        raise HTTPException(400, result.get("error", "Operation failed"))
    return result


@router.post("/collab/cursor")
async def update_cursor(req: UpdateCursorRequest) -> Dict[str, Any]:
    """Update a user's cursor position in a document."""
    success = _collaboration_engine.update_cursor(
        user_id=req.user_id,
        doc_id=req.doc_id,
        position=req.position,
        selection_start=req.selection_start,
        selection_end=req.selection_end,
    )
    return {"success": success}


@router.get("/collab/documents/{doc_id}/presence")
async def get_document_presence(doc_id: str) -> Dict[str, Any]:
    """Get all users currently viewing/editing a document."""
    presences = _collaboration_engine.get_document_presence(doc_id)
    return {"presences": presences, "count": len(presences)}


@router.post("/collab/documents/{doc_id}/snapshots")
async def create_snapshot(doc_id: str, label: str = "") -> Dict[str, Any]:
    """Create a version snapshot of a document."""
    snapshot = _collaboration_engine.create_snapshot(doc_id, label)
    if snapshot is None:
        raise HTTPException(404, "Document not found")
    return snapshot


@router.get("/collab/documents/{doc_id}/history")
async def get_document_history(
    doc_id: str, limit: int = Query(default=20, ge=1, le=100)
) -> Dict[str, Any]:
    """Get revision history of a document."""
    history = _collaboration_engine.get_history(doc_id, limit)
    return {"history": history, "count": len(history)}


@router.get("/collab/stats")
async def collaboration_stats() -> Dict[str, Any]:
    """Get collaboration engine statistics."""
    return _collaboration_engine.get_engine_stats()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — INTELLIGENT DATA MESH
# ══════════════════════════════════════════════════════════════════════════════


class RegisterAssetRequest(BaseModel):
    name: str
    description: str
    domain: str
    asset_type: str
    owner_team: str
    schema_fields: List[Dict[str, Any]] = Field(default_factory=list)
    access_policy: str = Field(default="internal")
    tags: List[str] = Field(default_factory=list)
    data_source: str = Field(default="unknown")
    update_frequency: str = Field(default="daily")
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None


class EvaluateQualityRequest(BaseModel):
    asset_id: str
    sample_data: Optional[List[Dict[str, Any]]] = None


class SearchAssetsRequest(BaseModel):
    query: str
    domain: Optional[str] = None
    asset_type: Optional[str] = None
    min_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    limit: int = Field(default=20, ge=1, le=100)


class CreateContractRequest(BaseModel):
    producer_asset_id: str
    consumer_team: str
    sla_freshness_minutes: int = Field(default=60, ge=1)
    sla_availability_pct: float = Field(default=99.0, ge=0.0, le=100.0)
    schema_version: str = Field(default="1.0.0")
    agreed_fields: List[str] = Field(default_factory=list)


@router.post("/data-mesh/assets")
async def register_data_asset(req: RegisterAssetRequest) -> Dict[str, Any]:
    """Register a new data asset in the data mesh catalog."""
    try:
        domain = DataDomain(req.domain)
        asset_type = DataAssetType(req.asset_type)
        access_policy = AccessPolicy(req.access_policy)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    asset = _data_mesh.register_data_asset(
        name=req.name,
        description=req.description,
        domain=domain,
        asset_type=asset_type,
        owner_team=req.owner_team,
        schema_fields=req.schema_fields,
        access_policy=access_policy,
        tags=req.tags,
        data_source=req.data_source,
        update_frequency=req.update_frequency,
        row_count=req.row_count,
        size_bytes=req.size_bytes,
    )
    return asset.to_dict()


@router.post("/data-mesh/assets/search")
async def search_data_assets(req: SearchAssetsRequest) -> Dict[str, Any]:
    """Search data assets by query, domain, and type."""
    assets = _data_mesh.search_assets(
        query=req.query,
        domain=req.domain,
        asset_type=req.asset_type,
        min_quality=req.min_quality,
        limit=req.limit,
    )
    return {"assets": assets, "count": len(assets)}


@router.get("/data-mesh/assets/{asset_id}")
async def get_data_asset(asset_id: str) -> Dict[str, Any]:
    """Get a specific data asset."""
    asset = _data_mesh.catalog.get_asset(asset_id)
    if asset is None:
        raise HTTPException(404, "Asset not found")
    return asset.to_dict()


@router.post("/data-mesh/quality/evaluate")
async def evaluate_data_quality(req: EvaluateQualityRequest) -> Dict[str, Any]:
    """Evaluate the data quality of an asset."""
    report = _data_mesh.evaluate_and_update_quality(
        asset_id=req.asset_id,
        sample_data=req.sample_data,
    )
    if report is None:
        raise HTTPException(404, "Asset not found")
    return report.to_dict()


@router.get("/data-mesh/assets/{asset_id}/quality-history")
async def get_quality_history(
    asset_id: str, limit: int = Query(default=10, ge=1, le=50)
) -> Dict[str, Any]:
    """Get quality evaluation history for an asset."""
    history = _data_mesh.quality.get_quality_history(asset_id, limit)
    return {"history": history, "count": len(history)}


@router.get("/data-mesh/assets/{asset_id}/lineage")
async def get_asset_lineage(
    asset_id: str,
    direction: str = Query(default="both"),
) -> Dict[str, Any]:
    """Get data lineage for an asset."""
    return _data_mesh.get_asset_lineage(asset_id, direction)


@router.get("/data-mesh/assets/{asset_id}/related")
async def get_related_assets(
    asset_id: str, top_k: int = Query(default=5, ge=1, le=20)
) -> Dict[str, Any]:
    """Get related assets based on tags and domain."""
    assets = _data_mesh.catalog.recommend_related(asset_id, top_k)
    return {"related": [a.to_dict() for a in assets], "count": len(assets)}


@router.post("/data-mesh/contracts")
async def create_data_contract(req: CreateContractRequest) -> Dict[str, Any]:
    """Create a data contract between a producer and consumer."""
    contract = _data_mesh.contracts.create_contract(
        producer_asset_id=req.producer_asset_id,
        consumer_team=req.consumer_team,
        sla_freshness_minutes=req.sla_freshness_minutes,
        sla_availability_pct=req.sla_availability_pct,
        schema_version=req.schema_version,
        agreed_fields=req.agreed_fields,
    )
    return contract.to_dict()


@router.get("/data-mesh/overview")
async def data_mesh_overview() -> Dict[str, Any]:
    """Get an overview of the data mesh."""
    return _data_mesh.get_mesh_overview()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — AI SECURITY OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════


class IngestEventRequest(BaseModel):
    event_category: str
    action: str
    outcome: str
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    resource: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None
    content: Optional[str] = None


class AddThreatIndicatorRequest(BaseModel):
    indicator_type: str
    value: str
    threat_name: str
    severity: str = Field(default="high")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    source: str = Field(default="manual")


class UpdateIncidentRequest(BaseModel):
    incident_id: str
    new_status: str
    analyst_notes: str = ""


@router.post("/security/events")
async def ingest_security_event(req: IngestEventRequest) -> Dict[str, Any]:
    """Ingest a security event for analysis, threat detection, and auto-response."""
    try:
        category = EventCategory(req.event_category)
    except ValueError:
        raise HTTPException(400, f"Invalid event category: {req.event_category}")
    return _soc.ingest_event(
        event_category=category,
        action=req.action,
        outcome=req.outcome,
        source_ip=req.source_ip,
        user_id=req.user_id,
        resource=req.resource,
        raw_data=req.raw_data,
        content=req.content,
    )


@router.post("/security/threat-indicators")
async def add_threat_indicator(req: AddThreatIndicatorRequest) -> Dict[str, Any]:
    """Add a threat indicator to the intelligence database."""
    try:
        ind_type = ThreatIndicatorType(req.indicator_type)
        severity = ThreatSeverity(req.severity)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    indicator = _soc.add_threat_indicator(
        indicator_type=ind_type,
        value=req.value,
        threat_name=req.threat_name,
        severity=severity,
        confidence=req.confidence,
        source=req.source,
    )
    return indicator.to_dict()


@router.get("/security/incidents")
async def list_incidents(
    status_filter: Optional[str] = Query(None),
    severity_filter: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """List security incidents with optional filters."""
    from infrastructure.ai_security.security_operations import IncidentStatus
    status_enum = None
    if status_filter:
        try:
            status_enum = IncidentStatus(status_filter)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status_filter}")
    severity_enum = None
    if severity_filter:
        try:
            severity_enum = ThreatSeverity(severity_filter)
        except ValueError:
            raise HTTPException(400, f"Invalid severity: {severity_filter}")
    incidents = _soc.incident_mgr.list_incidents(status_enum, severity_enum)
    return {"incidents": [i.to_dict() for i in incidents], "count": len(incidents)}


@router.get("/security/incidents/{incident_id}")
async def get_incident(incident_id: str) -> Dict[str, Any]:
    """Get details of a security incident."""
    incident = _soc.incident_mgr._incidents.get(incident_id)
    if incident is None:
        raise HTTPException(404, "Incident not found")
    return incident.to_dict()


@router.put("/security/incidents/{incident_id}/status")
async def update_incident_status(incident_id: str, req: UpdateIncidentRequest) -> Dict[str, Any]:
    """Update the status of a security incident."""
    from infrastructure.ai_security.security_operations import IncidentStatus
    try:
        new_status = IncidentStatus(req.new_status)
    except ValueError:
        raise HTTPException(400, f"Invalid status: {req.new_status}")
    success = _soc.incident_mgr.update_status(incident_id, new_status, req.analyst_notes)
    return {"success": success, "incident_id": incident_id, "new_status": req.new_status}


@router.get("/security/incidents/{incident_id}/sla")
async def get_incident_sla(incident_id: str) -> Dict[str, Any]:
    """Get SLA status for a security incident."""
    return _soc.incident_mgr.get_sla_status(incident_id)


@router.get("/security/blocked-ips")
async def get_blocked_ips() -> Dict[str, Any]:
    """Get list of currently blocked IP addresses."""
    blocked = _soc.get_blocked_ips()
    return {"blocked_ips": blocked, "count": len(blocked)}


@router.delete("/security/blocked-ips/{ip}")
async def unblock_ip(ip: str) -> Dict[str, Any]:
    """Remove an IP address from the block list."""
    _soc._blocked_ips.discard(ip)
    return {"success": True, "unblocked_ip": ip}


@router.get("/security/dashboard")
async def security_dashboard() -> Dict[str, Any]:
    """Get the Security Operations Center dashboard."""
    return _soc.get_soc_dashboard()


@router.get("/security/threat-intel/stats")
async def threat_intel_stats() -> Dict[str, Any]:
    """Get threat intelligence statistics."""
    return _soc.threat_intel.get_intel_stats()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ENTERPRISE WORKFLOW AUTOMATION
# ══════════════════════════════════════════════════════════════════════════════


class CreateWorkflowRequest(BaseModel):
    name: str
    description: str = ""
    owner_id: str
    steps: List[Dict[str, Any]]
    triggers: Optional[List[Dict[str, Any]]] = None
    variables: Optional[Dict[str, Any]] = None
    tags: List[str] = Field(default_factory=list)


class ExecuteWorkflowRequest(BaseModel):
    workflow_id: str
    input_data: Optional[Dict[str, Any]] = None
    trigger_type: str = Field(default="manual")


class UpdateWorkflowStatusRequest(BaseModel):
    status: str


@router.post("/workflows/definitions")
async def create_workflow(req: CreateWorkflowRequest) -> Dict[str, Any]:
    """Create a new workflow definition."""
    wf = _workflow_engine.create_workflow(
        name=req.name,
        description=req.description,
        owner_id=req.owner_id,
        steps=req.steps,
        triggers=req.triggers,
        variables=req.variables,
        tags=req.tags,
    )
    return wf.to_dict()


@router.get("/workflows/definitions")
async def list_workflows(
    owner_id: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """List workflow definitions."""
    status_enum = None
    if status_filter:
        try:
            status_enum = WorkflowStatus(status_filter)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status_filter}")
    workflows = _workflow_engine.list_workflows(owner_id, status_enum, tag)
    return {"workflows": workflows, "count": len(workflows)}


@router.get("/workflows/definitions/{workflow_id}")
async def get_workflow(workflow_id: str) -> Dict[str, Any]:
    """Get a workflow definition."""
    wf = _workflow_engine.get_workflow(workflow_id)
    if wf is None:
        raise HTTPException(404, "Workflow not found")
    return wf.to_dict()


@router.put("/workflows/definitions/{workflow_id}/activate")
async def activate_workflow(workflow_id: str) -> Dict[str, Any]:
    """Activate a workflow to allow executions."""
    success = _workflow_engine.activate_workflow(workflow_id)
    if not success:
        raise HTTPException(404, "Workflow not found")
    return {"success": True, "workflow_id": workflow_id, "status": "active"}


@router.post("/workflows/execute")
async def execute_workflow(req: ExecuteWorkflowRequest) -> Dict[str, Any]:
    """Execute a workflow with optional input data."""
    try:
        trigger = TriggerType(req.trigger_type)
    except ValueError:
        raise HTTPException(400, f"Invalid trigger type: {req.trigger_type}")
    try:
        execution = await _workflow_engine.execute_workflow(
            workflow_id=req.workflow_id,
            input_data=req.input_data,
            trigger_type=trigger,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return execution.to_dict()


@router.get("/workflows/executions")
async def list_executions(
    workflow_id: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None),
    limit: int = Query(default=20, ge=1, le=100),
) -> Dict[str, Any]:
    """List workflow executions."""
    status_enum = None
    if status_filter:
        try:
            status_enum = ExecutionStatus(status_filter)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status_filter}")
    executions = _workflow_engine.list_executions(workflow_id, status_enum, limit)
    return {"executions": executions, "count": len(executions)}


@router.get("/workflows/executions/{execution_id}")
async def get_execution(execution_id: str) -> Dict[str, Any]:
    """Get details of a workflow execution."""
    execution = _workflow_engine.get_execution(execution_id)
    if execution is None:
        raise HTTPException(404, "Execution not found")
    return execution.to_dict()


@router.get("/workflows/definitions/{workflow_id}/analytics")
async def get_workflow_analytics(workflow_id: str) -> Dict[str, Any]:
    """Get analytics for a workflow."""
    return _workflow_engine.get_workflow_analytics(workflow_id)


@router.get("/workflows/definitions/{workflow_id}/versions")
async def list_workflow_versions(workflow_id: str) -> Dict[str, Any]:
    """List versions of a workflow."""
    versions = _workflow_engine.version_mgr.list_versions(workflow_id)
    return {"versions": versions, "count": len(versions)}


@router.get("/workflows/audit-log")
async def get_audit_log(
    limit: int = Query(default=50, ge=1, le=200)
) -> Dict[str, Any]:
    """Get the workflow engine audit log."""
    log = _workflow_engine.get_audit_log(limit)
    return {"audit_log": log, "count": len(log)}


@router.get("/workflows/stats")
async def workflow_engine_stats() -> Dict[str, Any]:
    """Get workflow engine statistics."""
    return _workflow_engine.get_engine_stats()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PERFORMANCE PROFILING
# ══════════════════════════════════════════════════════════════════════════════


class RecordRequestRequest(BaseModel):
    endpoint: str
    method: str = Field(default="GET")
    latency_ms: float
    status_code: int = Field(default=200)
    user_id: Optional[str] = None


class RecordDbQueryRequest(BaseModel):
    query_type: str = Field(default="select")
    table: str
    duration_ms: float
    rows_affected: int = Field(default=0)


class UpdateSystemMetricsRequest(BaseModel):
    memory_mb: float = Field(default=0.0)
    cpu_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    active_connections: int = Field(default=0, ge=0)
    cache_hit_rate: float = Field(default=0.0, ge=0.0, le=100.0)


class DefineCustomMetricRequest(BaseModel):
    name: str
    metric_type: str = Field(default="gauge")
    unit: str = ""
    value: Optional[float] = None


class DefineSLORequest(BaseModel):
    name: str
    description: str = ""
    metric_name: str
    threshold: float
    comparison: str = Field(default="lt")
    window_seconds: int = Field(default=3600, ge=60)
    error_budget_pct: float = Field(default=0.1, ge=0.0, le=1.0)
    owner: str = Field(default="platform")


@router.post("/profiling/requests")
async def record_request(req: RecordRequestRequest) -> Dict[str, Any]:
    """Record an API request for performance profiling."""
    _profiler.record_request(
        endpoint=req.endpoint,
        method=req.method,
        latency_ms=req.latency_ms,
        status_code=req.status_code,
        user_id=req.user_id,
    )
    return {"recorded": True, "endpoint": req.endpoint, "latency_ms": req.latency_ms}


@router.post("/profiling/db-queries")
async def record_db_query(req: RecordDbQueryRequest) -> Dict[str, Any]:
    """Record a database query for performance profiling."""
    _profiler.record_db_query(
        query_type=req.query_type,
        table=req.table,
        duration_ms=req.duration_ms,
        rows_affected=req.rows_affected,
    )
    return {"recorded": True}


@router.post("/profiling/system-metrics")
async def update_system_metrics(req: UpdateSystemMetricsRequest) -> Dict[str, Any]:
    """Update system-level performance metrics."""
    _profiler.update_system_metrics(
        memory_mb=req.memory_mb,
        cpu_pct=req.cpu_pct,
        active_connections=req.active_connections,
        cache_hit_rate=req.cache_hit_rate,
    )
    return {"updated": True}


@router.post("/profiling/metrics/custom")
async def define_custom_metric(req: DefineCustomMetricRequest) -> Dict[str, Any]:
    """Define and optionally record a custom metric."""
    from infrastructure.profiling.performance_profiler import MetricType as MT
    try:
        mt = MT(req.metric_type)
    except ValueError:
        raise HTTPException(400, f"Invalid metric type: {req.metric_type}")
    series = _profiler.metrics.register(req.name, mt, req.unit)
    if req.value is not None:
        series.record(req.value)
    return {"metric_name": req.name, "type": req.metric_type, "registered": True}


@router.post("/profiling/slos")
async def define_slo(req: DefineSLORequest) -> Dict[str, Any]:
    """Define a new Service Level Objective."""
    slo = _profiler.slo_monitor.define_slo(
        name=req.name,
        description=req.description,
        metric_name=req.metric_name,
        threshold=req.threshold,
        comparison=req.comparison,
        window_seconds=req.window_seconds,
        error_budget_pct=req.error_budget_pct,
        owner=req.owner,
    )
    return slo.to_dict()


@router.get("/profiling/report")
async def get_performance_report() -> Dict[str, Any]:
    """Get a comprehensive performance report with hotspots, metrics, and recommendations."""
    return _profiler.get_performance_report()


@router.get("/profiling/dashboard")
async def get_profiling_dashboard() -> Dict[str, Any]:
    """Get a summarized performance dashboard."""
    return _profiler.get_dashboard_summary()


@router.get("/profiling/hotspots")
async def get_hotspots(
    top_k: int = Query(default=10, ge=1, le=50)
) -> Dict[str, Any]:
    """Get the top performance hotspots."""
    hotspots = _profiler.call_graph.get_hotspots(top_k)
    return {"hotspots": hotspots, "count": len(hotspots)}


@router.get("/profiling/metrics")
async def get_all_metrics() -> Dict[str, Any]:
    """Get all tracked performance metrics."""
    return {"metrics": _profiler.metrics.get_all_metrics()}


@router.get("/profiling/metrics/{name}")
async def get_metric(name: str) -> Dict[str, Any]:
    """Get a specific metric by name."""
    metric = _profiler.metrics.get_metric(name)
    if metric is None:
        raise HTTPException(404, f"Metric '{name}' not found")
    return metric


@router.get("/profiling/slos/status")
async def get_slo_status() -> Dict[str, Any]:
    """Evaluate and return the status of all SLOs."""
    results = _profiler.slo_monitor.evaluate_all(_profiler.metrics)
    breached = sum(1 for r in results if r.get("status") == "breached")
    return {"slo_results": results, "total": len(results), "breached": breached}


@router.get("/profiling/traces/recent")
async def get_recent_traces(
    limit: int = Query(default=20, ge=1, le=100)
) -> Dict[str, Any]:
    """Get recent performance traces."""
    traces = _profiler.tracer.get_recent_traces(limit)
    return {"traces": traces, "count": len(traces)}


@router.get("/profiling/latency")
async def get_latency_stats() -> Dict[str, Any]:
    """Get latency percentile statistics."""
    return _profiler.tracer.get_latency_stats()


@router.put("/profiling/enable")
async def enable_profiling() -> Dict[str, Any]:
    """Enable performance profiling."""
    _profiler.enable()
    return {"profiling_enabled": True}


@router.put("/profiling/disable")
async def disable_profiling() -> Dict[str, Any]:
    """Disable performance profiling."""
    _profiler.disable()
    return {"profiling_enabled": False}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — PHASE 6 OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════


@router.get("/overview")
async def phase6_overview() -> Dict[str, Any]:
    """Get an overview of all Phase 6 systems and their current status."""
    return {
        "phase": 6,
        "systems": [
            {
                "name": "Cognitive AI Engine",
                "description": (
                    "Advanced cognitive reasoning, planning, task decomposition, "
                    "self-evaluation, and hallucination reduction"
                ),
                "endpoints_prefix": "/api/v3/phase6/cognitive",
                "capabilities": [
                    "chain_of_thought",
                    "tree_of_thought",
                    "plan_and_solve",
                    "reflexion",
                    "hallucination_reduction",
                    "context_management",
                ],
                "stats": _cognitive_engine.get_statistics(),
            },
            {
                "name": "Multi-Agent Coordination System",
                "description": (
                    "Agent registry, message bus, intelligent task delegation, "
                    "consensus mechanisms, and performance tracking"
                ),
                "endpoints_prefix": "/api/v3/phase6/agents",
                "capabilities": [
                    "agent_registry",
                    "task_delegation",
                    "consensus_voting",
                    "capability_matching",
                    "load_balancing",
                    "performance_tracking",
                ],
                "stats": _multi_agent_coordinator.get_system_health(),
            },
            {
                "name": "Real-Time Collaboration Engine",
                "description": (
                    "Operational transformation, CRDT, presence awareness, "
                    "cursor sharing, conflict resolution, and version control"
                ),
                "endpoints_prefix": "/api/v3/phase6/collab",
                "capabilities": [
                    "operational_transform",
                    "crdt",
                    "presence_awareness",
                    "cursor_sharing",
                    "document_versioning",
                    "conflict_resolution",
                ],
                "stats": _collaboration_engine.get_engine_stats(),
            },
            {
                "name": "Intelligent Data Mesh",
                "description": (
                    "Data catalog, lineage tracking, quality scoring, "
                    "data contracts, and federated access control"
                ),
                "endpoints_prefix": "/api/v3/phase6/data-mesh",
                "capabilities": [
                    "data_catalog",
                    "lineage_tracking",
                    "quality_evaluation",
                    "data_contracts",
                    "impact_analysis",
                    "asset_discovery",
                ],
                "stats": _data_mesh.get_mesh_overview(),
            },
            {
                "name": "AI Security Operations Center",
                "description": (
                    "SIEM, threat intelligence, behavioral anomaly detection, "
                    "incident management, and automated response"
                ),
                "endpoints_prefix": "/api/v3/phase6/security",
                "capabilities": [
                    "siem_correlation",
                    "threat_intelligence",
                    "behavioral_anomaly_detection",
                    "incident_management",
                    "auto_response",
                    "detection_rules",
                ],
                "stats": _soc.get_soc_dashboard(),
            },
            {
                "name": "Enterprise Workflow Automation",
                "description": (
                    "Visual workflow DSL, conditional branching, retry strategies, "
                    "versioned workflows, and full audit logging"
                ),
                "endpoints_prefix": "/api/v3/phase6/workflows",
                "capabilities": [
                    "workflow_dsl",
                    "conditional_branching",
                    "parallel_execution",
                    "retry_strategies",
                    "workflow_versioning",
                    "audit_logging",
                ],
                "stats": _workflow_engine.get_engine_stats(),
            },
            {
                "name": "Performance Profiling System",
                "description": (
                    "Call graph analysis, latency percentiles, SLO monitoring, "
                    "metrics aggregation, and optimization recommendations"
                ),
                "endpoints_prefix": "/api/v3/phase6/profiling",
                "capabilities": [
                    "call_graph_analysis",
                    "latency_percentiles",
                    "slo_monitoring",
                    "metrics_aggregation",
                    "hotspot_detection",
                    "optimization_recommendations",
                ],
                "stats": _profiler.get_dashboard_summary(),
            },
        ],
        "total_endpoints": 75,
        "version": "6.0.0",
    }
