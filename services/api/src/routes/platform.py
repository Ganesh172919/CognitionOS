"""
Platform API Routes

Exposes the new platform-level systems:
- RBAC: role management, permission checks, policy enforcement
- Config: dynamic configuration management
- Onboarding: user journey, milestones, checklists, spotlights
- Workflow DSL: compile and validate workflow definitions
- Coordination: multi-agent coordination bus management
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v3/platform", tags=["Platform"])


# ──────────────────────────────────────────────────────────────────────────────
# Lazy imports (avoid circular imports at module load time)
# ──────────────────────────────────────────────────────────────────────────────


def _rbac():
    from infrastructure.auth.rbac import get_rbac_engine, Permission, Role, Policy
    return get_rbac_engine(), Permission, Role, Policy


def _config():
    from infrastructure.config.config_manager import get_config_manager
    return get_config_manager()


def _onboarding():
    from infrastructure.onboarding.onboarding_engine import (
        get_onboarding_engine,
        OnboardingPersona,
        TriggerType,
    )
    return get_onboarding_engine(), OnboardingPersona, TriggerType


def _dsl():
    from infrastructure.workflow.dsl_compiler import get_compiler
    return get_compiler()


def _bus():
    from infrastructure.agent.coordination_bus import (
        get_coordination_bus,
        AgentStatus,
        MessageType,
        MessagePriority,
    )
    return get_coordination_bus(), AgentStatus, MessageType, MessagePriority


# ──────────────────────────────────────────────────────────────────────────────
# RBAC Schemas & Routes
# ──────────────────────────────────────────────────────────────────────────────


class PermissionCheckRequest(BaseModel):
    principal_id: str
    resource: str
    action: str
    qualifier: str = "*"
    tenant_id: str
    context: Optional[Dict[str, Any]] = None


class AssignRoleRequest(BaseModel):
    principal_id: str
    role_id: str
    tenant_id: str
    granted_by: Optional[str] = None
    expires_in_seconds: Optional[float] = None


class RevokeRoleRequest(BaseModel):
    principal_id: str
    role_id: str
    tenant_id: str


class BulkPermissionCheckRequest(BaseModel):
    principal_id: str
    tenant_id: str
    permissions: List[Dict[str, str]]  # [{resource, action, qualifier?}]


@router.post("/rbac/check")
async def check_permission(request: PermissionCheckRequest) -> Dict[str, Any]:
    """Check if a principal has a specific permission"""
    engine, Permission, Role, Policy = _rbac()
    perm = Permission(request.resource, request.action, request.qualifier)
    result = engine.check(
        request.principal_id, perm, request.tenant_id, request.context
    )
    return result.to_dict()


@router.post("/rbac/check/bulk")
async def bulk_check_permissions(request: BulkPermissionCheckRequest) -> Dict[str, Any]:
    """Check multiple permissions at once"""
    engine, Permission, Role, Policy = _rbac()
    perms = [
        Permission(p["resource"], p["action"], p.get("qualifier", "*"))
        for p in request.permissions
    ]
    results = engine.bulk_check(request.principal_id, perms, request.tenant_id)
    return {"principal_id": request.principal_id, "results": results}


@router.post("/rbac/roles/assign")
async def assign_role(request: AssignRoleRequest) -> Dict[str, Any]:
    """Assign a role to a principal"""
    engine, Permission, Role, Policy = _rbac()
    expires_at = (
        time.time() + request.expires_in_seconds
        if request.expires_in_seconds
        else None
    )
    try:
        assignment = engine.assign_role(
            request.principal_id,
            request.role_id,
            request.tenant_id,
            request.granted_by,
            expires_at,
        )
        return {
            "assigned": True,
            "assignment_id": assignment.assignment_id,
            "role_id": assignment.role_id,
            "tenant_id": assignment.tenant_id,
            "expires_at": assignment.expires_at,
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/rbac/roles/revoke")
async def revoke_role(request: RevokeRoleRequest) -> Dict[str, Any]:
    """Revoke a role from a principal"""
    engine, Permission, Role, Policy = _rbac()
    removed = engine.revoke_role(request.principal_id, request.role_id, request.tenant_id)
    return {"revoked": True, "assignments_removed": removed}


@router.get("/rbac/roles")
async def list_roles(tenant_id: Optional[str] = None) -> Dict[str, Any]:
    """List all roles visible to a tenant"""
    engine, Permission, Role, Policy = _rbac()
    roles = engine.list_roles(tenant_id=tenant_id)
    return {"roles": [r.to_dict() for r in roles], "total": len(roles)}


@router.get("/rbac/principals/{principal_id}/permissions")
async def get_effective_permissions(
    principal_id: str,
    tenant_id: str,
) -> Dict[str, Any]:
    """Get all effective permissions for a principal"""
    engine, Permission, Role, Policy = _rbac()
    perms = engine.get_effective_permissions(principal_id, tenant_id)
    return {
        "principal_id": principal_id,
        "tenant_id": tenant_id,
        "permissions": [str(p) for p in perms],
        "total": len(perms),
    }


@router.get("/rbac/principals/{principal_id}/roles")
async def get_principal_roles(principal_id: str, tenant_id: str) -> Dict[str, Any]:
    """Get all roles assigned to a principal"""
    engine, Permission, Role, Policy = _rbac()
    roles = engine.get_principal_roles(principal_id, tenant_id)
    return {"principal_id": principal_id, "roles": [r.to_dict() for r in roles]}


# ──────────────────────────────────────────────────────────────────────────────
# Config Management Routes
# ──────────────────────────────────────────────────────────────────────────────


class ConfigSetRequest(BaseModel):
    key: str
    value: Any
    secret: bool = False
    description: str = ""
    changed_by: Optional[str] = None


class TenantConfigOverrideRequest(BaseModel):
    tenant_id: str
    key: str
    value: Any


@router.get("/config")
async def get_all_config() -> Dict[str, Any]:
    """Get all non-secret config values"""
    cfg = _config()
    return cfg.snapshot()


@router.get("/config/{key}")
async def get_config_value(key: str) -> Dict[str, Any]:
    """Get a specific config value"""
    cfg = _config()
    val = cfg.get(key)
    if val is None:
        raise HTTPException(status_code=404, detail=f"Config key '{key}' not found")
    return {"key": key, "value": val}


@router.put("/config")
async def set_config_value(request: ConfigSetRequest) -> Dict[str, Any]:
    """Set a runtime config override"""
    from infrastructure.config.config_manager import ConfigImmutableError
    cfg = _config()
    try:
        cfg.set(
            request.key,
            request.value,
            secret=request.secret,
            description=request.description,
            changed_by=request.changed_by,
        )
        return {"set": True, "key": request.key}
    except ConfigImmutableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/config/tenant-override")
async def set_tenant_config(request: TenantConfigOverrideRequest) -> Dict[str, Any]:
    """Set a tenant-specific config override"""
    cfg = _config()
    cfg.set_tenant_override(request.tenant_id, request.key, request.value)
    return {"set": True, "tenant_id": request.tenant_id, "key": request.key}


@router.get("/config/history/{key}")
async def get_config_history(key: str, limit: int = 20) -> Dict[str, Any]:
    """Get change history for a config key"""
    cfg = _config()
    return {
        "key": key,
        "history": cfg.get_change_history(key=key, limit=limit),
    }


@router.get("/config/checksum")
async def get_config_checksum() -> Dict[str, Any]:
    """Get a hash of the current config state"""
    cfg = _config()
    return {"checksum": cfg.checksum()}


# ──────────────────────────────────────────────────────────────────────────────
# Onboarding Routes
# ──────────────────────────────────────────────────────────────────────────────


class StartOnboardingRequest(BaseModel):
    user_id: str
    tenant_id: str
    persona: str = "general"


class CompleteMilestoneRequest(BaseModel):
    user_id: str
    milestone_id: str
    metadata: Optional[Dict[str, Any]] = None


class TriggerEventRequest(BaseModel):
    user_id: str
    trigger_type: str
    trigger_value: Optional[str] = None


@router.post("/onboarding/start")
async def start_onboarding(request: StartOnboardingRequest) -> Dict[str, Any]:
    """Start onboarding for a new user"""
    engine, OnboardingPersona, TriggerType = _onboarding()
    try:
        persona = OnboardingPersona(request.persona)
    except ValueError:
        persona = OnboardingPersona.GENERAL
    state = engine.start_onboarding(request.user_id, request.tenant_id, persona)
    return state.to_dict()


@router.post("/onboarding/milestone/complete")
async def complete_milestone(request: CompleteMilestoneRequest) -> Dict[str, Any]:
    """Mark an onboarding milestone as completed"""
    engine, OnboardingPersona, TriggerType = _onboarding()
    return engine.complete_milestone(request.user_id, request.milestone_id, request.metadata)


@router.post("/onboarding/event")
async def trigger_onboarding_event(request: TriggerEventRequest) -> Dict[str, Any]:
    """Fire an onboarding event (auto-completes matching milestones)"""
    engine, OnboardingPersona, TriggerType = _onboarding()
    try:
        ttype = TriggerType(request.trigger_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown trigger type: {request.trigger_type}")
    triggered = engine.trigger_event(request.user_id, ttype, request.trigger_value)
    return {
        "triggered_milestones": [m.to_dict() for m in triggered],
        "count": len(triggered),
    }


@router.get("/onboarding/{user_id}/checklist")
async def get_onboarding_checklist(user_id: str) -> Dict[str, Any]:
    """Get the onboarding checklist for a user"""
    engine, OnboardingPersona, TriggerType = _onboarding()
    checklist = engine.get_checklist(user_id)
    if not checklist:
        raise HTTPException(status_code=404, detail="Onboarding not started for this user")
    return checklist.to_dict()


@router.get("/onboarding/{user_id}/spotlights")
async def get_user_spotlights(user_id: str) -> Dict[str, Any]:
    """Get feature spotlights to show to a user"""
    engine, OnboardingPersona, TriggerType = _onboarding()
    spotlights = engine.get_spotlights_for_user(user_id)
    return {
        "user_id": user_id,
        "spotlights": [s.to_dict() for s in spotlights],
    }


@router.get("/onboarding/analytics/funnel")
async def get_onboarding_funnel() -> Dict[str, Any]:
    """Get aggregate onboarding funnel analytics"""
    engine, OnboardingPersona, TriggerType = _onboarding()
    return engine.get_funnel_analytics()


# ──────────────────────────────────────────────────────────────────────────────
# Workflow DSL Compiler Routes
# ──────────────────────────────────────────────────────────────────────────────


class CompileWorkflowRequest(BaseModel):
    definition: Dict[str, Any]
    format: str = "dict"   # "dict", "json", "yaml"
    source: Optional[str] = None  # raw string if format is json or yaml


@router.post("/workflow-dsl/compile")
async def compile_workflow(request: CompileWorkflowRequest) -> Dict[str, Any]:
    """Compile a workflow DSL definition into an execution plan"""
    compiler = _dsl()
    if request.format == "json" and request.source:
        result = compiler.compile_json(request.source)
    elif request.format == "yaml" and request.source:
        result = compiler.compile_yaml(request.source)
    else:
        result = compiler.compile_dict(request.definition)
    return result.to_dict()


@router.post("/workflow-dsl/validate")
async def validate_workflow(request: CompileWorkflowRequest) -> Dict[str, Any]:
    """Validate a workflow definition (compile without executing)"""
    compiler = _dsl()
    result = compiler.compile_dict(request.definition)
    return {
        "valid": result.success,
        "errors": [{"field": e.field, "message": e.message} for e in result.errors],
        "warnings": [{"field": w.field, "message": w.message, "severity": w.severity} for w in result.warnings],
        "step_count": len(result.workflow.steps) if result.workflow else 0,
        "execution_order": result.workflow.execution_order if result.workflow else [],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Agent Coordination Bus Routes
# ──────────────────────────────────────────────────────────────────────────────


class RegisterAgentRequest(BaseModel):
    agent_id: str
    capabilities: List[str] = Field(default_factory=list)
    max_concurrent_tasks: int = 1
    metadata: Optional[Dict[str, Any]] = None


class SendMessageRequest(BaseModel):
    message_type: str
    sender_id: str
    payload: Dict[str, Any]
    recipient_id: Optional[str] = None
    priority: int = 5
    ttl_seconds: float = 30.0


class AcquireLockRequest(BaseModel):
    resource_id: str
    holder_id: str
    ttl_seconds: float = 30.0


class ProposeConsensusRequest(BaseModel):
    topic: str
    proposed_value: Any
    proposer_id: str
    required_agents: Optional[List[str]] = None
    quorum_fraction: float = 0.5
    deadline_seconds: float = 30.0


class CastVoteRequest(BaseModel):
    proposal_id: str
    agent_id: str
    vote: bool


@router.post("/coordination/agents/register")
async def register_agent(request: RegisterAgentRequest) -> Dict[str, Any]:
    """Register an agent on the coordination bus"""
    bus, AgentStatus, MessageType, MessagePriority = _bus()
    reg = bus.register_agent(
        request.agent_id,
        capabilities=set(request.capabilities),
        max_concurrent_tasks=request.max_concurrent_tasks,
        metadata=request.metadata,
    )
    return reg.to_dict()


@router.get("/coordination/agents")
async def list_agents() -> Dict[str, Any]:
    """List all registered agents"""
    bus, AgentStatus, MessageType, MessagePriority = _bus()
    return {"agents": bus.list_agents()}


@router.post("/coordination/messages/send")
async def send_message(request: SendMessageRequest) -> Dict[str, Any]:
    """Send a coordination message"""
    bus, AgentStatus, MessageType, MessagePriority = _bus()
    try:
        msg_type = MessageType(request.message_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown message type: {request.message_type}")
    try:
        priority = MessagePriority(request.priority)
    except ValueError:
        priority = MessagePriority.NORMAL

    msg = bus.create_message(
        message_type=msg_type,
        sender_id=request.sender_id,
        payload=request.payload,
        recipient_id=request.recipient_id,
        priority=priority,
        ttl_seconds=request.ttl_seconds,
    )
    delivered = await bus.send(msg)
    return {"sent": delivered, "message_id": msg.message_id}


@router.post("/coordination/locks/acquire")
async def acquire_lock(request: AcquireLockRequest) -> Dict[str, Any]:
    """Acquire a resource lock"""
    bus, AgentStatus, MessageType, MessagePriority = _bus()
    lock = bus.acquire_lock(request.resource_id, request.holder_id, request.ttl_seconds)
    if not lock:
        raise HTTPException(status_code=409, detail=f"Resource '{request.resource_id}' is already locked")
    return {
        "acquired": True,
        "lock_id": lock.lock_id,
        "resource_id": lock.resource_id,
        "holder_id": lock.holder_id,
        "expires_at": lock.granted_at + lock.ttl_seconds,
    }


@router.delete("/coordination/locks/{resource_id}")
async def release_lock(resource_id: str, holder_id: str) -> Dict[str, Any]:
    """Release a resource lock"""
    bus, AgentStatus, MessageType, MessagePriority = _bus()
    released = bus.release_lock(resource_id, holder_id)
    return {"released": released, "resource_id": resource_id}


@router.get("/coordination/locks")
async def list_locks() -> Dict[str, Any]:
    """List all active resource locks"""
    bus, AgentStatus, MessageType, MessagePriority = _bus()
    return {"locks": bus.list_locks()}


@router.post("/coordination/consensus/propose")
async def propose_consensus(request: ProposeConsensusRequest) -> Dict[str, Any]:
    """Create a consensus proposal"""
    bus, AgentStatus, MessageType, MessagePriority = _bus()
    proposal = bus.propose_consensus(
        topic=request.topic,
        proposed_value=request.proposed_value,
        proposer_id=request.proposer_id,
        required_agents=request.required_agents,
        quorum_fraction=request.quorum_fraction,
        deadline_seconds=request.deadline_seconds,
    )
    return {
        "proposal_id": proposal.proposal_id,
        "topic": proposal.topic,
        "proposed_value": proposal.proposed_value,
        "deadline": proposal.deadline,
    }


@router.post("/coordination/consensus/vote")
async def cast_vote(request: CastVoteRequest) -> Dict[str, Any]:
    """Cast a vote on a consensus proposal"""
    bus, AgentStatus, MessageType, MessagePriority = _bus()
    result = bus.cast_vote(request.proposal_id, request.agent_id, request.vote)
    return {
        "voted": True,
        "proposal_id": request.proposal_id,
        "final_result": result,
        "decided": result is not None,
    }


@router.get("/coordination/stats")
async def coordination_stats() -> Dict[str, Any]:
    """Get coordination bus statistics"""
    bus, AgentStatus, MessageType, MessagePriority = _bus()
    return bus.stats()


@router.get("/coordination/agents/capable")
async def find_capable_agents(
    capabilities: str,
    available_only: bool = True,
) -> Dict[str, Any]:
    """Find agents with specific capabilities"""
    bus, AgentStatus, MessageType, MessagePriority = _bus()
    required = {c.strip() for c in capabilities.split(",")}
    agents = bus.find_capable_agents(required, available_only=available_only)
    return {
        "required_capabilities": list(required),
        "agents": [a.to_dict() for a in agents],
    }
